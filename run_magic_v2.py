# -*- coding: utf-8 -*-
"""
神奇公式 修改版 v2（对照实验，不覆盖原版）
============================================================
在原版（run_dogs_annual.py --strategy magic，年初调仓）基础上加三件套：

  ① EBIT 用近 3 年均值（Greenblatt 正常化盈利）
       —— 抑制"周期股盈利峰值陷阱"：煤炭/航运/航空在盈利顶点时
          单年 EBIT 虚高 → 原版反复买在周期顶。取近3年均值后，
          只有持续赚钱的公司才能排前。不足3年用可得年数均值。
  ② HS300 MA200 趋势过滤（月度检查，非日度，防止 V 型底震出）
       —— 每月第一个交易日，用 T-1 收盘 vs MA200(T-1)：
          跌破 → 降到半仓(卖出各持仓一半)；站回 → 恢复满仓。
  ④ 行业上限 + 固定持仓数
       —— 单一 stock_basic.industry ≤ 2 只；V2.1 默认持仓 15 只 + ⑧暴涨护栏 1.5（分散压回撤、防一次性暴利残留）
          （原版 top-n 30 实际只选出 6~17 只且行业高度集中）。

口径与原版对齐：
  - hfq 后复权（含分红再投）记账，买入日因子归一化（同 run_dogs_annual）
  - T-1 数据选股、T 日开盘执行；卖 0.99955 / 买 1.0002 费率
  - 基准 000300.SH 价格指数
  - 报告输出 hfq / raw / 真实趴账 三口径（同 run_dogs_annual 三轨道修正）

运行：
  venv_ml/Scripts/python.exe run_magic_v2.py 20140301 20260727
  # 持仓数/股票池/资金 默认均读 config.GLOBAL / config.BACKTEST 全局设置；
  # 如需覆盖：--top-n 8 --stock-pool zz500 --capital 200000
  可选: --no-trend 关闭②  --ebit-years 1 关闭①  --industry-cap 99 关闭④
  可选: --breadth-confirm 开启市场广度双重确认（道氏"双腿"，默认关闭）：
        第二条腿 = 股票池成分股中站上各自MA200的比例 ≥ --breadth-th(默认50%)。
        月检口径：双腿成立=满仓 / 仅一条腿=半仓 / 双腿皆失=1/4仓。
        结果存 backtest_v2_breadth_*.csv（不覆盖基线 v2），报告自动附
        v2基线 对照表（v2基线 vs v2+广度，均为本策略自身输出，独立文件无覆盖冲突）。
"""
import sys, os, sqlite3, bisect
# 印花税率复用共享引擎的「分段口径」（2023-08-28 起千1→千0.5）
from run_monthly_rebalance import stamp_duty_rate
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime

import run_magic_formula as mf          # 复用原版缓存/资格过滤（只读，不修改原版）
import config                             # 全局设置：stock_pool / top_n / 资金

DB_PATH = "D:/tu-shareData/astock_daily.db"

# 股票池 → 对应基准指数（与平台 STOCK_POOL_INDEX 口径一致）
_POOL_INDEX = {
    "hs300": "000300.SH", "zz500": "000905.SH", "zz800": "000906.SH",
    "zz1000": "000852.SH", "all": "000906.SH",
}

# ────────────────────────────────────────────────────────────
#  内存价格缓存（hfq 口径与 run_dogs_annual.get_hfq_price 完全一致，
#  前向填充因子、区间前缺失则后向取首个因子；全内存 bisect，快 100 倍）
# ────────────────────────────────────────────────────────────
_PX = {}    # code -> (dates, opens, closes)
_FAC = {}   # code -> (dates, factors)

def _conn():
    return sqlite3.connect(DB_PATH)

def _load_code(code):
    if code in _PX:
        return
    c = _conn()
    rows = c.execute(
        "SELECT CAST(trade_date AS TEXT), open, close FROM daily "
        "WHERE ts_code=? ORDER BY trade_date", (code,)).fetchall()
    fr = c.execute(
        "SELECT CAST(trade_date AS TEXT), adj_factor FROM adj_factor "
        "WHERE ts_code=? ORDER BY trade_date", (code,)).fetchall()
    c.close()
    _PX[code] = ([r[0] for r in rows],
                 [r[1] for r in rows],
                 [r[2] for r in rows])
    _FAC[code] = ([r[0] for r in fr], [r[1] for r in fr])

def _raw_price(code, td, kind="close"):
    _load_code(code)
    dates, opens, closes = _PX[code]
    i = bisect.bisect_right(dates, td) - 1
    if i < 0:
        return None
    v = opens[i] if kind == "open" else closes[i]
    return float(v) if v is not None else None

def _factor(code, td):
    _load_code(code)
    dates, facs = _FAC[code]
    i = bisect.bisect_right(dates, td) - 1
    if i >= 0 and facs[i] is not None:
        return float(facs[i])
    # 后向填充：区间前无因子 → 取首个可用因子（与原版修复后的口径一致）
    for f in facs:
        if f is not None:
            return float(f)
    return None

# ────────────────────────────────────────────────────────────
#  分红数据（dividend_detail: ex_date 除权日 / cash_div 每股分红）
#  用于"真实趴账"轨道：除权日把每股分红计入闲置现金，次年调仓日再投
# ────────────────────────────────────────────────────────────
_DIV = {}
def load_dividends():
    global _DIV
    if _DIV:
        return
    c = _conn()
    rows = c.execute(
        "SELECT ts_code, ex_date, cash_div FROM dividend_detail "
        "WHERE ex_date IS NOT NULL AND cash_div IS NOT NULL AND cash_div > 0").fetchall()
    c.close()
    d = {}
    for code, exd, dv in rows:
        d.setdefault(str(code), []).append((str(exd), float(dv)))
    _DIV = d

def hfq_price(code, td, kind="close"):
    p = _raw_price(code, td, kind)
    if p is None:
        return None
    f = _factor(code, td)
    return p * f if f else p

# ────────────────────────────────────────────────────────────
#  指数缓存 + MA200
# ────────────────────────────────────────────────────────────
_IDX = {}   # code -> (dates, closes, cumsum)

def _load_index(idx_code):
    if idx_code in _IDX:
        return
    c = _conn()
    rows = c.execute(
        "SELECT CAST(trade_date AS TEXT), close FROM index_daily "
        "WHERE ts_code=? ORDER BY trade_date", (idx_code,)).fetchall()
    c.close()
    dates = [r[0] for r in rows]
    closes = [float(r[1]) for r in rows]
    csum = np.concatenate([[0.0], np.cumsum(closes)])
    _IDX[idx_code] = (dates, closes, csum)

def index_close(idx_code, td):
    _load_index(idx_code)
    dates, closes, _ = _IDX[idx_code]
    i = bisect.bisect_right(dates, td) - 1
    return closes[i] if i >= 0 else None

def index_above_ma(idx_code, td, window=200):
    """T-1 收盘 >= MA(window)？数据不足 window 天返回 True（不干预）。"""
    _load_index(idx_code)
    dates, closes, csum = _IDX[idx_code]
    i = bisect.bisect_right(dates, td) - 1
    if i < window - 1:
        return True
    ma = (csum[i + 1] - csum[i + 1 - window]) / window
    return closes[i] >= ma

# ────────────────────────────────────────────────────────────
#  市场广度（道氏"双重确认"第二条腿）：
#  股票池成分股中，T-1 收盘站上各自 MA(window) 的比例
# ────────────────────────────────────────────────────────────
_BR_POOL = {}    # (pool, td) -> set(codes)  成分快照缓存

def _stock_above_ma(code, td, window=200):
    """个股 T-1 收盘 >= 自身 MA(window)？数据不足返回 None（不计入分母）。"""
    _load_code(code)
    dates, _opens, closes = _PX[code]
    i = bisect.bisect_right(dates, td) - 1
    if i < window - 1:
        return None
    seg = [c for c in closes[i + 1 - window: i + 1] if c is not None]
    if len(seg) < int(window * 0.8) or closes[i] is None:
        return None
    return float(closes[i]) >= (sum(seg) / len(seg))

def market_breadth(stock_pool, td, window=200):
    """返回 (广度比例0~1, 有效样本数)；无成分数据返回 (None, 0)。"""
    pool_key = stock_pool if stock_pool in ("hs300", "zz500", "zz800", "zz1000") else "hs300"
    key = (pool_key, td)
    if key not in _BR_POOL:
        _BR_POOL[key] = mf._get_pool_constituents(pool_key, td)
    pool = _BR_POOL[key]
    if not pool:
        return None, 0
    above = total = 0
    for code in pool:
        r = _stock_above_ma(code, td, window)
        if r is None:
            continue
        total += 1
        if r:
            above += 1
    if total < 30:
        return None, total
    return above / total, total

# ────────────────────────────────────────────────────────────
#  ① + ④ 选股：EBIT 3年均值 + 行业上限
# ────────────────────────────────────────────────────────────
def select_magic_v2(prev_date, top_n=15, industry_cap=2, ebit_years=3,
                    stock_pool="hs300", verbose=True,
                    profit_guard=0.0, ebit_conservative=False,
                    spike_guard=1.5, ebit_stat="mean"):
    """返回 DataFrame[ts_code, roc, ey, score, industry, avg_ebit, n_yr]

    ⑤ 利润回落护栏（profit_guard > 0 时启用）
        动机：EBIT 近3年均值虽压住"周期顶单年虚高"，但反向副作用是
        「一次性暴利会在均值里赖满 3 年」——达安基因/英科医疗/九安医疗
        这类新冠受益股利润早已崩塌，均值仍把 EBIT 撑高、EY 虚高，
        于是又变成价值陷阱（V2 在 2024 年跑输基准 21.95pp 的主因）。
        规则：最新年 EBIT < 3年均值 × profit_guard → 直接剔除。
        仅在有效年数 >= 2 时生效（单年样本 latest == avg 无意义）。

    ⑥ 保守 EBIT 口径（ebit_conservative=True 时启用）
        eff_ebit = min(最新年 EBIT, 基准统计量)，用于 roc/ey 计算。
        比 ⑤ 温和：不剔除，只让暴利滞后股的便宜程度不再虚高。

    ⑦ 正常化统计量（ebit_stat）
        mean(原) / median / min。median 天然抗"单年脉冲"：九安医疗在
        2023-12-29 时点 3年均值 66.18亿、中位数仅 9.29亿（差 86%），
        换中位数后 EY 打回原形，根本进不了前 5。

    ⑧ 利润暴涨护栏（spike_guard > 0 时启用）
        最新年 EBIT > 3年均值 × spike_guard → 剔除。
        补 ⑤ 的结构性盲区：⑤ 只能在"暴利已退潮"后才触发（九安 2024-12-31
        时点 最新/均值=0.11 才被拦），而亏损恰恰发生在"暴利当年"买入
        （2023-12-29 时点 最新/均值=2.80，⑤ 完全抓不到）。建议 1.5（实测 1.5 优于 2.0）。
    """
    basic = mf._load_basic()
    pool_set = (mf._get_pool_constituents(stock_pool, prev_date)
                if stock_pool and stock_pool != "all" else None)

    c = _conn()
    rows = c.execute("SELECT DISTINCT ts_code FROM daily WHERE trade_date=?",
                     (prev_date,)).fetchall()
    c.close()
    trading = set(str(r[0]) for r in rows)

    d_rb = datetime.strptime(prev_date, "%Y%m%d")
    eligible = set()
    for code in sorted(trading):
        if pool_set is not None and code not in pool_set:
            continue
        info = basic.get(code)
        if info is None:
            if code.startswith("688") or code.endswith(".BJ"):
                continue
            eligible.add(code)
            continue
        if info["excluded"]:
            continue
        ind = info["industry"]
        if ind in mf.FINANCIAL_INDUSTRIES or ind in mf.UTILITY_INDUSTRIES:
            continue
        ld = info["list_date"]
        if ld:
            try:
                if (d_rb - datetime.strptime(ld, "%Y%m%d")).days < mf.IPO_MIN_DAYS:
                    continue
            except Exception:
                pass
        eligible.add(code)
    if not eligible:
        return pd.DataFrame()

    fin = mf._load_financials(prev_date)      # point-in-time 最新年报（分母口径）
    mv_map = mf._get_mv_map(prev_date)
    mf._load_raw()

    recs = []
    n_guard_cut = 0        # ⑤ 被利润回落护栏剔除的只数
    n_conserv_hit = 0      # ⑥ 被保守口径下调 EBIT 的只数
    n_spike_cut = 0        # ⑧ 被利润暴涨护栏剔除的只数
    for code in sorted(eligible):
        f = fin.get(code)
        if f is None:
            continue
        ed0 = f["end_date"]
        # ① EBIT 近 N 年均值（end_date <= 最新可用年报期末，按期末去重）
        hist = [ir for ir in mf._RAW_INC.get(code, []) if ir["end_date"] <= ed0]
        hist.sort(key=lambda r: (r["end_date"], r["ann_date"]), reverse=True)
        ebits, seen = [], set()
        for ir in hist:
            if ir["end_date"] in seen:
                continue
            try:
                v = float(ir["ebit"])
            except (TypeError, ValueError):
                continue
            if np.isfinite(v):
                ebits.append(v)
                seen.add(ir["end_date"])
            if len(ebits) >= ebit_years:
                break
        if not ebits:
            continue
        avg_ebit = float(np.mean(ebits))
        if avg_ebit <= 0:
            continue
        latest_ebit = float(ebits[0])          # hist 已按 end_date 降序，[0]=最新年

        # ⑦ 正常化统计量：mean(原) / median(抗单年脉冲) / min(最保守)
        if ebit_stat == "median":
            base_ebit = float(np.median(ebits))
        elif ebit_stat == "min":
            base_ebit = float(min(ebits))
        else:
            base_ebit = avg_ebit
        if base_ebit <= 0:
            continue

        # ⑤ 利润回落护栏：最新年 EBIT 相对均值塌陷 → 暴利已退潮，剔除
        if profit_guard > 0 and len(ebits) >= 2:
            if latest_ebit < avg_ebit * profit_guard:
                n_guard_cut += 1
                continue

        # ⑧ 利润暴涨护栏：最新年相对均值异常膨胀 → 正处一次性暴利峰值，剔除
        #    （⑤只能抓"暴利之后"，抓不到"暴利当年"——九安医疗2022年EBIT 185亿
        #      vs 前两年 9.29/4.24亿，最新/均值=2.80，正是靠⑧才拦得住）
        if spike_guard > 0 and len(ebits) >= 2:
            if latest_ebit > avg_ebit * spike_guard:
                n_spike_cut += 1
                continue

        # ⑥ 保守 EBIT 口径：取 min(最新年, 基准统计量)，不让暴利滞后撑高 EY
        eff_ebit = base_ebit
        if ebit_conservative and len(ebits) >= 2 and latest_ebit < base_ebit:
            eff_ebit = latest_ebit
            n_conserv_hit += 1
        if eff_ebit <= 0:
            continue

        denom = f["nwc"] + f["fix"]
        if not np.isfinite(denom) or denom <= 0:
            continue
        mv = mv_map.get(code)
        if mv is None:
            continue
        ev = mv * 10000.0 + f["liab"] - f["cash"]
        if not np.isfinite(ev) or ev <= 0:
            continue
        roc = eff_ebit / denom
        ey = eff_ebit / ev
        info = basic.get(code) or {}
        ind = info.get("industry") or f"unk_{code}"
        recs.append((code, roc, ey, ind, avg_ebit, eff_ebit, len(ebits)))

    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame(recs, columns=["ts_code", "roc", "ey", "industry",
                                     "avg_ebit", "eff_ebit", "n_yr"])
    df["rank_roc"] = df["roc"].rank(ascending=False, method="first")
    df["rank_ey"] = df["ey"].rank(ascending=False, method="first")
    df["score"] = df["rank_roc"] + df["rank_ey"]
    # 稳定排序：并列分时用 ts_code 兜底，避免 set 遍历顺序导致结果漂移
    df = df.sort_values(["score", "ts_code"], kind="mergesort").reset_index(drop=True)

    # ④ 行业上限贪心：按 score 顺序取，单行业 ≤ industry_cap
    picks, ind_cnt = [], {}
    for _, r in df.iterrows():
        ind = r["industry"]
        if ind_cnt.get(ind, 0) >= industry_cap:
            continue
        picks.append(r)
        ind_cnt[ind] = ind_cnt.get(ind, 0) + 1
        if len(picks) >= top_n:
            break
    out = pd.DataFrame(picks).reset_index(drop=True)
    if verbose:
        n1 = int((out["n_yr"] < ebit_years).sum()) if len(out) else 0
        extra = ""
        if ebit_stat != "mean":
            extra += f"，EBIT口径={ebit_stat}"
        if profit_guard > 0:
            extra += f"，护栏⑤剔除 {n_guard_cut} 只(最新EBIT<均值×{profit_guard:g})"
        if spike_guard > 0:
            extra += f"，护栏⑧剔除 {n_spike_cut} 只(最新EBIT>均值×{spike_guard:g})"
        if ebit_conservative:
            extra += f"，护栏⑥下调 {n_conserv_hit} 只EBIT"
        print(f"  [v2选股] 候选 {len(df)} 只 → 行业上限{industry_cap}/固定{top_n}只"
              f" → 实选 {len(out)} 只（{n1} 只财务不足{ebit_years}年{extra}）")
    return out

# ────────────────────────────────────────────────────────────
#  回测引擎（hfq 单轨，买入日因子归一化，同原版口径）
# ────────────────────────────────────────────────────────────
SELL_MULT = 0.99955
BUY_MULT = 1.0002

def _prev_trade_date_db(td):
    """区间外的上一交易日（供首日调仓选股用，T-1 口径与原版一致）。"""
    c = _conn()
    r = c.execute("SELECT MAX(CAST(trade_date AS TEXT)) FROM daily "
                  "WHERE CAST(trade_date AS TEXT) < ?", (td,)).fetchone()
    c.close()
    return r[0] if r and r[0] else td

def _px_hfq(pos, code, td, kind):
    """归一化 hfq 价（买入日因子归一化，消除跨期复权台阶）。"""
    p = hfq_price(code, td, kind)
    if p is None:
        return pos.get("last_hfq")
    return p / (pos.get("buy_factor") or 1.0)

def _px_raw(pos, code, td, kind):
    """原始价（不含分红，不复权）。"""
    p = _raw_price(code, td, kind)
    return p if p is not None else pos.get("last_raw")

def _sell(positions, cash_h, cash_r, cash_i, code, td, frac=1.0):
    """按开盘价卖出 frac 比例（round 到 100 股）；同步更新三轨道现金与股数。
    返回 (cash_h, cash_r, cash_i)。"""
    pos = positions[code]
    px_h = _px_hfq(pos, code, td, "open")
    px_r = _px_raw(pos, code, td, "open")
    if px_h is None or px_r is None:
        return cash_h, cash_r, cash_i
    if frac >= 1.0:
        sell_sh = pos["shares"]
    else:
        sell_sh = int(pos["shares"] * frac / 100) * 100
        if sell_sh <= 0:
            return cash_h, cash_r, cash_i
        if pos["shares"] - sell_sh < 100:
            sell_sh = pos["shares"]
    cash_h += sell_sh * px_h * SELL_MULT
    cash_r += sell_sh * px_r * SELL_MULT
    cash_i += sell_sh * px_r * SELL_MULT
    pos["shares"] -= sell_sh
    pos["shares_raw"] -= sell_sh
    pos["shares_real"] -= sell_sh
    if pos["shares"] <= 0:
        del positions[code]
    return cash_h, cash_r, cash_i

def run_backtest_v2(start_date, end_date, top_n=15, industry_cap=2,
                    ebit_years=3, capital=None, stock_pool=None,
                    trend_filter=True, ma_window=200, half_ratio=0.5,
                    breadth_confirm=False, breadth_th=0.5,
                    profit_guard=0.0, ebit_conservative=False,
                    spike_guard=1.5, ebit_stat="mean"):
    # 全局设置兜底：持股数 / 股票池 / 资金 全部走 config.GLOBAL / config.BACKTEST
    # （.bat 已通过 --capital 显式传全局资金；这里保证直接命令行运行也吃全局）
    if top_n is None:
        top_n = 15   # V2.1 默认持仓：分散压回撤（覆盖 config.GLOBAL.top_n）
    if stock_pool is None:
        stock_pool = config.GLOBAL.get("stock_pool", "hs300")
    if capital is None:
        capital = config.BACKTEST.get("monthly_rebalance_capital", 100000)
    bench = _POOL_INDEX.get(stock_pool, "000300.SH")   # 基准 = 股票池对应指数
    c = _conn()
    trade_dates = [r[0] for r in c.execute(
        "SELECT DISTINCT CAST(trade_date AS TEXT) FROM daily "
        "WHERE trade_date BETWEEN ? AND ? ORDER BY trade_date",
        (start_date, end_date)).fetchall()]
    c.close()
    if not trade_dates:
        print("[ERROR] 无交易日")
        return None

    yearly_first, monthly_first = {}, {}
    for td in trade_dates:
        yearly_first.setdefault(td[:4], td)
        monthly_first.setdefault(td[:6], td)
    monthly_first_set = set(monthly_first.values())

    print("=" * 72)
    print("  神奇公式 v2（EBIT三年均值 + MA200趋势过滤 + 行业上限）· 年度调仓")
    print("=" * 72)
    print(f"  区间: {start_date} ~ {end_date} | 持仓: {top_n} 只 | 行业上限: {industry_cap}")
    print(f"  EBIT口径: 近{ebit_years}年均值 | 股票池: {stock_pool} | 总资金: {capital:,}")
    print(f"  趋势过滤: {'HS300<MA%d → 半仓(月检)' % ma_window if trend_filter else '关闭'}")
    if profit_guard > 0 or ebit_conservative or spike_guard > 0 or ebit_stat != "mean":
        gtxt = []
        if ebit_stat != "mean":
            gtxt.append(f"⑦正常化统计量={ebit_stat}")
        if profit_guard > 0:
            gtxt.append(f"⑤回落护栏(最新<均值×{profit_guard:g}→剔除)")
        if spike_guard > 0:
            gtxt.append(f"⑧暴涨护栏(最新>均值×{spike_guard:g}→剔除)")
        if ebit_conservative:
            gtxt.append("⑥保守口径(min(最新, 基准))")
        print(f"  利润护栏: {' + '.join(gtxt)}")
    else:
        print("  利润护栏: 关闭（原 v2 基线）")
    if breadth_confirm:
        print(f"  广度确认: 开启（成分股站上各自MA{ma_window}比例≥{breadth_th:.0%} 为第二条腿；"
              f"双腿=满仓 / 单腿={half_ratio:.0%} / 无腿={half_ratio/2:.0%}）")
    print("=" * 72)

    positions = {}   # code -> {shares, shares_raw, shares_real, buy_factor, last_hfq, last_raw, buy_date}
    cash_h = float(capital)      # hfq 轨道现金(后复权·含分红再投, 对年度调仓高估)
    cash_r = float(capital)      # raw 轨道现金(原始价·不含分红, 低估)
    cash_i = float(capital)      # 真实趴账轨道现金(原始价+分红趴账至调仓日再投)
    load_dividends()

    daily_vals = []
    ratio = 1.0                  # 当前目标仓位比例
    trend_log = []               # [(date, old, new)]
    last_holdings = []

    def port_value(td, kind="close"):
        """返回 (hfq, raw, real) 三轨道组合价值（开盘/收盘）。"""
        vh, vr, vi = cash_h, cash_r, cash_i
        for code, pos in positions.items():
            ph = _px_hfq(pos, code, td, kind)
            pr = _px_raw(pos, code, td, kind)
            if ph is not None:
                vh += pos["shares"] * ph
            if pr is not None:
                vr += pos["shares_raw"] * pr
                vi += pos["shares_real"] * pr
        return vh, vr, vi

    def _target_ratio(asof_td):
        """按 T-1 状态给出目标仓位。
        仅趋势：上方=满仓 / 下方=half。
        +广度确认（道氏双腿）：双腿=满仓 / 单腿=half / 无腿=half/2。
        返回 (ratio, 说明文字)"""
        leg1 = index_above_ma(bench, asof_td, ma_window)
        if not breadth_confirm:
            return ((1.0, f"HS300站上MA{ma_window}") if leg1
                    else (half_ratio, f"HS300跌破MA{ma_window}"))
        br, n = market_breadth(stock_pool, asof_td, ma_window)
        if br is None:
            # 广度样本不足 → 退化为纯趋势腿
            return ((1.0, f"广度样本不足,仅趋势腿:上方") if leg1
                    else (half_ratio, f"广度样本不足,仅趋势腿:下方"))
        leg2 = br >= breadth_th
        desc = (f"指数{'√' if leg1 else '×'}MA{ma_window} + "
                f"广度{br:.0%}{'≥' if leg2 else '<'}{breadth_th:.0%}(n={n})")
        if leg1 and leg2:
            return 1.0, "双腿确认: " + desc
        if leg1 or leg2:
            return half_ratio, "单腿: " + desc
        return half_ratio / 2, "无腿: " + desc

    # 首日按 T-1 趋势状态初始化仓位比例（原版是首日满仓建仓；
    # v2 若起点恰在闸门之下则按目标比例起步，口径自洽）
    if trend_filter:
        _p0 = _prev_trade_date_db(trade_dates[0])
        _r0, _d0 = _target_ratio(_p0)
        if _r0 != 1.0:
            ratio = _r0
            trend_log.append((trade_dates[0], 1.0, ratio, _d0))

    for i, td in enumerate(trade_dates):
        year = td[:4]
        # 首日(i==0)也调仓：选股基准取数据库中区间前的上一交易日（T-1 口径不变）
        is_rebal = (td == yearly_first.get(year))
        is_month_head = td in monthly_first_set

        # ── ② 月度趋势检查（用 T-1 收盘 vs MA200 [+广度确认]，无前视）──
        if trend_filter and is_month_head and i > 0:
            prev_td = trade_dates[i - 1]
            new_ratio, desc = _target_ratio(prev_td)
            if new_ratio != ratio:
                trend_log.append((td, ratio, new_ratio, desc))
                old_ratio = ratio
                ratio = new_ratio
                if not is_rebal and positions:
                    # 直接把持仓缩/扩到目标比例（调仓日则并入调仓逻辑）
                    if ratio < old_ratio:
                        frac = 1.0 - ratio / old_ratio
                        for code in list(positions.keys()):
                            cash_h, cash_r, cash_i = _sell(positions, cash_h, cash_r, cash_i, code, td, frac=frac)
                        print(f"  [趋势] {td} {desc} → 降至{ratio:.0%}仓")
                    else:
                        # 加回：三轨道各自按自身组合价值补到目标比例
                        for tag in ("h", "r", "i"):
                            ck = "shares" if tag == "h" else ("shares_raw" if tag == "r" else "shares_real")
                            cash = cash_h if tag == "h" else (cash_r if tag == "r" else cash_i)
                            val = cash
                            for code, pos in positions.items():
                                px = _px_hfq(pos, code, td, "open") if tag == "h" else _px_raw(pos, code, td, "open")
                                if px is not None:
                                    val += pos[ck] * px
                            target = val * ratio
                            invested = val - cash
                            budget = max(target - invested, 0) * 0.995
                            if positions and budget > 0:
                                per = budget / len(positions)
                                for code, pos in positions.items():
                                    # 与旧逻辑完全一致：开盘价缺失（停牌）则该票跳过加仓，不回退 last_hfq
                                    if tag == "h":
                                        op = hfq_price(code, td, "open")
                                        if op is None or op <= 0:
                                            continue
                                        px = op / (pos.get("buy_factor") or 1.0)
                                    else:
                                        op = _raw_price(code, td, "open")
                                        if op is None or op <= 0:
                                            continue
                                        px = op
                                    add = int(per / px / 100) * 100
                                    cost = add * px * BUY_MULT
                                    if add > 0 and cost <= cash:
                                        pos[ck] += add
                                        cash -= cost
                            if tag == "h":
                                cash_h = cash
                            elif tag == "r":
                                cash_r = cash
                            else:
                                cash_i = cash
                        print(f"  [趋势] {td} {desc} → 升至{ratio:.0%}仓")

        # ── 年度调仓 ──
        if is_rebal:
            prev_td = trade_dates[i - 1] if i > 0 else _prev_trade_date_db(td)
            print(f"\n── {year}年调仓日: {td} (选股基准: {prev_td}) ──")
            sel = select_magic_v2(prev_td, top_n=top_n, industry_cap=industry_cap,
                                  ebit_years=ebit_years, stock_pool=stock_pool,
                                  profit_guard=profit_guard,
                                  ebit_conservative=ebit_conservative,
                                  spike_guard=spike_guard, ebit_stat=ebit_stat)
            if sel is None or len(sel) == 0:
                print(f"  [WARN] {year}年选股为空，保持现有持仓")
            else:
                new_codes = sel["ts_code"].tolist()
                names = "、".join(
                    f"{r.ts_code}({(mf._load_basic().get(r.ts_code) or {}).get('name', '?')}"
                    f"/{r.industry})" for r in sel.itertuples())
                print(f"  本年入选 {len(new_codes)} 只: {names}")
                new_set = set(new_codes)
                # 卖出移出的（三轨道）
                for code in list(positions.keys()):
                    if code not in new_set:
                        cash_h, cash_r, cash_i = _sell(positions, cash_h, cash_r, cash_i, code, td, frac=1.0)
                # 三轨道目标投资额 = 组合价值 × 趋势仓位比例
                vh, vr, vi = port_value(td, "open")
                target_h, target_r, target_i = vh * ratio, vr * ratio, vi * ratio
                invested_h, invested_r, invested_i = vh - cash_h, vr - cash_r, vi - cash_i
                # 已有持仓超目标 → 按比例减（三轨道各自，按 hfq 口径触发）
                if invested_h > target_h * 1.02:
                    frac = 1.0 - target_h / invested_h
                    for code in list(positions.keys()):
                        cash_h, cash_r, cash_i = _sell(positions, cash_h, cash_r, cash_i, code, td, frac=frac)
                    vh, vr, vi = port_value(td, "open")
                    target_h, target_r, target_i = vh * ratio, vr * ratio, vi * ratio
                    invested_h, invested_r, invested_i = vh - cash_h, vr - cash_r, vi - cash_i
                budget_h = max(target_h - invested_h, 0) * 0.98
                budget_r = max(target_r - invested_r, 0) * 0.98
                budget_i = max(target_i - invested_i, 0) * 0.98
                to_buy = [cd for cd in new_codes if cd not in positions]
                bought, skipped = 0, []
                if to_buy and (budget_h > 0 or budget_r > 0 or budget_i > 0):
                    per_h = budget_h / len(to_buy)
                    per_r = budget_r / len(to_buy)
                    per_i = budget_i / len(to_buy)
                    for code in to_buy:
                        op = hfq_price(code, td, "open")
                        if op is None or op <= 0:
                            skipped.append(code)
                            continue
                        bf = _factor(code, td) or 1.0
                        px = op / bf                      # 归一化买入价 = raw开盘
                        # 三轨道股数：raw 与 hfq 同价→股数相同；real 用含分红现金→可能更多
                        sh_h = int(per_h / px / 100) * 100
                        sh_r = int(per_r / px / 100) * 100
                        sh_i = int(per_i / px / 100) * 100
                        cost_h = sh_h * px * BUY_MULT
                        cost_r = sh_r * px * BUY_MULT
                        cost_i = sh_i * px * BUY_MULT
                        if sh_h <= 0 or cost_h > cash_h:
                            skipped.append(code)
                            continue
                        positions[code] = {
                            "shares": sh_h, "shares_raw": sh_r, "shares_real": sh_i,
                            "buy_factor": bf, "last_hfq": px, "last_raw": px,
                            "buy_date": td}
                        cash_h -= cost_h
                        cash_r -= cost_r
                        cash_i -= cost_i
                        bought += 1
                if skipped:
                    print(f"  [跳过] {len(skipped)} 只: {', '.join(skipped)}")
                print(f"  买入 {bought} 只, 当前持仓 {len(positions)} 只, "
                      f"仓位比例 {ratio:.0%}, 现金(hfq/raw/real) {cash_h:,.0f}/{cash_r:,.0f}/{cash_i:,.0f}")
                last_holdings = list(positions.keys())

        # ── 每日估值（三轨道）──
        v_h = cash_h
        v_r = cash_r
        v_i = cash_i
        for code, pos in positions.items():
            ph = _px_hfq(pos, code, td, "close")
            pr = _px_raw(pos, code, td, "close")
            if ph is not None:
                pos["last_hfq"] = ph
                v_h += pos["shares"] * ph
            if pr is not None:
                pos["last_raw"] = pr
                v_r += pos["shares_raw"] * pr
                v_i += pos["shares_real"] * pr
            # 分红趴账：除权日把每股分红计入闲置现金（仅 real 轨道，留至下次调仓再投）
            for exd, dv in _DIV.get(code, []):
                if exd == td and exd > pos.get("buy_date", ""):
                    cash_i += pos["shares_real"] * dv
        daily_vals.append({"date": td, "value": v_h, "value_raw": v_r, "value_real": v_i})

    # ── 末日平仓（三轨道）──
    last_td = trade_dates[-1]
    for code in list(positions.keys()):
        pos = positions[code]
        ph = _px_hfq(pos, code, last_td, "close")
        pr = _px_raw(pos, code, last_td, "close")
        if ph is not None:
            rev = pos["shares"] * ph
            cash_h += rev - max(rev * 0.0002, 5.0) - rev * stamp_duty_rate(last_td)
        if pr is not None:
            rev_r = pos["shares_raw"] * pr
            cash_r += rev_r - max(rev_r * 0.0002, 5.0) - rev_r * stamp_duty_rate(last_td)
            rev_i = pos["shares_real"] * pr
            cash_i += rev_i - max(rev_i * 0.0002, 5.0) - rev_i * stamp_duty_rate(last_td)
        del positions[code]
    daily_vals[-1]["value"] = cash_h
    daily_vals[-1]["value_raw"] = cash_r
    daily_vals[-1]["value_real"] = cash_i

    # 护栏组合写进文件名后缀，避免不同参数互相覆盖产物
    tag_suffix = ""
    if ebit_stat != "mean":
        tag_suffix += f"_{ebit_stat}"
    if profit_guard > 0:
        tag_suffix += f"_pg{profit_guard:g}".replace(".", "")
    if spike_guard > 0:
        tag_suffix += f"_sg{spike_guard:g}".replace(".", "")
    if ebit_conservative:
        tag_suffix += "_cons"

    return _report(daily_vals, trade_dates, capital, bench, trend_log,
                   last_holdings, start_date, end_date,
                   top_n=top_n, trend_filter=trend_filter,
                   breadth_confirm=breadth_confirm, tag_suffix=tag_suffix)

# ────────────────────────────────────────────────────────────
#  报告（hfq / raw / 真实趴账 三口径）
# ────────────────────────────────────────────────────────────
def _metrics(vals):
    v = np.array(vals, dtype=float)
    total = v[-1] / v[0] - 1
    n = len(v)
    ann = (v[-1] / v[0]) ** (252.0 / n) - 1 if n > 1 else 0
    cummax = np.maximum.accumulate(v)
    dd = (v - cummax) / cummax
    mdd = float(dd.min())
    j = int(dd.argmin())
    pk = int(np.argmax(v[:j + 1])) if j > 0 else 0
    rets = np.diff(v) / v[:-1]
    sharpe = ((rets.mean() * 252 - 0.025) / (rets.std() * np.sqrt(252))
              if len(rets) > 1 and rets.std() > 0 else 0)
    return total, ann, mdd, sharpe, pk, j

def _yearly(dates, vals, capital):
    out = {}
    first_val, cur_year, start_v = None, None, None
    for d, v in zip(dates, vals):
        y = d[:4]
        if y != cur_year:
            start_v = capital if cur_year is None else out[cur_year][1]
            cur_year = y
            out[y] = [start_v, v, 1]
        else:
            out[y][1] = v
            out[y][2] += 1
    return out   # year -> [start, end, n_days]

def _report(daily_vals, trade_dates, capital, bench, trend_log,
            last_holdings, start_date, end_date, top_n, trend_filter,
            breadth_confirm=False, tag_suffix=""):
    dates = [d["date"] for d in daily_vals]
    vals = [d["value"] for d in daily_vals]
    vals_raw = [d["value_raw"] for d in daily_vals]
    vals_real = [d["value_real"] for d in daily_vals]
    total, ann, mdd, sharpe, pk, tr = _metrics(vals)
    total_raw, ann_raw, mdd_raw, sharpe_raw, pk_raw, tr_raw = _metrics(vals_raw)
    total_real, ann_real, mdd_real, sharpe_real, pk_real, tr_real = _metrics(vals_real)

    b0 = index_close(bench, dates[0])
    b1 = index_close(bench, dates[-1])
    b_total = b1 / b0 - 1 if b0 and b1 else 0

    print(f"\n{'=' * 82}")
    print("  📊 v2 年度收益对比（hfq含分红再投 / raw原始价 / 真实趴账 三口径）")
    print(f"{'=' * 82}")
    print(f"  {'年份':<7}{'v2(hfq)':>10}{'v2(raw)':>10}{'v2(真实)':>10}{'基准':>10}{'超额(真实)':>11}")
    print(f"  {'─' * 62}")
    yg = _yearly(dates, vals, capital)
    yg_raw = _yearly(dates, vals_raw, capital)
    yg_real = _yearly(dates, vals_real, capital)
    for y in sorted(yg):
        s0, s1, nd = yg[y]
        sret = s1 / s0 - 1
        yd = [d for d in dates if d[:4] == y]
        ib0, ib1 = index_close(bench, yd[0]), index_close(bench, yd[-1])
        bret = ib1 / ib0 - 1 if ib0 and ib1 else 0
        r0, r1, _ = yg_real[y]
        rret = r1 / r0 - 1
        tag = f"{y}*" if nd < 200 else y
        print(f"  {tag:<7}{sret:>+9.2%}{yg_raw[y][1] / yg_raw[y][0] - 1:>+9.2%}"
              f"{rret:>+9.2%}{bret:>+9.2%}{rret - bret:>+10.2%}")
    print(f"  {'─' * 62}")
    print(f"  {'全程':<6}{total:>+9.2%}{total_raw:>+9.2%}{total_real:>+9.2%}"
          f"{b_total:>+9.2%}{total_real - b_total:>+10.2%}")

    print(f"\n{'=' * 82}")
    print("  📈 v2 最终汇总（三口径）")
    print(f"{'=' * 82}")
    print(f"  初始资金: {capital:,.0f}   基准({bench}价格指数·不含分红): {b_total:+.2%}")
    print(f"  ── hfq（后复权·含分红再投·对年度调仓高估）──")
    print(f"    最终资产: {vals[-1]:,.0f}   总收益: {total:+.2%}   年化: {ann:+.2%}")
    print(f"    最大回撤: {mdd:+.2%} (峰 {dates[pk]} → 谷 {dates[tr]})   夏普: {sharpe:.4f}")
    print(f"  ── raw（原始价·不含分红·低估）──")
    print(f"    最终资产: {vals_raw[-1]:,.0f}   总收益: {total_raw:+.2%}   年化: {ann_raw:+.2%}")
    print(f"    最大回撤: {mdd_raw:+.2%} (峰 {dates[pk_raw]} → 谷 {dates[tr_raw]})   夏普: {sharpe_raw:.4f}")
    print(f"  ── 真实趴账（原始价+分红趴账至调仓日再投·主口径）──")
    print(f"    最终资产: {vals_real[-1]:,.0f}   总收益: {total_real:+.2%}   年化: {ann_real:+.2%}")
    print(f"    最大回撤(raw主): {mdd_raw:+.2%} | (real): {mdd_real:+.2%}   夏普: {sharpe_real:.4f}")
    print(f"    超额收益(真实 vs 基准): {total_real - b_total:+.2%}")
    print(f"  [口径] hfq=分红当日即再投(高估); raw=分红不入账(低估); 真实=分红趴账次年再投(真实总回报)。")
    print(f"  [口径] 主看真实轨道; 回撤以 raw 为主口径。")
    print(f"  末期持仓: {', '.join(last_holdings[:8])}")
    if trend_filter:
        print(f"  趋势切换 {len(trend_log)} 次:")
        for rec in trend_log:
            d, old, new = rec[0], rec[1], rec[2]
            desc = f"  ({rec[3]})" if len(rec) > 3 else ""
            print(f"    {d}: {old:.0%} → {new:.0%}{desc}")

    # ── v2 基线对照（开广度确认时，与不带广度的 v2 结果比；本策略自身输出，独立文件无覆盖冲突）──
    if breadth_confirm:
        base_csv = f"data/results/magic_v2/backtest_v2_{start_date}_{end_date}.csv"
        if os.path.exists(base_csv):
            bdf = pd.read_csv(base_csv, dtype={"date": str})
            bvals = bdf["value"].astype(float).tolist()
            bdates = bdf["date"].astype(str).tolist()
            bt, ba, bm, bs_, bpk, btr = _metrics(bvals)
            print(f"\n{'=' * 72}")
            print("  ⚖️  v2基线(纯MA200) vs v2+广度确认 对照")
            print(f"{'=' * 72}")
            print(f"  {'指标':<14}{'v2基线':>16}{'v2+广度':>16}{'差异':>12}")
            print(f"  {'─' * 60}")
            print(f"  {'总收益':<14}{bt:>+15.2%}{total:>+15.2%}{total - bt:>+11.2%}")
            print(f"  {'年化':<14}{ba:>+15.2%}{ann:>+15.2%}{ann - ba:>+11.2%}")
            print(f"  {'最大回撤':<12}{bm:>+15.2%}{mdd:>+15.2%}{mdd - bm:>+11.2%}")
            print(f"  {'夏普':<14}{bs_:>15.4f}{sharpe:>15.4f}{sharpe - bs_:>11.4f}")
            print(f"  {'回撤峰→谷':<11}{bdates[bpk]}→{bdates[btr]:<6}"
                  f"  {dates[pk]}→{dates[tr]}")
            byg = _yearly(bdates, bvals, bvals[0])
            print(f"\n  {'年份':<8}{'v2基线':>10}{'v2+广度':>10}{'差异':>10}")
            print(f"  {'─' * 40}")
            for y in sorted(yg):
                s0, s1, nd = yg[y]
                v2r = s1 / s0 - 1
                if y in byg:
                    b0_, b1_, _ = byg[y]
                    brr = b1_ / b0_ - 1
                    print(f"  {y:<8}{brr:>+9.2%}{v2r:>+9.2%}{v2r - brr:>+9.2%}")
        else:
            print(f"\n  [提示] 未找到 v2 基线日净值 {base_csv}，跳过基线对照")

    out_dir = "data/results/magic_v2"
    os.makedirs(out_dir, exist_ok=True)
    tag = ("_breadth" if breadth_confirm else "") + tag_suffix
    csv_path = f"{out_dir}/backtest_v2{tag}_{start_date}_{end_date}.csv"
    pd.DataFrame(daily_vals).to_csv(csv_path, index=False)
    print(f"\n  v2 日净值已保存 → {csv_path}（含 value / value_raw / value_real 三列）\n")
    return {"total": total, "total_raw": total_raw, "total_real": total_real,
            "annual_real": ann_real, "mdd_raw": mdd_raw, "mdd_real": mdd_real,
            "sharpe_real": sharpe_real}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="神奇公式修改版 v2（①EBIT3年均值+②MA200趋势过滤+④行业上限）")
    p.add_argument("start_date", nargs="?", default="20140301")
    p.add_argument("end_date", nargs="?", default="20260727")
    p.add_argument("--top-n", type=int, default=15,
                   help="持仓数量（默认=config.GLOBAL 选股数，随全局设置走）")
    p.add_argument("--industry-cap", type=int, default=2)
    p.add_argument("--ebit-years", type=int, default=3)
    p.add_argument("--capital", type=int, default=None,
                   help="总资金（默认=config.BACKTEST 月度调仓资金，随全局设置走）")
    p.add_argument("--stock-pool", default=None,
                   help="股票池（默认=config.GLOBAL 股票池，随全局设置走）")
    p.add_argument("--no-trend", action="store_true", help="关闭②MA200趋势过滤")
    p.add_argument("--ma", type=int, default=200)
    p.add_argument("--half-ratio", type=float, default=0.5)
    p.add_argument("--breadth-confirm", action="store_true",
                   help="开启市场广度双重确认（道氏双腿）：指数MA200 + 成分股站上"
                        "各自MA200比例≥阈值。双腿=满仓/单腿=半仓/无腿=1/4仓。默认关闭")
    p.add_argument("--breadth-th", type=float, default=0.5,
                   help="广度阈值（默认 0.5 = 50%%成分股站上各自MA200）")
    p.add_argument("--profit-guard", type=float, default=0.0,
                   help="⑤利润回落护栏：最新年EBIT < 3年均值×该值 → 剔除。"
                        "建议 0.5；0=关闭（默认，保持 v2 基线）")
    p.add_argument("--ebit-conservative", action="store_true",
                   help="⑥保守EBIT口径：roc/ey 改用 min(最新年EBIT, 基准统计量)，"
                        "抑制一次性暴利在均值里赖3年导致的EY虚高")
    p.add_argument("--spike-guard", type=float, default=1.5,
                   help="⑧利润暴涨护栏：最新年EBIT > 3年均值×该值 → 剔除（正处"
                        "一次性暴利峰值）。建议 2.0；0=关闭（默认）")
    p.add_argument("--ebit-stat", default="mean", choices=["mean", "median", "min"],
                   help="⑦正常化统计量：mean(默认·原v2) / median(抗单年脉冲) / min(最保守)")
    a = p.parse_args()
    run_backtest_v2(a.start_date, a.end_date, top_n=a.top_n,
                    industry_cap=a.industry_cap, ebit_years=a.ebit_years,
                    capital=a.capital, stock_pool=a.stock_pool,
                    trend_filter=not a.no_trend, ma_window=a.ma,
                    half_ratio=a.half_ratio,
                    breadth_confirm=a.breadth_confirm, breadth_th=a.breadth_th,
                    profit_guard=a.profit_guard,
                    ebit_conservative=a.ebit_conservative,
                    spike_guard=a.spike_guard, ebit_stat=a.ebit_stat)
