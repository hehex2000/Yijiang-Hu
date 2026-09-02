"""
PEG 低估值成长策略 — 回测
========================
选股逻辑：PEG = PE(TTM) / 净利润同比增长率(netprofit_yoy, 已是百分比口径)
  - PEG 越低 = 成长"性价比"越高（便宜的增长）。
  - 选股日取「最新已可用年报」的 netprofit_yoy（point-in-time，公告日≤选股日防前视）。

视频《低PE是打折还是陷阱 / PEG 价值陷阱》三道护栏（P1 落地到可跑回测）：
  ① 无稳定利润 / 负增长：PE>0 且 netprofit_yoy>0 才有意义，否则 PEG=NaN（负增长算出的
     负 PEG 绝非"便宜"，放进来会被反向因子当超便宜加分 → 误选）。
  ② 基数效应：净利高增但营收没同步(or_yoy<=0) 或 ROE<=0，可能是低基数/一次性收益虚增
     → PEG 不可信，置 NaN。or_yoy 缺失时按"不否决"放行（本地库 or_yoy 普遍为空，设计内）。
  ③ 增长稳定性：最近 N 个年报(年末1231) 净利润同比增速须全部为正；or_yoy 仅当"全部年份
     均有值"时才要求>0（缺失则放行）。—— 注意：与 src/factor_calculator.py 原实现不同，
     原实现把 or_yoy 缺失也判为不稳定会全盘拒掉 PEG；此处按本地数据现实放宽，详见脚本说明。

回测引擎：复用 run_monthly_rebalance 共享引擎（get_conn/get_open_price/get_price/
calc_fee/get_trade_dates/compute_reality_discounts 等），与 run_magic_formula.py 同构。

数据口径：
  - PE：daily_basic.pe_ttm（选股日 T-1；缺失回退到最近前一交易日）
  - 增长/质量：fina_indicator（netprofit_yoy / or_yoy / roe），年报(end_date LIKE '%1231')
  - 防偏：ann_date 有效且距期末≤200天用公告日，否则回退期末+120天（同魔法公式）

用法：
  venv_ml/Scripts/python.exe run_peg.py
  venv_ml/Scripts/python.exe run_peg.py --freq monthly --topn 20 --start 20140101 --end 20260715
  venv_ml/Scripts/python.exe run_peg.py --pool hs300 --stab-years 2 --capital 500000
  venv_ml/Scripts/python.exe run_peg.py --min-roe 8 --max-debt 70      (质量叠加：ROE>=8% 且 负债率<=70%)
  venv_ml/Scripts/python.exe run_peg.py --momentum 12                  (动量叠加：要求12月价格动量>0)
  venv_ml/Scripts/python.exe run_peg.py --weight liquidity             (流动性加权替代等权)
  venv_ml/Scripts/python.exe run_peg.py --min-roe 8 --max-debt 70 --momentum 12   (最优版：质量+动量)
  venv_ml/Scripts/python.exe run_peg.py --min-roe 8 --max-debt 70 --momentum 12 --var-guard   (叠加 VaR95% 回撤控制)

推荐默认配置（已设为 run_peg.py 默认值，直接 `run_peg.py` 即跑此版）：
  质量(ROE>=8% 且 负债率<=70%) + 动量(12月>0) + VaR(95%) 回撤控制(日度上限2.5%)
  全历史 2014-2026：总收益 +140.4% / 年化 +7.89% / 最大回撤 -41.2% / 夏普 0.38
  —— 唯一跑赢基准(中证全指+109%/沪深300+106%)的变体，且把 -72% 回撤回撤到 -41%。
  关闭 VaR 风控：加 --no-var；调更严阈值：--var-cap 0.015（回撤更小但收益更低）。
"""
import sqlite3
import os
import sys
import bisect
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_monthly_rebalance import (
    get_conn, calc_fee, calc_win_rate, get_trade_dates,
    get_open_price, get_price, get_stock_name,
    INIT_CAPITAL, COMMISSION_RATE, COMMISSION_MIN, STAMP_DUTY_RATE,
    SLIPPAGE_RATE, INDEX_DISPLAY_NAME, compute_reality_discounts,
)

# ════════════════════════════════════════════════════════════
#  常量
# ════════════════════════════════════════════════════════════
TOP_N            = 30        # 持仓数量
REBALANCE_MONTH  = 5         # 年度调仓月（annual 模式）
IPO_MIN_DAYS     = 60
BENCHMARKS       = ["000985.SH", "000300.SH"]   # 中证全指 / 沪深300
_CAPITAL         = 1_000_000
STAB_YEARS      = 3         # 护栏③：连续正增长年数

FINANCIAL_INDUSTRIES = {"银行", "证券", "保险", "多元金融"}
UTILITY_INDUSTRIES   = {"火力发电", "水力发电", "新型电力", "供气供热", "水务"}

_POOL_IDX = {
    "hs300": "000300.SH", "zz500": "000905.SH", "zz800": "000906.SH",
    "zz1000": "000852.SH", "zz2000": "932000.SH", "all": None,
}

# 缓存
_BASIC      = None
_TRADE_CAL  = None
_FIN_ANNUAL = None   # ts_code -> [ {end_date, ann_date, np_yoy, op_yoy, roe}, ... ] 按期末升序
_STATS      = {"eligible": 0, "g1": 0, "g2": 0, "g3": 0, "selected": 0, "rebal": 0}
_CLOSE_MAP  = {}      # trade_date -> {ts_code: close}  每日批量取收盘价缓存（避免逐笔查库拖垮性能）
_SLOW_CONN  = None     # 复用单连接，避免每日循环反复 connect/close

def _slow_conn():
    global _SLOW_CONN
    if _SLOW_CONN is None:
        _SLOW_CONN = get_conn()
    return _SLOW_CONN


# ════════════════════════════════════════════════════════════
#  NAV 计价口径: raw(漏分红) / hfq(含分红再投)
# ════════════════════════════════════════════════════════════
#  只切换 NAV 侧(估值/卖出所得)。信号侧(动量/选股)恒用 raw, 保证双跑差异可归因。
#  买入侧恒用 raw 价: 整手判定与 affordability 必须用真实成交价。
#  买入日因子缺失 → 该持仓永久锁 1.0(绝不能拿 f(今)/1.0, 否则绝对因子值整段虚增)。
PRICE_MODE   = "raw"
_ADJ_CACHE   = {}


def _adj_series(ts_code):
    """整条因子序列按 code 一次性载入(返回 日期list/因子list, 升序)。
    ⚠️ 不能按 (code, date) 逐日查库: 19 仓 × 3046 天 = 5.8 万次 SQL, 单次回测要 45+ 分钟。
    按 code 预载后 bisect 做 as-of, 查询量从 O(持仓×天数) 降到 O(股票数)。"""
    e = _ADJ_CACHE.get(ts_code)
    if e is not None:
        return e
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT trade_date, adj_factor FROM adj_factor WHERE ts_code=? "
        "ORDER BY trade_date", conn, params=(ts_code,))
    conn.close()
    e = ((df["trade_date"].astype(str).tolist(),
          df["adj_factor"].astype(float).tolist()) if len(df) else None)
    _ADJ_CACHE[ts_code] = e
    return e


def _adj_factor(ts_code, trade_date):
    """as-of 后复权因子(<=date 最近一条)。天然 ffill, 自动继承整市场缺行日。"""
    e = _adj_series(ts_code)
    if not e:
        return None
    ds, fs = e
    i = bisect.bisect_right(ds, str(trade_date)) - 1
    return float(fs[i]) if i >= 0 else None


def _hfq_ratio(ts_code, buy_factor, trade_date):
    """持仓的 hfq 折算比 = f(今)/f(买入)。
    raw 模式恒 1.0; 买入日因子缺失 → 1.0(该持仓不再补分红, 保守)。"""
    if PRICE_MODE != "hfq":
        return 1.0
    if not buy_factor:
        return 1.0
    ft = _adj_factor(ts_code, trade_date)
    if not ft:
        return 1.0
    return float(ft) / float(buy_factor)


def _day_close(trade_date):
    """某交易日全市场收盘价（一次性查询并缓存）。用于每日市值循环，
    避免对每只持仓逐日调用 get_price（每次都新建连接）导致回测极慢。"""
    d = _CLOSE_MAP.get(trade_date)
    if d is None:
        df = pd.read_sql_query(
            "SELECT ts_code, close FROM daily WHERE trade_date = ?",
            _slow_conn(), params=(trade_date,))
        d = {}
        for _, r in df.iterrows():
            if pd.notna(r["close"]):
                d[str(r["ts_code"])] = float(r["close"])
        _CLOSE_MAP[trade_date] = d
    return d


def _momentum(code, prev_date, past_date):
    """N 个月价格动量 = close(prev_date)/close(past_date) - 1。
    用 _day_close 缓存（按交易日批量取全市场收盘），避免逐笔查库。
    past_date 由 select_peg 在 trade_dates 上按 21 交易日/月回退算出。"""
    if past_date is None:
        return np.nan
    cur = _day_close(prev_date).get(code)
    past = _day_close(past_date).get(code)
    if cur is None or past is None or past <= 0:
        return np.nan
    return cur / past - 1


def _liquidity_map(trade_date, codes):
    """候选股的流动性权重代理 = daily.amount（成交额）。返回 {ts_code: amount}。
    缺失时回退到最近前一交易日。"""
    if not codes:
        return {}
    conn = get_conn()
    try:
        q = ("SELECT ts_code, amount FROM daily WHERE trade_date=? AND ts_code IN (%s)"
             % ",".join("?" * len(codes)))
        df = pd.read_sql_query(q, conn, params=(trade_date, *codes))
        if len(df) == 0:
            r = conn.execute(
                "SELECT MAX(trade_date) FROM daily WHERE trade_date < ?",
                (trade_date,)).fetchone()
            if r and r[0]:
                df = pd.read_sql_query(q, conn, params=(r[0], *codes))
    finally:
        conn.close()
    m = {}
    for _, row in df.iterrows():
        a = row["amount"]
        if pd.notna(a) and a > 0:
            m[str(row["ts_code"])] = float(a)
    return m


# ════════════════════════════════════════════════════════════
#  基础数据
# ════════════════════════════════════════════════════════════
def _load_basic():
    global _BASIC
    if _BASIC is not None:
        return _BASIC
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT ts_code, name, industry, list_date FROM stock_basic", conn)
    conn.close()
    m = {}
    for _, r in df.iterrows():
        code = str(r["ts_code"])
        name = str(r["name"]) if pd.notna(r["name"]) else ""
        ind = str(r["industry"]) if pd.notna(r["industry"]) else None
        ld = str(r["list_date"]) if pd.notna(r["list_date"]) else ""
        excluded = (code.endswith(".BJ")
                    or "ST" in name.upper() or name.startswith("*"))
        m[code] = {"name": name, "industry": ind, "list_date": ld,
                   "excluded": excluded}
    _BASIC = m
    return m


def _load_fin_annual():
    """一次性预载全部年报(end_date 末4位=='1231')财务指标，按 ts_code 建索引。"""
    global _FIN_ANNUAL
    if _FIN_ANNUAL is not None:
        return _FIN_ANNUAL
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT ts_code, end_date, ann_date, netprofit_yoy, or_yoy, roe, debt_to_assets "
        "FROM fina_indicator WHERE end_date LIKE '%1231'", conn)
    conn.close()
    d = {}
    for _, r in df.iterrows():
        code = str(r["ts_code"])
        ed = str(r["end_date"])
        ad = r["ann_date"]
        ad_s = ""
        if pd.notna(ad):
            try:
                ad_s = str(int(round(float(ad))))
            except (TypeError, ValueError):
                ad_s = ""
        d.setdefault(code, []).append({
            "end_date": ed, "ann_date": ad_s,
            "np_yoy": None if pd.isna(r["netprofit_yoy"]) else float(r["netprofit_yoy"]),
            "op_yoy": None if pd.isna(r["or_yoy"]) else float(r["or_yoy"]),
            "roe": None if pd.isna(r["roe"]) else float(r["roe"]),
            "debt": None if pd.isna(r["debt_to_assets"]) else float(r["debt_to_assets"]),
        })
    for c in d:
        d[c].sort(key=lambda x: (x["end_date"], x["ann_date"]))
    _FIN_ANNUAL = d
    return d


def _avail(end_date, ann_date):
    """年报可用日（point-in-time 防偏核心）。同 run_magic_formula._avail。"""
    ed = datetime.strptime(end_date, "%Y%m%d")
    if ann_date and ann_date != "":
        try:
            ad = datetime.strptime(ann_date, "%Y%m%d")
            if ad <= ed + timedelta(days=200):
                return ad
        except (TypeError, ValueError):
            pass
    return ed + timedelta(days=120)


def _get_pe_map(prev_date):
    """选股日 T-1 的 pe_ttm（万元）。缺失回退最近前一交易日。"""
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT ts_code, pe_ttm FROM daily_basic WHERE trade_date = ?",
        conn, params=(prev_date,))
    if len(df) == 0:
        r = conn.execute(
            "SELECT MAX(trade_date) FROM daily_basic WHERE trade_date < ?",
            (prev_date,)).fetchone()
        if r and r[0]:
            df = pd.read_sql_query(
                "SELECT ts_code, pe_ttm FROM daily_basic WHERE trade_date = ?",
                conn, params=(r[0],))
    conn.close()
    m = {}
    for _, row in df.iterrows():
        if pd.notna(row["pe_ttm"]):
            m[str(row["ts_code"])] = float(row["pe_ttm"])
    return m


def _get_pool_constituents(stock_pool, asof_date):
    idx = _POOL_IDX.get(stock_pool)
    if idx is None:
        return None
    conn = get_conn()
    try:
        snap = conn.execute(
            "SELECT MAX(CAST(trade_date AS INTEGER)) FROM index_constituent "
            "WHERE index_code=? AND CAST(trade_date AS INTEGER) <= CAST(? AS INTEGER)",
            (idx, asof_date)).fetchone()
        if not snap or snap[0] is None:
            print(f"  [⚠️] 股票池 {stock_pool}({idx}) 在 {asof_date} 无成分快照，退回全A股")
            return None
        rows = conn.execute(
            "SELECT ts_code FROM index_constituent WHERE index_code=? "
            "AND CAST(trade_date AS INTEGER)=CAST(? AS INTEGER)",
            (idx, snap[0])).fetchall()
    finally:
        conn.close()
    return set(str(r[0]) for r in rows)


# ════════════════════════════════════════════════════════════
#  PEG 计算 + 三护栏
# ════════════════════════════════════════════════════════════
def _peg_with_guardrails(code, visible_date, pe, stab_years):
    """返回 PEG(float) 或 np.nan（被任意护栏否决）。
    pe: 选股日 T-1 的 pe_ttm。visible_date: T-1 日期。
    """
    recs = _FIN_ANNUAL.get(code)
    if not recs or pe is None or not np.isfinite(pe):
        return np.nan
    vd = datetime.strptime(visible_date, "%Y%m%d")
    avail_recs = [x for x in recs if _avail(x["end_date"], x["ann_date"]) <= vd]
    if not avail_recs:
        return np.nan
    latest = avail_recs[-1]
    np_yoy = latest["np_yoy"]
    op_yoy = latest["op_yoy"]
    roe = latest["roe"]

    # 护栏① 无稳定利润 / 负增长
    if pe <= 0 or np_yoy is None or np_yoy <= 0:
        return np.nan
    peg = pe / np_yoy

    # 护栏② 基数效应（or_yoy 缺失→放行）
    if op_yoy is not None and (op_yoy <= 0 or (roe is not None and roe <= 0)):
        return np.nan

    # 护栏③ 增长稳定性
    tail = avail_recs[-stab_years:]
    # 3a. 净利润同比增速全部为正（核心）
    if not all(x["np_yoy"] is not None and x["np_yoy"] > 0 for x in tail):
        return np.nan
    # 3b. or_yoy 仅当全部年份均有值时要求>0，缺失则放行
    have_op = all(x["op_yoy"] is not None for x in tail)
    if have_op and not all(x["op_yoy"] > 0 for x in tail):
        return np.nan
    return peg


def select_peg(rebalance_date, top_n=TOP_N, prev_date=None, verbose=True,
               stock_pool="all", stab_years=STAB_YEARS,
               min_roe=0.0, max_debt=100.0, momentum_months=0, trade_dates=None):
    """返回 DataFrame: [ts_code, pe, np_yoy, peg]，按 peg 升序（最低=最便宜成长）。
    叠加层：min_roe/max_debt(质量)、momentum_months(>0 要求 N 月动量上行)。"""
    if prev_date is None:
        prev_date = rebalance_date
    basic = _load_basic()
    _load_fin_annual()
    conn = get_conn()

    pool_set = (_get_pool_constituents(stock_pool, prev_date)
                if stock_pool and stock_pool != "all" else None)
    if pool_set is not None and verbose:
        print(f"  [股票池] {stock_pool} 时点成分 {len(pool_set)} 只（asof {prev_date}）")

    rows = pd.read_sql_query(
        "SELECT DISTINCT ts_code FROM daily WHERE trade_date = ?",
        conn, params=(prev_date,))
    trading = set(str(c) for c in rows["ts_code"].tolist())
    conn.close()

    eligible = set()
    d_rb = datetime.strptime(prev_date, "%Y%m%d")
    for c in trading:
        if pool_set is not None and c not in pool_set:
            continue
        info = basic.get(c)
        if info is None:
            if c.endswith(".BJ"):
                continue
            eligible.add(c)
            continue
        if info["excluded"]:
            continue
        ind = info["industry"]
        if ind in FINANCIAL_INDUSTRIES or ind in UTILITY_INDUSTRIES:
            continue
        ld = info["list_date"]
        if ld:
            try:
                if (d_rb - datetime.strptime(ld, "%Y%m%d")).days < IPO_MIN_DAYS:
                    continue
            except Exception:
                pass
        eligible.add(c)

    if not eligible:
        return pd.DataFrame()

    pe_map = _get_pe_map(prev_date)
    _STATS["eligible"] += len(eligible)

    # 动量回看日期（按 21 交易日/月回退）
    past_date = None
    if momentum_months and momentum_months > 0 and trade_dates is not None and prev_date in trade_dates:
        i = trade_dates.index(prev_date)
        n = int(round(momentum_months * 21))
        if i - n >= 0:
            past_date = trade_dates[i - n]

    recs = []
    g1 = g3 = 0
    for c in eligible:
        pe = pe_map.get(c)
        # 逐护栏统计（便于诊断护栏松紧）
        recs_fin = _FIN_ANNUAL.get(c)
        if not recs_fin or pe is None or not np.isfinite(pe):
            continue
        vd = datetime.strptime(prev_date, "%Y%m%d")
        ar = [x for x in recs_fin if _avail(x["end_date"], x["ann_date"]) <= vd]
        if not ar:
            continue
        lat = ar[-1]
        if pe > 0 and lat["np_yoy"] is not None and lat["np_yoy"] > 0:
            g1 += 1
        peg = _peg_with_guardrails(c, prev_date, pe, stab_years)
        if np.isnan(peg):
            continue
        # 质量叠加：ROE 下限 / 资产负债率上限
        if min_roe > 0 and (lat["roe"] is None or lat["roe"] < min_roe):
            continue
        if max_debt < 100 and (lat["debt"] is None or lat["debt"] > max_debt):
            continue
        # 动量叠加：要求 N 月价格动量 > 0（上升趋势，避开下行通道中的"便宜"）
        if past_date is not None:
            mom = _momentum(c, prev_date, past_date)
            if not np.isfinite(mom) or mom <= 0:
                continue
        g3 += 1
        recs.append((c, pe, lat["np_yoy"], peg))
    # g2 近似 = g1 中未被基数效应否决者；用 g3/g1 推断即可，这里直接记 g1 与最终
    _STATS["g1"] += g1
    _STATS["g3"] += g3

    if not recs:
        if verbose:
            print(f"  [选股 {rebalance_date}] 候选池 {len(eligible)} 只，"
                  f"过护栏① {g1} 只 → 过护栏③ {g3} 只 → 取 0 只")
        return pd.DataFrame()

    df = pd.DataFrame(recs, columns=["ts_code", "pe", "np_yoy", "peg"])
    df = df.sort_values("peg").head(top_n).reset_index(drop=True)
    _STATS["selected"] += len(df)

    if verbose:
        print(f"  [选股 {rebalance_date}] 候选池 {len(eligible)} 只 → "
              f"过护栏① {g1} → 过护栏③ {g3} → 取 {top_n} 只")
    return df


# ════════════════════════════════════════════════════════════
#  回测引擎
# ════════════════════════════════════════════════════════════
def _rebalance_dates(trade_dates, freq, month=REBALANCE_MONTH):
    by_key = {}
    for td in trade_dates:
        key = td[:4] if freq == "annual" else td[:6]
        by_key.setdefault(key, []).append(td)
    out = set()
    for key, ds in by_key.items():
        # 第5个交易日（不足5个取最后）
        out.add(ds[4] if len(ds) >= 5 else ds[-1])
    return out


def run_backtest(start_date="20140101", end_date="20260715",
                 top_n=TOP_N, verbose=True, stock_pool="all",
                 freq="annual", stab_years=STAB_YEARS,
                 interrupt_start=None, interrupt_months=0, interrupt_pct=0.0,
                 min_roe=0.0, max_debt=100.0, momentum_months=0, weight="equal",
                 var_guard=False, var_cap=0.025, var_window=60):
    overlay = []
    if stock_pool and stock_pool != "all":
        overlay.append(f"股票池={stock_pool}")
    if min_roe > 0:
        overlay.append(f"质量ROE>={min_roe:.0f}%")
    if max_debt < 100:
        overlay.append(f"负债率<={max_debt:.0f}%")
    if momentum_months and momentum_months > 0:
        overlay.append(f"动量{int(momentum_months)}月>0")
    if weight == "liquidity":
        overlay.append("流动性加权")
    if var_guard:
        overlay.append(f"VaR95风控(上限{var_cap*100:.1f}%·窗{var_window}d)")
    overlay_str = " | ".join(overlay) if overlay else "无(纯PEG+三护栏)"
    print("=" * 72)
    print("  PEG 低估值成长策略回测（PE÷净利增速 + 价值陷阱三护栏）")
    print("=" * 72)
    print(f"  区间：{start_date} ~ {end_date}")
    wstr = "等权" if weight == "equal" else "流动性加权"
    print(f"  持仓：{top_n}只{wstr} | 调仓：{freq}(月{REBALANCE_MONTH}第5交易日) | "
          f"护栏③连续正增长年数：{stab_years}")
    print(f"  叠加层：{overlay_str}")
    print(f"  剔除：ST / .BJ / 金融 / 公用事业 / 上市<60天")
    print(f"  佣金万{COMMISSION_RATE*1e4:.1f}(最低{COMMISSION_MIN}) "
          f"印花税千1→千0.5(2023-08-28起) 滑点{SLIPPAGE_RATE*100:.1f}%")
    print(f"  初始资金：{_CAPITAL:,.0f}")
    print(f"  NAV 计价口径：{'hfq(后复权·含分红再投)' if PRICE_MODE == 'hfq' else 'raw(原始价·漏分红)'}"
          f"   [信号侧/买入成交价恒用 raw]")
    print()

    trade_dates = get_trade_dates(start_date, end_date)
    rebal_set = _rebalance_dates(trade_dates, freq)
    rebal_set = {d for d in rebal_set if start_date <= d <= end_date}
    print(f"  交易日 {len(trade_dates)} 天，调仓 {len(rebal_set)} 次\n")

    positions = {}
    cash = float(_CAPITAL)
    daily_vals = []
    trades = []
    name_cache = {}

    # VaR(95%) 风控层状态
    risk_off = False        # 当前是否已清仓至现金
    last_selected = []      # 最近一次调仓选出的代码（用于风控解除后回补参考）
    nav_hist = []           # 每日组合净值序列（算 VaR 用）
    last_month = None       # 上月标记，用于月度边界检测
    pending = "none"        # 待执行的风控动作：off=清仓 / on=回补（下个交易日开盘执行）
    var_off_days = 0        # 处于现金状态的天数（报告用）

    def _name(code):
        if code not in name_cache:
            name_cache[code] = get_stock_name(code)
        return name_cache[code]

    for i, td in enumerate(trade_dates):
        # ---- VaR 风控：上月边界触发的清仓动作在今日开盘执行（非调仓日）----
        if pending == "off" and i > 0 and td not in rebal_set:
            for code in list(positions.keys()):
                op = get_open_price(code, td)
                if op is None:
                    continue
                pos = positions[code]
                sell_px = op * _hfq_ratio(code, pos.get("buy_factor"), td)
                fee = calc_fee('sell', sell_px, pos["shares"])
                cash += pos["shares"] * sell_px - fee
                trades.append({"date": td, "action": "SELL", "code": code,
                               "name": _name(code), "price": op,
                               "shares": pos["shares"], "reason": "var_guard_off"})
                del positions[code]
            risk_off = True
            pending = "none"
            if verbose:
                print(f"  ⚠️ VaR(95%) 触发，清空至现金 @ {td}")

        if td in rebal_set and i > 0:
            prev_td = trade_dates[i - 1]
            sel = select_peg(td, top_n=top_n, prev_date=prev_td, verbose=verbose,
                             stock_pool=stock_pool, stab_years=stab_years,
                             min_roe=min_roe, max_debt=max_debt,
                             momentum_months=momentum_months, trade_dates=trade_dates)
            new_codes = sel["ts_code"].tolist() if not sel.empty else []
            new_set = set(new_codes)
            _STATS["rebal"] += 1
            last_selected = new_codes
            risk_off = False
            pending = "none"

            if not new_codes:
                if verbose:
                    print(f"\n调仓日 {td}：选股为空，保持现有 {len(positions)} 仓")
            else:
                cur = set(positions.keys())
                if cur != new_set:
                    for code in list(positions.keys()):
                        if code not in new_set:
                            op = get_open_price(code, td)
                            if op is None:
                                continue
                            pos = positions[code]
                            sell_px = op * _hfq_ratio(code, pos.get("buy_factor"), td)
                            fee = calc_fee('sell', sell_px, pos["shares"])
                            cash += pos["shares"] * sell_px - fee
                            trades.append({"date": td, "action": "SELL",
                                           "code": code, "name": _name(code),
                                           "price": op, "shares": pos["shares"],
                                           "reason": "rebalance"})
                            if verbose:
                                print(f"  ✅ 调仓卖出 {code}({_name(code)})："
                                      f"{pos['shares']}股 @ {op:.2f}")
                            del positions[code]
                    to_buy = [c for c in new_codes if c not in positions]
                    if to_buy:
                        if weight == "liquidity":
                            liq = _liquidity_map(prev_td, to_buy)
                            tot = sum(liq.values())
                            if tot > 0:
                                cash_for = {c: cash * (liq.get(c, 0) / tot) for c in to_buy}
                            else:
                                cash_for = {c: cash / len(to_buy) for c in to_buy}
                        else:
                            cash_for = {c: cash / len(to_buy) for c in to_buy}
                        for code in to_buy:
                            op = get_open_price(code, td)
                            if op is None:
                                continue
                            budget = cash_for[code]
                            max_shares = int(budget / op / 100) * 100
                            if max_shares < 100:
                                continue
                            cost = max_shares * op
                            fee = calc_fee('buy', op, max_shares)
                            if cost + fee <= cash:
                                cash -= cost + fee
                                positions[code] = {"shares": max_shares,
                                                   "buy_price": op, "last_price": op,
                                                   # 买入价/整手判定恒用 raw(真实 affordability);
                                                   # buy_factor 只用于 NAV 侧折算
                                                   "buy_factor": (_adj_factor(code, td)
                                                                  if PRICE_MODE == "hfq"
                                                                  else None)}
                                trades.append({"date": td, "action": "BUY",
                                               "code": code, "name": _name(code),
                                               "price": op, "shares": max_shares,
                                               "reason": "peg_low"})
                                if verbose:
                                    print(f"  ✅ 买入 {code}({_name(code)})："
                                          f"{max_shares}股 @ {op:.2f}")
                else:
                    if verbose:
                        print(f"\n调仓日 {td}：持仓不变")

        total = cash
        close_map = _day_close(td)
        for code, pos in list(positions.items()):
            px = close_map.get(code)
            if px is None:
                px = pos.get("last_price") or 0
            else:
                pos["last_price"] = px
            total += pos["shares"] * px * _hfq_ratio(code, pos.get("buy_factor"), td)
        daily_vals.append({"date": td, "value": total})
        nav_hist.append(total)
        if risk_off:
            var_off_days += 1

        # ---- VaR(95%) 月度监测：当日度 VaR(95%) 突破上限 → 下个交易日清仓至现金 ----
        if var_guard and len(nav_hist) >= var_window:
            m = td[:6]
            if m != last_month:
                last_month = m
                window = nav_hist[-var_window:]
                if len(window) >= 2:
                    rets = np.diff(window) / np.array(window[:-1])
                    rets = rets[np.isfinite(rets)]
                    if len(rets) > 0:
                        v = -np.percentile(rets, 5)
                        if not risk_off and v > var_cap:
                            pending = "off"
                            if verbose:
                                print(f"  [VaR] {td} 月度VaR(95%)={v*100:.2f}% > "
                                      f"上限{var_cap*100:.2f}% → 下交易日清仓")

    if trade_dates:
        last = trade_dates[-1]
        last_close = _day_close(last)
        for code in list(positions.keys()):
            px = last_close.get(code)
            if px is not None:
                pos = positions[code]
                sell_px = px * _hfq_ratio(code, pos.get("buy_factor"), last)
                fee = calc_fee('sell', sell_px, pos["shares"])
                cash += pos["shares"] * sell_px - fee
                trades.append({"date": last, "action": "SELL", "code": code,
                               "name": _name(code), "price": px,
                               "shares": pos["shares"], "reason": "backtest_end"})
                del positions[code]
        if daily_vals:
            daily_vals[-1]["value"] = cash

    return _report(daily_vals, trades, trade_dates, start_date, end_date,
                   top_n=top_n, freq=freq, stab_years=stab_years,
                   interrupt_start=interrupt_start,
                   interrupt_months=interrupt_months,
                   interrupt_pct=interrupt_pct,
                   stock_pool=stock_pool, min_roe=min_roe, max_debt=max_debt,
                   momentum_months=momentum_months, weight=weight,
                   var_guard=var_guard, var_cap=var_cap, var_window=var_window,
                   var_off_days=var_off_days)


def _yearly_returns(daily_vals):
    if not daily_vals:
        return {}
    df = pd.DataFrame(daily_vals)
    df["year"] = df["date"].str[:4]
    yrs = {}
    for y, g in df.groupby("year"):
        yrs[y] = (g["value"].iloc[-1] / g["value"].iloc[0] - 1) * 100
    return yrs


def _report(daily_vals, trades, trade_dates, start_date, end_date, top_n=TOP_N,
            freq="annual", stab_years=STAB_YEARS,
            interrupt_start=None, interrupt_months=0, interrupt_pct=0.0,
            stock_pool="all", min_roe=0.0, max_debt=100.0,
            momentum_months=0, weight="equal",
            var_guard=False, var_cap=0.025, var_window=60, var_off_days=0):
    final_value = daily_vals[-1]["value"] if daily_vals else _CAPITAL
    total_return = (final_value / _CAPITAL - 1) * 100
    days = len(trade_dates)
    years = days / 252
    annual_return = ((final_value / _CAPITAL) ** (1 / years) - 1) * 100 if years > 0 else 0

    vals = np.array([d["value"] for d in daily_vals], dtype=float)
    cummax = np.maximum.accumulate(vals)
    safe = np.where(cummax == 0, 1, cummax)
    max_dd = float(np.min((vals - cummax) / safe)) * 100

    rets = np.diff(vals) / np.where(vals[:-1] == 0, 1, vals[:-1])
    sharpe = ((np.mean(rets) * 252 - 0.025) / (np.std(rets) * np.sqrt(252))
              if len(rets) > 1 and np.std(rets) > 0 else 0.0)
    # 策略自身日度 VaR(95%)（报告用，与风控阈值口径一致）
    var95_daily = float(-np.percentile(rets, 5)) if len(rets) > 1 else 0.0
    var95_annual = var95_daily * np.sqrt(252)

    win_rate, win_cnt, tot_cnt = calc_win_rate(trades)

    bench = {}
    conn = get_conn()
    for idx in BENCHMARKS:
        b = pd.read_sql_query(
            "SELECT close FROM index_daily WHERE ts_code=? AND trade_date>=? "
            "ORDER BY trade_date ASC LIMIT 1", conn, params=(idx, trade_dates[0]))
        e = pd.read_sql_query(
            "SELECT close FROM index_daily WHERE ts_code=? AND trade_date<=? "
            "ORDER BY trade_date DESC LIMIT 1", conn, params=(idx, trade_dates[-1]))
        if len(b) > 0 and len(e) > 0:
            bench[idx] = (float(e.iloc[0]["close"]) / float(b.iloc[0]["close"]) - 1) * 100
    conn.close()

    print(f"\n{'=' * 72}")
    print(f"  回测结果")
    print(f"{'=' * 72}")
    print(f"  初始资金：{_CAPITAL:,.0f}")
    print(f"  最终资产：{final_value:,.0f}")
    print(f"  总收益率：{total_return:+.2f}%")
    print(f"  年化收益：{annual_return:+.2f}%")
    print(f"  最大回撤：{max_dd:.2f}%")
    print(f"  夏普比率：{sharpe:.2f}")
    if tot_cnt > 0:
        print(f"  胜率：{win_rate:.1f}%（{win_cnt}/{tot_cnt}）")
    print(f"  交易次数：{len(trades)}")
    for idx, r in bench.items():
        print(f"  {INDEX_DISPLAY_NAME.get(idx, idx)}：{r:+.2f}%  超额：{total_return - r:+.2f}%")
    print(f"  策略 VaR(95%)：日度 {var95_daily*100:.2f}% ｜ 年化 {var95_annual*100:.2f}%")
    if var_guard:
        print(f"  VaR(95%) 风控：上限日度 {var_cap*100:.2f}%（窗{var_window}d）｜ "
              f"期间处于现金 {var_off_days} 天（占 {var_off_days/max(days,1)*100:.1f}%）")

    # 护栏通过统计
    s = _STATS
    if s["rebal"] > 0:
        print(f"\n{'—' * 72}")
        print(f"  护栏通过统计（累计 {s['rebal']} 次调仓）")
        print(f"    候选池合计：{s['eligible']} 只次")
        print(f"    过护栏①(PE>0且净利增)：{s['g1']} 只次")
        print(f"    过全部护栏(①②③)：{s['g3']} 只次")
        print(f"    最终入选：{s['selected']} 只次（= {s['selected']//max(s['rebal'],1)} 只/次 均值）")

    # 现实折扣三件套
    disc = compute_reality_discounts(
        daily_vals, _CAPITAL,
        interrupt_start=interrupt_start,
        interrupt_months=interrupt_months,
        interrupt_pct=interrupt_pct,
    )
    if "real_total_return" in disc:
        print(f"  ── 现实折扣（预期管理，不改收益计算）──")
        print(f"  扣通胀真实总收益：{disc['real_total_return']:+.2f}% ｜ 真实年化：{disc['real_annual_return']:+.2f}%")
    if "dca_drag_pct" in disc:
        print(f"  定投对比(DCA)：一次性建仓较分12月定投 {disc['dca_drag_pct']:+.2f}%"
              f"（正=一次性占优·负=定投占优）｜ 终值 一次性 {disc['dca_lump_final']:,.0f} / 定投 {disc['dca_dca_final']:,.0f}")
    if "interrupt_loss_pct" in disc:
        print(f"  中断模拟：{interrupt_start}起撤{interrupt_pct*100:.0f}%持有{interrupt_months}月，"
              f"终值损失 {disc['interrupt_loss_pct']:+.2f}%（终值 {disc['interrupt_final']:,.0f}）")

    yr = _yearly_returns(daily_vals)
    if yr:
        print(f"\n{'—' * 72}")
        print(f"  逐年收益（组合）")
        for y in sorted(yr):
            print(f"    {y}: {yr[y]:+.2f}%")

    os.makedirs("data/results/peg", exist_ok=True)
    var_tag = f"_v{int(var_cap*1000)}" if var_guard else ""
    _pm = "_hfq" if PRICE_MODE == "hfq" else ""
    tag = (f"n{top_n}_c{int(_CAPITAL)}_{freq}_s{stab_years}"
           f"_p{stock_pool}_r{int(min_roe)}_d{int(max_debt)}_m{int(momentum_months)}_w{weight}{_pm}"
           f"{var_tag}"
           f"_{start_date}_{end_date}")
    csv_path = f"data/results/peg/backtest_{tag}.csv"
    pd.DataFrame(daily_vals).to_csv(csv_path, index=False)
    pd.DataFrame(trades).to_csv(csv_path.replace("backtest_", "trades_"), index=False)
    print(f"\n  结果已保存：{csv_path}")

    return {
        "final_value": final_value, "total_return": total_return,
        "annual_return": annual_return, "max_drawdown": max_dd,
        "sharpe": sharpe, "win_rate": win_rate, "trades": len(trades),
        "var95_daily": var95_daily, "var95_annual": var95_annual,
        "var_off_days": var_off_days,
        "bench": bench, "yearly": yr, "stats": dict(_STATS),
    }


# ════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="PEG 低估值成长策略回测")
    ap.add_argument("--start", default="20140101")
    ap.add_argument("--end", default="20260715")
    ap.add_argument("--topn", type=int, default=TOP_N)
    ap.add_argument("--capital", type=float, default=_CAPITAL)
    ap.add_argument("--freq", choices=["annual", "monthly"], default="annual")
    ap.add_argument("--pool", default="all",
                    help="all / hs300 / zz500 / zz800 / zz1000 / zz2000")
    ap.add_argument("--stab-years", type=int, default=STAB_YEARS,
                    help="护栏③：要求连续正增长的年报年数")
    ap.add_argument("--min-roe", type=float, default=8.0,
                    help="质量叠加：要求最新 ROE>=该值(%)，0=不启用（默认8）")
    ap.add_argument("--max-debt", type=float, default=70.0,
                    help="质量叠加：要求资产负债率<=该值(%)，100=不启用（默认70）")
    ap.add_argument("--momentum", type=int, default=12,
                    help="动量叠加：要求 N 个月价格动量>0，0=不启用（默认12）")
    ap.add_argument("--weight", choices=["equal", "liquidity"], default="equal",
                    help="仓位加权：equal 等权 / liquidity 按成交额加权")
    ap.add_argument("--no-var", action="store_true",
                    help="关闭默认开启的 VaR(95%) 风控（回撤控制兜底，默认开启）")
    ap.add_argument("--var-cap", type=float, default=0.025,
                    help="VaR(95%) 日度上限（小数），默认 0.025=2.5%/日")
    ap.add_argument("--var-window", type=int, default=60,
                    help="VaR 计算回看天数，默认 60")
    ap.add_argument("--price-mode", choices=["raw", "hfq"], default="raw",
                    help="NAV 计价口径: raw=原始价(漏分红, 旧行为) / "
                         "hfq=后复权(含分红再投, 正确总回报)。信号侧恒用 raw。")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--interrupt-start", default=None)
    ap.add_argument("--interrupt-months", type=int, default=0)
    ap.add_argument("--interrupt-pct", type=float, default=0.0)
    args = ap.parse_args()

    # if __name__ 块内赋值即改模块全局, 此处不需要(也不能)加 global
    _CAPITAL = args.capital  # 覆盖模块级常量供报告使用
    PRICE_MODE = args.price_mode
    run_backtest(
        start_date=args.start, end_date=args.end, top_n=args.topn,
        verbose=args.verbose, stock_pool=args.pool, freq=args.freq,
        stab_years=args.stab_years,
        interrupt_start=args.interrupt_start,
        interrupt_months=args.interrupt_months,
        interrupt_pct=args.interrupt_pct,
        min_roe=args.min_roe, max_debt=args.max_debt,
        momentum_months=args.momentum, weight=args.weight,
        var_guard=not args.no_var, var_cap=args.var_cap, var_window=args.var_window,
    )
