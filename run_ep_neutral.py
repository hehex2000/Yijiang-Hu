"""
纯行业中性 EP（Earnings Yield = 1/PE_TTM）月度选股策略 — 回测
==============================================================
选股逻辑（单因子 · 价值因子 EP 的行业中性版）：
  EP = 1 / PE_TTM   （PE_TTM>0 才有意义，负值/亏损股剔除）
  ① 全局 1%/99% 缩尾（Winsorize），抑制极端估值
  ② 在每个申万细分行业（stock_basic.industry）内，按 EP 降序 5 分组
  ③ 取每组「最便宜五分位」G5（EP 最高 = 估值最低）
  ④ 跨行业等权持有全部 G5 —— 这就是该因子本身的多头组合

调仓：每月第 5 个交易日，T-1 日数据选股、T 日开盘价执行；
     月度换手约 1/3（与独立研究一致），"持有至掉出 G5 才卖出、新进 G5 才买入"。

防偏措施（与平台其它策略一致）：
  · T-1 数据选股、T 日开盘执行，杜绝日内前视
  · 剔除 ST / .BJ(北交所) / 金融 / 公用事业 / 上市<60天
  · pe_ttm 来自 daily_basic 的 T-1 交易日，point-in-time 无前视

口径说明：
  上一轮独立研究（ep_factor_neutral_cost.py）的「多空净 10.54%/t=3.74」
  是 G5 多头 − G1 空头的行业中性多空组合（因子 alpha 度量）。
  本脚本落地的是**可实盘的多头组合**：仅持有行业中性后的便宜五分位 G5，
  故其多头收益口径与研究中的「多空」不同，但同源同因子、月度调仓一致。
"""
import sqlite3
import os
import sys
from datetime import datetime

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_index as bi   # 统一基准真相源：口径自动跟随 NAV（raw→价格指数 / hfq→全收益）
from run_monthly_rebalance import (
    get_conn, calc_fee, calc_win_rate, get_trade_dates,
    get_monthly_5th_trading_days, obv_accumulation_filter,
    INIT_CAPITAL, COMMISSION_RATE, COMMISSION_MIN, STAMP_DUTY_RATE,
    SLIPPAGE_RATE, INDEX_DISPLAY_NAME, compute_reality_discounts,
    set_trade_date_ctx, reset_fee_ctx,
)

# ════════════════════════════════════════════════════════════
#  常量
# ════════════════════════════════════════════════════════════
TOP_N            = None      # None = 持有全部 G5；也可 --top-n 设集中度上限
OBV_FILTER       = None      # None=关闭；整数=方案A OBV吸筹过滤窗口(交易日)
IPO_MIN_DAYS     = 60        # 上市<60天剔除
BENCHMARKS       = ["000985.SH", "000300.SH"]   # 中证全指 / 沪深300

# 因子组合通常较分散（G5 约数百只），需足额资金让等权充分建仓：
# 500万 ÷ ~600只 ≈ 8300/只，100股/手 → 单价<83元的 G5 票均可买，现金拖累小。
# 研究里的等权是无约束算术平均，故资金越大越接近研究口径（可用 --capital 提高）。
_CAPITAL         = 5_000_000

# 行业剔除（基于 stock_basic.industry）
FINANCIAL_INDUSTRIES = {"银行", "证券", "保险", "多元金融"}
UTILITY_INDUSTRIES   = {"火力发电", "水力发电", "新型电力", "供气供热", "水务"}

# 缩尾分位 & 分组数
WINSOR_LO, WINSOR_HI = 0.01, 0.99
QUINTILES = 5

# 盘后定价交易分界（2026-07-06 起开盘价改用收盘价，与引擎一致）
_POST_2026_OPEN_AS_CLOSE = 20260706

# ════════════════════════════════════════════════════════════
#  执行层 realism 参数
# ════════════════════════════════════════════════════════════
EXEC_PRICE = "open"     # 成交价假设: "open"(开盘价) 或 "vwap"(日级VWAP代理)

# ════════════════════════════════════════════════════════════
#  NAV 计价口径（2026-09-01 新增；2026-09-02 起默认 hfq，raw 变 opt-in）
# ════════════════════════════════════════════════════════════
# "raw" = 不复权（旧行为）：NAV **不含** 现金分红，且**不处理送转股**
#         → 持仓 shares 从不随送转增加，除权日市值凭空蒸发 (1 - 1/送转比例)，
#           单次 17%~45%（实测 300853.SZ 20210610 凭空亏损 -42.02pp），一击击穿 -15% 止损。
# "hfq" = 后复权（含分红再投 + 天然免疫送转）：NAV 是**总回报**口径。
#         必须与同口径基准比较 → 见 bench_index.resolve_mode("auto", PRICE_MODE)。
# ⚠️ 改计价口径必须同步改比较基准，两端必须同含或同不含分红。
PRICE_MODE = "hfq"
_ADJ_LAST = {}          # code -> 最近已知 adj_factor（跨交易日 ffill）
_ADJ_REF  = {}          # code -> 归一化基准因子（该股首个已知因子，首日 hfq 严格 == raw）


def reset_price_cache():
    """清空价格缓存（换回测区间时必须调用，否则 _ADJ_REF 会沿用上一段的归一化基准）。"""
    _DAY_PX.clear()
    _ADJ_LAST.clear()
    _ADJ_REF.clear()


def _scale(code, v, adj):
    """把 raw 价换算到 NAV 计价空间。

    hfq: v × adj_t / adj_ref —— 除以基准因子是为了**归一化**：
    否则绝对价位会被累计因子放大数十倍（实测 000651.SZ raw 67.9 → hfq 10589），
    而按整股下单 int(cash // px) 会导致买入 0 股 → 伪现金拖累。
    归一化后首日 hfq 严格等于 raw，之后差额 = 真实分红 + 送转贡献。
    """
    if PRICE_MODE != "hfq" or v is None or not adj:
        return v
    ref = _ADJ_REF.get(code) or adj
    return v * float(adj) / float(ref)


LIMIT_ON   = True       # 涨跌停约束: 涨停买不进、跌停卖不出
SLIPPAGE   = 0.001      # 滑点率(与 calc_fee 一致, 可通过 --slippage 调整)
LIMIT_UP_PCT   = 9.8    # 涨停阈值(覆盖10%/20%板块, pct_chg>=此值视为涨停)
LIMIT_DOWN_PCT = -9.8   # 跌停阈值

# ════════════════════════════════════════════════════════════
#  缓存
# ════════════════════════════════════════════════════════════
_BASIC      = None   # ts_code -> {industry, name, list_date, excluded}
_DAY_PX     = {}     # trade_date -> {ts_code: (open, close)}  批量缓存

_POOL_IDX = {
    "hs300": "000300.SH", "zz500": "000905.SH", "zz800": "000906.SH",
    "zz1000": "000852.SH", "zz2000": "932000.SH", "all": None,
}


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
#  按交易日批量取价（缓存，避免逐笔 DB 查询）
# ════════════════════════════════════════════════════════════
def _ensure_day(td):
    """批量取某日所有股票的 (open, close, high, low, amount, vol, pct_chg, adj) 并缓存（每天仅 1 次查询）。

    第 8 位 adj 仅在 PRICE_MODE=="hfq" 时有值，供 _scale() 换算到后复权空间。
    三个必须守住的口径细节：
      ① adj_factor 是**阶跃函数**（仅除权除息日变化），缺行必须 **ffill**，
         绝不能 fillna(1.0) —— 那会让缺行日价格掉回不复权、次日跳回复权，
         制造百倍级假跳空（实测某股 ffill=67.91 vs fillna(1.0)=0.4355，差 156 倍）。
         该表 2020-2026 有 132 个**整交易日全市场同缺**，不是个别股问题。
      ② pct_chg **永不缩放**：涨跌停是市场机制，判定必须基于实际成交价，
         与 NAV 计价口径无关。raw 与 hfq 下涨跌停判定结果完全一致。
      ③ adj 缺行时继承 _ADJ_LAST（跨交易日），并在首次见到该股时登记 _ADJ_REF。
    """
    if td in _DAY_PX:
        return _DAY_PX[td]
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT ts_code, open, close, high, low, amount, vol, pct_chg "
        "FROM daily WHERE trade_date = ?",
        conn, params=(td,))
    adj_map = {}
    if PRICE_MODE == "hfq":
        adf = pd.read_sql_query(
            "SELECT ts_code, adj_factor FROM adj_factor WHERE trade_date = ?",
            conn, params=(td,))
        adj_map = {str(r["ts_code"]): float(r["adj_factor"])
                   for _, r in adf.iterrows() if pd.notna(r["adj_factor"])}
    conn.close()
    d = {}
    for _, r in df.iterrows():
        c = str(r["ts_code"])
        o = float(r["open"]) if pd.notna(r["open"]) else None
        cl = float(r["close"]) if pd.notna(r["close"]) else None
        h = float(r["high"]) if pd.notna(r["high"]) else None
        l = float(r["low"]) if pd.notna(r["low"]) else None
        amt = float(r["amount"]) if pd.notna(r["amount"]) else None
        vol = float(r["vol"]) if pd.notna(r["vol"]) else None
        pct = float(r["pct_chg"]) if pd.notna(r["pct_chg"]) else None
        # adj: 当日有则更新 _ADJ_LAST，无（整表缺行/该股未覆盖）则继承 → 跨日 ffill
        a = adj_map.get(c)
        if a is not None and a > 0:
            _ADJ_LAST[c] = a
        else:
            a = _ADJ_LAST.get(c)
        if a is not None and a > 0:
            _ADJ_REF.setdefault(c, a)
        d[c] = (o, cl, h, l, amt, vol, pct, a)
    _DAY_PX[td] = d
    return d


def _px(code, td, which="close"):
    """取 code 在 td 的 open/close/high/low；缺失则向前（更早交易日）回找已缓存的价。"""
    set_trade_date_ctx(td)   # 登记成交日 → calc_fee 自动用对当期印花税率（2023-08-28 起千0.5）
    day = _ensure_day(td)
    rec = day.get(code)
    if rec is not None:
        o, cl, h, l, _, _, _, a = rec
        if which == "open":
            return _scale(code, cl if int(td) >= _POST_2026_OPEN_AS_CLOSE else o, a)
        if which == "high":
            return _scale(code, h, a)
        if which == "low":
            return _scale(code, l, a)
        return _scale(code, cl, a)
    # 回找更早交易日（已在循环中按序缓存）
    ti = int(td)
    for step in range(1, 30):
        p = str(ti - step)
        pd2 = _DAY_PX.get(p)
        if pd2 is not None and code in pd2:
            o, cl, h, l, _, _, _, a = pd2[code]
            if which == "open":
                return _scale(code, cl if int(p) >= _POST_2026_OPEN_AS_CLOSE else o, a)
            if which == "high":
                return _scale(code, h, a)
            if which == "low":
                return _scale(code, l, a)
            return _scale(code, cl, a)
    return None


def _exec_price(code, td):
    set_trade_date_ctx(td)   # 登记成交日 → calc_fee 用对当期印花税率 + 平方根冲击取流动性窗口
    """成交价（受 EXEC_PRICE 开关控制）：
       - "open": 当日开盘价（含 2026-07-06 后开盘=收盘约定）
       - "vwap": 日级 VWAP 代理 = amount(千元)*10 / vol(手) → 元/股
                 缺失/停牌则回退更早交易日的 VWAP，最终回退开盘价。
    """
    if EXEC_PRICE != "vwap":
        return _px(code, td, "open")
    day = _ensure_day(td)
    rec = day.get(code)
    if rec is not None:
        amt, vol = rec[4], rec[5]
        if amt and vol and vol > 0:
            v = amt * 10.0 / vol
            if v > 0:
                # VWAP 由 amount/vol 派生，同样是 raw 空间的元/股 → 必须同道缩放，
                # 否则成交价（raw）与估值价（hfq）不在同一空间 → 除权日产生虚假盈亏。
                return _scale(code, v, rec[7])
    ti = int(td)
    for step in range(1, 30):
        p = str(ti - step)
        pd2 = _DAY_PX.get(p)
        if pd2 is not None and code in pd2:
            amt, vol = pd2[code][4], pd2[code][5]
            if amt and vol and vol > 0 and amt * 10.0 / vol > 0:
                return _scale(code, amt * 10.0 / vol, pd2[code][7])
    return _px(code, td, "open")


def _pct(code, td):
    """当日涨跌幅(pct_chg)，用于涨跌停判定；缺失回找更早交易日。"""
    day = _ensure_day(td)
    rec = day.get(code)
    if rec is not None and rec[6] is not None:
        return rec[6]
    ti = int(td)
    for step in range(1, 30):
        p = str(ti - step)
        pd2 = _DAY_PX.get(p)
        if pd2 is not None and code in pd2 and pd2[code][6] is not None:
            return pd2[code][6]
    return None


def _is_limit_up(code, td):
    if not LIMIT_ON:
        return False
    p = _pct(code, td)
    return p is not None and p >= LIMIT_UP_PCT


def _is_limit_down(code, td):
    if not LIMIT_ON:
        return False
    p = _pct(code, td)
    return p is not None and p <= LIMIT_DOWN_PCT


def get_atr(code, td, period=14):
    """EP 内独立 ATR 计算（口径与 run_monthly_rebalance.get_atr 一致）：

        ATR = SMA(TR, period)，TR = max(H-L, |H-prevC|, |L-prevC|)
        取 td 当日及之前 period+30 日 high/low/close；数据不足返回 None。
    仅在建仓时调用一次（用于设定初始止损价），持仓期间用滚动窗口更新，不重复查库。
    """
    conn = get_conn()
    rows = pd.read_sql_query(
        "SELECT trade_date, high, low, close FROM daily "
        "WHERE ts_code=? AND trade_date<=? ORDER BY trade_date DESC LIMIT ?",
        conn, params=(code, td, period + 30))
    conn.close()
    if len(rows) < period + 1:
        return None
    rows = rows.iloc[::-1].reset_index(drop=True)
    trs = []
    for i in range(1, len(rows)):
        h = float(rows.iloc[i]["high"]); l = float(rows.iloc[i]["low"])
        pc = float(rows.iloc[i - 1]["close"])
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    if len(trs) < period:
        return None
    return float(sum(trs[-period:]) / period)


# ════════════════════════════════════════════════════════════
#  纯行业中性 EP 选股
# ════════════════════════════════════════════════════════════
def select_ep_neutral(rebalance_date, top_n=TOP_N, prev_date=None,
                      verbose=True, stock_pool="all", obv_filter=None):
    """返回 DataFrame: [ts_code, ep, pe_ttm, industry]，全部 G5（最便宜五分位），按 ep 降序。

    逻辑：每月 T-1(prev_date) 估值 → 行业内 5 分组 → 取 G5（最便宜五分位）→ 等权持有。
    obv_filter 给定天数时（方案A）：对 G5 再做 OBV 吸筹过滤，只留过去 N 日资金净流入的便宜股。
    """
    if prev_date is None:
        prev_date = rebalance_date
    basic = _load_basic()
    conn = get_conn()

    pool_set = (_get_pool_constituents(stock_pool, prev_date)
                if stock_pool and stock_pool != "all" else None)
    if pool_set is not None and verbose:
        print(f"  [股票池] {stock_pool} 时点成分 {len(pool_set)} 只（asof {prev_date}）")

    # 1) prev_date 有交易的股票
    rows = pd.read_sql_query(
        "SELECT DISTINCT ts_code FROM daily WHERE trade_date = ?",
        conn, params=(prev_date,))
    trading = set(str(c) for c in rows["ts_code"].tolist())

    # 2) 资格过滤
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

    # 3) T-1 日 PE_TTM
    pe = pd.read_sql_query(
        "SELECT ts_code, pe_ttm FROM daily_basic WHERE trade_date = ? AND pe_ttm > 0",
        conn, params=(prev_date,))
    conn.close()
    if pe.empty:
        if verbose:
            print(f"  [选股 {rebalance_date}] {prev_date} 无 PE_TTM 数据")
        return pd.DataFrame()

    pe["ts_code"] = pe["ts_code"].astype(str)
    pe = pe[pe["ts_code"].isin(eligible)].copy()
    if pe.empty:
        return pd.DataFrame()

    pe["industry"] = pe["ts_code"].map(
        lambda c: (basic.get(c) or {}).get("industry"))
    pe = pe[pe["industry"].notna()].copy()
    if pe.empty:
        return pd.DataFrame()

    # 4) EP = 1/PE_TTM，全局缩尾
    pe["ep"] = 1.0 / pe["pe_ttm"].astype(float)
    lo, hi = pe["ep"].quantile([WINSOR_LO, WINSOR_HI])
    pe["epw"] = pe["ep"].clip(lo, hi)

    # 5) 行业内 5 分组（每行业≥5只才分组）
    pe["g"] = np.nan
    for ind, g in pe.groupby("industry"):
        if len(g) < QUINTILES:
            continue
        pe.loc[g.index, "g"] = pd.qcut(
            g["epw"].rank(method="first"), QUINTILES, labels=False) + 1
    pe = pe[pe["g"] == QUINTILES].copy()   # G5 = 最便宜五分位
    if pe.empty:
        if verbose:
            print(f"  [选股 {rebalance_date}] 无可用 G5 行业分组")
        return pd.DataFrame()

    # 方案A：OBV 吸筹过滤（仅当 obv_filter 给定天数时启用）
    # 只保留过去 N 日 OBV 净流量>0 的 G5 票（便宜且有人在买），剔除无人接盘的派发票。
    if obv_filter:
        g5_before = len(pe)
        keep = set(obv_accumulation_filter(pe["ts_code"].tolist(), prev_date, lookback=obv_filter))
        pe = pe[pe["ts_code"].isin(keep)].copy()
        if verbose and g5_before:
            print(f"  [OBV吸筹过滤] G5 {g5_before} → 吸筹 {len(pe)} 只（lookback={obv_filter}日）")
        if pe.empty:
            if verbose:
                print(f"  [选股 {rebalance_date}] OBV过滤后无候选")
            return pd.DataFrame()

    # 6) 集中度上限（默认 None = 全持有 G5）
    if top_n is not None:
        pe = pe.sort_values("ep", ascending=False).head(top_n).reset_index(drop=True)
    else:
        pe = pe.sort_values("ep", ascending=False).reset_index(drop=True)

    if verbose:
        cap_note = f" → 取 {top_n} 只" if top_n else "（全持有）"
        print(f"  [选股 {rebalance_date}] 候选池 {len(eligible)} 只 → "
              f"G5 {len(pe)} 只{cap_note}")
    return pe[["ts_code", "ep", "pe_ttm", "industry"]]


# ════════════════════════════════════════════════════════════
#  月度回测引擎
# ════════════════════════════════════════════════════════════
def run_backtest(start_date="20100101", end_date="20260715",
                 top_n=TOP_N, verbose=True, stock_pool="all",
                 var_stop=False, atr_mult=2.0, atr_cooling=5,
                 exec_price="open", limit_on=True, slippage=0.001,
                 interrupt_start=None, interrupt_months=0, interrupt_pct=0.0):
    global EXEC_PRICE, LIMIT_ON
    EXEC_PRICE = exec_price
    LIMIT_ON = limit_on
    import run_monthly_rebalance as _rm
    _rm.SLIPPAGE_RATE = slippage   # 让 calc_fee 使用当前滑点
    reset_fee_ctx()                # 清空印花税率上下文，避免多次运行串味
    print("=" * 72)
    print("  纯行业中性 EP（Earnings Yield = 1/PE_TTM）月度选股策略回测")
    print("=" * 72)
    print(f"  区间：{start_date} ~ {end_date}")
    cap_note = f"{top_n}只等权" if top_n else "全部G5等权"
    print(f"  持仓：{cap_note} | 调仓：每月第5交易日（行业中性 G5）")
    print(f"  剔除：ST / .BJ / 金融 / 公用事业 / 上市<60天")
    print(f"  因子：EP=1/PE_TTM，全局1/99缩尾，行业内5分组取G5")
    if var_stop:
        print(f"  VAR动态止损（动量月度同款）：ATR追踪{atr_mult}倍 | 冷静期{atr_cooling}日 "
              f"| 跌破[最高收-{atr_mult}×ATR]次日开盘卖")
    else:
        print(f"  止损：无（月度调仓，掉出G5才卖）")
    ex = "开盘价" if EXEC_PRICE == "open" else "日级VWAP代理(amount×10/vol)"
    print(f"  成交价假设：{ex} | 涨跌停约束：{'开(涨停买不进/跌停卖不出)' if LIMIT_ON else '关'} "
          f"| T+1：月度调仓天然满足")
    print(f"  佣金万{COMMISSION_RATE*1e4:.1f}(最低{COMMISSION_MIN}) "
          f"印花税千1→千0.5(2023-08-28起) 滑点{slippage*100:.2f}%")
    print(f"  初始资金：{_CAPITAL:,.0f}\n")

    trade_dates = get_trade_dates(start_date, end_date)
    rebal_set = set(get_monthly_5th_trading_days(trade_dates))
    rebal_set = {d for d in rebal_set if start_date <= d <= end_date}
    print(f"  交易日 {len(trade_dates)} 天，月度调仓 {len(rebal_set)} 次\n")

    positions = {}   # code -> {shares, buy_price, last_price}
    cash = float(_CAPITAL)
    daily_vals = []
    trades = []
    name_cache = {}
    limit_up_skip = 0
    limit_down_skip = 0
    reset_price_cache()   # 含 _ADJ_REF 重置：换回测区间时归一化基准必须重算

    def _name(code):
        if code not in name_cache:
            name_cache[code] = _get_name(code)
        return name_cache[code]

    var_stop_count = 0
    for i, td in enumerate(trade_dates):
        # ── 1) 执行昨日标记的 VAR 动态止损卖出（开盘）──
        if var_stop:
            for code in list(positions.keys()):
                pos = positions[code]
                if pos.get("stop_pending"):
                    op = _px(code, td, "open")
                    if op is None:
                        continue
                    fee = calc_fee('sell', op, pos["shares"], ts_code=code)
                    cash += pos["shares"] * op - fee
                    trades.append({"date": td, "action": "SELL",
                                   "code": code, "name": _name(code),
                                   "price": op, "shares": pos["shares"],
                                   "reason": "var_atr_stop"})
                    var_stop_count += 1
                    if verbose:
                        print(f"  🔴 VAR止损 {code}({_name(code)})："
                              f"{pos['shares']}股 @ {op:.2f}（最高收{pos.get('highest_close',0):.2f} 跌破止损线）")
                    del positions[code]

        # ── 调仓日（首日跳过，用 T-1 数据选股、T 开盘执行）──
        if td in rebal_set and i > 0:
            prev_td = trade_dates[i - 1]
            sel = select_ep_neutral(td, top_n=top_n, prev_date=prev_td,
                                    verbose=verbose, stock_pool=stock_pool,
                                    obv_filter=OBV_FILTER)
            new_codes = sel["ts_code"].tolist() if not sel.empty else []
            new_set = set(new_codes)

            if not new_codes:
                if verbose:
                    print(f"\n调仓日 {td}：选股为空，保持现有 {len(positions)} 仓")
            else:
                cur = set(positions.keys())
                if cur != new_set:
                    for code in list(positions.keys()):
                        if code not in new_set:
                            op = _exec_price(code, td)
                            if op is None:
                                continue
                            if _is_limit_down(code, td):
                                limit_down_skip += 1
                                if verbose:
                                    print(f"  ⏸ 跌停未卖 {code}({_name(code)})："
                                          f"跌停约束，保留至下月再卖")
                                continue
                            pos = positions[code]
                            fee = calc_fee('sell', op, pos["shares"], ts_code=code)
                            cash += pos["shares"] * op - fee
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
                        cash_per = cash / len(to_buy)
                        for code in to_buy:
                            op = _exec_price(code, td)
                            if op is None:
                                continue
                            if _is_limit_up(code, td):
                                limit_up_skip += 1
                                if verbose:
                                    print(f"  ⏸ 涨停未买 {code}({_name(code)})："
                                          f"涨停约束，跳过买入")
                                continue
                            max_shares = int(cash_per / op / 100) * 100
                            if max_shares < 100:
                                continue
                            cost = max_shares * op
                            fee = calc_fee('buy', op, max_shares, ts_code=code)
                            if cost + fee <= cash:
                                cash -= cost + fee
                                pos = {"shares": max_shares,
                                       "buy_price": op, "last_price": op}
                                if var_stop:
                                    atr0 = get_atr(code, trade_dates[i - 1], 14)
                                    if atr0 and atr0 > 0:
                                        pos["highest_close"] = op
                                        pos["atr_stop_price"] = op - atr_mult * atr0
                                        pos["entry_idx"] = i
                                        pos["last_close"] = op
                                        pos["tr_window"] = []
                                    else:
                                        pos["atr_stop_price"] = None  # ATR不足，该只不追踪
                                positions[code] = pos
                                trades.append({"date": td, "action": "BUY",
                                               "code": code, "name": _name(code),
                                               "price": op, "shares": max_shares,
                                               "reason": "ep_neutral"})
                                if verbose:
                                    print(f"  ✅ 买入 {code}({_name(code)})："
                                          f"{max_shares}股 @ {op:.2f}")
                else:
                    if verbose:
                        print(f"\n调仓日 {td}：持仓不变")

        # ── 每日市值记录 + VAR 追踪更新（批量缓存取价）──
        total = cash
        for code, pos in list(positions.items()):
            px = _px(code, td, "close")
            if px is None:
                px = pos.get("last_price") or 0
            else:
                pos["last_price"] = px
            # VAR 追踪止损：更新最高收 → 滚动更新 ATR 止损线 → 过冷静期跌破则标记
            if var_stop and pos.get("atr_stop_price") is not None:
                if pos.get("highest_close") is None:
                    pos["highest_close"] = px
                else:
                    pos["highest_close"] = max(pos["highest_close"], px)
                h = _px(code, td, "high"); l = _px(code, td, "low")
                prev_c = pos.get("last_close", px)
                if h is not None and l is not None:
                    tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
                    tw = pos.setdefault("tr_window", [])
                    tw.append(tr)
                    if len(tw) > 14:
                        tw.pop(0)
                    if len(tw) >= 14:
                        atr = sum(tw) / len(tw)
                        new_stop = pos["highest_close"] - atr_mult * atr
                        if new_stop > pos["atr_stop_price"]:
                            pos["atr_stop_price"] = new_stop
                if (i - pos.get("entry_idx", -999)) > atr_cooling and px < pos["atr_stop_price"]:
                    pos["stop_pending"] = True
            pos["last_close"] = px
            total += pos["shares"] * px
        daily_vals.append({"date": td, "value": total})

    # ── 末日平仓 ──
    if trade_dates:
        last = trade_dates[-1]
        for code in list(positions.keys()):
            px = _px(code, last, "close")
            if px is not None:
                pos = positions[code]
                fee = calc_fee('sell', px, pos["shares"], ts_code=code)
                cash += pos["shares"] * px - fee
                trades.append({"date": last, "action": "SELL", "code": code,
                               "name": _name(code), "price": px,
                               "shares": pos["shares"], "reason": "backtest_end"})
                del positions[code]
        if daily_vals:
            daily_vals[-1]["value"] = cash

    return _report(daily_vals, trades, trade_dates, start_date, end_date,
                   top_n=top_n, var_stop_count=var_stop_count,
                   exec_price=exec_price, limit_on=limit_on, slippage=slippage,
                   limit_up_skip=limit_up_skip, limit_down_skip=limit_down_skip,
                   interrupt_start=interrupt_start, interrupt_months=interrupt_months,
                   interrupt_pct=interrupt_pct)


def _get_name(ts_code):
    conn = get_conn()
    row = pd.read_sql_query(
        "SELECT name FROM stock_basic WHERE ts_code = ? LIMIT 1",
        conn, params=(ts_code,))
    conn.close()
    return row.iloc[0]["name"] if len(row) > 0 else ts_code


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
            var_stop_count=0, exec_price="open", limit_on=True, slippage=0.001,
            limit_up_skip=0, limit_down_skip=0,
            interrupt_start=None, interrupt_months=0, interrupt_pct=0.0):
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

    win_rate, win_cnt, tot_cnt = calc_win_rate(trades)

    bench = {}
    bench_meta = {}      # idx -> meta（逐个保留：不同基准的全收益覆盖度不同，
    conn = get_conn()    # 若只留最后一个，会漏掉"某基准回退到价格指数"的告警）
    for idx in BENCHMARKS:
        # 经统一真相源 bench_index：基准口径自动跟随 NAV 口径
        #   raw NAV → 价格指数；hfq NAV → 官方全收益（缺失则回退自建/价格）
        # 两端必须同含或同不含分红，否则超额系统性失真（中证800 约 2.2%/年、沪深300 约 3.0%/年）。
        r, meta = bi.benchmark_return_between(idx, trade_dates[0], trade_dates[-1],
                                              conn=conn, nav_price_mode=PRICE_MODE)
        if r is not None:
            bench[idx] = r
            bench_meta[idx] = meta
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
    if var_stop_count > 0:
        print(f"  VAR动态止损触发：{var_stop_count} 笔")
    ex = "open" if exec_price == "open" else "vwap"
    # 注：_rm 原先是 run_backtest() 内的局部 import，_report() 访问不到 → NameError。
    #     此处补局部 import（不作为模块级依赖，避免改变导入顺序/循环依赖风险）。
    import run_monthly_rebalance as _rm2
    print(f"  成交价：{ex} | 涨跌停约束：{'开' if limit_on else '关'} | 滑点：{slippage*100:.2f}%"
          f" | 冲击模型：{'平方根(流动性感知)' if _rm2.USE_SQRT_IMPACT else 'flat'}"
          f"(MFS_SQRT_IMPACT={os.environ.get('MFS_SQRT_IMPACT','0')})")
    if limit_on:
        print(f"  涨跌停跳过：涨停未买 {limit_up_skip} 次 | 跌停未卖 {limit_down_skip} 次")
    print(f"  NAV 口径：{PRICE_MODE}"
          f"{'（含分红再投）' if PRICE_MODE == 'hfq' else '（不含分红）'}")
    for idx, r in bench.items():
        _m = bench_meta.get(idx)
        _lbl = bi.benchmark_meta_label(_m) if _m else ""
        print(f"  {INDEX_DISPLAY_NAME.get(idx, idx)}：{r:+.2f}%  超额：{total_return - r:+.2f}%"
              f"   {_lbl}")
        # 逐个基准检查：中证全指(000985.SH) 无官方全收益，会回退价格指数 → hfq 下超额被高估
        _w = bi.check_consistency(PRICE_MODE, _m)
        if _w:
            print(f"    ⚠️ {_w}")

    # ── 现实折扣三件套（扣通胀 / 定投拖累 / 中断模拟）──
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

    os.makedirs("data/results/ep_neutral", exist_ok=True)
    obv_tag = f"_obv{OBV_FILTER}" if OBV_FILTER else ""
    ex_tag = f"_{ex}{'' if limit_on else '_nolim'}"
    pm_tag = "_hfq" if PRICE_MODE == "hfq" else ""   # 口径后缀，避免 hfq 静默覆盖 raw
    tag = f"n{top_n}{obv_tag}{ex_tag}{pm_tag}" if top_n else f"nG5{obv_tag}{ex_tag}{pm_tag}"
    csv_path = (f"data/results/ep_neutral/"
                f"backtest_{tag}_c{int(_CAPITAL)}_{start_date}_{end_date}.csv")
    pd.DataFrame(daily_vals).to_csv(csv_path, index=False)
    pd.DataFrame(trades).to_csv(csv_path.replace("backtest_", "trades_"), index=False)
    print(f"\n  结果已保存：{csv_path}")

    return {
        "final_value": final_value, "total_return": total_return,
        "annual_return": annual_return, "max_drawdown": max_dd,
        "sharpe": sharpe, "win_rate": win_rate, "trades": len(trades),
        "bench": bench, "yearly": yr, "reality_discounts": disc,
    }


# ════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="纯行业中性 EP 月度选股策略")
    p.add_argument("start_date", nargs="?", default="20100101")
    p.add_argument("end_date", nargs="?", default="20260715")
    p.add_argument("--top-n", type=int, default=None,
                   help="集中度上限（默认None=持有全部G5等权）")
    p.add_argument("--capital", type=int, default=_CAPITAL,
                   help="初始资金（默认500万，G5约数百只需足额部署）")
    p.add_argument("--quiet", action="store_true", help="减少输出")
    p.add_argument("--stock-pool", default="all",
                   help="股票池 hs300/zz500/zz800/zz1000/zz2000/all（默认全A股）")
    p.add_argument("--exec-price", default="open", choices=["open", "vwap"],
                   help="成交价假设: open(开盘价,默认) / vwap(日级VWAP代理=amount×10/vol)")
    p.add_argument("--slippage", type=float, default=0.001,
                   help="滑点率(默认0.001=0.1%%, 买卖均含, 经 calc_fee 计入成本)")
    p.add_argument("--no-limit", action="store_true",
                   help="关闭涨跌停约束(默认开: 涨停买不进/跌停卖不出)")
    p.add_argument("--select-month", type=str, default=None,
                   help="仅输出某月(YYYYMM)选股结果，不回测")
    p.add_argument("--obv-filter", type=int, default=0,
                   help="方案A：OBV吸筹过滤窗口(交易日, 默认0=关闭); >0 时只留过去N日资金净流入的G5便宜股")
    p.add_argument("--var", action="store_true",
                   help="启用 VAR 动态止损（动量月度同款 ATR 追踪：跌破最高收-倍数×ATR 次日开盘卖）")
    p.add_argument("--atr-mult", type=float, default=2.0,
                   help="ATR 止损倍数（默认2.0，与动量月度一致）")
    p.add_argument("--atr-cooling", type=int, default=5,
                   help="买入后冷静期交易日数（默认5，期内不触发止损）")
    p.add_argument("--interrupt-start", type=str, default=None,
                   help="中断模拟起点(YYYYMM)，配合--interrupt-pct使用")
    p.add_argument("--interrupt-months", type=int, default=0,
                   help="中断模拟：撤出资金空仓月数（默认0=不模拟）")
    p.add_argument("--interrupt-pct", type=float, default=0.0,
                   help="中断模拟：撤出资金比例(0~1，如0.5=撤一半)，默认0=不模拟")
    p.add_argument("--price-mode", choices=["raw", "hfq"], default="hfq",
                   help="NAV 计价口径: hfq=后复权(默认,总回报,含分红再投+免疫送转,自动配全收益基准) / "
                        "raw=不复权(旧口径,漏分红且不处理送转,仅供复现历史结论)")
    args = p.parse_args()

    # 注：此赋值位于 `if __name__ == "__main__"` 的模块级块内，赋值即绑定模块全局，
    #     加 `global` 反而会报 "assigned to before global declaration"。
    PRICE_MODE = args.price_mode

    _CAPITAL = args.capital
    top_n = args.top_n
    OBV_FILTER = args.obv_filter if args.obv_filter and args.obv_filter > 0 else None

    if args.select_month:
        ym = args.select_month
        cal = (get_trade_dates(ym + "01", ym + "31")
               if len(ym) == 6 else get_trade_dates(ym[:4] + "0101", ym[:4] + "1231"))
        m5 = sorted([d for d in cal if d[:6] == ym]) if len(ym) == 6 else cal
        if not m5:
            print("该月无交易日")
        else:
            td = m5[4] if len(m5) >= 5 else m5[-1]
            idx = cal.index(td)
            prev = cal[idx - 1] if idx > 0 else td
            sel = select_ep_neutral(td, top_n=top_n, prev_date=prev,
                                    verbose=True, stock_pool=args.stock_pool,
                                    obv_filter=OBV_FILTER)
            print(f"\n=== {ym} 行业中性 EP 选股（{len(sel)} 只）===")
            for _, r in sel.iterrows():
                print(f"  {r['ts_code']}({_get_name(r['ts_code'])}) "
                      f"EP={r['ep']:.4f} PE_TTM={float(r['pe_ttm']):.1f} "
                      f"行业={r['industry']}")
    else:
        run_backtest(args.start_date, args.end_date,
                     top_n=top_n, verbose=not args.quiet, stock_pool=args.stock_pool,
                     var_stop=args.var, atr_mult=args.atr_mult,
                     atr_cooling=args.atr_cooling,
                     exec_price=args.exec_price, limit_on=not args.no_limit,
                     slippage=args.slippage,
                     interrupt_start=args.interrupt_start,
                     interrupt_months=args.interrupt_months,
                     interrupt_pct=args.interrupt_pct)
