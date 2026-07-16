"""
周度调仓「高股息 + 高波动」策略 — 忠实复现 BV13soeBWETV
=====================================================
复现 B站视频描述的四因子选股 + 周度调仓 + 涨停止盈 + 20日黑名单规则。

四因子（纯交集）：
  F1 高股息率 : daily_basic.dv_ttm          ≥ 全市场前25%分位
  F2 高换手率 : daily_basic.turnover_rate   ≥ 全市场前85%分位
  F3 低负债   : MLEV=(ME+总负债)/ME          ≤ 全市场最低55%分位
  F4 小市值   : daily_basic.circ_mv         ≤ 全市场最低50%分位

交易规则：
  买入：每周首个交易日，T-1数据选股、T开盘执行；等权分配，不满仓不加杠杆。
        候选需 ①非停牌 ②非一字跌停 ③不在20日黑名单。
  止盈：昨日涨停 + 今日涨停板打开 → T+1开盘卖出。
  调仓卖出：不再过四因子 → 一次性清仓。
  20日黑名单：所有被卖出股票未来20交易日禁买。

防偏措施（与上一版的关键区别）：
  ① 财务数据用 ann_date（公告日）≤ 选股日，杜绝前视偏差
  ② 股票池含退市股（从 daily 表取，不依赖 stock_basic），杜绝幸存者偏差
  ③ T-1 数据选股，T 开盘执行，首日不调仓，杜绝日内前视
  ④ 20日黑名单对所有卖出生效（忠实原描述，不再按卖出原因区分）
  ⑤ 末日平仓扣费后修正净值，杜绝收益高估
"""
import sqlite3, os, sys, bisect
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_monthly_rebalance import (
    get_conn, calc_fee, calc_win_rate, get_trade_dates,
    prefilter_by_liquidity,
    INIT_CAPITAL, COMMISSION_RATE, COMMISSION_MIN, STAMP_DUTY_RATE,
    SLIPPAGE_RATE, LIQUIDITY_MIN_AVG_AMOUNT, INDEX_DISPLAY_NAME,
)

# ════════════════════════════════════════════════════════════
#  常量
# ════════════════════════════════════════════════════════════
TOP_N      = 10
DIV_PCT    = 25       # 高股息：前25%
TURN_PCT   = 85       # 高换手：前85%
DEBT_PCT   = 55       # 低负债：最低55%
SIZE_PCT   = 50       # 小市值：最低50%
BLACKLIST_DAYS = 20   # 20日黑名单
IPO_MIN_DAYS   = 60   # 上市<60天剔除
BENCHMARKS = ["000906.SH", "000985.SH"]

# ════════════════════════════════════════════════════════════
#  缓存
# ════════════════════════════════════════════════════════════
_STOCK_BASIC = None   # {ts_code: {name, list_date, excluded, delisted}}
_ADJ_CACHE   = {}     # ts_code -> {dates, fac, ref, empty}
_LIAB_CACHE  = {}     # trade_date -> {ts_code: total_liab}
_DEBT_CACHE  = {}     # trade_date -> {ts_code: debt_to_assets}
_TRADE_CAL   = None
_BS_EMPTY    = None

# ════════════════════════════════════════════════════════════
#  1. 数据库工具
# ════════════════════════════════════════════════════════════
def _load_trade_cal():
    global _TRADE_CAL
    if _TRADE_CAL is None:
        conn = get_conn()
        rows = pd.read_sql_query(
            "SELECT DISTINCT trade_date FROM daily ORDER BY trade_date", conn)
        conn.close()
        _TRADE_CAL = [str(d) for d in rows["trade_date"].tolist()]
    return _TRADE_CAL


def _shift_date(date_str, n):
    """返回 date_str 之前第 n 个交易日。"""
    cal = _load_trade_cal()
    try:
        idx = cal.index(date_str)
    except ValueError:
        idx = bisect.bisect_left(cal, date_str) - 1
        if idx < 0:
            return date_str
    return cal[max(0, idx - n)]


def _load_stock_basic():
    """加载股票基础信息，**含退市股**（从 daily 表补充）。
    退市股不在 stock_basic 中，若不补充会导致幸存者偏差。"""
    global _STOCK_BASIC
    if _STOCK_BASIC is not None:
        return _STOCK_BASIC
    conn = get_conn()
    df = pd.read_sql_query("SELECT ts_code, name, list_date FROM stock_basic", conn)
    m = {}
    survivors = set()
    for _, r in df.iterrows():
        code = str(r["ts_code"])
        name = str(r["name"]) if pd.notna(r["name"]) else ""
        ld = str(r["list_date"]) if pd.notna(r["list_date"]) else ""
        excluded = (code.startswith("688") or code.endswith(".BJ")
                    or "ST" in name.upper() or name.startswith("*"))
        m[code] = {"name": name, "list_date": ld,
                   "excluded": excluded, "delisted": False}
        survivors.add(code)
    # 退市股补充
    dl = pd.read_sql_query(
        "SELECT ts_code, MIN(trade_date) AS first_dt FROM daily GROUP BY ts_code", conn)
    conn.close()
    n_dl = 0
    for _, r in dl.iterrows():
        code = str(r["ts_code"])
        if code in survivors:
            continue
        first_dt = str(r["first_dt"]) if pd.notna(r["first_dt"]) else ""
        excluded = code.startswith("688") or code.endswith(".BJ")
        m[code] = {"name": "", "list_date": first_dt,
                   "excluded": excluded, "delisted": True}
        n_dl += 1
    print(f"  [股票池] 存续 {len(survivors)} + 退市 {n_dl} = {len(m)} 只")
    _STOCK_BASIC = m
    return m


def _load_adj(code):
    if code in _ADJ_CACHE:
        return
    conn = get_conn()
    rows = pd.read_sql_query(
        "SELECT trade_date, adj_factor FROM adj_factor "
        "WHERE ts_code = ? ORDER BY trade_date", conn, params=(code,))
    conn.close()
    if len(rows) == 0:
        _ADJ_CACHE[code] = {"dates": [], "fac": [], "ref": 1.0, "empty": True}
        return
    ds = [str(d) for d in rows["trade_date"].tolist()]
    fs = [float(f) for f in rows["adj_factor"].tolist()]
    _ADJ_CACHE[code] = {"dates": ds, "fac": fs, "ref": fs[-1], "empty": False}


def _adj_factor(code, td):
    c = _ADJ_CACHE.get(code)
    if c is None:
        _load_adj(code)
        c = _ADJ_CACHE[code]
    if c.get("empty"):
        return None
    i = bisect.bisect_right(c["dates"], td) - 1
    if i < 0:
        return None
    return c["fac"][i]


def _raw_close(code, td):
    """不复权收盘价，精确匹配日期，无 fallback。"""
    conn = get_conn()
    row = pd.read_sql_query(
        "SELECT close FROM daily WHERE ts_code=? AND trade_date=?", conn,
        params=(code, td))
    conn.close()
    return float(row.iloc[0]["close"]) if len(row) > 0 else None


def _raw_open(code, td):
    """不复权开盘价，精确匹配日期，无 fallback。"""
    conn = get_conn()
    row = pd.read_sql_query(
        "SELECT open FROM daily WHERE ts_code=? AND trade_date=?", conn,
        params=(code, td))
    conn.close()
    return float(row.iloc[0]["open"]) if len(row) > 0 else None


def qfq_close(code, td):
    """前复权收盘价 = raw × ref / factor(td)。"""
    raw = _raw_close(code, td)
    if raw is None:
        return None
    f = _adj_factor(code, td)
    if not f:
        return raw
    return raw * _ADJ_CACHE[code]["ref"] / f


def qfq_open(code, td):
    """前复权开盘价。"""
    raw = _raw_open(code, td)
    if raw is None:
        return None
    f = _adj_factor(code, td)
    if not f:
        return raw
    return raw * _ADJ_CACHE[code]["ref"] / f


def get_daily_row(code, td):
    """返回某日 daily 行（open/high/low/close/pre_close/pct_chg/vol）。"""
    conn = get_conn()
    row = pd.read_sql_query(
        "SELECT open, high, low, close, pre_close, pct_chg, vol "
        "FROM daily WHERE ts_code=? AND trade_date=?", conn,
        params=(code, td))
    conn.close()
    if len(row) == 0:
        return None
    return {k: (float(row.iloc[0][k]) if pd.notna(row.iloc[0][k]) else None)
            for k in ["open", "high", "low", "close", "pre_close", "pct_chg", "vol"]}


def _limit_thr(code):
    """涨停幅度：主板 10%，创业板/科创板 20%，北交 30%。"""
    if code.startswith("688") or (code.startswith("300") and code.endswith(".SZ")):
        return 0.195
    if code.endswith(".BJ"):
        return 0.295
    return 0.095


def _bs_empty():
    global _BS_EMPTY
    if _BS_EMPTY is None:
        conn = get_conn()
        _BS_EMPTY = conn.execute("SELECT COUNT(*) FROM balance_sheet").fetchone()[0] == 0
        conn.close()
    return _BS_EMPTY


# ════════════════════════════════════════════════════════════
#  2. 财务因子（公告日口径，无前视）
# ════════════════════════════════════════════════════════════
def _liab_map(visible_date):
    """截至 visible_date 已公告（ann_date ≤ visible_date）的最新 total_liab。
    缓存按 visible_date（每个调仓日独立缓存）。"""
    if visible_date in _LIAB_CACHE:
        return _LIAB_CACHE[visible_date]
    conn = get_conn()
    df = pd.read_sql_query(
        "WITH ranked AS ("
        "  SELECT ts_code, total_liab,"
        "         ROW_NUMBER() OVER (PARTITION BY ts_code "
        "           ORDER BY end_date DESC, ann_date DESC) rn"
        "  FROM balance_sheet"
        "  WHERE ann_date <= ? AND ann_date IS NOT NULL AND ann_date != '')"
        "SELECT ts_code, total_liab FROM ranked WHERE rn = 1",
        conn, params=(visible_date,))
    conn.close()
    m = {str(r["ts_code"]): (float(r["total_liab"]) if pd.notna(r["total_liab"]) else None)
         for _, r in df.iterrows()}
    _LIAB_CACHE[visible_date] = m
    return m


def _debt_map(visible_date):
    """截至 visible_date 已公告的最新 debt_to_assets（资产负债率），仅作回退。"""
    if visible_date in _DEBT_CACHE:
        return _DEBT_CACHE[visible_date]
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT ts_code, end_date, ann_date, debt_to_assets FROM fina_indicator "
        "WHERE ann_date <= ? AND ann_date IS NOT NULL AND ann_date != ''",
        conn, params=(visible_date,))
    conn.close()
    if len(df) == 0:
        _DEBT_CACHE[visible_date] = {}
        return _DEBT_CACHE[visible_date]
    df = df.sort_values(["end_date", "ann_date"])
    latest = df.groupby("ts_code").tail(1)
    m = {str(r["ts_code"]): (float(r["debt_to_assets"]) if pd.notna(r["debt_to_assets"]) else None)
         for _, r in latest.iterrows()}
    _DEBT_CACHE[visible_date] = m
    return m


def _calc_mlev(ts_code, total_mv, liab_map, debt_map):
    """MLEV = (ME + 总负债) / ME。ME = total_mv(万元) × 10000 = 元。"""
    tl = liab_map.get(ts_code)
    if tl is not None and pd.notna(tl) and pd.notna(total_mv) and total_mv > 0:
        me = float(total_mv) * 10000.0
        return (me + float(tl)) / me
    # 回退：资产负债率
    v = debt_map.get(ts_code)
    if v is not None:
        return float(v)
    return np.nan


# ════════════════════════════════════════════════════════════
#  3. 四因子选股
# ════════════════════════════════════════════════════════════
def select_stocks(rebalance_date, top_n=TOP_N,
                  div_pct=DIV_PCT, turn_pct=TURN_PCT,
                  debt_pct=DEBT_PCT, size_pct=SIZE_PCT,
                  factor_lag=0, verbose=False):
    """四因子纯交集选股。返回 {ts_code: {name, score, ...}} 或空 dict。
    factor_lag > 0 时因子值取数日回挪 N 个交易日（前视隔离测试）。"""
    basic = _load_stock_basic()
    conn = get_conn()
    fdate = _shift_date(rebalance_date, factor_lag)

    # 1) 当日交易股票（时点存在性）
    rows = pd.read_sql_query(
        "SELECT DISTINCT ts_code FROM daily WHERE trade_date=?", conn,
        params=(rebalance_date,))
    trading = set(rows["ts_code"].tolist())

    # 2) 剔除 688/.BJ/ST/上市<60天
    eligible = set()
    for c in trading:
        info = basic.get(c)
        if info is None or info["excluded"]:
            continue
        ld = info["list_date"]
        if ld and rebalance_date < ld:
            continue
        if ld:
            try:
                d_ld = datetime.strptime(ld, "%Y%m%d")
                d_rb = datetime.strptime(rebalance_date, "%Y%m%d")
                if (d_rb - d_ld).days < IPO_MIN_DAYS:
                    continue
            except Exception:
                pass
        eligible.add(c)
    if not eligible:
        conn.close()
        return {}

    # 3) 流动性过滤
    kept = prefilter_by_liquidity(conn, eligible, fdate)
    if not kept:
        conn.close()
        return {}

    # 4) daily_basic 因子（股息/换手/市值）
    codes = list(kept)
    ph = ",".join("?" * len(codes))
    df = pd.read_sql_query(
        f"SELECT ts_code, dv_ttm, turnover_rate, circ_mv, total_mv "
        f"FROM daily_basic WHERE ts_code IN ({ph}) AND trade_date=?",
        conn, params=codes + [fdate])
    conn.close()
    if len(df) == 0:
        return {}
    df = df.dropna(subset=["turnover_rate", "circ_mv"])
    if len(df) == 0:
        return {}

    # 5) 杠杆因子（公告日口径）
    if _bs_empty():
        dm = _debt_map(fdate)
        df["mlev"] = df["ts_code"].map(lambda c: dm.get(c))
    else:
        lm = _liab_map(fdate)
        dm = _debt_map(fdate)
        df["mlev"] = df.apply(
            lambda r: _calc_mlev(r["ts_code"], r["total_mv"], lm, dm), axis=1)
    df_debt = df.dropna(subset=["mlev"]).copy()
    if len(df_debt) == 0:
        return {}

    # 6) 分位阈值
    p_div  = np.percentile(df["dv_ttm"].dropna(), 100 - div_pct)
    p_turn = np.percentile(df["turnover_rate"], 100 - turn_pct)
    p_debt = np.percentile(df_debt["mlev"], debt_pct)
    p_size = np.percentile(df["circ_mv"], size_pct)

    # 7) 四因子布尔
    df_debt["f_div"]  = (df_debt["dv_ttm"] > 0) & (df_debt["dv_ttm"] >= p_div)
    df_debt["f_turn"] = df_debt["turnover_rate"] >= p_turn
    df_debt["f_debt"] = df_debt["mlev"] <= p_debt
    df_debt["f_size"] = df_debt["circ_mv"] <= p_size
    df_debt["pass"] = df_debt["f_div"] & df_debt["f_turn"] & df_debt["f_debt"] & df_debt["f_size"]

    inter = df_debt[df_debt["pass"]].copy()
    if len(inter) == 0:
        return {}

    # 8) 综合打分（四因子等权百分位）
    df_debt["score"] = (
        df_debt["dv_ttm"].rank(pct=True, ascending=True)
        + df_debt["turnover_rate"].rank(pct=True, ascending=True)
        + (1 - df_debt["mlev"].rank(pct=True, ascending=True))
        + (1 - df_debt["circ_mv"].rank(pct=True, ascending=True))
    )
    inter = df_debt[df_debt["pass"]].copy()
    if len(inter) >= top_n:
        inter = inter.sort_values("score", ascending=False).head(top_n)

    result = {}
    for _, r in inter.iterrows():
        code = str(r["ts_code"])
        result[code] = {
            "name": basic.get(code, {}).get("name", code),
            "dv_ttm": r["dv_ttm"], "turnover_rate": r["turnover_rate"],
            "mlev": r["mlev"], "circ_mv": r["circ_mv"], "score": r["score"],
        }
    if verbose:
        print(f"  [选股 {rebalance_date}] 候选池={len(df_debt)} 交集={len(inter)} "
              f"阈值 dv>{p_div:.2f}% turn>{p_turn:.2f} mlev<{p_debt:.3f} size<{p_size:.0f}万")
    return result


# ════════════════════════════════════════════════════════════
#  4. 回测引擎
# ════════════════════════════════════════════════════════════
def _weekly_first_days(trade_dates):
    """每周首个交易日（ISO 周分组）。"""
    dmap = {}
    for td in trade_dates:
        iso = datetime.strptime(td, "%Y%m%d").isocalendar()
        key = (iso[0], iso[1])
        if key not in dmap:
            dmap[key] = td
    return [dmap[k] for k in sorted(dmap.keys())]


def run_backtest(start_date="20210104", end_date="20260710",
                 top_n=TOP_N, div_pct=DIV_PCT, turn_pct=TURN_PCT,
                 debt_pct=DEBT_PCT, size_pct=SIZE_PCT,
                 factor_lag=0, zero_fee=False, capital=None, verbose=True):
    """主回测循环。"""
    if capital is not None:
        global INIT_CAPITAL
        INIT_CAPITAL = float(capital)
    print("=" * 72)
    print(f"  周度「高股息+高波动」策略回测")
    print("=" * 72)
    print(f"  区间：{start_date} ~ {end_date}")
    print(f"  持仓：{top_n}只等权 | 股息前{div_pct}% 换手前{turn_pct}% "
          f"负债最低{debt_pct}% 市值最低{size_pct}%")
    print(f"  黑名单：{BLACKLIST_DAYS}日 | 财务口径：ann_date（公告日）")
    print(f"  含退市股 | 佣金万{COMMISSION_RATE*1e4:.1f}(最低{COMMISSION_MIN}) "
          f"印花税千{STAMP_DUTY_RATE*1e3:.0f} 滑点{SLIPPAGE_RATE*100:.2f}%"
          + (" | 零成本模式" if zero_fee else "") + "\n")

    trade_dates = get_trade_dates(start_date, end_date)
    weekly = set(_weekly_first_days(trade_dates))
    print(f"  交易日 {len(trade_dates)} 天，周度调仓 {len(weekly)} 次\n")

    positions = {}   # code -> {shares, buy_price}
    cash = float(INIT_CAPITAL)
    daily_vals = []
    trades = []
    blacklist = {}   # code -> 解禁索引
    pending_sell = set()  # 待止盈卖出
    name_cache = {}

    def _name(code):
        if code not in name_cache:
            conn = get_conn()
            row = pd.read_sql_query(
                "SELECT name FROM stock_basic WHERE ts_code=? LIMIT 1", conn,
                params=(code,))
            conn.close()
            name_cache[code] = row.iloc[0]["name"] if len(row) > 0 else code
        return name_cache[code]

    def _do_sell(code, td, reason, price=None):
        nonlocal cash
        px = price if price is not None else qfq_open(code, td)
        if px is None:
            return False
        shares = positions[code]["shares"]
        gross = shares * px
        fee = 0.0 if zero_fee else calc_fee('sell', px, shares)
        cash += gross - fee
        trades.append({"date": td, "action": "SELL", "code": code,
                       "name": _name(code), "price": px,
                       "shares": shares, "reason": reason})
        del positions[code]
        # 20日黑名单：所有卖出均生效
        blacklist[code] = i + BLACKLIST_DAYS
        return True

    for i, td in enumerate(trade_dates):
        # ── 开盘：执行昨日标记的止盈 ──
        if pending_sell:
            for code in list(pending_sell):
                if code in positions:
                    row = get_daily_row(code, td)
                    if row and row["vol"] is not None and row["vol"] > 0:
                        if _do_sell(code, td, "profit_take"):
                            if verbose:
                                print(f"  💰 止盈卖出 {code}({_name(code)}) @开")
            pending_sell.clear()

        # ── 调仓日（首日跳过，防前视）──
        if td in weekly and i > 0:
            prev_td = trade_dates[i - 1]
            cands = select_stocks(prev_td, top_n=top_n, div_pct=div_pct,
                                  turn_pct=turn_pct, debt_pct=debt_pct,
                                  size_pct=size_pct, factor_lag=factor_lag,
                                  verbose=verbose)
            new_codes = list(cands.keys())

            if not new_codes:
                if verbose:
                    print(f"  调仓日 {td}：交集为空 → 保持 {len(positions)} 仓")
            else:
                cur = set(positions.keys())
                new_set = set(new_codes)
                if cur != new_set:
                    # 卖出不再入选的持仓
                    for code in list(positions.keys()):
                        if code not in new_set:
                            row = get_daily_row(code, td)
                            if row and row["vol"] is not None and row["vol"] > 0:
                                if _do_sell(code, td, "rebalance_sell"):
                                    if verbose:
                                        print(f"  ✅ 调仓卖出 {code}({_name(code)})")
                    # 买入新候选
                    to_buy = [c for c in new_codes if c not in positions]
                    if to_buy:
                        cash_per = cash / len(to_buy)
                        for code in to_buy:
                            # 黑名单检查
                            if code in blacklist and i <= blacklist[code]:
                                if verbose:
                                    print(f"  ⏸️ 跳过 {code}({_name(code)})：黑名单冷却")
                                continue
                            row = get_daily_row(code, td)
                            if row is None:
                                continue
                            if row["vol"] is not None and row["vol"] == 0:
                                if verbose:
                                    print(f"  ⏸️ 跳过 {code}({_name(code)})：停牌")
                                continue
                            thr = _limit_thr(code)
                            if row["pct_chg"] is not None and row["pct_chg"] <= -thr + 0.005:
                                if verbose:
                                    print(f"  ⏸️ 跳过 {code}({_name(code)})：一字跌停")
                                continue
                            op = qfq_open(code, td)
                            if op is None:
                                continue
                            max_shares = int(cash_per / op / 100) * 100
                            if max_shares < 100:
                                continue
                            cost = max_shares * op
                            fee = 0.0 if zero_fee else calc_fee('buy', op, max_shares)
                            if cost + fee <= cash:
                                cash -= cost + fee
                                positions[code] = {"shares": max_shares, "buy_price": op, "last_price": op}
                                trades.append({"date": td, "action": "BUY",
                                               "code": code, "name": _name(code),
                                               "price": op, "shares": max_shares,
                                               "reason": "weekly_rebalance"})
                                if verbose:
                                    print(f"  ✅ 买入 {code}({_name(code)})："
                                          f"{max_shares}股 @ {op:.2f}")

        # ── 收盘：记账 + 止盈评估 ──
        total = cash
        for code, pos in list(positions.items()):
            px = qfq_close(code, td)
            if px is None:
                # 停牌/无数据时沿用最后已知价格（避免市值丢失）
                px = pos.get("last_price") if pos.get("last_price") else 0
            else:
                pos["last_price"] = px
            total += pos["shares"] * px
        daily_vals.append({"date": td, "value": total})

        # 止盈评估：昨日涨停 + 今日开板
        if positions and i > 0:
            for code in list(positions.keys()):
                if code in pending_sell:
                    continue
                y_row = get_daily_row(code, trade_dates[i - 1])
                t_row = get_daily_row(code, td)
                if y_row is None or t_row is None:
                    continue
                thr = _limit_thr(code)
                y_up = (y_row["pct_chg"] is not None
                        and y_row["pct_chg"] >= thr)
                if not y_up:
                    continue
                if t_row["pre_close"] and t_row["close"] is not None:
                    limit_px = t_row["pre_close"] * (1 + thr)
                    if t_row["close"] < limit_px - 1e-6:
                        pending_sell.add(code)

    # ── 末日平仓（扣费后修正净值）──
    if trade_dates:
        last = trade_dates[-1]
        if positions:
            for code in list(positions.keys()):
                px = qfq_close(code, last)
                if px is not None:
                    _do_sell(code, last, "backtest_end", price=px)
        # 不论是否有持仓，最终净值 = 现金（末日无持仓时 cash == 总资产）
        if daily_vals:
            daily_vals[-1]["value"] = cash
        else:
            daily_vals.append({"date": last, "value": cash})

    # ── 绩效 ──
    return _report(daily_vals, trades, trade_dates, start_date, end_date, zero_fee)


# ════════════════════════════════════════════════════════════
#  5. 绩效计算与报告
# ════════════════════════════════════════════════════════════
def _report(daily_vals, trades, trade_dates, start_date, end_date, zero_fee):
    final_value = daily_vals[-1]["value"] if daily_vals else INIT_CAPITAL
    total_return = (final_value / INIT_CAPITAL - 1) * 100
    days = len(trade_dates)
    years = days / 252
    annual_return = ((final_value / INIT_CAPITAL) ** (1 / years) - 1) * 100 if years > 0 else 0

    vals = np.array([d["value"] for d in daily_vals], dtype=float)
    cummax = np.maximum.accumulate(vals)
    safe = np.where(cummax == 0, 1, cummax)
    max_dd = float(np.min((vals - cummax) / safe)) * 100

    rets = np.diff(vals) / np.where(vals[:-1] == 0, 1, vals[:-1])
    sharpe = ((np.mean(rets) * 252 - 0.025) / (np.std(rets) * np.sqrt(252))
              if len(rets) > 1 and np.std(rets) > 0 else 0.0)

    win_rate, win_cnt, tot_cnt = calc_win_rate(trades)

    # 基准
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

    # 输出
    print(f"\n{'=' * 72}")
    print(f"  回测结果{'【零成本】' if zero_fee else ''}")
    print(f"{'=' * 72}")
    print(f"  初始资金：{INIT_CAPITAL:,.0f}")
    print(f"  最终资产：{final_value:,.0f}")
    print(f"  总收益率：{total_return:+.2f}%")
    print(f"  年化收益：{annual_return:+.2f}%    （视频宣称 +22%）")
    print(f"  最大回撤：{max_dd:.2f}%           （视频宣称 ≤18%）")
    print(f"  夏普比率：{sharpe:.2f}            （视频宣称 >1.6）")
    if tot_cnt > 0:
        print(f"  胜率：{win_rate:.1f}%（{win_cnt}/{tot_cnt}）  （视频宣称 58-65%）")
    print(f"  交易次数：{len(trades)}")
    for idx, r in bench.items():
        print(f"  {INDEX_DISPLAY_NAME.get(idx, idx)}：{r:+.2f}%  超额：{total_return-r:+.2f}%")

    # 保存
    os.makedirs("data/results/weekly_highdiv_vol", exist_ok=True)
    tag = "zero" if zero_fee else "cost"
    csv_path = (f"data/results/weekly_highdiv_vol/"
                f"backtest_n{TOP_N}_d{DIV_PCT}_t{TURN_PCT}_db{DEBT_PCT}_s{SIZE_PCT}"
                f"_{tag}_{start_date}_{end_date}.csv")
    pd.DataFrame(daily_vals).to_csv(csv_path, index=False)
    pd.DataFrame(trades).to_csv(
        csv_path.replace("backtest_", "trades_"), index=False)
    print(f"\n  结果已保存：{csv_path}")

    return {
        "final_value": final_value, "total_return": total_return,
        "annual_return": annual_return, "max_drawdown": max_dd,
        "sharpe": sharpe, "win_rate": win_rate,
        "trades": len(trades), "bench": bench,
    }


# ════════════════════════════════════════════════════════════
#  6. CLI
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="周度高股息高波动策略")
    p.add_argument("start_date", nargs="?", default="20210104")
    p.add_argument("end_date", nargs="?", default="20260710")
    p.add_argument("--top-n", type=int, default=TOP_N)
    p.add_argument("--div-pct", type=int, default=DIV_PCT)
    p.add_argument("--turn-pct", type=int, default=TURN_PCT)
    p.add_argument("--debt-pct", type=int, default=DEBT_PCT)
    p.add_argument("--size-pct", type=int, default=SIZE_PCT)
    p.add_argument("--factor-lag", type=int, default=0,
                   help="因子值回挪交易日数（前视隔离测试）")
    p.add_argument("--zero-fee", action="store_true", help="零成本模式")
    p.add_argument("--quiet", action="store_true", help="减少输出")
    args = p.parse_args()

    run_backtest(args.start_date, args.end_date,
                 top_n=args.top_n, div_pct=args.div_pct,
                 turn_pct=args.turn_pct, debt_pct=args.debt_pct,
                 size_pct=args.size_pct, factor_lag=args.factor_lag,
                  zero_fee=args.zero_fee, verbose=not args.quiet)
