"""
神奇公式（Magic Formula, Joel Greenblatt）年度选股策略 — 回测
============================================================
选股逻辑（原书两指标排名法）：
  ROC = EBIT / (净营运资本 + 净固定资产)
       净营运资本 = 流动资产合计 - 流动负债合计
       净固定资产 = fix_assets（固定资产净值）
  EY  = EBIT / EV
       EV = 总市值 + 负债合计 - 货币资金
           总市值 = daily_basic.total_mv(万元) × 10000 = 元
 排名：全市场按 ROC 降序排名、按 EY 降序排名，两排名相加，
       取总分最低（即"又好又便宜"）的 N 只，年度调仓。

防偏措施：
  ① 财务数据用 ann_date（公告日）≤ 选股日，杜绝前视偏差
  ② 取最新「年报」(end_date LIKE '%1231')，确保口径一致
  ③ T-1 日数据选股、T 日开盘执行，杜绝日内前视
  ④ 剔除 ST / 688(科创板) / .BJ(北交所) / 金融 / 公用事业 / 上市<60天
  ⑤ 仅用 EBIT>0 且 ROC、EY 分母>0 的票，避免指标无意义

说明：financials 字段在库中为 TEXT 类型，统一 float() 转换。
      total_mv 单位为「万元」，换算为元需 ×10000（已用 P/B 反推钉死）。
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
#  常量（按用户确认的推荐配置）
# ════════════════════════════════════════════════════════════
TOP_N            = 30        # 持仓数量
REBALANCE_MONTH  = 5         # 年度调仓月：5月
IPO_MIN_DAYS     = 60        # 上市<60天剔除
BENCHMARKS       = ["000985.SH", "000300.SH"]   # 中证全指 / 沪深300

# 初始资金：30只等权需足够部署。10万÷30≈3333元/只，A股100股/手
# 导致单价>33.3元的股票无法买入（被跳过）→ 大量空仓、收益被现金稀释。
# 故默认采用 100万，使每只≈3.3万有足额买入空间（可用 --capital 覆盖）。
_CAPITAL         = 1_000_000

# 行业剔除（基于 stock_basic.industry）
FINANCIAL_INDUSTRIES = {"银行", "证券", "保险", "多元金融"}
UTILITY_INDUSTRIES   = {"火力发电", "水力发电", "新型电力", "供气供热", "水务"}

# ════════════════════════════════════════════════════════════
#  缓存
# ════════════════════════════════════════════════════════════
_BASIC      = None   # ts_code -> {industry, name, list_date, excluded}
_FIN_CACHE  = {}     # visible_date -> {ts_code: {...}}
_MV_CACHE   = {}     # trade_date -> {ts_code: total_mv(万元)}
_TRADE_CAL  = None
_RAW_BS     = None   # ts_code -> {end_date: fields}  全部年报（一次性预载）
_RAW_INC    = None   # ts_code -> list of {end_date, ann_date, ebit}


# ════════════════════════════════════════════════════════════
#  基础数据
# ════════════════════════════════════════════════════════════
def _load_basic():
    """加载股票基础信息（含行业、ST/688/BJ 标记）。"""
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
        excluded = (code.startswith("688") or code.endswith(".BJ")
                    or "ST" in name.upper() or name.startswith("*"))
        m[code] = {"name": name, "industry": ind, "list_date": ld,
                   "excluded": excluded}
    _BASIC = m
    return m


def _load_trade_cal():
    global _TRADE_CAL
    if _TRADE_CAL is None:
        conn = get_conn()
        rows = pd.read_sql_query(
            "SELECT DISTINCT trade_date FROM daily ORDER BY trade_date", conn)
        conn.close()
        _TRADE_CAL = [str(d) for d in rows["trade_date"].tolist()]
    return _TRADE_CAL


# ════════════════════════════════════════════════════════════
#  财务数据（point-in-time，取最新年报）
# ════════════════════════════════════════════════════════════
def _avail(end_date, ann_date):
    """年报可用日（point-in-time 防偏核心）。
    - ann_date 有效（非缺失且距期末 ≤200 天，符合监管披露节奏）→ 用公告日
    - ann_date 缺失/污染（如旧年报被标成多年后）→ 回退到法定截止日
      期末 +120 天（A股年报须在次年4月30日前披露）
    返回 datetime。"""
    ed = datetime.strptime(end_date, "%Y%m%d")
    if ann_date and ann_date != "":
        try:
            ad = datetime.strptime(ann_date, "%Y%m%d")
            if ad <= ed + timedelta(days=200):
                return ad
        except (TypeError, ValueError):
            pass
    return ed + timedelta(days=120)


def _load_raw():
    """一次性预载全部年报（balance_sheet / income），按 ts_code 建索引。
    不依赖 visible_date，故只加载一次。"""
    global _RAW_BS, _RAW_INC
    if _RAW_BS is not None:
        return
    conn = get_conn()
    bs = pd.read_sql_query(
        "SELECT ts_code, end_date, ann_date, total_cur_assets, total_cur_liab, "
        "fix_assets, total_liab, money_cap FROM balance_sheet WHERE end_date LIKE '%1231'",
        conn)
    inc = pd.read_sql_query(
        "SELECT ts_code, end_date, ann_date, ebit FROM income "
        "WHERE end_date LIKE '%1231'", conn)
    conn.close()

    _RAW_BS = {}
    for _, r in bs.iterrows():
        code = str(r["ts_code"])
        ed = str(r["end_date"])
        _RAW_BS.setdefault(code, {})[ed] = {
            "ann_date": str(r["ann_date"]) if pd.notna(r["ann_date"]) else "",
            "tca": r["total_cur_assets"], "tcl": r["total_cur_liab"],
            "fix": r["fix_assets"], "liab": r["total_liab"], "cash": r["money_cap"],
        }
    _RAW_INC = {}
    for _, r in inc.iterrows():
        code = str(r["ts_code"])
        _RAW_INC.setdefault(code, []).append({
            "end_date": str(r["end_date"]),
            "ann_date": str(r["ann_date"]) if pd.notna(r["ann_date"]) else "",
            "ebit": r["ebit"],
        })


def _load_financials(visible_date):
    """截至 visible_date 已可用、且 end_date 最大的年报财务数据。
    返回 {ts_code: {ebit, nwc, fix, liab, cash, end_date}}。
    对每只股票：在其全部年报中，选出 avail<=visible_date 且 end_date 最大
    （同 end_date 取 ann_date 最大）的那份，并与 balance_sheet 同期末匹配。

    此举同时修复两类问题：
      ① 旧年报 ann_date 污染（如 2013 年报标成 2015）→ _avail 回退到期末+120天
      ② 不遗漏『最新年报尚未披露、但上一年年报已可用』的中间年份
    """
    if visible_date in _FIN_CACHE:
        return _FIN_CACHE[visible_date]
    _load_raw()
    vd = datetime.strptime(visible_date, "%Y%m%d")
    out = {}
    for code, ilist in _RAW_INC.items():
        best = None  # (end_date, ann_date)
        best_rec = None
        for ir in ilist:
            ed, ad = ir["end_date"], ir["ann_date"]
            if _avail(ed, ad) > vd:
                continue
            b = _RAW_BS.get(code, {}).get(ed)
            if b is None:
                continue
            cand = (ed, ad)
            if best is None or cand > best:
                best = cand
                best_rec = (ir, b)
        if best is None or best_rec is None:
            continue
        ir, b = best_rec
        try:
            ebit = float(ir["ebit"])
            tca = float(b["tca"]); tcl = float(b["tcl"])
            fix = float(b["fix"]); liab = float(b["liab"]); cash = float(b["cash"])
        except (TypeError, ValueError):
            continue
        out[code] = {
            "ebit": ebit, "nwc": tca - tcl, "fix": fix,
            "liab": liab, "cash": cash, "end_date": best[0],
        }
    _FIN_CACHE[visible_date] = out
    return out


def _get_mv_map(trade_date):
    """某日总市值（万元）。无数据则回退到最近的前一个交易日。"""
    if trade_date in _MV_CACHE:
        return _MV_CACHE[trade_date]
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT ts_code, total_mv FROM daily_basic WHERE trade_date = ?",
        conn, params=(trade_date,))
    if len(df) == 0:
        r = conn.execute(
            "SELECT MAX(trade_date) FROM daily_basic WHERE trade_date < ?",
            (trade_date,)).fetchone()
        if r and r[0]:
            df = pd.read_sql_query(
                "SELECT ts_code, total_mv FROM daily_basic WHERE trade_date = ?",
                conn, params=(r[0],))
    conn.close()
    m = {}
    for _, r in df.iterrows():
        if pd.notna(r["total_mv"]):
            m[str(r["ts_code"])] = float(r["total_mv"])
    _MV_CACHE[trade_date] = m
    return m


# ════════════════════════════════════════════════════════════
#  神奇公式选股
# ════════════════════════════════════════════════════════════
# 股票池 → 指数代码（与 run_dogs_annual.get_stock_pool_index 保持一致）
_POOL_IDX = {
    "hs300": "000300.SH",
    "zz500": "000905.SH",
    "zz800": "000906.SH",
    "zz1000": "000852.SH",
    "zz2000": "932000.SH",
    "all": None,
}

def _get_pool_constituents(stock_pool, asof_date):
    """返回指定股票池在 asof_date 时点的成分股 ts_code 集合；all/未知 返回 None（不过滤，全A股）。"""
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


def select_magic_formula(rebalance_date, top_n=TOP_N, prev_date=None, verbose=True, stock_pool="all"):
    """返回 DataFrame: [ts_code, roc, ey, score, ebit, ev, end_date]，
    已按 score 升序（总分最低 = 又好又便宜）。"""
    if prev_date is None:
        prev_date = rebalance_date
    basic = _load_basic()
    conn = get_conn()

    # 股票池过滤（point-in-time：取 asof 成分，避免用未来成分）
    pool_set = _get_pool_constituents(stock_pool, prev_date) if stock_pool and stock_pool != "all" else None
    if pool_set is not None and verbose:
        print(f"  [股票池] {stock_pool} 时点成分 {len(pool_set)} 只（asof {prev_date}）")

    # 1) prev_date 有交易的股票（时点存在性）
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
            # 退市股（不在 stock_basic）：按代码剔除 688/.BJ，保留其余
            if c.startswith("688") or c.endswith(".BJ"):
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
    conn.close()

    if not eligible:
        return pd.DataFrame()

    # 3) point-in-time 财务 + 市值
    fin = _load_financials(prev_date)
    mv_map = _get_mv_map(prev_date)

    recs = []
    for c in eligible:
        f = fin.get(c)
        if f is None:
            continue
        if not np.isfinite(f["ebit"]) or f["ebit"] <= 0:
            continue
        denom = f["nwc"] + f["fix"]
        if not np.isfinite(denom) or denom <= 0:
            continue
        mv = mv_map.get(c)
        if mv is None:
            continue
        ev = mv * 10000.0 + f["liab"] - f["cash"]
        if not np.isfinite(ev) or ev <= 0:
            continue
        roc = f["ebit"] / denom
        ey = f["ebit"] / ev
        if not np.isfinite(roc) or not np.isfinite(ey):
            continue
        recs.append((c, roc, ey, f["ebit"], ev, f["end_date"]))

    if not recs:
        if verbose:
            print(f"  [选股 {rebalance_date}] 候选池 {len(eligible)} 只，无有效财务指标")
        return pd.DataFrame()

    df = pd.DataFrame(recs, columns=["ts_code", "roc", "ey", "ebit", "ev", "end_date"])
    df["rank_roc"] = df["roc"].rank(ascending=False, method="first")
    df["rank_ey"] = df["ey"].rank(ascending=False, method="first")
    df["score"] = df["rank_roc"] + df["rank_ey"]
    df = df.sort_values("score").head(top_n).reset_index(drop=True)

    if verbose:
        print(f"  [选股 {rebalance_date}] 候选池 {len(eligible)} 只 → "
              f"有效 {len(recs)} 只 → 取 {top_n} 只")
    return df


# ════════════════════════════════════════════════════════════
#  年度回测引擎
# ════════════════════════════════════════════════════════════
def _may_rebalance_dates(trade_dates):
    """每年 5 月的第 5 个交易日（与月度引擎一致）。"""
    by_month = {}
    out = set()
    for td in trade_dates:
        if td[4:6] == f"{REBALANCE_MONTH:02d}":
            by_month.setdefault(td[:6], []).append(td)
    for ym, ds in by_month.items():
        out.add(ds[4] if len(ds) >= 5 else ds[-1])
    return out


def run_backtest(start_date="20140101", end_date="20260715",
                 top_n=TOP_N, verbose=True, stock_pool="all",
                 interrupt_start=None, interrupt_months=0, interrupt_pct=0.0):
    print("=" * 72)
    print("  神奇公式（Magic Formula）年度选股策略回测")
    print("=" * 72)
    print(f"  区间：{start_date} ~ {end_date}")
    print(f"  持仓：{top_n}只等权 | 调仓：每年{REBALANCE_MONTH}月第5交易日")
    print(f"  剔除：ST / 688 / .BJ / 金融 / 公用事业 / 上市<60天")
    print(f"  财务口径：ann_date（公告日）取最新年报")
    print(f"  佣金万{COMMISSION_RATE*1e4:.1f}(最低{COMMISSION_MIN}) "
          f"印花税千1→千0.5(2023-08-28起) 滑点{SLIPPAGE_RATE*100:.1f}%")
    print(f"  初始资金：{_CAPITAL:,.0f}（30只等权需足额部署，否则现金拖累）\n")

    trade_dates = get_trade_dates(start_date, end_date)
    rebal_set = _may_rebalance_dates(trade_dates)
    # 仅保留区间内的调仓日
    rebal_set = {d for d in rebal_set if start_date <= d <= end_date}
    print(f"  交易日 {len(trade_dates)} 天，年度调仓 {len(rebal_set)} 次\n")

    positions = {}   # code -> {shares, buy_price, last_price}
    cash = float(_CAPITAL)
    daily_vals = []
    trades = []
    name_cache = {}

    def _name(code):
        if code not in name_cache:
            name_cache[code] = get_stock_name(code)
        return name_cache[code]

    for i, td in enumerate(trade_dates):
        # ── 调仓日（首日跳过，用 T-1 数据选股、T 开盘执行）──
        if td in rebal_set and i > 0:
            prev_td = trade_dates[i - 1]
            sel = select_magic_formula(td, top_n=top_n, prev_date=prev_td, verbose=verbose, stock_pool=stock_pool)
            new_codes = sel["ts_code"].tolist() if not sel.empty else []
            new_set = set(new_codes)

            if not new_codes:
                if verbose:
                    print(f"\n调仓日 {td}：选股为空，保持现有 {len(positions)} 仓")
            else:
                cur = set(positions.keys())
                if cur != new_set:
                    # 卖出不在新池中的旧持仓
                    for code in list(positions.keys()):
                        if code not in new_set:
                            op = get_open_price(code, td)
                            if op is None:
                                continue
                            pos = positions[code]
                            fee = calc_fee('sell', op, pos["shares"])
                            cash += pos["shares"] * op - fee
                            trades.append({"date": td, "action": "SELL",
                                           "code": code, "name": _name(code),
                                           "price": op, "shares": pos["shares"],
                                           "reason": "rebalance"})
                            if verbose:
                                print(f"  ✅ 调仓卖出 {code}({_name(code)})："
                                      f"{pos['shares']}股 @ {op:.2f}")
                            del positions[code]
                    # 买入新选股票
                    to_buy = [c for c in new_codes if c not in positions]
                    if to_buy:
                        cash_per = cash / len(to_buy)
                        for code in to_buy:
                            op = get_open_price(code, td)
                            if op is None:
                                continue
                            max_shares = int(cash_per / op / 100) * 100
                            if max_shares < 100:
                                continue
                            cost = max_shares * op
                            fee = calc_fee('buy', op, max_shares)
                            if cost + fee <= cash:
                                cash -= cost + fee
                                positions[code] = {"shares": max_shares,
                                                   "buy_price": op, "last_price": op}
                                trades.append({"date": td, "action": "BUY",
                                               "code": code, "name": _name(code),
                                               "price": op, "shares": max_shares,
                                               "reason": "magic_formula"})
                                if verbose:
                                    print(f"  ✅ 买入 {code}({_name(code)})："
                                          f"{max_shares}股 @ {op:.2f}")
                else:
                    if verbose:
                        print(f"\n调仓日 {td}：持仓不变")

        # ── 每日市值记录 ──
        total = cash
        for code, pos in list(positions.items()):
            px = get_price(code, td)
            if px is None:
                px = pos.get("last_price") or 0
            else:
                pos["last_price"] = px
            total += pos["shares"] * px
        daily_vals.append({"date": td, "value": total})

    # ── 末日平仓 ──
    if trade_dates:
        last = trade_dates[-1]
        for code in list(positions.keys()):
            px = get_price(code, last)
            if px is not None:
                pos = positions[code]
                fee = calc_fee('sell', px, pos["shares"])
                cash += pos["shares"] * px - fee
                trades.append({"date": last, "action": "SELL", "code": code,
                               "name": _name(code), "price": px,
                               "shares": pos["shares"], "reason": "backtest_end"})
                del positions[code]
        if daily_vals:
            daily_vals[-1]["value"] = cash

    return _report(daily_vals, trades, trade_dates, start_date, end_date, top_n=top_n,
                   interrupt_start=interrupt_start, interrupt_months=interrupt_months,
                   interrupt_pct=interrupt_pct)


def _yearly_returns(daily_vals):
    """按自然年切分组合收益率（年未调仓日近似用年末市值）。"""
    if not daily_vals:
        return {}
    df = pd.DataFrame(daily_vals)
    df["year"] = df["date"].str[:4]
    yrs = {}
    for y, g in df.groupby("year"):
        yrs[y] = (g["value"].iloc[-1] / g["value"].iloc[0] - 1) * 100
    return yrs


def _report(daily_vals, trades, trade_dates, start_date, end_date, top_n=TOP_N,
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

    # 逐年收益
    yr = _yearly_returns(daily_vals)
    if yr:
        print(f"\n{'—' * 72}")
        print(f"  逐年收益（组合）")
        for y in sorted(yr):
            print(f"    {y}: {yr[y]:+.2f}%")

    # 保存
    os.makedirs("data/results/magic_formula", exist_ok=True)
    csv_path = (f"data/results/magic_formula/"
                f"backtest_n{top_n}_c{int(_CAPITAL)}_{start_date}_{end_date}.csv")
    pd.DataFrame(daily_vals).to_csv(csv_path, index=False)
    pd.DataFrame(trades).to_csv(csv_path.replace("backtest_", "trades_"), index=False)
    print(f"\n  结果已保存：{csv_path}")

    return {
        "final_value": final_value, "total_return": total_return,
        "annual_return": annual_return, "max_drawdown": max_dd,
        "sharpe": sharpe, "win_rate": win_rate, "trades": len(trades),
        "bench": bench, "yearly": yr,
    }


# ════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="神奇公式年度选股策略")
    p.add_argument("start_date", nargs="?", default="20140101")
    p.add_argument("end_date", nargs="?", default="20260715")
    p.add_argument("--top-n", type=int, default=TOP_N)
    p.add_argument("--capital", type=int, default=_CAPITAL,
                   help="初始资金（默认100万，30只等权需足额部署）")
    p.add_argument("--quiet", action="store_true", help="减少输出")
    p.add_argument("--stock-pool", default="all",
                   help="股票池 hs300/zz500/zz800/zz1000/zz2000/all（默认全A股）")
    p.add_argument("--select-year", type=str, default=None,
                   help="仅输出某年(YYYY)5月选股结果，不回测")
    p.add_argument("--interrupt-start", type=str, default=None,
                   help="现实折扣-中断模拟：从 YYYYMM 起撤出部分资金（配合 --interrupt-months/--interrupt-pct）")
    p.add_argument("--interrupt-months", type=int, default=0,
                   help="中断模拟持续月数（默认0=关闭）")
    p.add_argument("--interrupt-pct", type=float, default=0.0,
                   help="中断模拟撤出比例(0~1，如 0.5=撤一半)，默认0")
    args = p.parse_args()

    _CAPITAL = args.capital

    if args.select_year:
        # 找到该年5月第5交易日作为调仓日，前一交易日选股
        cal = get_trade_dates(args.start_date if args.start_date > args.select_year + "0101" else args.select_year + "0101",
                              args.select_year + "1231")
        rb = sorted([d for d in cal if d[4:6] == "05"])
        if not rb:
            print("该年无5月交易日")
        else:
            td = rb[4] if len(rb) >= 5 else rb[-1]
            idx = cal.index(td)
            prev = cal[idx - 1] if idx > 0 else td
            sel = select_magic_formula(td, top_n=args.top_n, prev_date=prev, verbose=True, stock_pool=args.stock_pool)
            print(f"\n=== {args.select_year} 神奇公式选股（{len(sel)} 只）===")
            for _, r in sel.iterrows():
                print(f"  {r['ts_code']}({get_stock_name(r['ts_code'])}) "
                      f"ROC={r['roc']:.3f} EY={r['ey']:.4f} "
                      f"score={r['score']:.0f} 财报={r['end_date']}")
    else:
        run_backtest(args.start_date, args.end_date,
                     top_n=args.top_n, verbose=not args.quiet, stock_pool=args.stock_pool,
                     interrupt_start=args.interrupt_start,
                     interrupt_months=args.interrupt_months,
                     interrupt_pct=args.interrupt_pct)
