# -*- coding: utf-8 -*-
"""
多因子等权打分选股（价值 + 质量 + 动量）季度调仓回测 — 原型
================================================================
打分选秀模型（参考视频《别再单看市盈率了！机构选股其实是场打分选秀》）：
  每个股票在三个维度上打分，等权平均后全市场排名，买 Top N。
  ① 价值(Value)   : EP = 1/PE_TTM（越便宜分越高）
  ② 质量(Quality) : ROE 高 + 负债率低（ROE 升序分高、负债率降序分高 → 等权平均）
  ③ 动量(Momentum): 12-1 月收益率（跳过最近 1 月，规避短期反转）
  三因子各做截面 percentile(0~1)，等权平均 → 总分 → 降序取前 N。

调仓：每季度第 5 个交易日（1/4/7/10 月），T-1 数据选股、T 开盘执行。
防前视：估值用 T-1 的 daily_basic；财务用 ann_date ≤ T-1 的最新报告；
        动量用 T-1 往前推 12 月/1 月的收盘价（日历对齐到最近交易日，不依赖回测窗口起点）。

复用：run_monthly_rebalance（执行层/费用/基准） + run_ep_neutral（价格缓存、涨跌停助手、基础资料）。
"""
import os
import sys
import calendar
from datetime import datetime

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_monthly_rebalance import (
    get_conn, calc_fee, calc_win_rate, get_trade_dates,
    get_monthly_5th_trading_days,
    INIT_CAPITAL, COMMISSION_RATE, COMMISSION_MIN, STAMP_DUTY_RATE,
    SLIPPAGE_RATE, INDEX_DISPLAY_NAME, compute_reality_discounts,
)
import run_ep_neutral as _ep   # 复用价格缓存 / 涨跌停助手 / 基础资料

# ════════════════════════════════════════════════════════════
#  常量
# ════════════════════════════════════════════════════════════
TOP_N = 30                      # 持仓数量（Top N）
_CAPITAL = 1_000_000           # 初始资金：30只等权，每只~3.3万，足额部署
WINSOR_LO, WINSOR_HI = 0.01, 0.99
EXEC_PRICE = "open"            # open(开盘价) / vwap(日级VWAP代理)
LIMIT_ON = True                # 涨停买不进/跌停卖不出
BENCHMARKS = ["000985.SH", "000300.SH"]   # 中证全指 / 沪深300

_POOL_IDX = {
    "hs300": "000300.SH", "zz500": "000905.SH", "zz800": "000906.SH",
    "zz1000": "000852.SH", "zz2000": "932000.SH", "all": None,
}


# ════════════════════════════════════════════════════════════
#  调仓日历 & 日期助手
# ════════════════════════════════════════════════════════════
def get_quarterly_5th_trading_days(trade_dates):
    """每季度第5交易日（1/4/7/10 月），0基索引第4个。"""
    by_month = {}
    for d in trade_dates:
        by_month.setdefault(d[:6], []).append(d)
    res = []
    for ym in sorted(by_month):
        mm = int(ym[4:6])
        if mm in (1, 4, 7, 10):
            days = by_month[ym]
            if len(days) >= 5:
                res.append(days[4])
    return res


def _shift_month(ymd, delta_months):
    """ymd 'YYYYMMDD' 加减 delta_months 个月（日取原日，月末截断）。"""
    y = int(ymd[:4]); m = int(ymd[4:6]); d = int(ymd[6:8])
    total = (y * 12 + (m - 1)) + delta_months
    ny = total // 12
    nm = total % 12 + 1
    last = calendar.monthrange(ny, nm)[1]
    nd = min(d, last)
    return f"{ny:04d}{nm:02d}{nd:02d}"


def _nearest_td_le(conn, target_ymd):
    """返回 ≤ target_ymd 的最近交易日（字符串），无则 None。"""
    row = conn.execute(
        "SELECT MAX(trade_date) FROM daily WHERE trade_date <= ?",
        (target_ymd,)).fetchone()
    return row[0] if row and row[0] else None


# ════════════════════════════════════════════════════════════
#  选股：三因子等权打分
# ════════════════════════════════════════════════════════════
def select_multifactor(rebalance_date, prev_date, top_n=TOP_N,
                       stock_pool="all", verbose=True,
                       use_value=True, use_quality=True, use_momentum=True):
    """返回 DataFrame[ts_code, score, ep, roe, debt, mom]，等权打分 Top N。

    逻辑：T-1(prev_date) 估值/财务/动量 → 三因子截面 percentile → 等权平均 → 取前 N。
    use_value/use_quality/use_momentum 可独立关闭，用于因子消融实验（同一引擎公平对照）。
    """
    basic = _ep._load_basic()
    conn = get_conn()

    pool_set = (_ep._get_pool_constituents(stock_pool, prev_date)
                if stock_pool and stock_pool != "all" else None)
    if pool_set is not None and verbose:
        print(f"  [股票池] {stock_pool} 时点成分 {len(pool_set)} 只（asof {prev_date}）")

    # 1) 资格过滤（同 run_ep_neutral：剔除 ST / .BJ/金融/公用事业/上市<60天）
    rows = pd.read_sql_query(
        "SELECT DISTINCT ts_code FROM daily WHERE trade_date = ?",
        conn, params=(prev_date,))
    trading = set(str(c) for c in rows["ts_code"].tolist())

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
        if ind in _ep.FINANCIAL_INDUSTRIES or ind in _ep.UTILITY_INDUSTRIES:
            continue
        ld = info["list_date"]
        if ld:
            try:
                if (d_rb - datetime.strptime(ld, "%Y%m%d")).days < _ep.IPO_MIN_DAYS:
                    continue
            except Exception:
                pass
        eligible.add(c)

    # 2) 价值因子 EP = 1/pe_ttm（T-1 daily_basic）
    pe = pd.read_sql_query(
        "SELECT ts_code, pe_ttm FROM daily_basic WHERE trade_date = ? AND pe_ttm > 0",
        conn, params=(prev_date,))
    pe["ts_code"] = pe["ts_code"].astype(str)
    pe = pe[pe["ts_code"].isin(eligible)].copy()
    if pe.empty:
        conn.close()
        return pd.DataFrame()

    # 3) 质量因子（高且稳的 ROE）：取 ann_date ≤ T-1 的近 4 年年度 ROE，
    #    计算 ROE 水平(近3年均值) 与 稳定性(变异系数 CV=std/|均值|，越低越稳)。
    #    用 _shift_month(-48) 限定近4年，避免拉全历史 fina_indicator 撑爆内存。
    lo_date = _shift_month(prev_date, -48)
    fi = pd.read_sql_query(
        "SELECT ts_code, roe, ann_date FROM fina_indicator "
        "WHERE ann_date IS NOT NULL AND ann_date <= ? AND ann_date >= ? "
        "AND roe IS NOT NULL AND roe <> 0",
        conn, params=(prev_date, lo_date))
    conn.close()
    roe_mean_map, roe_cv_map = {}, {}
    if not fi.empty:
        fi["ts_code"] = fi["ts_code"].astype(str)
        fi = fi.sort_values("ann_date")
        for c, g in fi.groupby("ts_code"):
            vals = g["roe"].tail(3).tolist()      # 最近 3 个年度 ROE
            if len(vals) >= 2:
                m = float(np.mean(vals))
                s = float(np.std(vals))
                roe_mean_map[c] = m
                roe_cv_map[c] = (s / abs(m)) if m != 0 else 99.0

    # 4) 动量因子（12-1 月，原始收盘价，日历对齐到最近交易日）
    mom = {}
    if use_momentum:
        conn2 = get_conn()
        t1 = _nearest_td_le(conn2, _shift_month(prev_date, -1))    # ~1个月前
        t12 = _nearest_td_le(conn2, _shift_month(prev_date, -12))  # ~12个月前
        close_map = {}
        for td in (t1, t12):
            if td:
                r = pd.read_sql_query(
                    "SELECT ts_code, close FROM daily WHERE trade_date = ?",
                    conn2, params=(td,))
                close_map[td] = dict(zip(r["ts_code"].astype(str), r["close"].astype(float)))
        conn2.close()
        if t1 and t12:
            for c in eligible:
                p1 = close_map[t1].get(c)
                p12 = close_map[t12].get(c)
                if p1 and p12 and p12 > 0:
                    mom[c] = p1 / p12 - 1

    # 5) 合并因子（按开关保留列）
    df = pe[["ts_code"]].copy()
    df["ep"] = df["ts_code"].map(dict(zip(pe["ts_code"], pe["pe_ttm"])))
    df["ep"] = 1.0 / df["ep"].astype(float)
    if use_quality:
        df["roe"] = df["ts_code"].map(roe_mean_map)
        df["roecv"] = df["ts_code"].map(roe_cv_map)
    if use_momentum:
        df["mom"] = df["ts_code"].map(mom)
    need = ["ep"]
    if use_quality:
        need += ["roe", "roecv"]
    if use_momentum:
        need += ["mom"]
    df = df.dropna(subset=need).copy()
    if df.empty:
        if verbose:
            print(f"  [选股 {rebalance_date}] 因子齐备候选为空")
        return pd.DataFrame()

    # 6) 缩尾 + 截面 percentile 打分（0~1，越大越好）
    score_cols = []
    if use_value:
        lo, hi = df["ep"].quantile([WINSOR_LO, WINSOR_HI])
        df["ep_w"] = df["ep"].clip(lo, hi)
        df["value_s"] = df["ep_w"].rank(pct=True)                 # 便宜→高分
        score_cols.append("value_s")
    if use_quality:
        # 质量 = 高 ROE(水平) + 稳 ROE(低CV)，两者等权
        lo, hi = df["roe"].quantile([WINSOR_LO, WINSOR_HI])
        df["roe_w"] = df["roe"].clip(lo, hi)
        df["roe_level_s"] = df["roe_w"].rank(pct=True)            # ROE高→高分
        lo, hi = df["roecv"].quantile([WINSOR_LO, WINSOR_HI])
        df["roecv_w"] = df["roecv"].clip(lo, hi)
        df["roe_stable_s"] = df["roecv_w"].rank(pct=True, ascending=False)  # 低CV→高分
        df["qual_s"] = df[["roe_level_s", "roe_stable_s"]].mean(axis=1)
        score_cols.append("qual_s")
    if use_momentum:
        lo, hi = df["mom"].quantile([WINSOR_LO, WINSOR_HI])
        df["mom_w"] = df["mom"].clip(lo, hi)
        df["mom_s"] = df["mom_w"].rank(pct=True)                # 动量大→高分
        score_cols.append("mom_s")
    df["score"] = df[score_cols].mean(axis=1)

    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    if top_n is not None:
        df = df.head(top_n)
    if verbose:
        parts = []
        if use_value:
            parts.append("价值")
        if use_quality:
            parts.append("质量(高且稳ROE)")
        if use_momentum:
            parts.append("动量")
        print(f"  [选股 {rebalance_date}] 候选池 {len(eligible)} → "
              f"{'+'.join(parts)}齐备 {len(df)} 只 → 等权打分取 Top {top_n or '全部'}")
    out_cols = ["ts_code", "score", "ep"]
    if use_quality:
        out_cols += ["roe", "roecv"]
    if use_momentum:
        out_cols += ["mom"]
    return df[out_cols]


# ════════════════════════════════════════════════════════════
#  季度回测引擎（结构照搬 run_ep_neutral.run_backtest，季度调仓 + 三因子）
# ════════════════════════════════════════════════════════════
def run_backtest(start_date="20120401", end_date="20260715", top_n=TOP_N,
                 verbose=True, stock_pool="all", exec_price="open",
                 limit_on=True, slippage=0.001,
                 use_value=True, use_quality=True, use_momentum=True,
                 rebalance="quarterly",
                 interrupt_start=None, interrupt_months=0, interrupt_pct=0.0):
    global EXEC_PRICE, LIMIT_ON
    EXEC_PRICE = exec_price
    LIMIT_ON = limit_on
    _ep.EXEC_PRICE = exec_price
    _ep.LIMIT_ON = limit_on
    import run_monthly_rebalance as _rm
    _rm.SLIPPAGE_RATE = slippage

    fac = []
    if use_value:
        fac.append("价值(EP)")
    if use_quality:
        fac.append("质量(高且稳ROE)")
    if use_momentum:
        fac.append("动量(12-1月)")
    fac_str = " + ".join(fac) if fac else "（无因子）"

    cadence_note = ("每季度第5交易日（1/4/7/10月）" if rebalance == "quarterly"
                    else "每月第5交易日（月度调仓）")

    print("=" * 72)
    print("  多因子等权打分调仓回测 — 原型（因子消融 / 单因子验证）")
    print("=" * 72)
    print(f"  区间：{start_date} ~ {end_date}")
    cap_note = f"{top_n}只等权" if top_n else "全部等权"
    print(f"  持仓：{cap_note} | 调仓：{cadence_note}")
    print(f"  剔除：ST / .BJ / 金融 / 公用事业 / 上市<60天")
    print(f"  因子：{fac_str}，各截面pct等权平均")
    ex = "开盘价" if EXEC_PRICE == "open" else "日级VWAP代理(amount×10/vol)"
    print(f"  成交价假设：{ex} | 涨跌停约束：{'开(涨停买不进/跌停卖不出)' if LIMIT_ON else '关'} "
          f"| T+1：{'季度' if rebalance=='quarterly' else '月度'}调仓天然满足")
    print(f"  佣金万{COMMISSION_RATE*1e4:.1f}(最低{COMMISSION_MIN}) "
          f"印花税千1→千0.5(2023-08-28起) 滑点{slippage*100:.2f}%")
    print(f"  初始资金：{_CAPITAL:,.0f}\n")

    trade_dates = get_trade_dates(start_date, end_date)
    if rebalance == "monthly":
        rebal_list = get_monthly_5th_trading_days(trade_dates)
    else:
        rebal_list = get_quarterly_5th_trading_days(trade_dates)
    rebal_set = {d for d in rebal_list if start_date <= d <= end_date}
    cadence_label = "月度" if rebalance == "monthly" else "季度"
    print(f"  交易日 {len(trade_dates)} 天，{cadence_label}调仓 {len(rebal_set)} 次\n")

    positions = {}   # code -> {shares, buy_price, last_price}
    cash = float(_CAPITAL)
    daily_vals = []
    trades = []
    name_cache = {}
    limit_up_skip = 0
    limit_down_skip = 0
    _ep._DAY_PX.clear()

    def _name(code):
        if code not in name_cache:
            name_cache[code] = _ep._get_name(code)
        return name_cache[code]

    for i, td in enumerate(trade_dates):
        # ── 调仓日（首日跳过，用 T-1 数据选股、T 开盘执行）──
        if td in rebal_set and i > 0:
            prev_td = trade_dates[i - 1]
            sel = select_multifactor(td, prev_date=prev_td, top_n=top_n,
                                     stock_pool=stock_pool, verbose=verbose,
                                     use_value=use_value, use_quality=use_quality,
                                     use_momentum=use_momentum)
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
                            op = _ep._exec_price(code, td)
                            if op is None:
                                continue
                            if _ep._is_limit_down(code, td):
                                limit_down_skip += 1
                                if verbose:
                                    print(f"  ⏸ 跌停未卖 {code}({_name(code)})："
                                          f"跌停约束，保留至下季再卖")
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
                    to_buy = [c for c in new_codes if c not in positions]
                    if to_buy:
                        cash_per = cash / len(to_buy)
                        for code in to_buy:
                            op = _ep._exec_price(code, td)
                            if op is None:
                                continue
                            if _ep._is_limit_up(code, td):
                                limit_up_skip += 1
                                if verbose:
                                    print(f"  ⏸ 涨停未买 {code}({_name(code)})："
                                          f"涨停约束，跳过买入")
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
                                               "reason": "multifactor"})
                                if verbose:
                                    print(f"  ✅ 买入 {code}({_name(code)})："
                                          f"{max_shares}股 @ {op:.2f}")
                else:
                    if verbose:
                        print(f"\n调仓日 {td}：持仓不变")

        # ── 每日市值记录 ──
        total = cash
        for code, pos in list(positions.items()):
            px = _ep._px(code, td, "close")
            if px is None:
                px = pos.get("last_price") or 0
            else:
                pos["last_price"] = px
            total += pos["shares"] * px
        daily_vals.append({"date": td, "value": total})

        # 每日清掉价格缓存，避免全区间交易日堆积撑爆内存（沙箱/长区间回测必需）
        _ep._DAY_PX.clear()

    # ── 末日平仓 ──
    if trade_dates:
        last = trade_dates[-1]
        for code in list(positions.keys()):
            px = _ep._px(code, last, "close")
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

    return _report(daily_vals, trades, trade_dates, start_date, end_date,
                   top_n=top_n, exec_price=exec_price, limit_on=limit_on,
                   slippage=slippage, limit_up_skip=limit_up_skip,
                   limit_down_skip=limit_down_skip,
                   use_value=use_value, use_quality=use_quality,
                   use_momentum=use_momentum, rebalance=rebalance,
                   interrupt_start=interrupt_start, interrupt_months=interrupt_months,
                   interrupt_pct=interrupt_pct)


# ════════════════════════════════════════════════════════════
#  报告（照搬 run_ep_neutral._report，改标签与落盘目录）
# ════════════════════════════════════════════════════════════
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
            exec_price="open", limit_on=True, slippage=0.001,
            limit_up_skip=0, limit_down_skip=0,
            use_value=True, use_quality=True, use_momentum=True,
            rebalance="quarterly",
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
    ex = "open" if exec_price == "open" else "vwap"
    print(f"  成交价：{ex} | 涨跌停约束：{'开' if limit_on else '关'} | 滑点：{slippage*100:.2f}%")
    if limit_on:
        print(f"  涨跌停跳过：涨停未买 {limit_up_skip} 次 | 跌停未卖 {limit_down_skip} 次")
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

    yr = _yearly_returns(daily_vals)
    if yr:
        print(f"\n{'—' * 72}")
        print(f"  逐年收益（组合）")
        for y in sorted(yr):
            print(f"    {y}: {yr[y]:+.2f}%")

    os.makedirs("data/results/multifactor", exist_ok=True)
    ex_tag = f"_{ex}{'' if limit_on else '_nolim'}"
    fac_tag = ("V" if use_value else "") + ("Q" if use_quality else "") + ("M" if use_momentum else "")
    if not fac_tag:
        fac_tag = "none"
    reb_tag = "m" if rebalance == "monthly" else "q"
    tag = f"n{top_n}{ex_tag}_{fac_tag}_{reb_tag}"
    csv_path = (f"data/results/multifactor/"
                f"backtest_{tag}_c{int(_CAPITAL)}_{start_date}_{end_date}.csv")
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
    p = argparse.ArgumentParser(description="多因子等权打分（价值+质量+动量）季度调仓")
    p.add_argument("start_date", nargs="?", default="20120401")
    p.add_argument("end_date", nargs="?", default="20260715")
    p.add_argument("--top-n", type=int, default=TOP_N, help="持仓数量（默认30）")
    p.add_argument("--capital", type=int, default=_CAPITAL, help="初始资金（默认100万）")
    p.add_argument("--stock-pool", default="all",
                   help="股票池 hs300/zz500/zz800/zz1000/zz2000/all（默认全A股）")
    p.add_argument("--quiet", action="store_true", help="减少输出")
    p.add_argument("--exec-price", default="open", choices=["open", "vwap"],
                   help="成交价假设: open(开盘价,默认) / vwap(日级VWAP代理)")
    p.add_argument("--slippage", type=float, default=0.001, help="滑点率(默认0.001)")
    p.add_argument("--no-limit", action="store_true", help="关闭涨跌停约束")
    p.add_argument("--no-momentum", dest="use_momentum", action="store_false",
                   help="关闭动量因子（消融实验）")
    p.add_argument("--no-quality", dest="use_quality", action="store_false",
                   help="关闭质量因子（消融实验）")
    p.add_argument("--no-value", dest="use_value", action="store_false",
                   help="关闭价值因子（消融实验）")
    p.add_argument("--rebalance", default="quarterly", choices=["quarterly", "monthly"],
                   help="调仓频率: quarterly(季,默认) / monthly(月)")
    p.add_argument("--select-quarter", type=str, default=None,
                   help="仅输出某季度选股结果(YYYYMM，须为1/4/7/10月)，不回测")
    p.add_argument("--interrupt-start", type=str, default=None,
                   help="现实折扣-中断模拟：从 YYYYMM 起撤出部分资金（配合 --interrupt-months/--interrupt-pct）")
    p.add_argument("--interrupt-months", type=int, default=0,
                   help="中断模拟持续月数（默认0=关闭）")
    p.add_argument("--interrupt-pct", type=float, default=0.0,
                   help="中断模拟撤出比例(0~1，如 0.5=撤一半)，默认0")
    args = p.parse_args()

    _CAPITAL = args.capital
    top_n = args.top_n
    verbose = not args.quiet

    if args.select_quarter:
        ym = args.select_quarter
        if len(ym) != 6 or int(ym[4:6]) not in (1, 4, 7, 10):
            print("  [错误] --select-quarter 须为季度月份 YYYYMM（1/4/7/10）")
            sys.exit(1)
        td_all = get_trade_dates(ym + "01", ym + "31")
        if len(td_all) < 5:
            print(f"  [错误] {ym} 交易日不足5天")
            sys.exit(1)
        rb = td_all[4]
        # 找 rb 之前最近交易日作为 T-1
        prev_all = get_trade_dates("20000101", rb)
        prev_td = prev_all[-2] if len(prev_all) >= 2 else None
        print(f"  季度选股 {ym}：调仓日={rb} (T-1={prev_td})")
        sel = select_multifactor(rb, prev_date=prev_td, top_n=top_n,
                                 stock_pool=args.stock_pool, verbose=verbose)
        if sel.empty:
            print("  无候选")
        else:
            pd.set_option("display.width", 200)
            pd.set_option("display.max_rows", 100)
            print(sel.to_string(index=False))
        sys.exit(0)

    run_backtest(
        start_date=args.start_date, end_date=args.end_date, top_n=top_n,
        verbose=verbose, stock_pool=args.stock_pool,
        exec_price=args.exec_price, limit_on=not args.no_limit,
        slippage=args.slippage,
        use_value=args.use_value, use_quality=args.use_quality,
        use_momentum=args.use_momentum, rebalance=args.rebalance,
        interrupt_start=args.interrupt_start,
        interrupt_months=args.interrupt_months,
        interrupt_pct=args.interrupt_pct,
    )
