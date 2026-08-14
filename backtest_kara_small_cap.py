# -*- coding: utf-8 -*-
"""
Kara 小市值策略 · 平台适配版复现（v2：日频模拟 + 可选移动止损）
===============================================================
视频：BV15X5t6rE12（Kara说量化，3年135%/最大回撤31%/夏普2.43）

原版宇宙：北交所(AKShare) 最小20只 · 月频 · 无过滤器。
平台适配（用户要求）：
  - 数据源改用本地库 D:/tu-shareData/astock_daily.db
  - 宇宙：全市场(all)，【屏蔽老三板(4xx)与北交所(8xx/920)，放行科创板(688)】
  - 选股：流通市值(circ_mv)最小 N 只，月频（每月第5交易日）等权调仓，零过滤器
  - 成本：复用 run_monthly_rebalance 的 calc_fee / get_open_price / 历史印花税分段 / 0.1% 滑点
  - 平台既有护栏：剔除ST、剔除次新股(上市<1年)、2<=价<=100、流动性门槛(可配)

v2 改动（2026-08-14，受 tigerman88 BV17t3k65Eva 启发）：
  - 模拟从"仅月频估值"升级为"日频循环"：调仓日仍只在每月第5交易日，
    但日间逐日更新持仓峰价并监控移动止损，使回撤/夏普更真实
    （月频快照会系统性低估回撤）。stop=0 时总收益与原版完全一致。
  - 新增 --trailing-stop（默认 0 = 关闭；设 0.10 = 10% 移动止损）：
    持仓收盘价跌破 自买入以来峰值 的 (1-stop) 时，于次日开盘止损卖出。

用法：
  python backtest_kara_small_cap.py --start 20200101 --end 20251231 --hold-count 20
  python backtest_kara_small_cap.py --start 20200101 --end 20251231 --trailing-stop 0.10
  python backtest_kara_small_cap.py --start 20230101 --end 20251231 --trailing-stop 0.08
"""
import sys, os, sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from bisect import bisect_right

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_monthly_rebalance import (
    calc_fee, get_open_price, get_price, get_trade_dates,
    get_monthly_5th_trading_days, INIT_CAPITAL, reset_fee_ctx, set_trade_date_ctx,
    get_conn, COMMISSION_RATE, COMMISSION_MIN, SLIPPAGE_RATE,
)
from src.small_cap_rotation_selector import (SmallCapRotationSelector, MIN_AVG_AMOUNT_K,
                                             limit_up_ratio, limit_down_ratio)


def _prev_trade_date(trade_dates, td):
    i = trade_dates.index(td)
    return trade_dates[i - 1] if i > 0 else td


def _bench_total_return(index_code, start, end):
    """指数买入持有总收益（覆盖 [start,end]）。"""
    conn = get_conn()
    row = pd.read_sql_query(
        "SELECT trade_date, close FROM index_daily WHERE ts_code=? AND trade_date BETWEEN ? AND ? ORDER BY trade_date",
        conn, params=(index_code, start, end))
    conn.close()
    if len(row) < 2:
        return None, None, None
    s = float(row.iloc[0]["close"]); e = float(row.iloc[-1]["close"])
    return e / s - 1.0, s, e


def run_backtest(start="20200101", end="20251231", hold_count=20,
                 pool_mode="zz2000", min_avg_amount_k=None, exclude_688=False,
                 trailing_stop=0.0, verbose=False):
    reset_fee_ctx()
    trade_dates = get_trade_dates(start, end)
    if len(trade_dates) < 2:
        print("交易日不足"); return None
    reb_days = get_monthly_5th_trading_days(trade_dates)
    reb_days = [d for d in reb_days if start <= d <= end]

    # 第一轮：收集所有曾入选的标的，便于批量预载收盘价/开盘价
    sel = SmallCapRotationSelector(
        hold_count=hold_count, pool_mode=pool_mode,
        min_avg_amount_k=(min_avg_amount_k if min_avg_amount_k is not None else MIN_AVG_AMOUNT_K),
        fundamental_filter=False, exclude_delisted=False, exclude_st=True, quality_filter=False,
        growth_tilt=False, vol_filter=False, industry_cap=0, exclude_688=exclude_688,
    )
    targets_by_day = {}
    all_codes = set()
    for T in reb_days:
        snap = _prev_trade_date(trade_dates, T)
        codes = sel.select_stocks(snap)
        targets_by_day[T] = codes
        all_codes.update(codes)

    # 批量预载收盘价 / 开盘价（覆盖全部交易日，供日频模拟使用）
    conn = get_conn()
    close_df = pd.read_sql_query(
        "SELECT ts_code, trade_date, close FROM daily WHERE ts_code IN (%s)"
        % ",".join("?" * len(all_codes)),
        conn, params=list(all_codes)) if all_codes else pd.DataFrame(columns=["ts_code","trade_date","close"])
    open_df = pd.read_sql_query(
        "SELECT ts_code, trade_date, open FROM daily WHERE ts_code IN (%s)"
        % ",".join("?" * len(all_codes)),
        conn, params=list(all_codes)) if all_codes else pd.DataFrame(columns=["ts_code","trade_date","open"])
    conn.close()

    # 每只股票的 {td: price} 与排序后的 td 列表（bisect 加速前复权回退）
    code_close, code_open, code_dates = {}, {}, {}
    for r in close_df.itertuples():
        code_close.setdefault(r.ts_code, {})[r.trade_date] = float(r.close)
    for r in open_df.itertuples():
        code_open.setdefault(r.ts_code, {})[r.trade_date] = float(r.open)
    for code, dmap in code_close.items():
        code_dates[code] = sorted(dmap.keys())

    def cprice(code, td):
        dmap = code_close.get(code)
        if not dmap: return None
        if td in dmap: return dmap[td]
        dl = code_dates[code]
        i = bisect_right(dl, td) - 1
        return dmap[dl[i]] if i >= 0 else None

    def oprice(code, td):
        v = code_open.get(code, {}).get(td)
        if v is not None and v > 0: return v
        return cprice(code, td)

    # 模拟状态
    cash = INIT_CAPITAL
    positions = {}      # code -> shares
    entry_px = {}       # code -> 加权买入均价
    peak_px = {}        # code -> 自买入以来收盘价峰值
    pending_sell = set()  # 昨日触发止损、今日开盘卖出
    stopped_today = set()  # 今日已止损（本月不再买回）
    n_stop_exits = 0
    equity_curve = []

    reb_set = set(reb_days)

    def equity_at(td):
        eq = cash
        for code, sh in positions.items():
            p = cprice(code, td)
            if p: eq += sh * p
        return eq

    for td in trade_dates:
        # 1) 执行昨日触发的止损卖出（今日开盘价）
        if pending_sell:
            for code in list(pending_sell):
                sh = positions.get(code, 0)
                if sh <= 0:
                    pending_sell.discard(code); stopped_today.discard(code); continue
                op = oprice(code, td)
                if op is None or op <= 0:
                    continue  # 无开盘价，留给次日
                pc = cprice(code, _prev_trade_date(trade_dates, td))
                if pc and op / pc <= limit_down_ratio(code, td) + 1e-9:
                    continue  # 跌停卖不出，继续挂起
                proceeds = sh * op - calc_fee("sell", op, sh, td)
                cash += proceeds
                positions.pop(code, None); entry_px.pop(code, None); peak_px.pop(code, None)
                pending_sell.discard(code); stopped_today.add(code); n_stop_exits += 1

        # 2) 调仓日：月频等权再平衡（逻辑与原版一致）
        if td in reb_set:
            target = targets_by_day[td]
            if target:
                # 调仓前总权益（含当日所有持仓开盘价）
                eq_before = cash
                for code, sh in positions.items():
                    eq_before += sh * oprice(code, td)
                per = eq_before / len(target)
                new_pos = {}
                # loop1：卖出不在目标里的 / 把目标里的调整到目标权重
                for code, sh in list(positions.items()):
                    op = oprice(code, td)
                    if op is None or op <= 0:
                        new_pos[code] = sh; continue
                    if code not in target:
                        pc = cprice(code, _prev_trade_date(trade_dates, td))
                        if pc and op / pc <= limit_down_ratio(code, td) + 1e-9:
                            new_pos[code] = sh; continue  # 跌停卖不出 -> 续持
                        proceeds = sh * op - calc_fee("sell", op, sh, td)
                        cash += proceeds
                    else:
                        if code in stopped_today:
                            new_pos[code] = sh; continue  # 今日已止损，本月不再调
                        desired = int(per / op // 100) * 100
                        if desired < 100: desired = 0
                        if desired < sh:
                            sell_sh = sh - desired
                            cash += sell_sh * op - calc_fee("sell", op, sell_sh, td)
                            new_pos[code] = desired
                        elif desired > sh:
                            buy_sh = desired - sh
                            cost = buy_sh * op + calc_fee("buy", op, buy_sh, td)
                            if cost <= cash:
                                cash -= cost
                                if sh > 0:
                                    entry_px[code] = (entry_px.get(code, op) * sh + op * buy_sh) / (sh + buy_sh)
                                else:
                                    entry_px[code] = op
                                peak_px[code] = max(peak_px.get(code, op), op)
                                new_pos[code] = desired
                            else:
                                aff = int(cash / (op * (1 + COMMISSION_RATE + SLIPPAGE_RATE)) // 100) * 100
                                if aff > 0:
                                    cash -= aff * op + calc_fee("buy", op, aff, td)
                                    if sh > 0:
                                        entry_px[code] = (entry_px.get(code, op) * sh + op * aff) / (sh + aff)
                                    else:
                                        entry_px[code] = op
                                    peak_px[code] = max(peak_px.get(code, op), op)
                                    new_pos[code] = sh + aff
                                else:
                                    new_pos[code] = sh
                        else:
                            new_pos[code] = sh
                # loop2：买入目标里的新票
                for code in target:
                    if code in new_pos: continue
                    if code in stopped_today: continue
                    op = oprice(code, td)
                    if op is None or op <= 0: continue
                    pc = cprice(code, _prev_trade_date(trade_dates, td))
                    if pc and op / pc >= limit_up_ratio(code, td) - 1e-9: continue  # 涨停买不进
                    desired = int(per / op // 100) * 100
                    if desired < 100: continue
                    cost = desired * op + calc_fee("buy", op, desired, td)
                    if cost <= cash:
                        cash -= cost; entry_px[code] = op
                        peak_px[code] = max(peak_px.get(code, op), op)
                        new_pos[code] = desired
                    else:
                        aff = int(cash / (op * (1 + COMMISSION_RATE + SLIPPAGE_RATE)) // 100) * 100
                        if aff >= 100:
                            cash -= aff * op + calc_fee("buy", op, aff, td)
                            entry_px[code] = op
                            peak_px[code] = max(peak_px.get(code, op), op)
                            new_pos[code] = aff
                positions = {k: v for k, v in new_pos.items() if v > 0}

        # 3) 用今日收盘价更新持仓峰值
        for code, sh in positions.items():
            pc = cprice(code, td)
            if pc: peak_px[code] = max(peak_px.get(code, pc), pc)

        # 4) 移动止损监控（收盘触发，次日开盘卖）
        if trailing_stop > 0:
            for code, sh in list(positions.items()):
                if code in pending_sell: continue
                pc = cprice(code, td)
                pk = peak_px.get(code)
                if pc and pk and pc <= pk * (1 - trailing_stop) + 1e-9:
                    pending_sell.add(code)

        # 5) 记录当日权益
        equity_curve.append((td, equity_at(td)))
        stopped_today = set()

    # 末日清算估值
    last_td = trade_dates[-1]
    final_eq = equity_at(last_td)
    if equity_curve and equity_curve[-1][0] != last_td:
        equity_curve.append((last_td, final_eq))

    eq_series = pd.Series({td: e for td, e in equity_curve}).sort_index()
    rets = eq_series.pct_change().dropna()
    total_ret = final_eq / INIT_CAPITAL - 1.0
    n_years = (datetime.strptime(end, "%Y%m%d") - datetime.strptime(start, "%Y%m%d")).days / 365.25
    cagr = (final_eq / INIT_CAPITAL) ** (1 / n_years) - 1 if n_years > 0 else 0
    peak = eq_series.cummax()
    dd = eq_series / peak - 1
    max_dd = dd.min()
    # 夏普（日频 -> 年化 sqrt(252)）
    if len(rets) > 1 and rets.std() > 0:
        sharpe = rets.mean() / rets.std() * np.sqrt(252)
    else:
        sharpe = 0.0
    calmar = (cagr / abs(max_dd)) if max_dd < 0 else 0.0

    # 年度收益
    eq_df = eq_series.reset_index(); eq_df.columns = ["td", "eq"]
    eq_df["year"] = eq_df["td"].str[:4]
    yearly = {}
    for yr, g in eq_df.groupby("year"):
        g = g.sort_values("td")
        yearly[yr] = g["eq"].iloc[-1] / g["eq"].iloc[0] - 1

    # 基准
    bench = {}
    for nm, code in [("中证2000", "932000.SH"), ("沪深300", "000300.SH"), ("中证全指", "000985.SH")]:
        tr, _, _ = _bench_total_return(code, start, end)
        bench[nm] = tr

    return {
        "start": start, "end": end, "hold_count": hold_count, "pool_mode": pool_mode,
        "exclude_688": exclude_688, "trailing_stop": trailing_stop,
        "total_ret": total_ret, "cagr": cagr, "max_dd": max_dd, "sharpe": sharpe,
        "calmar": calmar, "final_eq": final_eq, "init": INIT_CAPITAL,
        "yearly": yearly, "bench": bench, "n_reb": len(reb_days),
        "n_stop_exits": n_stop_exits,
        "reb_days": reb_days, "targets_by_day": targets_by_day,
    }


def _print_report(r):
    if not r: return
    print("=" * 64)
    stop_s = f" · 移动止损{r['trailing_stop']*100:.0f}%" if r.get("trailing_stop", 0) > 0 else " · 无止损"
    print(f"Kara 适配版 · 全市场最小{r['hold_count']}只 · {r['start']}~{r['end']} · 月频等权{stop_s}")
    print(f"宇宙：all（屏蔽老三板/北交所，含科创板）· 零过滤器 · 平台成本模型"
          + ("" if r.get("exclude_688") else " · [含688]") + (" · [剔除688]" if r.get("exclude_688") else ""))
    print("-" * 64)
    print(f"总收益   : {r['total_ret']*100:7.2f}%")
    print(f"年化     : {r['cagr']*100:7.2f}%")
    print(f"最大回撤 : {r['max_dd']*100:7.2f}%")
    print(f"夏普     : {r['sharpe']:7.3f}")
    print(f"Calmar   : {r['calmar']:7.3f}")
    print(f"期末净值 : {r['final_eq']:,.0f}  (起始 {r['init']:,.0f})  调仓 {r['n_reb']} 次"
          + (f"  止损卖出 {r.get('n_stop_exits',0)} 次" if r.get("trailing_stop",0) > 0 else ""))
    print("-" * 64)
    print("年度收益:")
    for yr in sorted(r['yearly']):
        print(f"  {yr}: {r['yearly'][yr]*100:7.2f}%")
    print("-" * 64)
    print("基准买入持有:")
    for nm, tr in r['bench'].items():
        print(f"  {nm}: {('%.2f%%'%(tr*100)) if tr is not None else 'N/A'}")
    print("=" * 64)
    print(f"（Kara原版声称：3年135% / 31%回撤 / 夏普2.43 — 其宇宙含北交所，本复现已剔除）")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20200101")
    ap.add_argument("--end", default="20251231")
    ap.add_argument("--hold-count", type=int, default=20)
    ap.add_argument("--pool-mode", default="zz2000")
    ap.add_argument("--min-avg-amount-k", type=float, default=None)
    ap.add_argument("--exclude-688", action="store_true", help="剔除科创板(688)，用于对照加科创板选股的贡献")
    ap.add_argument("--trailing-stop", type=float, default=0.0,
                    help="移动止损比例，0=关闭；例如 0.10 表示 10% 回撤止损（次日开盘卖出）")
    args = ap.parse_args()
    res = run_backtest(start=args.start, end=args.end, hold_count=args.hold_count,
                       pool_mode=args.pool_mode, min_avg_amount_k=args.min_avg_amount_k,
                       exclude_688=args.exclude_688, trailing_stop=args.trailing_stop)
    _print_report(res)
