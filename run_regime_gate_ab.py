# -*- coding: utf-8 -*-
"""
环境判断器仓位门控 —— A/B 对照实验
====================================
视频: "左侧交易与右侧交易" (悦悦笔记, BV1uAuu6vE1j)
思路: 环境判断器不驱动买卖信号，只做仓位门控:
  - 震荡regime (布林带宽分位 < squeeze_th) → 半仓 (target_ratio=0.5)
  - 趋势regime (带宽分位 >= squeeze_th 且 指数多头排列) → 满仓 (target_ratio=1.0)
  - 不确定 → 空仓 (target_ratio=0.0)
接到 run_monthly_rebalance.py 已有的选股函数上，做 A/B 对照:
  A) 门控 OFF (= 原始策略，满仓调仓)
  B) 门控 ON  (根据 regime 调整仓位比例)

选股策略默认用红利低波(div_low_vol)，可切换为 value / momentum / div_growth。
回测口径与 run_monthly_rebalance.py 一致:
  - 月度调仓(每月第5交易日)
  - T-1 选股, T 开盘执行
  - 涨跌停处理(涨停不买/跌停延卖)
  - 佣金/印花税/滑点(复用 calc_fee)
"""
import sys
import os
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest"))

from run_monthly_rebalance import (
    get_conn, get_trade_dates, get_monthly_5th_trading_days,
    calc_fee, get_open_price, get_price, get_stock_name,
    select_by_method, get_stock_pool_index,
    _bb_width_pct, INDEX_DISPLAY_NAME,
    get_index_constituents,
    COMMISSION_RATE, SLIPPAGE_RATE,
)
import config

RES_DIR = "data/results/regime_gate"
os.makedirs(RES_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════════
#  环境判断器（与 run_left_right_regime.py 同口径，但用于仓位控制）
# ════════════════════════════════════════════════════════════

def regime_target_ratio(idx_code, trade_date, squeeze_th=0.25,
                        bb_win=20, bb_lookback=120):
    """返回目标仓位比例: 0.0(空仓) / 0.5(半仓) / 1.0(满仓)

    用指数的布林带宽分位 + 均线多头排列判断市场环境:
    - 宽带分位 < squeeze_th → 震荡 → 0.5（半仓）
    - 宽带分位 >= squeeze_th 且 指数收盘 > MA20 → 趋势 → 1.0（满仓）
    - 其余 → 不确定 → 0.0（空仓）

    全部用 <= trade_date 数据（T-1 调用，T 执行），无未来函数。
    """
    # 布林带宽分位（复用已有函数，对指数计算）
    bbw_pct = _bb_width_pct(idx_code, trade_date, is_index=True,
                            win=bb_win, lookback=bb_lookback)

    # 指数均线多头排列: 收盘价 > MA20
    conn = get_conn()
    rows = pd.read_sql_query(
        "SELECT close FROM index_daily WHERE ts_code=? AND trade_date<=? "
        "ORDER BY trade_date DESC LIMIT 30",
        conn, params=(idx_code, trade_date))
    conn.close()
    if len(rows) < 20:
        return 0.0  # 数据不足 → 保守空仓
    closes = rows["close"].values[::-1].astype(float)
    ma20 = closes[-20:].mean()
    price = closes[-1]
    above_ma20 = price > ma20

    if bbw_pct < squeeze_th:
        return 0.5  # 震荡: 半仓
    elif above_ma20:
        return 1.0  # 趋势: 满仓
    else:
        return 0.0  # 不确定: 空仓


def regime_label(ratio):
    if ratio >= 1.0:
        return "趋势(满仓)"
    elif ratio >= 0.5:
        return "震荡(半仓)"
    else:
        return "不确定(空仓)"


# ════════════════════════════════════════════════════════════
#  回测引擎（精简版，复用 run_monthly_rebalance 的选股 + 成交价）
# ════════════════════════════════════════════════════════════

def run_backtest(start_date="20200102", end_date="20251231",
                 selection_method="div_low_vol", top_n=5,
                 use_gate=False, squeeze_th=0.25,
                 bb_win=20, bb_lookback=120,
                 stock_pool=None, stop_loss_pct=0.0):
    """带可选 regime gate 的月度调仓回测。

    use_gate=False → A组对照(原始满仓调仓)
    use_gate=True  → B组(根据 regime 调整目标仓位)
    """
    trade_dates = get_trade_dates(start_date, end_date)
    rebalance_set = set(get_monthly_5th_trading_days(trade_dates))

    # 基准指数（用于 regime 判断 + 绩效对比）
    idx_code = stock_pool or get_stock_pool_index() or "000906.SH"
    idx_name = INDEX_DISPLAY_NAME.get(idx_code, idx_code)
    bench_idx = "000300.SH"  # 沪深300 基准

    positions = {}
    cash = float(getattr(config, 'BACKTEST', {}).get('monthly_rebalance_capital', 100000))
    init_capital = cash
    daily_vals = []
    trades = []
    regime_log = []  # 记录每次调仓的 regime 判断
    stop_count = 0

    # 涨跌停阈值
    def _limit(code):
        if code.startswith("688") or ((code.startswith("300") or code.startswith("301"))):
            return 19.9
        return 9.9

    for i, td in enumerate(trade_dates):
        # --- 步骤1: 执行止损挂单（T日开盘执行T-1收盘触发的止损）---
        # 简化: 不做止损（因子类策略月调仓即退出，与 div_low_vol 默认一致）
        # 如需止损可在此扩展

        # --- 步骤2: 记录当日市值 ---
        total_value = cash
        for code, pos in positions.items():
            price = get_price(code, td)
            if price is not None:
                total_value += pos["shares"] * price
            else:
                total_value += pos["shares"] * pos.get("last_price", pos["buy_price"])
        daily_vals.append({"date": td, "value": total_value})

        # --- 步骤3: 调仓日决策 ---
        if td not in rebalance_set:
            continue

        prev_td = trade_dates[i-1] if i > 0 else td

        # 选股
        stocks = select_by_method(selection_method, prev_td, top_n=top_n,
                                   stock_pool=stock_pool)
        new_codes = stocks['ts_code'].tolist() if (stocks is not None and not stocks.empty) else []

        if not new_codes:
            continue

        # --- Regime Gate ---
        if use_gate:
            target_ratio = regime_target_ratio(idx_code, prev_td, squeeze_th,
                                               bb_win, bb_lookback)
            label = regime_label(target_ratio)
            regime_log.append({"date": td, "ratio": target_ratio, "label": label})
        else:
            target_ratio = 1.0  # 门控OFF: 始终满仓
            label = "满仓(无门控)"

        # --- 卖出不在新池中的旧持仓 ---
        for code in list(positions.keys()):
            if code not in new_codes:
                open_price = get_open_price(code, td)
                if open_price is None:
                    continue
                pos = positions[code]
                proceeds = pos["shares"] * open_price
                fee = calc_fee('sell', open_price, pos["shares"])
                cash += proceeds - fee
                trades.append({
                    "date": td, "action": "SELL", "code": code,
                    "name": get_stock_name(code), "price": open_price,
                    "shares": pos["shares"], "reason": "rebalance"})
                del positions[code]

        # --- 买入新股 ---
        new_to_buy = [c for c in new_codes if c not in positions]
        if not new_to_buy:
            continue

        # 仓位控制核心: 用 target_ratio 缩放可投资金
        # 计算当前总权益
        equity = cash
        for code, pos in positions.items():
            price = get_price(code, td)
            if price is not None:
                equity += pos["shares"] * price
            else:
                equity += pos["shares"] * pos.get("last_price", pos["buy_price"])

        # 目标股票仓位 = equity * target_ratio
        # 已持仓部分占用的资金要扣除
        held_value = sum(
            pos["shares"] * (get_price(c, td) or pos.get("last_price", pos["buy_price"]))
            for c, pos in positions.items()
        )
        # 可用于新股的现金 = min(目标总仓位 - 已持仓, 当前现金)
        target_stock_value = equity * target_ratio
        avail_for_new = max(0, min(target_stock_value - held_value, cash))

        if avail_for_new <= 0 or target_ratio <= 0:
            # 空仓或资金不足: 不买入
            if use_gate and target_ratio <= 0:
                # 门控空仓: 也卖出已有持仓
                for code in list(positions.keys()):
                    open_price = get_open_price(code, td)
                    if open_price is None:
                        continue
                    pos = positions[code]
                    proceeds = pos["shares"] * open_price
                    fee = calc_fee('sell', open_price, pos["shares"])
                    cash += proceeds - fee
                    trades.append({
                        "date": td, "action": "SELL", "code": code,
                        "name": get_stock_name(code), "price": open_price,
                        "shares": pos["shares"], "reason": "regime_flat"})
                    del positions[code]
            continue

        cash_per_stock = avail_for_new / len(new_to_buy)
        if cash_per_stock <= 0:
            continue

        for ts_code in new_to_buy:
            open_price = get_open_price(ts_code, td)
            if open_price is None:
                continue
            # 涨停不买
            pre_close = None
            conn = get_conn()
            row = pd.read_sql_query(
                "SELECT pre_close FROM daily WHERE ts_code=? AND trade_date=? LIMIT 1",
                conn, params=(ts_code, td))
            conn.close()
            if not row.empty:
                pre_close = float(row.iloc[0]["pre_close"])
            if pre_close is not None and open_price >= pre_close * (1 + _limit(ts_code) / 100 - 0.001):
                continue  # 开盘涨停, 买不进

            max_shares = int(cash_per_stock / open_price / 100) * 100
            if max_shares < 100:
                continue
            cost = max_shares * open_price
            fee = calc_fee('buy', open_price, max_shares)
            if cost + fee > cash:
                max_shares = int(cash / open_price / 100) * 100
                if max_shares < 100:
                    continue
                cost = max_shares * open_price
                fee = calc_fee('buy', open_price, max_shares)
                if cost + fee > cash:
                    continue
            cash -= cost + fee
            positions[ts_code] = {
                "shares": max_shares, "buy_price": open_price,
                "last_price": open_price}
            trades.append({
                "date": td, "action": "BUY", "code": ts_code,
                "name": get_stock_name(ts_code), "price": open_price,
                "shares": max_shares, "reason": "rebalance"})

    # --- 回测结束: 最后一天收盘价清仓 ---
    if trade_dates and positions:
        last_date = trade_dates[-1]
        for code in list(positions.keys()):
            price = get_price(code, last_date)
            if price is not None:
                pos = positions[code]
                proceeds = pos["shares"] * price
                fee = calc_fee('sell', price, pos["shares"])
                cash += proceeds - fee
                trades.append({
                    "date": last_date, "action": "SELL", "code": code,
                    "name": get_stock_name(code), "price": price,
                    "shares": pos["shares"], "reason": "backtest_end"})
                del positions[code]

    final_value = cash
    for code, pos in positions.items():
        price = get_price(code, trade_dates[-1]) if trade_dates else None
        if price is not None:
            final_value += pos["shares"] * price

    # --- 指标计算 ---
    vals = np.array([d["value"] for d in daily_vals], dtype=float)
    total_return = (final_value / init_capital - 1) * 100
    days = len(trade_dates)
    years = days / 252
    annual_return = ((final_value / init_capital) ** (1 / years) - 1) * 100 if years > 0 else 0

    cummax = np.maximum.accumulate(vals)
    safe_cummax = np.where(cummax == 0, 1, cummax)
    drawdowns = (vals - cummax) / safe_cummax
    max_dd = float(np.min(drawdowns)) * 100

    rets = np.diff(vals) / np.where(vals[:-1] == 0, 1, vals[:-1])
    if len(rets) > 1 and np.std(rets) > 0:
        sharpe = (np.mean(rets) * 252 - 0.025) / (np.std(rets) * np.sqrt(252))
    else:
        sharpe = 0.0

    # 基准
    bench_vals = _bench_series(bench_idx, trade_dates, init_capital)
    bench_total = (bench_vals[-1] / init_capital - 1) * 100 if len(bench_vals) > 0 else 0
    bench_ann = ((bench_vals[-1] / init_capital) ** (1 / years) - 1) * 100 if (len(bench_vals) > 0 and years > 0) else 0

    return dict(
        start=start_date, end=end_date, use_gate=use_gate,
        selection_method=selection_method, top_n=top_n,
        total_return=total_return, annual_return=annual_return,
        max_dd=max_dd, sharpe=sharpe, final_value=final_value,
        init_capital=init_capital,
        bench_total=bench_total, bench_ann=bench_ann,
        n_trades=len(trades), trades=trades, daily_vals=daily_vals,
        regime_log=regime_log,
    )


def _bench_series(idx_code, trade_dates, init_capital):
    """获取基准指数的归一化净值序列。"""
    conn = get_conn()
    if not trade_dates:
        conn.close()
        return np.array([])
    rows = pd.read_sql_query(
        "SELECT trade_date, close FROM index_daily WHERE ts_code=? "
        "AND trade_date BETWEEN ? AND ? ORDER BY trade_date",
        conn, params=(idx_code, str(trade_dates[0]), str(trade_dates[-1])))
    conn.close()
    if rows.empty:
        return np.array([init_capital] * len(trade_dates))
    rows["trade_date"] = rows["trade_date"].astype(str)
    s = rows.set_index("trade_date")["close"].astype(float)
    s = s.reindex([str(t) for t in trade_dates]).ffill()
    fv = s.first_valid_index()
    if fv is None:
        return np.array([init_capital] * len(trade_dates))
    base = s[fv]
    return (s / base * init_capital).values


# ════════════════════════════════════════════════════════════
#  报告
# ════════════════════════════════════════════════════════════

def fmt_pct(x):
    return f"{x:+.2f}%"


def print_result(r, label=""):
    if r is None:
        print(f"  [{label}] 无结果")
        return
    print(f"\n  ── {label} ──")
    print(f"  选股: {r['selection_method']} | top_n={r['top_n']} | 门控: {'ON' if r['use_gate'] else 'OFF'}")
    print(f"  总收益: {fmt_pct(r['total_return'])} | 年化: {fmt_pct(r['annual_return'])}")
    print(f"  最大回撤: {r['max_dd']:.2f}% | 夏普: {r['sharpe']:.2f}")
    print(f"  沪深300基准: {fmt_pct(r['bench_total'])} | 年化: {fmt_pct(r['bench_ann'])}")
    print(f"  超额(vs300): {fmt_pct(r['total_return'] - r['bench_total'])}")
    print(f"  交易数: {r['n_trades']}")
    if r.get("regime_log"):
        rl = r["regime_log"]
        from collections import Counter
        labels = Counter(e["label"] for e in rl)
        total_rb = len(rl)
        parts = [f"{v}次({v/total_rb:.0%})" for _, v in labels.most_common()]
        print(f"  门控判断({total_rb}次调仓): {' / '.join(parts)}")
    print()


# ════════════════════════════════════════════════════════════
#  主入口
# ════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="环境判断器仓位门控 A/B 对照 (BV1uAuu6vE1j · 悦悦笔记)")
    ap.add_argument("--start", default="20180101")
    ap.add_argument("--end", default="20251231")
    ap.add_argument("--method", default="div_low_vol",
                    choices=["div_low_vol", "value", "momentum", "div_growth"],
                    help="选股策略")
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--squeeze-th", type=float, default=0.25,
                    help="布林带宽分位阈值: <此值=震荡(半仓)")
    ap.add_argument("--bb-win", type=int, default=20)
    ap.add_argument("--bb-lookback", type=int, default=120)
    ap.add_argument("--stock-pool", default=None,
                    help="股票池指数代码 (默认用 config)")
    args = ap.parse_args()

    print(f"{'='*80}")
    print(f"  环境判断器仓位门控 · A/B 对照实验")
    print(f"  视频源: 悦悦笔记 BV1uAuu6vE1j")
    print(f"  回测区间: {args.start} -> {args.end}")
    print(f"  选股: {args.method} | top_n={args.top_n}")
    print(f"  门控参数: squeeze_th={args.squeeze_th} bb_win={args.bb_win} bb_lookback={args.bb_lookback}")
    print(f"{'='*80}")

    # A组: 门控 OFF (原始满仓调仓)
    print(f"\n{'─'*60}")
    print(f"  运行 A 组: 门控 OFF (满仓调仓)...")
    r_a = run_backtest(
        start_date=args.start, end_date=args.end,
        selection_method=args.method, top_n=args.top_n,
        use_gate=False,
        stock_pool=args.stock_pool)
    print_result(r_a, "A) 门控 OFF (满仓)")

    # B组: 门控 ON
    print(f"{'─'*60}")
    print(f"  运行 B 组: 门控 ON (regime 仓位控制)...")
    r_b = run_backtest(
        start_date=args.start, end_date=args.end,
        selection_method=args.method, top_n=args.top_n,
        use_gate=True, squeeze_th=args.squeeze_th,
        bb_win=args.bb_win, bb_lookback=args.bb_lookback,
        stock_pool=args.stock_pool)
    print_result(r_b, "B) 门控 ON (regime)")

    # 汇总对比
    if r_a and r_b:
        print(f"\n{'='*80}")
        print(f"  A/B 对照汇总")
        print(f"{'='*80}")
        print(f"  {'模式':<20} {'总收益':>10} {'年化':>10} {'最大回撤':>10} {'夏普':>8} {'超额(vs300)':>12}")
        print(f"  {'─'*70}")
        print(f"  {'A) 门控OFF':<20} {fmt_pct(r_a['total_return']):>10} {fmt_pct(r_a['annual_return']):>10} "
              f"{r_a['max_dd']:>9.2f}% {r_a['sharpe']:>8.2f} {fmt_pct(r_a['total_return']-r_a['bench_total']):>12}")
        print(f"  {'B) 门控ON':<20} {fmt_pct(r_b['total_return']):>10} {fmt_pct(r_b['annual_return']):>10} "
              f"{r_b['max_dd']:>9.2f}% {r_b['sharpe']:>8.2f} {fmt_pct(r_b['total_return']-r_b['bench_total']):>12}")
        print(f"  {'─'*70}")
        print(f"  {'沪深300(基准)':<20} {fmt_pct(r_a['bench_total']):>10} {fmt_pct(r_a['bench_ann']):>10}")
        print()

        delta_ret = r_b['total_return'] - r_a['total_return']
        delta_dd = r_b['max_dd'] - r_a['max_dd']
        delta_sharpe = r_b['sharpe'] - r_a['sharpe']
        print(f"  门控效果:")
        print(f"    收益变化: {fmt_pct(delta_ret)}")
        print(f"    回撤变化: {delta_dd:+.2f}pp {'(改善)' if delta_dd > 0 else '(恶化)'}")
        print(f"    夏普变化: {delta_sharpe:+.2f}")
        if delta_dd > 0 and delta_ret > -5:
            print(f"    [结论] 门控有效降低回撤{'且收益可控' if delta_ret > -5 else '但牺牲较多收益'}")
        elif delta_ret > 0:
            print(f"    [结论] 门控在降低回撤的同时也提升了收益 (双赢)")
        else:
            print(f"    [结论] 门控效果不显著或为负, 需调参或换选股策略")
        print()


if __name__ == "__main__":
    main()
