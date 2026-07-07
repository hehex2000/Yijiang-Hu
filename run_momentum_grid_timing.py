"""
方案①：网格作为市场择时钟 —— 动量策略 + 网格持仓择时

核心逻辑：
  网格持仓 < 20%  → 市场偏贵 → 动量减仓至 3 只
  网格持仓 20-80% → 正常市场 → 保持 5 只
  网格持仓 > 80%  → 市场便宜 → 动量增加到 8 只（趁机低吸）

设计原则：
  - 不修改现有策略代码（run_monthly_rebalance.py / run_grid_backtest.py）
  - 独立实现，导入复用现有模块
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_monthly_rebalance import (
    get_conn, get_price, get_open_price, calc_fee,
    get_trade_dates, get_monthly_5th_trading_days,
    get_stock_pool_index, get_stock_name,
    select_momentum_stocks, is_above_ma,
    INIT_CAPITAL, INDEX_DISPLAY_NAME, calc_win_rate,
)
from run_grid_backtest import generate_grid_levels

# ── 默认参数 ──
GRID_PCT = 0.02          # 网格间距 2%
PER_GRID_CASH = 5000     # 每格交易金额
INIT_POSITION_PCT = 0.5  # 网格初始仓位 50%
DEFAULT_GRID_INDEX = "000300.SH"  # 网格参考指数


def get_grid_position_series(ts_code="000300.SH", start_date="20200102", end_date="20251231",
                              grid_pct=GRID_PCT, per_grid_cash=PER_GRID_CASH,
                              init_position_pct=INIT_POSITION_PCT, initial_capital=INIT_CAPITAL):
    """
    运行网格模拟，返回每日持仓比例 {trade_date: position_pct}

    静默运行（不打印），仅输出数据。

    position_pct = 持仓市值 / 总资产
    范围: 0.0 ~ 1.0
    0.0 = 空仓（市场最贵）
    1.0 = 满仓（市场最便宜）

    Returns:
        dict: {int(trade_date): float(position_pct)}
    """
    # ETF/指数判断
    is_index = (ts_code.endswith(".SH") and ts_code[:3] == "000" and len(ts_code) == 9)
    table = "index_daily" if is_index else "daily"
    price_scale = 0.001 if is_index else 1.0
    lot_size = 1 if is_index else 100

    conn = get_conn()
    df = pd.read_sql_query(f"""
        SELECT trade_date, open, high, low, close
        FROM {table}
        WHERE ts_code = ? AND trade_date >= ? AND trade_date <= ?
        ORDER BY trade_date
    """, conn, params=(ts_code, start_date, end_date))
    conn.close()

    if len(df) < 2:
        return {}

    # 价格缩放
    df['scaled_open'] = df['open'] * price_scale
    df['scaled_high'] = df['high'] * price_scale
    df['scaled_low'] = df['low'] * price_scale
    df['scaled_close'] = df['close'] * price_scale

    # 生成网格线
    first_scaled_close = float(df.iloc[0]['scaled_close'])
    levels_down, levels_up = generate_grid_levels(first_scaled_close, grid_pct)

    # 初始化仓位
    cash = initial_capital * (1 - init_position_pct)
    position_amount = initial_capital * init_position_pct
    first_open = float(df.iloc[0]['scaled_open'])
    raw_units = position_amount / first_open if first_open > 0 else 0
    if lot_size > 1:
        units = int(raw_units / lot_size) * lot_size
    else:
        units = raw_units
    fee = calc_fee('buy', first_open, units) if units > 0 else 0
    cash = initial_capital - units * first_open - fee

    # 网格标记
    grid_bought = {lv: False for lv in levels_down}
    grid_sold   = {lv: False for lv in levels_up}

    position_series = {}
    prev_close = float(df.iloc[0]['scaled_close'])

    for _, row in df.iterrows():
        td = int(row['trade_date'])
        op = row['scaled_open']
        hi = row['scaled_high']
        lo = row['scaled_low']
        cl = row['scaled_close']

        # 买入触发
        for lv in reversed(levels_down):
            if grid_bought.get(lv, False):
                continue
            if lo <= lv < prev_close:
                buy_units = per_grid_cash / lv if lv > 0 else 0
                if lot_size > 1:
                    buy_units = int(buy_units / lot_size) * lot_size
                if buy_units <= 0:
                    continue
                buy_cost = buy_units * lv
                buy_fee = calc_fee('buy', lv, buy_units)
                if buy_cost + buy_fee > cash:
                    if lot_size > 0:
                        min_units = int(cash / (lv * lot_size + calc_fee('buy', lv, lot_size))) * lot_size
                    else:
                        min_units = 0
                    if min_units >= lot_size:
                        buy_units = min_units
                        buy_cost = buy_units * lv
                        buy_fee = calc_fee('buy', lv, buy_units)
                    else:
                        continue
                if buy_cost + buy_fee > cash:
                    continue
                cash -= buy_cost + buy_fee
                units += buy_units
                grid_bought[lv] = True

        # 卖出触发
        for lv in levels_up:
            if grid_sold.get(lv, False):
                continue
            if lv <= 0:
                continue
            if hi >= lv > prev_close and units > 0:
                sell_units = per_grid_cash / lv
                if lot_size > 1:
                    sell_units = int(sell_units / lot_size) * lot_size
                if sell_units > units:
                    sell_units = int(units / lot_size) * lot_size
                if sell_units <= 0:
                    continue
                proceeds = sell_units * lv
                fee = calc_fee('sell', lv, sell_units)
                cash += proceeds - fee
                units -= sell_units
                grid_sold[lv] = True

        # 状态重置：空仓后重置所有格线标记
        if units <= 0 and cash > 0:
            for lv in grid_sold:
                grid_sold[lv] = False
            for lv in grid_bought:
                grid_bought[lv] = False

        # 计算持仓比例
        tv = cash + units * cl
        pos_pct = (tv - cash) / tv if tv > 0 else 0
        position_series[td] = round(pos_pct, 4)
        prev_close = cl

    return position_series


def get_dynamic_top_n(position_pct, default_top_n=5):
    """
    根据网格持仓比例确定动量策略选股数量

    Args:
        position_pct: 网格持仓比例 (0~1)
        default_top_n: 默认选股数量

    Returns:
        tuple: (top_n, regime_label)
    """
    if position_pct < 0.2:
        top_n = max(3, default_top_n - 2)
        regime = "🔴 市场偏贵"
    elif position_pct > 0.8:
        top_n = min(10, default_top_n + 3)
        regime = "🟢 市场便宜"
    else:
        top_n = default_top_n
        regime = "🟡 正常市场"

    return top_n, regime


def run_momentum_with_grid_timing(start_date="20200101", end_date="20251231",
                                   default_top_n=5, lookback_months=6, stock_pool=None,
                                   rebalance_freq_months=1, atr_stop_multiple=0,
                                   atr_cooling_days=0, trailing_stop_pct=0,
                                   skip_recent_months=1, trend_filter_ma=0,
                                   grid_pct=GRID_PCT, per_grid_cash=PER_GRID_CASH,
                                   grid_index=DEFAULT_GRID_INDEX):
    """
    动量策略 + 网格持仓择时回测

    流程：
    1. 对指定指数的网格策略进行模拟，得到每日持仓比例
    2. 在动量回测循环中，每次调仓前查询当日网格持仓比例
    3. 根据持仓比例动态调整选股数量（3/5/8只）
    """
    if stock_pool is None:
        pool_display = "全A股"
    else:
        pool_display = INDEX_DISPLAY_NAME.get(stock_pool, stock_pool)

    freq_label = f"每{rebalance_freq_months}个月" if rebalance_freq_months > 1 else "每月"

    # ── 打印策略概要 ──
    grid_display = INDEX_DISPLAY_NAME.get(grid_index, grid_index)
    print("=" * 70)
    print("方案①：网格作为市场择时钟 —— 动量 + 网格持仓择时")
    print("=" * 70)
    print(f"  股票池：{pool_display}")
    print(f"  基准选股：{default_top_n}只（网格择时动态调为 3/5/8只）")
    print(f"  网格参考：{grid_index} ({grid_display})")
    print(f"  网格间距：{grid_pct*100:.0f}% | 每格 {per_grid_cash:,}元")
    print(f"  动量回看：{lookback_months}个月")
    print(f"  调仓频率：{freq_label}调仓")
    if trend_filter_ma > 0:
        print(f"  市场过滤：指数<{trend_filter_ma}日MA时空仓")
    if atr_stop_multiple > 0:
        print(f"  ATR止损：{atr_stop_multiple}倍")
    elif trailing_stop_pct > 0:
        print(f"  固定止损：最高价回撤{trailing_stop_pct:.0%}")
    print(f"  回测区间：{start_date} ~ {end_date}")
    print(f"  佣金：万2.5（最低5元）| 印花税：千1 | 滑点：0.1%")
    print()

    # ── 第1步：获取网格每日持仓比例 ──
    print("📡 正在运行网格模拟，获取市场温度数据...")
    grid_positions = get_grid_position_series(
        ts_code=grid_index,
        start_date=start_date,
        end_date=end_date,
        grid_pct=grid_pct,
        per_grid_cash=per_grid_cash,
    )
    if not grid_positions:
        print("⚠️  网格模拟数据为空，无法进行择时，使用默认选股数量")
        grid_positions = {}

    print(f"  网格交易日数：{len(grid_positions)}")
    if grid_positions:
        dates_sorted = sorted(grid_positions.keys())
        pos_vals = [grid_positions[d] for d in dates_sorted]
        print(f"  持仓比例范围：{min(pos_vals):.1%} ~ {max(pos_vals):.1%}")
        print(f"  平均持仓比例：{np.mean(pos_vals):.1%}")

        # 统计各区间天数
        cheap_days = sum(1 for v in pos_vals if v > 0.8)
        normal_days = sum(1 for v in pos_vals if 0.2 <= v <= 0.8)
        expensive_days = sum(1 for v in pos_vals if v < 0.2)
        print(f"  市场便宜(>80%)：{cheap_days}天 | 正常(20-80%)：{normal_days}天 | 偏贵(<20%)：{expensive_days}天")
    print()

    # ── 第2步：动量回测循环（动态top_n）──
    trade_dates = get_trade_dates(start_date, end_date)
    monthly_rebalance = get_monthly_5th_trading_days(trade_dates)
    rebalance_set = set(list(monthly_rebalance)[::rebalance_freq_months])
    print(f"交易日总数：{len(trade_dates)}，调仓日：{len(rebalance_set)}次\n")

    # 初始化
    positions = {}
    cash = INIT_CAPITAL
    daily_vals = []
    trades = []
    stop_count = 0
    name_cache = {}
    top_n_history = []  # 记录每次调仓的top_n

    def get_name(code):
        if code not in name_cache:
            name_cache[code] = get_stock_name(code)
        return name_cache[code]

    for i, td in enumerate(trade_dates):
        # ===== 止损卖出 =====
        use_stop = atr_stop_multiple > 0 or trailing_stop_pct > 0
        if use_stop and positions:
            for code in list(positions.keys()):
                pos = positions[code]
                buy_idx = pos.get("buy_idx", 0)
                holding_days = i - buy_idx
                if atr_cooling_days > 0 and holding_days < atr_cooling_days:
                    continue

                if pos.get("stop_triggered", False):
                    open_price = get_open_price(code, td)
                    if open_price is not None:
                        proceeds = pos["shares"] * open_price
                        fee = calc_fee('sell', open_price, pos["shares"])
                        cash += proceeds - fee
                        stop_count += 1
                        print(f"  🔴 止损卖出 {code}({get_name(code)})：{pos['shares']}股 @ {open_price:.2f}")
                        trades.append({
                            "date": td, "action": "SELL", "code": code, "name": get_name(code),
                            "price": open_price, "shares": pos["shares"], "reason": "stop_loss"
                        })
                        del positions[code]
                    continue

                close_price = get_price(code, td)
                if close_price is None:
                    continue

                if close_price > pos.get("highest_close", 0):
                    pos["highest_close"] = close_price

                if atr_stop_multiple > 0:
                    from run_monthly_rebalance import get_atr
                    atr = get_atr(code, td, period=14)
                    if atr is None or atr <= 0:
                        continue
                    stop_price = pos["highest_close"] - atr_stop_multiple * atr
                elif trailing_stop_pct > 0:
                    stop_price = pos["highest_close"] * (1 - trailing_stop_pct)
                else:
                    continue

                if close_price < stop_price:
                    positions[code]["stop_triggered"] = True
                    mode = "ATR" if atr_stop_multiple > 0 else "固定比例"
                    print(f"  ⚠️ {mode}止损触发 {code}({get_name(code)})：收盘{close_price:.2f} < 止损{stop_price:.2f}（持有{holding_days}日）")

        # ===== 调仓日 =====
        if td in rebalance_set:
            # 市场趋势过滤
            benchmark_idx = stock_pool if stock_pool else "000906.SH"
            market_ok = True
            if trend_filter_ma > 0:
                market_ok = is_above_ma(benchmark_idx, td, period=trend_filter_ma, is_index=True)
                if not market_ok:
                    print(f"\n  ⏸️ {td} 指数<{trend_filter_ma}日MA，空仓等待")

            prev_td = trade_dates[i - 1] if i > 0 else td

            # 熊市卖出
            if not market_ok and positions:
                for code in list(positions.keys()):
                    open_price = get_open_price(code, td)
                    if open_price is None:
                        continue
                    pos = positions[code]
                    proceeds = pos["shares"] * open_price
                    fee = calc_fee('sell', open_price, pos["shares"])
                    cash += proceeds - fee
                    print(f"  ✅ 卖出 {code}({get_name(code)})：{pos['shares']}股 @ {open_price:.2f}")
                    trades.append({
                        "date": td, "action": "SELL", "code": code, "name": get_name(code),
                        "price": open_price, "shares": pos["shares"], "reason": "trend_filter_sell"
                    })
                    del positions[code]
                print(f"  💤 空仓等待中证800重回{trend_filter_ma}日MA上方")
                continue

            # === 关键：根据网格持仓比例动态确定选股数量 ===
            grid_pos = grid_positions.get(int(td), 0.5)  # 默认0.5（正常），int(td)因调仓日为str类型
            dynamic_top_n, regime = get_dynamic_top_n(grid_pos, default_top_n)
            top_n_history.append({"date": td, "grid_pos": grid_pos, "top_n": dynamic_top_n, "regime": regime})

            print(f"\n调仓日 {td}：网格持仓 {grid_pos:.1%} → {regime} → 选股 {dynamic_top_n} 只")

            # 选股
            stocks = select_momentum_stocks(
                prev_td,
                lookback_months=lookback_months,
                top_n=dynamic_top_n,
                index_code=stock_pool,
                skip_recent_months=skip_recent_months,
            )
            new_codes = stocks['ts_code'].tolist() if not stocks.empty else []

            if not new_codes:
                print(f"  ⚠️ 选股为空，保持现有仓位")
            else:
                current_codes = set(positions.keys())
                new_set = set(new_codes)

                if current_codes == new_set:
                    print(f"  持仓不变")
                else:
                    print(f"  新选：{[f'{c}({get_name(c)})' for c in new_codes]}")

                    # 卖出旧仓
                    for code in list(positions.keys()):
                        if code not in new_set:
                            open_price = get_open_price(code, td)
                            if open_price is None:
                                continue
                            pos = positions[code]
                            proceeds = pos["shares"] * open_price
                            fee = calc_fee('sell', open_price, pos["shares"])
                            cash += proceeds - fee
                            print(f"  ✅ 卖出 {code}({get_name(code)})：{pos['shares']}股 @ {open_price:.2f}")
                            trades.append({
                                "date": td, "action": "SELL", "code": code, "name": get_name(code),
                                "price": open_price, "shares": pos["shares"], "reason": "dynamic_rebalance"
                            })
                            del positions[code]

                    # 买入新仓
                    new_to_buy = [c for c in new_codes if c not in positions]
                    if new_to_buy:
                        cash_per_stock = cash / len(new_to_buy)
                        for ts_code in new_to_buy:
                            open_price = get_open_price(ts_code, td)
                            if open_price is None:
                                continue
                            max_shares = int(cash_per_stock / open_price / 100) * 100
                            if max_shares < 100:
                                continue
                            cost = max_shares * open_price
                            fee = calc_fee('buy', open_price, max_shares)
                            if cost + fee <= cash:
                                cash -= cost + fee
                                positions[ts_code] = {
                                    "shares": max_shares,
                                    "buy_price": open_price,
                                    "buy_idx": i,
                                    "highest_close": open_price,
                                    "stop_triggered": False,
                                }
                                print(f"  ✅ 买入 {ts_code}({get_name(ts_code)})：{max_shares}股 @ {open_price:.2f}")
                                trades.append({
                                    "date": td, "action": "BUY", "code": ts_code, "name": get_name(ts_code),
                                    "price": open_price, "shares": max_shares, "reason": "dynamic_rebalance"
                                })

        # ===== 每日市值记录 =====
        total_value = cash
        for code, pos in list(positions.items()):
            price = get_price(code, td)
            if price is not None:
                total_value += pos["shares"] * price
        daily_vals.append({"date": td, "value": total_value})

    # === 平仓结算 ===
    if trade_dates:
        last_date = trade_dates[-1]
        for code in list(positions.keys()):
            price = get_price(code, last_date)
            if price is not None:
                pos = positions[code]
                proceeds = pos["shares"] * price
                fee = calc_fee('sell', price, pos["shares"])
                cash += proceeds - fee
                trades.append({
                    "date": last_date, "action": "SELL", "code": code, "name": get_name(code),
                    "price": price, "shares": pos["shares"], "reason": "backtest_end"
                })
                del positions[code]

    # === 绩效计算 ===
    final_value = cash
    total_return = (final_value / INIT_CAPITAL - 1) * 100
    days = len(trade_dates)
    years = days / 252
    annual_return = ((final_value / INIT_CAPITAL) ** (1 / years) - 1) * 100 if years > 0 else 0

    vals = np.array([d["value"] for d in daily_vals])
    cummax = np.maximum.accumulate(vals)
    safe_cummax = np.where(cummax == 0, 1, np.array(cummax, dtype=float))
    drawdowns = (vals - cummax) / safe_cummax
    max_dd = float(np.min(drawdowns)) * 100

    rets = np.diff(vals) / np.where(vals[:-1] == 0, 1, vals[:-1])
    if len(rets) > 1 and np.std(rets) > 0:
        sharpe = (np.mean(rets) * 252 - 0.025) / (np.std(rets) * np.sqrt(252))
    else:
        sharpe = 0.0

    # === 基准指数 ===
    benchmark_idx = stock_pool if stock_pool else "000906.SH"
    conn = get_conn()
    b_start = pd.read_sql_query(
        "SELECT close FROM index_daily WHERE ts_code = ? AND trade_date >= ? ORDER BY trade_date ASC LIMIT 1",
        conn, params=(benchmark_idx, trade_dates[0])
    )
    b_end = pd.read_sql_query(
        "SELECT close FROM index_daily WHERE ts_code = ? AND trade_date <= ? ORDER BY trade_date DESC LIMIT 1",
        conn, params=(benchmark_idx, trade_dates[-1])
    )
    conn.close()

    idx_return = 0
    if len(b_start) > 0 and len(b_end) > 0:
        idx_start = float(b_start.iloc[0]['close'])
        idx_end = float(b_end.iloc[0]['close'])
        idx_return = (idx_end / idx_start - 1) * 100

    # === 择时统计 ===
    grid_display_name = INDEX_DISPLAY_NAME.get(grid_index, grid_index)
    if top_n_history:
        top3_count = sum(1 for h in top_n_history if h["top_n"] == 3)
        top5_count = sum(1 for h in top_n_history if h["top_n"] == 5)
        top8_count = sum(1 for h in top_n_history if h["top_n"] == 8)
        print(f"\n{'=' * 70}")
        print(f"  网格择时统计（{grid_display_name}）")
        print(f"{'=' * 70}")
        print(f"  调仓总次数：{len(top_n_history)}")
        print(f"  选3只（市场偏贵）：{top3_count}次")
        print(f"  选5只（正常市场）：{top5_count}次")
        print(f"  选8只（市场便宜）：{top8_count}次")
        print(f"\n  调仓明细：")
        for h in top_n_history:
            print(f"    {h['date']}  网格持仓 {h['grid_pos']:.1%}  {h['regime']}  选{h['top_n']}只")

    # === 输出 ===
    print(f"\n{'=' * 70}")
    print(f"  动量{lookback_months}个月 × {freq_label}调仓 + 网格择时 回测结果")
    print(f"{'=' * 70}")
    profit_amount = final_value - INIT_CAPITAL
    print(f"  初始资金：{INIT_CAPITAL:,.2f}")
    print(f"  最终资产：{final_value:,.2f}")
    print(f"  总盈亏：{profit_amount:+,.2f} 元")
    print(f"  总收益率：{total_return:+.2f}%")
    print(f"  年化收益率：{annual_return:+.2f}%")
    print(f"  最大回撤：{max_dd:.2f}%")
    print(f"  夏普比率：{sharpe:.2f}")
    print(f"  交易次数：{len(trades)}")
    win_rate, win_cnt, tot_cnt = calc_win_rate(trades)
    if tot_cnt > 0:
        print(f"  胜率：{win_rate:.1f}%（{win_cnt}/{tot_cnt}）")
    if atr_stop_multiple > 0 or trailing_stop_pct > 0:
        print(f"  止损次数：{stop_count}")
    print(f"  基准涨幅：{idx_return:+.2f}%")
    print(f"  超额收益：{total_return - idx_return:+.2f}%")

    # 保存结果
    csv_dir = "data/results/momentum_grid_timing"
    os.makedirs(csv_dir, exist_ok=True)
    freq_suffix = f"_{rebalance_freq_months}m_rebal"
    if atr_stop_multiple > 0:
        stop_suffix = f"_atr{atr_stop_multiple}"
    elif trailing_stop_pct > 0:
        stop_suffix = f"_trail{int(trailing_stop_pct*100)}"
    else:
        stop_suffix = ""
    csv_path = f"{csv_dir}/momentum_grid_{lookback_months}m{freq_suffix}{stop_suffix}_{start_date}_{end_date}.csv"
    pd.DataFrame(daily_vals).to_csv(csv_path, index=False)
    print(f"\n  结果已保存：{csv_path}")

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "trades": len(trades),
        "idx_return": idx_return,
        "stop_count": stop_count,
        "daily_values": daily_vals,
        "top_n_history": top_n_history,
    }


# ══════════════════════════════════════════
#  入口
# ══════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="方案①：网格作为市场择时钟")
    parser.add_argument("start_date", nargs="?", default="20200101", help="开始日期 YYYYMMDD")
    parser.add_argument("end_date", nargs="?", default="20251231", help="结束日期 YYYYMMDD")
    parser.add_argument("--top-n", type=int, default=5, help="默认选股数量（默认5）")
    parser.add_argument("--lookback", type=int, default=12, help="动量回看月数（默认12）")
    parser.add_argument("--stock-pool", type=str, default=None, help="股票池代码（如000300.SH）")
    parser.add_argument("--rebalance-freq", type=int, default=3, help="调仓频率月数（默认3=季度）")
    parser.add_argument("--atr-stop", type=float, default=2.0, help="ATR止损倍数（默认2.0）")
    parser.add_argument("--trailing-stop", type=float, default=0, help="固定比例trailing stop")
    parser.add_argument("--grid-pct", type=float, default=GRID_PCT, help="网格间距（默认0.02=2%%）")
    parser.add_argument("--per-grid", type=int, default=PER_GRID_CASH, help="每格金额（默认5000）")
    parser.add_argument("--grid-index", type=str, default=DEFAULT_GRID_INDEX, help="网格参考指数（默认000300.SH）")
    parser.add_argument("--trend-filter", type=int, default=200, help="MA趋势过滤（默认200）")
    args = parser.parse_args()

    run_momentum_with_grid_timing(
        start_date=args.start_date,
        end_date=args.end_date,
        default_top_n=args.top_n,
        lookback_months=args.lookback,
        stock_pool=args.stock_pool,
        rebalance_freq_months=args.rebalance_freq,
        atr_stop_multiple=args.atr_stop,
        trailing_stop_pct=args.trailing_stop,
        trend_filter_ma=args.trend_filter,
        grid_pct=args.grid_pct,
        per_grid_cash=args.per_grid,
        grid_index=args.grid_index,
    )
