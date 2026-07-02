"""
百分比网格交易回测 —— 不猜方向，靠波劙收割差价

原理：
  价格每涨  grid_pct  →  卖出一份（锁定利润）
  价格每跌  grid_pct  →  买入一份（降低成本）
  市场波劙本身就是利润来源，不需要预测涨跌。

参考：Rundle et al. (2019) MDPI Applied Sciences
"""

import sys
import os
import sqlite3
import numpy as np
import pandas as pd

# ── 复用现有模块 ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_monthly_rebalance import (
    get_conn, get_price, get_open_price, calc_fee,
    INIT_CAPITAL, INDEX_DISPLAY_NAME,
)

# ── 默认参数 ──
GRID_PCT = 0.02          # 每格百分比 (2%)
PER_GRID_CASH = 5000     # 每格交易金额
INIT_POSITION_PCT = 0.5  # 初始持仓比例（50%建仓）


def generate_grid_levels(base_price, grid_pct, num_levels_up=20, num_levels_down=20):
    """
    生成百分比网格线

    从 base_price 向上下展开：
      向上: base × (1+pct)^1, base × (1+pct)^2, ...
      向下: base × (1-pct)^1, base × (1-pct)^2, ...

    Returns:
        levels_down: 买单价格线（从高到低）
        levels_up:   卖单价格线（从低到高）
    """
    levels_down = []
    for i in range(1, num_levels_down + 1):
        levels_down.append(base_price * (1 - grid_pct) ** i)

    levels_up = []
    for i in range(1, num_levels_up + 1):
        levels_up.append(base_price * (1 + grid_pct) ** i)

    return levels_down, levels_up


def run_grid_backtest(ts_code="000300.SH", start_date="20200102", end_date="20251231",
                      grid_pct=GRID_PCT, per_grid_cash=PER_GRID_CASH,
                      init_position_pct=INIT_POSITION_PCT, initial_capital=INIT_CAPITAL):
    """
    百分比网格交易回测

    Args:
        ts_code:           标的代码（指数或ETF）
        start_date:        回测开始日期 YYYYMMDD
        end_date:          回测结束日期 YYYYMMDD
        grid_pct:          每格百分比 (0.02 = 2%)
        per_grid_cash:     每格交易金额（元）
        init_position_pct: 初始持仓比例
        initial_capital:   初始资金

    Returns:
        dict: 绩效指标
    """
    # ETF/指数映射：ETF代码→模拟指数代理
    ETF_PROXY = {
        "510300.SH": "000300.SH",  # 沪深300ETF→沪深300指数
        "510500.SH": "000905.SH",  # 中证500ETF→中证500指数
        "512100.SH": "000906.SH",  # 中证800ETF→中证800指数
    }
    if ts_code in ETF_PROXY:
        ts_code = ETF_PROXY[ts_code]

    # 判断标的类型（000XXX.SH=指数，其余=个股）
    is_index = (ts_code.endswith(".SH") and ts_code[:3] == "000" and len(ts_code) == 9)
    table = "index_daily" if is_index else "daily"
    price_scale = 0.001 if is_index else 1.0
    lot_size = 1 if is_index else 100

    display = INDEX_DISPLAY_NAME.get(ts_code, ts_code)
    print("=" * 70)
    print(f"百分比网格交易回测")
    print("=" * 70)
    print(f"  标的：{ts_code} ({display})")
    print(f"  网格：每涨跌 {grid_pct*100:.0f}% 触发买卖")
    print(f"  每格金额：{per_grid_cash:,.0f} 元")
    print(f"  初始仓位：{init_position_pct*100:.0f}%")
    print(f"  回测区间：{start_date} ~ {end_date}")
    print(f"  佣金：万2.5（最低5元）| 印花税：千1 | 滑点：0.1%")
    print()

    # ===== 1. 获取历史数据 =====
    conn = get_conn()
    df = pd.read_sql_query(f"""
        SELECT trade_date, open, high, low, close
        FROM {table}
        WHERE ts_code = ? AND trade_date >= ? AND trade_date <= ?
        ORDER BY trade_date
    """, conn, params=(ts_code, start_date, end_date))
    conn.close()

    if len(df) < 2:
        print(f"⚠️ 数据不足（{len(df)}条），无法回测")
        return None

    # 价格缩放（指数÷1000模拟ETF，约4元/份）
    df['scaled_open']  = df['open']  * price_scale
    df['scaled_high']  = df['high']  * price_scale
    df['scaled_low']   = df['low']   * price_scale
    df['scaled_close'] = df['close'] * price_scale

    print(f"  交易日数：{len(df)}")
    print(f"  原始价格范围：{df['close'].min():.2f} ~ {df['close'].max():.2f}")
    print(f"  缩放后价格（模拟ETF）：{df['scaled_close'].min():.2f} ~ {df['scaled_close'].max():.2f}")

    # ===== 2. 生成网格线 =====
    first_scaled_close = float(df.iloc[0]['scaled_close'])
    base_price = first_scaled_close
    levels_down, levels_up = generate_grid_levels(base_price, grid_pct)
    all_levels = sorted(set(levels_down + [base_price] + levels_up))
    all_levels = [round(lv, 4) for lv in all_levels]

    print(f"  基准价：{base_price:.2f}")
    print(f"  网格最低：{levels_down[-1]:.2f}，最高：{levels_up[-1]:.2f}")
    print(f"  共 {len(all_levels)} 条格线\n")

    # ===== 3. 初始化仓位（按金额，股票取整手）=====
    cash = initial_capital * (1 - init_position_pct)
    position_amount = initial_capital * init_position_pct
    first_open = float(df.iloc[0]['scaled_open'])
    raw_units = position_amount / first_open if first_open > 0 else 0
    if lot_size > 1:
        units = int(raw_units / lot_size) * lot_size  # 股票取整手
    else:
        units = raw_units  # 指数取实际份数
    fee = calc_fee('buy', first_open, units) if units > 0 else 0
    cash = initial_capital - units * first_open - fee
    print(f"  建仓：{units:,.0f}份 @ {first_open:.2f}，投入 {units*first_open:,.0f}，现金 {cash:,.0f}")

    # 跟踪每条网格线是否已触发
    grid_bought = {lv: False for lv in levels_down}
    grid_sold   = {lv: False for lv in levels_up}

    daily_vals = []
    trades = []
    prev_close = float(df.iloc[0]['scaled_close'])
    buy_count = 0
    sell_count = 0

    for _, row in df.iterrows():
        td = int(row['trade_date']) if hasattr(row['trade_date'], 'item') else int(row['trade_date'])
        op = row['scaled_open']
        hi = row['scaled_high']
        lo = row['scaled_low']
        cl = row['scaled_close']

        # ── 检查买入触发（价格跌穿买入格线）──
        for lv in reversed(levels_down):
            if grid_bought.get(lv, False):
                continue
            if lo <= lv < prev_close:
                # 按金额买入：per_grid_cash 元等值份数
                buy_units = per_grid_cash / lv if lv > 0 else 0
                if lot_size > 1:
                    buy_units = int(buy_units / lot_size) * lot_size
                if buy_units <= 0:
                    continue
                buy_cost = buy_units * lv
                buy_fee = calc_fee('buy', lv, buy_units)
                if buy_cost + buy_fee > cash or buy_units < lot_size:
                    # 钱不够，尝试买最小整数手
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
                    continue    # 确实买不起了
                cash -= buy_cost + buy_fee
                units += buy_units
                grid_bought[lv] = True
                buy_count += 1
                print(f"  📥 买入 {td} 格线{lv:.2f}：{buy_units:,.0f}份 @ {lv:.2f}，现金{cash:,.0f}")
                trades.append({
                    "date": td, "action": "BUY", "price": lv, "shares": buy_units, "reason": f"grid_{lv:.2f}"
                })

        # ── 检查卖出触发（价格涨破卖出格线）──
        for lv in levels_up:
            if grid_sold.get(lv, False):
                continue
            if lv <= 0:
                continue
            if hi >= lv > prev_close and units > 0:
                # 按金额卖出：per_grid_cash 元等值份数
                sell_units = per_grid_cash / lv
                if lot_size > 1:
                    sell_units = int(sell_units / lot_size) * lot_size
                if sell_units > units:
                    sell_units = int(units / lot_size) * lot_size  # 全卖
                if sell_units <= 0:
                    continue

                proceeds = sell_units * lv
                fee = calc_fee('sell', lv, sell_units)
                cash += proceeds - fee
                units -= sell_units
                grid_sold[lv] = True
                sell_count += 1
                print(f"  📤 卖出 {td} 格线{lv:.2f}：{sell_units:,.0f}份 @ {lv:.2f}，现金{cash:,.0f}")
                trades.append({
                    "date": td, "action": "SELL", "price": lv, "shares": sell_units, "reason": f"grid_{lv:.2f}"
                })

        # ── 状态重置：空仓后重置所有格线标记，等待下一轮循环 ──
        if units <= 0 and cash > 0:
            for lv in grid_sold:
                grid_sold[lv] = False
            for lv in grid_bought:
                grid_bought[lv] = False  # Bug #1修复：买入标记同步重置

        # ── 每日净值记录 ──
        tv = cash + units * cl
        daily_vals.append({"date": td, "value": tv, "units": units, "cash": cash})
        prev_close = cl

    # ===== 4. 平仓结算 =====
    if units > 0:
        last_close = float(df.iloc[-1]['scaled_close'])
        proceeds = units * last_close
        fee = calc_fee('sell', last_close, units)
        cash += proceeds - fee
        print(f"\n  平仓：{units:,.0f}份 @ {last_close:.2f}，现金 {cash:,.0f}")
        trades.append({
            "date": df.iloc[-1]['trade_date'], "action": "SELL",
            "price": last_close, "shares": units, "reason": "backtest_end"
        })

    # ===== 5. 绩效计算 =====
    final_value = cash
    total_return = (final_value / initial_capital - 1) * 100
    days = len(df)
    years = days / 252
    annual_return = ((final_value / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0

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

    # ===== 6. 基准对比 =====
    idx_return = (float(df.iloc[-1]['scaled_close']) / float(df.iloc[0]['scaled_close']) - 1) * 100

    # ===== 7. 输出 =====
    print(f"\n{'=' * 70}")
    print(f"  网格交易回测结果")
    print(f"{'=' * 70}")
    profit_amount = final_value - initial_capital
    print(f"  初始资金：{initial_capital:,.0f}")
    print(f"  最终资产：{final_value:,.0f}")
    print(f"  总盈亏：{profit_amount:+,.0f} 元")
    print(f"  总收益率：{total_return:+.2f}%")
    print(f"  年化收益率：{annual_return:+.2f}%")
    print(f"  最大回撤：{max_dd:.2f}%")
    print(f"  夏普比率：{sharpe:.2f}")
    print(f"  交易次数：{len(trades)}（买{buy_count}次 / 卖{sell_count}次）")
    print(f"  {display}涨幅：{idx_return:+.2f}%")
    print(f"  超额收益：{total_return - idx_return:+.2f}%")

    # 保存
    csv_dir = "data/results/grid_backtest"
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = f"{csv_dir}/grid_{ts_code.replace('.','_')}_{grid_pct*100:.0f}pct_{start_date}_{end_date}.csv"
    pd.DataFrame(daily_vals).to_csv(csv_path, index=False)
    print(f"\n  结果已保存：{csv_path}")

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "trades": len(trades),
        "buy_count": buy_count,
        "sell_count": sell_count,
        "idx_return": idx_return,
        "daily_values": daily_vals,
    }


# ══════════════════════════════════════════
#  入口
# ══════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="百分比网格交易回测")
    parser.add_argument("ts_code", nargs="?", default="000300.SH", help="标的代码")
    parser.add_argument("start_date", nargs="?", default="20200102", help="开始日期 YYYYMMDD")
    parser.add_argument("end_date", nargs="?", default="20251231", help="结束日期 YYYYMMDD")
    parser.add_argument("--grid-pct", type=float, default=GRID_PCT, help="每格百分比（默认0.02=2%%）")
    parser.add_argument("--per-grid", type=int, default=PER_GRID_CASH, help="每格交易金额（默认5000）")
    parser.add_argument("--init-pos", type=float, default=INIT_POSITION_PCT, help="初始持仓比例（默认0.5）")
    parser.add_argument("--capital", type=int, default=INIT_CAPITAL, help="初始资金（默认100000）")
    args = parser.parse_args()

    run_grid_backtest(
        ts_code=args.ts_code,
        start_date=args.start_date,
        end_date=args.end_date,
        grid_pct=args.grid_pct,
        per_grid_cash=args.per_grid,
        init_position_pct=args.init_pos,
        initial_capital=args.capital,
    )
