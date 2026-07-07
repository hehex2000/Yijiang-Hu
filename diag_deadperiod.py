"""诊断：510300 在 2025-2026 为何一年无网格交易。

复用 run_grid_backtest 的数据加载与网格逻辑，仅对 2025 之后的交易日
逐日打印：价格、持仓、pos_min、当日是否触发卖线/买线、被什么挡住。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_grid_backtest import (
    _load_etf_adjusted, get_conn, calc_fee, INIT_CAPITAL, POS_MIN_FRAC, POS_MAX_FRAC,
)
import numpy as np

ts_code = "510300.SH"
GRID_PCT = 0.04
PER_GRID_CASH = 5000
INIT_POSITION_PCT = 0.5
start_date, end_date = "20180102", "20260703"

conn = get_conn()
df, note = _load_etf_adjusted(conn, ts_code, start_date, end_date)
conn.close()
df["trade_date"] = df["trade_date"].astype(int)

# 复刻建仓
base_price = float(df.iloc[0]['close'])
first_open = float(df.iloc[0]['open'])
units = int((INIT_CAPITAL * INIT_POSITION_PCT) / first_open)
cash = INIT_CAPITAL - units * first_open
pos_min = units * POS_MIN_FRAC
pos_max = units * POS_MAX_FRAC

sell_gap = buy_gap = GRID_PCT
buy_lines = sorted([base_price*(1-buy_gap)**k for k in range(1,401)], reverse=True)
sell_lines = sorted([base_price*(1+sell_gap)**k for k in range(1,401)])

print(f"base_price={base_price:.4f} first_open={first_open:.4f} 初始units={units} "
      f"pos_min={pos_min:.0f} pos_max={pos_max:.0f}")
print("sell_lines(前10):", [f"{x:.2f}" for x in sell_lines[:10]])
print("buy_lines(前5):", [f"{x:.2f}" for x in buy_lines[:5]])
print("="*80)

prev_close = float(df.iloc[0]['close'])
# 仅打印 2025-01-01 之后的日子，并在每年首笔打印分隔
started = False
for _, row in df.iterrows():
    td = int(row['trade_date'])
    if td < 20250101:
        prev_close = float(row['close'])
        # 仍要推进 units/cash 以得到 2025 年初的真实持仓
        op, hi, lo, cl = float(row['open']), float(row['high']), float(row['low']), float(row['close'])
        if cl <= prev_close:
            for line in buy_lines:
                if lo <= line < prev_close and units < pos_max:
                    bu = PER_GRID_CASH/line
                    bu = int(bu/100)*100
                    if bu > 0 and bu*line + calc_fee('buy',line,bu) <= cash:
                        cash -= bu*line + calc_fee('buy',line,bu); units += bu
        else:
            if units > pos_min:
                for line in sell_lines:
                    if prev_close < line <= hi and units > pos_min:
                        su = int((PER_GRID_CASH/line)/100)*100
                        if su > 0 and units - su >= pos_min:
                            cash += su*line - calc_fee('sell',line,su); units -= su
        prev_close = cl
        continue

    op, hi, lo, cl = float(row['open']), float(row['high']), float(row['low']), float(row['close'])
    units_before = units
    # 检测卖线穿越
    sell_triggered = [f"{line:.2f}" for line in sell_lines if prev_close < line <= hi and units > pos_min]
    sell_blocked_posmin = [f"{line:.2f}" for line in sell_lines if prev_close < line <= hi and units <= pos_min]
    buy_triggered = [f"{line:.2f}" for line in buy_lines if lo <= line < prev_close and units < pos_max]

    # 推进逻辑
    if cl <= prev_close:
        for line in buy_lines:
            if lo <= line < prev_close and units < pos_max:
                bu = int((PER_GRID_CASH/line)/100)*100
                if bu > 0 and bu*line + calc_fee('buy',line,bu) <= cash:
                    cash -= bu*line + calc_fee('buy',line,bu); units += bu
    else:
        if units > pos_min:
            for line in sell_lines:
                if prev_close < line <= hi and units > pos_min:
                    su = int((PER_GRID_CASH/line)/100)*100
                    if su > 0 and units - su >= pos_min:
                        cash += su*line - calc_fee('sell',line,su); units -= su

    if td % 10000 == 101 or td == 20250102 or (sell_triggered or sell_blocked_posmin or buy_triggered):
        print(f"{td} O{op:.2f} H{hi:.2f} L{lo:.2f} C{cl:.2f} | units_before={units_before:.0f} "
              f"pos_min={pos_min:.0f} | 卖触发={sell_triggered} 卖被posmin挡={sell_blocked_posmin} 买触发={buy_triggered}")

    prev_close = cl

print("="*80)
print(f"最终 units={units:.0f} cash={cash:.0f} 净值={cash+units*float(df.iloc[-1]['close']):.0f}")
