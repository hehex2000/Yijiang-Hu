"""从已保存的权益曲线 CSV 计算并打印「年度盈亏窗口」（复用 run_monthly_rebalance._print_annual_pnl）。"""
import pandas as pd
from run_monthly_rebalance import _print_annual_pnl, INIT_CAPITAL

csv_path = "data/results/monthly_rebalance/backtest_20140108_20260807.csv"
df = pd.read_csv(csv_path)
daily_vals = [{"date": int(r["date"]), "value": float(r["value"])} for _, r in df.iterrows()]
init = daily_vals[0]["value"]  # 实际初始资金（CSV 首个值）
print(f"区间 {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}  初始净值={init:,.2f}  末值={daily_vals[-1]['value']:,.2f}")
print(f"(注: 平台菜单显示总初始资金 200000，但 run_backtest 实际用 INIT_CAPITAL={INIT_CAPITAL:,.0f}；本窗口以 CSV 真实初值 {init:,.0f} 为基准)")
annual_rows = _print_annual_pnl(daily_vals, init, benchmark_idx="000906.SH")
out = "data/results/monthly_rebalance/annual_pnl_20140108_20260807.csv"
pd.DataFrame(annual_rows).to_csv(out, index=False)
print(f"\n年度盈亏明细已保存：{out}")
