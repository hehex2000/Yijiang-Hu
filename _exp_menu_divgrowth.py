import sys, io, contextlib
from run_monthly_rebalance import run_backtest

def run(label, top_n, start, end):
    print(f"\n########## {label}  top_n={top_n}  {start}~{end} ##########")
    r = run_backtest(start_date=start, end_date=end, top_n=top_n, selection_method="div_growth")
    print(f"[{label}] total_return={r['total_return']:.2f}%  annual={r['annual_return']:.2f}%  "
          f"maxdd={r['max_drawdown']:.2f}%  sharpe={r['sharpe']:.2f}%  final={r['final_value']:,.0f}")

run("USER_CONFIG", 5, "20140108", "20260807")
run("MY_CONFIG_MENU_ENGINE", 10, "20140108", "20260720")
print("\nDONE")
