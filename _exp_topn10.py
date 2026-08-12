from run_monthly_rebalance import run_backtest
print("########## MENU_ENGINE top_n=10  20140101~20260720 ##########")
run_backtest(start_date="20140101", end_date="20260720", top_n=10, selection_method="div_growth")
print("DONE_TOPN10")
