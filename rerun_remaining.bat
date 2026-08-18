@echo off
cd /d C:\Users\99395\WorkBuddy\multi_factor_selection

rem ===== [NEW] original ETF rotation (menu B mode 4 baseline) =====
.\venv_ml\Scripts\python.exe run_etf_rotation.py

rem ===== [NEW] kara small cap rotation (stock) =====
.\venv_ml\Scripts\python.exe backtest_kara_small_cap.py

rem ===== [NEW] dividend low vol quality (stock, monthly) =====
.\venv_ml\Scripts\python.exe run_dividend_low_vol_quality_bt.py

rem ===== [NEW] momentum grid timing (stock; menu A equivalent) =====
.\venv_ml\Scripts\python.exe run_momentum_grid_timing.py

rem ===== [NEW] selection sensitivity (stock) =====
.\venv_ml\Scripts\python.exe run_selection_sensitivity.py

rem ===== [RERUN] ab compare was truncated at price stage, rerun for full metrics =====
.\venv_ml\Scripts\python.exe run_selection_ab_compare.py

rem ===== [OPTIONAL] selection annual = control group (locked Jan, unaffected); rerun only for full terminal output =====
.\venv_ml\Scripts\python.exe run_selection_annual.py

rem ===== [ALREADY RERUN] v6 merged (+179.91% verified earlier, skip unless you want to confirm) =====
rem .\venv_ml\Scripts\python.exe run_etf_rotation_v6_merged.py
