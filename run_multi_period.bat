@echo off
chcp 65001 >nul
cd /d C:\Users\99395\WorkBuddy\multi_factor_selection
"C:\Users\99395\.workbuddy\binaries\python\versions\3.13.12\python.exe" run_multi_period.py
echo.
echo ==============================
echo Multi-period backtest finished!
echo Check results above for each year
echo ==============================
pause
