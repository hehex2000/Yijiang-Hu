@echo off
chcp 65001 >nul
cd /d C:\Users\99395\WorkBuddy\multi_factor_selection

echo.
echo ============================================
echo   多因子选股 + 回测系统
echo ============================================
echo.
echo   请选择回测方案：
echo.
echo   [1] 多因子选股  (--source multi)
echo   [2] 机器学习选股 (--source ml)
echo   [3] 列出所有可用 CSV 文件
echo   [4] 使用 config.py 默认配置
echo.
set /p CHOICE=请选择 (1-4): 
echo.

if "%CHOICE%"=="1" (
    "venv_ml\Scripts\python.exe" run_backtest.py --source multi
) else if "%CHOICE%"=="2" (
    "venv_ml\Scripts\python.exe" run_backtest.py --source ml
) else if "%CHOICE%"=="3" (
    "venv_ml\Scripts\python.exe" run_backtest.py --list
    echo.
    pause
    goto :eof
) else if "%CHOICE%"=="4" (
    "venv_ml\Scripts\python.exe" run_backtest.py
) else (
    echo 无效选择，使用默认配置...
    "venv_ml\Scripts\python.exe" run_backtest.py
)

pause
