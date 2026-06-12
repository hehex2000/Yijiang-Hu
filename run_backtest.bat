@echo off
chcp 65001 >nul
cd /d C:\Users\99395\WorkBuddy\multi_factor_selection

REM ── 有命令行参数 → 直接透传 ──
if not "%*"=="" (
    "venv_ml\Scripts\python.exe" run_backtest.py %*
    goto :eof
)

:menu
cls
echo.
echo ============================================
echo   Multi-Factor Stock Selection + Backtest
echo ============================================
echo.
echo   Select mode:
echo.
echo   [1] Multi-factor selection + backtest (--source multi)
echo   [2] ML selection + backtest (--source ml)
echo   [3] List all CSV files
echo   [4] Use config.py default
echo   [5] Backtest only (use latest selection result)
echo   [6] Select only (no backtest)
echo   [7] Quit
echo.
set /p CHOICE=Select (1-7): 
echo.

if "%CHOICE%"=="1" (
    "venv_ml\Scripts\python.exe" run_backtest.py --source multi
    echo.
    pause
    goto :menu
) else if "%CHOICE%"=="2" (
    "venv_ml\Scripts\python.exe" run_backtest.py --source ml
    echo.
    pause
    goto :menu
) else if "%CHOICE%"=="3" (
    "venv_ml\Scripts\python.exe" run_backtest.py --list
    echo.
    pause
    goto :menu
) else if "%CHOICE%"=="4" (
    "venv_ml\Scripts\python.exe" run_backtest.py
    echo.
    pause
    goto :menu
) else if "%CHOICE%"=="5" (
    "venv_ml\Scripts\python.exe" run_backtest.py --source csv --auto
    echo.
    pause
    goto :menu
) else if "%CHOICE%"=="6" (
    goto :select_only
) else if "%CHOICE%"=="7" (
    goto :eof
) else (
    echo Invalid choice, please select 1-7.
    pause
    goto :menu
)

:select_only
cls
echo.
echo   Select only mode - choose method:
echo.
echo   [1] Multi-factor selection
echo   [2] Machine Learning selection
echo   [3] Back to main menu
echo.
set /p SUBCHOICE=Select (1-3): 
echo.

if "%SUBCHOICE%"=="1" (
    "venv_ml\Scripts\python.exe" run_backtest.py --source multi --select-only
    echo.
    pause
    goto :menu
) else if "%SUBCHOICE%"=="2" (
    "venv_ml\Scripts\python.exe" run_backtest.py --source ml --select-only
    echo.
    pause
    goto :menu
) else if "%SUBCHOICE%"=="3" (
    goto :menu
) else (
    echo Invalid choice.
    pause
    goto :select_only
)

:eof
