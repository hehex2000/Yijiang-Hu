@echo off
REM ============================================================
REM  高股息 + 基本面成长 双因子月度调仓（B站视频策略复刻）
REM  平台月度回测入口：三筛(股息率前10%%+PE/PEG/ROE/营收/净利五关)
REM  + 月调仓 + 涨停跑路日规则；无个股止损
REM
REM  说明：
REM    - 选股池固定全A（视频口径），不受平台股票池配置影响
REM    - 基准/净值统计沿用平台月度引擎（raw 口径）
REM    - 研究版（含 hfq 双轨净值 + 多窗口对比）：
REM        run_dividend_growth_monthly.py --windows ...
REM ============================================================
cd /d C:\Users\99395\WorkBuddy\multi_factor_selection

if exist user_config.bat (
    call user_config.bat
)
if not defined P_BACKTEST_START set P_BACKTEST_START=20140101
if not defined P_BACKTEST_END set P_BACKTEST_END=20260720
if not defined P_TOP_N set P_TOP_N=10
if not defined P_TOTAL_CAPITAL set P_TOTAL_CAPITAL=100000

echo ========================================
echo   高股息+基本面成长 月度调仓回测
echo ========================================
echo   区间: %P_BACKTEST_START% ~ %P_BACKTEST_END%
echo   持仓: %P_TOP_N% 只等权 | 资金: %P_TOTAL_CAPITAL% 元
echo.
echo   策略: 股息率前10%% + PE/PEG/ROE/营收/净利五关
echo         + 月第5交易日调仓 + 涨停跑路日规则
echo.

venv_ml\Scripts\python.exe run_backtest.py --source monthly_rebalance --start-date %P_BACKTEST_START% --end-date %P_BACKTEST_END% --top-n %P_TOP_N% --capital %P_TOTAL_CAPITAL% --selection-method div_growth

echo.
pause
