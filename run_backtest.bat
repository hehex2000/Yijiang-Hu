@echo off
cd /d C:\Users\99395\WorkBuddy\multi_factor_selection

REM ================================
REM  加载持久化配置（user_config.bat）
REM ================================
if exist user_config.bat (
    call user_config.bat
)

REM 如果持久化配置未定义，使用默认值
if not defined P_BACKTEST_START set P_BACKTEST_START=20200102
if not defined P_BACKTEST_END set P_BACKTEST_END=20211231
if not defined P_TOP_N set P_TOP_N=5
if not defined P_SELECTION_METHOD set P_SELECTION_METHOD=multi
if not defined P_STOCK_POOL set P_STOCK_POOL=all

REM 应用到会话变量
set BACKTEST_START=%P_BACKTEST_START%
set BACKTEST_END=%P_BACKTEST_END%
set TOP_N=%P_TOP_N%
set SELECTION_METHOD=%P_SELECTION_METHOD%

REM ================================
REM  保存持久化配置到 user_config.bat
REM  （子程序，用 call :save_config 调用）
REM ================================
goto :menu_start

:save_config
(
echo set P_BACKTEST_START=%BACKTEST_START%
echo set P_BACKTEST_END=%BACKTEST_END%
echo set P_TOP_N=%TOP_N%
echo set P_SELECTION_METHOD=%SELECTION_METHOD%
) > user_config.bat
goto :eof

REM ================================
REM  多因子选股 + 回测系统 - 交互式菜单
REM ================================
:menu_start

:menu
cls
echo.
echo ================
echo   多因子选股 + 回测系统
echo ================
echo.
echo   当前配置:
echo   ----------------
echo   回测区间: %BACKTEST_START% ~ %BACKTEST_END%
echo   选股数量: %TOP_N% 只
echo   选股策略: %SELECTION_METHOD%
echo   股票池: %STOCK_POOL%
echo.
echo   ================
echo   主菜单:
echo   ================
echo.
echo   [1] 运行选股 + 回测（使用当前配置）
echo   [2] 设置回测区间
echo   [3] 设置选股数量
echo   [4] 选择选股策略
echo   [5] 设置股票池
echo   [6] 仅选股（不回测）
echo   [7] 仅回测（使用最新选股结果）
echo   [8] 月度调仓回测
echo   [9] 退出
echo.
set /p CHOICE=请选择 (1-9):
echo.

if "%CHOICE%"=="1" goto :run_all
if "%CHOICE%"=="2" goto :set_date
if "%CHOICE%"=="3" goto :set_top_n
if "%CHOICE%"=="4" goto :set_method
if "%CHOICE%"=="5" goto :set_stock_pool
if "%CHOICE%"=="6" goto :select_only
if "%CHOICE%"=="7" goto :backtest_only
if "%CHOICE%"=="8" goto :monthly_rebalance
if "%CHOICE%"=="9" goto :eof
goto :menu

:set_date
cls
echo.
echo ================
echo   设置回测区间
echo ================
echo.
echo   当前: %BACKTEST_START% ~ %BACKTEST_END%
echo.
set /p BACKTEST_START=  回测开始日期 (YYYYMMDD, 如 20260102):
set /p BACKTEST_END=  回测结束日期 (YYYYMMDD, 如 20260618):
echo.
echo   [OK] 已更新: %BACKTEST_START% ~ %BACKTEST_END%
echo.
REM 同步写入 config.py GLOBAL 配置
venv_ml\Scripts\python.exe update_dates.py %BACKTEST_START% %BACKTEST_END%
if errorlevel 1 (
    echo   [错误] 更新 config.py 失败！
    pause
)
call :save_config
goto :menu

:set_top_n
cls
echo.
echo ================
echo   设置选股数量
echo ================
echo.
echo   当前: %TOP_N% 只
echo.
set /p TOP_N=  选股数量 (如 5, 10, 20):
echo.
echo   [OK] 已更新: %TOP_N% 只
echo.
REM 同步写入 config.py（使用专用脚本，确保正确性）
venv_ml\Scripts\python.exe update_top_n.py %TOP_N%
if errorlevel 1 (
    echo   [错误] 更新 config.py 失败！
    pause
)
call :save_config
goto :menu

:set_method
cls
echo.
echo ================
echo   选择选股策略
echo ================
echo.
echo   [1] 多因子选股（技术+基本面因子）
echo   [2] 价值投资选股（价值投资量化策略）
echo   [3] 红利低波选股（高股息率+低波动率）
echo.
set /p METHOD_CHOICE=请选择 (1-3):
echo.

if "%METHOD_CHOICE%"=="1" (
    set SELECTION_METHOD=multi
    echo   [OK] 已选择: 多因子选股
    call :save_config
    goto :menu
)
if "%METHOD_CHOICE%"=="2" (
    set SELECTION_METHOD=value
    echo   [OK] 已选择: 价值投资选股
    call :save_config
    goto :menu
)
if "%METHOD_CHOICE%"=="3" (
    set SELECTION_METHOD=div_low_vol
    echo   [OK] 已选择: 红利低波选股
    call :save_config
    goto :menu
)
goto :set_method

:run_all
cls
echo.
echo ================
echo   运行选股 + 回测
echo ================
echo.
echo   配置:
echo     回测区间: %BACKTEST_START% ~ %BACKTEST_END%
echo     选股数量: %TOP_N% 只
echo     选股策略: %SELECTION_METHOD%
echo.
"venv_ml\Scripts\python.exe" run_backtest.py --source %SELECTION_METHOD% --start-date %BACKTEST_START% --end-date %BACKTEST_END% --top-n %TOP_N%
echo.
pause
goto :menu

:select_only
cls
echo.
echo ================
echo   仅选股（不回测）
echo ================
echo.
echo   配置:
echo     选股策略: %SELECTION_METHOD%
echo     选股数量: %TOP_N% 只
echo     股票池: %STOCK_POOL%
echo.
"venv_ml\Scripts\python.exe" run_backtest.py --source %SELECTION_METHOD% --select-only --top-n %TOP_N% --stock-pool %STOCK_POOL%
echo.
pause
goto :menu

:backtest_only
cls
echo.
echo ================
echo   仅回测（使用最新选股结果）
echo ================
echo.
echo   配置:
echo     回测区间: %BACKTEST_START% ~ %BACKTEST_END%
echo.
"venv_ml\Scripts\python.exe" run_backtest.py --source csv --auto --start-date %BACKTEST_START% --end-date %BACKTEST_END%
echo.
pause
goto :menu

:monthly_rebalance
cls
echo.
echo ================
echo   月度调仓回测
echo ================
echo.
echo   请选择调仓时的选股策略:
echo   ----------------
echo   [1] 价值选股（PB破净+ROE质量）
echo   [2] 红利低波选股（高股息+低波动）
echo.
set /p MR_METHOD=请选择 (1-2, 默认1):

if "%MR_METHOD%"=="2" (
    set MR_ARG=--selection-method div_low_vol
    echo.
    echo   已选择: 红利低波选股
) else (
    set MR_ARG=--selection-method value
    echo.
    echo   已选择: 价值选股
)
echo.
echo   配置:
echo     回测区间: %BACKTEST_START% ~ %BACKTEST_END%
echo.
"venv_ml\Scripts\python.exe" run_backtest.py --source monthly_rebalance --start-date %BACKTEST_START% --end-date %BACKTEST_END% --top-n %TOP_N% %MR_ARG%
echo.
pause
goto :menu

:set_stock_pool
cls
echo.
echo ===============
echo   设置股票池
echo ===============
echo.
echo   当前: %STOCK_POOL%
echo.
echo   [1] 沪深300 (hs300)
echo   [2] 中证500 (zz500)
echo   [3] 中证800 (zz800)
echo   [4] 中证1000 (zz1000)
echo   [5] 全A股 (all)
echo.
set /p SP_CHOICE=请选择 (1-5, 默认5):
echo.
if "%SP_CHOICE%"=="1" set STOCK_POOL=hs300
if "%SP_CHOICE%"=="2" set STOCK_POOL=zz500
if "%SP_CHOICE%"=="3" set STOCK_POOL=zz800
if "%SP_CHOICE%"=="4" set STOCK_POOL=zz1000
if "%SP_CHOICE%"=="5" set STOCK_POOL=all
if "%SP_CHOICE%"=="" set STOCK_POOL=all
echo   [OK] 已设置股票池: %STOCK_POOL%
echo.
REM 同步到 config.py
venv_ml\Scripts\python.exe update_stock_pool.py %STOCK_POOL%
call :save_config
pause
goto :menu

:eof
cls
echo.
echo   再见！
echo.
