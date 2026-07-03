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

:momentum_grid_timing
cls
echo.
echo ========================================
echo   动量策略 + 网格持仓择时（方案①）
echo ========================================
echo.
echo   核心逻辑：
echo     网格持仓 ^< 20%%  → 市场偏贵 → 选3只（减仓防守）
echo     网格持仓 20-80%% → 正常市场 → 选5只（默认）
echo     网格持仓 ^> 80%%  → 市场便宜 → 选8只（低吸进攻）
echo.
echo   默认配置:
echo     动量12个月 + 季度调仓 + 2×ATR止损 + MA200过滤
echo     网格参考：沪深300指数 (2%%间距, 每格5000元)
echo.
echo   回测区间: %BACKTEST_START% ~ %BACKTEST_END%
echo.
echo   请选择股票池:
echo   ----------------
echo   [1] 沪深300 (推荐·与网格参考指数一致)
echo   [2] 中证800
echo   [3] 中证500
echo   [0] 返回主菜单
echo.
set /p MT_POOL=请选择 (1-3, 默认1):

if "%MT_POOL%"=="0" goto :menu
if "%MT_POOL%"=="2" set MT_POOL_ARG=--stock-pool 000906.SH
if "%MT_POOL%"=="3" set MT_POOL_ARG=--stock-pool 000905.SH
if "%MT_POOL%"=="" set MT_POOL_ARG=--stock-pool 000300.SH
if not defined MT_POOL_ARG set MT_POOL_ARG=--stock-pool 000300.SH

echo.
echo   正在运行...
"venv_ml\Scripts\python.exe" run_momentum_grid_timing.py %BACKTEST_START% %BACKTEST_END% --top-n 5 --lookback 12 --rebalance-freq 3 --atr-stop 2.0 --trend-filter 200 --grid-pct 0.02 --per-grid 5000 --grid-index 000300.SH %MT_POOL_ARG%
echo.
echo   [返回主菜单]
pause
goto :menu

:etf_rotation
cls
echo.
echo ========================================
echo   ETF轮动策略
echo ========================================
echo.
echo   标的池（7只国内ETF · 真实价格）：
echo     沪深300 - 上证50 - 中证500 - 中证1000
echo     创业板 - 黄金 - 货币
echo.
echo   核心逻辑：
echo     ROC动量×0.5 + 中期动量×0.3 - 波动率×0.2
echo     MA60过滤：跌破均线不买入，全部走弱时转现金
echo.
echo   回测区间: %BACKTEST_START% ~ %BACKTEST_END%
echo.
echo   请选择调仓方法:
echo   ----------------
echo   [1] 双动量法（默认·前2名等权·最常用）
echo   [2] 单动量法（满仓第1名·激进）
echo   [3] 均线过滤法（MA60过滤·保守）
echo   [0] 返回主菜单
echo.
set /p ETF_METHOD=请选择 (1-3, 默认1):

if "%ETF_METHOD%"=="0" goto :menu
set ETF_METHOD_ARG=dual
if "%ETF_METHOD%"=="1" set ETF_METHOD_ARG=dual
if "%ETF_METHOD%"=="2" set ETF_METHOD_ARG=single
if "%ETF_METHOD%"=="3" set ETF_METHOD_ARG=ma_filter

echo.
echo   正在运行...
"venv_ml\Scripts\python.exe" run_etf_rotation.py %BACKTEST_START% %BACKTEST_END% --method %ETF_METHOD_ARG%
echo.
echo   [返回主菜单]
pause
goto :menu

:pairs_trading
cls
echo.
echo ========================================
echo   配对套利策略（均值回归·多头轮动）
echo ========================================
echo.
echo   逻辑：两只高相关ETF，价差拉大时从贵的换到便宜的
echo   市场过滤：沪深300ETF跌破MA60时强制空仓
echo.
echo   回测区间: %BACKTEST_START% ~ %BACKTEST_END%
echo.
echo   请选择配对:
echo   ----------------
echo   [1] 沪深300ETF vs 上证50ETF（宽基轮动）
echo   [2] 中证500ETF vs 中证1000ETF（中小盘）
echo   [3] 创业板ETF vs 创业板50ETF（创业板）
echo   [0] 返回主菜单
echo.
set /p PAIR_CHOICE=请选择 (1-3, 默认1):

if "%PAIR_CHOICE%"=="0" goto :menu
if "%PAIR_CHOICE%"=="" set PAIR_CHOICE=1
set PAIR_ARG=%PAIR_CHOICE%

echo.
echo   正在运行...
"venv_ml\Scripts\python.exe" run_pairs_trading.py %BACKTEST_START% %BACKTEST_END% --pair %PAIR_ARG%
echo.
echo   [返回主菜单]
pause
goto :menu

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
echo   [8] 月度调仓回测（价值/红利低波/动量）
echo   [9] 狗股年度调仓
echo   [A] 短线逆转策略（超跌反弹）
echo   [B] 网格交易策略（波劙收割）
echo   [C] 动量+网格择时（方案①·网格作为市场温度计）
echo   [D] ETF轮动策略（动量轮动·纯国内资产）
echo   [E] 配对套利（均值回归·多头轮动）
echo   [0] 退出
echo.
set /p CHOICE=请选择 (1-0):
echo.

if "%CHOICE%"=="1" goto :run_all
if "%CHOICE%"=="2" goto :set_date
if "%CHOICE%"=="3" goto :set_top_n
if "%CHOICE%"=="4" goto :set_method
if "%CHOICE%"=="5" goto :set_stock_pool
if "%CHOICE%"=="6" goto :select_only
if "%CHOICE%"=="7" goto :backtest_only
if "%CHOICE%"=="8" goto :monthly_rebalance
if "%CHOICE%"=="9" goto :dogs_annual
if /i "%CHOICE%"=="A" goto :reversal
if /i "%CHOICE%"=="B" goto :grid
if /i "%CHOICE%"=="C" goto :momentum_grid_timing
if /i "%CHOICE%"=="D" goto :etf_rotation
if /i "%CHOICE%"=="E" goto :pairs_trading
if "%CHOICE%"=="0" goto :eof
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
echo   [4] 狗股策略选股（高股息+低PB+均值回归）
echo.
set /p METHOD_CHOICE=请选择 (1-4):
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
if "%METHOD_CHOICE%"=="4" (
    set SELECTION_METHOD=dogs
    echo   [OK] 已选择: 狗股策略选股
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
venv_ml\Scripts\python.exe run_backtest.py --source %SELECTION_METHOD% --start-date %BACKTEST_START% --end-date %BACKTEST_END% --top-n %TOP_N%
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
venv_ml\Scripts\python.exe run_backtest.py --source %SELECTION_METHOD% --select-only --top-n %TOP_N% --stock-pool %STOCK_POOL%
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
venv_ml\Scripts\python.exe run_backtest.py --source csv --auto --start-date %BACKTEST_START% --end-date %BACKTEST_END%
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
echo   [3] 动量效应追涨（12月动量+2×ATR止损）
echo   [0] 返回主菜单
echo.
set MR_ARG=
set /p MR_METHOD=请选择 (1-3, 0返回):

if "%MR_METHOD%"=="0" goto :menu
if "%MR_METHOD%"=="2" (
    set MR_ARG=--selection-method div_low_vol
    echo.
    echo   已选择: 红利低波选股
) else if "%MR_METHOD%"=="3" (
    set MR_ARG=--selection-method momentum
    echo.
    echo   已选择: 动量效应追涨
) else (
    set MR_ARG=--selection-method value
    echo.
    echo   已选择: 价值选股
)
echo.
echo   配置:
echo     回测区间: %BACKTEST_START% ~ %BACKTEST_END%
echo.
venv_ml\Scripts\python.exe run_backtest.py --source monthly_rebalance --start-date %BACKTEST_START% --end-date %BACKTEST_END% --top-n %TOP_N% %MR_ARG%
echo.
echo   [返回主菜单]
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

:dogs_annual
cls
echo.
echo ================
echo   狗股策略年度调仓回测
echo ================
echo.
echo   配置:
echo     回测区间: %BACKTEST_START% ~ %BACKTEST_END%
echo     选股数量: %TOP_N% 只
echo.
"venv_ml\Scripts\python.exe" run_backtest.py --source dogs_annual --start-date %BACKTEST_START% --end-date %BACKTEST_END% --top-n %TOP_N%
echo.
pause
goto :menu

:eof
cls
echo.
echo   再见！
echo.

:reversal
cls
echo.
echo ================
echo   短线逆转策略（超跌反弹·20日跌幅·8日持有）
echo ================
echo.
echo   请选择股票池:
echo   ----------------
echo   [1] 沪深300
echo   [2] 中证800
echo   [3] 中证500
echo.
set /p REV_POOL=请选择 (1-3, 默认1):

if "%REV_POOL%"=="2" set REV_ARG=--stock-pool 000906.SH
if "%REV_POOL%"=="3" set REV_ARG=--stock-pool 000905.SH
if "%REV_POOL%"=="" set REV_ARG=--stock-pool 000300.SH
if not defined REV_ARG set REV_ARG=--stock-pool 000300.SH

echo.
echo   配置: 20日跌幅排名, 8日持有, 5只, MACD金叉过滤, 8%%止损
echo   区间: %BACKTEST_START% ~ %BACKTEST_END%
echo.
echo   正在运行...
"venv_ml\Scripts\python.exe" run_monthly_rebalance.py %BACKTEST_START% %BACKTEST_END% --selection-method reversal %REV_ARG% --top-n 5 --reversal-lookback 20 --reversal-hold 8 --market-filter macd --reversal-stop 0.08
echo.
pause
goto :menu

:grid
cls
echo.
echo ================
echo   网格交易策略（百分比网格·波劙收割）
echo ================
echo.
echo   请选择标的:
echo   ----------------
echo   [1] 沪深300 ETF (510300，指数代理模拟)
echo   [2] 中证500 ETF (510500，指数代理模拟)
echo   [3] 中证800 ETF (512100，指数代理模拟)
echo   [0] 返回主菜单
echo.
set GRID_ARG=
set /p GRID_CODE=请选择 (1-3, 0返回):

if "%GRID_CODE%"=="0" goto :menu
if "%GRID_CODE%"=="2" set GRID_ARG=000905.SH
if "%GRID_CODE%"=="3" set GRID_ARG=000906.SH
if "%GRID_CODE%"=="" set GRID_ARG=000300.SH
if not defined GRID_ARG set GRID_ARG=000300.SH

echo.
echo   请设置网格间距:
echo   ----------------
echo   [1] 1%% (窄网格，高频交易)
echo   [2] 2%% (标准，推荐)
echo   [3] 3%% (宽网格，低频交易)
echo.
set /p GRID_PCT=请选择 (1-3, 默认2):

if "%GRID_PCT%"=="1" set GRID_PCT_ARG=--grid-pct 0.01
if "%GRID_PCT%"=="3" set GRID_PCT_ARG=--grid-pct 0.03
if "%GRID_PCT%"=="" set GRID_PCT_ARG=--grid-pct 0.02
if not defined GRID_PCT_ARG set GRID_PCT_ARG=--grid-pct 0.02

echo.
echo   配置: 网格间距已设置, 每格5000元, 初始50%%仓位
echo   区间: %BACKTEST_START% ~ %BACKTEST_END%
echo.
echo   正在运行...
"venv_ml\Scripts\python.exe" run_grid_backtest.py %GRID_ARG% %BACKTEST_START% %BACKTEST_END% %GRID_PCT_ARG% --per-grid 5000
echo.
echo   [返回主菜单]
pause
goto :menu
