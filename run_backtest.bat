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
if not defined P_TOTAL_CAPITAL set P_TOTAL_CAPITAL=100000
if not defined P_SELECTION_METHOD set P_SELECTION_METHOD=multi
if not defined P_STOCK_POOL set P_STOCK_POOL=all
if not defined P_PAIRS_CAPITAL set P_PAIRS_CAPITAL=500000
if not defined P_VALUE_ENHANCED set P_VALUE_ENHANCED=
if not defined P_VALUE_QGATES set P_VALUE_QGATES=on

REM 应用到会话变量
set BACKTEST_START=%P_BACKTEST_START%
set BACKTEST_END=%P_BACKTEST_END%
set TOP_N=%P_TOP_N%
set SELECTION_METHOD=%P_SELECTION_METHOD%
set VALUE_ENHANCED=%P_VALUE_ENHANCED%
set VALUE_QGATES=%P_VALUE_QGATES%
set TOTAL_CAPITAL=%P_TOTAL_CAPITAL%
set STOCK_POOL=%P_STOCK_POOL%

REM ================================
REM  保存持久化配置到 user_config.bat
REM  （子程序，用 call :save_config 调用）
REM ================================
goto :menu_start

:momentum_grid_timing
cls
echo.
echo ========================================
echo   动量策略 + 网格持仓择时[方案①]
echo ========================================
echo.
echo   核心逻辑：
echo     网格持仓 ^< 20%%  → 市场偏贵 → 选3只[减仓防守]
echo     网格持仓 20-80%% → 正常市场 → 选5只[默认]
echo     网格持仓 ^> 80%%  → 市场便宜 → 选8只[低吸进攻]
echo.
echo   默认配置:
echo     动量12个月 + 可调仓频率(季度/每月/每周) + 2×ATR止损 + MA200过滤
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

:: 先设默认值，再按输入覆盖（修复变量残留bug）
set MT_POOL_ARG=--stock-pool 000300.SH
if "%MT_POOL%"=="2" set MT_POOL_ARG=--stock-pool 000906.SH
if "%MT_POOL%"=="3" set MT_POOL_ARG=--stock-pool 000905.SH

echo.
echo   请选择调仓频率:
echo   ----------------
echo   [1] 季度调仓 (每3个月) [默认]
echo   [2] 每月调仓
echo   [3] 每周调仓
echo.
set /p MT_FREQ=请选择 (1-3, 默认1):

:: 先设默认值，再按输入覆盖（修复变量残留bug）
set MT_FREQ_ARG=--rebalance-freq 3
if "%MT_FREQ%"=="2" set MT_FREQ_ARG=--rebalance-freq 1
if "%MT_FREQ%"=="3" set MT_FREQ_ARG=--rebalance-freq weekly

echo.
echo   正在运行...
"venv_ml\Scripts\python.exe" run_momentum_grid_timing.py %BACKTEST_START% %BACKTEST_END% --top-n 5 --lookback 12 %MT_FREQ_ARG% --atr-stop 2.0 --trend-filter 200 --grid-pct 0.02 --per-grid 5000 --grid-index 000300.SH %MT_POOL_ARG%
echo.
echo   [返回主菜单]
pause
goto :menu

:etf_rotation
cls
echo.
echo ========================================
echo   ETF轮动策略 [V6 + Regime兜底 · 生产版]
echo ========================================
echo.
echo   标的池[20只精选ETF · 真实价格] + Regime牛熊识别(沪深300 MA200+市场宽度):
echo     沪深300 - 上证50 - 中证500 - 中证1000
echo     创业板50 - 科创50 - 证券 - 医药
echo     消费 - 5G通信 - 新能源车 - 旅游
echo     黄金 - 红利 - 货币 - 更多...
echo.
echo   核心逻辑：
echo     ROC动量×0.4 + 中期动量×0.4 - 波动率×0.2[手册推荐权重]
echo     MA60过滤+追高保护+最小切换阈值[防抖动]
echo     + RSRS质量分(可开关) + Regime兜底(牛市强制≥40%%宽基)
echo     全部走弱时转货币基金避险，保留10%%现金缓冲
echo.
echo   回测区间: %BACKTEST_START% ~ %BACKTEST_END%
echo.
echo   请选择运行模式:
echo   ----------------
echo   [1] V6+Regime生产版[默认·已替代原ETF轮动]  --rsrs 0
echo   [2] V6+Regime+RSRS0.3[带质量分]
echo   [3] V6合并版(无Regime·纯对照)  --regime off
echo   [4] 原版平台ETF轮动(审计对照基线·保留)
echo   [5] (V6 4只版) 红利/创业50/纳指/黄金 · 纯RSRS+动量
echo   [0] 返回主菜单
echo.
set ETF_POOL_ARG=
set ETF_VAR_ARG=
set ETF_TOPN_ARG=
set ETF_VARSEL=
set ETF_PREMIUM_ARG=
set ETF_MA_ARG=
set ETF_MODE=
set /p ETF_MODE=请选择 (1-5, 默认1):
if "%ETF_MODE%"=="0" goto :menu
if "%ETF_MODE%"=="" set ETF_MODE=1
set ETF_REGIME=off
set ETF_RSRS=0
set ETF_LEGACY=0
if "%ETF_MODE%"=="1" set ETF_REGIME=on
if "%ETF_MODE%"=="2" (set ETF_REGIME=on & set ETF_RSRS=0.3)
if "%ETF_MODE%"=="3" set ETF_REGIME=off
if "%ETF_MODE%"=="4" set ETF_LEGACY=1
goto :etf_var

:etf_var
echo.
echo   VaR 仓位缩放（可选·按篮子VaR反解投入比例，未投部分留现金·默认关闭）:
echo   [0] 关闭=默认   [1] 95%%置信   [2] 99%%置信
set /p ETF_VARSEL=请选择 (0/1/2, 回车=0关闭):
if "%ETF_VARSEL%"=="1" set ETF_VAR_ARG=--var-control 95 --var-maxdd 15 --var-n 5
if "%ETF_VARSEL%"=="2" set ETF_VAR_ARG=--var-control 99 --var-maxdd 15 --var-n 5
goto :etf_premium

:etf_premium
echo.
echo   ETF折溢价过滤（可选·实盘防泡沫护栏·默认关闭）:
echo   [0] 关闭=默认（回测可复现）
echo   [1] rolling=自适应分位数（推荐·实盘护栏）
echo   [2] uniform=统一5%%硬阈值
echo   [3] strict=跨境更严（3%%/5%%）
echo   [4] qdii=仅限申赎受限品种8%%
set /p ETF_PREMIUM_SEL=请选择 (0/1/2/3/4, 回车=0关闭):
if "%ETF_PREMIUM_SEL%"=="1" set ETF_PREMIUM_ARG=--premium-filter rolling
if "%ETF_PREMIUM_SEL%"=="2" set ETF_PREMIUM_ARG=--premium-filter uniform
if "%ETF_PREMIUM_SEL%"=="3" set ETF_PREMIUM_ARG=--premium-filter strict
if "%ETF_PREMIUM_SEL%"=="4" set ETF_PREMIUM_ARG=--premium-filter qdii
if not defined ETF_PREMIUM_ARG set ETF_PREMIUM_ARG=
goto :etf_exec

:etf_exec
echo.
echo   正在运行...
if "%ETF_MODE%"=="5" (
  "venv_ml\Scripts\python.exe" run_etf_rotation_v6.py
  echo.
  echo   [返回主菜单]
  pause
  goto :menu
)
if "%ETF_LEGACY%"=="1" (
  "venv_ml\Scripts\python.exe" run_etf_rotation.py %BACKTEST_START% %BACKTEST_END% --method dual %ETF_MA_ARG% %ETF_POOL_ARG% %ETF_TOPN_ARG% %ETF_VAR_ARG% %ETF_PREMIUM_ARG%
) else (
  "venv_ml\Scripts\python.exe" run_etf_rotation_v6_merged.py %BACKTEST_START% %BACKTEST_END% --method dual %ETF_MA_ARG% %ETF_POOL_ARG% %ETF_TOPN_ARG% %ETF_VAR_ARG% %ETF_PREMIUM_ARG% --rsrs-weight %ETF_RSRS% --regime %ETF_REGIME%
)
echo.
echo   [返回主菜单]
pause
goto :menu

:pairs_trading
cls
echo.
echo =======================================
echo   配对套利策略[均值回归·多头轮动]
echo =======================================
echo.
echo   逻辑：两只高相关ETF，价差拉大时从贵的换到便宜的
echo   市场过滤：沪深300ETF跌破MA60时强制空仓
echo   初始资金：%P_PAIRS_CAPITAL% 元
echo.
echo   回测区间: %BACKTEST_START% ~ %BACKTEST_END%
echo.
echo   请选择配对:
echo   -----------------
echo   [1] 沪深300   vs 上证50    [宽基·大盘内部]
echo   [2] 中证500   vs 中证800   [宽基·中大盘]
echo   [3] 创业板     vs 创业板50   [宽基·科技成长]
echo   [4] 科创50     vs 创业板50   [宽基·科技成长]
echo   [5] 半导体     vs 新能源车   [行业·科技制造]
echo   [6] 沪深300   vs 中证800   [宽基·大中盘]
echo   [7] 恒生ETF   vs 沪深300   [跨境·AH溢价]
echo   [8] 黄金ETF   vs 国债ETF   [避险·股债轮动]
echo   [9] 红利ETF   vs 红利低波ETF [红利·同主题高相关]
echo   [0] 返回主菜单
echo.
set PAIR_CHOICE=
set /p PAIR_CHOICE=请选择 (1-9, 默认1):

if "%PAIR_CHOICE%"=="0" goto :menu
if "%PAIR_CHOICE%"=="" set PAIR_CHOICE=1
set PAIR_ARG=%PAIR_CHOICE%

echo.
set /p PAIRS_CAPITAL_INPUT=输入初始资金(元, 回车默认%P_PAIRS_CAPITAL%):
if "%PAIRS_CAPITAL_INPUT%"=="" set PAIRS_CAPITAL_INPUT=%P_PAIRS_CAPITAL%
set P_PAIRS_CAPITAL=%PAIRS_CAPITAL_INPUT%

echo.
echo   正在运行(初始资金 %P_PAIRS_CAPITAL% 元)...
"venv_ml\Scripts\python.exe" run_pairs_trading.py %BACKTEST_START% %BACKTEST_END% --pair %PAIR_ARG% --capital %P_PAIRS_CAPITAL%
echo.
echo   [返回配对菜单]
pause
goto :pairs_trading
:save_config
(
echo set P_BACKTEST_START=%BACKTEST_START%
echo set P_BACKTEST_END=%BACKTEST_END%
echo set P_TOP_N=%TOP_N%
echo set P_TOTAL_CAPITAL=%TOTAL_CAPITAL%
echo set P_SELECTION_METHOD=%SELECTION_METHOD%
echo set P_VALUE_ENHANCED=%VALUE_ENHANCED%
echo set P_VALUE_QGATES=%VALUE_QGATES%
echo set P_PAIRS_CAPITAL=%P_PAIRS_CAPITAL%
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
echo   初始【总】资金: %P_TOTAL_CAPITAL% 元 (将均分到每只)
echo   选股策略: %SELECTION_METHOD%
echo   股票池: %STOCK_POOL%
echo.
echo   ================
echo   主菜单:
echo   ================
echo.
echo   [1] 运行选股 + 回测[使用当前配置]
echo   [2] 设置回测区间
echo   [3] 设置选股数量 / 初始资金
echo   [4] 选择选股策略
echo   [5] 设置股票池
echo   [6] 月度调仓回测[价值/红利低波/动量/质量复合/高股息成长]
echo   [7] 年度调仓·狗股组[狗股/价值选股/神奇公式/神奇公式v2]
echo   [8] 短线逆转策略[超跌反弹]
echo   [9] 网格交易策略[波段收割]
echo   [A] 动量+网格择时 [方案①·网格作为市场温度计]
echo   [B] ETF轮动策略[动量轮动·纯国内资产]
echo   [C] 配对套利[均值回归·多头轮动]
echo   [D] 指数/ETF 涨跌一览[回测周期表现]
echo   [E] 小市值轮动 [①周频止损版 ②Kara纯最小市值月频版]
echo   [F] ETF定投 [单产品/宽基篮子·月/周对比]
echo   [0] 退出
echo.
set /p CHOICE=请选择 (1-9/A-F, 0退出):
echo.

if "%CHOICE%"=="1" goto :run_all
if "%CHOICE%"=="2" goto :set_date
if "%CHOICE%"=="3" goto :set_top_n_capital
if "%CHOICE%"=="4" goto :set_method_menu
if "%CHOICE%"=="5" goto :set_stock_pool
if "%CHOICE%"=="6" goto :monthly_rebalance
if "%CHOICE%"=="7" goto :dogs_annual
if "%CHOICE%"=="8" goto :reversal
if "%CHOICE%"=="9" goto :grid
if /i "%CHOICE%"=="A" goto :momentum_grid_timing
if /i "%CHOICE%"=="B" goto :etf_rotation
if /i "%CHOICE%"=="C" goto :pairs_trading
if /i "%CHOICE%"=="D" goto :index_etf_changes
if /i "%CHOICE%"=="E" goto :sc_rotation
if /i "%CHOICE%"=="F" goto :dca_etf
if "%CHOICE%"=="0" goto :eof
goto :menu
:dca_etf
cls
echo.
echo ================
echo   ETF 定投策略 (DCA)
echo ================
echo.
echo   国内被动宽基 ETF 定投：月度 / 周度 / 一次性投入对比
echo   月投 / 周投金额可在下方自行设定（默认 4000 / 1000）
echo.
echo   请选择定投标的：
echo   [1] 沪深300ETF  (510300)
echo   [2] 中证500ETF  (510500)
echo   [3] 创业板ETF   (159915)
echo   [4] 上证50ETF   (510050)
echo   [5] 宽基核心篮子（沪深300+中证500+创业板+上证50，4只等权）
echo   [6] 红利ETF     (510880·上证红利，全历史)
echo   [7] 红利低波ETF (512890·中证红利低波)
echo   [8] 红利核心篮子（红利+红利低波，dividend）
echo   [9] 红利+科技弹性篮子（红利+红利低波+创业板，div_tech）
echo   [0] 返回主菜单
echo.
set DCA_CODE=
set DCA_PRESET=
set DCA_MONTHLY=4000
set DCA_WEEKLY=1000
set /p DCA_CHOICE=请选择 (0-9):
if "%DCA_CHOICE%"=="0" goto :menu
if "%DCA_CHOICE%"=="1" ( set DCA_CODE=510300.SH )
if "%DCA_CHOICE%"=="2" ( set DCA_CODE=510500.SH )
if "%DCA_CHOICE%"=="3" ( set DCA_CODE=159915.SZ )
if "%DCA_CHOICE%"=="4" ( set DCA_CODE=510050.SH )
if "%DCA_CHOICE%"=="5" ( set DCA_PRESET=core4 )
if "%DCA_CHOICE%"=="6" ( set DCA_CODE=510880.SH )
if "%DCA_CHOICE%"=="7" ( set DCA_CODE=512890.SH )
if "%DCA_CHOICE%"=="8" ( set DCA_PRESET=dividend )
if "%DCA_CHOICE%"=="9" ( set DCA_PRESET=div_tech )
if not defined DCA_CODE ( if not defined DCA_PRESET ( goto :dca_etf ) )
echo.
echo   选择频率：
echo   [1] 月度定投   [2] 周度定投   [3] 月/周对比 + 一次性投入基准
echo.
set /p DCA_FREQ_CHOICE=请选择 (1-3):
if "%DCA_FREQ_CHOICE%"=="2" ( set DCA_FREQ=weekly ) else if "%DCA_FREQ_CHOICE%"=="3" ( set DCA_FREQ=both ) else ( set DCA_FREQ=monthly )
echo.
echo   定投模式（smart=文章《玩红利ETF的实战思路》5周/20周线操作法）：
echo   [1] 普通纪律定投(plain)   [2] 均线增强(smart)
echo.
set /p DCA_MODE_CHOICE=请选择 (1-2, 回车默认1):
if "%DCA_MODE_CHOICE%"=="2" ( set DCA_MODE=smart ) else ( set DCA_MODE=plain )
echo.
echo   投入金额（月投 / 周投，回车用默认值）:
set /p DCA_MONTHLY=  月投金额 (默认 4000):
if "%DCA_MONTHLY%"=="" set DCA_MONTHLY=4000
set /p DCA_WEEKLY=  周投金额 (默认 1000):
if "%DCA_WEEKLY%"=="" set DCA_WEEKLY=1000
echo.
echo   回测区间: %BACKTEST_START% ~ %BACKTEST_END%
echo.
if defined DCA_PRESET (
  "venv_ml\Scripts\python.exe" run_backtest.py --source dca_etf --dca-preset %DCA_PRESET% --dca-freq %DCA_FREQ% --dca-mode %DCA_MODE% --dca-monthly %DCA_MONTHLY% --dca-weekly %DCA_WEEKLY% --start-date %BACKTEST_START% --end-date %BACKTEST_END%
) else (
  "venv_ml\Scripts\python.exe" run_backtest.py --source dca_etf --dca-freq %DCA_FREQ% --dca-code %DCA_CODE% --dca-mode %DCA_MODE% --dca-monthly %DCA_MONTHLY% --dca-weekly %DCA_WEEKLY% --start-date %BACKTEST_START% --end-date %BACKTEST_END%
)
echo.
echo   [返回主菜单]
pause
goto :menu

:set_top_n_capital
cls
echo.
echo ================
echo   设置选股数量 / 初始资金
echo ================
echo.
echo   当前选股数量: %TOP_N% 只
echo   当前初始(总)资金: %P_TOTAL_CAPITAL% 元 (将均分到每只)
echo.
set /p TOP_N_INPUT=  选股数量 (回车默认 %P_TOP_N%):
if "%TOP_N_INPUT%"=="" set TOP_N_INPUT=%P_TOP_N%
set P_TOP_N=%TOP_N_INPUT%
set TOP_N=%TOP_N_INPUT%
echo.
set /p CAPITAL_INPUT=  初始总资金(元, 回车默认 %P_TOTAL_CAPITAL%):
if "%CAPITAL_INPUT%"=="" set CAPITAL_INPUT=%P_TOTAL_CAPITAL%
set P_TOTAL_CAPITAL=%CAPITAL_INPUT%
set TOTAL_CAPITAL=%CAPITAL_INPUT%
echo.
echo   [OK] 已更新: 选股 %TOP_N% 只 / 总资金 %TOTAL_CAPITAL% 元
echo.
call :save_config
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


:set_method_menu
cls
echo.
echo ================
echo   选择选股策略
echo ================
echo.
echo   [1] 多因子选股[技术+基本面因子]
echo   [2] 价值投资选股[价值投资量化策略]
echo   [3] 红利低波动选股策略[高股息+低波动]
echo   [4] 狗股策略选股[高股息+低PB+均值回归]
echo.
set /p METHOD_CHOICE=请选择 (1-4):
echo.

if "%METHOD_CHOICE%"=="1" (
    set SELECTION_METHOD=multi
    echo   [OK] 已选择: 多因子选股
    call :save_config
    goto :menu
)
if "%METHOD_CHOICE%"=="2" goto :method_value
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
goto :set_method_menu

:method_value
set SELECTION_METHOD=value
echo   [OK] 已选择: 价值投资选股
echo.
echo   价值 BM 增强(纯BM·全市场BM前30%%+市值中性化^):
echo   [1] 不开(原版·破净+ROE质量^)  [2] 一键开启
set /p VE_CHOICE=请选择 1-2, 默认1不开:
if "%VE_CHOICE%"=="2" (
    set VALUE_ENHANCED=--value-mode pure_bm --value-pct 0.3 --value-size-neutral
    echo   [OK] BM增强: 已开启(纯BM + 前30%% + 市值中性化^)
) else (
    set VALUE_ENHANCED=
    echo   [OK] BM增强: 未开启(原版破净^)
)
echo.
echo   四道质量门槛(盈余质量/杠杆/应收/估值纵向分位·防价值陷阱^):
echo   [1] 启用(默认·过滤高杠杆现金流差的伪便宜^)  [2] 关闭(仅破净+ROE·用于对照^)
set /p QG_CHOICE=请选择 1-2, 默认1启用:
if "%QG_CHOICE%"=="2" (
    set VALUE_QGATES=off
    echo   [OK] 质量门槛: 已关闭(仅破净+ROE^)
) else (
    set VALUE_QGATES=on
    echo   [OK] 质量门槛: 已启用(②盈余质量 ③杠杆 ④应收 ⑤纵向分位^)
)
call :save_config
goto :menu

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
set VB_ARGS=
if "%SELECTION_METHOD%"=="value" set VB_ARGS=%VALUE_ENHANCED% --value-quality-gates %VALUE_QGATES%
:run_all_exec
venv_ml\Scripts\python.exe run_backtest.py --source %SELECTION_METHOD% --start-date %BACKTEST_START% --end-date %BACKTEST_END% --top-n %TOP_N% --capital %TOTAL_CAPITAL% %VB_ARGS%
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
echo   [1] 价值选股[PB破净+ROE质量]
echo   [2] 红利低波选股[高股息+低波动]
echo   [3] 动量效应追涨[动量选股]
echo   [4] 红利低波质量复合[季度调仓]
echo   [5] MACD择时[逐股DIF^>DEA·无KDJ·跟随全局股票池]
echo   [6] 高股息+基本面成长[股息率前10%%+PE/PEG/ROE/增长五关·月调仓·涨停跑路]
echo   [7] 价值选股+组合层分散[70%%权益+15%%国债+15%%黄金·月度再平衡]
echo   [0] 返回主菜单
echo.
set MR_ARG=
set MR_VAR_ARG=
set VAR_CTRL=0
set VAR_MDD=15
    set VAR_N=5
    set VA_LOOK=0
    set VA_PCT=70
    set VAR_SEL=
    set VA_SEL=
    set /p MR_METHOD=请选择 (1-7, 0返回):

if "%MR_METHOD%"=="0" goto :menu
if "%MR_METHOD%"=="6" goto :mr_dg
if "%MR_METHOD%"=="2" goto :mr_div
if "%MR_METHOD%"=="4" goto :mr_divq
if "%MR_METHOD%"=="5" goto :macd_timing
if "%MR_METHOD%"=="3" goto :mr_mom
if "%MR_METHOD%"=="7" goto :mr_blended
set MR_ARG=--selection-method value
echo.
echo   已选择: 价值选股
goto :mr_after

:mr_div
set MR_ARG=--selection-method div_low_vol
echo.
echo   已选择: 红利低波选股
goto :mr_after

:mr_dg
set MR_ARG=--selection-method div_growth
echo.
echo   已选择: 高股息+基本面成长[股息率前10%%+PE/PEG/ROE/营收/净利五关]
echo   说明: 月调仓 + 涨停跑路日规则（昨涨停今未封→当日收盘卖）；无个股止损
goto :mr_after

:mr_divq
set MR_ARG=--selection-method div_low_vol_quality --dlvq-mode official_compact
echo.
echo   调仓频率:
echo   ----------------
echo   [1] 季度档[默认·沿用官方编制法]
echo   [2] 年度档[12月官方调整日·降换手·60.3%% vs 季度160.7%%]
echo.
set MR_RB_TXT=季度档
set /p MR_RB=请选择 (1-2, 回车=1季度档):
if "%MR_RB%"=="2" set MR_ARG=%MR_ARG% --dlvq-rebal year
if "%MR_RB%"=="2" set MR_RB_TXT=年度档[降换手]
echo.
echo   已选择: 红利低波质量复合[%MR_RB_TXT%]
echo.
echo   红利通道仓位 overlay（000922通道位置→权益仓位，贵减仓/便宜满仓·已验证正向）:
echo   [1] 平衡档[默认·rolling w756/k0.5·开启]
echo   [2] 小账户档[rolling w504/k0.3·开启·卡玛0.40]
echo   [3] 关闭[满仓基线·对照普通红利低波]
echo   [4] 实盘前瞻版[--live-forward·今日重选+今日k]
echo   [5] 历史买列表[--live·上期已选票]
set DCO_SEL=
set /p DCO_SEL=请选择 (1-5, 回车=1平衡档):
if "%DCO_SEL%"=="2" set MR_ARG=%MR_ARG% --div-channel-window 504 --k-min 0.3
if "%DCO_SEL%"=="3" set MR_ARG=%MR_ARG% --no-div-channel-overlay
if "%DCO_SEL%"=="4" set MR_ARG=%MR_ARG% --live-forward
if "%DCO_SEL%"=="5" set MR_ARG=%MR_ARG% --live
goto :mr_after

:mr_mom
set MR_ARG=--selection-method momentum
echo.
echo   已选择: 动量效应追涨
echo.
echo   VaR 仓位缩放（凶策略默认开启 95%%，锁定目标回撤）:
echo   [0] 关闭   [1] 95%%置信=默认   [2] 99%%置信
set /p VAR_SEL=请选择 (0/1/2, 回车=1·95%%开启):
set VAR_CTRL=95
if "%VAR_SEL%"=="0" set VAR_CTRL=0
if "%VAR_SEL%"=="2" set VAR_CTRL=99
if not "%VAR_CTRL%"=="0" set /p VAR_MDD=目标最大回撤上限 %%（默认15）:
if "%VAR_MDD%"=="" set VAR_MDD=15
if not "%VAR_CTRL%"=="0" set /p VAR_N=连续下跌周期数 N（趋势=5/反转=3, 默认5）:
if "%VAR_N%"=="" set VAR_N=5
echo.
echo   价值区过滤（动量只接价值区内/上方，剔除价值区下方弱势票）:
echo   [0] 关闭=默认   [1] 开启·回看20日
set /p VA_SEL=请选择 (0/1, 回车=0关闭):
if "%VA_SEL%"=="1" set VA_LOOK=20
goto :mr_after

:mr_after
echo.
set VB_ARGS=
if "%MR_METHOD%"=="2" goto :mr_exec
if "%MR_METHOD%"=="3" goto :mr_exec
if "%MR_METHOD%"=="4" goto :mr_exec
if "%MR_METHOD%"=="6" goto :mr_exec
set VB_ARGS=%VALUE_ENHANCED% --value-quality-gates %VALUE_QGATES%
:mr_exec
REM === VaR 仓位缩放参数（动量/红利低波生效；价值忽略，var-control=0 即关闭）===
set VB_ARGS=%VB_ARGS% --var-control %VAR_CTRL% --var-maxdd %VAR_MDD% --var-n %VAR_N%
if not "%VA_LOOK%"=="0" set VB_ARGS=%VB_ARGS% --value-area %VA_LOOK% --va-pct %VA_PCT%
echo.
echo   配置:
echo     回测区间: %BACKTEST_START% ~ %BACKTEST_END%
echo.
venv_ml\Scripts\python.exe run_backtest.py --source monthly_rebalance --start-date %BACKTEST_START% --end-date %BACKTEST_END% --top-n %TOP_N% --capital %TOTAL_CAPITAL% %MR_ARG% %VB_ARGS%
echo.
echo   [返回主菜单]
pause
goto :menu

:mr_blended
cls
echo.
echo ================
echo   价值选股 + 组合层分散 [70%%权益 + 15%%国债 + 15%%黄金]
echo ================
echo.
echo   底层: run_value_backtest.py 价值选股(破净+ROE质量, 月度第5交易日调仓)
echo   组合层: 把价值选股日频NAV与 国债ETF 511260 + 黄金ETF 518880 做月度再平衡
echo   作用: 降波动/降回撤/提夏普 (机制真, 收益溢价依赖债金双牛窗口, 非免费午餐)
echo   区间: %BACKTEST_START% ~ %BACKTEST_END%
echo.
echo   权重方案:
echo   [1] static 固定 0.7/0.15/0.15 (推荐·已验证价值选股夏普 0.31->0.54, MDD -36.8%%->-26.4%%)
echo   [2] invvol 逆波动月度 (债金顶满, 2018-2025窗口虚高, 谨慎)
echo.
set PL_SCHEME=static
set /p PL_SCHEME_SEL=请选择 (1/2, 回车=1 static):
if "%PL_SCHEME_SEL%"=="2" set PL_SCHEME=invvol
echo.
set PL_POOL=zz800
if "%STOCK_POOL%"=="hs300" set PL_POOL=hs300
if "%STOCK_POOL%"=="zz500" set PL_POOL=zz500
if "%STOCK_POOL%"=="zz1000" set PL_POOL=zz1000
if "%STOCK_POOL%"=="all" set PL_POOL=zz800
echo.
echo   正在运行(池=%PL_POOL% 权益70%%/国债15%%/黄金15%%/方案=%PL_SCHEME%)...
"venv_ml\Scripts\python.exe" run_value_backtest.py --start %BACKTEST_START% --end %BACKTEST_END% --initial-cash %TOTAL_CAPITAL% --top-n %TOP_N% --pool %PL_POOL% --portfolio-layer 0.7,0.15,0.15 --layer-scheme %PL_SCHEME%
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
echo   年度调仓·狗股组[1月首个交易日·等权·非指数复制]
echo ================
echo.
echo   配置:
echo     回测区间: %BACKTEST_START% ~ %BACKTEST_END%
echo     选股数量: %TOP_N% 只
echo     总资金: %TOTAL_CAPITAL% 元 (将均分到每只)
echo.
echo   请选择子策略:
echo   ----------------
echo   [1] 狗股策略[高股息 + 低PB + 均值回归]
echo   [2] 价值选股[破净 + ROE质量 + 自由现金流]
echo   [3] 神奇公式（Magic Formula · ROC+EY双排名 · 年度调仓）
echo   [4] 神奇公式修改版v2（EBIT3年均值+MA200趋势过滤+行业上限 · 抗回撤）
echo   [0] 返回主菜单
echo.
set DA_STRAT=dogs
set DA_QG=
set /p DA_CHOICE=请选择 (1-4, 默认1狗股):
if "%DA_CHOICE%"=="0" goto :menu
if "%DA_CHOICE%"=="2" set DA_STRAT=value
if "%DA_CHOICE%"=="3" set DA_STRAT=magic
if "%DA_CHOICE%"=="4" goto :magic_v2_run
if "%DA_CHOICE%"=="" set DA_STRAT=dogs
if "%DA_STRAT%"=="value" set DA_QG=--value-quality-gates %VALUE_QGATES%

echo.
echo   正在运行[子策略: %DA_STRAT%]...
"venv_ml\Scripts\python.exe" run_backtest.py --source dogs_annual --start-date %BACKTEST_START% --end-date %BACKTEST_END% --top-n %TOP_N% --capital %TOTAL_CAPITAL% --dogs-strategy %DA_STRAT% %DA_QG%
echo.
pause
goto :menu

:magic_v2_run
echo.
echo   正在运行[神奇公式修改版 v2]...
echo     ①EBIT近3年均值(抑制周期股盈利峰值陷阱)
echo     ②HS300小于MA200时降半仓(月检趋势过滤)
echo     ③单行业上限 2 只/行业（持仓数=全局选股数 %TOP_N%）
"venv_ml\Scripts\python.exe" run_magic_v2.py %BACKTEST_START% %BACKTEST_END% --top-n %TOP_N% --capital %TOTAL_CAPITAL%
echo.
echo   [对照] 报告尾部自动附带原版(magic)同区间对照表
pause
goto :menu

:index_etf_changes
cls
echo.
echo ================
echo   指数 / ETF 涨跌一览
echo ================
echo.
echo   覆盖主要宽基与风格指数[优先显示ETF，无ETF则显示指数]
echo.
echo   当前回测区间设置: %BACKTEST_START% ~ %BACKTEST_END%
echo.
set IEC_END=%BACKTEST_END%
set IEC_END_IN=
set /p IEC_END_IN=结束日期 YYYYMMDD (回车沿用 %BACKTEST_END%, 输入 auto 用数据库最新日):
if not "%IEC_END_IN%"=="" set IEC_END=%IEC_END_IN%
if "%IEC_END%"=="" set IEC_END=auto
echo.
echo   实际统计区间: %BACKTEST_START% ~ %IEC_END%
echo.
echo   正在统计...
echo.
"venv_ml\Scripts\python.exe" show_index_etf_changes.py %BACKTEST_START% %IEC_END%
echo.
echo   [HTML报告] outputs\index_etf_changes_%BACKTEST_START%_*.html
echo.
echo   [返回主菜单]
pause
goto :menu


:reversal
cls
echo.
echo ================
echo   短线逆转策略[超跌反弹·20日跌幅·8日持有]
echo ================
echo.
echo   请选择股票池:
echo   ----------------
echo   [1] 沪深300
echo   [2] 中证800
echo   [3] 中证500
echo   [0] 返回主菜单
echo.
set /p REV_POOL=请选择 (1-3, 0返回主菜单):

if "%REV_POOL%"=="0" goto :menu

set REV_ARG=--stock-pool 000300.SH
if "%REV_POOL%"=="2" set REV_ARG=--stock-pool 000906.SH
if "%REV_POOL%"=="3" set REV_ARG=--stock-pool 000905.SH

set REV_VAR=--var-control 95 --var-maxdd 15 --var-n 3
set REV_VA=0
set REV_FO=0
echo.
echo   VaR 仓位缩放（反转=肥尾策略，默认开启 95%%·目标回撤15%%·N=3）:
echo   [0] 关闭   [1] 95%%置信(默认)   [2] 99%%置信
set /p REV_VARSEL=请选择 (0/1/2, 回车=1·95%%开启):
if "%REV_VARSEL%"=="0" (set REV_VAR=) else if "%REV_VARSEL%"=="2" (set REV_VAR=--var-control 99 --var-maxdd 15 --var-n 3)
echo.
echo   价值区过滤(反转只接价值区下沿/下方超跌区):
echo   [0] 关闭(默认)   [1] 开启(回看20日)
set /p REV_VASEL=请选择 (0/1, 回车=0关闭):
if "%REV_VASEL%"=="1" (set REV_VA=20)
echo.
echo   反转 fakeout-reclaim 优先(扫止损后快速收回的标的排前):
echo   [0] 关闭(默认)   [1] 开启
set /p REV_FOSEL=请选择 (0/1, 回车=0关闭):
if "%REV_FOSEL%"=="1" (set REV_FO=1)
if "%REV_FO%"=="1" (set REV_FO_ARG=--fakeout-reclaim) else (set REV_FO_ARG=)

echo.
echo   配置: 20日跌幅排名, 8日持有, 5只, MACD金叉过滤, 8%%止损
echo   区间: %BACKTEST_START% ~ %BACKTEST_END%
echo.
echo   正在运行...
"venv_ml\Scripts\python.exe" run_monthly_rebalance.py %BACKTEST_START% %BACKTEST_END% --selection-method reversal %REV_ARG% --top-n 5 --reversal-lookback 20 --reversal-hold 8 --market-filter macd --reversal-stop 0.08 %REV_VAR% --value-area %REV_VA% --va-pct 70 %REV_FO_ARG%
echo.
pause
goto :reversal

:grid
cls
echo.
echo ================
echo   网格交易策略[百分比网格·波段收割]
echo ================
echo.
echo   请选择标的:
echo   ----------------
echo   [1]  沪深300 ETF (510300，真实ETF)
echo   [2]  中证500 ETF (510500，真实ETF)
echo   [4]  上证指数 (000001.SH)
echo   [5]  深证成指 (399001.SZ)
echo   [7]  中证1000 (000852.SH)
echo   [8]  科创50 (000688.SH)
echo   [9]  科创100 (000698.SH)
echo   [10] 创业板指 (399006.SZ)
echo   [11] 创业板50 (399673.SZ)
echo   [12] 中证2000 (932000.SH)
echo   [0] 返回主菜单
echo.
set GRID_ARG=
set /p GRID_CODE=请选择 (1-12, 0返回):

if "%GRID_CODE%"=="0" goto :menu
if "%GRID_CODE%"=="" set GRID_ARG=510300.SH
if "%GRID_CODE%"=="1"  set GRID_ARG=510300.SH
if "%GRID_CODE%"=="2"  set GRID_ARG=510500.SH
if "%GRID_CODE%"=="3"  set GRID_ARG=515800.SH
if "%GRID_CODE%"=="4"  set GRID_ARG=000001.SH
if "%GRID_CODE%"=="5"  set GRID_ARG=399001.SZ
if "%GRID_CODE%"=="6"  set GRID_ARG=000016.SH
if "%GRID_CODE%"=="7"  set GRID_ARG=000852.SH
if "%GRID_CODE%"=="8"  set GRID_ARG=000688.SH
if "%GRID_CODE%"=="9"  set GRID_ARG=000698.SH
if "%GRID_CODE%"=="10" set GRID_ARG=399006.SZ
if "%GRID_CODE%"=="11" set GRID_ARG=399673.SZ
if "%GRID_CODE%"=="12" set GRID_ARG=932000.SH
if not defined GRID_ARG (
    echo ? 无效选择，请重试。
    goto :grid
)

echo.
echo   请设置网格间距:
echo   ----------------
echo   [1] 2%% 对称网格    (窄间距·高频收割)
echo   [2] 4%% 对称网格    (宽间距·低频·减少踏空·推荐·默认)
echo.
set GRID_STRAT=
set /p GRID_STRAT=请选择 (1-2, 默认2):

rem 默认 = 4%% 对称网格；仅 [1] 切换为 2%%
set GRID_MODE=symmetric
set GRID_PCT_ARG=--grid-pct 0.04
set GRID_SELL_ARG=
if "%GRID_STRAT%"=="1" set GRID_PCT_ARG=--grid-pct 0.02

rem ── 对照买入持有：默认常开（评判网格好坏的标尺）──
set GRID_BH_ARG=--compare-buyhold

rem ── 高级选项默认全部关闭 ──
set GRID_TF_ARG=
set GRID_SL_ARG=
set GRID_VF_ARG=
set GRID_CV_ARG=
rem ── 中枢模式/初始仓位默认(稳健通用) ──
set GRID_CM_ARG=--center-mode fixed
set GRID_CMW_ARG=
set GRID_IP_ARG=

rem ── 网格中枢模式：主流程主问题（决定买卖线是否随市场浮动，是"零成交"痛点的核心开关）──
echo.
echo   网格中枢模式[决定买卖线如何锚定]:
echo   [1] 固定价位(旧版·锚定首日收盘价·通用稳健·默认)
echo   [2] 滚动MA中枢(买卖线随中枢浮动·深跌能补仓/反弹能收割·震荡深跌市更优)
set GRID_CM=
set /p GRID_CM=请选择 (1-2, 默认1固定):
if "%GRID_CM%"=="2" set GRID_CM_ARG=--center-mode ma
if "%GRID_CM%"=="2" set /p GRID_CMW=  MA窗口(日, 默认60):
if "%GRID_CM%"=="2" if "%GRID_CMW%"=="" set GRID_CMW=60
if "%GRID_CM%"=="2" set GRID_CMW_ARG=--center-ma-window %GRID_CMW%

rem ── 初始仓位：主流程主问题（越低深跌时购买力越强；中枢模式配低仓位效果最佳）──
echo.
echo   初始仓位[预留干火药·越低深跌时购买力越强]:
echo   [1] 50%% (默认·上涨市占优)   [2] 30%%   [3] 20%% (震荡/深跌市占优·配中枢模式最佳)
set GRID_IP=
set /p GRID_IP=请选择 (1-3, 默认1=50%%):
if "%GRID_IP%"=="2" set GRID_IP_ARG=--init-pos 0.30
if "%GRID_IP%"=="3" set GRID_IP_ARG=--init-pos 0.20

echo.
echo   [A] 高级选项 (进阶间距 / 趋势过滤 / 止损 / 波动率关网 / 情景提示)
echo       默认全部关闭，回车直接跑；输入 A 逐项设置。
echo.
set GRID_ADV=
set /p GRID_ADV=请选择 (回车跳过 / A 进入高级):
if /i "%GRID_ADV%"=="A" goto :grid_advanced
goto :grid_run

:grid_advanced
echo.
echo   [进阶间距] 覆盖上面的间距选择:
echo   [1] 保持不变   [2] 3%% 中网格   [3] 非对称 2/8%% (锚定成本·浮亏不割·浮盈宽卖)
set GRID_ADVPCT=
set /p GRID_ADVPCT=请选择 (1-3, 默认1保持):
if "%GRID_ADVPCT%"=="2" set GRID_PCT_ARG=--grid-pct 0.03
if "%GRID_ADVPCT%"=="3" set GRID_MODE=asymmetric
if "%GRID_ADVPCT%"=="3" set GRID_PCT_ARG=--grid-pct 0.02
if "%GRID_ADVPCT%"=="3" set GRID_SELL_ARG=--sell-pct 0.08

echo.
echo   趋势过滤[站上250日均线只持有不卖]:
echo   注意: 会抑制卖出, 单边涨市≈买入持有
echo   [1] 关闭[纯网格]   [2] 开启
set GRID_TF=
set /p GRID_TF=请选择 (1-2, 默认1关闭):
if "%GRID_TF%"=="2" set GRID_TF_ARG=--trend-filter

echo.
echo   总止损线[净值峰回撤超阈值即停止新建网格·仅持有]:
echo   [1] 关闭   [2] 开启 (-30%%)
set GRID_SL=
set /p GRID_SL=请选择 (1-2, 默认1关闭):
if "%GRID_SL%"=="2" set GRID_SL_ARG=--stop-loss -0.30

echo.
echo   波动率关网[短期波动突升暂停网格·恢复自动重启]:
echo   [1] 关闭   [2] 开启 (20日年化波动^^>基准×2.5)
set GRID_VF=
set /p GRID_VF=请选择 (1-2, 默认1关闭):
if "%GRID_VF%"=="2" set GRID_VF_ARG=--vol-filter

echo.
echo   情景提示[幸存者偏差/未来收益下行/集中度认知]:
echo   [1] 关闭   [2] 开启
set GRID_CV=
set /p GRID_CV=请选择 (1-2, 默认1关闭):
if "%GRID_CV%"=="2" set GRID_CV_ARG=--caveat

goto :grid_run

:grid_run
echo.
echo   配置: 每格5000元 ^| 中枢%GRID_CM_ARG% %GRID_CMW_ARG% ^| 仓位%GRID_IP_ARG% ^| 对照买入持有:常开
echo   (仓位/中枢为空=默认 初始50%%仓位·固定价位网格)
echo   区间: %BACKTEST_START% ~ %BACKTEST_END%
echo.
echo   正在运行...
"venv_ml\Scripts\python.exe" run_grid_backtest.py %GRID_ARG% %BACKTEST_START% %BACKTEST_END% %GRID_PCT_ARG% --mode %GRID_MODE% %GRID_SELL_ARG% %GRID_TF_ARG% %GRID_SL_ARG% %GRID_VF_ARG% %GRID_BH_ARG% %GRID_CV_ARG% %GRID_CM_ARG% %GRID_CMW_ARG% %GRID_IP_ARG% --per-grid 5000
echo.
echo   [返回主菜单]
pause
goto :menu

:sc_rotation
cls
echo.
echo ================
echo   小市值轮动策略 · 版本选择
echo ================
echo.
echo   请选择要运行的小市值策略版本：
echo   [1] 周频止损版 (原 sc_rotation · 周频换仓 + 三层止损 + VaR仓位缩放)
echo   [2] Kara 纯最小市值月频版 (sc_kara · 月频等权 + 零过滤器 + 含科创板)
echo   [0] 返回主菜单
echo.
set /p SC_VER=  选择 (1/2/0):
if "%SC_VER%"=="1" goto :sc_rotation_run
if "%SC_VER%"=="2" goto :sc_kara
if "%SC_VER%"=="0" goto :menu
goto :sc_rotation

:sc_rotation_run
cls
echo.
echo ================
echo   小市值轮动策略[全市场小市值·中证2000风格]
echo ================
echo.
echo   规则：沪深两市(剔除北交所)按流通市值升序取最小约2000只作候选宇宙，每周二等权换仓，持有其中最小N只。
echo   已对齐《投研手册》：无前视(前日快照选股·次日开盘成交) / 流动性过滤 / 三层止损 / 退市清仓。
echo.
echo   当前配置:
echo     回测区间: %BACKTEST_START% ~ %BACKTEST_END%
echo     总资金: %TOTAL_CAPITAL% 元
echo.
echo   请设置持仓只数[默认7，回车=7]:
set SC_HOLD=7
set /p SC_HOLD_INPUT=  持仓只数:
if not "%SC_HOLD_INPUT%"=="" set SC_HOLD=%SC_HOLD_INPUT%
echo.
echo   选择「选股宇宙」？[说明：本策略取候选池内市值最小N只，故"合并"无意义——合并池的最小N只恒等于其中较小档]
echo   [1] 中证2000风格[默认，全市场流通市值最小2000只·含微盘尾·剔除北交所]
echo   [2] 纯创业板 (300/301)
echo   [4] 中证1000风格[市值排名801-1800·剔除微盘尾·剔除北交所，更稳健]
set SC_POOL=zz2000
set /p SC_POOL_INPUT=  选择 (1/2/4, 默认1):
if "%SC_POOL_INPUT%"=="2" set SC_POOL=cyb
if "%SC_POOL_INPUT%"=="4" set SC_POOL=zz1000
echo.
echo   请选择运行方式？
echo   [1] 单回测（可选小/中/大桶，出明细+HTML）
echo   [2] 三桶对比（小/中/大桶一起跑，只生成HTML对比图）
set SC_RUN=single
set /p SC_RUN_INPUT=  选择 (1/2, 默认1单回测):
if "%SC_RUN_INPUT%"=="2" set SC_RUN=compare
echo.
if "%SC_RUN%"=="compare" goto :sc_compare
echo   选择「市值分位桶」？(替代默认最小桶)
echo   [1] 最小桶[默认·最小N只]
echo   [2] 中桶[宇宙40%%分位档·跳过最小800只取其后N只]
echo   [3] 大桶[宇宙最大N只]
set SC_BUCKET=
set /p SC_BUCKET_INPUT=  选择 (1/2/3, 默认1最小桶):
if "%SC_BUCKET_INPUT%"=="2" set SC_BUCKET=mid
if "%SC_BUCKET_INPUT%"=="3" set SC_BUCKET=large
echo.
echo   是否开启「1月/4月空仓」(年报/一季报窗口避险)？
echo   [1] 关闭[默认，全年持有]   [2] 开启
set SC_EMPTY=0
set /p SC_EMPTY_INPUT=  选择 (1/2, 默认1):
if "%SC_EMPTY_INPUT%"=="2" set SC_EMPTY=1
echo.
echo   是否开启「三层止损」？
echo    层1 单票自买入价回撤^>12%% 清仓该股
echo    层2 中证2000单日跌幅^>6.6%% 清仓全部当周空仓
echo    层3 昨涨停今炸板(周二开盘低于昨涨停价) 清仓保利润
echo   [1] 关闭[默认]   [2] 开启
set SC_SL=0
set /p SC_SL_INPUT=  选择 (1/2, 默认1):
if "%SC_SL_INPUT%"=="2" set SC_SL=1
echo.
echo   输出格式？(明细=逐笔文本+CSV, 你一定看得到; HTML=净值曲线辅助可视化)
echo   [1] 明细 + HTML[默认·推荐: 两者都给, 不只甩HTML]
echo   [2] 仅明细(文本+CSV, 不生成HTML)
echo   [3] 仅HTML(不导出明细)
set SC_OUT=
set /p SC_OUT_INPUT=  选择 (1/2/3, 默认1明细+HTML):
if "%SC_OUT_INPUT%"=="2" set SC_OUT=--sc-no-html
if "%SC_OUT_INPUT%"=="3" set SC_OUT=--sc-no-detail

echo.
echo   VaR 仓位缩放（小市值=高波动凶策略，默认开启 95%%·目标回撤15%%·N=5）:
echo   [0] 关闭   [1] 95%%置信(默认)   [2] 99%%置信
set /p SC_VARSEL=请选择 (0/1/2, 回车=1·95%%开启):
if "%SC_VARSEL%"=="0" (set SC_VAR=) else if "%SC_VARSEL%"=="2" (set SC_VAR=--var-control 99 --var-maxdd 15 --var-n 5) else (set SC_VAR=--var-control 95 --var-maxdd 15 --var-n 5)
echo.
set SC_ARGS=--hold-count %SC_HOLD% --sc-pool-mode %SC_POOL% --sc-mode single
if not "%SC_BUCKET%"=="" set SC_ARGS=%SC_ARGS% --sc-bucket %SC_BUCKET%
if "%SC_EMPTY%"=="1" set SC_ARGS=%SC_ARGS% --empty-jan-apr
if "%SC_SL%"=="1" set SC_ARGS=%SC_ARGS% --stop-loss
echo   配置: 宇宙=%SC_POOL% 持仓%SC_HOLD%只 桶=%SC_BUCKET%^空仓1/4月=%SC_EMPTY% 止损=%SC_SL%
echo   正在运行...
echo.
"venv_ml\Scripts\python.exe" run_backtest.py --source sc_rotation --start-date %BACKTEST_START% --end-date %BACKTEST_END% --capital %TOTAL_CAPITAL% %SC_ARGS% %SC_OUT% %SC_VAR%
echo.
echo   [返回小市值策略菜单]
pause
goto :sc_rotation

:sc_compare
set SC_ARGS=--hold-count %SC_HOLD% --sc-pool-mode %SC_POOL% --sc-mode size_quintile
echo   配置: 宇宙=%SC_POOL% 持仓%SC_HOLD%只 模式=三桶对比(仅生成HTML, 不含明细)
echo   正在运行...
echo.
"venv_ml\Scripts\python.exe" run_backtest.py --source sc_rotation --start-date %BACKTEST_START% --end-date %BACKTEST_END% --capital %TOTAL_CAPITAL% %SC_ARGS% --sc-no-detail
echo.
echo   [返回小市值策略菜单]
pause
goto :sc_rotation

:sc_kara
cls
echo.
echo ================
echo   Kara 小市值轮动策略[全市场最小流通市值·月频等权·零过滤器]
echo ================
echo.
echo   规则：全市场(剔除老三板/北交所·含科创板)按流通市值升序取最小N只，每月第5交易日等权换仓，零过滤器、无止损。
echo   来源：B站 UP Kara说量化(BV15X5t6rE12) 极简最小市值策略的平台适配版，已用平台成本/滑点/历史印花税模型复现。
echo   实测(2020-2026)：含科创板 +80.47%% 跑赢中证2000(+9.53pp)；远不及原版声称135%%(其含北交所+幸存者偏差+窗口美化)。
echo.
echo   当前配置:
echo     回测区间: %BACKTEST_START% ~ %BACKTEST_END%
echo.
echo   请设置持仓只数[默认20，回车=20]:
set KARA_HOLD=20
set /p KARA_HOLD_INPUT=  持仓只数:
if not "%KARA_HOLD_INPUT%"=="" set KARA_HOLD=%KARA_HOLD_INPUT%
echo.
echo   选择「选股宇宙」？
echo   [1] 中证2000风格[默认，全市场流通市值最小2000只·含微盘尾·含科创板]
echo   [2] 纯创业板 (300/301)
echo   [4] 中证1000风格[市值排名801-1800·剔除微盘尾]
set KARA_POOL=zz2000
set /p KARA_POOL_INPUT=  选择 (1/2/4, 默认1):
if "%KARA_POOL_INPUT%"=="2" set KARA_POOL=cyb
if "%KARA_POOL_INPUT%"=="4" set KARA_POOL=zz1000
echo.
echo   是否「剔除科创板(688)」？(用于对照加科创板选股的贡献，默认含科创板)
echo   [1] 含科创板[默认]   [2] 剔除科创板
set KARA_X688=
set /p KARA_X688_INPUT=  选择 (1/2, 默认1含688):
if "%KARA_X688_INPUT%"=="2" set KARA_X688=--kara-exclude-688
echo.
echo   配置: 宇宙=%KARA_POOL% 持仓%KARA_HOLD%只 剔除688=%KARA_X688%
echo   正在运行...
echo.
"venv_ml\Scripts\python.exe" run_backtest.py --source sc_kara --start-date %BACKTEST_START% --end-date %BACKTEST_END% --hold-count %KARA_HOLD% --sc-pool-mode %KARA_POOL% %KARA_X688%
echo.
echo   [返回小市值策略菜单]
pause
goto :sc_rotation

:macd_timing
cls
echo.
echo ========================================
echo   MACD 择时策略 [逐股DIF^>DEA·无KDJ]（月度调仓入口）
echo ========================================
echo.
echo   逐股 timing：每只股票独立按 MACD 状态多/空（DIF^>DEA=多头区，金叉进/死叉出），
echo   无 KDJ 按钮。顶替已退休的 macd_kdj 逐股择时插件。
echo   股票池: %STOCK_POOL% （跟随全局，主菜单[5]设置）
echo   记账口径与平台一致（hfq 后复权）。指数门控默认关（MA200 太慢、牛市易长期空仓）；
echo   如需风控可加 --regime（CLI 用：run_macd_timing.py ... --regime --ma 200）。
echo   回测区间: %BACKTEST_START% ~ %BACKTEST_END%
echo.
if "%STOCK_POOL%"=="" set STOCK_POOL=all
echo   正在运行...（%STOCK_POOL% 池，逐股独立子账户等权）
"venv_ml\Scripts\python.exe" run_macd_timing.py %BACKTEST_START% %BACKTEST_END% --pool %STOCK_POOL% --capital %TOTAL_CAPITAL%
echo.
echo   [返回主菜单]
pause
goto :menu
