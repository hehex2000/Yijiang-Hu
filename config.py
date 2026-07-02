# -*- coding: utf-8 -*-
"""
多因子选股 + 回测系统 - 统一配置文件
========================================
使用方法：修改本文件参数 -> 运行 python run_backtest.py
所有选股、回测、策略参数均在此配置，无需改代码
"""

# ============================================================
# 一、全局配置（所有策略共享）
# ============================================================

GLOBAL = {
    # 回测时间范围
    "backtest_start": "20230102",
    "backtest_end": "20231231",

    # 选股日期（自动使用回测开始日前一交易日，此处仅作fallback）
    "selection_date": "20260102",

    # 股票池: "hs300" | "zz500" | "zz800" | "all"
    "stock_pool": "zz800",

    # 选股数量
    "top_n": 5,
}

# ============================================================
# 二、数据源配置
# ============================================================

DATA = {

    # 主数据源: "local_db" | "akshare" | "tushare"
    # 行情数据优先本地DB，财务/估值数据优先Tushare
    "primary_source": "local_db",      # ← 优先使用本地数据库

    "local_db_path": r"C:\Users\99395\WorkBuddy\2026-07-01-10-38-55\astock_daily.db",

    "tushare_token": "761165a821532fe625262d6b33e144b9859a887c004acbcb981c319b",

    "use_akshare_backup": False,      # ← 禁用AkShare备份（用户确认不通）

    "use_tushare_backup": True,      # ← 启用Tushare备份（替代本地数据库）
}


# ============================================================
# 二、选股配置
# ============================================================

SELECTION = {

    # 是否先执行选股
    #   True  = 自动选股 -> 回测
    #   False = 使用下面 stocks_manual 中的股票直接回测
    "enabled": True,

    # 选股基准日期（YYYYMMDD，自动使用回测开始日的前一交易日）
    # 【已自动计算】无需手动修改！由 run_backtest.py 在执行 run_selection() 时自动
    # 从数据库查询 BACKTEST["start_date"] 前最近交易日并覆盖此值。
    "date": GLOBAL["selection_date"],  # ← 从GLOBAL读取

    # 股票池: "hs300" | "zz500" | "zz800" | "all"
    "stock_pool": GLOBAL["stock_pool"],  # ← 从GLOBAL读取

    # 选股数量
    "top_n": GLOBAL["top_n"],  # ← 从GLOBAL读取

    # 排除ST / 停牌
    "exclude_st": True,
    "exclude_suspended": True,
}

# 选股结果保存路径（修改为你指定的目录）
SELECTION["output_file"] = f"outputs/selection_{SELECTION['date']}.csv"


# ============================================================
# 三、因子计算器配置
# ============================================================

FACTOR_CALCULATOR = {
    
    "enable_quality": True,       # 质量因子
    
    "enable_momentum": True,      # 动量因子
    
    "enable_technical": True,     # 技术因子（TA-Lib）
    
    "enable_volatility": True,     # 低波动因子
    
    "enable_money_flow": False,    # 资金流因子（历史回测建议关）
    
    "enable_industry_momentum": True,   # 行业动量因子（需先填充 industry_momentum 表）
    
    "enable_risk": True,         # 风险因子（夏普比率、贝塔等）
}


# ============================================================
# 四、因子处理器配置（权重）
# ============================================================

FACTOR_PROCESSOR = {

    "standardization_method": "zscore",

    "winsorize_limits": (0.01, 0.99),

    # 加权方式: "equal" | "custom"
    "weighting_method": "custom",

    # 各大类因子权重（总和自动归一化，无需凑整）
    "weights": {
        "value_score": 0.20,
        "growth_score": 0.10,
        "quality_score": 0.10,
        "momentum_score": 0.15,
        "technical_score": 0.10,
        "volatility_score": 0.15,
        "money_flow_score": 0.0,
        "industry_momentum_score": 0.10,
        "risk_score": 0.10,       # 风险因子（夏普比率、贝塔等）
    },
}


# ============================================================
# 五、回测配置
# ============================================================

BACKTEST = {
    
    # 回测时间范围（从 GLOBAL 读取）
    "start_date": GLOBAL["backtest_start"],  # ← 从GLOBAL读取
    "end_date": GLOBAL["backtest_end"],      # ← 从GLOBAL读取

    # 每只股票初始资金（元）
    "initial_capital": 100000,

    # 基准指数（用于对比）
    "benchmark": "000906.SH",  # ← 中证800指数

    # ═══════════════════════════════════════════════════════
    # 股票来源（可被命令行参数覆盖）
    # ═══════════════════════════════════════════════════════
    #
    # 【推荐】使用命令行参数（无需修改此文件）：
    #   python run_backtest.py                # 使用下面默认配置
    #   python run_backtest.py --source multi # 使用最新 multi-*.csv
    #   python run_backtest.py --source ml    # 使用最新 ml-*.csv
    #   python run_backtest.py --list         # 列出所有可用 CSV 文件
    #
    # 【或】修改下面配置（被 --source 参数覆盖）：
    #   "selection": 自动运行选股 -> 回测（同时保存CSV）
    #   "csv":       从 stocks_file 读取选股结果（无需重跑选股）
    #   "manual":    使用下面 stocks_manual 列表
    "stocks_source": "selection",  # ← 先选股再回测

    # CSV 文件路径（stocks_source="csv" 时生效）
    # 命名规则:
    #   ML选股:     ml-YYYYMMDD-topN.csv    (如 ml-20260603-top20.csv)
    #   多因子选股: multi-YYYYMM-xxx.csv   (如 multi-202606-selection.csv)
    "stocks_file": "data/results/ml-20260603-top20.csv",  # ← ML选股结果（示例）

    # 手动指定股票（stocks_source="manual" 时生效）
    "stocks_manual": [],
}


# ============================================================
# 六、回测策略配置
# 每个策略 enabled=True/False 控制是否启用
# 策略参数可自由调整
# ============================================================

STRATEGIES = {
    "buy_hold": {
        "enabled": True,
        "name": "买入持有",
        # ATR动态止损（默认禁用，买入持有策略不建议启用止损）
        "use_atr_stop": False,       # ← 禁用ATR动态止损（恢复纯买入持有）
        "atr_period": 14,            # ATR计算周期
        "atr_mult": 3.0,            # 初始止损倍数
        "trail_mult": 3.0,           # 追踪止损倍数
    },
    "rsi": {
        "enabled": True,
        "name": "RSI策略（优化参数）",
        "rsi_period": 10,         # 更敏感，更快反应
        "oversold": 40,            # RSI < 40 超卖买入（增加交易次数）
        "overbought": 60,          # RSI > 60 超买卖出（增加交易次数）
        "take_profit": 0.50,       # 止盈 +50%
        "stop_loss": 0.15,         # 止损 -15%（ATR启用时失效）
        # ── ATR 动态止损（可选，默认关闭）──
        "atr_stop_loss": True,     # True=启用ATR止损，False=使用固定stop_loss
        "atr_period": 14,          # ATR 计算周期
        "atr_mult": 3.0,           # 初始止损倍数
        "trail_mult": 3.0,         # 追踪止损倍数
    },
    "macd_kdj": {
        "enabled": True,
        "name": "MACD/KDJ策略",
        "fast": 12, "slow": 26, "signal": 9,
        "kdj_period": 9,
        "take_profit": 0.20,
        "stop_loss": 0.10,
        # ── ATR 动态止损（可选，默认关闭）──
        "atr_stop_loss": True,
        "atr_period": 14,
        "atr_mult": 3.0,
        "trail_mult": 3.0,
    },
    # ── bollinger 布林带策略（已改造-加入凯利仓位）──
    "bollinger": {
        "enabled": True,
        "name": "布林带策略（凯利仓位版）",
        # ── 布林带参数 ──
        "bb_period": 20,
        "bb_std": 2.5,
        # ── 止盈止损 ──
        "take_profit": 0.50,
        "stop_loss": 0.15,
        # ── 仓位模式（use_kelly=False 时生效）──
        "position_mode": "half",        # "half"=半仓, "full"=全仓
        # ── ATR 动态止损 ──
        "atr_stop_loss": True,
        "atr_period": 14,
        "atr_mult": 3.0,
        "trail_mult": 3.0,
        # ── 凯利公式仓位（use_kelly=True 时生效）──
        "use_kelly": True,
        "kelly_win_rate": 0.50,         # BB策略胜率保守估计
        "kelly_win_loss_ratio": 1.8,     # 盈亏比（赚1.8元亏1元）
        "kelly_fraction": 0.5,           # 半凯利
        "kelly_max_position": 0.20,      # 最大仓位20%
        "kelly_min_position": 0.05,      # 最小仓位5%
        "kelly_safety_discount": 0.8,    # 参数不确定性再打8折
    },
    # ── BB+RSI组合策略（视频版-带宽挤压+半凯利）──
    "bb_rsi_combo": {
        "enabled": False,
        "name": "BB+RSI组合策略（带宽挤压+半凯利）",
        # ── 布林带参数 ──
        "bb_period": 20,
        "bb_std": 2.0,
        # ── RSI参数 ──
        "rsi_period": 14,
        "rsi_oversold": 35,             # RSI超卖（入场确认，放宽至35）
        "rsi_overbought": 70,           # RSI超买（出场确认）
        # ── 带宽挤压参数 ──
        "squeeze_lookback": 50,         # 挤压检测回看期
        "squeeze_threshold": 1.20,      # 挤压阈值（带宽<历史最低×1.20，放宽至20%以内）
        # ── 止盈止损 ──
        "stop_loss": 0.05,              # 5%硬止损
        "take_profit": 0.30,            # 30%止盈
        # ── ATR 动态止损 ──
        "atr_stop_loss": True,
        "atr_period": 14,
        "atr_mult": 2.0,                # 收紧止损（2倍ATR）
        "trail_mult": 2.0,              # 收紧追踪止损
        # ── 凯利公式仓位 ──
        "use_kelly": True,
        "kelly_win_rate": 0.55,         # 估计胜率55%（BB+RSI双确认更高）
        "kelly_win_loss_ratio": 1.5,     # 估计盈亏比
        "kelly_fraction": 0.5,           # 半凯利
        "kelly_max_position": 0.20,      # 最大仓位20%
        "kelly_min_position": 0.05,      # 最小仓位5%
        "kelly_safety_discount": 0.8,    # 参数不确定性再打8折
    },
    # ── RSI+布林带双确认策略（视频版-RSI超买超卖+布林带过滤+半凯利）──
    "rsi_bb_dual": {
        "enabled": True,
        "name": "RSI+布林带双确认策略（A股优化参数+半凯利）",
        # ── RSI参数（A股适配：9天比14更敏感）──
        "rsi_period": 9,                # RSI周期（A股震荡市推荐9天）
        "rsi_oversold": 30,             # RSI超卖阈值（入场确认）
        "rsi_overbought": 70,           # RSI超买阈值（出场确认）
        "rsi_center": 50,               # RSI中轴线
        # ── 布林带参数 ──
        "bb_period": 20,
        "bb_std": 2.0,
        # ── 止盈止损 ──
        "stop_loss": 0.05,              # 5%硬止损
        "take_profit": 0.20,            # 20%止盈
        # ── ATR 动态止损 ──
        "atr_stop_loss": True,
        "atr_period": 14,
        "atr_mult": 2.0,
        "trail_mult": 2.0,
        # ── 凯利公式仓位 ──
        "use_kelly": True,
        "kelly_win_rate": 0.53,         # RSI策略回测胜率约53%
        "kelly_win_loss_ratio": 1.5,
        "kelly_fraction": 0.5,           # 半凯利
        "kelly_max_position": 0.20,
        "kelly_min_position": 0.05,
        "kelly_safety_discount": 0.8,
    },
    # ── 均值回归策略 ──
    "mean_reversion": {
        "enabled": True,
        "name": "均值回归策略（视频版）",
        "bb_period": 20,           # 布林带周期
        "bb_std": 2.0,            # 布林带标准差倍数
        "rsi_period": 14,          # RSI计算周期
        "rsi_oversold": 30,        # RSI超卖阈值（入场）
        "rsi_overbought": 70,      # RSI超买阈值（出场）
        "zscore_threshold": 2.0,   # Z-Score阈值
        "band_width_ma": 20,        # 布林带宽度MA周期（检测喇叭口）
        "stop_loss": 0.03,         # 3%硬止损（防止均值永久下移）
        # ── ATR 动态止损（可选，默认关闭）──
        "atr_stop_loss": True,
        "atr_period": 14,
        "atr_mult": 3.0,
        "trail_mult": 3.0,
        "position_mode": "half",     # "half"=半仓,"full"=全仓
    },
    "turtle": {
        "enabled": False,  # ← 禁用简化版，使用完整版
        "name": "海龟策略（双周期+ATR动态风控）",
        # ── 双周期趋势识别（海龟原版）──
        "short_period": 20,       # 短期系统：20日高点突破
        "long_period": 55,        # 长期系统：55日高点突破
        "short_exit_period": 10,   # 短期离场：10日低点跌破
        "long_exit_period": 20,    # 长期离场：20日低点跌破
        # ── ATR 波动量化 ──
        "atr_period": 14,         # ATR 计算周期
        "risk_pct": 0.02,         # 单笔风险：总资金的 2%（A股高价股适配）
        "max_risk_per_day": 0.02,  # 单日最大亏损：总资金的 2%
        "max_pos_pct": 1.0,       # 单品种最大仓位：总资金的 100%（取消限制，由1%风险原则决定仓位）
        # ── 金字塔加仓（海龟原版）──
        "max_adds": 4,            # 最多加仓次数（海龟原版）
        "add_step_atr": 0.5,      # 加仓步长：0.5×ATR
        "pos_unit_decay": True,     # 加仓单位递减（True=海龟原版）
        # ── 动态止损（基于 ATR）──
        "stop_atr_mult": 4.0,     # 止损距离：4×ATR（与追踪止损一致）
        "trail_atr_mult": 4.0,    # 追踪止损：4×ATR（放宽，拿住趋势）
        # ── A股本土化过滤 ──
        "trend_filter": False,     # 临时关闭，验证是否是这个问题
        "volume_filter": False,    # 暂时关闭，验证策略逻辑
        "volume_ma_period": 20,    # 成交量 MA 周期
        "min_listed_days": 250,    # 上市不足 250 个交易日过滤
        # ── 系统选择 ──
        "use_short_system": True,   # 恢复双系统同时运行（海龟原版）
        "use_long_system": True,    # 启用长期系统
        "system_weight": [0.5, 0.5],  # 短期/长期系统资金分配权重
    },
    "turtle_full": {
        "enabled": True,   # ← 启用完整版海龟策略（与插件key匹配）
        "name": "海龟策略（完整版-双周期+ATR动态风控）",
        # ── 双周期趋势识别（海龟原版）──
        "short_period": 20,       # 短期系统：20日高点突破
        "long_period": 55,        # 长期系统：55日高点突破
        "short_exit_period": 10,   # 短期离场：10日低点跌破
        "long_exit_period": 20,    # 长期离场：20日低点跌破
        # ── ATR 波动量化 ──
        "atr_period": 14,         # ATR 计算周期
        "risk_pct": 0.02,         # 单笔风险：总资金的 2%
        "max_risk_per_day": 0.02,  # 单日最大亏损：总资金的 2%
        "max_pos_pct": 1.0,       # 单品种最大仓位：总资金的 100%
        # ── 金字塔加仓（海龟原版）──
        "max_adds": 4,            # 最多加仓次数（海龟原版）
        "add_step_atr": 0.5,      # 加仓步长：0.5×ATR
        "pos_unit_decay": True,     # 加仓单位递减（True=海龟原版）
        # ── 动态止损（基于 ATR）──
        "stop_atr_mult": 4.0,     # 止损距离：4×ATR
        "trail_atr_mult": 4.0,    # 追踪止损：4×ATR
        # ── A股本土化过滤 ──
        "trend_filter": False,     # 暂时关闭趋势过滤
        "volume_filter": False,    # 暂时关闭成交量过滤
        "volume_ma_period": 20,    # 成交量 MA 周期
        "min_listed_days": 250,    # 上市不足 250 个交易日过滤
        # ── 系统选择 ──
        "use_short_system": True,   # 启用短期系统
        "use_long_system": True,    # 启用长期系统
        "system_weight": [0.5, 0.5],  # 短期/长期系统资金分配权重
    },
    "rsi_trend": {
        "enabled": False,  # ← 禁用RSI趋势策略
        "name": "RSI趋势策略",
        "rsi_period": 14,        # RSI计算周期
        "rsi_center": 50,        # RSI中轴线（上穿买入，下穿卖出）
        "take_profit": 0.50,     # 止盈 +50%
        "stop_loss": 0.15,       # 止损 -15%
        # ── ATR 动态止损（可选，默认关闭）──
        "atr_stop_loss": True,
        "atr_period": 14,
        "atr_mult": 3.0,
        "trail_mult": 3.0,
        "position_mode": "half",   # 半仓操作（50%资金）
    },
    # ── 定投策略（已禁用）──
    "dca": {
        "enabled": False,  # ← 已禁用（2026-06-23）
        "name": "月定投策略",
        "amount_per_month": 5000,
        "take_profit": 0.30,
        "stop_loss": 0.20,
        "enable_tp_sl": True,
    },
    "weekly_dca": {
        "enabled": False,  # ← 已禁用（2026-06-23）
        "name": "周定投策略",
        "shares_per_week": 100,
        "take_profit": 0.30,
        "stop_loss": 0.10,
        "enable_tp_sl": True,
    },
    # ── 均线策略 ──
    "dual_ma": {
        "enabled": True,
        "name": "双均线策略（MA10/MA30）",
        "ma_short": 10,           # 短期均线周期（从20改为10）
        "ma_long": 30,            # 长期均线周期（从60改为30）
        "position_pct": 0.5,      # 单次买入仓位比例（50%）
        "take_profit": 0.30,      # 止盈 +30%
        "stop_loss": 0.10,        # 止损 -10%
        # ── ATR 动态止损（可选，默认关闭）──
        "atr_stop_loss": True,
        "atr_period": 14,
        "atr_mult": 3.0,
        "trail_mult": 3.0,
    },
    # ── 能量指标策略 ──
    "energy": {
        "enabled": False,
        "name": "能量指标策略（AR/BR/CR/VR）",
        "indicator_period": 26,     # 指标计算周期
        "buy_threshold": 100,      # 买入阈值（低于此值视为超卖）
        "sell_threshold": 150,     # 卖出阈值（高于此值视为过热）
        "position_pct": 0.5,       # 单次买入仓位比例（50%）
        "take_profit": 0.25,      # 止盈 +25%
        "stop_loss": 0.10,        # 止损 -10%
    },
    # ── 动态网格策略（VR 驱动）──
    "energy_grid": {
        "enabled": False,
        "name": "动态网格策略（VR驱动）",
        "grid_levels": 10,          # 网格层数
        "base_grid_pct": 0.05,    # 基础网格间距（5%）
        "low_vr_grid_pct": 0.03,  # VR<80时网格间距（3%）
        "high_vr_grid_pct": 0.08,  # VR>120时网格间距（8%）
        "vr_period": 26,           # VR计算周期
        "position_per_grid": 0.1,  # 每层网格仓位比例（10%）
    },
}


# ============================================================
# 七、行业动量因子（可选）
# ============================================================

INDUSTRY_MOMENTUM = {
    "enabled": True,           # 是否启用（需先填充 industry_momentum 表）
    "lookback_months": 6,
}


# ============================================================
# 八、输出配置
# ============================================================

OUTPUT = {
    # ══ 输出根目录
    "dir": "data/results",

    # ══ 选股结果子目录（多因子 / ML 选股结果）
    "selection_dir": "data/results/selection",

    # ══ 回测结果子目录（回测收益汇总等）
    "backtest_dir": "data/results/backtest",

    "save_csv": True,        # ← 选股时保存CSV
    "print_top_n": 20,
}


# ============================================================
# 九、价值投资策略配置
# ============================================================

VALUE_STRATEGY = {
    # 选股日期（格式: "YYYYMMDD"，必须是交易日）
    "date": GLOBAL["selection_date"],  # ← 从GLOBAL读取

    # 财务数据报告期（格式: "YYYYMMDD"，如 "20260331" 表示2026年一季报）
    "report_date": "20260331",  # 最新报告期（数据库已更新到2026-03-31）

    # 股票池: "hs300" | "zz500" | "zz800" | "all"
    "stock_pool": GLOBAL["stock_pool"],  # ← 从GLOBAL读取

    # 市值阈值（分位数，0.5 = 中位数）
    "market_cap_quantile": 0.5,

    # 流动比率阈值（分位数）
    "current_ratio_quantile": 0.5,

    # ROE阈值（分位数）
    "roe_quantile": 0.5,

    # 自由现金流要求（连续为正年数，0 = 不检查）
    "free_cash_flow_years": 5,

    # 营收增长率区间（max=0 表示无上限）
    "revenue_growth_min": 0.06,
    "revenue_growth_max": 0,

    # EPS区间（max=0 表示无上限）
    "eps_min": 0.08,
    "eps_max": 0,

    # 选股数量（0 = 不限制，按条件筛选）
    "top_n": GLOBAL["top_n"],  # ← 从GLOBAL读取

    # 输出配置
    "output_dir": "data/results/value_strategy",
    "output_file": "value_selection_{date}.csv",
}

# ============================================================
# 十、红利低波策略配置
# ============================================================

DIVIDEND_LOW_VOL = {
    # 选股日期（格式: "YYYYMMDD"，必须是交易日）
    # 【自动计算】由 run_backtest.py 在执行时自动设为回测开始日前一交易日
    "date": GLOBAL["selection_date"],  # ← 从GLOBAL读取

    # 股票池: "hs300" | "zz500" | "zz800" | "all"
    "stock_pool": GLOBAL["stock_pool"],  # ← 从GLOBAL读取

    # 选股数量（0 = 按条件筛选，不限制数量）
    "top_n": GLOBAL["top_n"],  # ← 从GLOBAL读取

    # ── 因子阈值 ──
    "dividend_yield_min": 0.0,  # 股息率下限（dv_ttm，0=不限制）
    "pe_min": 0,                   # PE_TTM 下限
    "pe_max": 50,                  # PE_TTM 上限
    "pb_min": 0,                   # PB 下限
    "pb_max": 10,                  # PB 上限

    # ── 波动率计算 ──
    "volatility_window": 120,       # 波动率计算窗口（交易日）

    # ── MACD 过滤 ──
    "macd_filter": True,            # 是否启用个GU MACD金叉过滤

    # ── 输出配置 ──
    "output_dir": "data/results/dividend_low_vol",
    "output_file": "dividend_low_vol_{date}.csv",
}

# ============================================================
# 十一、狗股策略配置（Dogs of the Market）
# 视频参考：《凯利公式——只看一个指标，每年操作一次》
# 核心理念：高股息率不是买分红，而是买错杀后的均值回归
# ============================================================

DOGS_OF_MARKET = {
    # 选股日期（格式: "YYYYMMDD"，必须是交易日）
    # 【自动计算】由 run_backtest.py 在执行时自动设为回测开始日前一交易日
    "date": GLOBAL["selection_date"],  # ← 从GLOBAL读取

    # 股票池: "hs300" | "zz500" | "zz800" | "all"
    "stock_pool": GLOBAL["stock_pool"],  # ← 从GLOBAL读取

    # 选股数量
    "top_n": GLOBAL["top_n"],  # ← 从GLOBAL读取

    # ── 因子阈值 ──
    "dividend_yield_percentile": 0.5,      # 股息率高于市场中位数（50%分位）
    "pb_percentile": 0.5,                  # PB低于市场中位数
    "min_dividend_years": 3,               # 连续分红年数要求
    "dividend_lookback_years": 3,          # 分红检查回溯年数

    # ── 调仓配置 ──
    "annual_rebalance_month": 4,           # 每年4月调仓（年报出完）

    # ── 波动率计算 ──
    "volatility_window": 120,               # 波动率计算窗口（交易日）

    # ── 输出配置 ──
    "output_dir": "data/results/dogs_of_market",
    "output_file": "dogs_of_market_{date}.csv",
}
