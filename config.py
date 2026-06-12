# -*- coding: utf-8 -*-
"""
多因子选股 + 回测系统 - 统一配置文件
========================================
使用方法：修改本文件参数 -> 运行 python run_backtest.py
所有选股、回测、策略参数均在此配置，无需改代码
"""

# ============================================================
# 一、数据源配置
# ============================================================

DATA = {

    # 主数据源: "local_db" | "akshare" | "tushare"
    # 行情数据优先本地DB，财务/估值数据优先Tushare
    "primary_source": "local_db",      # ← 优先使用本地数据库

    "local_db_path": r"D:\tu-shareData\astock_daily.db",  # 注意大小写：tu-shareData

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

    # 选股基准日期（YYYYMMDD，必须是交易日）
    "date": "20220103",  # ← 2022年初第一个交易日

    # 股票池: "hs300" | "zz500" | "zz800" | "all"
    "stock_pool": "zz500",  # ← 修改为中证500

    # 选股数量
    "top_n": 20,

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
    
    # 回测时间范围
    "start_date": "20220102",
    "end_date": "20220331",

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
    },
    "rsi": {
        "enabled": True,
        "name": "RSI策略（优化参数）",
        "rsi_period": 10,         # 更敏感，更快反应
        "oversold": 40,            # RSI < 40 超卖买入（增加交易次数）
        "overbought": 60,          # RSI > 60 超买卖出（增加交易次数）
        "take_profit": 0.50,       # 止盈 +50%
        "stop_loss": 0.15,         # 止损 -15%
    },
    "macd_kdj": {
        "enabled": True,
        "name": "MACD/KDJ策略",
        "fast": 12, "slow": 26, "signal": 9,
        "kdj_period": 9,
        "take_profit": 0.20,
        "stop_loss": 0.10,
    },
    "bollinger": {
        "enabled": True,  # ← 启用布林带策略
        "name": "布林带策略",
        "period": 20,
        "std": 2,
        "take_profit": 0.20,
        "stop_loss": 0.10,
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
        "position_mode": "half",   # 半仓操作（50%资金）
    },
    # ── 定投策略 ──
    "dca": {
        "enabled": True,
        "name": "月定投策略",
        "amount_per_month": 5000,  # 每月定投金额（元）
        "take_profit": 0.30,      # 止盈 +30%
        "stop_loss": 0.20,        # 止损 -20%
        "enable_tp_sl": True,      # 是否启用止盈止损
    },
    "weekly_dca": {
        "enabled": True,
        "name": "周定投策略",
        "shares_per_week": 100,   # 每周买入股数
        "take_profit": 0.30,      # 止盈 +30%
        "stop_loss": 0.10,        # 止损 -10%
        "enable_tp_sl": True,      # 是否启用止盈止损
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
    },
    # ── 能量指标策略 ──
    "energy": {
        "enabled": True,
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
        "enabled": True,
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
    "date": "20260102",  # 2026年第一个交易日

    # 财务数据报告期（格式: "YYYYMMDD"，如 "20260331" 表示2026年一季报）
    "report_date": "20260331",  # 最新报告期（数据库已更新到2026-03-31）

    # 股票池: "hs300" | "zz500" | "zz800" | "all"
    "stock_pool": "zz800",

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
    "top_n": 0,

    # 输出配置
    "output_dir": "data/results/value_strategy",
    "output_file": "value_selection_{date}.csv",
}
