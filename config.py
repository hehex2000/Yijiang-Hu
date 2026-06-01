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
    "primary_source": "local_db",       # 改为本地数据库优先（避免网络问题）

    "local_db_path": r"D:\tu-sharedata\astock_daily.db",

    "tushare_token": "761165a821532fe625262d6b33e144b9859a887c004acbcb981c319b",

    "use_akshare_backup": True,      # AkShare作为备用

    "use_tushare_backup": True,      # Tushare作为备用（付费账户，数据更全）
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
    "date": "20170102",

    # 股票池: "hs300" | "zz500" | "zz800" | "all"
    "stock_pool": "hs300",

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
        "value_score": 0.25,
        "growth_score": 0.15,
        "quality_score": 0.10,
        "momentum_score": 0.20,
        "technical_score": 0.10,
        "volatility_score": 0.15,
        "money_flow_score": 0.05,
    },
}


# ============================================================
# 五、回测配置
# ============================================================

BACKTEST = {

    # 回测时间范围
    "start_date": "20170102",
    "end_date": "20171229",

    # 每只股票初始资金（元）
    "initial_capital": 100000,

    # 基准指数（用于对比）
    "benchmark": "000300.SH",

    # 股票来源: "selection" | "csv" | "manual"
    #   "selection": 自动运行选股 -> 回测（同时保存CSV）
    #   "csv":       从 stocks_file 读取选股结果（无需重跑选股）
    #   "manual":    使用下面 stocks_manual 列表
    "stocks_source": "selection",

    # CSV 文件路径（stocks_source="csv" 时生效）
    "stocks_file": "",

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
        "take_profit": 0.50,
        "stop_loss": 0.15,
    },
    "bollinger": {
        "enabled": False,
        "name": "布林带策略",
        "period": 20,
        "std": 2,
        "take_profit": 0.50,
        "stop_loss": 0.15,
    },
    "turtle": {
        "enabled": True,
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
    "rsi_trend": {
        "enabled": True,
        "name": "RSI趋势策略",
        "rsi_period": 14,        # RSI计算周期
        "rsi_center": 50,        # RSI中轴线（上穿买入，下穿卖出）
        "take_profit": 0.50,     # 止盈 +50%
        "stop_loss": 0.15,       # 止损 -15%
        "position_mode": "half",   # 半仓操作（50%资金）
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
    "dir": "outputs",
    "save_csv": False,       # 禁用CSV保存，直接打印结果
    "print_top_n": 20,
}
