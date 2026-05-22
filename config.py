"""
配置文件 - 多因子选股系统
通过修改本文件的参数，然后运行 python run_selection.py 即可

修改后保存，直接运行 run_selection.py 即可生效
"""

# ============ 数据源配置 ============
# DataFetcher 初始化参数
# primary_source: "local_db"(本地数据库), "akshare", "tushare"
# local_db_path: 本地 SQLite 数据库路径
DATA_FETCHER = {
    "primary_source": "local_db",
    "tushare_token": None,
    "use_akshare_backup": True,
    "use_tushare_backup": False,
    "local_db_path": "D:/tu-shareData/astock_daily.db"
}

# ============ 因子计算器配置 ============
# FactorCalculator 初始化参数
# 启用哪些因子类型（True/False）
FACTOR_CALCULATOR = {
    "enable_quality": True,      # 质量因子
    "enable_momentum": True,     # 动量因子
    "enable_technical": True,    # 技术因子（TA-Lib）
    "enable_volatility": True,    # 低波动因子
    "enable_money_flow": False,  # 资金流因子（历史数据时不准确）
}

# ============ 因子处理器配置 ============
# FactorProcessor 初始化参数
FACTOR_PROCESSOR = {
    "winsorize_limits": (0.01, 0.99),  # 去极值范围
    "standardization_method": "zscore"  # 标准化方法: "zscore" 或 "rank"
}

# ============ 选股器配置 ============
# StockSelector 初始化参数
STOCK_SELECTOR = {
    "top_n": 20,          # 选股数量（TOP N）
    "min_score": 0.0,      # 最低得分阈值
    "scoring_method": "equal_weight"  # 打分方法: "equal_weight" 或 "custom_weight"
}

# ============ 选股日期配置 ============
# 选股日期（格式：YYYYMMDD）
# 用于计算因子和选股的基准日期
SELECTION_DATE = "20230101"  # 默认：2023-01-01

# ============ 行业动量因子配置 ============
# 是否在选股时计算并添加行业动量因子
ENABLE_INDUSTRY_MOMENTUM = True   # 启用行业动量因子（简化版）
INDUSTRY_MOMENTUM_LOOKBACK = 6    # 回看月数（默认6个月）

# ============ 回测配置 ============
BACKTEST = {
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "total_capital": 200000,  # 初始资金（元）
    "benchmark": "hs300"         # 基准指数
}

# ============ 回测策略选择配置 ============
# 要测试的策略列表（留空 = 测试所有策略）
# 可选值: "买入持有", "双均线", "海龟", "MACD/RSI", "MACD/KDJ", "月定投", "周定投", "智能切换", "RSI", "布林带"
# 示例:
#   - 只测试买入持有: ["买入持有"]
#   - 测试多个策略: ["买入持有", "双均线", "海龟"]
#   - 测试所有策略: [] (空列表)
STRATEGIES_TO_TEST = ["买入持有", "海龟", "MACD/KDJ", "RSI", "布林带"]  # 回测5个策略

# ============ 输出配置 ============
OUTPUT = {
    "output_dir": "data/results",
    "export_csv": True,
    "print_top_n": 20,        # 控制台打印前 N 只
    "save_log": True,
    "log_file": "stock_selection.log"
}
