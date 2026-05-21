"""
多因子选股系统 - 主程序
集成所有模块，执行完整的选股流程
"""

import sys
import os
import argparse
import pandas as pd
from loguru import logger
from datetime import datetime


def main():
    """主程序"""
    print("\n" + "="*70)
    print("多因子选股系统 (Multi-Factor Stock Selection System)")
    print("="*70 + "\n")
    
    # 1. 解析命令行参数
    args = parse_arguments()
    
    # 2. 读取配置文件
    config = load_config(args.config)
    
    # 如果命令行指定了top-n，则覆盖配置文件
    if args.top_n is not None:
        config.setdefault("selection", {})["top_n"] = args.top_n
        print(f"✓ 使用命令行参数 top_n={args.top_n}")
    
    # 如果命令行指定了output-dir，则覆盖配置文件
    if args.output_dir is not None:
        config.setdefault("output", {})["output_dir"] = args.output_dir
    
    # 3. 初始化日志
    setup_logging(config)
    logger.info("="*50)
    logger.info("多因子选股系统启动")
    logger.info("="*50)
    
    try:
        # 4. 初始化所有模块
        print("\n[1/6] 初始化模块...")
        data_fetcher, factor_calculator, factor_processor, stock_selector = init_modules(config, args)
        print("✓ 模块初始化完成\n")
        
        # 5. 获取股票池（沪深300成分股）
        print("[2/6] 获取股票池...")
        stock_pool = get_stock_pool(data_fetcher, config)
        print(f"✓ 获取到 {len(stock_pool)} 只股票\n")
        
        # 6. 计算因子
        print("[3/6] 计算因子...")
        factors_df = calculate_factors(stock_pool, data_fetcher, factor_calculator, args)
        print(f"✓ 因子计算完成: {len(factors_df)} 只股票\n")
        
        # 7. 处理因子（清洗、标准化、打分）
        print("[4/6] 处理因子...")
        processed_df = process_factors(factors_df, factor_processor, config)
        print(f"✓ 因子处理完成\n")
        
        # 8. 执行选股
        print("[5/6] 执行选股...")
        selected_df = select_stocks(processed_df, stock_selector, config)
        print(f"✓ 选股完成: {len(selected_df)} 只股票\n")
        
        # 9. 导出结果
        print("[6/6] 导出结果...")
        output_path = export_results(selected_df, stock_selector, config)
        print(f"✓ 结果已保存到: {output_path}\n")
        
        # 10. 打印TOP 10
        stock_selector.print_top_stocks(selected_df, n=10)
        
        logger.info("="*50)
        logger.info("选股完成!")
        logger.info("="*50 + "\n")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        raise
    
    print("="*70)
    print("程序执行完成!")
    print("="*70 + "\n")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="多因子选股系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                       # 使用默认配置运行
  python main.py -c config/my_config.yaml  # 使用自定义配置
  python main.py --top-n 30            # 选择TOP 30股票
  python main.py --output-dir output    # 指定输出目录
        """
    )
    
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="config/config.yaml",
        help="配置文件路径 (默认: config/config.yaml)"
    )
    
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="选择TOP N股票 (覆盖配置文件)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录 (覆盖配置文件)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="日志级别 (覆盖配置文件)"
    )
    
    parser.add_argument(
        "--factors",
        type=str,
        default=None,
        help="指定要计算的因子类型（逗号分隔）：value,growth,quality,momentum,technical。默认：value,growth"
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="多线程并行数（默认：5）"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """
    读取配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    import yaml
    
    # 默认配置
    default_config = {
        "data_source": {
            "primary": "akshare",
            "use_tushare_backup": False,
            "tushare_token": "",
        },
        "stock_pool": {
            "index_code": "000300",
            "filter_st": True,
            "filter_suspended": True,
        },
        "processing": {
            "winsorize_method": "winsorize",
            "winsorize_lower": 0.01,
            "winsorize_upper": 0.99,
            "missing_value_method": "fill_median",
            "standardization_method": "zscore",
            "neutralize_industry": True,
            "neutralize_market_cap": True,
        },
        "selection": {
            "weighting_method": "equal",
            "top_n": 50,
            "min_score": 0.0,
        },
        "output": {
            "output_dir": "data/results",
            "format": "csv",
            "save_intermediate": True,
        },
        "logging": {
            "level": "INFO",
            "file": "multi_factor_selection.log",
            "console": True,
        },
    }
    
    # 如果配置文件存在，则读取
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                user_config = yaml.safe_load(f)
            
            # 合并配置（用户配置覆盖默认配置）
            merge_config(default_config, user_config)
            logger.info(f"✓ 配置文件已加载: {config_path}")
        
        except Exception as e:
            logger.warning(f"无法读取配置文件 {config_path}: {e}，使用默认配置")
    else:
        logger.warning(f"配置文件不存在: {config_path}，使用默认配置")
    
    return default_config


def merge_config(default_config: dict, user_config: dict):
    """递归合并配置"""
    for key, value in user_config.items():
        if key in default_config and isinstance(default_config[key], dict) and isinstance(value, dict):
            merge_config(default_config[key], value)
        else:
            default_config[key] = value


def setup_logging(config: dict):
    """配置日志"""
    log_config = config.get("logging", {})
    
    # 移除默认handler
    logger.remove()
    
    # 控制台输出
    if log_config.get("console", True):
        logger.add(
            sys.stdout,
            level=log_config.get("level", "INFO"),
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>"
        )
    
    # 文件输出
    log_file = log_config.get("file", "multi_factor_selection.log")
    logger.add(
        log_file,
        rotation="500 MB",
        level=log_config.get("level", "INFO"),
        encoding="utf-8"
    )


def init_modules(config: dict, args=None):
    """
    初始化所有模块
    
    Args:
        config: 配置字典
        args: 命令行参数（可选）
        
    Returns:
        (data_fetcher, factor_calculator, factor_processor, stock_selector)
    """
    from src.data_fetcher import DataFetcher
    from src.factor_calculator import FactorCalculator
    from src.factor_processor import FactorProcessor
    from src.stock_selector import StockSelector
    
    # 数据源配置
    data_source_config = config.get("data_source", {})
    primary_source = data_source_config.get("primary", "akshare")
    tushare_token = data_source_config.get("tushare_token", "")
    use_akshare_backup = data_source_config.get("use_akshare_backup", True)
    use_tushare_backup = data_source_config.get("use_tushare_backup", False)
    
    # 初始化数据获取器
    data_fetcher = DataFetcher(
        primary_source=primary_source,
        tushare_token=tushare_token,
        use_akshare_backup=use_akshare_backup,
        use_tushare_backup=use_tushare_backup
    )
    
    # 因子配置（从命令行参数或配置文件读取）
    enable_quality = False
    enable_momentum = False
    enable_technical = False
    
    if args and args.factors:
        # 从命令行参数解析
        factors = [f.strip().lower() for f in args.factors.split(',')]
        enable_quality = 'quality' in factors
        enable_momentum = 'momentum' in factors
        enable_technical = 'technical' in factors
    else:
        # 从配置文件读取
        factors_config = config.get("factors", {})
        enable_quality = factors_config.get("enable_quality", False)
        enable_momentum = factors_config.get("enable_momentum", False)
        enable_technical = factors_config.get("enable_technical", False)
    
    # 初始化因子计算器
    factor_calculator = FactorCalculator(
        enable_quality=enable_quality,
        enable_momentum=enable_momentum,
        enable_technical=enable_technical
    )
    
    # 初始化因子处理器
    factor_processor = FactorProcessor(config=config.get("processing", {}))
    
    # 初始化选股器
    stock_selector = StockSelector(config=config.get("selection", {}))
    
    return data_fetcher, factor_calculator, factor_processor, stock_selector


def get_stock_pool(data_fetcher, config: dict) -> pd.DataFrame:
    """
    获取股票池
    
    Returns:
        股票池DataFrame
    """
    # 获取沪深300成分股
    stock_pool = data_fetcher.get_hs300_components()
    
    # 过滤ST股票
    if config.get("stock_pool", {}).get("filter_st", True):
        # TODO: 实现ST股票过滤
        pass
    
    # 过滤停牌股票
    if config.get("stock_pool", {}).get("filter_suspended", True):
        # TODO: 实现停牌股票过滤
        pass
    
    return stock_pool


def calculate_factors(stock_pool: pd.DataFrame, 
                      data_fetcher, 
                      factor_calculator,
                      args=None) -> pd.DataFrame:
    """
    计算因子
    
    Args:
        stock_pool: 股票池DataFrame
        data_fetcher: DataFetcher实例
        factor_calculator: FactorCalculator实例
        args: 命令行参数（可选，用于获取max_workers）
        
    Returns:
        因子DataFrame
    """
    # 获取股票代码列表
    stock_codes = stock_pool["code"].tolist()
    
    # 获取max_workers参数
    max_workers = args.max_workers if args else 5
    
    # 计算所有股票的因子（多线程）
    factors_df = factor_calculator.calculate_all_factors(
        stock_codes, data_fetcher, max_workers=max_workers
    )
    
    return factors_df


def process_factors(factors_df: pd.DataFrame,
                     factor_processor,
                     config: dict) -> pd.DataFrame:
    """
    处理因子（清洗、标准化、打分）
    
    Args:
        factors_df: 因子DataFrame
        factor_processor: FactorProcessor实例
        config: 配置字典
        
    Returns:
        处理后的因子DataFrame
    """
    # 处理因子
    processed_df = factor_processor.process(factors_df)
    
    return processed_df


def select_stocks(processed_df: pd.DataFrame,
                  stock_selector,
                  config: dict) -> pd.DataFrame:
    """
    执行选股
    
    Args:
        processed_df: 处理后的因子DataFrame
        stock_selector: StockSelector实例
        config: 配置字典
        
    Returns:
        选股结果DataFrame
    """
    selection_config = config.get("selection", {})
    
    # 执行选股
    selected_df = stock_selector.select(
        processed_df,
        top_n=selection_config.get("top_n", 50),
        min_score=selection_config.get("min_score", 0.0)
    )
    
    return selected_df


def export_results(selected_df: pd.DataFrame,
                    stock_selector,
                    config: dict) -> str:
    """
    导出结果
    
    Args:
        selected_df: 选股结果DataFrame
        stock_selector: StockSelector实例
        config: 配置字典
        
    Returns:
        保存的文件路径
    """
    output_config = config.get("output", {})
    output_dir = output_config.get("output_dir", "data/results")
    output_format = output_config.get("format", "csv")
    
    # 生成文件名（使用更精确的时间戳，避免冲突）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if output_format == "csv":
        filename = f"selection_{timestamp}.csv"
        output_path = stock_selector.export_to_csv(selected_df, filename, output_dir)
    elif output_format == "excel":
        filename = f"selection_{timestamp}.xlsx"
        output_path = stock_selector.export_to_excel(selected_df, filename, output_dir)
    else:
        raise ValueError(f"不支持的输出格式: {output_format}")
    
    return output_path


def print_top_stocks(selected_df: pd.DataFrame, n: int = 10):
    """
    打印TOP N股票（简化格式：代码、名称、总得分、当前股价）
    
    Args:
        selected_df: 选股结果DataFrame
        n: 打印前N只股票
    """
    print("\n" + "="*70)
    print(f"TOP {min(n, len(selected_df))} 股票:")
    print("="*70)
    
    # 只显示4列：排名、代码、名称、总评分、当前股价
    display_cols = ["rank", "code", "name", "total_score", "current_price"]
    
    # 只显示存在的列
    display_cols = [col for col in display_cols if col in selected_df.columns]
    
    # 打印
    print(selected_df[display_cols].head(n).to_string(index=False))
    print("="*70 + "\n")


if __name__ == "__main__":
    # 添加项目根目录到路径
    project_root = os.path.join(os.path.dirname(__file__))
    sys.path.insert(0, project_root)
    
    # 运行主程序
    main()
