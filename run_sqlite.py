"""
使用本地 SQLite 数据库运行多因子选股系统
从 D:\tu-shareData\astock_daily.db 读取数据
"""

import sys
import os
import pandas as pd
from loguru import logger
from datetime import datetime

# 添加项目根目录到路径
project_root = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.data_fetcher import SQLiteDataFetcher
from src.factor_calculator import FactorCalculator
from src.factor_processor import FactorProcessor
from src.stock_selector import StockSelector


def main():
    """主程序"""
    print("\n" + "="*70)
    print("多因子选股系统 (使用本地 SQLite 数据库)")
    print("="*70 + "\n")
    
    # 1. 初始化日志
    logger.remove()
    logger.add(sys.stdout, level="INFO", 
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")
    logger.add("multi_factor_selection.log", rotation="500 MB", level="INFO", encoding="utf-8")
    
    logger.info("="*50)
    logger.info("多因子选股系统启动 (SQLite版本)")
    logger.info("="*50)
    
    try:
        # 2. 初始化所有模块
        print("\n[1/6] 初始化模块...")
        data_fetcher, factor_calculator, factor_processor, stock_selector = init_modules()
        print("✓ 模块初始化完成\n")
        
        # 3. 获取股票池（所有A股股票）
        print("[2/6] 获取股票池...")
        stock_pool = get_stock_pool(data_fetcher)
        print(f"✓ 获取到 {len(stock_pool)} 只股票\n")
        
        # 4. 计算因子
        print("[3/6] 计算因子（这可能需要几分钟）...")
        factors_df = calculate_factors(stock_pool, data_fetcher, factor_calculator)
        print(f"✓ 因子计算完成: {len(factors_df)} 只股票\n")
        
        # 5. 处理因子（清洗、标准化、打分）
        print("[4/6] 处理因子...")
        processed_df = process_factors(factors_df, factor_processor)
        print(f"✓ 因子处理完成\n")
        
        # 6. 执行选股
        print("[5/6] 执行选股...")
        selected_df = select_stocks(processed_df, stock_selector)
        print(f"✓ 选股完成: {len(selected_df)} 只股票\n")
        
        # 7. 导出结果
        print("[6/6] 导出结果 (Excel格式)...")
        output_path = export_results(selected_df, stock_selector, output_format="excel")
        print(f"✓ 结果已保存到: {output_path}\n")
        
        # 8. 打印TOP 10
        print_top_stocks(selected_df, n=10)
        
        logger.info("="*50)
        logger.info("选股完成!")
        logger.info("="*50 + "\n")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        raise
    
    print("="*70)
    print("程序执行完成!")
    print("="*70 + "\n")


def init_modules():
    """初始化所有模块"""
    from src.data_fetcher import SQLiteDataFetcher
    from src.factor_calculator import FactorCalculator
    from src.factor_processor import FactorProcessor
    from src.stock_selector import StockSelector
    
    # 初始化数据获取器（使用本地SQLite）
    data_fetcher = SQLiteDataFetcher(db_path="D:/tu-shareData/astock_daily.db")
    
    # 初始化因子计算器（启用所有因子：价值、成长、质量、动量、技术）
    factor_calculator = FactorCalculator(
        enable_quality=True,
        enable_momentum=True,
        enable_technical=True
    )
    
    # 初始化因子处理器
    factor_processor = FactorProcessor()
    
    # 初始化选股器
    stock_selector = StockSelector()
    
    return data_fetcher, factor_calculator, factor_processor, stock_selector


def get_stock_pool(data_fetcher) -> pd.DataFrame:
    """获取股票池（沪深300成分股）"""
    # 方法1：尝试从 AkShare 获取沪深300成分股
    try:
        import akshare as ak
        print("  正在从 AkShare 获取沪深300成分股...")
        df = ak.index_stock_cons_csindex(symbol="000300")
        # AkShare 返回格式: 日期, 指数代码, 指数名称, 指数英文名称, 成分券代码, 成分券名称
        result = df[['成分券代码', '成分券名称']].copy()
        result.columns = ['code', 'name']
        # 移除后缀（如果有）
        result['code'] = result['code'].str.replace(r'\.(SH|SZ)$', '', regex=True)
        print(f"  ✓ 从 AkShare 获取到 {len(result)} 只沪深300成分股")
        return result
    except Exception as e:
        print(f"  从 AkShare 获取失败: {e}")
    
    # 方法2：如果 AkShare 失败，返回所有股票（从数据库）
    print("  正在从数据库获取所有股票...")
    stock_pool = data_fetcher.get_hs300_components()  # 实际上是所有A股
    print(f"  ✓ 从数据库获取到 {len(stock_pool)} 只股票（所有A股）")
    print("  警告：由于无法获取沪深300成分股，将计算所有A股股票")
    
    return stock_pool


def calculate_factors(stock_pool: pd.DataFrame, 
                      data_fetcher, 
                      factor_calculator,
                      max_workers: int = 5) -> pd.DataFrame:
    """
    计算因子
    
    Args:
        stock_pool: 股票池DataFrame
        data_fetcher: 数据获取器
        factor_calculator: 因子计算器
        max_workers: 最大线程数
        
    Returns:
        因子DataFrame
    """
    # 获取股票代码列表
    stock_codes = stock_pool["code"].tolist()
    
    # 计算所有股票的因子（单线程，避免多线程数据库访问冲突）
    factors_df = factor_calculator.calculate_all_factors(
        stock_codes, data_fetcher, max_workers=1  # SQLite 使用单线程
    )
    
    return factors_df


def process_factors(factors_df: pd.DataFrame,
                     factor_processor) -> pd.DataFrame:
    """处理因子（清洗、标准化、打分）"""
    # 处理因子
    processed_df = factor_processor.process(factors_df)
    
    return processed_df


def select_stocks(processed_df: pd.DataFrame,
                  stock_selector,
                  top_n: int = 50,
                  min_score: float = 0.0) -> pd.DataFrame:
    """执行选股"""
    # 执行选股
    selected_df = stock_selector.select(
        processed_df,
        top_n=top_n,
        min_score=min_score
    )
    
    return selected_df


def export_results(selected_df: pd.DataFrame,
                   stock_selector,
                   output_dir: str = "data/results",
                   output_format: str = "csv") -> str:
    """导出结果"""
    # 生成文件名
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
    """打印TOP N股票"""
    print("\n" + "="*70)
    print(f"TOP {min(n, len(selected_df))} 股票:")
    print("="*70)
    
    # 显示列：排名、代码、名称、总评分、当前股价、市值
    display_cols = ["rank", "code", "name", "total_score", "current_price", "market_cap"]
    
    # 只显示存在的列
    display_cols = [col for col in display_cols if col in selected_df.columns]
    
    # 打印
    print(selected_df[display_cols].head(n).to_string(index=False))
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
