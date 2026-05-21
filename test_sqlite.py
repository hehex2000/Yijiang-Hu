"""
测试脚本：从 SQLite 数据库读取数据，计算几只股票的因子
"""

import sys
import os
import pandas as pd
from loguru import logger

# 添加项目根目录到路径
project_root = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.data_fetcher import SQLiteDataFetcher
from src.factor_calculator import FactorCalculator
from src.factor_processor import FactorProcessor
from src.stock_selector import StockSelector


def main():
    """主测试程序"""
    print("\n" + "="*70)
    print("测试：多因子选股系统 (使用本地 SQLite 数据库)")
    print("="*70 + "\n")
    
    # 初始化日志
    logger.remove()
    logger.add(sys.stdout, level="DEBUG", 
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")
    
    try:
        # 1. 初始化所有模块
        print("\n[1/5] 初始化模块...")
        data_fetcher = SQLiteDataFetcher(db_path="D:/tu-shareData/astock_daily.db")
        factor_calculator = FactorCalculator(
            enable_quality=False,
            enable_momentum=False,
            enable_technical=False
        )
        factor_processor = FactorProcessor()
        stock_selector = StockSelector()
        print("✓ 模块初始化完成\n")
        
        # 2. 测试几只股票
        test_codes = ["000001", "000002", "600000", "600036", "601318"]
        print(f"[2/5] 测试计算 {len(test_codes)} 只股票的因子...")
        
        factors_df = factor_calculator.calculate_all_factors(
            test_codes, data_fetcher, max_workers=1
        )
        print(f"✓ 因子计算完成: {len(factors_df)} 只股票\n")
        
        # 3. 处理因子
        print("[3/5] 处理因子...")
        processed_df = factor_processor.process(factors_df)
        print(f"✓ 因子处理完成\n")
        
        # 4. 执行选股
        print("[4/5] 执行选股...")
        selected_df = stock_selector.select(processed_df, top_n=10, min_score=0.0)
        print(f"✓ 选股完成: {len(selected_df)} 只股票\n")
        
        # 5. 打印结果
        print("[5/5] 打印TOP 5股票:")
        print("="*70)
        
        # 显示列：排名、代码、名称、总评分、当前股价、市值
        display_cols = ["rank", "code", "name", "total_score", "current_price", "market_cap"]
        display_cols = [col for col in display_cols if col in selected_df.columns]
        
        print(selected_df[display_cols].head(5).to_string(index=False))
        print("="*70 + "\n")
        
        logger.info("测试完成!")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
