"""
快速测试脚本 - 测试多因子选股系统
使用前10只股票进行快速测试
"""

import sys
import os

# 添加项目根目录到路径
project_root = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from loguru import logger
from src.data_fetcher import DataFetcher
from src.factor_calculator import FactorCalculator
from src.factor_processor import FactorProcessor
from src.stock_selector import StockSelector


def test_system():
    """测试整个系统"""
    print("\n" + "="*70)
    print("多因子选股系统 - 快速测试")
    print("="*70 + "\n")
    
    try:
        # 1. 初始化模块
        print("[1/6] 初始化模块...")
        data_fetcher = DataFetcher(use_tushare=False)
        factor_calculator = FactorCalculator()
        factor_processor = FactorProcessor()
        stock_selector = StockSelector(config={"top_n": 10})
        print("✓ 模块初始化完成\n")
        
        # 2. 获取股票池（测试用前10只）
        print("[2/6] 获取股票池...")
        hs300 = data_fetcher.get_hs300_components()
        test_codes = hs300["code"].head(10).tolist()
        print(f"✓ 测试股票池: {len(test_codes)} 只股票")
        print(f"  股票列表: {test_codes}\n")
        
        # 3. 计算因子
        print("[3/6] 计算因子...")
        factors_df = factor_calculator.calculate_all_factors(test_codes, data_fetcher)
        print(f"✓ 因子计算完成: {len(factors_df)} 只股票")
        print(f"  因子数量: {len(factors_df.columns) - 1} 个\n")
        
        # 4. 处理因子（清洗、标准化、打分）
        print("[4/6] 处理因子...")
        processed_df = factor_processor.process(factors_df)
        print(f"✓ 因子处理完成")
        print(f"  处理后数据量: {len(processed_df)} 只股票\n")
        
        # 5. 选股和导出
        print("[5/6] 执行选股...")
        selected_df = stock_selector.select(processed_df, top_n=10)
        print(f"✓ 选股完成: {len(selected_df)} 只股票\n")
        
        # 6. 打印TOP 10
        print("[6/6] 展示结果...")
        stock_selector.print_top_stocks(selected_df, n=10)
        
        # 导出结果
        output_path = stock_selector.export_to_csv(selected_df)
        print(f"\n✓ 结果已保存到: {output_path}")
        
        print("\n" + "="*70)
        print("测试完成!")
        print("="*70 + "\n")
        
        return True
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 配置日志
    logger.add("test_log.log", rotation="500 MB", level="INFO")
    
    # 运行测试
    success = test_system()
    
    if success:
        print("✓ 系统测试通过！")
    else:
        print("✗ 系统测试失败，请检查日志")
