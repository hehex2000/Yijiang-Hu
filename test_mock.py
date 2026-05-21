"""
使用模拟数据的测试脚本 - 验证系统核心功能（简化输出格式）
"""
import sys
import os
import pandas as pd
import numpy as np

# 添加项目根目录到路径
project_root = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from loguru import logger
from src.factor_calculator import FactorCalculator
from src.factor_processor import FactorProcessor
from src.stock_selector import StockSelector


def generate_mock_data(codes):
    """生成模拟因子数据（包含股票名称和市值）"""
    np.random.seed(42)  # 固定随机种子，保证结果可重复
    
    stock_names = {
        '000001': '平安银行',
        '000002': '万科A',
        '000063': '中兴通讯',
        '000100': 'TCL科技',
        '000157': '中联重科',
        '000166': '申万宏源',
        '000301': '东方盛虹',
        '000333': '美的集团',
        '000338': '潍柴动力',
        '000408': '藏格矿业'
    }
    
    data = []
    for code in codes:
        row = {
            'code': code,
            'name': stock_names.get(code, f'股票{code}'),
            'market_cap': round(np.random.uniform(100, 5000), 2)  # 市值：100亿-5000亿
        }
        
        # 价值因子（5个）
        row['VF1_PE'] = np.random.uniform(5, 50)  # PE
        row['VF2_PB'] = np.random.uniform(0.5, 5)  # PB
        row['VF3_PS'] = np.random.uniform(0.5, 10)  # PS
        row['VF4_PEG'] = np.random.uniform(0.5, 3)  # PEG
        row['VF5_EV_EBITDA'] = np.random.uniform(5, 30)  # EV/EBITDA
        
        # 成长因子（5个）
        row['GF1_ROE_growth'] = np.random.uniform(-0.1, 0.3)  # ROE增长率
        row['GF2_profit_growth'] = np.random.uniform(-0.2, 0.5)  # 净利润增长率
        row['GF3_revenue_growth'] = np.random.uniform(-0.1, 0.4)  # 营收增长率
        row['GF4_asset_growth'] = np.random.uniform(-0.05, 0.3)  # 总资产增长率
        row['GF5_EPS_growth'] = np.random.uniform(-0.15, 0.4)  # EPS增长率
        
        # 质量因子（5个）
        row['QF1_ROE'] = np.random.uniform(0.05, 0.3)  # ROE
        row['QF2_ROA'] = np.random.uniform(0.02, 0.15)  # ROA
        row['QF3_AssetLiabRatio'] = np.random.uniform(0.3, 0.7)  # 资产负债率
        row['QF4_current_ratio'] = np.random.uniform(0.8, 3)  # 流动比率
        row['QF5_inventory_turnover'] = np.random.uniform(2, 10)  # 存货周转率
        
        # 动量因子（5个）
        row['MF1_1m_return'] = np.random.uniform(-0.2, 0.3)  # 1个月收益率
        row['MF2_3m_return'] = np.random.uniform(-0.3, 0.5)  # 3个月收益率
        row['MF3_6m_return'] = np.random.uniform(-0.4, 0.8)  # 6个月收益率
        row['MF4_12m_return'] = np.random.uniform(-0.5, 1.0)  # 12个月收益率
        row['MF5_volume_momentum'] = np.random.uniform(-0.2, 0.3)  # 成交量动量
        
        # 技术因子（5个）
        row['TF1_MA5'] = np.random.uniform(10, 100)  # 5日均线
        row['TF2_MA20'] = np.random.uniform(10, 100)  # 20日均线
        row['TF3_MA_bullish'] = np.random.choice([0, 1])  # 均线多头排列
        row['TF4_RSI'] = np.random.uniform(30, 70)  # RSI
        row['TF5_BOLL_position'] = np.random.uniform(0, 1)  # 布林带位置
        
        data.append(row)
    
    return pd.DataFrame(data)


def test_system_with_mock_data():
    """使用模拟数据测试整个系统（简化输出）"""
    print("\n" + "="*70)
    print("多因子选股系统 - 模拟数据测试（简化输出）")
    print("="*70 + "\n")
    
    try:
        # 1. 初始化模块
        print("[1/4] 初始化模块...")
        factor_calculator = FactorCalculator()
        factor_processor = FactorProcessor()
        stock_selector = StockSelector(config={"top_n": 10})
        print("✓ 模块初始化完成\n")
        
        # 2. 生成模拟数据
        print("[2/4] 生成模拟数据...")
        test_codes = ['000001', '000002', '000063', '000100', '000157', 
                     '000166', '000301', '000333', '000338', '000408']
        factors_df = generate_mock_data(test_codes)
        print(f"✓ 模拟数据生成完成: {len(factors_df)} 只股票, {len(factors_df.columns)-3} 个因子\n")
        
        # 3. 处理因子（清洗、标准化、打分）
        print("[3/4] 处理因子...")
        processed_df = factor_processor.process(factors_df)
        print(f"✓ 因子处理完成\n")
        
        # 4. 选股和导出
        print("[4/4] 执行选股...")
        selected_df = stock_selector.select(processed_df, top_n=10)
        print(f"✓ 选股完成: {len(selected_df)} 只股票\n")
        
        # 5. 打印TOP 10（简化格式）
        stock_selector.print_top_stocks(selected_df, n=10)
        
        # 导出结果（简化格式）
        output_path = stock_selector.export_to_csv(selected_df)
        print(f"\n✓ 结果已保存到CSV: {output_path}")
        
        # 同时导出到Excel
        excel_path = stock_selector.export_to_excel(selected_df)
        print(f"✓ 结果已保存到Excel: {excel_path}")
        
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
    logger.add("test_mock_log.log", rotation="500 MB", level="INFO")
    
    # 运行测试
    success = test_system_with_mock_data()
    
    if success:
        print("✓ 系统测试通过！")
    else:
        print("✗ 系统测试失败，请检查日志")
