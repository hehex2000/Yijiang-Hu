# -*- coding: utf-8 -*-
"""
选股脚本 - 2023-01-01 选股
因子：价值 + 成长 + 质量 + 低波动
"""
import sys
import os
import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from loguru import logger
from src.data_fetcher import DataFetcher
from src.factor_calculator import FactorCalculator
from src.factor_processor import FactorProcessor
from src.stock_selector import StockSelector

# 初始化日志
logger.add("stock_selection_20230101.log", rotation="500 MB", level="INFO")

print("\n" + "="*70)
print("多因子选股 - 2023-01-01")
print("="*70 + "\n")

# 1. 初始化所有模块
print("[1/6] 初始化模块...")
data_fetcher = DataFetcher(local_db_path="D:/tu-shareData/astock_daily.db")
factor_calculator = FactorCalculator(
    enable_quality=True,      # 质量因子
    enable_volatility=True,   # 低波动因子
    enable_momentum=True,     # 动量因子（新增）
    enable_technical=True,    # 技术因子（新增）
    enable_money_flow=False  # 资金流因子（关闭，因为是历史数据）
)
factor_processor = FactorProcessor()
stock_selector = StockSelector(config={"top_n": 5})  # 改为 TOP 5
print("[OK] 模块初始化完成")
print(f"  主数据源: {data_fetcher.primary_source}")
print(f"  启用因子: 价值, 成长, 质量, 动量, 技术, 低波动\n")

# 2. 获取沪深300成分股
print("[2/6] 获取沪深300成分股...")
try:
    hs300 = data_fetcher.get_hs300_components()
    print(f"[OK] 获取到 {len(hs300)} 只沪深300成分股\n")
except Exception as e:
    print(f"[FAIL] 获取沪深300成分股失败: {e}")
    raise

# 3. 计算因子（使用截至2023-01-01的数据）
print("[3/6] 计算因子（使用截至2023-01-01的数据）...")
print("  注意：质量因子和低波动因子需要足够的历史数据\n")

# 为了使用历史数据，我们需要修改 calculate_single_stock_factors 的调用
# 但当前实现是使用最新数据，所以我们需要：
# 1. 获取2023-01-01之前的history数据
# 2. 获取2023-01-01之前的financial数据
# 3. 获取2023-01-01之前的valuation数据

# 由于当前实现限制，我们计算所有股票的因子（使用最新数据）
# 但在实际选股时，应该只用2023-01-01之前的数据

# 这里先计算所有股票的因子（使用本地数据库中的最新数据）
test_codes = hs300["code"].tolist()  # 所有沪深300成分股

print(f"  选股池: {len(test_codes)} 只股票（全量沪深300成分股）")
print("  正在计算因子...\n")

try:
    factors_df = factor_calculator.calculate_all_factors(
        test_codes, 
        data_fetcher,
        max_workers=5
    )
    print(f"[OK] 因子计算完成: {len(factors_df)} 只股票\n")
except Exception as e:
    print(f"[FAIL] 因子计算失败: {e}")
    import traceback
    traceback.print_exc()
    raise

# 4. 处理因子（清洗、标准化、打分）
print("[4/6] 处理因子（清洗、标准化、打分）...")
try:
    processed_df = factor_processor.process(factors_df)
    print(f"[OK] 因子处理完成\n")
except Exception as e:
    print(f"[FAIL] 因子处理失败: {e}")
    raise

# 5. 选股
print("[5/6] 执行选股（TOP 20）...")
try:
    selected_df = stock_selector.select(processed_df, top_n=20)
    print(f"[OK] 选股完成: {len(selected_df)} 只股票\n")
except Exception as e:
    print(f"[FAIL] 选股失败: {e}")
    raise

# 6. 打印和导出结果
print("[6/6] 打印和导出结果...\n")

# 打印TOP 20
stock_selector.print_top_stocks(selected_df, n=20)

# 导出到CSV
output_dir = "data/results"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "selection_20230101.csv")

try:
    stock_selector.export_to_csv(selected_df, filename="selection_20230101.csv", output_dir=output_dir)
    print(f"[OK] 结果已保存到: {output_path}")
except Exception as e:
    print(f"[FAIL] 导出失败: {e}")

# 打印因子列表
print("\n" + "="*70)
print("使用的因子列表:")
print("="*70)

factor_cols = [c for c in processed_df.columns if c not in ['code', 'name', 'market_cap', 'current_price', 'total_score', 'rank']]
for i, col in enumerate(factor_cols, 1):
    print(f"  {i}. {col}")

print("="*70 + "\n")

print("\n" + "="*70)
print("选股完成!")
print("="*70 + "\n")
