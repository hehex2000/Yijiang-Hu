# -*- coding: utf-8 -*-
"""
简化测试 - 验证因子计算（只用本地数据库）
"""
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from loguru import logger
from src.data_fetcher import DataFetcher
from src.factor_calculator import FactorCalculator
from src.factor_processor import FactorProcessor
from src.stock_selector import StockSelector

# 初始化日志（只显示INFO及以上）
logger.remove()
logger.add(sys.stderr, level="INFO")

print("\n" + "="*70)
print("简化测试 - 验证因子计算（只用本地数据库）")
print("="*70 + "\n")

# 1. 初始化（只用本地数据库）
print("[1/5] 初始化模块...")
data_fetcher = DataFetcher(local_db_path="D:/tu-shareData/astock_daily.db")
factor_calculator = FactorCalculator(
    enable_quality=True,
    enable_volatility=True,
    enable_money_flow=False,
    enable_momentum=False,
    enable_technical=False
)
factor_processor = FactorProcessor()
stock_selector = StockSelector(config={"top_n": 10})
print("[OK] 初始化完成\n")

# 2. 获取HS300成分股（前20只用于测试）
print("[2/5] 获取HS300成分股（前20只）...")
hs300 = data_fetcher.get_hs300_components()
test_codes = hs300["code"].head(20).tolist()
print(f"[OK] 测试股票池: {len(test_codes)} 只\n")

# 3. 计算因子
print("[3/5] 计算因子...")
factors_df = factor_calculator.calculate_all_factors(
    test_codes, 
    data_fetcher,
    max_workers=5
)
print(f"[OK] 因子计算完成: {len(factors_df)} 只股票\n")

# 4. 处理因子
print("[4/5] 处理因子（清洗、标准化、打分）...")
processed_df = factor_processor.process(factors_df)
print(f"[OK] 因子处理完成\n")

# 5. 选股
print("[5/5] 执行选股（TOP 10）...")
selected_df = stock_selector.select(processed_df, top_n=10)
print(f"[OK] 选股完成: {len(selected_df)} 只股票\n")

# 打印结果
print("="*70)
print("TOP 10 股票:")
print("="*70)

# 显示关键列
display_cols = ["rank", "code", "name", "total_score"]
# 添加几个关键因子
quality_cols = [c for c in processed_df.columns if c.startswith("QF")][:2]
vol_cols = [c for c in processed_df.columns if c.startswith("LVF")][:2]
display_cols.extend(quality_cols)
display_cols.extend(vol_cols)

# 只显示存在的列
display_cols = [c for c in display_cols if c in selected_df.columns]
print(selected_df[display_cols].to_string(index=False))
print()

# 打印因子统计
print("="*70)
print("因子统计:")
print("="*70)

factor_cols = [c for c in processed_df.columns 
              if c.startswith(("VF", "GF", "QF", "LVF"))]
for col in factor_cols[:10]:  # 只显示前10个因子
    if col in processed_df.columns:
        non_nan = processed_df[col].notna().sum()
        mean_val = processed_df[col].mean()
        print(f"  {col}: 非空={non_nan}/{len(processed_df)}, 均值={mean_val:.4f}" 
              if not pd.isna(mean_val) else f"  {col}: 非空={non_nan}/{len(processed_df)}, 均值=NaN")

print("\n" + "="*70)
print("测试完成!")
print("="*70 + "\n")
