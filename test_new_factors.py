# -*- coding: utf-8 -*-
"""
测试脚本 - 验证质量因子和资金流因子
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

# 初始化日志
logger.add("test_new_factors.log", rotation="500 MB", level="DEBUG")

print("\n" + "="*70)
print("测试：质量因子（Quality）和资金流因子（Money Flow）")
print("="*70 + "\n")

# 1. 初始化数据获取器
print("[1/4] 初始化数据获取器...")
data_fetcher = DataFetcher(primary_source="akshare")
print("[OK] 数据获取器初始化完成\n")

# 2. 初始化因子计算器（启用质量因子和资金流因子）
print("[2/4] 初始化因子计算器（启用质量+资金流因子）...")
factor_calculator = FactorCalculator(
    enable_quality=True,      # 质量因子
    enable_money_flow=True,   # 资金流因子
    enable_momentum=False,    # 动量因子（关闭）
    enable_technical=False,   # 技术因子（关闭）
    enable_volatility=False    # 低波动因子（关闭）
)
print("[OK] 因子计算器初始化完成\n")

# 3. 测试单只股票的因子计算
test_code = "000001"  # 平安银行

print(f"[3/4] 计算 {test_code} 的因子...")
try:
    factors = factor_calculator.calculate_single_stock_factors(test_code, data_fetcher)
    
    print(f"[OK] 成功计算 {len(factors)} 个因子\n")
    
    # 打印所有因子
    print("="*70)
    print(f"股票 {test_code} 的因子详情:")
    print("="*70)
    
    for key, value in factors.items():
        # 格式化输出
        if isinstance(value, float) and not pd.isna(value):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print("="*70 + "\n")
    
    # 检查质量因子
    print("质量因子检查:")
    quality_keys = [k for k in factors.keys() if k.startswith("QF")]
    if len(quality_keys) > 0:
        print(f"[OK] 找到 {len(quality_keys)} 个质量因子:")
        for key in quality_keys:
            value = factors[key]
            if not pd.isna(value):
                print(f"    {key}: {value:.4f}")
            else:
                print(f"    {key}: NaN (数据不可用)")
    else:
        print("[FAIL] 未找到质量因子")
    print()
    
    # 检查资金流因子
    print("资金流因子检查:")
    money_flow_keys = [k for k in factors.keys() if k.startswith("MF") and "return" not in k]
    if len(money_flow_keys) > 0:
        print(f"[OK] 找到 {len(money_flow_keys)} 个资金流因子:")
        for key in money_flow_keys:
            value = factors[key]
            if not pd.isna(value):
                print(f"    {key}: {value:.4f}")
            else:
                print(f"    {key}: NaN (数据不可用)")
    else:
        print("[FAIL] 未找到资金流因子")
    print()
    
except Exception as e:
    print(f"[FAIL] 计算因子时出错: {e}")
    import traceback
    traceback.print_exc()

# 4. 测试多只股票的因子计算
print(f"[4/4] 测试多只股票的因子计算（前5只沪深300成分股）...")

try:
    # 获取沪深300成分股（前5只）
    hs300 = data_fetcher.get_hs300_components()
    test_codes = hs300["code"].head(5).tolist()
    
    print(f"测试股票: {test_codes}\n")
    
    # 计算所有股票的因子
    df_factors = factor_calculator.calculate_all_factors(test_codes, data_fetcher, max_workers=2)
    
    print(f"[OK] 成功计算 {len(df_factors)} 只股票的因子\n")
    
    # 检查DataFrame的列
    print("因子DataFrame列名:")
    for col in df_factors.columns:
        print(f"  {col}")
    print()
    
    # 检查质量因子列
    quality_cols = [c for c in df_factors.columns if c.startswith("QF")]
    print(f"质量因子列 ({len(quality_cols)} 个): {quality_cols}")
    print()
    
    # 检查资金流因子列
    mf_cols = [c for c in df_factors.columns if c.startswith("MF") and "return" not in c]
    print(f"资金流因子列 ({len(mf_cols)} 个): {mf_cols}")
    print()
    
    # 打印前几行数据
    print("前3只股票的因子数据（节选）:")
    if len(quality_cols) > 0 or len(mf_cols) > 0:
        display_cols = ["code", "name"] + quality_cols[:3] + mf_cols[:3]
        display_cols = [c for c in display_cols if c in df_factors.columns]
        print(df_factors[display_cols].head(3).to_string(index=False))
    print()
    
except Exception as e:
    print(f"[FAIL] 计算多只股票因子时出错: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("测试完成!")
print("="*70 + "\n")
