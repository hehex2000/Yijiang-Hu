#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试低波动因子计算
"""

from loguru import logger
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data_fetcher import DataFetcher
from src.factor_calculator import FactorCalculator


def test_volatility_factors():
    """测试低波动因子计算"""
    
    print("=" * 70)
    print("测试低波动因子（Low Volatility Factors）")
    print("=" * 70 + "\n")
    
    # 初始化日志
    logger.add("test_volatility.log", rotation="100 MB", level="DEBUG")
    
    # 创建数据获取器
    print("[1/3] 初始化数据获取器...")
    data_fetcher = DataFetcher(primary_source="akshare", use_tushare_backup=False)
    print("  ✓ 完成\n")
    
    # 测试1: 不启用低波动因子（默认）
    print("[2/3] 测试1: 默认配置（不启用低波动因子）...")
    calculator_default = FactorCalculator(
        enable_quality=False,
        enable_momentum=False,
        enable_technical=False,
        enable_volatility=False  # 默认关闭
    )
    
    factors_default = calculator_default.calculate_single_stock_factors(
        "000001", data_fetcher
    )
    
    # 检查是否包含低波动因子
    volatility_keys = [k for k in factors_default.keys() if k.startswith("LVF")]
    print(f"  - 低波动因子数: {len(volatility_keys)} (期望: 0)")
    print(f"  ✓ 完成\n")
    
    # 测试2: 启用低波动因子
    print("[3/3] 测试2: 启用低波动因子...")
    calculator_vol = FactorCalculator(
        enable_quality=False,
        enable_momentum=False,
        enable_technical=False,
        enable_volatility=True  # 启用低波动因子
    )
    
    factors_vol = calculator_vol.calculate_single_stock_factors(
        "000001", data_fetcher
    )
    
    # 检查是否包含低波动因子
    volatility_keys = [k for k in factors_vol.keys() if k.startswith("LVF")]
    print(f"  - 低波动因子数: {len(volatility_keys)} (期望: 5)")
    print(f"  - 低波动因子列表:")
    for key in sorted(volatility_keys):
        value = factors_vol[key]
        print(f"    {key}: {value:.6f}" if not pd.isna(value) else f"    {key}: NaN")
    
    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)
    
    # 打印所有因子
    print("\n所有因子值:")
    for key, value in sorted(factors_vol.items()):
        if key == "code" or key == "name":
            print(f"  {key}: {value}")
        elif isinstance(value, float):
            print(f"  {key}: {value:.6f}" if not pd.isna(value) else f"  {key}: NaN")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    import pandas as pd
    test_volatility_factors()
