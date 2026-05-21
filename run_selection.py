# -*- coding: utf-8 -*-
"""
统一选股脚本 - 读取 config.py 配置
修改 config.py 后直接运行本脚本即可

使用方法：
1. 修改 config.py 中的参数
2. 运行：python run_selection.py
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

# 读取配置
try:
    from config import (
        DATA_FETCHER, FACTOR_CALCULATOR, FACTOR_PROCESSOR,
        STOCK_SELECTOR, OUTPUT,
        ENABLE_INDUSTRY_MOMENTUM, INDUSTRY_MOMENTUM_LOOKBACK,
        SELECTION_DATE,
    )
    print("[OK] 已加载 config.py 配置")
except ImportError as e:
    print(f"[WARN] 未找到 config.py，使用默认配置: {e}")
    # 默认配置
    DATA_FETCHER = {"local_db_path": "D:/tu-shareData/astock_daily.db"}
    FACTOR_CALCULATOR = {"enable_quality": True, "enable_volatility": True}
    FACTOR_PROCESSOR = {}
    STOCK_SELECTOR = {"top_n": 5}
    OUTPUT = {"output_dir": "data/results", "export_csv": True}
    ENABLE_INDUSTRY_MOMENTUM = False
    INDUSTRY_MOMENTUM_LOOKBACK = 6
    SELECTION_DATE = "20230101"

# 初始化日志
log_file = OUTPUT.get("log_file", "stock_selection.log")
logger.add(log_file, rotation="500 MB", level="INFO")

print("\n" + "="*70)
print("多因子选股系统")
print("="*70 + "\n")

# 0.5 计算行业动量因子（如果启用）
if ENABLE_INDUSTRY_MOMENTUM:
    print("[0.5/7] 计算行业动量因子...")
    
    # 导入计算函数
    try:
        from scripts.calc_industry_momentum import calc_industry_momentum_for_date, save_industry_momentum_to_db
        
        # 计算行业动量
        print(f"  选股日期: {SELECTION_DATE}，回看 {INDUSTRY_MOMENTUM_LOOKBACK} 个月")
        result = calc_industry_momentum_for_date(
            DATA_FETCHER.get("local_db_path", "D:/tu-shareData/astock_daily.db"),
            SELECTION_DATE,
            lookback_months=INDUSTRY_MOMENTUM_LOOKBACK
        )
        
        if result is not None:
            # 保存到数据库
            count = save_industry_momentum_to_db(
                DATA_FETCHER.get("local_db_path", "D:/tu-shareData/astock_daily.db"),
                result
            )
            print(f"  ✅ 保存 {count} 条行业动量因子")
        else:
            print("  ⚠️ 计算失败")
            
    except Exception as e:
        print(f"  ❌ 计算行业动量因子失败: {e}")
        import traceback
        traceback.print_exc()

# 1. 初始化所有模块
print("[1/7] 初始化模块...")
data_fetcher = DataFetcher(**DATA_FETCHER)
factor_calculator = FactorCalculator(**FACTOR_CALCULATOR)
factor_processor = FactorProcessor()
stock_selector = StockSelector(config=STOCK_SELECTOR)
print("[OK] 模块初始化完成")
print(f"  主数据源: {data_fetcher.primary_source}")
print(f"  启用因子: 价值, 成长", end="")
if FACTOR_CALCULATOR.get("enable_quality"): print(", 质量", end="")
if FACTOR_CALCULATOR.get("enable_momentum"): print(", 动量", end="")
if FACTOR_CALCULATOR.get("enable_technical"): print(", 技术", end="")
if FACTOR_CALCULATOR.get("enable_volatility"): print(", 低波动", end="")
if FACTOR_CALCULATOR.get("enable_money_flow"): print(", 资金流", end="")
if ENABLE_INDUSTRY_MOMENTUM: print(", 行业动量", end="")
print("\n")

# 2. 获取沪深300成分股
print("[2/7] 获取沪深300成分股...")
try:
    hs300 = data_fetcher.get_hs300_components()
    print(f"[OK] 获取到 {len(hs300)} 只沪深300成分股\n")
except Exception as e:
    print(f"[FAIL] 获取沪深300成分股失败: {e}")
    raise

# 3. 计算因子
print("[3/7] 计算因子...")
print(f"  选股池: {len(hs300)} 只股票（全量沪深300成分股）")
print("  正在计算因子...\n")

try:
    factors_df = factor_calculator.calculate_all_factors(
        hs300["code"].tolist(),
        data_fetcher,
        max_workers=5
    )
    print(f"[OK] 因子计算完成: {len(factors_df)} 只股票\n")
except Exception as e:
    print(f"[FAIL] 因子计算失败: {e}")
    import traceback
    traceback.print_exc()
    raise

# 3.5 读取行业动量因子（如果启用）
if ENABLE_INDUSTRY_MOMENTUM:
    print("[3.5/7] 读取行业动量因子...")
    
    try:
        import sqlite3
        db_path = DATA_FETCHER.get("local_db_path", "D:/tu-shareData/astock_daily.db")
        
        # 从数据库读取行业动量因子
        conn = sqlite3.connect(db_path)
        industry_df = pd.read_sql("""
            SELECT ts_code, industry_momentum, industry_momentum_z
            FROM industry_momentum
            WHERE trade_date = ?
        """, conn, params=(SELECTION_DATE,))
        conn.close()
        
        if not industry_df.empty:
            # 合并到 factors_df
            factors_df = factors_df.merge(industry_df, left_on='code', right_on='ts_code', how='left')
            
            # 重命名列（符合因子命名规范）
            factors_df = factors_df.rename(columns={
                'industry_momentum': 'IMF1_industry_momentum',
                'industry_momentum_z': 'IMF2_industry_momentum_z'
            })
            
            print(f"  ✅ 合并 {len(industry_df)} 只股票的行业动量因子")
        else:
            print(f"  ⚠️ 未找到 {SELECTION_DATE} 的行业动量因子")
            
    except Exception as e:
        print(f"  ❌ 读取行业动量因子失败: {e}")

# 4. 处理因子（清洗、标准化、打分）
print("[4/7] 处理因子（清洗、标准化、打分）...")
try:
    processed_df = factor_processor.process(factors_df)
    print(f"[OK] 因子处理完成\n")
except Exception as e:
    print(f"[FAIL] 因子处理失败: {e}")
    raise

# 5. 选股
top_n = STOCK_SELECTOR.get("top_n", 5)
print(f"[5/7] 执行选股（TOP {top_n}）...")
try:
    selected_df = stock_selector.select(processed_df, top_n=top_n)
    print(f"[OK] 选股完成: {len(selected_df)} 只股票\n")
except Exception as e:
    print(f"[FAIL] 选股失败: {e}")
    raise

# 6. 打印和导出结果
print("[6/7] 打印和导出结果...\n")

# 打印TOP N
print_top_n = OUTPUT.get("print_top_n", 20)
stock_selector.print_top_stocks(selected_df, n=print_top_n)

# 导出到CSV
if OUTPUT.get("export_csv", True):
    output_dir = OUTPUT.get("output_dir", "data/results")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "selection_results.csv")
    
    try:
        stock_selector.export_to_csv(
            selected_df,
            filename="selection_results.csv",
            output_dir=output_dir
        )
        print(f"\n[OK] 结果已保存到: {output_path}")
    except Exception as e:
        print(f"\n[FAIL] 导出失败: {e}")

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
