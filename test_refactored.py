"""
测试重构后的因子计算（使用 TA-Lib）
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from loguru import logger
from src.data_fetcher import DataFetcher
from src.factor_calculator import FactorCalculator

# 初始化日志
logger.add("test_refactored.log", rotation="500 MB", level="DEBUG")

print("\n" + "="*70)
print("测试：重构后的因子计算（使用 TA-Lib）")
print("="*70 + "\n")

# 1. 初始化数据获取器（使用本地数据库）
print("[1/4] 初始化数据获取器（本地数据库）...")
data_fetcher = DataFetcher(
    primary_source="local_db",
    local_db_path=r"D:\tu-shareData\astock_daily.db"
)
print(f"[OK] 主数据源: {data_fetcher.primary_source}\n")

# 2. 初始化因子计算器（启用所有因子）
print("[2/4] 初始化因子计算器（启用所有因子）...")
factor_calc = FactorCalculator(
    enable_quality=True,       # 质量因子
    enable_momentum=True,     # 动量因子
    enable_technical=True,     # 技术因子（使用 TA-Lib）
    enable_volatility=True,    # 低波动因子
    enable_money_flow=False    # 资金流因子（暂时关闭）
)
print("[OK] 因子计算器初始化完成\n")

# 3. 测试单只股票的因子计算
test_code = "000001.SZ"
print(f"[3/4] 计算单只股票因子: {test_code}")
try:
    factors = factor_calc.calculate_single_stock_factors(test_code, data_fetcher)
    
    print(f"\n[OK] 因子计算成功！共 {len(factors)} 个因子\n")
    print("="*70)
    print("因子列表：")
    print("="*70)
    
    # 导入 pandas（用于判断 NaN）
    import pandas as pd
    
    # 按类别打印因子
    categories = {
        "基本信息": ["code", "name", "current_price", "market_cap"],
        "价值因子": [k for k in factors.keys() if k.startswith("VF")],
        "成长因子": [k for k in factors.keys() if k.startswith("GF")],
        "质量因子": [k for k in factors.keys() if k.startswith("QF")],
        "动量因子": [k for k in factors.keys() if k.startswith("MF")],
        "技术因子": [k for k in factors.keys() if k.startswith("TF")],
        "低波动因子": [k for k in factors.keys() if k.startswith("LVF")]
    }
    
    for cat, keys in categories.items():
        if len(keys) > 0:
            print(f"\n【{cat}】")
            for key in keys:
                val = factors[key]
                if pd.isna(val):
                    print(f"  {key}: NaN")
                elif isinstance(val, str):
                    print(f"  {key}: {val}")
                else:
                    try:
                        print(f"  {key}: {val:.4f}")
                    except:
                        print(f"  {key}: {val}")
    
    print("\n" + "="*70)
    print("✅ 单只股票因子计算测试通过！")
    print("="*70 + "\n")
    
except Exception as e:
    print(f"\n❌ 因子计算失败: {e}")
    import traceback
    traceback.print_exc()

# 4. 测试多只股票的因子计算（前 10 只）
print(f"[4/4] 计算多只股票因子（前 10 只沪深 300 成分股）...")
try:
    # 获取沪深 300 成分股
    hs300 = data_fetcher.get_hs300_components()
    
    if hs300 is not None and len(hs300) > 0:
        test_codes = hs300["code"].head(10).tolist()
        print(f"  测试股票: {test_codes}")
        
        factors_df = factor_calc.calculate_all_factors(
            test_codes, 
            data_fetcher, 
            max_workers=5
        )
        
        print(f"\n[OK] 多只股票因子计算成功！共 {len(factors_df)} 只股票\n")
        print("="*70)
        print("因子 DataFrame 预览：")
        print("="*70)
        print(factors_df.head())
        print("\n因子 DataFrame 信息：")
        print(factors_df.info())
        print("\n" + "="*70)
        print("[OK] 多只股票因子计算测试通过！")
        print("="*70 + "\n")
    else:
        print("  [OK] 获取沪深 300 成分股失败")
        
except Exception as e:
    print(f"\n[ERR] 多只股票因子计算失败: {e}")
    import traceback
    traceback.print_exc()

print("="*70)
print("所有测试完成！")
print("="*70)
