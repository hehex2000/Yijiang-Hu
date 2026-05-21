# -*- coding: utf-8 -*-
"""
测试脚本 - 验证 DataFetcher 优先使用本地数据库
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from loguru import logger
from src.data_fetcher import DataFetcher

# 初始化日志
logger.add("test_local_db_priority.log", rotation="500 MB", level="INFO")

print("\n" + "="*70)
print("测试：验证 DataFetcher 优先使用本地数据库")
print("="*70 + "\n")

# 测试1: 只指定 local_db_path，不指定 primary_source
print("[测试1] 只指定 local_db_path，不指定 primary_source...")
print("代码: data_fetcher = DataFetcher(local_db_path='D:/tu-shareData/astock_daily.db')")
print()

data_fetcher = DataFetcher(local_db_path="D:/tu-shareData/astock_daily.db")

print(f"\n结果:")
print(f"  primary_source: {data_fetcher.primary_source}")
print(f"  local_db_available: {data_fetcher.local_db_available}")
print(f"  期望: primary_source='local_db', local_db_available=True")
print()

if data_fetcher.primary_source == "local_db" and data_fetcher.local_db_available:
    print("[OK] 测试1通过: 自动设置 primary_source 为 'local_db'")
else:
    print("[FAIL] 测试1失败: primary_source 未自动切换")
print("\n" + "-"*70 + "\n")

# 测试2: 指定 primary_source="akshare"，但同时指定 local_db_path
print("[测试2] 指定 primary_source='akshare'，但同时指定 local_db_path...")
print("代码: data_fetcher = DataFetcher(primary_source='akshare', local_db_path='D:/tu-shareData/astock_daily.db')")
print()

data_fetcher2 = DataFetcher(
    primary_source="akshare",
    local_db_path="D:/tu-shareData/astock_daily.db"
)

print(f"\n结果:")
print(f"  primary_source: {data_fetcher2.primary_source}")
print(f"  local_db_available: {data_fetcher2.local_db_available}")
print(f"  期望: primary_source='local_db' (自动覆盖), local_db_available=True")
print()

if data_fetcher2.primary_source == "local_db" and data_fetcher2.local_db_available:
    print("[OK] 测试2通过: 即使指定了 primary_source='akshare'，仍自动切换为 'local_db'")
else:
    print("[FAIL] 测试2失败: primary_source 未自动切换")
print("\n" + "-"*70 + "\n")

# 测试3: 不指定 local_db_path
print("[测试3] 不指定 local_db_path...")
print("代码: data_fetcher = DataFetcher()")
print()

data_fetcher3 = DataFetcher()

print(f"\n结果:")
print(f"  primary_source: {data_fetcher3.primary_source}")
print(f"  local_db_available: {data_fetcher3.local_db_available}")
print(f"  期望: primary_source='akshare' (默认值), local_db_available=False")
print()

if data_fetcher3.primary_source == "akshare" and not data_fetcher3.local_db_available:
    print("[OK] 测试3通过: 未指定 local_db_path，使用默认的 'akshare'")
else:
    print("[FAIL] 测试3失败")
print("\n" + "-"*70 + "\n")

# 测试4: 指定无效的 local_db_path
print("[测试4] 指定无效的 local_db_path...")
print("代码: data_fetcher = DataFetcher(local_db_path='C:/invalid/path/db.db')")
print()

data_fetcher4 = DataFetcher(local_db_path="C:/invalid/path/db.db")

print(f"\n结果:")
print(f"  primary_source: {data_fetcher4.primary_source}")
print(f"  local_db_available: {data_fetcher4.local_db_available}")
print(f"  期望: primary_source='akshare' (默认值), local_db_available=False")
print()

if data_fetcher4.primary_source == "akshare" and not data_fetcher4.local_db_available:
    print("[OK] 测试4通过: 无效的 local_db_path，使用默认的 'akshare'")
else:
    print("[FAIL] 测试4失败")
print("\n" + "="*70)
print("测试完成!")
print("="*70 + "\n")

# 总结
print("总结:")
print("="*70)
print("修改后的逻辑:")
print("  1. 如果指定了 local_db_path 且数据库有效 -> 自动设置 primary_source='local_db'")
print("  2. 如果未指定 local_db_path 或数据库无效 -> 使用默认的 primary_source='akshare'")
print("  3. 这样可以确保：有本地数据库时优先使用，无需手动设置 primary_source")
print("="*70 + "\n")
