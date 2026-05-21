#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试Tushare API调用（中证500/中证1000）"""

import tushare as ts
import os

# 读取token
config_path = r"C:\Users\99395\WorkBuddy\Tushare-Downloader\config.py"
with open(config_path, 'r') as f:
    for line in f:
        if line.startswith('TUSHARE_TOKEN'):
            token = line.split('=')[1].strip().strip('"').strip("'")
            break

ts.set_token(token)
pro = ts.pro_api()

print("✓ Token设置成功")
print("\n测试1: index_member (中证500)...")
try:
    df = pro.index_member(index_code='000905.SH')
    if df is not None and len(df) > 0:
        print(f"  ✓ 成功！获取到 {len(df)} 条数据")
        print(f"  列名: {list(df.columns)}")
        print(f"  前3行:\n{df.head(3)}")
    else:
        print(f"  ❌ 无数据")
except Exception as e:
    print(f"  ❌ 失败: {e}")
    import traceback
    traceback.print_exc()

print("\n测试2: index_member (中证1000)...")
try:
    df = pro.index_member(index_code='000852.SH')
    if df is not None and len(df) > 0:
        print(f"  ✓ 成功！获取到 {len(df)} 条数据")
        print(f"  列名: {list(df.columns)}")
        print(f"  前3行:\n{df.head(3)}")
    else:
        print(f"  ❌ 无数据")
except Exception as e:
    print(f"  ❌ 失败: {e}")

print("\n测试3: index_weight (中证500, trade_date=20240102)...")
try:
    df = pro.index_weight(index_code='000905.SH', trade_date='20240102')
    if df is not None and len(df) > 0:
        print(f"  ✓ 成功！获取到 {len(df)} 条数据")
        print(f"  列名: {list(df.columns)}")
    else:
        print(f"  ❌ 无数据")
except Exception as e:
    print(f"  ❌ 失败: {e}")

print("\n全部测试完成！")
