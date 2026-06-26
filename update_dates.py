#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修改 config.py 中的 GLOBAL["backtest_start"] 和 GLOBAL["backtest_end"] 参数
使用方法: python update_dates.py <start_date> <end_date>
"""

import sys
import re

def update_dates(start_date, end_date):
    """修改 config.py 中 GLOBAL 字典里的日期参数"""
    with open('config.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 验证日期格式（YYYYMMDD）
    if not (len(start_date) == 8 and start_date.isdigit()):
        print(f"[ERROR] 开始日期格式错误: {start_date} (应为 YYYYMMDD)")
        sys.exit(1)
    if not (len(end_date) == 8 and end_date.isdigit()):
        print(f"[ERROR] 结束日期格式错误: {end_date} (应为 YYYYMMDD)")
        sys.exit(1)
    
    # 找到 GLOBAL 字典的范围（匹配花括号）
    global_start = content.find('GLOBAL = {')
    if global_start == -1:
        print("[ERROR] 在 config.py 中未找到 GLOBAL 字典！")
        sys.exit(1)
    
    brace_count = 0
    global_end = None
    for i in range(global_start, len(content)):
        if content[i] == '{':
            brace_count += 1
        elif content[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                global_end = i + 1
                break
    
    if global_end is None:
        print("[ERROR] 无法解析 GLOBAL 字典的范围！")
        sys.exit(1)
    
    global_content = content[global_start:global_end]
    
    # 按行查找并替换
    lines = global_content.split('\n')
    found_start = False
    found_end = False
    for i, line in enumerate(lines):
        if '"backtest_start"' in line:
            lines[i] = re.sub(r'"backtest_start"\s*:\s*"[^"]*"', f'"backtest_start": "{start_date}"', line)
            found_start = True
        elif '"backtest_end"' in line:
            lines[i] = re.sub(r'"backtest_end"\s*:\s*"[^"]*"', f'"backtest_end": "{end_date}"', line)
            found_end = True
        if found_start and found_end:
            break
    
    if not found_start or not found_end:
        print(f"[ERROR] 在 GLOBAL 字典中未找到日期参数！(start={found_start}, end={found_end})")
        sys.exit(1)
    
    new_global_content = '\n'.join(lines)
    new_content = content[:global_start] + new_global_content + content[global_end:]
    
    with open('config.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"[OK] config.py 已成功更新:")
    print(f"   GLOBAL[\"backtest_start\"] = \"{start_date}\"")
    print(f"   GLOBAL[\"backtest_end\"]   = \"{end_date}\"")
    print(f"   (所有策略的回测时间范围将自动使用此值)")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("使用方法: python update_dates.py <开始日期> <结束日期>")
        print("示例: python update_dates.py 20260102 20260618")
        sys.exit(1)
    
    update_dates(sys.argv[1], sys.argv[2])
