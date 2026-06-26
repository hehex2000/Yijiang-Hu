#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修改 config.py 中的 GLOBAL["top_n"] 参数（统一配置）
使用方法: python update_top_n.py <新的top_n值>
"""

import sys
import re

def update_top_n(new_value):
    """修改 config.py 中 GLOBAL 字典里的 top_n 参数"""
    with open('config.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
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
    
    # 按行查找并替换 "top_n": <数字>
    lines = global_content.split('\n')
    found = False
    for i, line in enumerate(lines):
        if '"top_n"' in line:
            lines[i] = re.sub(r'"top_n"\s*:\s*\d+', f'"top_n": {new_value}', line)
            found = True
            break
    
    if not found:
        print("[ERROR] 在 GLOBAL 字典中未找到 top_n 参数！")
        sys.exit(1)
    
    new_global_content = '\n'.join(lines)
    new_content = content[:global_start] + new_global_content + content[global_end:]
    
    with open('config.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"[OK] config.py 已成功更新: GLOBAL[\"top_n\"] = {new_value}")
    print(f"   (所有策略的 top_n 将自动使用此值)")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用方法: python update_top_n.py <新的top_n值>")
        print("示例: python update_top_n.py 10")
        sys.exit(1)
    
    try:
        new_value = int(sys.argv[1])
        if new_value <= 0:
            print("[ERROR] top_n 必须是正整数")
            sys.exit(1)
        update_top_n(new_value)
    except ValueError:
        print("[ERROR] 请输入有效的整数")
        sys.exit(1)
