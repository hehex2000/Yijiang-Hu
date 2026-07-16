#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修改 config.py 中的 GLOBAL["stock_pool"] 参数（统一配置）
使用方法: python update_stock_pool.py <新的stock_pool值>
"""

import sys
import re

def update_stock_pool(new_pool):
    """修改 config.py 中 GLOBAL 字典里的 stock_pool 参数"""
    with open('config.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 验证输入值
    valid_pools = ['hs300', 'zz500', 'zz800', 'zz1000', 'all']
    if new_pool not in valid_pools:
        print(f"[ERROR] 无效的股票池: {new_pool}")
        print(f"  有效值: {', '.join(valid_pools)}")
        sys.exit(1)

    # 找到 GLOBAL 字典的范围（匹配花括号）
    global_start = content.find('GLOBAL = {')
    if global_start == -1:
        print("[ERROR] 在 config.py 中未找到 GLOBAL 字典！")
        sys.exit(1)

    # 找到 GLOBAL 字典的结束位置（匹配的 }）
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

    # 调试：显示 GLOBAL 字典内容
    # print(f"[DEBUG] GLOBAL 字典内容:\n{global_content}\n")

    # 在 GLOBAL 字典范围内替换 "stock_pool": "xxx"
    # 使用更可靠的方案：按行查找并替换
    lines = global_content.split('\n')
    found = False
    for i, line in enumerate(lines):
        if '"stock_pool"' in line:
            # 替换这一行中的值
            lines[i] = re.sub(r'"stock_pool"\s*:\s*"[^"]*"', f'"stock_pool": "{new_pool}"', line)
            found = True
            break

    if not found:
        print("[ERROR] 在 GLOBAL 字典中未找到 stock_pool 参数！")
        sys.exit(1)

    new_global_content = '\n'.join(lines)

    # 写回文件
    new_content = content[:global_start] + new_global_content + content[global_end:]

    with open('config.py', 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"[OK] config.py 已成功更新: GLOBAL[\"stock_pool\"] = \"{new_pool}\"")
    print(f"   (所有策略的 stock_pool 将自动使用此值)")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用方法: python update_stock_pool.py <新的stock_pool值>")
        print("示例: python update_stock_pool.py all")
        print("有效值: hs300, zz500, zz800, zz1000, all")
        sys.exit(1)

    update_stock_pool(sys.argv[1])
