#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修改 config.py 中的 BACKTEST["total_capital"] 参数（选股族回测总初始资金）
使用方法: python update_capital.py <新的总资金值>
"""

import sys
import re


def update_capital(new_value):
    """修改 config.py 中 BACKTEST 字典里的 total_capital 参数"""
    with open('config.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 找到 BACKTEST 字典的范围（匹配花括号）
    backtest_start = content.find('BACKTEST = {')
    if backtest_start == -1:
        print("[ERROR] 在 config.py 中未找到 BACKTEST 字典！")
        sys.exit(1)

    brace_count = 0
    backtest_end = None
    for i in range(backtest_start, len(content)):
        if content[i] == '{':
            brace_count += 1
        elif content[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                backtest_end = i + 1
                break

    if backtest_end is None:
        print("[ERROR] 无法解析 BACKTEST 字典的范围！")
        sys.exit(1)

    backtest_content = content[backtest_start:backtest_end]

    # 按行查找并替换 "total_capital": <数字>
    lines = backtest_content.split('\n')
    found = False
    for i, line in enumerate(lines):
        if '"total_capital"' in line:
            lines[i] = re.sub(r'"total_capital"\s*:\s*\d+', f'"total_capital": {new_value}', line)
            found = True
            break

    if not found:
        print("[ERROR] 在 BACKTEST 字典中未找到 total_capital 参数！")
        sys.exit(1)

    new_backtest_content = '\n'.join(lines)
    new_content = content[:backtest_start] + new_backtest_content + content[backtest_end:]

    with open('config.py', 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"[OK] config.py 已成功更新: BACKTEST[\"total_capital\"] = {new_value}")
    print(f"   (选股族回测总初始资金已更新，每支资金 = 总资金 // 选股数量)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用方法: python update_capital.py <新的总资金值>")
        print("示例: python update_capital.py 200000")
        sys.exit(1)

    try:
        new_value = int(sys.argv[1])
        if new_value < 100000:
            print("[ERROR] 总初始资金必须 >= 100000")
            sys.exit(1)
        update_capital(new_value)
    except ValueError:
        print("[ERROR] 请输入有效的整数")
        sys.exit(1)
