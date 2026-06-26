import re

with open("run_backtest.bat", "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    
    # 1. 在选项2后插入选项3
    if 'echo   [2] 价值投资选股' in line:
        new_lines.append(line)
        new_lines.append('\necho   [3] 红利低波选股（高股息率+低波动率）\n')
        i += 1
        continue
    
    # 2. 更新提示从 (1-2) 到 (1-3)
    if '请选择 (1-2):' in line:
        line = line.replace('请选择 (1-2):', '请选择 (1-3):')
        new_lines.append(line)
        i += 1
        continue
    
    # 3. 在 goto :set_method 前插入选项3的处理
    if 'goto :set_method' in line and i > 0 and 'METHOD_CHOICE' not in lines[i-1]:
        # 找到 goto :set_method，在其前插入
        new_lines.append('if "%METHOD_CHOICE%"=="3" (\n')
        new_lines.append('    set SELECTION_METHOD=div_low_vol\n')
        new_lines.append('    echo   ✓ 已选择: 红利低波选股\n')
        new_lines.append('    pause\n')
        new_lines.append('    goto :menu\n')
        new_lines.append(')\n')
        new_lines.append(line)
        i += 1
        continue
    
    new_lines.append(line)
    i += 1

with open("run_backtest.bat", "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("✓ run_backtest.bat 更新完成")
