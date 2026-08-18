# -*- coding: utf-8 -*-
"""安全落地: 在 run_backtest.bat 的 ETF 轮动菜单新增 [5] (V6 4只版) 选项。
调用原始4ETF纯版 run_etf_rotation_v6.py (红利/创业50/纳指/黄金, 不参与其他ETF轮动)。
GBK 安全: 单次 pass, 锚点定位, 不依赖行索引, 不破坏现有 1-4 模式。
"""
import sys

BAT = r"run_backtest.bat"

raw = open(BAT, "rb").read()
try:
    text = raw.decode("gbk")
except Exception as e:
    print("GBK 解码失败:", e); sys.exit(1)
lines = text.split("\r\n")

# --- 1) 菜单提示 (1-4 -> 1-5) ---
hit1 = False
for i, ln in enumerate(lines):
    if ln.strip().startswith("set /p ETF_MODE="):
        if "(1-4," in ln:
            lines[i] = ln.replace("(1-4,", "(1-5,")
            print("菜单提示已改:", lines[i].strip())
            hit1 = True
        break
if not hit1:
    print("WARN: 未找到 set /p ETF_MODE= 的 (1-4 提示, 跳过")

# --- 2) 菜单项 [5] 插入 (在 [4] 原版平台ETF轮动 行之后) ---
hit2 = False
for i, ln in enumerate(lines):
    if "[4] 原版平台ETF轮动" in ln:
        lines.insert(i + 1, 'echo   [5] (V6 4只版) 红利/创业50/纳指/黄金 · 纯RSRS+动量')
        print("已插入菜单项 [5]:", lines[i + 1].strip())
        hit2 = True
        break
if not hit2:
    print("ERROR: 未找到 [4] 原版平台ETF轮动 锚点, 中止")
    sys.exit(1)

# --- 3) 执行块 mode5 分支 (在 if "%ETF_LEGACY%"=="1" 之前) ---
hit3 = False
for i, ln in enumerate(lines):
    if ln.strip().startswith('if "%ETF_LEGACY%"=="1"'):
        block = [
            'if "%ETF_MODE%"=="5" (',
            '  "venv_ml\\Scripts\\python.exe" run_etf_rotation_v6.py',
            '  echo.',
            '  echo   [返回主菜单]',
            '  pause',
            '  goto :menu',
            ')',
        ]
        for off, b in enumerate(block):
            lines.insert(i + off, b)
        print("已插入执行块 mode5 分支 (共 %d 行)" % len(block))
        hit3 = True
        break
if not hit3:
    print("ERROR: 未找到 if \"%ETF_LEGACY%\"==\"1\" 锚点, 中止")
    sys.exit(1)

# --- 写回 GBK ---
out = "\r\n".join(lines)
open(BAT, "wb").write(out.encode("gbk"))
print("已写回", BAT, "| GBK 编码 OK | 总行数", len(lines))
