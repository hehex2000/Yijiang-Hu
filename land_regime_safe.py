# -*- coding: utf-8 -*-
"""安全落地 ETF 轮动 Regime 到 run_backtest.bat（恢复自 HEAD 的干净原版）。
做法：GBK 解码 -> 单次 pass 两段替换 -> GBK 写回。
关键：先在列表上换菜单块，再于"已改列表"上重定位 :etf_exec 块，避免索引漂移。
新菜单文本不含希腊字母 β（GBK 不含），规避编码报错。"""
BAT = r"C:\Users\99395\WorkBuddy\multi_factor_selection\run_backtest.bat"

with open(BAT, "rb") as f:
    raw = f.read()
text = raw.decode("gbk")
lines = text.split("\r\n")

# ---- 1) 菜单块：:etf_rotation .. goto :etf_var ----
i_menu = lines.index(":etf_rotation")
i_goto_var = next(j for j in range(i_menu + 1, len(lines)) if lines[j].strip() == "goto :etf_var")
new_menu = """:etf_rotation
cls
echo.
echo ========================================
echo   ETF轮动策略 [V6 + Regime兜底 · 生产版]
echo ========================================
echo.
echo   标的池[20只精选ETF · 真实价格] + Regime牛熊识别(沪深300 MA200+市场宽度):
echo     沪深300 - 上证50 - 中证500 - 中证1000
echo     创业板50 - 科创50 - 证券 - 医药
echo     消费 - 5G通信 - 新能源车 - 旅游
echo     黄金 - 红利 - 货币 - 更多...
echo.
echo   核心逻辑：
echo     ROC动量×0.4 + 中期动量×0.4 - 波动率×0.2[手册推荐权重]
echo     MA60过滤+追高保护+最小切换阈值[防抖动]
echo     + RSRS质量分(可开关) + Regime兜底(牛市强制≥40%%宽基)
echo     全部走弱时转货币基金避险，保留10%%现金缓冲
echo.
echo   回测区间: %BACKTEST_START% ~ %BACKTEST_END%
echo.
echo   请选择运行模式:
echo   ----------------
echo   [1] V6+Regime生产版[默认·已替代原ETF轮动]  --rsrs 0
echo   [2] V6+Regime+RSRS0.3[带质量分]
echo   [3] V6合并版(无Regime·纯对照)  --regime off
echo   [4] 原版平台ETF轮动(审计对照基线·保留)
echo   [0] 返回主菜单
echo.
set ETF_POOL_ARG=
set ETF_VAR_ARG=
set ETF_TOPN_ARG=
set ETF_VARSEL=
set ETF_PREMIUM_ARG=
set ETF_MA_ARG=
set ETF_MODE=
set /p ETF_MODE=请选择 (1-4, 默认1):
if "%ETF_MODE%"=="0" goto :menu
if "%ETF_MODE%"=="" set ETF_MODE=1
set ETF_REGIME=off
set ETF_RSRS=0
set ETF_LEGACY=0
if "%ETF_MODE%"=="1" set ETF_REGIME=on
if "%ETF_MODE%"=="2" (set ETF_REGIME=on & set ETF_RSRS=0.3)
if "%ETF_MODE%"=="3" set ETF_REGIME=off
if "%ETF_MODE%"=="4" set ETF_LEGACY=1
goto :etf_var""".split("\n")
lines[i_menu:i_goto_var + 1] = new_menu

# ---- 2) 执行块：:etf_exec 与 goto :menu 之间的 body（重定位，已含菜单替换）----
i_exec = lines.index(":etf_exec")
i_goto_menu = next(j for j in range(i_exec + 1, len(lines)) if lines[j].strip() == "goto :menu")
new_body = """echo.
echo   正在运行...
if "%ETF_LEGACY%"=="1" (
  "venv_ml\\Scripts\\python.exe" run_etf_rotation.py %BACKTEST_START% %BACKTEST_END% --method dual %ETF_MA_ARG% %ETF_POOL_ARG% %ETF_TOPN_ARG% %ETF_VAR_ARG% %ETF_PREMIUM_ARG%
) else (
  "venv_ml\\Scripts\\python.exe" run_etf_rotation_v6_merged.py %BACKTEST_START% %BACKTEST_END% --method dual %ETF_MA_ARG% %ETF_POOL_ARG% %ETF_TOPN_ARG% %ETF_VAR_ARG% %ETF_PREMIUM_ARG% --rsrs-weight %ETF_RSRS% --regime %ETF_REGIME%
)
echo.
echo   [返回主菜单]
pause""".split("\n")
lines[i_exec + 1:i_goto_menu] = new_body

out = "\r\n".join(lines)
with open(BAT, "wb") as f:
    f.write(out.encode("gbk"))

# ---- 校验 ----
checks = {
    "主菜单标签 :menu": ":menu" in [l.strip() for l in lines],
    "入口标签 :menu_start": ":menu_start" in [l.strip() for l in lines],
    "set /p CHOICE 存在": "set /p CHOICE" in out,
    "无 ETF_V6 残留": "ETF_V6" not in out,
    "菜单[1]存在": "[1] V6+Regime生产版" in out,
    "原版[4]基线存在": "原版平台ETF轮动(审计对照基线·保留)" in out,
}
print("替换完成。菜单块 %d 行 -> %d 行；执行块重定位 idx %d -> %d" % (
    i_goto_var - i_menu + 1, len(new_menu), i_goto_menu, i_exec + 1 + len(new_body)))
for k, v in checks.items():
    print(("  [OK] " if v else "  [!!] ") + k + ": " + str(v))
