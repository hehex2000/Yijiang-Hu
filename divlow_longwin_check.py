# -*- coding: utf-8 -*-
"""
长窗口验证：降频（--rebal year）够不够格进回测平台？

背景：降频把年化单边换手从 142.3% 打到 50.7%（−64%），这是**算术恒等式**级的机制，可信。
但 2020-2026 窗口的年度档只有 **6 个调仓点**，收益差异全部不显著（配对 t |t|<0.95）。
进平台 = 变成可复用的默认策略 → 必须先补样本量。

窗口选择依据（实测股息覆盖率，红利策略的命脉）：
    2010 46.4% / 2011 53.1% / 2012 62.2% / 2013 65.2% / 2014+ 稳定 60~65%
→ 2013 年之前候选池被**数据缺失**人为压缩，不是市场真实状态，会扭曲结果。
→ 安全窗口 = 2013-01-01 ~ 最新，年度档 13 期（比 2020 起点的 6 期翻倍）。

用法（🔴 必须串行，不要并行：_preload_allA_prices 会灌全A全历史，内存吃紧会 OOM）：
    venv_ml/Scripts/python.exe -u divlow_longwin_check.py year
    venv_ml/Scripts/python.exe -u divlow_longwin_check.py quarter
"""
import sys

import run_dividend_low_vol_quality_bt as E

# 🔴 引擎窗口是硬编码模块级常量（L357-358），CLI 没有 --start/--end
#    → 只能通过模块属性覆盖（平台层 run_backtest.py 就是这么做的）
E.START = "20130101"
E.END = "20260903"

freq = (sys.argv[1] if len(sys.argv) > 1 else "year")
if freq not in E.REBAL_SPECS:
    raise SystemExit(f"rebal 必须是 {list(E.REBAL_SPECS)} 之一，收到 {freq!r}")

# 🔴 必须改 MODE_SPECS 本体（全局），不是局部副本 —— 与 PRICE_MODE 同类坑
E.MODE_SPECS["official_compact"]["rebal"] = freq
print(f"[longwin] 调仓频率={freq}（{E.REBAL_SPECS[freq][1]}）  窗口 {E.START} ~ {E.END}")
print(f"[longwin] 口径: 全A池 / 持仓12 / hfq / 关闭红利通道 overlay(满仓基线)")

E.run_official_backtest("official_compact", pool="all", top_n=12, overlay=False)
