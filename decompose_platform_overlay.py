# -*- coding: utf-8 -*-
"""分解 platform_stop_overlay：止损 vs MACD 各自贡献多少。
复用 macd_plugin_validate 的基线与指标函数，仅改变 hold 掩码做 A/B/C/D 对照。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

import macd_plugin_validate as M
from regime_cash_overlay import load_index_close, BENCH, apply_overlay, cash_ratio

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_decomp_cache.pkl')
START, END = '20100101', '20251231'

# ── 加载基线 + 沪深300（缓存避免每次2分钟重算）──
if os.path.exists(CACHE):
    base, hs = pd.read_pickle(CACHE)
    print(f"[cache] 命中基线/沪深300，长度 {len(base)}")
else:
    base = M.load_base_zz800_eq(START, END)
    hs = load_index_close(BENCH, START, END)
    hs = hs.reindex(base.index).ffill()
    pd.to_pickle((base, hs), CACHE)
    print(f"[load] 已缓存")

base_v = base.values.astype(float)
hs_v = hs.values.astype(float)

# ── 三层信号 ──
peak = pd.Series(base_v).cummax().values
trailing_hit = base_v < peak * (1 - 0.15)                      # 组合自身 -15% 触止损
golden = M.macd_golden(hs_v).values                           # 沪深300 MACD 金叉
# 止损锁存
stopped = np.zeros(len(base_v), dtype=bool); s = False
for i in range(len(base_v)):
    if trailing_hit[i]: s = True
    if golden[i]:       s = False
    stopped[i] = s

A = golden & (~stopped)      # 当前：MACD + 止损
B = ~stopped                 # 仅止损
C = golden                   # 仅MACD
D = np.ones(len(base_v), dtype=bool)  # 无控制（满仓）

def run(mask, name):
    nav = apply_overlay(base_v, mask)
    rb, ab, mdb, sb = M.metrics(nav)
    cr = cash_ratio(mask) * 100
    return name, rb, ab, mdb, sb, cr, nav

rows = [run(D, 'D 无控制(满仓等权800)'),
        run(B, 'B 仅15%止损'),
        run(C, 'C 仅沪深300 MACD'),
        run(A, 'A 止损+MACD(当前)')]

def pct(x): return f"{x*100:+.2f}%"
print(f"\n{'='*92}")
print(f"  等权中证800 风控层分解 | 止损=组合峰回撤15% | MACD=沪深300金叉")
print(f"{'='*92}")
print(f"  {'方案':<22}{'总收益':>10}{'年化':>9}{'最大回撤':>10}{'Sharpe':>9}{'持币%':>8}")
for nm, rb, ab, mdb, sb, cr, _ in rows:
    print(f"  {nm:<22}{pct(rb):>10}{pct(ab):>9}{pct(mdb):>10}{sb:>9.2f}{cr:>7.1f}%")

# 增量贡献
_, rbD,_,mdbD,_,_,_ = rows[0]
_, rbB,_,mdbB,_,_,_ = rows[1]
_, rbC,_,mdbC,_,_,_ = rows[2]
_, rbA,_,mdbA,_,_,_ = rows[3]
print(f"\n  ── 增量贡献（相对无控制 D）──")
print(f"  仅止损 B : Δ总收益={pct(rbB-rbD)}  Δ回撤={pct(mdbB-mdbD)}")
print(f"  仅MACD C : Δ总收益={pct(rbC-rbD)}  Δ回撤={pct(mdbC-mdbD)}")
print(f"  A-B(纯MACD增量) : Δ总收益={pct(rbA-rbB)}  Δ回撤={pct(mdbA-mdbB)}")
print(f"  A-C(纯止损增量) : Δ总收益={pct(rbA-rbC)}  Δ回撤={pct(mdbA-mdbC)}")
print(f"  结论：若 A-B≈0 则 MACD 几乎无增量（只是止损）；若 A-C≈0 则止损无增量。")
