# -*- coding: utf-8 -*-
"""逐年度 ON vs OFF 表（rsrs=0 纯平台），用于报告 §8.6。"""
import contextlib, io
import run_etf_rotation_v6_merged as M
from regime_core import build_regime_hook
RSRS = 0.0
YEARS = [("2020","20200101","20201231"),("2021","20210101","20211231"),
         ("2022","20220101","20221231"),("2023","20230101","20231231"),
         ("2024","20240101","20241231"),("2025","20250101","20251231"),
         ("2026(至08)","20260101","20260831")]
def run(s,e,on):
    h = build_regime_hook() if on else None
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf):
        return M.run_etf_rotation(s,e,rsrs_weight=RSRS,regime_hook=h,verbose=False)
print(f"{'年度':<10} | {'OFF总收益':>9} {'OFF超额':>8} | {'ON总收益':>9} {'ON超额':>8} | {'Δ超额':>7}")
print("-"*70)
for name,s,e in YEARS:
    o=run(s,e,False); n=run(s,e,True)
    eo=o['total_return']-o['idx_return']; en=n['total_return']-n['idx_return']
    print(f"{name:<10} | {o['total_return']:+8.2f}% {eo:+7.2f}pp | {n['total_return']:+8.2f}% {en:+7.2f}pp | {en-eo:+6.2f}pp")
