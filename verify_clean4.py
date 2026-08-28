# -*- coding: utf-8 -*-
"""④ 自利性分红护栏 A/B 对照：2023/2024/2025 三调仓日，④关 vs ④开。"""
import os, sys, io, contextlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_monthly_rebalance as M

DATES = ['20230331', '20240329', '20250328']
BANKS = {'601398.SH', '601288.SH', '601939.SH', '601988.SH', '601328.SH'}

print(f"{'调仓日':<12}{'④关篮内':>10}{'④开篮内':>10}{'④额外剔除':>12}")
print("-" * 46)
for d in DATES:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        off = M.select_dividend_low_vol_clean(d, top_n=20, require_low_pledge=False)
        on = M.select_dividend_low_vol_clean(d, top_n=20, require_low_pledge=True)
    off_set = set(off['ts_code'])
    on_set = set(on['ts_code'])
    dropped = off_set - on_set
    print(f"{d:<12}{len(off):>10}{len(on):>10}{len(dropped):>12}")
    if dropped:
        names = ', '.join(f"{c}({M.get_stock_name(c)})" for c in sorted(dropped))
        print(f"   ④剔除: {names}")
    banks_off = BANKS & off_set
    banks_on = BANKS & on_set
    print(f"   银行核心(工农中建交) ④关保留 {len(banks_off)}/5  ④开保留 {len(banks_on)}/5")
