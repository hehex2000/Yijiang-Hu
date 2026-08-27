# -*- coding: utf-8 -*-
"""ETF版MACD择时 分年表：逐年纪录 满仓 vs MACD择时 收益差，钉死超额来源年份。
用法:
  ./venv_ml/Scripts/python.exe etf_macd_yearly.py --etf 515800 --start 20100101 --split-year 2013
  ./venv_ml/Scripts/python.exe etf_macd_yearly.py --etf 515800 --start 20190701
  ./venv_ml/Scripts/python.exe etf_macd_yearly.py --etf 510300 --start 20130701 --split-year 2019
"""
import argparse
import numpy as np
import pandas as pd

import macd_plugin_validate as M
from regime_cash_overlay import load_index_close, apply_overlay

ETF_MAP = {
    '510300': ('000300.SH', '沪深300ETF(510300.SH, 2012-05-28上市)'),
    '515800': ('000906.SH', '中证800ETF(515800.SH, 2019-01上市)'),
}
SIG = '000300.SH'


def load(etf, start, end, etf_cost=0.002):
    base_code, _ = ETF_MAP[etf]
    base = M.load_base_index(base_code, start, end)
    dc = 1.0 - (1.0 - etf_cost) ** (1.0 / 252.0)
    r = base.pct_change().fillna(0.0)
    nav = (1.0 + r - dc).cumprod()
    nav = nav / nav.iloc[0]
    nav.index = base.index
    hs = load_index_close(SIG, start, end).reindex(base.index).ffill()
    golden = M.macd_golden(hs.values).values
    nav_m = apply_overlay(nav.values, golden, cash_growth=1.0)
    nav_m.index = base.index   # apply_overlay 传数组时索引退化，补回交易日
    return nav, nav_m, golden


def yearly(nav, nav_m, golden):
    idx = np.asarray(nav.index)
    nav_arr = np.asarray(nav, float)
    nav_m_arr = np.asarray(nav_m, float)
    years = sorted(set(idx // 10000))
    rows = []
    for y in years:
        pos = np.where(idx // 10000 == y)[0]
        last = pos[-1]; prev = pos[0] - 1
        s_bh = nav_arr[prev] if prev >= 0 else 1.0
        s_md = nav_m_arr[prev] if prev >= 0 else 1.0
        rb = nav_arr[last] / s_bh - 1
        rm = nav_m_arr[last] / s_md - 1
        sw = sum(1 for i in pos if i > 0 and golden[i] != golden[i-1])
        cash = int(np.sum(~golden[pos]))
        rows.append((y, rb, rm, rm - rb, sw, cash))
    return rows


def cum(nav, y0, y1):
    idx = np.asarray(nav.index); arr = np.asarray(nav, float)
    pos = np.where((idx >= y0 * 10000) & (idx <= y1 * 10000))[0]
    if len(pos) == 0: return None
    prev = pos[0] - 1
    s = arr[prev] if prev >= 0 else 1.0
    return arr[pos[-1]] / s - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--etf', default='515800', choices=sorted(ETF_MAP))
    ap.add_argument('--start', default='20190701')
    ap.add_argument('--end', default='20251231')
    ap.add_argument('--split-year', type=int, default=None, help='在N年处切分前后段累计对比')
    args = ap.parse_args()

    nav, nav_m, golden = load(args.etf, args.start, args.end)
    rows = yearly(nav, nav_m, golden)

    def pct(x): return f"{x*100:+.2f}"
    base_code, label = ETF_MAP[args.etf]
    print(f"\n{'='*88}")
    print(f"  分年表 | {label} | {args.start}~{args.end} | 信号=沪深300 MACD | ETF年成本0.2%")
    print(f"{'='*88}")
    print(f"  {'年份':<8}{'满仓%':>10}{'MACD%':>10}{'Δpp':>9}{'切换':>6}{'空仓天':>8}")
    for y, rb, rm, dd, sw, cash in rows:
        print(f"  {y:<8}{pct(rb):>10}{pct(rm):>10}{dd*100:+8.1f} {sw:>5} {cash:>7}")
    # 全段
    rb_all = nav.iloc[-1] / nav.iloc[0] - 1
    rm_all = nav_m.iloc[-1] / nav_m.iloc[0] - 1
    print(f"  {'全段':<8}{pct(rb_all):>10}{pct(rm_all):>10}{ (rm_all-rb_all)*100:+8.1f}")
    # 分年胜率
    wins = sum(1 for _, rb, rm, dd, _, _ in rows if rm > rb)
    print(f"  [MACD跑赢年份] {wins}/{len(rows)}")
    if args.split_year:
        s = args.split_year
        e = int(args.end[:4])
        y0a = int(args.start[:4])
        for seg, y0, y1 in [('前段', y0a, s - 1), ('后段', s, e)]:
            cb = cum(nav, y0, y1); cm = cum(nav_m, y0, y1)
            if cb is not None:
                print(f"  [{seg} {y0}-{y1}] 满仓{pct(cb)} | MACD{pct(cm)} | Δ{(cm-cb)*100:+.1f}pp")


if __name__ == '__main__':
    main()
