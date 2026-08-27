# -*- coding: utf-8 -*-
"""ETF 替代可行性：把"等权800篮子"换成可交易的宽基指数(ETF proxies)。
- 000906.SH 市值加权 = 中证800ETF(如515800) 跟踪标的
- 000300.SH 市值加权 = 沪深300ETF(如510300) 跟踪标的（同时与MACD信号同源）
对比理论等权800(已证 +433%)，看可操作版本的真实回报。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import macd_plugin_validate as M
from regime_cash_overlay import load_index_close, BENCH, apply_overlay, cash_ratio

def run(base_code, sig_code, label):
    base = M.load_base_index(base_code)
    sig = load_index_close(sig_code).reindex(base.index).ffill()
    base_v, sig_v = base.values.astype(float), sig.values.astype(float)
    golden = M.macd_golden(sig_v).values
    peak = pd.Series(base_v).cummax().values
    trailing = base_v < peak*0.85
    stopped = np.zeros(len(base_v), bool); s=False
    for i in range(len(base_v)):
        if trailing[i]: s=True
        if golden[i]:  s=False
        stopped[i]=s
    masks = {'满仓持有': np.ones(len(base_v),bool),
             '仅MACD择时': golden,
             '止损+MACD': golden & (~stopped)}
    print(f"\n── {label} (base={base_code}, signal={sig_code}) ──")
    print(f"  {'方案':<14}{'总收益':>10}{'年化':>9}{'最大回撤':>10}{'Sharpe':>9}{'持币%':>8}")
    for nm,m in masks.items():
        nav = apply_overlay(base_v, m); rb,ab,mdb,sb = M.metrics(nav)
        cr = cash_ratio(m)*100
        print(f"  {nm:<14}{rb*100:+9.2f}%{ab*100:+8.2f}%{mdb*100:+9.2f}%{sb:>9.2f}{cr:>7.1f}%")

run('000906.SH','000300.SH','中证800ETF代理(市值加权, 信号用沪深300)')
run('000300.SH','000300.SH','沪深300ETF代理(信号=持仓同源, 最干净可落地)')
print("\n[对照] 理论等权800 + MACD择时 = +433.84% / -32.77% / Sharpe0.73 / 持币49.5%")
