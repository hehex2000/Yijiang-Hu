# -*- coding: utf-8 -*-
"""
A 档严格同窗口 A/B：你的版(旧红利低波月度) vs 我的版(M1 红利低波月度)
=====================================================================
用 M1 已校验费用/换手的 run_sim 引擎，同一窗口/同一月度首日调仓/同一 N=5/
同一等权/同一无择时(golden=全 True)，只换「选股器+池」，隔离框架效应：
  NEW  : M1.select_div_low_vol        (zz800, 无逐只MACD过滤)
  OLD-H: run_monthly_rebalance.select_dividend_low_vol_stocks
         (hs300, 纯 dv_ttm+波动率双排序, 关闭逐只MACD过滤)
  OLD-H-F: 同上但保留逐只MACD金叉过滤 (验证 M3「MACD类过滤减收益」)
关键修正：vol_lookup 与 macd_state 必须基于【全历史】行情(与 M1 main 一致)，
         否则 2015 年初波动率/历史不足→选股为空→首仓迟到牛市顶→数字失真。
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

import run_daily20_macd as D
import run_monthly_rebalance as RM
import macd_plugin_validate as M
from run_monthly_rebalance import get_trade_dates

START, END = '20150101', '20251231'
TOPN, CAP = 5, 1_000_000

# ── 数据：load_closes 返回全历史(2010-2025) ──
_, full_cf = D.load_closes()
closes = full_cf.loc[(full_cf.index >= int(START)) & (full_cf.index <= int(END))]
closes_ff = closes.ffill()

# 波动率查表（window=120，与 M1 / config 对齐）—— 必须基于全历史
vol_lookup = D.build_vol_lookup(full_cf, window=120)
_vol_dict = {c: vol_lookup[c].dropna().to_dict() for c in vol_lookup.columns}

def _fast_calvol(code, actual_date, window=None):
    d = _vol_dict.get(str(code))
    if not d:
        return None
    cand = [k for k in d if k <= int(actual_date)]
    if not cand:
        return None
    return d[max(cand)]

def _fast_macd_state(ts_code, trade_date, is_index_signal=False, regime_code=None,
                     regime_is_index=False, mode="golden"):
    """基于【全历史】closes 计算逐只 MACD 金叉状态。"""
    s = full_cf.get(str(ts_code))
    if s is None:
        return "death"
    c = s[s.index <= int(trade_date)].dropna()
    if len(c) < 35:
        return "death"
    v = c.values.astype(float)
    ef = pd.Series(v).ewm(span=12, adjust=False).mean().values
    es = pd.Series(v).ewm(span=26, adjust=False).mean().values
    dif = ef - es
    dea = pd.Series(dif).ewm(span=9, adjust=False).mean().values
    return "golden" if dif[-1] > dea[-1] else "death"

RM.calc_volatility = _fast_calvol
RM.macd_state = _fast_macd_state

# ── 月度首日调仓（与 M1 main 完全一致）──
trade_dates = get_trade_dates(START, END)
dates_i = [int(d) for d in trade_dates]
month_starts = set()
prev_ym = None
for d in trade_dates:
    ym = d[:6]
    if ym != prev_ym:
        month_starts.add(d); prev_ym = ym

golden = [True] * len(trade_dates)

# ── 选择器封装 ──
def sel_new(prev_td, top_n, vl, verbose=False):
    return D.select_div_low_vol(prev_td, top_n, vl, verbose=False)

def _old(prev_td, top_n, pool, vl=None, verbose=False):
    df = RM.select_dividend_low_vol_stocks(prev_td, top_n=top_n,
                                          macd_filter_mode="golden", stock_pool=pool)
    if df is None or len(df) == 0:
        return []
    return df['ts_code'].tolist()

def sel_old_hs(prev_td, top_n, vl=None, verbose=False):
    return _old(prev_td, top_n, "000300.SH")

def sel_old_hs_nofilter(prev_td, top_n, vl=None, verbose=False):
    # 关闭逐只MACD过滤：让 macd_state 恒返回 golden（所有候选都通过）
    saved = RM.macd_state
    RM.macd_state = lambda *a, **k: "golden"
    try:
        return _old(prev_td, top_n, "000300.SH")
    finally:
        RM.macd_state = saved

def run_one(sel_fn, tag, quiet=False):
    t0 = time.time()
    nav, tr, st = D.run_sim(trade_dates, dates_i, golden, closes, closes_ff,
                            TOPN, CAP, sel_fn, vol_lookup,
                            rebal_freq='monthly', month_starts=month_starts, verbose=False)
    rb, ab, mdb, sb = M.metrics(pd.Series(nav))
    msg = (f"{tag}: 总收={rb*100:7.2f}% 年化={ab*100:6.2f}% MDD={mdb*100:7.2f}% "
           f"Sharpe={sb:5.2f} 年化换手={st['turnover']/CAP*100:6.1f}% "
           f"总费={st['total_fee']:>9,.0f}  重选={st['n_reselect']}  ({time.time()-t0:.1f}s)")
    if not quiet:
        print(msg, flush=True)
    else:
        print(msg, flush=True)
    return nav, st

print("=" * 100, flush=True)
print(f"A/B 同窗口 {START}~{END} | N={TOPN} | 月度首日调仓 | 等权 | 无择时 | 本金{CAP:,}", flush=True)
print("=" * 100, flush=True)

nav_new, _ = run_one(sel_new, "NEW(zz800, 无逐只MACD过滤)")
nav_old_hs_nf, _ = run_one(sel_old_hs_nofilter, "OLD-H(hs300, 纯双排序无MACD过滤)")
nav_old_hs_f, _ = run_one(sel_old_hs, "OLD-H-F(hs300, 含逐只MACD金叉过滤)")

# 基准
b800 = M.load_base_index('000906.SH', START, END)
b300 = M.load_base_index('000300.SH', START, END)
rb8, ab8, md8, _ = M.metrics(b800)
rb3, ab3, md3, _ = M.metrics(b300)
print(f"基准中证800: 总收={rb8*100:7.2f}% 年化={ab8*100:6.2f}% MDD={md8*100:7.2f}%", flush=True)
print(f"基准沪深300: 总收={rb3*100:7.2f}% 年化={ab3*100:6.2f}% MDD={md3*100:7.2f}%", flush=True)

out = pd.DataFrame({'trade_date': trade_dates,
                    'nav_new': nav_new, 'nav_old_hs_nofilter': nav_old_hs_nf,
                    'nav_old_hs_filter': nav_old_hs_f,
                    'nav_zz800': b800.values, 'nav_hs300': b300.values})
os.makedirs('data/results/daily20_divlow', exist_ok=True)
out.to_csv('data/results/daily20_divlow/ab_oldvsnew_20150101_20251231.csv',
           index=False, encoding='utf-8-sig')
print("\nNAV → data/results/daily20_divlow/ab_oldvsnew_20150101_20251231.csv", flush=True)
print("DONE", flush=True)
