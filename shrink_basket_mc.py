# -*- coding: utf-8 -*-
"""缩小篮子可行性蒙特卡洛：从800池随机抽 N(5/10) 只，固定窗口起点不再换，
跑与等权800完全一致算法(等权每日再平衡 + 沪深300 MACD 开关)，看 30 个随机组合分布。
核心目的：验证"缩到5-10只、算法不变"是否是个靠谱方法，还是纯运气。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import macd_plugin_validate as M
from regime_cash_overlay import load_index_close, BENCH, apply_overlay

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_retmat_cache.pkl')
START, END = '20100101', '20251231'

if os.path.exists(CACHE):
    ret, codes, hs = pd.read_pickle(CACHE)
    print(f"[cache] 命中收益矩阵 {ret.shape}, 沪深300 {len(hs)}")
else:
    conn = M.get_conn()
    codes = [r[0] for r in conn.execute(
        "SELECT DISTINCT ts_code FROM index_constituent WHERE index_code='000906.SH'")]
    print(f"[load] {len(codes)} 只，分批 bulk...", flush=True)
    frames = []
    for i in range(0, len(codes), 400):
        b = codes[i:i+400]; ph = ",".join("?"*len(b))
        d = pd.read_sql_query(
            f"SELECT ts_code,trade_date,close FROM daily WHERE ts_code IN ({ph}) "
            f"AND trade_date>=? AND trade_date<=? ORDER BY ts_code,trade_date",
            conn, params=b+[int(START), int(END)])
        if len(d): frames.append(d)
    conn.close()
    alld = pd.concat(frames, ignore_index=True)
    alld['trade_date'] = alld['trade_date'].astype(int)
    panel = alld.pivot(index='trade_date', columns='ts_code', values='close').sort_index()
    ret = panel.pct_change()
    hs = load_index_close(BENCH, START, END).reindex(panel.index).ffill()
    pd.to_pickle((ret, codes, hs), CACHE)
    print(f"[load] 收益矩阵就绪 {ret.shape}", flush=True)

ret_v = ret.values.astype(float)
dates = ret.index
hs_v = hs.values.astype(float)
golden = M.macd_golden(hs_v).values

def port_nav(subset_cols, mask):
    r = ret_v[:, subset_cols]
    pr = np.nanmean(r, axis=1)            # 等权每日再平衡(与800一致)
    nav = (1.0 + np.nan_to_num(pr, 0.0)).cumprod()
    return apply_overlay(nav, mask)

def total(mask, subset):
    nav = port_nav(subset, mask)
    return nav.iloc[-1]/nav.iloc[0] - 1

rng = np.random.default_rng(20260825)
K = 30
results = {}
for N in (5, 10):
    tot_macd, tot_bh = [], []
    for k in range(K):
        idx = rng.choice(len(codes), size=N, replace=False)
        sub = [codes[i] for i in idx]
        cols = [ret.columns.get_loc(c) for c in sub]
        tot_bh.append(total(np.ones(len(dates), bool), cols))
        tot_macd.append(total(golden, cols))
    results[N] = (np.array(tot_bh), np.array(tot_macd))

def stats(a):
    a = np.sort(a); return f"min{pct(a[0])} p10{pct(a[K//10])} med{pct(np.median(a))} p90{pct(a[-K//10])} max{pct(a[-1])} 正{a[a>0].size}/{K}"

def pct(x): return f"{x*100:+6.1f}%"

print(f"\n{'='*96}")
print(f"  缩小篮子蒙特卡洛 | 800池随机抽N只(窗口起点定死) | 等权每日再平衡+沪深300 MACD")
print(f"  对照: 全800+MACD=+433.8%  全800满仓=+83.4%")
print(f"{'='*96}")
for N in (5,10):
    bh, mc = results[N]
    print(f"\n  N={N} (K={K} 随机组合):")
    print(f"    等权每日再平衡(无MACD): {stats(bh)}")
    print(f"    等权每日再平衡+MACD   : {stats(mc)}")
print(f"\n  [解读] 若分布跨度巨大/中位远低于全800，则'缩到5-10只算法不变'本质是运气抽样，")
print(f"        不可作为可信方法——结果取决于抽到哪几只，而非算法。")
