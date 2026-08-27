# -*- coding: utf-8 -*-
"""精确复刻 run_monthly_rebalance 的 价值初筛(top BM) -> F>=8 门槛，看早年是否真的空。"""
import sqlite3, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA
DB = DATA["local_db_path"]
from piotroski_fscore import build_fscore_maps, compute_fscore

con = sqlite3.connect(DB)
M = build_fscore_maps(con)

def value_candidates(t, top_n=30):
    """复刻价值初筛：zz800 成员中，取 t 前最近交易日的 pb，按 bm=1/pb 降序取前 top_n。"""
    mem = [r[0] for r in con.execute(
        "SELECT DISTINCT ts_code FROM index_constituent WHERE index_code='000906.SH' AND trade_date <= ?",
        (t,)).fetchall()]
    cand = []
    for c in mem:
        row = con.execute(
            "SELECT pb FROM daily_basic WHERE ts_code=? AND trade_date <= ? ORDER BY trade_date DESC LIMIT 1",
            (c, t)).fetchone()
        if row and row[0] is not None and row[0] > 0:
            cand.append((c, 1.0/row[0]))
    cand.sort(key=lambda x: -x[1])
    return cand[:top_n]

print("== 价值候选(top30 by BM) 中 F>=8 的数量（复刻引擎门槛）==")
for t in ["20100505","20110505","20120505","20140505","20150505","20170505"]:
    cand = value_candidates(t)
    fs = [(c, compute_fscore(c, t, M)[0]) for c, _ in cand]
    ge8 = [c for c, s in fs if s >= 8]
    top5 = fs[:5]
    print(f"  t={t}: 价值候选={len(cand)} 其中F>=8={len(ge8)}  Top5的F={[s for _,s in top5]}  Top5代码={[c for c,_ in top5]}")

con.close()
