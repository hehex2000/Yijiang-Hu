# -*- coding: utf-8 -*-
"""诊断：fina_indicator 回补后，早年 zz800 成员能否算出 F-score、能否达 F>=8。"""
import sqlite3, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA
DB = DATA["local_db_path"]
from piotroski_fscore import build_fscore_maps, compute_fscore

con = sqlite3.connect(DB)
M = build_fscore_maps(con)

# ---- 1) zz800 成员在 fina_indicator 年报(年度)覆盖行数（按年）----
print("== fina_indicator 年报行数（zz800 历史成员内，端到端年度）==")
rows = con.execute("""
    SELECT substr(end_date,1,4) yr, COUNT(*) n,
           SUM(CASE WHEN roa IS NOT NULL THEN 1 ELSE 0 END) roa_nn
    FROM fina_indicator
    WHERE ts_code IN (SELECT DISTINCT ts_code FROM index_constituent WHERE index_code='000906.SH')
      AND end_date LIKE '%1231'
    GROUP BY yr ORDER BY yr
""").fetchall()
for yr, n, roa_nn in rows:
    print(f"  {yr}: rows={n:5d}  roa非null={roa_nn:5d}")

# ---- 2) 各早年调仓日：成员数 / 有roa map数 / F>=8数 / F分布 ----
def members_at(t):
    return [r[0] for r in con.execute(
        "SELECT DISTINCT ts_code FROM index_constituent WHERE index_code='000906.SH' AND trade_date <= ?",
        (t,)).fetchall()]

print("\n== 早年调仓日 F-score 分布（严格 PIT, ann_date<t）==")
for t in ["20100505","20110505","20120505","20140505","20150505"]:
    mem = members_at(t)
    scores, has_roa = [], 0
    for c in mem:
        s, _ = compute_fscore(c, t, M)
        scores.append(s)
        if c in M["roa"]:
            has_roa += 1
    n = len(mem)
    ge8 = sum(1 for x in scores if x >= 8)
    from collections import Counter
    dist = dict(sorted(Counter(scores).items()))
    print(f"  t={t}: members={n} 有roa数据={has_roa}  F>=8={ge8}  F分布={dist}")

con.close()
