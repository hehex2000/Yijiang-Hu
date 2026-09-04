# -*- coding: utf-8 -*-
"""重建持仓，量化 raw 口径下「送转股」与「现金分红」各自造成的凭空亏损

做法：
  1. 从 trades CSV 重建每段持仓（买入日 → 卖出日）
  2. 对每段持仓，取该股 adj_factor 序列，找出持仓期内的因子跳变
  3. 因子单次跳变 > 10% → 判为送转/拆股；<= 10% → 判为现金分红
  4. 送转的凭空亏损 = 1 - 1/送转比例（raw 模式下 shares 不变，市值直接蒸发）
     现金分红的漏计 = 因子增长（分红从未计入 NAV）
"""
import sqlite3
import sys
import os
import bisect
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_monthly_rebalance as m
import pandas as pd

BASE = "data/results/monthly_rebalance"
con0 = m.get_conn()
db = con0.execute("PRAGMA database_list").fetchone()[2]
con0.close()
con = sqlite3.connect(db)

tr = pd.read_csv(f"{BASE}/trades_value_20200102_20260723.csv")
tr["date"] = tr["date"].astype(str)
tr["code"] = tr["code"].astype(str)

# ── 重建持仓段 ──
holds = {}   # code -> list of (buy_date, shares)
segs = []    # (code, buy_date, sell_date)
for _, r in tr.iterrows():
    c = r["code"]
    if str(r["action"]).lower().startswith("buy") or str(r["action"]) == "买入":
        holds.setdefault(c, []).append((r["date"], r["shares"]))
    else:
        if holds.get(c):
            bd, sh = holds[c].pop(0)
            segs.append((c, bd, r["date"]))
for c, lst in holds.items():
    for bd, sh in lst:
        segs.append((c, bd, "20260723"))
print(f"重建持仓段 {len(segs)} 段")

# ── 取所有涉及股票的因子序列 ──
codes = sorted({c for c, _, _ in segs})
print(f"涉及股票 {len(codes)} 只，读取 adj_factor ...")
qmarks = ",".join("?" * len(codes))
adj = pd.read_sql_query(
    f"SELECT ts_code, trade_date, adj_factor FROM adj_factor "
    f"WHERE ts_code IN ({qmarks}) AND trade_date BETWEEN '20200101' AND '20260723' "
    f"ORDER BY ts_code, trade_date", con, params=codes)
con.close()
SER = {}
for c, g in adj.groupby("ts_code"):
    SER[c] = ([str(x) for x in g["trade_date"].tolist()],
              [float(v) for v in g["adj_factor"].tolist()])

SPLIT_TH = 0.10   # 因子单次跳变 >10% 判为送转/拆股
rows = []
for c, bd, sd in segs:
    if c not in SER:
        continue
    dates, vals = SER[c]
    i0 = bisect.bisect_left(dates, bd)
    i1 = bisect.bisect_right(dates, sd) - 1
    if i1 <= i0:
        continue
    for k in range(i0 + 1, i1 + 1):
        chg = vals[k] / vals[k - 1] - 1
        if abs(chg) < 1e-9:
            continue
        if chg > SPLIT_TH:
            # 送转：raw 模式下 shares 不变 → 市值蒸发 (1 - 1/ratio)
            phantom = 1 - 1 / (vals[k] / vals[k - 1])
            rows.append((c, dates[k], "送转", chg, -phantom))
        elif chg > 0:
            # 现金分红：漏计，金额即因子增长
            rows.append((c, dates[k], "分红", chg, -chg))

ev = pd.DataFrame(rows, columns=["code", "date", "type", "factor_chg", "phantom_loss"])
print(f"\n持仓期内因子事件 {len(ev)} 次")
print(ev["type"].value_counts().to_string())

print()
print("=" * 78)
print("凭空亏损 Top 15（按单次亏损幅度）")
print("=" * 78)
print(ev.reindex(ev["phantom_loss"].abs().sort_values(ascending=False).index).head(15).to_string(
    index=False, float_format=lambda x: f"{x:,.4f}"))

# ── 按年汇总 ──
ev["year"] = ev["date"].str[:4]
print()
print("=" * 78)
print("按年汇总（损失幅度为持仓市值占比，未加权到组合）")
print("=" * 78)
agg = ev.groupby(["year", "type"]).agg(
    次数=("phantom_loss", "size"),
    损失合计=("phantom_loss", "sum"),
    最大单次=("phantom_loss", "min")).reset_index()
print(agg.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

tot_split = ev[ev["type"] == "送转"]["phantom_loss"].sum()
tot_div = ev[ev["type"] == "分红"]["phantom_loss"].sum()
print(f"\n  送转累计（持仓市值占比求和）: {tot_split*100:,.2f}%   [{len(ev[ev['type']=='送转'])} 次]")
print(f"  分红累计（持仓市值占比求和）: {tot_div*100:,.2f}%   [{len(ev[ev['type']=='分红'])} 次]")
print(f"  → 送转占比 {tot_split/(tot_split+tot_div)*100:.1f}%   分红占比 {tot_div/(tot_split+tot_div)*100:.1f}%")
print()
print("注：这里按「持仓市值占比」直接求和，未做组合权重加权与时间复利，")
print("    仅用于判断两类事件的相对重要性，不等于 NAV 口径的最终差额。")
