import csv
from collections import defaultdict
import statistics

tf = "data/results/weekly_highdiv_vol/trades_n10_d25_t85_db55_s50_cost_20210104_20260710.csv"
rows = list(csv.DictReader(open(tf)))
by_date = defaultdict(list)
for r in rows:
    by_date[r["date"]].append(r)
ef = "data/results/weekly_highdiv_vol/backtest_n10_d25_t85_db55_s50_cost_20210104_20260710.csv"
ed = list(csv.reader(open(ef)))[1:]
edates = [r[0] for r in ed]
evals = [float(r[1]) for r in ed]

prev = {}
held = {}
for d in edates:
    if d in by_date:
        for r in by_date[d]:
            c = r["code"]; s = int(r["shares"])
            prev[c] = prev.get(c, 0) + (s if r["action"] == "BUY" else -s)
            if prev[c] <= 0:
                prev.pop(c, None)
    held[d] = set(prev.keys())

inv_rets = []; cash_rets = []
for i in range(1, len(edates)):
    r = (evals[i]-evals[i-1])/evals[i-1]
    if held[edates[i]] or held[edates[i-1]]:
        inv_rets.append(r)
    else:
        cash_rets.append(r)

print("INVESTED-day daily ret: mean=%.5f std=%.5f n=%d" % (statistics.mean(inv_rets), statistics.pstdev(inv_rets), len(inv_rets)))
print("CASH-day     daily ret: mean=%.6f std=%.6f n=%d" % (statistics.mean(cash_rets), statistics.pstdev(cash_rets), len(cash_rets)))
print()
# check: are there flat runs WHILE invested (price freeze)?
flat=0; runs=[]
for i in range(1,len(edates)):
    if held[edates[i]] and abs(evals[i]-evals[i-1])<1e-6:
        flat+=1
    else:
        if flat>=3: runs.append((edates[i-flat],edates[i-1],flat))
        flat=0
if flat>=3: runs.append((edates[-flat],edates[-1],flat))
print("INVESTED flat runs (price-freeze suspect) >=3d:", len(runs))
for s,e,n in sorted(runs,key=lambda x:-x[2])[:8]:
    print("   ",s,"->",e,"%dd held but flat"%n)

# invested-day max drawdown (real risk when actually in market)
peak=0; mdd=0; mdd_d=None
cum=1.0
for r in inv_rets:
    cum*= (1+r)
    if cum>peak: peak=cum
    dd=(cum-peak)/peak
    if dd<mdd: mdd=dd; mdd_d=r
print("\nInvested-only compounded drawdown approx: %.2f%%" % (mdd*100))
