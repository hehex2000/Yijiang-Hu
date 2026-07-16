import csv
from collections import defaultdict, deque
import statistics

base = "data/results/weekly_highdiv_vol/"
bt = "backtest_n10_d25_t85_db55_s50_cost_rb0_20210104_20260710.csv"
tr = bt.replace("backtest_", "trades_")
ef = base + bt; tf = base + tr

rows = list(csv.DictReader(open(tf)))
by_date = defaultdict(list)
for r in rows:
    by_date[r["date"]].append(r)
ed = list(csv.reader(open(ef)))[1:]
edates = [r[0] for r in ed]; evals = [float(r[1]) for r in ed]

prev = {}
held = {}
for d in edates:
    if d in by_date:
        for r in by_date[d]:
            c = r["code"]; s = int(r["shares"])
            prev[c] = prev.get(c, 0) + (s if r["action"] == "BUY" else -s)
            if prev[c] <= 0: prev.pop(c, None)
    held[d] = set(prev.keys())

# (1) invested freeze runs (>=2d flat while holding)
flat=0; runs=[]
for i in range(1,len(edates)):
    if held[edates[i]] and abs(evals[i]-evals[i-1])<1e-6:
        flat+=1
    else:
        if flat>=2: runs.append((edates[i-flat],edates[i-1],flat))
        flat=0
if flat>=2: runs.append((edates[-flat],edates[-1],flat))
print("INVESTED freeze runs >=2d:", len(runs), "longest:", max([r[2] for r in runs],default=0))
for s,e,n in sorted(runs,key=lambda x:-x[2])[:6]:
    print("   ",s,"->",e,"%dd"%n)

# (2) per-trade round-trip return
buys = defaultdict(deque)
rt = []
for r in rows:
    if r["action"]=="BUY":
        buys[r["code"]].append((float(r["price"]), int(r["shares"])))
    else:
        # SELL: match FIFO with buys
        q = buys[r["code"]]
        sp = float(r["price"]); sq = int(r["shares"])
        while sq>0 and q:
            bp, bsh = q[0]
            m = min(bsh, sq)
            rt.append((sp-bp)/bp)
            sq-=m; bsh-=m
            if bsh<=0: q.popleft()
            else: q[0]=(bp,bsh)
print("\nround-trips:", len(rt))
print("median rt ret: %.2f%%  mean: %.2f%%" % (statistics.median(rt)*100, statistics.mean(rt)*100))
print("rt > +20%%: %d   rt > +50%%: %d   rt < -20%%: %d" % (
    sum(1 for x in rt if x>0.2), sum(1 for x in rt if x>0.5), sum(1 for x in rt if x<-0.2)))
print("TOP 10 single-trade returns:")
for x in sorted(rt,reverse=True)[:10]:
    print("   +%.1f%%" % (x*100))
print("BOTTOM 5:")
for x in sorted(rt)[:5]:
    print("   %.1f%%" % (x*100))

# (3) biggest single-day portfolio jumps
jumps = sorted(range(1,len(evals)), key=lambda i: evals[i]-evals[i-1])
print("\nTOP 5 single-day GAINS:")
for i in jumps[-5:][::-1]:
    print("   %s +%.2f%%" % (edates[i], (evals[i]-evals[i-1])/evals[i-1]*100))
print("TOP 5 single-day LOSSES:")
for i in jumps[:5]:
    print("   %s %.2f%%" % (edates[i], (evals[i]-evals[i-1])/evals[i-1]*100))
