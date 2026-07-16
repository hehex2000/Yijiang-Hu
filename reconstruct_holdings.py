import csv
from collections import defaultdict

tf = "data/results/weekly_highdiv_vol/trades_n10_d25_t85_db55_s50_cost_20210104_20260710.csv"
rows = list(csv.DictReader(open(tf)))
# replay holdings per date
hold = defaultdict(int)  # code -> shares held after this date's trades
# group trades by date
by_date = defaultdict(list)
for r in rows:
    by_date[r["date"]].append(r)

# build a sorted list of all dates from equity file
ef = "data/results/weekly_highdiv_vol/backtest_n10_d25_t85_db55_s50_cost_20210104_20260710.csv"
edates = [r[0] for r in list(csv.reader(open(ef)))[1:]]

# For each date, simulate trades then count held codes
held_counts = {}
prev = {}
for d in edates:
    if d in by_date:
        for r in by_date[d]:
            c = r["code"]
            s = int(r["shares"])
            if r["action"] == "BUY":
                prev[c] = prev.get(c, 0) + s
            else:
                prev[c] = prev.get(c, 0) - s
                if prev[c] <= 0:
                    prev.pop(c, None)
    held_counts[d] = len(prev)

# stats
import statistics
vals = [float(r[1]) for r in list(csv.reader(open(ef)))[1:]]
# invested fraction = days where held>0
invested_days = sum(1 for d in edates if held_counts[d] > 0)
print("total dates", len(edates))
print("dates with >=1 holding:", invested_days, "(%.1f%%)" % (100*invested_days/len(edates)))
print("dates with 0 holding (all cash):", len(edates)-invested_days)
hc = [held_counts[d] for d in edates]
print("avg held codes when invested: %.1f" % statistics.mean([h for h in hc if h>0]))
print("max held codes:", max(hc), " min:", min(hc))

# Check the 13-day flat run 20210818..20210903
flat = ["20210818","20210819","20210820","20210823","20210824","20210825","20210826","20210827","20210830","20210831","20210901","20210902","20210903"]
print("\nDuring 13d flat run, held codes per day:")
for d in flat:
    print("  ", d, held_counts.get(d))
