import csv, statistics

f = "data/results/weekly_highdiv_vol/backtest_n10_d25_t85_db55_s50_cost_20210104_20260710.csv"
rows = list(csv.reader(open(f)))
data = rows[1:]
vals = [float(r[1]) for r in data]
dates = [r[0] for r in data]

peak = vals[0]; mdd = 0; mdd_trough = None; mdd_peak = None
flat = 0; maxflat = 0; flat_start = None; runs = []
for i, v in enumerate(vals):
    if v > peak:
        peak = v
    dd = (v - peak) / peak
    if dd < mdd:
        mdd = dd; mdd_trough = i; mdd_peak = peak
    if i > 0 and abs(v - vals[i-1]) < 1e-6:
        if flat == 0:
            flat_start = i
        flat += 1
        maxflat = max(maxflat, flat)
    else:
        if flat >= 5:
            runs.append((dates[flat_start], dates[i-1], flat))
        flat = 0
if flat >= 5:
    runs.append((dates[flat_start], dates[-1], flat))

print("rows", len(vals))
print("first", dates[0], round(vals[0], 2), "last", dates[-1], round(vals[-1], 2))
print("max drawdown %.4f%% at %s (peak before=%.2f)" % (mdd*100, dates[mdd_trough], mdd_peak))
rets = [(vals[i]-vals[i-1])/vals[i-1] for i in range(1, len(vals))]
print("daily ret mean %.6f std %.6f" % (statistics.mean(rets), statistics.pstdev(rets)))
print("days |ret|>2%%:", sum(1 for r in rets if abs(r) > 0.02), "of", len(rets))
print("days |ret|>1%%:", sum(1 for r in rets if abs(r) > 0.01))
print("num flat runs>=5d:", len(runs), " longest:", maxflat)
for s, e, n in sorted(runs, key=lambda x: -x[2])[:10]:
    print("   flat", s, "->", e, "%dd" % n)
