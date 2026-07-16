import csv, statistics, sys
from collections import defaultdict

base = "data/results/weekly_highdiv_vol/"
bt = sys.argv[1] if len(sys.argv) > 1 else "backtest_n10_d25_t85_db55_s50_cost_rb0_20210104_20260710.csv"
tr = bt.replace("backtest_", "trades_")
ef = base + bt
tf = base + tr

rows = list(csv.DictReader(open(tf)))
by_date = defaultdict(list)
for r in rows:
    by_date[r["date"]].append(r)
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

inv = sum(1 for d in edates if held[d])
hc = [len(held[d]) for d in edates]
inv_hc = [h for h in hc if h > 0]
print(f"file: {bt}")
print(f"交易日={len(edates)}  持仓日={inv}({100*inv/len(edates):.1f}%)  空仓日={len(edates)-inv}")
print(f"持仓只数: 均值(持仓时)={statistics.mean(inv_hc):.1f}  全样本均值={statistics.mean(hc):.1f}  min={min(hc)} max={max(hc)}")
# drawdown
peak=evals[0]; mdd=0
for v in evals:
    if v>peak: peak=v
    dd=(v-peak)/peak
    if dd<mdd: mdd=dd
print(f"总收益末端={evals[-1]:.0f}  最大回撤={mdd*100:.2f}%")
# invested-day vol
inv_rets=[(evals[i]-evals[i-1])/evals[i-1] for i in range(1,len(edates)) if held[edates[i]] or held[edates[i-1]]]
print(f"持仓日日波动 std={statistics.pstdev(inv_rets)*100:.2f}%  n={len(inv_rets)}")
