import csv, statistics
from collections import defaultdict, OrderedDict

f = "data/results/weekly_highdiv_vol/trades_n10_d25_t85_db55_s50_cost_rb0_20210104_20260710.csv"
rows = list(csv.DictReader(open(f)))
print("总成交记录:", len(rows), " BUY=", sum(1 for r in rows if r['action']=='BUY'),
      " SELL=", sum(1 for r in rows if r['action']=='SELL'))

# 配对每支股票的 BUY->SELL，得到每笔 round-trip 收益
# 按出现顺序配对（同一股票可能多次进出，用队列）
positions = defaultdict(list)   # code -> list of (buy_price, buy_shares, buy_date)
rt = []  # round trips: (code, ret, hold_days, pnl, buy_date, sell_date)
pnl_by_year = defaultdict(float)
cnt_by_year = defaultdict(int)
win_by_year = defaultdict(int)
stock_pnl = defaultdict(float)
stock_cnt = defaultdict(int)

for r in rows:
    code = r['code']; act = r['action']; d = r['date']; p = float(r['price']); s = int(r['shares'])
    if act == 'BUY':
        positions[code].append([p, s, d])
    else:  # SELL
        if positions[code]:
            bp, bs, bd = positions[code].pop(0)
            ret = (p - bp) / bp
            hold = int(d) - int(bd)
            rt.append((code, ret, hold, (p-bp)*bs, bd, d))
            yr = d[:4]
            pnl_by_year[yr] += (p-bp)*bs
            cnt_by_year[yr] += 1
            if ret > 0: win_by_year[yr] += 1
            stock_pnl[code] += (p-bp)*bs
            stock_cnt[code] += 1

print("\n=== 配对得到的 round-trip 数:", len(rt))
rets = [x[1] for x in rt]
print("每笔收益: 中位=%.2f%% 均值=%.2f%% 最小=%.1f%% 最大=%.1f%%" %
      (statistics.median(rets)*100, statistics.mean(rets)*100, min(rets)*100, max(rets)*100))
print("胜率(笔数): %.1f%%" % (100*sum(1 for r in rets if r>0)/len(rets)))
print("收益分布: <-10%%:%d  -10~-2%%:%d  -2~0%%:%d  0~2%%:%d  2~10%%:%d  >10%%:%d" %
      (sum(1 for r in rets if r<-0.10), sum(1 for r in rets if -0.10<=r<-0.02),
       sum(1 for r in rets if -0.02<=r<0), sum(1 for r in rets if 0<=r<0.02),
       sum(1 for r in rets if 0.02<=r<0.10), sum(1 for r in rets if r>=0.10)))

print("\n=== 按年盈亏贡献 (元) ===")
for y in sorted(pnl_by_year):
    wr = 100*win_by_year[y]/cnt_by_year[y] if cnt_by_year[y] else 0
    print("  %s: 盈亏=%+10.0f  交易=%4d  胜率=%.1f%%" % (y, pnl_by_year[y], cnt_by_year[y], wr))

print("\n=== 单股贡献 TOP15 (按累计盈亏) ===")
top = sorted(stock_pnl.items(), key=lambda x:-x[1])[:15]
tot_pnl = sum(stock_pnl.values())
print("总盈亏(成交口径)=%.0f" % tot_pnl)
cum = 0
for c, p in top:
    cum += p
    print("  %s 次数=%3d 累计盈亏=%+10.0f  占总盈亏=%.1f%%  累计=%.1f%%" %
          (c, stock_cnt[c], p, 100*p/tot_pnl, 100*cum/tot_pnl))

print("\n=== 集中度 ===")
sp = sorted(stock_pnl.values(), reverse=True)
print("贡献最大的10支股票占总盈亏: %.1f%%" % (100*sum(sp[:10])/tot_pnl))
print("贡献最大的20支股票占总盈亏: %.1f%%" % (100*sum(sp[:20])/tot_pnl))
# 亏损股占比
losers = [(c,p) for c,p in stock_pnl.items() if p<0]
print("亏损股票数=%d / 总参与股票数=%d (%.1f%%)" % (len(losers), len(stock_pnl), 100*len(losers)/len(stock_pnl)))
