# -*- coding: utf-8 -*-
"""验证实际持仓只数 —— 若 top_n=20 但质量门过滤后只剩 5 只，
则单只权重 ~20%，正好解释「比值跳变/因子跳变 ≈ 23%」的隐含权重。"""
import pandas as pd

BASE = "data/results/monthly_rebalance"
tr = pd.read_csv(f"{BASE}/trades_value_20200102_20260723.csv")
tr["date"] = tr["date"].astype(str)
print("action 取值:", tr["action"].unique().tolist())
print(tr.head(6).to_string(index=False))

buy_kw = [a for a in tr["action"].unique() if str(a).startswith(("buy", "买入", "BUY"))]
print("识别为买入的 action:", buy_kw)

# 逐日持仓只数
opens = []
events = []
for _, r in tr.iterrows():
    events.append((r["date"], r["action"], r["code"]))
events.sort()

# 用 FIFO 重建
from collections import defaultdict, deque
q = defaultdict(deque)
cur = set()
series = []
for d, a, c in events:
    if a in buy_kw:
        q[c].append(d)
        cur.add(c)
    else:
        if q[c]:
            q[c].popleft()
        if not q[c]:
            cur.discard(c)
    series.append((d, len(cur)))

s = pd.DataFrame(series, columns=["date", "n"]).groupby("date")["n"].last()
print(f"\n持仓只数统计: 均值 {s.mean():.1f}  中位 {s.median():.0f}  最小 {s.min()}  最大 {s.max()}")
print(f"\n按年均值:")
print(s.groupby(s.index.str[:4]).mean().to_string())

print("\n抽样几个时点:")
for d in ["20221214", "20220613", "20210610", "20240705", "20260105"]:
    if d in s.index:
        print(f"  {d}: {s[d]} 只")
    else:
        near = s.index[s.index <= d]
        if len(near):
            print(f"  {d}: (非交易日) 最近 {near[-1]} → {s[near[-1]]} 只")
