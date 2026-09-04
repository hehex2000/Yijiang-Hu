# -*- coding: utf-8 -*-
"""NAV 路径级验证：hfq/raw 比值必须平滑累积（股息复利），不得有跳变

跳变 = 假跳空 = 价格空间 bug。平滑 = 股息逐日累积 = 正确。
"""
import pandas as pd
import os

BASE = "data/results/monthly_rebalance"
raw = pd.read_csv(f"{BASE}/backtest_20200102_20260723.csv")
hfq = pd.read_csv(f"{BASE}/backtest_hfq_20200102_20260723.csv")
print("raw 列:", list(raw.columns))
print("hfq 列:", list(hfq.columns))

dc = [c for c in raw.columns if "date" in c.lower()][0]
vc = [c for c in raw.columns if "value" in c.lower() or "nav" in c.lower()][0]
print(f"用列: date={dc}  value={vc}")

r = raw[[dc, vc]].rename(columns={vc: "raw"})
h = hfq[[dc, vc]].rename(columns={vc: "hfq"})
df = r.merge(h, on=dc).sort_values(dc)
df[dc] = df[dc].astype(str)
print(f"\n合并 {len(df)} 行  {df[dc].iloc[0]} → {df[dc].iloc[-1]}")

df["ratio"] = df["hfq"] / df["raw"]
df["d_ratio"] = df["ratio"].pct_change()

print(f"\n比值: 首日 {df['ratio'].iloc[0]:.6f}  → 末日 {df['ratio'].iloc[-1]:.6f}")
print(f"      累计 {(df['ratio'].iloc[-1]/df['ratio'].iloc[0]-1)*100:+.2f}%  "
      f"年化 {((df['ratio'].iloc[-1]/df['ratio'].iloc[0])**(252/len(df))-1)*100:+.2f}%")

# 跳变检测：单日比值变动超过 ±1% 视为异常（除息日单日股息不可能>1%）
thr = 0.01
jumps = df[df["d_ratio"].abs() > thr]
print(f"\n单日比值变动 > ±{thr:.0%} 的天数: {len(jumps)} / {len(df)-1}")
if len(jumps):
    print(jumps[[dc, "ratio", "d_ratio"]].head(20).to_string(index=False))
    print(f"  最大正跳变 {jumps['d_ratio'].max()*100:+.2f}%   最大负跳变 {jumps['d_ratio'].min()*100:+.2f}%")
else:
    print("  → 无跳变 ✅")

print(f"\n单日比值变动统计: 均值 {df['d_ratio'].mean()*100:+.4f}%  "
      f"std {df['d_ratio'].std()*100:.4f}%  最大 {df['d_ratio'].max()*100:+.3f}%  "
      f"最小 {df['d_ratio'].min()*100:+.3f}%")

# 年度比值增长
df["year"] = df[dc].str[:4]
yr = df.groupby("year")["ratio"].last()
prev = yr.shift(1)
growth = (yr / prev - 1) * 100
print("\n比值逐年增长（≈当年股息贡献）:")
for y in yr.index:
    g = f"{growth[y]:+6.2f}%" if pd.notna(growth[y]) else "   n/a"
    print(f"  {y}: 比值 {yr[y]:.4f}   当年增长 {g}")

# ── 止损成交笔数对照 ──
print()
print("=" * 60)
print("止损成交笔数对照")
print("=" * 60)
for tag, path in [("raw", f"{BASE}/trades_value_20200102_20260723.csv"),
                  ("hfq", f"{BASE}/trades_hfq_value_20200102_20260723.csv")]:
    if not os.path.exists(path):
        print(f"  {tag}: 文件不存在 {path}")
        continue
    t = pd.read_csv(path)
    print(f"\n  [{tag}] 共 {len(t)} 笔")
    if "reason" in t.columns:
        print(t["reason"].value_counts().to_string())
