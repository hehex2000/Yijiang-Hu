# -*- coding: utf-8 -*-
"""杠杆点A 回归测试：run_daily20_macd.load_closes(hfq=True) 的计价空间不变量。

同时构建 raw / hfq 两个缓存（首次运行较慢，之后走 pickle）。
"""
import sys
import numpy as np
import pandas as pd
import run_daily20_macd as D

OK = True


def check(name, cond, detail=""):
    global OK
    tag = "PASS" if cond else "FAIL"
    if not cond:
        OK = False
    print(f"[{tag}] {name}" + (f"  {detail}" if detail else ""), flush=True)


print("=" * 70)
print("杠杆点A：run_daily20_macd hfq 收盘矩阵测试")
print("=" * 70, flush=True)

codes_r, raw_df = D.load_closes(hfq=False)
print(f"  raw 矩阵 {raw_df.shape}", flush=True)
codes_h, hfq_df = D.load_closes(hfq=True)
print(f"  hfq 矩阵 {hfq_df.shape}", flush=True)

check("0. 两版股票池一致", codes_r == codes_h,
      f"raw={len(codes_r)} hfq={len(codes_h)}")

common = [c for c in raw_df.columns if c in hfq_df.columns]
check("0b. 列集合一致", len(common) == len(raw_df.columns),
      f"common={len(common)}/{len(raw_df.columns)}")

# ── 1. 首个交易日：hfq 应严格等于 raw（归一化基准正确）──
first_day = raw_df.index[0]
fr = raw_df.loc[first_day, common].dropna()
fh = hfq_df.loc[first_day, common].dropna()
j = fr.index.intersection(fh.index)
maxdiff = float((fr[j] - fh[j]).abs().max())
check("1. 首日 hfq == raw（归一化基准正确）", maxdiff < 1e-6,
      f"首日 {first_day}，最大绝对差 {maxdiff:.2e}（{len(j)} 只）")

# ── 2. 末日：hfq 应 >= raw（分红+送转累计，单调不减）──
last_day = raw_df.index[-1]
lr = raw_df.loc[last_day, common].dropna()
lh = hfq_df.loc[last_day, common].dropna()
j2 = lr.index.intersection(lh.index)
ratio = (lh[j2] / lr[j2])
check("2. 末日 hfq/raw >= 1（因子单调不减）", bool((ratio >= 1 - 1e-6).all()),
      f"最小比值 {ratio.min():.4f}，中位 {ratio.median():.4f}，最大 {ratio.max():.4f}")

# ── 3. 绝对价位不得被放大到失真（归一化有效性）──
#    若忘了除以 ref，格力类个股会从 67.9 变成 10589，整股下单 int(cash//px) 会买 0 股
mx_raw = float(raw_df.max().max())
mx_hfq = float(hfq_df.max().max())
check("3. hfq 最大价位在合理区间（< 20× raw 最大价，说明已归一化）",
      mx_hfq < mx_raw * 20,
      f"raw 最大 {mx_raw:.2f} → hfq 最大 {mx_hfq:.2f}（{mx_hfq/mx_raw:.2f}×）")

# ── 4. 逐日比值应平滑累积，不得有整表跳变（ffill 生效、无 fillna(1.0) 假跳空）──
sub_r = raw_df.loc[(raw_df.index >= 20200101) & (raw_df.index <= 20260723), common]
sub_h = hfq_df.loc[(hfq_df.index >= 20200101) & (hfq_df.index <= 20260723), common]
ratio_df = (sub_h / sub_r).replace([np.inf, -np.inf], np.nan)
# 按日取所有股票比值的中位数，看是否有单日整表跳变
med = ratio_df.median(axis=1).dropna()
chg = med.pct_change().abs()
worst = chg.max()
worst_day = chg.idxmax()
check("4. 逐日比值中位数无整表跳变（单日 < 3%）", worst < 0.03,
      f"最大单日跳变 {worst:.4%} @ {worst_day}；若 ffill 失效会出现 10%+ 跳变")

# ── 5. 找一个真实的除权案例：单日 raw 跌 >20% 但 hfq 跌幅小得多 ──
found = None
for c in common:
    s_r = raw_df[c].dropna()
    s_h = hfq_df[c].dropna()
    if len(s_r) < 100:
        continue
    r_ret = s_r.pct_change()
    h_ret = s_h.pct_change()
    idx = r_ret[(r_ret < -0.25) & (r_ret.index.isin(h_ret.index))].index
    for d in idx:
        gap = h_ret.get(d, np.nan) - r_ret[d]
        if np.isfinite(gap) and gap > 0.20:
            if found is None or gap > found[3]:
                found = (c, d, s_r.shift(1)[d], s_r[d], gap,
                         r_ret[d], h_ret.get(d, np.nan))
if found:
    c, d, p0, p1, gap, rr, hr = found
    print(f"\n  最大凭空亏损案例：{c} @ {d}")
    print(f"    raw  {p0:.2f} → {p1:.2f}  ({rr:+.2%})")
    print(f"    hfq  跌幅 {hr:+.2%}    凭空亏损 {gap*100:.2f}pp")
    check("5. 存在 raw 凭空亏损 > 20pp 的除权事件（送转/分红）", gap > 0.20,
          f"{c} @ {d}: {gap*100:.2f}pp")
    check("5b. hfq 在该事件上跌幅显著更小", hr > rr + 0.15,
          f"hfq {hr:+.2%} vs raw {rr:+.2%}")
else:
    print("[SKIP] 5. zz800 池内未找到单日 >20pp 的除权事件")

print("=" * 70)
print("结果:", "全部 PASS ✅" if OK else "存在 FAIL ❌", flush=True)
sys.exit(0 if OK else 1)
