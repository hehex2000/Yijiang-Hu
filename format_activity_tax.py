# -*- coding: utf-8 -*-
"""把 run_activity_tax_check.py 的全平台扫描结果整理成可读分组表。

为什么要单独一个脚本：activity_tax 的控制台输出用 `{name:<14}` 定宽，
而全平台扫描的行名是相对路径（如 `daily20_divlow_bugfixed_20260902/n5/trades_...`），
115 行会挤成一团没法看。这里按「顶层目录 = 策略」分组、组内按活跃税降序，
并把路径拆成 策略/档位 两列。

用法：
  python format_activity_tax.py [输入CSV] [输出CSV]
默认 输入 data/results/activity_tax_all_platform.csv
     输出 data/results/activity_tax_all_platform_grouped.csv
"""
import os
import sys
import pandas as pd

pd.set_option("display.width", 240)
pd.set_option("display.max_rows", 300)

IN = sys.argv[1] if len(sys.argv) > 1 else "data/results/activity_tax_all_platform.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "data/results/activity_tax_all_platform_grouped.csv"

BENCH = 0.065  # Barber & Odean 6.5%/年

df = pd.read_csv(IN)
# 路径分隔符 Windows 是 `\`、POSIX 是 `/`，两种都要能切（否则 策略/档位 两列会
# 退化成同一个整串，115 行的长路径挤在一起根本没法看）
def _split(p):
    return str(p).replace("\\", "/").split("/")

df["策略"] = df["name"].map(lambda x: _split(x)[0] if len(_split(x)) > 1 else "(根目录)")
df["档位"] = df["name"].map(
    lambda x: _split(x)[-1] if len(_split(x)) > 1 else _split(x)[0]
).str.replace(r"^trades_|\.csv$", "", regex=True)

# 可信度：分母塌缩 / 流水不自洽 → 数字不可信
df["可信"] = ~(df["init_cap"].fillna(0) <= 0) & ~(df["neg_hold_codes"].fillna(0) > 0)
df["超阈值"] = df["可信"] & (df["active_tax_nav"] > BENCH)

if "price_mode" in df.columns:
    df["口径"] = df["price_mode"]
show = ["策略", "档位", "口径", "n_trades", "years", "mean_nav", "total_cost",
        "active_tax_nav", "active_tax_yr", "round_trip_cost", "annualized",
        "max_dd", "可信", "超阈值", "flag"]
show = [c for c in show if c in df.columns]
out = df[show].copy()
for c in ["active_tax_nav", "active_tax_yr", "round_trip_cost", "annualized", "max_dd"]:
    if c in out.columns:
        out[c] = (out[c] * 100).round(2)
if "mean_nav" in out.columns:
    out["mean_nav"] = (out["mean_nav"] / 1e4).round(0)
if "total_cost" in out.columns:
    out["total_cost"] = (out["total_cost"] / 1e4).round(1)
out = out.rename(columns={
    "n_trades": "笔数", "years": "年数", "mean_nav": "均净值(万)",
    "total_cost": "总成本(万)", "active_tax_nav": "活跃税/年(净值)%",
    "active_tax_yr": "活跃税/年(本金)%", "round_trip_cost": "单边摩擦%",
    "annualized": "年化%", "max_dd": "回撤%", "可信": "数字可信",
    "超阈值": "超B&O6.5%", "flag": "标记"})
out = out.sort_values(["策略", "活跃税/年(净值)%"], ascending=[True, False])
out.to_csv(OUT, index=False, encoding="utf-8-sig")

# ── 控制台：先给可信策略的完整排行，再单列不可信的 ──────────────
ok = out[out["数字可信"]].sort_values("活跃税/年(净值)%", ascending=False)
bad = out[~out["数字可信"]]

print(f"=== 全平台活跃税排行（可信样本 {len(ok)} 个，B&O 基准 {BENCH*100:.1f}%/年）===")
cols = ["策略", "档位", "口径", "笔数", "年数", "活跃税/年(净值)%", "活跃税/年(本金)%",
        "单边摩擦%", "超B&O6.5%"]
cols = [c for c in cols if c in ok.columns]
print(ok[cols].head(30).to_string(index=False))

n_over = int(ok["超B&O6.5%"].sum())
print(f"\n→ {n_over}/{len(ok)} 个策略活跃税超过 6.5%/年")
print(f"   活跃税/年(净值) 中位 {ok['活跃税/年(净值)%'].median():.2f}% "
      f"| 均值 {ok['活跃税/年(净值)%'].mean():.2f}% "
      f"| 最高 {ok['活跃税/年(净值)%'].max():.2f}%（{ok.iloc[0]['策略']} / {ok.iloc[0]['档位']}）")

if len(bad):
    print(f"\n⚠️ 数字不可信（{len(bad)} 个，已排除出排行）：")
    print(bad[["策略", "档位", "标记"]].to_string(index=False))

print(f"\n分组明细已写出：{OUT}")
