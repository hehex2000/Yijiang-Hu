# -*- coding: utf-8 -*-
"""
V2 护栏 第二轮：⑧暴涨阈值敏感性 + 持仓集中度对回撤的影响
────────────────────────────────────────────────────────────
第一轮结论：
  C ⑧暴涨2.0 是唯一全面小胜基线的护栏（+83.52% / -57.17% / 0.235
  vs 基线 +80.53% / -60.40% / 0.230），但改善幅度有限。
  ⑤回落护栏、⑦中位数反而显著变差。

第一轮诊断出的更深问题：
  2023 年 ⑤护栏仅替换了 1 只票（世纪华通→华新建材），全年收益就从
  +6.87% 崩到 -16.33%（差 23pp）。top_n=5 时单票权重 20%，
  任何护栏的统计效应都被单票噪声淹没。
  ∴ -60.4% 回撤的主因很可能是「持仓过度集中」，而非「暴利股污染」。
  本轮用 top_n=10/15 直接验证。

文件名冲突处理：run_backtest_v2 的产物名只含 start/end/护栏tag，不含 top_n，
  不同 top_n 会互相覆盖（甚至覆盖用户手动跑的基线）→ 先备份基线，
  每个变体跑完立刻 rename 到带 _n{top_n} 的唯一名。
"""
import os
import sys
import time
import io
import shutil
import contextlib
import pandas as pd
import numpy as np

import run_magic_v2 as v2
from tune_magic_v2_guard import (yearly_from_csv, metrics_from_csv,
                                 bench_yearly, START, END, POOL, CAP,
                                 OUT_DIR, BASE_CSV)

BASE_KEEP = f"{OUT_DIR}/_baseline_keep_{START}_{END}.csv"

# (标签, 护栏参数, top_n)
VARIANTS = [
    ("J ⑧1.5 · 5只",       {"spike_guard": 1.5}, 5),
    ("K ⑧3.0 · 5只",       {"spike_guard": 3.0}, 5),
    ("G 无护栏 · 10只",     {},                   10),
    ("H ⑧2.0 · 10只",      {"spike_guard": 2.0}, 10),
    ("I ⑧2.0 · 15只",      {"spike_guard": 2.0}, 15),
]


def guard_tag(kw):
    t = ""
    if kw.get("ebit_stat", "mean") != "mean":
        t += f"_{kw['ebit_stat']}"
    if kw.get("profit_guard", 0) > 0:
        t += f"_pg{kw['profit_guard']:g}".replace(".", "")
    if kw.get("spike_guard", 0) > 0:
        t += f"_sg{kw['spike_guard']:g}".replace(".", "")
    if kw.get("ebit_conservative"):
        t += "_cons"
    return t


def main():
    if not os.path.exists(BASE_CSV):
        print(f"[ERROR] 缺少 v2 基线 CSV: {BASE_CSV}")
        return 1
    # 备份用户手动跑的基线，防被 top_n≠5 的无护栏变体覆盖
    shutil.copy(BASE_CSV, BASE_KEEP)
    print(f"  [safe] 已备份基线 → {BASE_KEEP}", flush=True)

    results = {"A 基线(原v2) · 5只": BASE_KEEP,
               "C ⑧2.0 · 5只": f"{OUT_DIR}/backtest_v2_sg2_{START}_{END}.csv"}

    for label, kw, tn in VARIANTS:
        raw_path = f"{OUT_DIR}/backtest_v2{guard_tag(kw)}_{START}_{END}.csv"
        final_path = raw_path.replace(".csv", f"_n{tn}.csv")
        if os.path.exists(final_path):
            print(f"  [skip] {label} 已存在", flush=True)
            results[label] = final_path
            continue
        print(f"\n  ▶ {label}  {kw} top_n={tn}", flush=True)
        t0 = time.time()
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                v2.run_backtest_v2(START, END, top_n=tn, stock_pool=POOL,
                                   capital=CAP, industry_cap=2, ebit_years=3,
                                   trend_filter=True, **kw)
        except Exception as e:
            print(f"    [FAIL] {type(e).__name__}: {e}", flush=True)
            print("\n".join(buf.getvalue().splitlines()[-8:]), flush=True)
            continue
        if os.path.exists(raw_path):
            os.replace(raw_path, final_path)      # 立刻改名，避免下个变体覆盖
            results[label] = final_path
            print(f"    [done] {time.time()-t0:.0f}s → {os.path.basename(final_path)}",
                  flush=True)
        else:
            print(f"    [WARN] 未找到产物 {raw_path}", flush=True)

    # 基线文件若被覆盖则恢复
    shutil.copy(BASE_KEEP, BASE_CSV)

    byr, btotal = bench_yearly()
    rows, yr_tbl = [], {}
    for label, path in results.items():
        if not os.path.exists(path):
            continue
        m = metrics_from_csv(path)
        rows.append({"变体": label,
                     "总收益%": round(m["total"], 2),
                     "年化%": round(m["annual"], 2),
                     "最大回撤%": round(m["mdd"], 2),
                     "夏普": round(m["sharpe"], 4),
                     "终值": round(m["final"], 0),
                     "超额基准pp": round(m["total"] - btotal, 2)})
        yr_tbl[label] = yearly_from_csv(path)

    summary = pd.DataFrame(rows).sort_values("夏普", ascending=False)
    print("\n" + "=" * 96)
    print(f"  📊 第二轮：⑧阈值敏感性 + 持仓集中度（zz800 | 20万 | 基准000906 {btotal:+.2f}%）")
    print("=" * 96)
    print(summary.to_string(index=False))

    years = sorted(set().union(*[set(v) for v in yr_tbl.values()]))
    ydf = pd.DataFrame({"年份": years,
                        "基准%": [round(byr.get(y, np.nan), 2) for y in years]})
    for label in yr_tbl:
        ydf[label] = [round(yr_tbl[label].get(y, np.nan), 2) for y in years]
    print("\n" + "=" * 96)
    print("  📅 逐年收益对比（%，真实趴账口径）")
    print("=" * 96)
    print(ydf.to_string(index=False))

    summary.to_csv("data/results/magic_v2_guard2_summary.csv",
                   index=False, encoding="utf-8-sig")
    ydf.to_csv("data/results/magic_v2_guard2_yearly.csv",
               index=False, encoding="utf-8-sig")
    print("\n  已保存 → data/results/magic_v2_guard2_summary.csv")
    print("  已保存 → data/results/magic_v2_guard2_yearly.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
