# -*- coding: utf-8 -*-
"""
V2 护栏 第三轮：确定最优点 + 分离「护栏贡献」与「分散贡献」
────────────────────────────────────────────────────────────
前两轮结论：
  轮1（5只固定）：⑤回落护栏/⑦中位数都变差；只有 ⑧暴涨2.0 小胜
                  （+83.52% / -57.17% / 0.235 vs 基线 +80.53% / -60.40% / 0.230）
  轮2（放开持仓数）：持仓集中度才是主因——
                  5只→10只(无护栏) 就从 +80.53% 提到 +120.45%、回撤 -60.40%→-56.81%；
                  ⑧2.0+15只 达 +145.49% / -54.00% / 0.348 / 超额 +51.44pp（首次跑赢基准）。

本轮要回答三件事：
  1) ⑧ 阈值 1.5 还是 2.0 更好（在 10/15 只下复检，避免 5 只样本噪声误导）
  2) 分散到 20 只是否还有增益（找拐点）
  3) 15 只时护栏还剩多少独立贡献（用「无护栏·15只」对照分离）
另：轮2 的 rename 逻辑把轮1 的 C(⑧2.0·5只) 产物覆盖了，本轮一并补回。
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
from tune_magic_v2_guard2 import guard_tag, BASE_KEEP

VARIANTS = [
    ("C ⑧2.0 · 5只",    {"spike_guard": 2.0}, 5),    # 补回被覆盖的轮1产物
    ("M ⑧1.5 · 10只",   {"spike_guard": 1.5}, 10),
    ("L ⑧1.5 · 15只",   {"spike_guard": 1.5}, 15),
    ("O 无护栏 · 15只",  {},                   15),   # 分离护栏贡献的对照组
    ("N ⑧2.0 · 20只",   {"spike_guard": 2.0}, 20),
    ("P ⑧1.5 · 20只",   {"spike_guard": 1.5}, 20),
]

# 前两轮已落盘、可直接复用的产物
CARRY = {
    "A 基线(无护栏) · 5只": BASE_KEEP,
    "J ⑧1.5 · 5只":        f"{OUT_DIR}/backtest_v2_sg15_{START}_{END}_n5.csv",
    "G 无护栏 · 10只":      f"{OUT_DIR}/backtest_v2_{START}_{END}_n10.csv",
    "H ⑧2.0 · 10只":       f"{OUT_DIR}/backtest_v2_sg2_{START}_{END}_n10.csv",
    "I ⑧2.0 · 15只":       f"{OUT_DIR}/backtest_v2_sg2_{START}_{END}_n15.csv",
}


def main():
    if os.path.exists(BASE_CSV) and not os.path.exists(BASE_KEEP):
        shutil.copy(BASE_CSV, BASE_KEEP)

    results = dict(CARRY)
    for label, kw, tn in VARIANTS:
        raw_path = f"{OUT_DIR}/backtest_v2{guard_tag(kw)}_{START}_{END}.csv"
        final_path = raw_path.replace(".csv", f"_n{tn}.csv")
        if os.path.exists(final_path):
            print(f"  [skip] {label}", flush=True)
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
            os.replace(raw_path, final_path)
            results[label] = final_path
            print(f"    [done] {time.time()-t0:.0f}s", flush=True)

    shutil.copy(BASE_KEEP, BASE_CSV)          # 恢复用户手动跑的基线

    byr, btotal = bench_yearly()
    rows, yr_tbl = [], {}
    for label, path in results.items():
        if not os.path.exists(path):
            print(f"  [miss] {label} → {path}")
            continue
        m = metrics_from_csv(path)
        # 从标签解析持仓数与护栏，便于交叉看
        tn = label.split("·")[-1].strip()
        gd = "无" if "无护栏" in label or "基线" in label else label.split("⑧")[1].split("·")[0].strip()
        rows.append({"变体": label, "持仓": tn, "⑧阈值": gd,
                     "总收益%": round(m["total"], 2),
                     "年化%": round(m["annual"], 2),
                     "最大回撤%": round(m["mdd"], 2),
                     "夏普": round(m["sharpe"], 4),
                     "终值": round(m["final"], 0),
                     "超额基准pp": round(m["total"] - btotal, 2)})
        yr_tbl[label] = yearly_from_csv(path)

    summary = pd.DataFrame(rows).sort_values("夏普", ascending=False)
    print("\n" + "=" * 104)
    print(f"  📊 V2 护栏×分散 全变体总表（zz800 | 20万 | {START}~{END} | 基准000906 {btotal:+.2f}%）")
    print("=" * 104)
    print(summary.to_string(index=False))

    years = sorted(set().union(*[set(v) for v in yr_tbl.values()]))
    ydf = pd.DataFrame({"年份": years,
                        "基准%": [round(byr.get(y, np.nan), 2) for y in years]})
    for label in yr_tbl:
        ydf[label] = [round(yr_tbl[label].get(y, np.nan), 2) for y in years]
    print("\n" + "=" * 104)
    print("  📅 逐年收益（%，真实趴账口径）")
    print("=" * 104)
    print(ydf.to_string(index=False))

    summary.to_csv("data/results/magic_v2_guard_final_summary.csv",
                   index=False, encoding="utf-8-sig")
    ydf.to_csv("data/results/magic_v2_guard_final_yearly.csv",
               index=False, encoding="utf-8-sig")
    print("\n  已保存 → data/results/magic_v2_guard_final_summary.csv")
    print("  已保存 → data/results/magic_v2_guard_final_yearly.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
