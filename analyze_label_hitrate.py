# -*- coding: utf-8 -*-
"""
体检标签说对率分析（基于 backtest_diagnose 输出的 CSV）
================================================================================
从 outputs/backtest_diagnose_*.csv 读取逐股回测结果，按标签分组统计：
  - 每个历史体检点 T 各出一张分组表（机会/中性/陷阱）
  - 跨所有 T 的整体汇总表
  - 结论：到底哪个组真正涨幅最高（ret 与 alpha 双口径）

"说对率"定义（按标签方向）：
  - 机会：预期跑赢大盘 → alpha_12m > 0 算说对
  - 陷阱：预期跑输大盘 → alpha_12m < 0 算说对
  - 中性：本无明确方向预测 → 不计入"方向说对率"（标 —），但给"跑赢大盘率"供参考
"跑赢大盘率"：统一 P(alpha_12m > 0)，直观看该标签选出的票有多少跑赢中证800等权。

用法：
  python analyze_label_hitrate.py --csv outputs/backtest_diagnose_zz800_-35.csv
================================================================================
"""
import argparse
import os
import sys

import pandas as pd
import numpy as np


def hit_rate(row):
    a = row["alpha_12m"]
    if pd.isna(a):
        return np.nan
    if row["label"] == "机会":
        return 1.0 if a > 0 else 0.0
    if row["label"] == "陷阱":
        return 1.0 if a < 0 else 0.0
    return np.nan  # 中性不计入方向说对率


def aggregate(g):
    return pd.Series({
        "N": len(g),
        "均ret_6m": g["ret_6m"].mean(),
        "均ret_12m": g["ret_12m"].mean(),
        "均alpha_6m": g["alpha_6m"].mean(),
        "均alpha_12m": g["alpha_12m"].mean(),
        "跑赢大盘率12M": g["win12"].mean(),
        "方向说对率12M": g["hit12"].mean(),
    })


def fmt_pct(v):
    if pd.isna(v):
        return "  —  "
    return f"{v*100:>6.1f}%"


def print_group_table(df_sub, title):
    print("\n" + "=" * 92)
    print(title)
    print("=" * 92)
    hdr = (f"{'标签':<6}{'N':>5}{'均收益6M':>10}{'均收益12M':>11}"
           f"{'均超额6M':>10}{'均超额12M':>11}{'跑赢大盘率':>11}{'方向说对率':>11}")
    print(hdr)
    for lab in ("机会", "中性", "陷阱"):
        s = df_sub[df_sub["label"] == lab]
        if s.empty:
            continue
        n = len(s)
        r6 = s["ret_6m"].mean(); r12 = s["ret_12m"].mean()
        a6 = s["alpha_6m"].mean(); a12 = s["alpha_12m"].mean()
        win = s["win12"].mean()
        hit = s["hit12"].mean()
        print(f"{lab:<6}{n:>5}{fmt_pct(r6):>10}{fmt_pct(r12):>11}"
              f"{fmt_pct(a6):>10}{fmt_pct(a12):>11}{fmt_pct(win):>11}"
              f"{fmt_pct(hit) if not pd.isna(hit) else '  —  ':>11}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="outputs/backtest_diagnose_zz800_-35.csv")
    p.add_argument("--out-dir", default="outputs")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    df["win12"] = (df["alpha_12m"] > 0).astype(float)
    df["win6"] = (df["alpha_6m"] > 0).astype(float)
    df["hit12"] = df.apply(hit_rate, axis=1)

    # ---- 分时点分组表 ----
    ts = sorted(df["T"].unique())
    print("\n" + "#" * 92)
    print(f"# 体检标签说对率分析  源文件: {os.path.basename(args.csv)}")
    print(f"# 体检点: {len(ts)} 个  ({ts[0]} ~ {ts[-1]})")
    print("#" * 92)

    for T in ts:
        sub = df[df["T"] == T]
        print_group_table(sub, f"【体检点 {T}】各标签分组统计（12M口径）")

    # ---- 整体汇总 ----
    overall = df.groupby("label").apply(aggregate).reindex(["机会", "中性", "陷阱"])
    print("\n" + "#" * 92)
    print("【整体汇总】跨所有体检点累积")
    print("#" * 92)
    print_group_table(df, "整体（全部T合并）")

    # ---- 结论：哪个组真正涨幅最高 ----
    print("\n" + "=" * 92)
    print("结论：哪个组真正涨幅最高（按 均收益12M / 均超额12M 排序）")
    print("=" * 92)
    rk = overall.sort_values("均ret_12m", ascending=False)
    for i, (lab, row) in enumerate(rk.iterrows(), 1):
        print(f"  {i}. {lab:<6} 均收益12M={fmt_pct(row['均ret_12m']).strip()}  "
              f"均超额12M={fmt_pct(row['均alpha_12m']).strip()}  "
              f"N={int(row['N'])}")
    best_ret = rk.index[0]
    rk2 = overall.sort_values("均alpha_12m", ascending=False)
    best_alpha = rk2.index[0]
    print("-" * 92)
    print(f"  按绝对涨幅(均ret_12m)：最高 = {best_ret}")
    print(f"  按相对大盘(均alpha_12m)：最高 = {best_alpha}")
    # 说对率
    opp = df[df.label == '机会']['hit12'].mean()
    trp = df[df.label == '陷阱']['hit12'].mean()
    print(f"  方向说对率12M：机会={fmt_pct(opp).strip()}  陷阱={fmt_pct(trp).strip()}")
    print("=" * 92)

    # ---- 导出 CSV ----
    os.makedirs(args.out_dir, exist_ok=True)
    byT = df.groupby(["T", "label"]).apply(aggregate).reset_index()
    byT_path = os.path.join(args.out_dir, "label_hitrate_byT.csv")
    overall_path = os.path.join(args.out_dir, "label_hitrate_overall.csv")
    byT.to_csv(byT_path, index=False, encoding="utf-8-sig")
    overall.reset_index().to_csv(overall_path, index=False, encoding="utf-8-sig")
    print(f"\n[输出] 分时点表: {byT_path}")
    print(f"[输出] 整体汇总: {overall_path}")


if __name__ == "__main__":
    main()
