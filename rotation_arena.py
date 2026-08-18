#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轮动擂台 (rotation arena) —— 统一记分牌
=========================================
用途：把平台上每一个"周期再平衡 / 轮动"类策略，按同一张表对齐到它的基准指数，
给出 跑赢/跑输 的干净 verdict。避免"这个轮动亏了→轮动都亏"的过度外推。

用法：
    python rotation_arena.py                 # 打印当前记分牌
    python rotation_arena.py --add file.csv  # 把 file.csv(同 schema) 的新行追加进记分牌
    python rotation_arena.py --win           # 只看跑赢的
    python rotation_arena.py --lose          # 只看跑输的

记分牌文件：data/results/rotation_scoreboard.csv
schema: strategy,category,universe,rebalance_freq,period,total_ret_pct,
        bench_name,bench_ret_pct,excess_pp,ann_pct,mdd_pct,sharpe,verdict,note
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BOARD = os.path.join(HERE, "data", "results", "rotation_scoreboard.csv")


def load():
    with open(BOARD, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def fmt_row(r):
    excess = float(r["excess_pp"])
    tag = "▲WIN " if r["verdict"].upper() == "WIN" else "▼LOSE"
    return (f"  {tag} {r['strategy']:<20} | {r['universe']:<22} | "
            f"{r['rebalance_freq']:<14} | {r['period']:<12} | "
            f"策略 {float(r['total_ret_pct']):>7.1f}% vs {r['bench_name']} "
            f"{float(r['bench_ret_pct']):>7.1f}% | 超额 {excess:>+7.1f}pp | {r['verdict']}")


def main():
    only = None
    addfile = None
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--win":
            only = "WIN"
        elif a == "--lose":
            only = "LOSE"
        elif a == "--add" and i + 1 < len(args):
            addfile = args[i + 1]
            i += 1
        i += 1

    if addfile:
        with open(addfile, encoding="utf-8-sig") as f:
            new = list(csv.DictReader(f))
        rows = load()
        rows.extend(new)
        with open(BOARD, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[arena] 已追加 {len(new)} 行 -> {BOARD}")
        return

    rows = load()
    if only:
        rows = [r for r in rows if r["verdict"].upper() == only]

    _rows = load()
    n_win = sum(1 for r in _rows if r["verdict"].strip().upper() == "WIN")
    n_lose = sum(1 for r in _rows if r["verdict"].strip().upper() == "LOSE")

    print("=" * 110)
    print("  轮动擂台 · 策略 vs 基准 记分牌")
    print(f"  总样本 {len(load())} 个  |  ▲WIN {n_win}  ▼LOSE {n_lose}")
    print("=" * 110)
    for r in rows:
        print(fmt_row(r))
    print("=" * 110)
    print("  判定规则：excess_pp = 策略总收益 - 基准总收益；>0 即跑赢基准。")
    print("  结论不是'轮动 vs 指数'，而是'因子/池子/regime 三维'决定 edge 归属。")


if __name__ == "__main__":
    main()
