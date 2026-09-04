#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""全平台 trades CSV **流水健康度体检**（第一遍：不需要价格数据）。

为什么要有这个
--------------
SKILL §6 第 12 条立的规矩是「从 trades CSV 重建任何指标前先过流水自洽性三查」。
但那三查里只有第 ① 条（action 前缀匹配）是纯文本检查，**第 ② 条「逐标的累计持仓
不得为负」其实完全不需要价格**——只看买/卖的股数就够了。

这意味着可以**先把全库 100+ 个流水文件用近乎零成本的方式过一遍**，把坏文件挑出来，
再只对健康文件做要拉全市场行情的 NAV 重建 / 活跃税计算。否则 123 个文件直接上
重建要拉约 5000 只标的 × 16 年（约 2000 万行），内存和时间都吃不消。

检查项（全部不依赖行情）
------------------------
C1 缺列 / 读不了            → 直接判死
C2 action 无法归一化        → 方向识别失败（会静默反向现金流）
C3 shares <= 0 或 NaN       → 非法成交
C4 price <= 0 或 NaN        → 无法估值
C5 **逐标的累计持仓出现负值** → 卖出了日志里从未记录的买入（导出漏 append 的典型症状）
C6 终值持仓为负的标的数      → 平不掉，等于流水账对不上
C7 成交日不在交易日历        → 非交易日成交（日期口径错 or 数据源错位）

用法
----
    # 全库体检（默认扫 data/results 下所有 trades_*.csv）
    python run_trades_health_check.py

    # 只看坏文件
    python run_trades_health_check.py --bad-only

    # 换个根目录 / 限制数量
    python run_trades_health_check.py --root data/results/monthly_rebalance --max 200
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nav_recon_util as U                      # noqa: E402
from run_monthly_rebalance import get_conn      # noqa: E402

EPS = 1e-6


def _calendar(start, end):
    """交易日历（daily 表 distinct trade_date），用于 C7。"""
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT DISTINCT trade_date FROM daily "
            "WHERE trade_date>=? AND trade_date<=? ORDER BY trade_date",
            (str(int(start)), str(int(end)))).fetchall()
    finally:
        conn.close()
    return {int(r[0]) for r in rows}


def check_file(path, cal=None):
    """对单个 trades CSV 做 C1~C7 检查。返回 dict（缺列/异常时用 floats('nan') 占位）。"""
    out = {"file": path, "n": 0, "err": ""}
    try:
        df = pd.read_csv(path)
    except Exception as e:                       # C1
        out["err"] = f"读取失败 {type(e).__name__}: {e}"
        return out

    df = df.rename(columns={c: str(c).strip() for c in df.columns})
    need = ["date", "action", "code", "price", "shares"]
    miss = [c for c in need if c not in df.columns]
    if miss:                                     # C1
        # `*_fills_assumption.csv` 是 run_fill_assumption_compare.py 的成交成本
        # 对比产物，schema 本就是 (amount, mkt_slip, lim_slip, ...)，不含
        # price/shares —— 它是**另一种产物**，不是坏流水，不能算进不健康。
        if os.path.basename(path).endswith("_fills_assumption.csv"):
            out["err"] = "非流水文件（成交成本对比产物），已跳过"
            out["status_hint"] = "skip"
            return out
        out["err"] = f"缺列 {miss}；现有列={list(df.columns)}"
        return out

    n0 = len(df)
    out["n"] = n0
    df["date"] = pd.to_numeric(df["date"], errors="coerce")
    df["code"] = df["code"].astype(str).str.strip()
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # C2 action 可识别性
    acts = df["action"].map(U._norm_action)
    bad_act = int((~acts.isin(["BUY", "SELL"])).sum())
    df["_dir"] = acts.map({"BUY": 1.0, "SELL": -1.0})

    # C3 / C4
    bad_sh = int((df["shares"].isna() | (df["shares"] <= 0)).sum())
    bad_px = int((df["price"].isna() | (df["price"] <= 0)).sum())

    # C5 / C6：逐标的累计持仓
    d = df.dropna(subset=["_dir", "shares"])
    d = d.assign(_qty=d["_dir"] * d["shares"])
    grp = d.groupby("code")["_qty"]
    cum = grp.cumsum()
    # 终值必须是「累计和的最后一个」，不能写 grp.transform("last")——
    # 后者取的是**最后一笔的原始股数**（末笔多为清仓卖出、天然为负），
    # 会把每一个正常平仓的标的都误判成"终值为负"（实测 123/123 全误报）。
    last = cum.groupby(d["code"]).last()
    neg_codes = d.loc[cum < -EPS, "code"].nunique()
    min_hold = float(cum.min()) if len(cum) else 0.0
    end_neg = int((last < -EPS).sum())

    # C7 交易日历
    off_cal = -1
    if cal is not None:
        dd = df["date"].dropna()
        if len(dd):
            off_cal = int((~dd.astype(int).isin(cal)).sum())

    out.update({"bad_action": bad_act, "bad_shares": bad_sh, "bad_price": bad_px,
                "neg_hold_codes": neg_codes, "min_hold": min_hold,
                "end_neg_codes": end_neg, "off_calendar": off_cal,
                "n_codes": int(d["code"].nunique()),
                "d0": int(df["date"].min()) if df["date"].notna().any() else 0,
                "d1": int(df["date"].max()) if df["date"].notna().any() else 0})
    return out


def main():
    ap = argparse.ArgumentParser(description="全平台 trades CSV 流水健康度体检")
    ap.add_argument("--root", default="data/results")
    ap.add_argument("--glob", default="trades_*.csv")
    ap.add_argument("--max", type=int, default=500)
    ap.add_argument("--bad-only", action="store_true", help="只列不健康的文件")
    ap.add_argument("--out", default="data/results/trades_health_check.csv")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.root, "**", args.glob),
                             recursive=True))
    files = [f for f in files if os.sep + ".cache" + os.sep not in f]
    # 排除本脚本自己的输出（上一轮结果会被下一次扫描当成输入）
    files = [f for f in files
             if os.path.abspath(f) != os.path.abspath(args.out)]
    # 🔴 批量扫描不得静默截断：--max 按**路径排序**砍尾部，可能整族漏扫
    #    （实例：全平台 211 个 trades 用 --max 200，漏的 11 个全是问题最严重、
    #     字母序最靠后的那一族 → 危险信号被少报）。0 = 不限。
    if args.max and len(files) > args.max:
        print(f"⚠️ 共 {len(files)} 个文件，--max {args.max} 只处理前 {args.max} 个，"
              f"漏掉 {len(files) - args.max} 个（按路径排序，可能整族漏扫）"
              f"→ 用 --max 0 表示不限")
    files = files[: args.max] if args.max else files
    if not files:
        print(f"未找到匹配文件：{args.root}/**/{args.glob}")
        return
    print(f"扫描 {len(files)} 个流水文件（根目录 {args.root}）...")

    # 先收集日期范围，一次性取日历
    lo, hi = None, None
    for f in files:
        try:
            d = pd.read_csv(f, usecols=["date"])["date"]
            d = pd.to_numeric(d, errors="coerce").dropna()
            if len(d):
                a, b = int(d.min()), int(d.max())
                lo = a if lo is None else min(lo, a)
                hi = b if hi is None else max(hi, b)
        except Exception:
            pass
    cal = _calendar(lo, hi) if (lo and hi) else None
    if cal is not None:
        print(f"交易日历 {min(cal)}~{max(cal)}，{len(cal)} 个交易日")

    rows = [check_file(f, cal) for f in files]
    df = pd.DataFrame(rows)

    def _status(r):
        if r.get("status_hint") == "skip":
            return "－ 非流水文件"
        if r.get("err"):
            return "✗ 不可用"
        if (r["neg_hold_codes"] > 0 or r["end_neg_codes"] > 0
                or r["bad_action"] > 0):
            return "✗ 流水不自洽"
        if r["bad_shares"] > 0 or r["bad_price"] > 0 or r["off_calendar"] > 0:
            return "△ 数值可疑"
        return "✓ 健康"

    df["status"] = df.apply(_status, axis=1)
    # 显式排序：坏文件在前（直接按 status 字符串排会把 ✗ 排到最后，反了）
    _ord = {"✗ 流水不自洽": 0, "✗ 不可用": 1, "△ 数值可疑": 2,
            "✓ 健康": 3, "－ 非流水文件": 4}
    df = (df.assign(_o=df["status"].map(_ord).fillna(9))
            .sort_values(["_o", "file"]).drop(columns="_o"))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")

    show = df[df["status"] != "✓ 健康"] if args.bad_only else df
    print(f"\n{'文件':<58}{'笔数':>7}{'标的':>6}{'负持仓':>8}{'终值负':>7}"
          f"{'方向错':>7}{'价异常':>7}{'非交易日':>8}  状态")
    print("-" * 130)
    for _, r in show.iterrows():
        if r.get("err"):
            print(f"{r['file'][:56]:<58}{'':>7}{'':>6}{'':>8}{'':>7}{'':>7}"
                  f"{'':>7}{'':>8}  {r['status']}  {str(r['err'])[:40]}")
            continue
        print(f"{r['file'][:56]:<58}{r['n']:>7,}{r['n_codes']:>6}"
              f"{r['neg_hold_codes']:>8}{r['end_neg_codes']:>7}"
              f"{r['bad_action']:>7}{r['bad_price']:>7}{r['off_calendar']:>8}"
              f"  {r['status']}")

    n_bad = int((df["status"].str.startswith("✗")).sum())
    n_warn = int((df["status"].str.startswith("△")).sum())
    n_ok = int((df["status"].str.startswith("✓")).sum())
    n_skip = int((df["status"].str.startswith("－")).sum())
    print(f"\n合计 {len(df)} 个文件：✓ 健康 {n_ok} | △ 数值可疑 {n_warn} | "
          f"✗ 不可用 {n_bad} | － 非流水文件 {n_skip}")
    print("明细已写出：" + args.out)


if __name__ == "__main__":
    main()
