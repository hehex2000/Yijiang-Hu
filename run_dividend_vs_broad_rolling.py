# -*- coding: utf-8 -*-
"""
方向①：红利 vs 宽基 全滚动窗口分析
========================================
视频《2020—2024，沪深300竟没跑赢存款？》最值钱的研究问题：
    "红利跑赢宽基，是 A 股常态，还是 2021-2024 特定 regime？"

方法
----
- 从 index_tr_official 直接读取 4 条官方全收益日线（已确认对齐 2010-01-04 ~ 2026-08-28）：
    H00300.CSI  沪深300全收益
    H00906.CSI  中证800全收益
    H00922.CSI  中证红利全收益
    H20955.CSI  红利低波100全收益
- 生成所有 5 年（≈1260 交易日）滚动窗口，日频滚动（约 2780 个窗口）
- 每窗口计算：红利全收益累计收益 − 宽基全收益累计收益 = 累计超额（含年度化）
- 全分布统计：胜率 / 中位数 / 分位数
- 按窗口起点年份做 regime 归因（哪类起点 regime 下红利占优）

输出
----
- 控制台纯文本表格（胡老师偏好，不花哨）
- 可选 --csv 写出每个窗口的明细（start/end/div_cum/broad_cum/excess/ann_excess）

注意：本脚本是纯研究脚本，直接读官方全收益表，刻意不经 bench_index 映射，
以免 .SH→.CSI 映射在边缘情形干扰；口径已在输出头显式声明。
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

try:
    from config import DATA
except Exception:
    DATA = {"local_db_path": r"D:\tu-shareData\astock_daily.db"}

DB = DATA.get("local_db_path", "")

# 官方全收益代码（已确认 index_tr_official 中 2010-01-04~2026-08-28 完整覆盖）
TR_SERIES = {
    "HS300_TR": "H00300.CSI",   # 沪深300全收益
    "ZZ800_TR": "H00906.CSI",   # 中证800全收益
    "ZHHL_TR":  "H00922.CSI",   # 中证红利全收益
    "HLDB_TR":  "H20955.CSI",   # 红利低波100全收益
}

# 对比配对：红利全收益 vs 宽基全收益
PAIRS = [
    ("红利低波100", "HLDB_TR", "沪深300", "HS300_TR"),
    ("红利低波100", "HLDB_TR", "中证800", "ZZ800_TR"),
    ("中证红利",   "ZHHL_TR",  "沪深300", "HS300_TR"),
    ("中证红利",   "ZHHL_TR",  "中证800", "ZZ800_TR"),
]

WINDOW_YEARS = 5
TRADING_DAYS_PER_YEAR = 252
WIN = WINDOW_YEARS * TRADING_DAYS_PER_YEAR  # 1260 默认窗口长度


def load_tr(conn, tr_code):
    df = pd.read_sql_query(
        "SELECT trade_date, close FROM index_tr_official "
        "WHERE tr_code=? ORDER BY trade_date",
        conn, params=(tr_code,))
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna()
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df = df.set_index("trade_date")["close"].sort_index()
    return df


def rolling_cum_ret(series, win=WIN):
    """向量化：窗口长度 win 的累计收益（末/初 - 1）。
    返回 (start_dates, end_dates, cum_ret)，长度 = len - win + 1。
    """
    arr = series.values.astype(float)
    idx = series.index
    if len(arr) < win + 1:
        return None, None, None
    start_px = arr[:-win + 1]
    end_px = arr[win - 1:]
    cum = end_px / start_px - 1.0
    s_dates = idx[:-win + 1]
    e_dates = idx[win - 1:]
    return s_dates, e_dates, cum


def ann_from_cum(cum, years=WINDOW_YEARS):
    """累计收益 -> 年度化（几何）。cum 为 fraction。"""
    return np.sign(cum + 1) * (np.abs(cum + 1) ** (1.0 / years) - 1.0)


def fmt_pct(x, nd=2):
    return f"{x*100:+.2f}%" if x is not None else "  n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="", help="写出每窗口明细 CSV 路径")
    ap.add_argument("--window", type=int, default=WIN, help="滚动窗口交易日数（默认1260=5年）")
    args = ap.parse_args()
    win = args.window

    conn = __import__("sqlite3").connect(DB)

    print("=" * 78)
    print("方向①：红利 vs 宽基 全滚动窗口分析（官方全收益口径 index_tr_official）")
    print("=" * 78)
    print(f"数据源 : {DB}")
    print(f"窗口   : {WINDOW_YEARS} 年 = {win} 交易日（日频滚动）")
    print(f"序列   : " + ", ".join(f"{k}={v}" for k, v in TR_SERIES.items()))
    print()

    # 载入并对齐 4 条序列（inner join 共同交易日）
    raw = {k: load_tr(conn, v) for k, v in TR_SERIES.items()}
    conn.close()
    common = None
    for s in raw.values():
        common = s.index if common is None else common.intersection(s.index)
    aligned = {k: v.reindex(common).dropna() for k, v in raw.items()}
    for k in aligned:
        print(f"  {k:10s} 行数={len(aligned[k])}  区间={aligned[k].index[0].date()} ~ {aligned[k].index[-1].date()}")
    print()

    results = []
    for div_name, div_key, broad_name, broad_key in PAIRS:
        s_dates, e_dates, div_cum = rolling_cum_ret(aligned[div_key], win)
        _, _, broad_cum = rolling_cum_ret(aligned[broad_key], win)
        if div_cum is None or broad_cum is None:
            print(f"[跳过] {div_name} vs {broad_name} 数据不足")
            continue
        excess = div_cum - broad_cum
        ann_excess = ann_from_cum(div_cum + 1) - ann_from_cum(broad_cum + 1)
        n = len(excess)
        win_rate = float(np.mean(excess > 0)) * 100
        med = float(np.median(excess))
        q10 = float(np.percentile(excess, 10))
        q25 = float(np.percentile(excess, 25))
        q75 = float(np.percentile(excess, 75))
        q90 = float(np.percentile(excess, 90))
        ann_med = float(np.median(ann_excess))
        results.append(dict(div=div_name, broad=broad_name, n=n,
                            excess=excess, ann_excess=ann_excess,
                            s_dates=s_dates, e_dates=e_dates,
                            win=win_rate, med=med, q10=q10, q25=q25,
                            q75=q75, q90=q90, ann_med=ann_med))

        print("-" * 78)
        print(f"配对：{div_name}全收益 (H)  −  {broad_name}全收益 (H)   [窗口数={n}]")
        print(f"  累计超额胜率        : {win_rate:5.1f}%   （{int((excess>0).sum())}/{n} 窗口红利跑赢）")
        print(f"  累计超额中位数      : {fmt_pct(med)}")
        print(f"  累计超额分位数 10/25: {fmt_pct(q10)} / {fmt_pct(q25)}")
        print(f"  累计超额分位数 75/90: {fmt_pct(q75)} / {fmt_pct(q90)}")
        print(f"  年度化超额中位数    : {fmt_pct(ann_med)}")
        # regime 归因：按起点年份
        yr = pd.Series(excess, index=s_dates).groupby(s_dates.year)
        print(f"  ── 按窗口起点年份 regime 归因 ──")
        print(f"  {'起点年':>6s} {'窗口':>5s} {'胜率':>7s} {'中位累计超额':>12s} {'中位年化超额':>12s}")
        for y, grp in yr:
            g = grp.values
            gw = float(np.mean(g > 0)) * 100
            gm = float(np.median(g))
            ga = float(np.median(ann_excess[s_dates.year == y]))
            print(f"  {y:>6d} {len(g):>5d} {gw:>6.1f}% {fmt_pct(gm):>12s} {fmt_pct(ga):>12s}")
        print()

    # 聚焦：起点在 2017-2021 的窗口（覆盖 2021-2024 红利/中特估 regime）
    print("=" * 78)
    yr_label = f"{win} 交易日（≈{win/TRADING_DAYS_PER_YEAR:.0f}年）"
    print(f"聚焦：起点 2017-2021 的 {yr_label} 窗口（覆盖 2021-2024 红利/中特估 regime）")
    print("=" * 78)
    for r in results:
        mask = (r["s_dates"].year >= 2017) & (r["s_dates"].year <= 2021)
        sub = r["excess"][mask]
        sub_ann = r["ann_excess"][mask]
        if len(sub) == 0:
            continue
        w = float(np.mean(sub > 0)) * 100
        print(f"  {r['div']:8s}−{r['broad']:6s} : 窗口={len(sub):4d}  胜率={w:5.1f}%  "
              f"中位累计={fmt_pct(float(np.median(sub)))}  中位年化={fmt_pct(float(np.median(sub_ann)))}")

    # CSV 明细
    if args.csv:
        out = []
        for div_name, div_key, broad_name, broad_key in PAIRS:
            sd, ed, dc = rolling_cum_ret(aligned[div_key], win)
            _, _, bc = rolling_cum_ret(aligned[broad_key], win)
            out.append(pd.DataFrame(dict(
                pair=f"{div_name}-{broad_name}",
                start=sd, end=ed,
                div_cum=dc, broad_cum=bc,
                excess_cum=dc - bc,
                excess_ann=ann_from_cum(dc + 1) - ann_from_cum(bc + 1),
            )))
        big = pd.concat(out, ignore_index=True)
        big.to_csv(args.csv, index=False)
        print(f"\n[CSV] 每窗口明细已写出：{args.csv}  （{len(big)} 行）")

    print("\n结论速读：")
    print("  - 全历史胜率≈50% 即'红利-宽基'无系统性偏向；>60% 才称得上常态占优。")
    print("  - 若仅 2019-2021 起点年胜率显著高于其它年，则'红利跑赢'是 2021-2024 特定 regime，")
    print("    而非 A 股常态 —— 视频标题的'沪深300没跑赢存款'是窗口选择偏差（survivorship/区间选择）。")


if __name__ == "__main__":
    main()
