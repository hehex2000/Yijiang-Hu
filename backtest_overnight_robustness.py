# -*- coding: utf-8 -*-
"""
隔夜策略 · 周末效应 分年稳健性检验
================================
对 all/momentum/strong × 普通/周末 六组，按年份拆分 base_open(次日/周一开盘卖)
与 hold_close(次日/周一收盘卖) 的毛均值与胜率，判断"周末正效应"是整段持续
还是被某几年撑着。
⚠️ 按合理代理定义复刻，非逐字复刻原视频。
"""
import sys, os, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest_overnight as B
import backtest_overnight_scenarios as S

START, END = "20150101", "20260630"


def add_ma(df):
    df = df.sort_values(["ts_code", "trade_date"])
    g = df.groupby("ts_code", sort=False)
    df["ma20"] = g["close"].transform(lambda x: x.rolling(20, min_periods=10).mean())
    df["ma60"] = g["close"].transform(lambda x: x.rolling(60, min_periods=30).mean())
    return df


def get_sub(df, mode, weekend):
    if mode == "all":
        sub = B.select(df, "all")
    elif mode == "momentum":
        sub = B.select(df, "momentum")
    elif mode == "strong":
        base = B.select(df, "all")
        sub = base[(base["close"] > base["ma20"]) & (base["ma20"] > base["ma60"])]
    else:
        raise ValueError(mode)
    if weekend:
        td = pd.to_datetime(sub["trade_date"].astype(str), format="%Y%m%d")
        sub = sub[td.dt.weekday == 4]
    return sub


def yearly(sub):
    sub = sub.copy().reset_index(drop=True)
    sub["year"] = sub["trade_date"] // 10000
    c = sub["close"].to_numpy(dtype="float64")
    bo = sub["next_open"].to_numpy(dtype="float64") / c - 1.0
    hc = sub["next_close"].to_numpy(dtype="float64") / c - 1.0
    rows = []
    for y, idx in sub.groupby("year").groups.items():
        b = bo[idx.to_numpy()]
        h = hc[idx.to_numpy()]
        if len(b) == 0:
            continue
        rows.append(dict(year=int(y), n=len(b),
                         base_open_mean_pct=round(float(b.mean()) * 100, 4),
                         base_open_win_pct=round(float((b > 0).mean()) * 100, 2),
                         hold_close_mean_pct=round(float(h.mean()) * 100, 4),
                         hold_close_win_pct=round(float((h > 0).mean()) * 100, 2)))
    return rows


def main():
    t0 = time.time()
    print(f"[load] {START}~{END} ...  注: 按合理代理定义复刻, 非逐字复刻原视频", flush=True)
    df = B.load_data(START, END)
    df = B.build_signals(df)
    df = add_ma(df)
    df = S.add_next(df)
    print(f"[load done] {len(df):,} rows, {time.time()-t0:.1f}s", flush=True)

    all_rows = []
    for mode in ["all", "momentum", "strong"]:
        for wknd in [False, True]:
            tag = f"{mode}{'_wknd' if wknd else ''}"
            sub = get_sub(df, mode, wknd)
            rows = yearly(sub)
            for r in rows:
                r["mode"] = mode
                r["weekend"] = wknd
                all_rows.append(r)
            print(f"\n=== {tag} 分年 ===", flush=True)
            for r in rows:
                print(f"  {r['year']}  N={r['n']:>7,}  base_open {r['base_open_mean_pct']:+.3f}%(胜{r['base_open_win_pct']:.0f})  "
                      f"hold_close {r['hold_close_mean_pct']:+.3f}%(胜{r['hold_close_win_pct']:.0f})", flush=True)

    od = "data/results/overnight"
    os.makedirs(od, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(f"{od}/variants_by_year.csv", index=False, encoding="utf-8-sig")
    print(f"\n[done] 已保存 {od}/variants_by_year.csv  ({time.time()-t0:.1f}s)  注: 按合理代理定义复刻, 非逐字复刻原视频")


if __name__ == "__main__":
    main()
