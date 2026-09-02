# -*- coding: utf-8 -*-
"""
⑩ 20日新高占比 —— Gate 1 信号强度检验
======================================
问题：低占比（超卖极值）之后，市场/策略的未来收益真的更高吗？（视频：<18% = 历史大底）

方法（按项目时序信号方法论）：
  - 信号 = 20日新高占比的 750 日滚动分位（分位口径；原始值水平漂移污染，见⑤教训）
  - 前瞻收益：value NAV（hfq）与基准 000906 全收益（H00906.CSI 官方口径），未来 20 / 60 日
  - Spearman IC + NW t（lag=H，主口径）；五分位前向收益单调性；分年 IC 稳健性
用法：python new_high20_gate1.py
"""
import sys
import sqlite3

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, ".")
from bench_index import load_benchmark

H_LIST = [20, 60]
SIG = "data/results/new_high20/ratio20.csv"
NAV = "data/results/value_strategy/backtest_result_hfq_20100104_20260831.csv"
BENCH = "000906.SH"


def nw_t(x, y, lag):
    """OLS slope y~x 的 NW t（lag=H，重叠样本校正）。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 60:
        return np.nan, n
    X = np.column_stack([np.ones(n), x])
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ (X.T @ y)
    e = y - X @ beta
    u = X * e[:, None]
    S = u.T @ u
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)
        S += w * (u[k:].T @ u[: len(u) - k] + u[: len(u) - k].T @ u[k:])
    V_nw = XtX_inv @ S @ XtX_inv
    se = np.sqrt(V_nw[1, 1])
    return (beta[1] / se if se > 0 else np.nan), n


def main():
    sig = pd.read_csv(SIG, dtype={"trade_date": str})
    sig["pct"] = sig.ratio.rolling(750, min_periods=250).apply(
        lambda w: (w[-1] >= w[:-1]).mean() * 100, raw=True)

    nav = pd.read_csv(NAV, dtype={"trade_date": str})[["trade_date", "portfolio_value_full"]]
    conn = sqlite3.connect("D:/tu-shareData/astock_daily.db")
    bench, meta = load_benchmark(BENCH, "20100104", "20260831",
                                 conn=conn, nav_price_mode="hfq")
    assert bench is not None and len(bench) > 3000, "基准加载失败（先自证再往下跑）"
    print(f"基准：{meta['resolved_code']}  {meta['note']}  {len(bench)} 日")

    df = sig.merge(nav, on="trade_date", how="inner").merge(
        bench[["trade_date", "close"]].rename(columns={"close": "bench"}),
        on="trade_date", how="inner").sort_values("trade_date").reset_index(drop=True)
    print(f"合并样本 {len(df)} 日  {df.trade_date.min()}~{df.trade_date.max()}")
    assert len(df) > 3000, "样本不足"

    for target, px in [("value NAV(hfq)", "portfolio_value_full"), ("基准000906全收益", "bench")]:
        for H in H_LIST:
            fwd = df[px].shift(-H) / df[px] - 1
            x, y = df.pct, fwd
            ok = np.isfinite(x) & np.isfinite(y)
            rho, p = stats.spearmanr(x[ok], y[ok])
            t_nw, n = nw_t(x, y, lag=H)
            print(f"  {target:16s} H={H:3d}  Spearman={rho:+.4f}(p={p:.3g})  "
                  f"NW_t={t_nw:+.2f}(n={n})")

    print("\n=== 五分位前向收益（基准 000906 全收益；若视频对：Q1 低占比应最高）===")
    for H in H_LIST:
        fwd = df.bench.shift(-H) / df.bench - 1
        q = pd.qcut(df.pct, 5, labels=False, duplicates="drop")
        g = fwd.groupby(q).agg(["mean", "median", "count"])
        g.index = [f"Q{i+1}" for i in g.index]
        print(f" H={H}:")
        print("  " + g.round(4).to_string().replace("\n", "\n  "))

    print("\n=== 分年 Spearman IC（市场口径, H=20）===")
    fwd = df.bench.shift(-20) / df.bench - 1
    tmp = pd.DataFrame({"year": df.trade_date.str[:4], "pct": df.pct, "fwd": fwd}).dropna()
    for y, s in tmp.groupby("year"):
        rho, p = stats.spearmanr(s.pct, s.fwd)
        print(f"  {y}: IC={rho:+.3f} (n={len(s)})")


if __name__ == "__main__":
    main()
