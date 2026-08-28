# -*- coding: utf-8 -*-
"""
dist_to_poc 反过拟合三件套之(3): walk-forward 验证
- 复用宽宇宙 point-in-time 面板 m2_wide_panel.csv (n=664443 只·月, 2010-07~2026-06)
- 成本: 组合层净成本 0.60%/月 (整本书 2 round-trip, 见 vp_factor_wide_net.py)
- 方向: 负IC因子 -> 做多低 dist_to_poc(贴近成交密集区) / 做空高 dist_to_poc(远离)
- 三件套其余两件已在 vp_factor_wide_net.py 完成(宽成本 + 扩展候选池)
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PANEL = "data/results/volume_profile/m2_wide_panel.csv"
OUT_DIR = "data/results/volume_profile"
FACTOR = "vp_dist_to_poc"
RET_GROSS = "fwd_ret"
MONTHLY_TURNOVER_COST = 0.0060   # 0.60%/月 (组合层净成本)
MIN_N = 20


def load_panel():
    p = pd.read_csv(PANEL)
    p["year"] = (p["date"] // 10000).astype(int)
    return p


def portfolio_ls(sub, factor, ret):
    """负IC因子: 做多低因子组 / 做空高因子组, 返回逐月毛利差序列"""
    months = []
    for d, g in sub.groupby("date"):
        if len(g) < MIN_N:
            continue
        r = g[factor].rank(method="first")
        n = max(5, len(g) // 5)
        longs = g.loc[r <= n]
        shorts = g.loc[r >= len(g) - n]
        months.append(longs[ret].mean() - shorts[ret].mean())
    return pd.Series(months) if months else pd.Series([0.0])


def ic_of(sub, factor, ret):
    ics = []
    for d, g in sub.groupby("date"):
        if len(g) < MIN_N:
            continue
        rho, _ = spearmanr(g[factor], g[ret])
        if np.isfinite(rho):
            ics.append(rho)
    return float(np.nanmean(ics)), len(ics)


def ls_stats(ls_series):
    """输入逐月毛利差; 返回 毛月均/净月均/t/年化毛/年化净/胜率"""
    g = ls_series.mean()
    sd = ls_series.std()
    t = g / sd * np.sqrt(len(ls_series)) if sd > 0 else 0.0
    net = g - MONTHLY_TURNOVER_COST
    win = (ls_series > 0).mean()
    return dict(
        ls_gross_m=g * 100,
        ls_net_m=net * 100,
        ls_t=t,
        ann_gross=g * 12 * 100,
        ann_net=net * 12 * 100,
        win_rate=win * 100,
        n_months=len(ls_series),
    )


def per_year_table(p):
    rows = []
    for y, g in p.groupby("year"):
        ls = portfolio_ls(g, FACTOR, RET_GROSS)
        ic, nic = ic_of(g, FACTOR, RET_GROSS)
        st = ls_stats(ls)
        rows.append(dict(year=y, n_stocks=len(g), ic=ic,
                         ls_gross_m=st["ls_gross_m"], ls_net_m=st["ls_net_m"],
                         ls_t=st["ls_t"], ann_net=st["ann_net"], win=st["win_rate"]))
    return pd.DataFrame(rows)


def rolling_walkforward(p, train_months=60, test_months=12, step=12):
    """滚动 train/test 折叠, 仅 test 期计 out-of-sample 净 LS"""
    dates = sorted(p["date"].unique())
    folds = []
    eq = 1.0
    eq_log = []
    i = train_months
    while i + test_months <= len(dates):
        train_d = dates[max(0, i - train_months): i]
        test_d = dates[i: i + test_months]
        sub = p[p["date"].isin(test_d)]
        ls = portfolio_ls(sub, FACTOR, RET_GROSS)
        st = ls_stats(ls)
        # 折叠内逐月净复利累计
        net_monthly = ls - MONTHLY_TURNOVER_COST
        for r in net_monthly.values:
            eq *= (1 + r)
        y0 = test_d[0] // 10000
        y1 = test_d[-1] // 10000
        folds.append(dict(test_window=f"{y0}-{y1}", ic=ic_of(sub, FACTOR, RET_GROSS)[0],
                          ann_net=st["ann_net"], ls_t=st["ls_t"], n_months=st["n_months"],
                          cum_eq=eq))
        eq_log.append(eq)
        i += step
    return pd.DataFrame(folds)


def expanding_equity(p):
    """全期逐月净 LS 复利累计 (组合层净成本), 诚实 P&L"""
    ls_all = []
    for d, g in p.groupby("date"):
        if len(g) < MIN_N:
            continue
        r = g[FACTOR].rank(method="first")
        n = max(5, len(g) // 5)
        longs = g.loc[r <= n]
        shorts = g.loc[r >= len(g) - n]
        ls_all.append((d, longs[RET_GROSS].mean() - shorts[RET_GROSS].mean()))
    s = pd.Series([x[1] for x in ls_all], index=[x[0] for x in ls_all])
    net = s - MONTHLY_TURNOVER_COST
    eq = (1 + net).cumprod()
    df = pd.DataFrame({"date": eq.index, "net_ls": net.values, "cum_eq": eq.values})
    return df


def main():
    p = load_panel()
    print("=== 全期基线 (宽宇宙净成本, 已知) ===")
    ls = portfolio_ls(p, FACTOR, RET_GROSS)
    st = ls_stats(ls)
    ic, nic = ic_of(p, FACTOR, RET_GROSS)
    print("IC=%.4f (n=%d月) | LS净月均=%.3f%% t=%.2f | 年化净=%.1f%% | 胜率=%.1f%%"
          % (ic, nic, st["ls_net_m"], st["ls_t"], st["ann_net"], st["win_rate"]))

    print("\n=== (A) 逐年稳定性 (walk-forward 逐窗口 OOS) ===")
    yt = per_year_table(p)
    yt.to_csv(os.path.join(OUT_DIR, "wf_year_table.csv"), index=False)
    print(yt.to_string(index=False))

    print("\n=== (B) 滚动 walk-forward 折叠 (train60m/test12m, 仅test期计OOS净LS) ===")
    wf = rolling_walkforward(p)
    wf.to_csv(os.path.join(OUT_DIR, "wf_rolling_folds.csv"), index=False)
    print(wf.to_string(index=False))

    print("\n=== (C) 扩展窗口累计净净值 (全期逐月净LS复利) ===")
    eq = expanding_equity(p)
    eq.to_csv(os.path.join(OUT_DIR, "wf_equity.csv"), index=False)
    final_eq = eq["cum_eq"].iloc[-1]
    max_dd = ((eq["cum_eq"].cummax() - eq["cum_eq"]) / eq["cum_eq"].cummax()).max() * 100
    print("最终累计净值=%.2fx | 最大回撤=%.1f%% | 月数=%d" % (final_eq, max_dd, len(eq)))

    # 分布统计 (fold 级)
    print("\n=== (D) 折叠分布统计 (抗过拟合判定) ===")
    pos_folds = (wf["ann_net"] > 0).sum()
    print("滚动折叠数: %d | 净正年份/折叠: %d (%.0f%%)"
          % (len(wf), pos_folds, pos_folds / len(wf) * 100))
    print("最佳折叠: %.1f%% (%s) | 最差折叠: %.1f%% (%s)"
          % (wf["ann_net"].max(), wf.loc[wf["ann_net"].idxmax(), "test_window"],
             wf["ann_net"].min(), wf.loc[wf["ann_net"].idxmin(), "test_window"]))
    print("逐年净正年数: %d / %d" % ((yt["ann_net"] > 0).sum(), len(yt)))
    # holdout 近期
    recent = yt[yt["year"] >= 2024]
    print("HOLDOUT 2024-2026: 净=%s"
          % ", ".join("%.1f%%(%d)" % (r, y) for y, r in zip(recent["year"], recent["ann_net"])))
    print("\n输出: wf_year_table.csv / wf_rolling_folds.csv / wf_equity.csv")


if __name__ == "__main__":
    main()
