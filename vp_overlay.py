# -*- coding: utf-8 -*-
"""
dist_to_poc 作为现有反转信号的[确认 overlay] 验证
- 基线: 平台 factors/reversal.py 的 rev_21 (21d 收益, raw, 做多跌最多)
- overlay: dist_to_poc (价相对成交密集区中心偏离, raw, 做多贴近POC)
- 语义: 仅当两者同向(都指均值回复同一方向)才持仓, 否则空仓
- 成本: 组合层净 0.60%/月 (与 vp_factor_wide_net / vp_walkforward 一致)
- 公平对比: base 全仓换手扣全额成本; overlay 持仓更少 -> 成本按 coverage 比例折算
"""
import os
import sqlite3
import numpy as np
import pandas as pd
import config

PANEL = "data/results/volume_profile/m2_wide_panel.csv"
OUT_DIR = "data/results/volume_profile"
FACTOR = "vp_dist_to_poc"
BASE = "rev_21"
TC_MONTH = 0.0060
Q = 5
MIN_N = 20


def load_panel():
    p = pd.read_csv(PANEL)
    p["date"] = p["date"].astype(int)
    return p


def compute_rev21():
    conn = sqlite3.connect(config.DATA.get("local_db_path", ""))
    df = pd.read_sql_query("SELECT ts_code, trade_date, close FROM daily", conn)
    conn.close()
    df = df.sort_values(["ts_code", "trade_date"])
    df["rev_21"] = df.groupby("ts_code")["close"].pct_change(21)
    df = df.rename(columns={"trade_date": "date"})
    df["date"] = df["date"].astype(int)
    return df[["ts_code", "date", "rev_21"]]


def overlay_portfolio(m):
    """逐月: base=rev_21 五分位多空; overlay=base 多/空中再要求 vp 同向确认(中位分割)"""
    base_g, ov_g, covs = [], [], []
    for d, g in m.groupby("date"):
        if len(g) < MIN_N:
            continue
        r_base = g[BASE].rank(method="first")
        r_vp = g[FACTOR].rank(method="first")
        n = max(5, len(g) // Q)
        longs = g.loc[r_base <= n]
        shorts = g.loc[r_base >= len(g) - n]
        base_g.append(longs["fwd_ret"].mean() - shorts["fwd_ret"].mean())

        # overlay: 多头确认=base多头且vp偏低(贴近POC); 空头确认=base空头且vp偏高
        half = len(g) // 2
        ov_longs = longs.loc[r_vp <= half]
        ov_shorts = shorts.loc[r_vp >= half]
        if len(ov_longs) > 0 and len(ov_shorts) > 0:
            ov_g.append(ov_longs["fwd_ret"].mean() - ov_shorts["fwd_ret"].mean())
            covs.append((len(ov_longs) + len(ov_shorts)) / (len(longs) + len(shorts)))
        else:
            ov_g.append(np.nan)
            covs.append(np.nan)
    return (pd.Series(base_g), pd.Series(ov_g, dtype=float),
            pd.Series(covs, dtype=float))


def stats(s, cost_mult=1.0):
    s = s.dropna()
    g = s.mean()
    sd = s.std()
    t = g / sd * np.sqrt(len(s)) if sd > 0 else 0.0
    net = g - TC_MONTH * cost_mult
    return dict(n=len(s), gross_m=g * 100, ls_t=t,
                ann_gross=g * 12 * 100, ann_net=net * 12 * 100,
                win=(s > 0).mean() * 100)


def main():
    p = load_panel()
    rev = compute_rev21()
    m = p.merge(rev, on=["ts_code", "date"], how="inner").dropna(
        subset=[BASE, FACTOR, "fwd_ret"])
    print("合并面板: %d 只·月 (rev_21 + dist_to_poc 对齐)" % len(m))

    base, ov, cov = overlay_portfolio(m)
    cov_mean = cov.mean()
    print("\n=== 对比: base rev_21 多空 vs dist_to_poc 确认 overlay ===")
    sb = stats(base, 1.0)
    so = stats(ov, cov_mean)   # overlay 持仓少 -> 成本按 coverage 折算(公平)
    sgb = stats(base, 0.0)     # 毛(无成本) 看纯信号质量
    sgo = stats(ov, 0.0)
    rows = [
        dict(策略="base rev_21 (毛)", 月均_g=sgb["gross_m"], 年化_g=sgb["ann_gross"],
             月均净=sgb["gross_m"], 年化净=sgb["ann_gross"], t=sgb["ls_t"], 胜率=sgb["win"], 覆盖=100.0),
        dict(策略="base rev_21 (净,全额成本)", 月均_g=sb["gross_m"], 年化_g=sb["ann_gross"],
             月均净=sb["ann_net"], 年化净=sb["ann_net"], t=sb["ls_t"], 胜率=sb["win"], 覆盖=100.0),
        dict(策略="overlay 确认 (毛)", 月均_g=sgo["gross_m"], 年化_g=sgo["ann_gross"],
             月均净=sgo["gross_m"], 年化净=sgo["ann_gross"], t=sgo["ls_t"], 胜率=sgo["win"],
             覆盖=cov_mean * 100),
        dict(策略="overlay 确认 (净,比例成本)", 月均_g=so["gross_m"], 年化_g=so["ann_gross"],
             月均净=so["ann_net"], 年化净=so["ann_net"], t=so["ls_t"], 胜率=so["win"],
             覆盖=cov_mean * 100),
    ]
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "overlay_confirm.csv"), index=False)
    print(out.to_string(index=False))

    # 确认率 / 条件 IC
    agree = (m.groupby("date").apply(
        lambda g: ((g[BASE].rank() <= len(g) // 2) == (g[FACTOR].rank() <= len(g) // 2)).mean()
    ) if False else None)
    # 简化: 全样本同向率
    r_base = m[BASE].rank(method="first")
    r_vp = m[FACTOR].rank(method="first")
    same_dir = ((r_base <= len(m) // 2) == (r_vp <= len(m) // 2))
    print("\n=== 全局同向率(两因子指同一方向的占比) ===")
    print("%.1f%%" % (same_dir.mean() * 100))

    print("\n结论要点:")
    print("- overlay 覆盖(持仓占比)约 %.0f%%" % (cov_mean * 100))
    print("- 毛信号: base 月均 %.3f%% vs overlay %.3f%% (overlay %s)"
          % (sgb["gross_m"], sgo["gross_m"],
             "更优" if sgo["gross_m"] > sgb["gross_m"] else "更弱"))
    print("- 净(公平成本): base 年化 %.1f%% vs overlay 年化 %.1f%%"
          % (sb["ann_net"], so["ann_net"]))
    print("输出: overlay_confirm.csv")


if __name__ == "__main__":
    main()
