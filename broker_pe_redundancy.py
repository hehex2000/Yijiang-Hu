# -*- coding: utf-8 -*-
"""
Gate 5 冗余检验 —— 券商 PB 分位的信号到底是「券商特有」还是「市场估值择时」

核心对照（同一套择时规则，只换信号和标的）：
  A: 券商PB分位   → 择时买卖【券商】      （原始策略）
  B: 全市场PB分位 → 择时买卖【券商】      ← 若 B≈A，券商PB 是冗余的
  C: 券商PB分位   → 择时买卖【中证800】   ← 若 C 好，说明本质是市场择时信号
  D: 全市场PB分位 → 择时买卖【中证800】   （市场择时基准线）
  E: 券商PB分位对全市场PB分位做【滚动残差化】后 → 择时买卖券商
     ← 若 E 塌缩，说明券商PB 的增量 = 0，判冗余

残差化用【滚动窗口】估计 beta（只用 ≤t 的数据），不用全样本 OLS，
否则会引入前视偏差（用未来的 beta 修正今天的信号）。

用法
----
  python broker_pe_redundancy.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

OUT_DIR = os.path.join("data", "results", "broker_pe")
SEC = os.path.join(OUT_DIR, "sector_daily.csv")
MKT = os.path.join(OUT_DIR, "market_daily.csv")

COST_BUY = 0.00125
COST_SELL = 0.00175
WINDOW = 750


# ------------------------------------------------------------ 工具
def rolling_percentile(s, window=WINDOW, min_p=None):
    min_p = min_p or int(window * 0.8)
    r = s.rolling(window, min_periods=min_p)
    return r.apply(lambda x: (x < x.iloc[-1]).mean() * 100, raw=False)


def rolling_residual(y, x, window=WINDOW, min_periods=None):
    """滚动 OLS 残差：y_t - (a_t + b_t * x_t)，beta 只用 [t-window, t] 的数据估。"""
    min_periods = min_periods or int(window * 0.8)
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(y)
    resid = np.full(n, np.nan)
    for i in range(min_periods, n):
        s = slice(max(0, i - window), i)          # 不含 i 本身 → 无前视
        yy, xx = y[s], x[s]
        m = np.isfinite(yy) & np.isfinite(xx)
        if m.sum() < min_periods:
            continue
        X = np.column_stack([np.ones(m.sum()), xx[m]])
        try:
            beta = np.linalg.lstsq(X, yy[m], rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        if np.isfinite(x[i]):
            resid[i] = y[i] - (beta[0] + beta[1] * x[i])
    return resid


def timing_backtest(sig_pct, asset_ret, lo=30.0, hi=70.0, rebal_n=60):
    """按分位信号做仓位择时（≤lo 满仓 / ≥hi 空仓 / 中间半仓）。"""
    sig_pct = np.asarray(sig_pct, dtype=float)
    asset_ret = np.asarray(asset_ret, dtype=float)
    n = len(sig_pct)
    target = np.where(sig_pct <= lo, 1.0, np.where(sig_pct >= hi, 0.0, 0.5))
    target = np.where(np.isfinite(sig_pct), target, 0.0)

    if rebal_n and rebal_n > 1:
        tgt = target.copy()
        for i in range(1, n):
            if i % rebal_n != 0:
                tgt[i] = tgt[i - 1]
        target = tgt

    nav = np.ones(n)
    w = 0.0
    cost_tot = 0.0
    for i in range(1, n):
        r = asset_ret[i] if np.isfinite(asset_ret[i]) else 0.0
        gross = w * r
        delta = target[i] - w
        c = delta * COST_BUY if delta > 0 else (-delta) * COST_SELL
        cost_tot += c
        nav[i] = nav[i - 1] * (1 + gross - c)
        w = target[i]

    years = n / 250.0
    tot = nav[-1] - 1
    ann = (1 + tot) ** (1 / years) - 1 if tot > -1 else np.nan
    dd = (pd.Series(nav) / pd.Series(nav).cummax() - 1).min()
    rets = pd.Series(nav).pct_change().fillna(0.0)
    sharpe = rets.mean() / rets.std() * np.sqrt(250) if rets.std() > 0 else np.nan
    return dict(ann=ann, tot=tot, mdd=dd, sharpe=sharpe,
                exposure=float(np.nanmean(target)), cost=cost_tot)


def buy_hold_stats(asset_ret):
    r = pd.Series(asset_ret).fillna(0.0)
    nav = (1 + r).cumprod()
    n = len(nav)
    tot = nav.iloc[-1] - 1
    ann = (1 + tot) ** (250.0 / n) - 1 if tot > -1 else np.nan
    dd = (nav / nav.cummax() - 1).min()
    sharpe = r.mean() / r.std() * np.sqrt(250) if r.std() > 0 else np.nan
    return dict(ann=ann, tot=tot, mdd=dd, sharpe=sharpe, exposure=1.0, cost=0.0)


# ------------------------------------------------------------ 主流程
def main():
    sec = pd.read_csv(SEC, dtype={"trade_date": str}).sort_values("trade_date")
    mkt = pd.read_csv(MKT, dtype={"trade_date": str}).sort_values("trade_date")
    print(f"[券商] {len(sec)} 行 {sec.trade_date.min()}~{sec.trade_date.max()}")
    print(f"[全市场] {len(mkt)} 行")

    m = sec.merge(mkt[["trade_date", "pb_pct750"]].rename(
        columns={"pb_pct750": "mkt_pb_pct750"}), on="trade_date", how="left")
    m["mkt_pb_pct750"] = m["mkt_pb_pct750"].ffill()

    sec_ret = m["ret"].fillna(0.0).values
    bch_ret = pd.Series(m["bench_close"].astype(float)).pct_change().fillna(0.0).values

    brk_pct = m["pb_pct750"].values
    mkt_pct = m["mkt_pb_pct750"].values

    # 滚动残差化（无前视）
    print("[计算] 滚动残差化（券商PB分位 ~ 全市场PB分位）...")
    resid = rolling_residual(brk_pct, mkt_pct)
    resid_pct = rolling_percentile(pd.Series(resid))
    m["brk_resid_pct"] = resid_pct

    print("\n" + "=" * 96)
    print("【Gate 5】信号 × 标的 交叉对照（每60日调仓，已扣组合层成本）")
    print("=" * 96)
    print(f"{'#':<3}{'信号':<22}{'标的':<12}{'年化%':>9}{'回撤%':>10}{'夏普':>8}"
          f"{'仓位%':>8}{'成本%':>8}")
    print("-" * 96)

    cases = [
        ("A", "券商PB分位", brk_pct, "券商", sec_ret),
        ("B", "全市场PB分位", mkt_pct, "券商", sec_ret),
        ("C", "券商PB分位", brk_pct, "中证800", bch_ret),
        ("D", "全市场PB分位", mkt_pct, "中证800", bch_ret),
        ("E", "券商PB残差分位", resid_pct, "券商", sec_ret),
    ]
    results = {}
    for tag, sname, sig, aname, aret in cases:
        r = timing_backtest(sig, aret)
        results[tag] = r
        print(f"{tag:<3}{sname:<22}{aname:<12}{r['ann']*100:>9.2f}{r['mdd']*100:>10.2f}"
              f"{r['sharpe']:>8.2f}{r['exposure']*100:>8.1f}{r['cost']*100:>8.2f}")

    print("-" * 96)
    bh_sec = buy_hold_stats(sec_ret)
    bh_bch = buy_hold_stats(bch_ret)
    print(f"{'—':<3}{'（无择时）':<22}{'券商':<12}{bh_sec['ann']*100:>9.2f}{bh_sec['mdd']*100:>10.2f}"
          f"{bh_sec['sharpe']:>8.2f}{100.0:>8.1f}{0.0:>8.2f}")
    print(f"{'—':<3}{'（无择时）':<22}{'中证800':<12}{bh_bch['ann']*100:>9.2f}{bh_bch['mdd']*100:>10.2f}"
          f"{bh_bch['sharpe']:>8.2f}{100.0:>8.1f}{0.0:>8.2f}")

    # ---------------- 判定 ----------------
    print("\n" + "=" * 96)
    print("【判定】")
    print("=" * 96)
    a, b, e = results["A"], results["B"], results["E"]
    print(f"  A 券商PB→券商      年化 {a['ann']*100:6.2f}%   夏普 {a['sharpe']:.2f}")
    print(f"  B 全市场PB→券商    年化 {b['ann']*100:6.2f}%   夏普 {b['sharpe']:.2f}")
    print(f"  E 残差信号→券商    年化 {e['ann']*100:6.2f}%   夏普 {e['sharpe']:.2f}")
    print()
    if np.isfinite(e["ann"]) and e["ann"] < a["ann"] * 0.6:
        print("  ❌ 残差化后年化塌缩到原始的 60% 以下 → 券商PB 的绝大部分信息"
              "已被【全市场估值】解释，判冗余，不单独入库。")
    elif np.isfinite(e["ann"]) and e["ann"] >= a["ann"] * 0.85:
        print("  ✅ 残差化后年化基本保持 → 券商PB 含【独立于市场估值】的增量信息。")
    else:
        print("  🟡 残差化后部分衰减 → 有增量但不强，可当 overlay 确认层。")
    print()
    if np.isfinite(b["ann"]) and b["ann"] >= a["ann"] * 0.85:
        print("  ⚠️ 用【全市场PB】择时券商的效果接近甚至优于用券商PB"
              " → 直接换成市场估值信号更简单、更不易过拟合。")

    # 相关性
    msk = np.isfinite(brk_pct) & np.isfinite(mkt_pct)
    from scipy import stats as st
    corr = st.spearmanr(brk_pct[msk], mkt_pct[msk]).correlation
    print(f"\n  券商PB分位 vs 全市场PB分位 的 Spearman 相关: {corr:.3f}")
    print(f"  （接近 1 说明券商估值分位几乎就是市场估值分位的镜像）")


if __name__ == "__main__":
    main()
