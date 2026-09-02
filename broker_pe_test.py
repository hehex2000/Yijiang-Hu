# -*- coding: utf-8 -*-
"""
券商 PE 陷阱因子 —— 时序检验层（Gate 1 / Gate 4）

与 broker_pe_factor.py 配套。核心是正确处理【样本重叠】：
日频采样 + H 日 forward return → 相邻样本重叠 (H-1)/H，OLS t 值严重虚高。
本脚本同时给出三种口径，任何一个不达标都不下结论：
  1) 朴素 t（仅供参考，明知虚高）
  2) Newey-West HAC t（lag = H，主口径）
  3) 非重叠子样本 t（每 H 日抽一个，最保守）

方向约定
--------
  IC > 0 : PE 分位越高 → 未来超额越高 → 即「低 PE = 差」→ PE 陷阱成立（视频假说）
  IC < 0 : PE 分位越低 → 未来超额越高 → 即「低 PE = 便宜」→ 传统价值成立

用法
----
  python broker_pe_test.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

OUT_DIR = os.path.join("data", "results", "broker_pe")
SRC = os.path.join(OUT_DIR, "sector_daily.csv")

HORIZONS = (60, 120, 250)
WINDOWS = (500, 750, 1000)
QUANTILES = 5


# ------------------------------------------------------------ 统计工具
def nw_tstat(y, x, lag):
    """OLS y = a + b*x，返回 (beta, Newey-West HAC t)。

    V = (X'X)^-1 · S · (X'X)^-1，其中
    S = Σ_t u_t u_t' + Σ_{l=1..lag} (1 - l/(lag+1)) · Σ_t (u_t u_{t-l}' + u_{t-l} u_t')

    ⚠️ 踩过的坑：V 后面**不要**再乘 n。S 已经是对全部 t 求和，
       再乘 n 会让 se 放大 √n 倍（约 55 倍 @n=3000），t 值塌缩到 ~0.02，
       把强信号误判成「完全不显著」。这个 bug 差点让整批结果被误读。
    """
    m = np.isfinite(y) & np.isfinite(x)
    y, x = np.asarray(y)[m], np.asarray(x)[m]
    n = len(y)
    if n < lag + 10:
        return np.nan, np.nan
    X = np.column_stack([np.ones(n), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    XtX_inv = np.linalg.inv(X.T @ X)
    U = X * resid[:, None]
    S = U.T @ U
    for l in range(1, lag + 1):
        w = 1.0 - l / (lag + 1.0)
        G = U[l:].T @ U[:-l]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv          # 不乘 n
    se = np.sqrt(max(np.diag(V)[1], 0.0))
    return beta[1], (beta[1] / se if se > 0 else np.nan)


def nw_tstat_auto(y, x):
    """Newey-West 自动带宽：lag = floor(4*(n/100)^(2/9))，用于交叉验证 lag=H 的结果。"""
    n = int(np.isfinite(y).sum())
    lag = int(np.floor(4 * (n / 100.0) ** (2 / 9.0)))
    return nw_tstat(y, x, max(lag, 1))


def naive_tstat(y, x):
    m = np.isfinite(y) & np.isfinite(x)
    y, x = np.asarray(y)[m], np.asarray(x)[m]
    if len(y) < 10:
        return np.nan, np.nan
    r, p = stats.pearsonr(x, y)
    # 用相关系数的 t
    n = len(y)
    t = r * np.sqrt((n - 2) / max(1e-12, 1 - r ** 2))
    return r, t


def spearman_ic(x, y, min_n=30):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < min_n:
        return np.nan
    return stats.spearmanr(np.asarray(x)[m], np.asarray(y)[m]).correlation


# ------------------------------------------------------------ 主检验
def run_horizon(sec, factor, H, window):
    col = f"{factor}_pct{window}"
    ret = f"fwd_exc_{H}"
    if col not in sec.columns or ret not in sec.columns:
        return None
    d = sec[[col, ret, "trade_date"]].dropna()
    if len(d) < 200:
        return None

    x = d[col].values
    y = d[ret].values

    ic = spearman_ic(x, y)
    r_naive, t_naive = naive_tstat(y, x)
    b_nw, t_nw = nw_tstat(y, x, lag=H)
    _, t_nw_auto = nw_tstat_auto(y, x)

    # 非重叠子样本：每 H 日取一个
    idx = np.arange(0, len(d), H)
    dsub = d.iloc[idx]
    ic_sub = spearman_ic(dsub[col].values, dsub[ret].values, min_n=8)
    _, t_sub = naive_tstat(dsub[ret].values, dsub[col].values)

    n_eff = len(dsub)                      # 有效独立样本
    return dict(factor=factor, window=window, H=H, n_obs=len(d), n_eff=n_eff,
                ic=ic, t_naive=t_naive, beta_nw=b_nw, t_nw=t_nw, t_nw_auto=t_nw_auto,
                ic_sub=ic_sub, t_sub=t_sub)


def run_quantile(sec, factor, H, window):
    """分位分层：按因子分位分 5 档，看各档平均未来超额。"""
    col = f"{factor}_pct{window}"
    ret = f"fwd_exc_{H}"
    d = sec[[col, ret]].dropna()
    if len(d) < 200:
        return None
    d = d.copy()
    d["q"] = pd.qcut(d[col], QUANTILES, labels=False, duplicates="drop")
    g = d.groupby("q")[ret].agg(["mean", "median", "count", "std"])
    g["胜率%"] = d.groupby("q")[ret].apply(lambda s: (s > 0).mean() * 100)
    return g


def main():
    sec = pd.read_csv(SRC, dtype={"trade_date": str})
    print(f"[数据] {len(sec)} 行 | {sec.trade_date.min()} ~ {sec.trade_date.max()}")
    print(f"[行业] {sec.industry_name.iloc[0]} ({sec.industry_code.iloc[0]})\n")

    # ---------------- Gate 1: 主检验矩阵 ----------------
    print("=" * 96)
    print("【Gate 1】时序 IC 与显著性  ——  IC>0 = PE陷阱成立(低PE→差)  IC<0 = 传统价值成立")
    print("=" * 96)
    print(f"{'因子':<6}{'窗口':>6}{'H':>6}{'n_obs':>8}{'n_eff':>7}"
          f"{'IC':>9}{'t_朴素':>9}{'t_NW(H)':>10}{'t_NW(auto)':>12}"
          f"{'IC_非重叠':>11}{'t_非重叠':>10}  判定")
    print("-" * 104)
    rows = []
    for factor in ("pe", "pb"):
        for window in WINDOWS:
            for H in HORIZONS:
                r = run_horizon(sec, factor, H, window)
                if r is None:
                    continue
                rows.append(r)
                # 判定：主口径 NW，且非重叠同号
                sig = abs(r["t_nw"]) >= 2.0
                consis = (np.sign(r["ic"]) == np.sign(r["ic_sub"])) if \
                    np.isfinite(r["ic"]) and np.isfinite(r["ic_sub"]) else False
                if sig and consis:
                    verdict = "✅ 显著且稳健"
                elif sig:
                    verdict = "⚠️ NW显著但不稳健"
                elif abs(r["t_nw"]) >= 1.5:
                    verdict = "🟡 边缘"
                else:
                    verdict = "❌ 不显著"
                print(f"{factor:<6}{window:>6}{H:>6}{r['n_obs']:>8}{r['n_eff']:>7}"
                      f"{r['ic']:>9.3f}{r['t_naive']:>9.2f}{r['t_nw']:>10.2f}"
                      f"{r['t_nw_auto']:>12.2f}"
                      f"{r['ic_sub']:>11.3f}{r['t_sub']:>10.2f}  {verdict}")
    print("-" * 104)
    print("注：t_朴素 已知虚高（样本重叠），仅作对照；判定以 t_NW 为准，且要求非重叠子样本同号。")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT_DIR, "ic_report.csv"),
               index=False, encoding="utf-8-sig")

    # ---------------- 分层明细 ----------------
    print("\n" + "=" * 96)
    print("【分层】按分位分 5 档的平均未来超额收益（Q0=估值最低 / Q4=估值最高）")
    print("=" * 96)
    qrows = []
    for factor in ("pe", "pb"):
        for H in (60, 120, 250):
            g = run_quantile(sec, factor, H, 750)
            if g is None:
                continue
            print(f"\n--- {factor.upper()} 分位(750日窗口) → 未来 {H} 日超额 ---")
            t = g.copy()
            t["mean"] = (t["mean"] * 100).round(2)
            t["median"] = (t["median"] * 100).round(2)
            t["std"] = (t["std"] * 100).round(2)
            t["胜率%"] = t["胜率%"].round(1)
            t.index = [f"Q{i}" for i in t.index]
            print(t.to_string())
            qrows.append((factor, H, g))

    # ---------------- 有效样本警示 ----------------
    print("\n" + "=" * 96)
    print("【样本量硬约束】")
    print("=" * 96)
    eff = res[["H", "n_eff"]].drop_duplicates().sort_values("H")
    for _, r in eff.iterrows():
        print(f"  H={int(r.H):>3}日 → 非重叠有效样本仅 {int(r.n_eff)} 个"
              f"{'   ⚠️ <30，统计检验力不足' if r.n_eff < 30 else ''}")
    print("\n  结论：这是低频择时因子，独立样本极少。任何 |t|<3 的结果都不足以支撑入库决策。")


if __name__ == "__main__":
    main()
