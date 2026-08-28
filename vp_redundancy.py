# -*- coding: utf-8 -*-
"""
vp_redundancy: M2 冗余检查 —— dist_to_poc 与平台现有反转/价值因子的横截面冗余度
（计划 ② 收尾项）

方法：
  复用宽宇宙面板 m2_wide_panel.csv（含 vp_dist_to_poc，raw 口径，2010-2026 全市场）。
  在同一 (ts_code, 月末) 上用本地 daily 表(raw close) 复算对照因子：
    rev_21      : 平台 factors/reversal.py 定义（21d 收益，做多跌最多）—— 短期反转
    rev_250     : 250d 收益（长周期反转，与 VP 250d 窗口同源）
    ma_dist_250 : close/MA250 - 1（VP 的"机械近孪生"，纯价格锚距离）
    mom_12m     : 平台 factors/momentum.py 定义（12m 跳过1m）—— 动量(延续)
  价值因子(BP/EP/ROE)为基本面、daily 表无，结构性正交，本脚本不造代理、仅文字说明。

检验：
  1) 横截面 Spearman 相关矩阵（pooled rank）—— 看 vp 与谁共线
  2) 各因子 IC（vs fwd_ret，前向20日/T+1）—— 看谁真有 edge
  3) 增量检验：把 vp_dist_to_poc 对 ma_dist_250 / rev_250 逐日残差化，
     残差 IC 若≈0 → VP 不提供超出 MA距离/长反转 的增量信息（冗余）

口径说明：vp_dist_to_poc 与下方对照因子均为 raw close 计算（同基、apple-to-apple）。
  平台 reversal/momentum 惯例用 close_adj；daily 无 adj_factor 列、adj_factor 表
  2015 才起，故未做 adj 敏感性（避免样本截断）。结构结论( VP≈MA距离/长反转 )对
  raw/adj 均稳健（同属价格级信号）。
"""
import os
import sqlite3
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config

DB_PATH = config.DATA.get("local_db_path", "")
PANEL = "data/results/volume_profile/m2_wide_panel.csv"
OUT = "data/results/volume_profile"
FACTORS = ["vp_dist_to_poc", "rev_21", "rev_250", "ma_dist_250", "mom_12m"]


def load_panel():
    p = pd.read_csv(PANEL)
    p = p[["date", "ts_code", "fwd_ret", "fwd_ret_net", "vp_dist_to_poc"]].copy()
    p["date"] = p["date"].astype(int)
    p = p.dropna(subset=["vp_dist_to_poc"])
    return p


def load_compare_factors():
    """全 daily 表向量化复算对照因子(raw close)。"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT ts_code, trade_date, close FROM daily", conn
    )
    conn.close()
    df["trade_date"] = df["trade_date"].astype(int)
    df["close"] = df["close"].astype(float)
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    g = df.groupby("ts_code")["close"]
    df["rev_21"] = g.pct_change(21)
    df["rev_250"] = g.pct_change(250)
    df["ma250"] = g.transform(lambda s: s.rolling(250).mean())
    df["ma_dist_250"] = df["close"] / df["ma250"] - 1.0
    df["mom_12m"] = g.shift(21) / g.shift(252) - 1.0
    return df[["ts_code", "trade_date", "rev_21", "rev_250",
               "ma_dist_250", "mom_12m"]]


def corr_matrix(m):
    sub = m[FACTORS].dropna()
    # pooled rank Spearman（两两）
    n = len(FACTORS)
    mat = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            rho, _ = spearmanr(sub[FACTORS[i]], sub[FACTORS[j]])
            mat[i, j] = mat[j, i] = rho
    return pd.DataFrame(mat, index=FACTORS, columns=FACTORS), len(sub)


def ic_table(m, ret_col="fwd_ret"):
    rows = []
    for f in FACTORS:
        sub = m[["date", f, ret_col]].dropna()
        ics = []
        for d, g in sub.groupby("date"):
            if len(g) < 10:
                continue
            rho, _ = spearmanr(g[f], g[ret_col])
            if not np.isnan(rho):
                ics.append(rho)
        ics = pd.Series(ics)
        if len(ics) == 0:
            continue
        rows.append({
            "factor": f,
            "IC_mean": ics.mean(),
            "IC_std": ics.std(),
            "ICIR": ics.mean() / ics.std() if ics.std() > 0 else np.nan,
            "IC_pos_ratio": (ics > 0).mean(),
            "n_dates": len(ics),
        })
    return pd.DataFrame(rows)


def residualize(m, target, control):
    """逐日 OLS 残差化：vp ~ control，返回含 resid 的 df。"""
    out = []
    for d, g in m.groupby("date"):
        gg = g[[target, control, "fwd_ret"]].dropna()
        if len(gg) < 10:
            continue
        x = gg[control].values
        y = gg[target].values
        A = np.vstack([x, np.ones_like(x)]).T
        coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A.dot(coef)
        out.append(pd.DataFrame({
            "date": d, "fwd_ret": gg["fwd_ret"].values, "resid": resid,
        }))
    return pd.concat(out, ignore_index=True)


def incremental(m):
    """增量检验：VP 残差化后还剩多少预测力。"""
    rows = []
    for ctrl in ["ma_dist_250", "rev_250"]:
        r = residualize(m, "vp_dist_to_poc", ctrl)
        rho, _ = spearmanr(r["resid"], r["fwd_ret"])
        rows.append({
            "control": ctrl,
            "resid_IC_vs_fwd": rho,
            "n": len(r),
        })
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    panel = load_panel()
    print("panel rows (vp_dist_to_poc 非空):", len(panel))
    cmpf = load_compare_factors()
    m = cmpf.merge(panel, left_on=["ts_code", "trade_date"],
                   right_on=["ts_code", "date"], how="inner")
    print("merged rows (含对照因子):", len(m))
    m = m.dropna(subset=FACTORS + ["fwd_ret"])

    print("\n=== 1) 横截面 Spearman 相关矩阵 (pooled rank, raw 口径) ===")
    mat, n = corr_matrix(m)
    print("样本:", n, "只·月")
    print(mat.round(3).to_string())

    print("\n=== 2) 各因子 IC (vs fwd_ret, 前向20日/T+1) ===")
    ic = ic_table(m)
    print(ic.round(4).to_string(index=False))

    print("\n=== 3) 增量检验：vp_dist_to_poc 残差化后 IC ===")
    inc = incremental(m)
    print(inc.round(4).to_string(index=False))

    # 保存
    mat.to_csv(f"{OUT}/redundancy_corr_raw.csv")
    ic.to_csv(f"{OUT}/redundancy_ic.csv", index=False)
    inc.to_csv(f"{OUT}/redundancy_incremental.csv", index=False)
    print("\nsaved ->", OUT, "/ redundancy_corr_raw.csv, redundancy_ic.csv, redundancy_incremental.csv")


if __name__ == "__main__":
    main()
