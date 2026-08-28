# -*- coding: utf-8 -*-
"""
vp_factor: Volume Profile 单因子化 + zoo 式 IC/分层验证（计划 M2）
把 vp_core 的输出接成横截面因子，用本地库 daily 表直接算：
  - vp_dist_to_poc      : (收盘 - POC)/POC，正=价在价值区上方
  - vp_support_dist_pct : 距最近下方支撑的百分比（正=价在支撑上方）
  - vp_va_pass          : 价>=价值区下沿(动量模式)记 1，否则 0
无前视：因子窗口只用 <=t 数据；前向收益 open[t+1] -> close[t+20] (T+1 进场)。
"""
import os
import sqlite3
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config
import vp_core

DB_PATH = config.DATA.get("local_db_path", "")
FACTORS = ["vp_dist_to_poc", "vp_support_dist_pct", "vp_va_pass"]


def value_area_band(profile, centers, va_pct=0.70):
    """从 profile 峰值(POC)向两侧扩展，覆盖 va_pct 成交量 -> 价值区上下沿。"""
    total = float(profile.sum())
    if total <= 0:
        return float(centers[0]), float(centers[-1])
    poc_bin = int(np.argmax(profile))
    cum = profile[poc_bin]
    lo = hi = poc_bin
    n = len(profile)
    while cum < va_pct * total and (lo > 0 or hi < n - 1):
        lv = profile[lo - 1] if lo > 0 else -1.0
        rv = profile[hi + 1] if hi < n - 1 else -1.0
        if rv >= lv and hi < n - 1:
            hi += 1
            cum += profile[hi]
        elif lo > 0:
            lo -= 1
            cum += profile[lo]
        else:
            break
    return float(centers[lo]), float(centers[hi])


def factors_for_window(df_window, last_close):
    """给定截至信号日的窗口(升序)，返回因子 dict；无足够数据返回 None。"""
    res = vp_core.volume_profile(df_window)
    if res is None:
        return None
    centers, raw, sm = res
    zones, poc = vp_core.detect_zones(centers, sm)
    if poc <= 0:
        return None
    dist_to_poc = (last_close - poc) / poc
    supports = [p for p, _ in zones if p < last_close]
    if supports:
        nearest = max(supports)
        support_dist_pct = (last_close - nearest) / last_close * 100.0
    else:
        support_dist_pct = np.nan
    va_low, _ = value_area_band(raw, centers)
    va_pass = 1.0 if last_close >= va_low else 0.0
    return {
        "vp_dist_to_poc": dist_to_poc,
        "vp_support_dist_pct": support_dist_pct,
        "vp_va_pass": va_pass,
    }


def load_universe(n=300, min_rows=1000):
    """最长历史的 n 只票（注：含幸存者偏差，仅 M2 可行性用，非样本外结论）。"""
    conn = sqlite3.connect(DB_PATH)
    codes = pd.read_sql_query(
        "SELECT ts_code, COUNT(*) c FROM daily GROUP BY ts_code "
        "HAVING c>=? ORDER BY c DESC LIMIT ?",
        conn, params=(min_rows, n),
    )
    conn.close()
    return codes["ts_code"].tolist()


def month_end_dates(sorted_dates):
    s = pd.Series(sorted_dates)
    ym = s.astype(str).str[:6]
    return s.groupby(ym).max().tolist()


def build_panel(universe, window=120, fwd=20, rebal=None):
    conn = sqlite3.connect(DB_PATH)
    q = ",".join("?" * len(universe))
    df = pd.read_sql_query(
        "SELECT ts_code, trade_date, open, high, low, close, vol "
        "FROM daily WHERE ts_code IN (%s)" % q,
        conn, params=list(universe),
    )
    conn.close()
    df["trade_date"] = df["trade_date"].astype(int)
    for c in ["open", "high", "low", "close", "vol"]:
        df[c] = df[c].astype(float)
    if rebal is None:
        rebal = month_end_dates(sorted(df["trade_date"].unique()))
    by_stock = {c: g.sort_values("trade_date").reset_index(drop=True)
                for c, g in df.groupby("ts_code")}
    rows = []
    for code, g in by_stock.items():
        dates = g["trade_date"].values
        closes = g["close"].values
        opens = g["open"].values
        for d in rebal:
            idx = int(np.searchsorted(dates, d))
            if idx >= len(dates) or int(dates[idx]) != d:
                continue
            if idx < window:
                continue
            if idx + fwd >= len(closes):
                continue
            w = g.iloc[idx - window : idx + 1]
            last_close = closes[idx]
            fac = factors_for_window(w, last_close)
            if fac is None:
                continue
            entry = opens[idx + 1]
            exitr = closes[idx + fwd]
            fwd_ret = exitr / entry - 1.0
            r = {"date": int(d), "ts_code": code, "fwd_ret": fwd_ret}
            r.update(fac)
            rows.append(r)
    return pd.DataFrame(rows)


def ic_report(panel):
    out = []
    for f in FACTORS:
        sub = panel[["date", "ts_code", f, "fwd_ret"]].dropna()
        ics = []
        for d, g in sub.groupby("date"):
            if len(g) < 10:
                continue
            rho, _ = spearmanr(g[f], g["fwd_ret"])
            if not np.isnan(rho):
                ics.append(rho)
        ics = pd.Series(ics)
        if len(ics) == 0:
            continue
        out.append({
            "factor": f,
            "IC_mean": ics.mean(),
            "IC_std": ics.std(),
            "ICIR": ics.mean() / ics.std() if ics.std() > 0 else np.nan,
            "IC_pos_ratio": (ics > 0).mean(),
            "n_dates": len(ics),
        })
    return pd.DataFrame(out)


def quantile_ls(panel, factor, q=5):
    sub = panel[["date", "ts_code", factor, "fwd_ret"]].dropna()
    ic_all, _ = spearmanr(sub[factor], sub["fwd_ret"])
    longs, shorts = [], []
    for d, g in sub.groupby("date"):
        if len(g) < q * 3:
            continue
        g = g.sort_values(factor).reset_index(drop=True)
        k = int(len(g) / q)
        if ic_all >= 0:
            lq = g.iloc[-k:]["fwd_ret"].mean()
            sq = g.iloc[:k]["fwd_ret"].mean()
        else:
            lq = g.iloc[:k]["fwd_ret"].mean()
            sq = g.iloc[-k:]["fwd_ret"].mean()
        longs.append(lq)
        shorts.append(sq)
    ls = pd.Series(longs) - pd.Series(shorts)
    return {
        "factor": factor,
        "IC": ic_all,
        "long_mean": np.mean(longs),
        "short_mean": np.mean(shorts),
        "LS_mean": ls.mean(),
        "LS_t": ls.mean() / ls.std() * np.sqrt(len(ls)) if ls.std() > 0 else np.nan,
        "n_dates": len(ls),
    }


def main():
    uni = load_universe(300, 1000)
    print("universe size: %d (最长历史 top300, 含幸存者偏差)" % len(uni))
    panel = build_panel(uni, window=120, fwd=20)
    print("panel rows: %d | 日期数: %d" % (len(panel), panel["date"].nunique()))
    ic = ic_report(panel)
    print("\n=== IC 报告 (前向20日 / T+1进场) ===")
    print(ic.to_string(index=False))
    print("\n=== 分层多空(LS, 顶/底20%组, IC符号定方向) ===")
    for f in FACTORS:
        q = quantile_ls(panel, f)
        print("%-20s IC=%.3f  LS_mean=%+.3f%%  LS_t=%.2f  n=%d"
              % (f, q["IC"], q["LS_mean"] * 100, q["LS_t"], q["n_dates"]))
    os.makedirs("data/results/volume_profile", exist_ok=True)
    panel.to_csv("data/results/volume_profile/factor_panel.csv", index=False)
    ic.to_csv("data/results/volume_profile/ic_report.csv", index=False)
    print("\nsaved -> data/results/volume_profile/")


if __name__ == "__main__":
    main()
