# -*- coding: utf-8 -*-
"""
vp_factor_wide: M2 复验 —— 宽宇宙(point-in-time 全市场) + 净成本
同一套算法/同一代码路径，只换宇宙与是否扣成本：
  A) 最长历史 top300 + 毛收益   (原 M2 口径，同路径重跑作对照)
  B) 宽宇宙(每月全市场可用票) + 毛收益
  C) 宽宇宙(每月全市场可用票) + 净成本(round-trip, 卖出印花税按日期)
无前视：因子窗口只用 <=t；前向收益 open[t+1] -> close[t+20] (T+1 进场)。
性能：逐只查询(内存 O(1)) + numpy 内层(无 pandas 逐月切片)。
"""
import os
import sqlite3
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import spearmanr

import config
import vp_core

DB_PATH = config.DATA.get("local_db_path", "")
FACTORS = ["vp_dist_to_poc", "vp_support_dist_pct", "vp_va_pass"]
WINDOW = 120
FWD = 20
N_BINS = 80
SMOOTH = 2.0

COMMISSION = 0.00025
SLIP = 0.001
STAMP_CUT = 20230828
STAMP_PRE = 0.001
STAMP_POST = 0.0005


def rt_cost_components(entry_date):
    stamp = STAMP_PRE if entry_date < STAMP_CUT else STAMP_POST
    return COMMISSION + SLIP, COMMISSION + stamp + SLIP  # buy, sell


def month_end_dates():
    conn = sqlite3.connect(DB_PATH)
    rows = pd.read_sql_query(
        "SELECT max(trade_date) m FROM daily GROUP BY substr(trade_date,1,6)", conn
    )["m"].tolist()
    conn.close()
    return sorted(int(x) for x in rows)


def vp_from_arrays(lows, highs, vols, n_bins=N_BINS, smooth_sigma=SMOOTH):
    """向量化分箱（与 vp_core.volume_profile 同算法/同输出），吃 numpy 数组。"""
    if len(lows) < 5:
        return None
    lo = float(lows.min())
    hi = float(highs.max())
    if hi <= lo:
        return None
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    cov = (centers[None, :] >= lows[:, None]) & (centers[None, :] <= highs[:, None])
    k = cov.sum(axis=1).astype(float)
    valid = (k > 0) & (vols > 0)
    profile = np.zeros(n_bins, dtype=float)
    if valid.any():
        safe_k = np.where(valid, k, 1.0)
        contrib = np.where(cov & valid[:, None], (vols / safe_k)[:, None], 0.0)
        profile = contrib.sum(axis=0)
    smoothed = gaussian_filter1d(profile, sigma=smooth_sigma)
    return centers, profile, smoothed


def value_area_band(profile, centers, va_pct=0.70):
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


def load_stock_arrays(conn, code):
    df = pd.read_sql_query(
        "SELECT trade_date, open, high, low, close, vol FROM daily "
        "WHERE ts_code=? ORDER BY trade_date",
        conn, params=(code,),
    )
    if df.empty or len(df) < WINDOW + FWD + 2:
        return None
    return {
        "date": df["trade_date"].values.astype(int),
        "open": df["open"].values.astype(float),
        "close": df["close"].values.astype(float),
        "low": df["low"].values.astype(float),
        "high": df["high"].values.astype(float),
        "vol": df["vol"].values.astype(float),
    }


def build_panel_codes(codes, rebal, conn):
    rows = []
    n = len(codes)
    for si, code in enumerate(codes):
        if (si + 1) % 500 == 0:
            print("  ... %d/%d stocks" % (si + 1, n), flush=True)
        arr = load_stock_arrays(conn, code)
        if arr is None:
            continue
        dates = arr["date"]
        closes = arr["close"]
        opens = arr["open"]
        lows = arr["low"]
        highs = arr["high"]
        vols = arr["vol"]
        L = len(dates)
        for d in rebal:
            idx = int(np.searchsorted(dates, d))
            if idx >= L or int(dates[idx]) != d:
                continue
            if idx < WINDOW or idx + FWD >= L:
                continue
            res = vp_from_arrays(lows[idx - WINDOW:idx + 1],
                                 highs[idx - WINDOW:idx + 1],
                                 vols[idx - WINDOW:idx + 1])
            if res is None:
                continue
            centers, raw, sm = res
            zones, poc = vp_core.detect_zones(centers, sm)
            if poc <= 0:
                continue
            last = closes[idx]
            dist_to_poc = (last - poc) / poc
            supports = [p for p, _ in zones if p < last]
            sup_pct = ((last - max(supports)) / last * 100.0) if supports else np.nan
            va_low, _ = value_area_band(raw, centers)
            va_pass = 1.0 if last >= va_low else 0.0
            entry = opens[idx + 1]
            exitr = closes[idx + FWD]
            ed = int(dates[idx + 1])
            buy_c, sell_c = rt_cost_components(ed)
            gross = exitr / entry - 1.0
            net = exitr * (1 - sell_c) / (entry * (1 + buy_c)) - 1.0
            rows.append((int(d), code, gross, net, dist_to_poc, sup_pct, va_pass))
    return pd.DataFrame(
        rows,
        columns=["date", "ts_code", "fwd_ret", "fwd_ret_net",
                 "vp_dist_to_poc", "vp_support_dist_pct", "vp_va_pass"],
    )


def ic_report(panel, ret_col="fwd_ret"):
    out = []
    for f in FACTORS:
        sub = panel[["date", "ts_code", f, ret_col]].dropna()
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
        out.append({
            "factor": f, "IC_mean": ics.mean(), "IC_std": ics.std(),
            "ICIR": ics.mean() / ics.std() if ics.std() > 0 else np.nan,
            "IC_pos_ratio": (ics > 0).mean(), "n_dates": len(ics),
        })
    return pd.DataFrame(out)


def quantile_ls(panel, factor, ret_col="fwd_ret", q=5):
    sub = panel[["date", "ts_code", factor, ret_col]].dropna()
    ic_all, _ = spearmanr(sub[factor], sub[ret_col])
    longs, shorts = [], []
    for d, g in sub.groupby("date"):
        if len(g) < q * 3:
            continue
        g = g.sort_values(factor).reset_index(drop=True)
        k = int(len(g) / q)
        if ic_all >= 0:
            lq = g.iloc[-k:][ret_col].mean()
            sq = g.iloc[:k][ret_col].mean()
        else:
            lq = g.iloc[:k][ret_col].mean()
            sq = g.iloc[-k:][ret_col].mean()
        longs.append(lq)
        shorts.append(sq)
    ls = pd.Series(longs) - pd.Series(shorts)
    return {
        "factor": factor, "IC": ic_all,
        "LS_mean": ls.mean(),
        "LS_t": ls.mean() / ls.std() * np.sqrt(len(ls)) if ls.std() > 0 else np.nan,
        "n_dates": len(ls),
    }


def main():
    import vp_factor as vf
    rebal = month_end_dates()
    print("月份数(全局): %d" % len(rebal))

    conn = sqlite3.connect(DB_PATH)

    print("=== top300(原 M2 口径, 同路径重跑) ===")
    top300 = vf.load_universe(300, 1000)
    panel_top = build_panel_codes(top300, rebal, conn)
    print("top300 panel rows: %d | 股票: %d" % (len(panel_top), panel_top["ts_code"].nunique()))
    ic_t = ic_report(panel_top)

    print("=== 宽宇宙(全市场, point-in-time) ===")
    all_codes = pd.read_sql_query(
        "SELECT ts_code FROM daily GROUP BY ts_code", conn
    )["ts_code"].tolist()
    print("全市场股票数: %d" % len(all_codes))
    panel_wide = build_panel_codes(all_codes, rebal, conn)
    print("wide panel rows: %d | 月份: %d | 股票: %d"
          % (len(panel_wide), panel_wide["date"].nunique(), panel_wide["ts_code"].nunique()))
    ic_w = ic_report(panel_wide)

    conn.close()

    cmp_rows = []
    for f in FACTORS:
        qt = quantile_ls(panel_top, f, "fwd_ret")
        qw = quantile_ls(panel_wide, f, "fwd_ret")
        qn = quantile_ls(panel_wide, f, "fwd_ret_net")
        ic_t_f = ic_t.loc[ic_t.factor == f, "IC_mean"].iloc[0] if len(ic_t) else np.nan
        ic_w_f = ic_w.loc[ic_w.factor == f, "IC_mean"].iloc[0] if len(ic_w) else np.nan
        cmp_rows.append({
            "factor": f,
            "IC_top300": ic_t_f, "IC_wide": ic_w_f,
            "LS_top300_gross_%": qt["LS_mean"] * 100,
            "LS_wide_gross_%": qw["LS_mean"] * 100,
            "LS_wide_net_%": qn["LS_mean"] * 100,
            "LS_wide_net_t": qn["LS_t"],
            "n_wide": qw["n_dates"],
        })
    cmp = pd.DataFrame(cmp_rows)
    print("\n=== M2 复验对比 (毛=未扣成本, 净=round-trip) ===")
    print(cmp.to_string(index=False))

    os.makedirs("data/results/volume_profile", exist_ok=True)
    cmp.to_csv("data/results/volume_profile/m2_wide_recheck.csv", index=False)
    ic_w.to_csv("data/results/volume_profile/m2_wide_ic.csv", index=False)
    panel_wide.to_csv("data/results/volume_profile/m2_wide_panel.csv", index=False)
    print("\nsaved -> data/results/volume_profile/")


if __name__ == "__main__":
    main()
