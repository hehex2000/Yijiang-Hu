# -*- coding: utf-8 -*-
"""
vp_core: Volume Profile 核心算法（计划 M1，对标 Kara BV1tgVs6SE4C）
把 .openclaw spike_volume_profile.py 的算法拿过来，数据源改为本地库(vp_data)，
去掉 akshare / matplotlib 依赖，保留纯算法 + 无未来函数滚动回测。

算法四步（Kara 核心）：分箱(range-weighted) → 加权 → scipy 平滑 → find_peaks 检测。
设计哲学：不预测方向，只识别"哪里成交最密集"。
"""
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

import vp_data


def volume_profile(df, n_bins=80, smooth_sigma=2.0):
    """每根K线覆盖 [low, high] 区间，当日成交量按覆盖箱数均摊投票；
    箱内累计成交量 = 该价位"人气"。返回 (centers, raw, smoothed) 或 None。
    """
    if df is None or len(df) < 5:
        return None
    lo, hi = float(df["low"].min()), float(df["high"].max())
    if hi <= lo:
        return None
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    profile = np.zeros(n_bins, dtype=float)
    lows = df["low"].values.astype(float)
    highs = df["high"].values.astype(float)
    vols = df["vol"].values.astype(float)
    for l, h, v in zip(lows, highs, vols):
        if v <= 0 or h <= l:
            continue
        mask = (centers >= l) & (centers <= h)
        k = int(mask.sum())
        if k > 0:
            profile[mask] += v / k
    smoothed = gaussian_filter1d(profile, sigma=smooth_sigma)
    return centers, profile, smoothed


def detect_zones(centers, smoothed, min_prom_ratio=0.15, distance=3):
    """峰值 = 密集区(支撑/压力候选)；全局最高 = POC(控制点)。
    返回 (zones[(price, strength)], poc)。strength 已归一化到 smoothed 峰值=1.0。
    """
    if smoothed is None or len(smoothed) == 0:
        return [], 0.0
    prom = float(smoothed.max()) * min_prom_ratio
    peaks, _ = find_peaks(smoothed, prominence=prom, distance=distance)
    if len(peaks) == 0:
        return [], float(centers[np.argmax(smoothed)])
    zones = sorted(
        [(float(centers[p]), float(smoothed[p]) / smoothed.max()) for p in peaks],
        key=lambda x: -x[1],
    )
    poc = float(centers[int(np.argmax(smoothed))])
    return zones, poc


def rolling_backtest(df, window=120, touch_pct=0.02, hold=10, n_bins=80):
    """无未来函数滚动回测（三道防线：只用[t-window,t)历史 / 次日开盘成交 / T+1 持有）。
    t 日收盘在近支撑(touch_pct 内) → 次日开盘买入，持 hold 个交易日收盘卖出。
    返回 trades DataFrame[date, support, entry, exit, ret] 或 None。
    """
    if df is None or len(df) < window + hold + 2:
        return None
    closes = df["close"].values.astype(float)
    opens = df["open"].values.astype(float)
    dates = df["date"].values
    trades = []
    for t in range(window, len(df) - hold - 1):
        hist = df.iloc[t - window : t]
        res = volume_profile(hist, n_bins=n_bins)
        if res is None:
            continue
        centers, _, sm = res
        zones, _ = detect_zones(centers, sm)
        px = closes[t]
        supports = [p for p, _ in zones if p < px]
        if not supports:
            continue
        nearest = max(supports)  # 距现价最近的支撑
        if abs(px - nearest) / nearest <= touch_pct:
            entry = opens[t + 1]  # 次日开盘成交（T+1）
            exit_ = closes[t + 1 + hold]
            trades.append(
                (int(dates[t]), float(nearest), float(entry), float(exit_),
                 (exit_ - entry) / entry)
            )
    if not trades:
        return None
    return pd.DataFrame(trades, columns=["date", "support", "entry", "exit", "ret"])
