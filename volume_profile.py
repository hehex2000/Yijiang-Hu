"""
成交量分布 / 价值区（Volume Profile / Point of Control）因子模块
================================================================
灵感来源：视频《技术分析学得越多为什么反而越亏》中"成交密集→停下(价值区)，
成交稀薄→快速通过(趋势)"的市场微结构观察。我们用日线 OHLCV 近似 volume profile
（没有逐笔/分钟数据，就把每个交易日的成交量均摊到它的 [low, high] 价格区间），
从而得到：
  - POC        : 成交量最集中的价格（市场最认可的"价值"）
  - 价值区上下沿: 从 POC 向两侧扩展、累计覆盖 va_pct(默认70%) 成交量的价格区间
  - dist_to_poc: 当前价相对 POC 的偏离（动量=正越强，反转=负越强）

本模块与平台其他策略共用本地 SQLite(astock_daily.db)取数，零 tushare 在线依赖。

典型用法
--------
from volume_profile import value_area_pass, fakeout_reclaim
ok, why = value_area_pass(ts_code, td, lookback=20, va_pct=0.70, mode="momentum")
hit, why = fakeout_reclaim(ts_code, td, lookback=20)
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from config import DATA
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DB_PATH = DATA.get("local_db_path", "")
    if not DB_PATH or not os.path.exists(DB_PATH):
        DB_PATH = os.path.join(_BASE_DIR, "data", "tu-sharedata", "astock_daily.db")
except Exception:
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DB_PATH = os.path.join(_BASE_DIR, "data", "tu-sharedata", "astock_daily.db")


def _get_conn():
    return __import__("sqlite3").connect(DB_PATH)


def get_window_bars(ts_code, trade_date, lookback=20):
    """取 lookback 个交易日的 OHLCV（按日期升序），不足返回 None。

    index_daily（指数）无 vol 字段，价值区对指数意义不大，直接返回 None。
    """
    if str(ts_code).endswith((".SH", ".SZ")) and str(ts_code).startswith(("000", "399", "932")):
        # 宽基指数：日线表为 index_daily，无成交量分布 → 不计算
        pass
    conn = _get_conn()
    try:
        df = pd.read_sql_query(
            """
            SELECT trade_date, high, low, close, vol
            FROM daily
            WHERE ts_code = ? AND trade_date <= ?
            ORDER BY trade_date DESC
            LIMIT ?
            """,
            conn,
            params=(ts_code, trade_date, int(lookback)),
        )
    finally:
        conn.close()
    if len(df) < max(5, lookback // 2):
        return None
    df = df.iloc[::-1].reset_index(drop=True)
    return df


def build_volume_profile_from_bars(df, va_pct=0.70, bins=None):
    """从 OHLCV DataFrame 计算 volume profile。

    返回 dict: poc / va_high / va_low / va_pct / last_close / in_value_area / dist_to_poc
    或 None（数据不足 / 无成交量）。
    """
    if df is None or len(df) < 5:
        return None
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values
    vols = df["vol"].astype(float).values
    closes = df["close"].astype(float).values

    all_low = float(lows.min())
    all_high = float(highs.max())
    if all_high <= all_low:
        return None

    if bins is None:
        bins = max(12, len(df) + 4)
    edges = np.linspace(all_low, all_high, bins + 1)
    mids = (edges[:-1] + edges[1:]) / 2.0
    vol_sum = np.zeros(bins, dtype=float)

    for h, l, v in zip(highs, lows, vols):
        if v <= 0 or h <= l:
            continue
        # 把当日成交量均摊到 [l, h] 覆盖到的 bin
        mask = (edges[:-1] >= l) & (edges[1:] <= h)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            idx = np.where((mids >= l) & (mids <= h))[0]
            if len(idx) == 0:
                continue
        vol_sum[idx] += v / len(idx)

    total = float(vol_sum.sum())
    if total <= 0:
        return None

    poc_bin = int(np.argmax(vol_sum))
    poc = float(mids[poc_bin])

    # 从 POC 向两侧扩展，直到累计成交量覆盖 va_pct
    cum = vol_sum[poc_bin]
    lo, hi = poc_bin, poc_bin
    while cum < va_pct * total and (lo > 0 or hi < bins - 1):
        left_v = vol_sum[lo - 1] if lo > 0 else -1.0
        right_v = vol_sum[hi + 1] if hi < bins - 1 else -1.0
        if right_v >= left_v and hi < bins - 1:
            hi += 1
            cum += vol_sum[hi]
        elif lo > 0:
            lo -= 1
            cum += vol_sum[lo]
        else:
            break

    va_low = float(edges[lo])
    va_high = float(edges[hi + 1])
    last_close = float(closes[-1])
    in_va = (va_low <= last_close <= va_high)
    dist = (last_close - poc) / poc if poc > 0 else 0.0
    return {
        "poc": poc,
        "va_high": va_high,
        "va_low": va_low,
        "va_pct": va_pct,
        "last_close": last_close,
        "in_value_area": in_va,
        "dist_to_poc": dist,
    }


def build_volume_profile(ts_code, trade_date, lookback=20, va_pct=0.70, bins=None):
    df = get_window_bars(ts_code, trade_date, lookback)
    return build_volume_profile_from_bars(df, va_pct=va_pct, bins=bins)


def value_area_pass(ts_code, trade_date, lookback=20, va_pct=0.70, mode="momentum"):
    """可选的价值区过滤。

    mode="momentum": 价格处于价值区内或上方（>= va_low）才通过——
        即"市场接受当前/更高价格"，不接处于价值区下方(被拒绝/扫止损)的弱势票。
    mode="reversal" : 价格处于价值区下沿或下方（<= va_low）才通过——
        即"已扫到价值区下方"的潜在超跌反弹区。
    返回 (bool 通过, str 说明)。
    """
    prof = build_volume_profile(ts_code, trade_date, lookback=lookback, va_pct=va_pct)
    if prof is None:
        return True, "无价值区数据(放行)"
    price = prof["last_close"]
    if mode == "momentum":
        ok = price >= prof["va_low"]
        why = (f"价{price:.2f} {'≥' if ok else '<'}价值区下沿{prof['va_low']:.2f}"
               f"(POC{prof['poc']:.2f})→{'通过' if ok else '剔除'}")
        return ok, why
    else:  # reversal
        ok = price <= prof["va_low"]
        why = (f"价{price:.2f} {'≤' if ok else '>'}价值区下沿{prof['va_low']:.2f}"
               f"(POC{prof['poc']:.2f})→{'通过(超跌区)' if ok else '剔除'}")
        return ok, why


def fakeout_reclaim(ts_code, trade_date, lookback=20):
    """反转 fakeout-reclaim 检测（视频核心微结构之一）。

    逻辑：在 lookback 窗口内，最近一个交易日创出窗口新低（扫止损/诱空），
    但当日收盘价又收回该新低之上（市场不接受更低价 → 快速反弹）。
    返回 (bool 命中, str 说明)。
    """
    df = get_window_bars(ts_code, trade_date, lookback)
    if df is None or len(df) < 5:
        return False, "数据不足"
    lows = df["low"].astype(float).values
    closes = df["close"].astype(float).values
    window_low = float(lows.min())
    last_low = float(lows[-1])
    last_close = float(closes[-1])
    # 最近一日创窗口新低（扫止损），且收盘收回新低之上（reclaim）
    swept = abs(last_low - window_low) < 1e-9 and last_low < float(lows[:-1].min()) + 1e-9
    reclaim = last_close > window_low
    if swept and reclaim:
        return True, f"新低{last_low:.2f}扫止损→收盘{last_close:.2f}收回→fakeout-reclaim"
    return False, f"无fakeout(低{last_low:.2f}/窗低{window_low:.2f},收{last_close:.2f})"
