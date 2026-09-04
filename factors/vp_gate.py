# -*- coding: utf-8 -*-
"""VP 确认闸门（overlay gate）：用 dist_to_poc 对反转类信号做同向确认。

定位（务必先读，别误用）
----------------------
这不是 alpha 因子，是一个**廉价的成本/质量滤镜**。证据来自 vp_overlay.py
（宽宇宙 664443 只·月，组合层净成本 0.60%/月）：

    基线 rev_21        年化净 10.5%   t=4.26   胜率 60.9%   覆盖 100%
    + VP 同向确认      年化净 13.1%   t=4.27   胜率 60.4%   覆盖  82%

+2.6pp/年，但 **t 值几乎不动** —— 增益主要来自"少下 18% 的仓"省下的摩擦，
不是新增 alpha。且 dist_to_poc 本身约 64% 冗余于 close/MA250−1
（残差 IC 仅 -0.022），按样本外零增量纪律属"边界、弱增量"。

用法
----
在策略**按反转类因子排序取到候选集之后**调用，绝不要对全市场算：

    from factors import vp_gate
    kept = vp_gate.apply_gate(cands, trade_date, rev_col="reversal")

之所以要求"先排序再闸门"，是因为闸门只在候选集（如 top 20%）上算 VP，
比全市场算便宜约 5 倍；而在候选集上做确认与全市场筛完再排序，
对最终持仓的影响等价（闸门只做剔除，不改变候选间的相对顺序）。

口径
----
dist_to_poc = (收盘 - POC) / POC，与 vp_factor.factors_for_window 一致。
确认规则：rev < 0（过去跌，均值回复看多）与 dist < 0（价在 POC 下方，看多）
同向即确认，即 (dist < 0) == (rev < 0)。
"""

import argparse
import os
import sys
import time

# 直接 `python factors/vp_gate.py` 时 sys.path[0] 是 factors/，
# 需把项目根目录加进来才能 import vp_core / vp_data。
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd

import vp_core
import vp_data

WINDOW = 250          # VP 回看交易日
N_BINS = 80           # 分箱数（与 vp_core.volume_profile 默认一致）
SMOOTH_SIGMA = 2.0


def dist_to_poc(ts_code, trade_date, window=WINDOW, n_bins=N_BINS):
    """单只票在 trade_date 的 dist_to_poc；数据不足返回 None。"""
    df = vp_data.get_window(ts_code, trade_date, lookback=window)
    if df is None or len(df) < window or len(df) == 0:
        return None
    last_close = float(df["close"].iloc[-1])
    if last_close <= 0:
        return None
    res = vp_core.volume_profile(df, n_bins=n_bins, smooth_sigma=SMOOTH_SIGMA)
    if res is None:
        return None
    centers, _, sm = res
    _, poc = vp_core.detect_zones(centers, sm)
    if poc is None or poc <= 0:
        return None
    return (last_close - poc) / poc


def dist_to_poc_batch(ts_codes, trade_date, window=WINDOW, n_bins=N_BINS):
    """对候选集批量算 dist_to_poc，返回 {ts_code: float}；算不出的不进字典。"""
    out = {}
    for ts in ts_codes:
        try:
            v = dist_to_poc(ts, trade_date, window=window, n_bins=n_bins)
        except Exception:
            v = None
        if v is not None and np.isfinite(v):
            out[ts] = v
    return out


def confirmed(rev, dist):
    """同向确认：rev<0（过去跌）与 dist<0（价在 POC 下方）同向才算确认。"""
    if rev is None or dist is None:
        return False
    try:
        rev = float(rev)
        dist = float(dist)
    except (TypeError, ValueError):
        return False
    if not (np.isfinite(rev) and np.isfinite(dist)):
        return False
    return (dist < 0) == (rev < 0)


def apply_gate(cands, trade_date, rev_col="reversal", window=WINDOW,
               n_bins=N_BINS, topn=None, keep_rank=True):
    """对候选集做 VP 同向确认过滤。

    参数
    ----
    cands : DataFrame，须含 ts_code 与 rev_col 列，且已按策略逻辑排好序
    trade_date : 信号日（形如 20260703 或 '20260703'）
    topn : 不为 None 时，确认后按原顺序取前 topn 只
    keep_rank : True 时结果沿用 cands 的原顺序（只剔除、不重排）

    返回
    ----
    (kept_df, dist_map)
        kept_df : 通过确认的候选（保留原列）
        dist_map: {ts_code: dist_to_poc}，含全部能算出的候选（含被剔除的）
    """
    if cands is None or len(cands) == 0:
        return cands, {}
    codes = list(cands["ts_code"])
    dist_map = dist_to_poc_batch(codes, trade_date, window=window, n_bins=n_bins)
    rev_map = dict(zip(cands["ts_code"], cands[rev_col])) if rev_col in cands else {}

    def ok(ts):
        return confirmed(rev_map.get(ts), dist_map.get(ts))

    order = {ts: i for i, ts in enumerate(codes)}
    kept_codes = [ts for ts in codes if ok(ts)]
    if not keep_rank:
        kept_codes.sort(key=lambda ts: dist_map[ts])
    if topn is not None:
        kept_codes = kept_codes[:topn]
        if keep_rank:
            kept_codes.sort(key=lambda ts: order[ts])
    kept = cands[cands["ts_code"].isin(set(kept_codes))].copy()
    if keep_rank:
        kept["_rank"] = kept["ts_code"].map(order)
        kept = kept.sort_values("_rank").drop(columns=["_rank"])
    return kept.reset_index(drop=True), dist_map


# ------------------------- 自检 -------------------------

def _selftest(trade_date, top_pct=0.20, topn=15):
    """漏斗自检：HS300 池 -> 按 21 日反转排序 -> 取前 top_pct -> VP 闸门。"""
    import sqlite3
    conn = sqlite3.connect(vp_data.DB_PATH)
    snap = conn.execute(
        "SELECT MAX(trade_date) FROM index_constituent "
        "WHERE index_code='000300.SH' AND trade_date<=?", (trade_date,)).fetchone()[0]
    codes = [r[0] for r in conn.execute(
        "SELECT DISTINCT ts_code FROM index_constituent "
        "WHERE index_code='000300.SH' AND trade_date=?", (snap,))]
    conn.close()
    print("信号日 %s   沪深300 快照 %s  (%d 只)" % (trade_date, snap, len(codes)))

    t0 = time.time()
    rows = []
    for ts in codes:
        df = vp_data.get_window(ts, trade_date, lookback=250)
        if df is None or len(df) < 250:
            continue
        rev = df["close"].pct_change(21).iloc[-1]
        if rev is None or not np.isfinite(rev):
            continue
        rows.append({"ts_code": ts, "reversal": float(rev)})
    uni = pd.DataFrame(rows)
    t_uni = time.time() - t0
    print("取反转值: %d 只  耗时 %.1fs" % (len(uni), t_uni))

    # 做多"跌最多" -> reversal 升序
    uni = uni.sort_values("reversal").reset_index(drop=True)
    n_cand = max(1, int(len(uni) * top_pct))
    cands = uni.head(n_cand).copy()
    print("候选集(top %.0f%%): %d 只" % (top_pct * 100, n_cand))

    rev_map = dict(zip(cands["ts_code"], cands["reversal"]))
    t0 = time.time()
    kept, dmap = apply_gate(cands, trade_date, rev_col="reversal", topn=topn)
    t_gate = time.time() - t0
    n_ok = sum(1 for c in cands["ts_code"] if confirmed(rev_map.get(c), dmap.get(c)))
    print("VP 闸门: 确认 %d / %d  (%.0f%%)  取前 %d  耗时 %.2fs" % (
        n_ok, n_cand, 100.0 * n_ok / n_cand, len(kept), t_gate))
    print("闸门单只耗时: %.1f ms  (候选集 %d 只)" % (t_gate / n_cand * 1000, n_cand))
    print("\n确认后前 %d 只:" % len(kept))
    for _, r in kept.iterrows():
        print("  %s  rev21=%6.2f%%  dist_to_poc=%7.4f" % (
            r["ts_code"], r["reversal"] * 100, dmap[r["ts_code"]]))


def main():
    ap = argparse.ArgumentParser(description="VP 确认闸门自检")
    ap.add_argument("--date", default="20260703", help="信号日 YYYYMMDD")
    ap.add_argument("--top-pct", type=float, default=0.20, help="候选集比例")
    ap.add_argument("--topn", type=int, default=15, help="最终取前 N 只")
    a = ap.parse_args()
    _selftest(a.date, top_pct=a.top_pct, topn=a.topn)


if __name__ == "__main__":
    main()
