"""周频调仓 + 沪深300 MACD 择时 统计实验（纯统计，非实盘策略）。

口径（与看板"✅机会"一致 + 指数级择时）：
- 调仓日 W = 每周最后一个交易日（ISO 周分组取各组最后一日）。
- 信号：W 日收盘后，用截至 W 的日线(window=120)算每只沪深300成分股 VP，
       判"机会"(支撑附近3%内 + 反转overlay双确认)，按离支撑由近到远取前15只。
- 指数择时：W 日沪深300日线 MACD(12/26/9) 多头(DIF>DEA) -> 下周建仓；空头 -> 空仓(持币, 周收益0)。
- 交易(轻微未来函数, 同隔夜版T-1收盘近似)：W 日收盘价买入，下一周 W' 收盘价卖出（持有约一周）。
- 对比基准：忽略 MACD，每周都满仓建仓（同样选股），看指数择时是否真增厚。

复用 vp_stat_overnight 的 get_calendar / get_hs300_pool / is_st 与常量。
VP 直接用 vp_core（scipy 实现），与 vp_scan / 看板同一口径。

历史更正：本文件一度改用「纯 python VP + 子进程隔离」以规避所谓
"scipy 在 Windows 连续调用会 segfault"。2026-08-31 压力实测证伪该说法
（单进程连续 30000 次 vp_core 调用无崩溃、无内存泄漏），故改回 scipy。
真正的慢是 sqlite 连接 close() 约 79ms/次（7.6GB 库），已由
_get_window_fast 的 SQL 端取窗口缓解，与 scipy 无关。
"""

import os

# 限制 BLAS/OMP 线程数，避免 numpy/BLAS 侧多线程抖动
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sqlite3
import time
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

import vp_core
from vp_stat_overnight import (
    get_calendar,
    get_hs300_pool,
    is_st,
    NEAR,
    RT_COST,
    DB,
)

INDEX_CODE = "000300.SH"


def get_index(ts_code, t, lookback=200):
    """取指数日线(截止t, 升序)，来自 index_daily 表。"""
    conn = sqlite3.connect(DB)
    try:
        cur = conn.execute(
            "SELECT trade_date, close FROM index_daily WHERE ts_code=? AND trade_date<=? "
            "ORDER BY trade_date DESC LIMIT ?",
            (ts_code, t, lookback),
        )
        rows = cur.fetchall()
    finally:
        conn.close()
    if not rows:
        return None
    d = pd.DataFrame(rows, columns=["trade_date", "close"])
    d["close"] = d["close"].astype(float)
    d = d.sort_values("trade_date").reset_index(drop=True)
    return d


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def macd_long(ts_code, t, fast=12, slow=26, sig=9):
    """W 日指数 MACD 是否多头(DIF>DEA)。数据不足返回 None。"""
    d = get_index(ts_code, t, lookback=slow + sig + 30)
    if d is None or len(d) < slow + sig:
        return None
    close = d["close"]
    dif = ema(close, fast) - ema(close, slow)
    dea = ema(dif, sig)
    return float(dif.iloc[-1]) > float(dea.iloc[-1])


def weekly_ends(cal):
    """每周最后一个交易日（按 ISO 周分组，时间序）。"""
    weeks = {}
    for d in cal:
        dt = datetime.strptime(str(d), "%Y%m%d")
        iso = dt.isocalendar()
        key = (iso[0], iso[1])  # 必须用 ISO 年份，用 dt.year 会把跨年周拆成两个桶
        weeks.setdefault(key, []).append(d)
    return [v[-1] for v in weeks.values()]


def _get_window_fast(ts_code, t1, lookback):
    """取 t1 及之前 lookback 个交易日（升序）。
    vp_data.get_window 会先把该票全部历史(~4000行)拉进 DataFrame 再 tail，
    300只×26周下极慢；这里在 SQL 端 DESC LIMIT 只取需要的行数。"""
    conn = sqlite3.connect(DB)
    try:
        cur = conn.execute(
            "SELECT trade_date, open, high, low, close, vol FROM daily "
            "WHERE ts_code=? AND trade_date<=? ORDER BY trade_date DESC LIMIT ?",
            (ts_code, t1, lookback),
        )
        rows = cur.fetchall()
    finally:
        conn.close()
    if not rows:
        return None
    rows.reverse()
    return pd.DataFrame(
        rows, columns=["date", "open", "high", "low", "close", "vol"]
    ).astype({"open": float, "high": float, "low": float, "close": float, "vol": float})


def judge_opportunity_w(ts_code, t1, window):
    """判"机会"（复刻看板✅：支撑附近3%内 + 反转overlay双确认）。
    VP 直接用 vp_core（scipy），与 vp_scan / 看板 / 隔夜版同一实现。"""
    d = _get_window_fast(ts_code, t1, window)
    if d is None or len(d) < window or len(d) <= 21:
        return None
    price = float(d["close"].iloc[-1])
    if price <= 0:
        return None
    # 分箱数与隔夜版保持同一口径：n_bins = max(20, window//3)
    #（看板 vp_scan 是 250日/80箱；这里 window=120 -> 40箱，与 overnight 实验可比）
    res = vp_core.volume_profile(d, n_bins=max(20, window // 3), smooth_sigma=2.0)
    if res is None:
        return None
    centers, _, sm = res
    zones, poc = vp_core.detect_zones(centers, sm)
    if not zones:
        return None
    supports = [p for p, _ in zones if p < price]
    if not supports:
        return None
    support = max(supports)
    sup_dist = (price - support) / price
    if sup_dist > NEAR:
        return None
    rev21 = float(d["close"].pct_change(21).iloc[-1])
    vp_long = price < poc
    rev_long = rev21 < 0
    if not (vp_long == rev_long):
        return None
    return dict(ts_code=ts_code, price=price, support=support,
                sup_dist=sup_dist, poc=poc, rev21=rev21)


def select_topn_w(t1, window, topn, st_filter, pool):
    """进程内串行选股（用 vp_core/scipy，不做子进程隔离——实测无必要）。"""
    conn = sqlite3.connect(DB)
    try:
        nm = pd.read_sql_query("SELECT ts_code, name FROM stock_basic", conn)
    finally:
        conn.close()
    namemap = dict(zip(nm["ts_code"], nm["name"]))
    # ST 过滤必须用「股票名称」判，不能拿 ts_code 判（ts_code 里永远没有 ST）
    codes = []
    for ts in pool:
        if st_filter and is_st(namemap.get(ts, "")):
            continue
        codes.append(ts)
    out = []
    for ts in codes:
        try:
            j = judge_opportunity_w(ts, t1, window)
            if j:
                out.append(j)
        except Exception:
            continue
    for j in out:
        j["name"] = namemap.get(j["ts_code"], "")
    out.sort(key=lambda x: x["sup_dist"])
    return out[:topn], len(out)


def bar_ret(ts_code, buy_d, sell_d):
    """buy_d 收盘买 -> sell_d 收盘卖 的收益率；取不到返回 None。"""
    b = _get_window_fast(ts_code, buy_d, lookback=5)
    s = _get_window_fast(ts_code, sell_d, lookback=5)
    if b is None or s is None or len(b) == 0 or len(s) == 0:
        return None
    bp = float(b["close"].iloc[-1])
    sp = float(s["close"].iloc[-1])
    if bp <= 0 or sp <= 0:
        return None
    return sp / bp - 1.0


def run_range(start_s, end_s, window, topn, st_filter):
    cal = get_calendar()
    cal_sub = [d for d in cal if start_s <= str(d) <= end_s]
    if len(cal_sub) < 10:
        print("区间交易日不足: %s~%s" % (start_s, end_s))
        return None
    wends = weekly_ends(cal_sub)
    print("区间 %s~%s: 周数=%d" % (start_s, end_s, len(wends) - 1))
    pool, snap = get_hs300_pool(cal_sub[0])
    print("沪深300池(快照%s): %d 只" % (snap, len(pool)), flush=True)

    rows = []
    nav_wt = 1.0  # 择时 nav
    nav_no = 1.0  # 满仓 nav
    for i in range(len(wends) - 1):
        w = wends[i]
        wn = wends[i + 1]
        print("WEEK %d %s" % (i, w), flush=True)
        # 进程内串行选股（vp_core/scipy），不做子进程隔离——实测无必要且更慢
        top, n_cand = select_topn_w(w, window, topn, st_filter, pool)
        print("  sel=%d (候选%d)" % (len(top), n_cand), flush=True)
        long = macd_long(INDEX_CODE, w)
        print("  macd=%s" % long, flush=True)
        # 满仓基准：忽略 MACD，有股就建
        if top:
            rets = []
            for c in top:
                r = bar_ret(c["ts_code"], w, wn)
                if r is not None:
                    rets.append(r)
            ret_no = float(np.mean(rets)) if rets else 0.0
        else:
            ret_no = 0.0
        # 择时：仅多头建仓
        if long and top:
            ret_wt = ret_no  # 选股相同
            built = True
        else:
            ret_wt = 0.0
            built = False
        nav_wt *= (1 + ret_wt)
        nav_no *= (1 + ret_no)
        rows.append(
            dict(
                buy_d=w,
                sell_d=wn,
                idx_macd="多头" if long else ("空头" if long is False else "NA"),
                built=built,
                n_sel=len(top),
                names=";".join(c["name"] for c in top[:topn]),
                ret_wt=ret_wt,
                ret_no=ret_no,
                nav_wt=nav_wt,
                nav_no=nav_no,
            )
        )
    return pd.DataFrame(rows)


def summarize(df):
    if df is None or df.empty:
        print("无数据")
        return
    n = len(df)
    bw = df[df["built"]]
    print("\n=== 周频统计汇总（共 %d 周）===" % n)
    print("【择时版(仅指数多头建仓)】")
    print("  建仓周数: %d / %d" % (len(bw), n))
    print("  周胜率(建仓周): %.1f%%" % ((bw["ret_wt"] > 0).mean() * 100 if len(bw) else 0))
    print("  累计净值: %.3f  (涨%.1f%%)" % (df["nav_wt"].iloc[-1], (df["nav_wt"].iloc[-1] - 1) * 100))
    print("  最佳/最差周: %.2f%% / %.2f%%" % (df["ret_wt"].max() * 100, df["ret_wt"].min() * 100))
    print("【满仓基准(忽略MACD)】")
    print("  周胜率: %.1f%%" % ((df["ret_no"] > 0).mean() * 100))
    print("  累计净值: %.3f  (涨%.1f%%)" % (df["nav_no"].iloc[-1], (df["nav_no"].iloc[-1] - 1) * 100))
    print("  最佳/最差周: %.2f%% / %.2f%%" % (df["ret_no"].max() * 100, df["ret_no"].min() * 100))
    # 扣成本
    dfc = df.copy()
    dfc["ret_wt_c"] = dfc["ret_wt"] - np.where(dfc["built"], RT_COST, 0)
    dfc["ret_no_c"] = dfc["ret_no"] - RT_COST
    nav_wtc = (1 + dfc["ret_wt_c"]).cumprod().iloc[-1]
    nav_noc = (1 + dfc["ret_no_c"]).cumprod().iloc[-1]
    print("【扣成本(单边RT=%.2f%%, 买卖共扣)】" % (RT_COST * 100))
    print("  择时累计: %.3f  满仓累计: %.3f" % (nav_wtc, nav_noc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20260201")
    ap.add_argument("--end", default="20260731")
    ap.add_argument("--window", type=int, default=120)
    ap.add_argument("--topn", type=int, default=15)
    ap.add_argument("--st", action="store_true", help="剔除ST/*")
    ap.add_argument("--probe", action="store_true", help="只跑首周(验证)")
    args = ap.parse_args()

    if args.probe:
        start, end = "20260601", "20260731"
    else:
        start, end = args.start, args.end
    print("区间 %s~%s  window=%d  topn=%d  st=%s" % (start, end, args.window, args.topn, args.st))

    t0 = time.time()
    df = run_range(start, end, args.window, args.topn, args.st)
    print("耗时 %.1fs" % (time.time() - t0))
    if df is None or df.empty:
        return
    summarize(df)
    out = "data/results/volume_profile/weekly_%s_%s_top%d_w%d.csv" % (
        start[:6], end[:6], args.topn, args.window)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print("已写", out)


if __name__ == "__main__":
    main()
