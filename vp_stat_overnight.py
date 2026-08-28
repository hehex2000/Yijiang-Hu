# -*- coding: utf-8 -*-
"""
vp_stat_overnight: 统计学实验（非策略，仅测"✅机会"信号的隔夜胜率/盈亏比）

口径（用户定义，见报告）：
  - T-1 日收盘：全市场逐只算 VP(window)，判"机会"(支撑位附近<=NEAR + 反转overlay双确认)
    按离支撑由近到远取前 topn(默认15)
  - 买入价 = T-1 收盘价
  - T 日：开盘价 > 买入价 -> 开盘卖出(止盈)；否则 T 日收盘卖出(无论盈亏)
  - 统计一个月：胜率/盈亏比/平均收益/月度累计

注：本实验接近可执行(T-1收盘竞价买 + T日开/收盘卖)，未来函数很低；无分钟数据故用T-1收盘。
"""
import os
import sqlite3
import time
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import config
import vp_data
import vp_core

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "results", "volume_profile")
DB = config.DATA.get("local_db_path", "")
NEAR = 0.03
RT_COST = 0.003  # 双边成本约0.3% (佣万2.5*2 + 滑点0.1%*2 + 印花0.05%卖)


def get_calendar():
    conn = sqlite3.connect(DB)
    try:
        cal = pd.read_sql_query(
            "SELECT trade_date FROM daily WHERE ts_code='600519.SH' ORDER BY trade_date",
            conn,
        )["trade_date"].tolist()
    finally:
        conn.close()
    return cal


def is_st(name):
    s = str(name)
    return ("ST" in s) or ("*" in s)


def judge_opportunity(ts_code, t1, window):
    """截止 t1 收盘判该票是否"机会"。返回 dict 或 None（单只异常吞掉，返回 None）。"""
    try:
        d = vp_data.get_window(ts_code, t1, lookback=window)
        if d is None or len(d) < window or len(d) <= 21:
            return None
        price = float(d["close"].iloc[-1])
        if price <= 0 or not np.isfinite(price):
            return None
        n_bins = max(20, window // 3)  # 120日窗口->40箱, 与250日/80箱不同的独立口径
        res = vp_core.volume_profile(d, n_bins=n_bins, smooth_sigma=2.0)
        if res is None:
            return None
        centers, raw, sm = res
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
    except Exception:
        return None


def get_hs300_pool(trade_date):
    """取 <= trade_date 最近快照日的沪深300成分股 distinct ts_code 列表。"""
    conn = sqlite3.connect(DB)
    try:
        snap = pd.read_sql_query(
            "SELECT MAX(trade_date) AS d FROM index_constituent "
            "WHERE index_code='000300.SH' AND trade_date<='%s'" % trade_date,
            conn)["d"].iloc[0]
        codes = pd.read_sql_query(
            "SELECT DISTINCT ts_code FROM index_constituent "
            "WHERE index_code='000300.SH' AND trade_date='%s'" % snap,
            conn)
    finally:
        conn.close()
    return codes["ts_code"].tolist(), snap


def _judge_worker(args):
    """进程池 worker：单只判机会；C 层崩溃(Windows scipy 偶发 segfault)由
    主进程 ProcessPoolExecutor 捕获后退化串行，不会整轮作废。"""
    ts_code, t1, window = args
    try:
        return ts_code, judge_opportunity(ts_code, t1, window)
    except Exception:
        return ts_code, None


def select_topn(t1, window, topn, st_filter, pool=None):
    conn = sqlite3.connect(DB)
    try:
        codes = pd.read_sql_query("SELECT ts_code, name FROM stock_basic", conn)
    finally:
        conn.close()
    if pool is not None:
        codes = codes[codes["ts_code"].isin(pool)]
    name_map = dict(zip(codes["ts_code"], codes["name"]))
    tasks = [(ts, t1, window) for ts in codes["ts_code"].tolist()]

    def collect(tasks_iter):
        cands = []
        for ts_code, j in tasks_iter:
            if not j:
                continue
            nm = name_map.get(ts_code, "")
            if st_filter and is_st(nm):
                continue
            j["name"] = nm
            cands.append(j)
        return cands

    cands = []
    try:
        with ProcessPoolExecutor(max_workers=4) as ex:
            cands = collect(ex.map(_judge_worker, tasks, chunksize=8))
    except BaseException:
        # 进程池异常(BrokenProcessPool 等)：退化串行，保底跑完
        cands = collect((ts, judge_opportunity(ts, t1, window)) for ts, _, _ in tasks)
    cands.sort(key=lambda x: x["sup_dist"])
    return cands[:topn], len(cands)


def simulate_t(ts_code, t1, t, cost):
    df = vp_data.get_daily(ts_code)
    if df is None or df.empty:
        return None
    d1 = df[df["date"] == int(t1)]
    dt = df[df["date"] == int(t)]
    if d1.empty or dt.empty:
        return None
    buy = float(d1["close"].iloc[0])
    open_t = float(dt["open"].iloc[0])
    close_t = float(dt["close"].iloc[0])
    if open_t > buy:
        sell, tag = open_t, "open"
    else:
        sell, tag = close_t, "close"
    r = sell / buy - 1
    if cost:
        r = r - RT_COST
    return dict(t1=t1, t=t, ts_code=ts_code, buy=buy,
                open_t=open_t, close_t=close_t, sell=sell, ret=r, tag=tag)


def run_month(month, window, topn, st_filter, pool):
    cal = get_calendar()
    md = [d for d in cal if d.startswith(month)]
    if len(md) < 2:
        print("月份 %s 交易日不足" % month)
        return None
    rows = []
    for i in range(len(md) - 1):
        t1, t = md[i], md[i + 1]
        top, _ = select_topn(t1, window, topn, st_filter, pool=pool)
        for c in top:
            sim = simulate_t(c["ts_code"], t1, t, cost=False)
            if sim:
                sim["name"] = c["name"]
                sim["sup_dist"] = c["sup_dist"]
                rows.append(sim)
    res = pd.DataFrame(rows)
    if not res.empty:
        res["ret_cost"] = res["ret"] - RT_COST
    return res


def summarize(res, cost_col):
    if res is None or res.empty:
        print("无交易")
        return
    col = "ret_cost" if cost_col else "ret"
    N = len(res)
    win = (res[col] > 0).sum()
    mean_r = res[col].mean()
    win_r = res.loc[res[col] > 0, col].mean()
    loss_r = res.loc[res[col] <= 0, col].mean()
    pf = win_r / abs(loss_r) if loss_r < 0 else float("nan")
    gross_win = res.loc[res[col] > 0, col].sum()
    gross_loss = -res.loc[res[col] <= 0, col].sum()
    pf_total = gross_win / gross_loss if gross_loss > 0 else float("nan")
    cum = ((1 + res[col]).prod() - 1)
    open_n = (res["tag"] == "open").sum()
    label = "扣成本(双边0.3%)" if cost_col else "免成本"
    print("=== 隔夜统计 [%s] ===" % label)
    print("  总笔数: %d   选股日: %d   日均笔数: %.1f" %
          (N, res["t1"].nunique(), N / max(res["t1"].nunique(), 1)))
    print("  胜率: %.1f%%  (%d/%d)" % (win / N * 100, win, N))
    print("  平均单笔: %.3f%%   最佳: %.2f%%   最差: %.2f%%" %
          (mean_r * 100, res[col].max() * 100, res[col].min() * 100))
    print("  盈亏比(均盈/均亏): %.2f   总盈亏比(总盈/总亏): %.2f" % (pf, pf_total))
    print("  月度累计(等权): %.2f%%" % (cum * 100))
    print("  开盘止盈占比: %.1f%%   收盘离场: %.1f%%" %
          (open_n / N * 100, (N - open_n) / N * 100))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default=None, help="YYYYMM, 默认上个月完整月")
    ap.add_argument("--window", type=int, default=120)
    ap.add_argument("--topn", type=int, default=15)
    ap.add_argument("--st", action="store_true", help="剔除ST/*")
    ap.add_argument("--one", action="store_true", help="只跑该月最后T-1日(测速/验证)")
    ap.add_argument("--pool", default="hs300", choices=["hs300", "all"],
                    help="候选池: hs300(沪深300) / all(全市场)")
    args = ap.parse_args()

    cal = get_calendar()
    if args.month:
        month = args.month
    else:
        last = cal[-1]
        y, m = int(last[:4]), int(last[4:6])
        m -= 1
        if m == 0:
            m = 12
            y -= 1
        month = "%04d%02d" % (y, m)
    pool = None
    pool_txt = "全市场"
    if args.pool == "hs300":
        pool, snap = get_hs300_pool(month + "31")
        pool_txt = "沪深300(快照%s, %d只)" % (snap, len(pool))
    print("样本月: %s  池: %s  window=%d  topn=%d  st_filter=%s" %
          (month, pool_txt, args.window, args.topn, args.st))

    if args.one:
        md = [d for d in cal if d.startswith(month)]
        t1, t = md[-2], md[-1]
        t0 = time.time()
        top, total = select_topn(t1, args.window, args.topn, args.st, pool=pool)
        dt = time.time() - t0
        print("T-1=%s 机会候选总数=%d 取前%d (耗时%.1fs)" % (t1, total, args.topn, dt))
        print("前%d只:" % args.topn)
        for c in top:
            print("  %s %s 价%.2f 支撑%.2f 距支%.2f%% rev21=%.2f%%" %
                  (c["ts_code"], c["name"], c["price"], c["support"],
                   c["sup_dist"] * 100, c["rev21"] * 100))
        rows = []
        for c in top:
            sim = simulate_t(c["ts_code"], t1, t, cost=False)
            if sim:
                sim["name"] = c["name"]
                rows.append(sim)
        if rows:
            r = pd.DataFrame(rows)
            print("T=%s 模拟(免成本): 胜率%.0f%% 均收%.3f%%" %
                  (t, (r["ret"] > 0).mean() * 100, r["ret"].mean() * 100))
        return

    res = run_month(month, args.window, args.topn, args.st, pool=pool)
    summarize(res, False)
    summarize(res, True)
    if res is not None and not res.empty:
        os.makedirs(OUT, exist_ok=True)
        fp = os.path.join(OUT, "overnight_%s_top%d_w%d.csv" %
                          (month, args.topn, args.window))
        res.to_csv(fp, index=False, encoding="utf-8-sig")
        print("已存: %s" % fp)


if __name__ == "__main__":
    main()
