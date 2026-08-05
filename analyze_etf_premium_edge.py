# -*- coding: utf-8 -*-
"""
ETF 折溢价的预测力事件研究
================================
问题：调仓日买入时，标的的折溢价率对未来 20 个交易日收益有无预测力？
      如果没有，把折溢价过滤接进 run_etf_rotation.py 就是纯噪音。

方法：
  · 样本 = 轮动池 20 只 ETF × 每月调仓日（月度第5交易日）
  · 溢价率 = (当日收盘价 - 当日或之前最近单位净值) / 单位净值
  · 前瞻收益 = 后 20 个交易日的收盘价涨跌幅（不含费用）
  · 按溢价分档统计均值/中位数/胜率，并单独看跨境 ETF

注意：货币ETF(511990)等计价单位不一致的标的自动剔除。
"""
import sys
import sqlite3
import statistics as st

sys.path.insert(0, ".")
import config
from backtest import etf_premium_filter as epf
from run_monthly_rebalance import get_monthly_5th_trading_days

POOL = [
    "510300.SH", "510050.SH", "515800.SH", "510980.SH", "510500.SH",
    "512100.SH", "159915.SZ", "159949.SZ", "588000.SH", "512480.SH",
    "515030.SH", "512010.SH", "159928.SZ", "512880.SH", "159920.SZ",
    "513100.SH", "518880.SH", "501018.SH", "511010.SH", "511990.SH",
]
FWD_DAYS = 20
START, END = "20180101", "20260803"


def main():
    conn = sqlite3.connect(config.DATA["local_db_path"])

    # ── 交易日历（用沪深300指数日线代理）──
    tds = [str(r[0]) for r in conn.execute(
        "SELECT DISTINCT trade_date FROM index_daily "
        "WHERE ts_code='000300.SH' AND trade_date BETWEEN ? AND ? "
        "ORDER BY trade_date", (START, END))]
    if not tds:
        print("[ERR] 无交易日数据")
        return
    rebal = [str(d) for d in get_monthly_5th_trading_days(tds)]
    print(f"交易日 {len(tds)} 天 | 调仓日 {len(rebal)} 次 ({rebal[0]}~{rebal[-1]})")

    # ── 预载每只 ETF 的价格序列 ──
    px = {}
    for code in POOL:
        rows = conn.execute(
            "SELECT trade_date, close FROM etf_daily WHERE ts_code=? "
            "AND trade_date BETWEEN ? AND ? ORDER BY trade_date",
            (code, START, END)).fetchall()
        px[code] = {str(d): float(c) for d, c in rows if c}

    # ── 预载净值序列（按日期排序，便于 as-of 查找）──
    nav_series = {}
    for code in POOL:
        rows = conn.execute(
            "SELECT nav_date, unit_nav FROM etf_nav WHERE ts_code=? "
            "ORDER BY nav_date", (code,)).fetchall()
        nav_series[code] = [(str(d), float(v)) for d, v in rows if v and v > 0]

    def nav_as_of(code, day):
        """该日或之前最近的一个单位净值 (nav_date, nav)。"""
        arr = nav_series.get(code) or []
        lo, hi, best = 0, len(arr) - 1, None
        while lo <= hi:
            mid = (lo + hi) // 2
            if arr[mid][0] <= day:
                best = arr[mid]
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    td_idx = {d: i for i, d in enumerate(tds)}
    samples = []

    for rd in rebal:
        i = td_idx.get(rd)
        if i is None or i + FWD_DAYS >= len(tds):
            continue
        fwd_day = tds[i + FWD_DAYS]
        for code in POOL:
            p0 = px[code].get(rd)
            p1 = px[code].get(fwd_day)
            if not p0 or not p1:
                continue
            nv = nav_as_of(code, rd)
            if not nv:
                continue
            nav_date, nav = nv
            # 计价单位不一致（货币ETF）→ 跳过
            if p0 / nav > epf.PRICE_NAV_RATIO_CAP:
                continue
            prem = (p0 - nav) / nav
            # 净值滞后过久 → 溢价读数不可信，记录但标记
            stale = epf.staleness_days(nav_date, rd)
            samples.append({
                "date": rd, "code": code, "prem": prem,
                "fwd": p1 / p0 - 1.0,
                "cross": epf.is_crossborder(code),
                "stale": stale,
            })

    print(f"有效样本 {len(samples)} 条\n")
    if not samples:
        return

    # ── 分档统计 ──
    buckets = [
        ("溢价 >5%",      lambda x: x >= 0.05),
        ("溢价 3~5%",     lambda x: 0.03 <= x < 0.05),
        ("溢价 1~3%",     lambda x: 0.01 <= x < 0.03),
        ("平价 -1~1%",    lambda x: -0.01 <= x < 0.01),
        ("折价 -3~-1%",   lambda x: -0.03 <= x < -0.01),
        ("折价 <-3%",     lambda x: x < -0.03),
    ]

    def report(title, data):
        print("=" * 78)
        print(f"  {title}  (N={len(data)})")
        print("=" * 78)
        print(f"{'档位':<14}{'样本':>6}{'占比':>8}{'后20日均值':>12}{'中位数':>10}{'胜率':>8}")
        print("-" * 78)
        total = len(data)
        for name, fn in buckets:
            sub = [s["fwd"] for s in data if fn(s["prem"])]
            if not sub:
                print(f"{name:<14}{0:>6}{'-':>8}{'-':>12}{'-':>10}{'-':>8}")
                continue
            mean = sum(sub) / len(sub)
            med = st.median(sub)
            win = sum(1 for v in sub if v > 0) / len(sub)
            print(f"{name:<14}{len(sub):>6}{len(sub)/total:>7.1%}"
                  f"{mean:>11.2%}{med:>10.2%}{win:>8.1%}")
        allf = [s["fwd"] for s in data]
        print("-" * 78)
        print(f"{'全样本':<14}{len(allf):>6}{1:>7.1%}"
              f"{sum(allf)/len(allf):>11.2%}{st.median(allf):>10.2%}"
              f"{sum(1 for v in allf if v>0)/len(allf):>8.1%}")
        print()

    report("全部 ETF", samples)
    report("跨境 ETF（恒生/纳指）", [s for s in samples if s["cross"]])
    report("境内 ETF", [s for s in samples if not s["cross"]])

    # ── 关键对比：过滤线两侧 ──
    print("=" * 78)
    print("  关键对比：现行过滤阈值两侧的收益差")
    print("=" * 78)
    for label, cross_flag in [("境内(阈值 warn3%/block5%)", False),
                              ("跨境(阈值 warn1%/block3%)", True)]:
        data = [s for s in samples if s["cross"] == cross_flag]
        if not data:
            continue
        hard = epf.PREMIUM_HARD - (epf.CROSSBORDER_EXTRA if cross_flag else 0)
        warn = epf.PREMIUM_WARN - (epf.CROSSBORDER_EXTRA if cross_flag else 0)
        blocked = [s["fwd"] for s in data if s["prem"] >= hard]
        warned = [s["fwd"] for s in data if warn <= s["prem"] < hard]
        passed = [s["fwd"] for s in data if s["prem"] < warn]
        for nm, arr in [("被BLOCK", blocked), ("被WARN", warned), ("放行OK", passed)]:
            if arr:
                mean = sum(arr) / len(arr)
                win = sum(1 for v in arr if v > 0) / len(arr)
                print(f"  {label:<26} {nm:<8} N={len(arr):>4}  "
                      f"后20日均值={mean:>7.2%}  胜率={win:>6.1%}")
            else:
                print(f"  {label:<26} {nm:<8} N=   0")
        print()

    conn.close()


if __name__ == "__main__":
    main()
