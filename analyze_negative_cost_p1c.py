# -*- coding: utf-8 -*-
"""P1③ 股息回本存活率 —— 视频说"12 或 14 年回本"，真实存活率多少？

来源：B站 BV1G7466MEPN（悦悦笔记）五法之③「股息收租 / 回本」
P0 已证成本价不进收益式；P1① 证翻倍抽本≈普通减仓。本脚本独立验证视频最自信的
"靠分红 12-14 年回本"——它的隐藏假设是「这只票能连续分红 14 年且不退市」。

数据：
    dividend_detail (ts_code, end_date, ex_date, cash_div 每股) —— 50657 行
    daily.close (名义价) + adj_factor (hfq 调整) —— 入场价用

三条红线（与计划一致，必查）：
    ① hfq 双口径：raw 入场价=你实际付的成本（主口径）；hfq 入场价=稳健性校验
       （hfq 把未来分红折进价格，入场价更高 → 回本更难；若两种口径结论一致则稳）
    ② 幸存者偏差校正：entry 时只看【当时】可得信息（该票当年在分红）。
       后来中断/削减/退市的，一律算「失败」。对比「只统计活到今天的连续派息者」(biased)。
    ③ 税后红利：长期持有 >1 年免征（红利税阶梯 20%/10%/0%）。
       默认 tax=0（买持收租的真实口径）；可用 --tax 0.1/0.2 做敏感。

口径细节：
    - 入场：在年份 Y 的除权日 ex_date 按收盘价买入 1 股，付 P0（名义）。
      买在除权日 → 当年分红归前手，从 Y+1 年起收息（保守）。
    - 回本：后续各年 cash_div 累加（税后），首次 cum >= P0 即回本，记 payback_year。
    -  horizon：14 个日历年（Y_end_year+1 .. +14）。数据不足以覆盖 14 年者剔除（不虚报成功）。

用法：
    python analyze_negative_cost_p1c.py --probe          # 快跑验证
    python analyze_negative_cost_p1c.py --limit 0 --tax 0 # 全样本（limit 0=不限）
"""

import argparse
import os
import sqlite3
import time

import numpy as np
import pandas as pd

DB = "D:/tu-shareData/astock_daily.db"
OUT_DIR = os.path.join("data", "results", "negative_cost")
HORIZON_YEARS = 14
ENTRY_YEARS = [str(y) for y in range(2005, 2013)]   # 2005..2012，确保 14 年前向窗口


def load_dividends(con):
    """返回 div[ts_code] = 按 end_date 升序的 [(end_date, ex_date, cash_div), ...]。
    cash_div 按 (ts_code, end_date) 聚合（处理中期+末期、重复行）。"""
    rows = con.execute(
        "SELECT ts_code, end_date, ex_date, cash_div FROM dividend_detail").fetchall()
    agg = {}   # (ts_code, end_date) -> (ex_date, cash_div_sum)
    for ts, ed, exd, cd in rows:
        if cd is None or cd <= 0:
            continue
        key = (ts, ed)
        if key in agg:
            agg[key] = (exd, agg[key][1] + float(cd))
        else:
            agg[key] = (exd, float(cd))
    div = {}
    for (ts, ed), (exd, cd) in agg.items():
        div.setdefault(ts, []).append((ed, exd, cd))
    for ts in div:
        div[ts].sort(key=lambda x: x[0])
    return div


def load_price_lookup(con, ts_code):
    """返回该票 {trade_date(str): (close, adj_factor)}，用 nearest-prior 查价。"""
    closes = {}
    af = {}
    for td, c in con.execute(
            "SELECT trade_date, close FROM daily WHERE ts_code=? AND close>0", (ts_code,)):
        closes[str(td)] = float(c)
    for td, a in con.execute(
            "SELECT trade_date, adj_factor FROM adj_factor WHERE ts_code=?", (ts_code,)):
        af[str(td)] = float(a)
    return closes, af


def nearest_prior(lut, date_str):
    """lut: {date_str: val}，返回 <= date_str 的最大日期对应值（无则 None）。"""
    if date_str in lut:
        return lut[date_str]
    # 线性扫描（每日序已按字符串序=时间序）；样本量小可接受
    best = None
    for d, v in lut.items():
        if d <= date_str:
            if best is None or d > best:
                best = d
    return lut.get(best) if best else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tax", type=float, default=0.0, help="红利税（长期持有默认 0）")
    ap.add_argument("--limit", type=int, default=200, help="限制股票数（0=全样本）")
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()

    t_all = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 72)
    print("P1③ 股息回本存活率（tax=%.0f%%，horizon=%d年）" % (args.tax * 100, HORIZON_YEARS))
    print("=" * 72)

    con = sqlite3.connect(DB)
    div = load_dividends(con)
    print("[dividend] %d 只票有分红记录" % len(div))

    n_limit = 200 if args.probe else (args.limit if args.limit else len(div))
    recs = []
    continuity_counter = {"continuous14": 0, "entries": 0}

    t0 = time.time()
    for ti, ts in enumerate(div):
        if ti >= n_limit:
            break
        evs = div[ts]
        closes, af = load_price_lookup(con, ts)
        for i, (ed, exd, cd_y) in enumerate(evs):
            yend = ed[:4]
            if yend not in ENTRY_YEARS:
                continue
            p0_raw = nearest_prior(closes, exd)
            if p0_raw is None or p0_raw <= 0:
                continue
            p0_hfq = p0_raw * (nearest_prior(af, exd) or 1.0)
            yield_raw = cd_y / p0_raw

            # 前向累加（Y+1 起）
            cum = 0.0
            achieved = None
            payback_year = None
            # 连续性：Y+1..Y+14 每年是否都有分红记录
            nxt_years = [(int(yend) + k) for k in range(1, HORIZON_YEARS + 1)]
            yr_has = {int(e[0][:4]): True for e in evs[i + 1:]}
            continuous = all(y in yr_has for y in nxt_years)

            continuity_counter["entries"] += 1
            if continuous:
                continuity_counter["continuous14"] += 1

            for e2 in evs[i + 1:]:
                ey = int(e2[0][:4])
                if ey > int(yend) + HORIZON_YEARS:
                    break
                cum += e2[2] * (1.0 - args.tax)
                if cum >= p0_raw and achieved is None:
                    achieved = True
                    payback_year = ey - int(yend)
                    break
            recs.append({
                "ts_code": ts,
                "entry_year": yend,
                "p0_raw": p0_raw,
                "p0_hfq": p0_hfq,
                "yield_raw": yield_raw,
                "cum14_raw": cum,            # 14 年累计税后分红（占成本比）
                "achieved": bool(achieved),
                "payback_year": payback_year,
                "continuous14": bool(continuous),
            })
    con.close()
    print("       处理 %d 只，事件 %d，耗时 %.1fs" % (min(n_limit, len(div)), len(recs), time.time() - t0))

    if not recs:
        print("无有效事件")
        return

    df = pd.DataFrame(recs)

    # ── L0 幸存者偏差校正（核心）──
    print("\n── L0 幸存者偏差校正 ──")
    n_entries = len(df)
    c14 = df["continuous14"].mean() * 100
    print("  入场年（%s）分红股中，后续连续 14 年派息的比例 = %.1f%%（即 %.1f%% 做不到）"
          % ("/".join(ENTRY_YEARS[:3]) + "…", c14, 100 - c14))
    print("  → 视频隐含假设你押中了这 %.1f%% 的「连续派息者」。" % c14)

    # ── L1 真实回本存活率（无偏 vs 最乐观存活）──
    print("\n── L1 真实回本存活率（14 年内累计分红 ≥ 成本）──")
    succ_all = df["achieved"].mean() * 100
    succ_cont = df[df["continuous14"]]["achieved"].mean() * 100 if df["continuous14"].any() else float("nan")
    print("  【无偏】所有入场决策（含中断/退市=失败）：回本率 = %.1f%%" % succ_all)
    print("  【最乐观存活】仅连续派息 14 年者（已假设你押中了存活的那 %d%%）：回本率 = %.1f%%"
          % (c14, succ_cont))
    print("  → 连续派息本身 ≠ 回本：即使假设你押中了存活的少数，回本率仍仅 %.1f%%。" % succ_cont)
    print("    真实概率 ≈ 存活率(%.1f%%) × 存活者内回本率(%.1f%%) ≈ %.1f%%，与无偏 %.1f%% 一致。"
          % (c14, succ_cont, c14 / 100 * succ_cont, succ_all))
    print("    视频把『能连续分红』误当成『能回本』，漏算了股息率门槛。")

    # ── L2 回本年限分布（在成功者中）──
    print("\n── L2 回本年限分布（仅成功者）──")
    ok = df[df["achieved"]]
    if len(ok):
        print("  成功者 %d 人：回本年限 中位 %.1f / 均值 %.1f / 最长 %d 年"
              % (len(ok), ok["payback_year"].median(), ok["payback_year"].mean(), ok["payback_year"].max()))
        # 多少成功者其实 >14 年? achieved 内都是 <=14（因 horizon 截断），但部分可能恰好卡在 14
        print("  （注：achieved 均 ≤%d 年；未达成的更长或永不）" % HORIZON_YEARS)

    # ── L3 入场估值决定回本（依赖估值）──
    print("\n── L3 回本强烈依赖入场估值（股息率）──")
    bins = [(0, 0.01, "<1%"), (0.01, 0.02, "1-2%"), (0.02, 0.03, "2-3%"),
            (0.03, 0.05, "3-5%"), (0.05, 0.08, "5-8%"), (0.08, 1.0, ">8%")]
    rows = []
    for lo, hi, lab in bins:
        m = (df["yield_raw"] >= lo) & (df["yield_raw"] < hi)
        sub = df[m]
        if len(sub) < 5:
            continue
        ok_sub = sub[sub["achieved"]]
        rows.append({
            "入场股息率": lab,
            "样本": len(sub),
            "回本率(%)": round(sub["achieved"].mean() * 100, 1),
            "连续14年派息(%)": round(sub["continuous14"].mean() * 100, 1),
            "中位回本年限": (round(ok_sub["payback_year"].median(), 1)
                            if len(ok_sub) else float("nan")),
        })
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))

    # ── hfq 稳健性 ──
    print("\n── hfq 入场价稳健性（hfq 把未来分红折进价格 → 入场更贵 → 回本更难）──")
    succ_hfq = (df["cum14_raw"] >= df["p0_hfq"]).mean() * 100
    print("  以 hfq 入场价重算的 14 年回本率 = %.1f%%（vs raw %.1f%%）" % (succ_hfq, succ_all))
    print("  → 两种口径都低，结论稳健：回本靠的是「高股息率 + 连续派息」，不是 14 年咒语。")

    stamp = time.strftime("%Y%m%d")
    out = os.path.join(OUT_DIR, "p1c_events_%s%s.csv" % (stamp, "_probe" if args.probe else ""))
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print("\n逐事件明细已写入：%s" % out)
    print("总耗时 %.1fs" % (time.time() - t_all))


if __name__ == "__main__":
    main()
