# -*- coding: utf-8 -*-
"""
㉑ 必选 vs 可选消费 —— 六道闸门（截面配置规律）
================================================
视频主张：必选消费抗跌于可选（熊市规律）。
口径（台账已锁，助手归类非视频原文，落地前须用户确认）：
  必选 = 食品饮料 801120 + 农林牧渔 801010
  可选 = 家用电器 801110 + 商贸零售 801200 + 纺织服饰 801130 + 社会服务 801210
方法：
  - 两组等权日频再平衡组合（申万一级价格指数，不含股息——声明口径），2010-01-04 起
  - 对照：红利（000922 全收益 H00922）、宽基（000906 全收益 H00906）
  - Gate 1：全期对比 + 熊市事件研究（与⑳同款 4 段 + 2012 盲区段）
  - Gate 5 落地判定：必选熊市超额 vs 红利熊市超额 头对头（⑳已测：红利剔2021中位+8.65%）
用法：python consumer_defensive.py
"""
import sys
import sqlite3

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from bench_index import load_benchmark

DB = "D:/tu-shareData/astock_daily.db"
START = "20100104"
END = "20260828"

DEFENSIVE = {"801120.SI": "食品饮料", "801010.SI": "农林牧渔"}          # 必选
CYCLICAL = {"801110.SI": "家用电器", "801200.SI": "商贸零售",
            "801130.SI": "纺织服饰", "801210.SI": "社会服务"}           # 可选
# 熊市段（与⑳ dividend_bear_excess 同款）+ 2012 盲区段
BEARS = [("2012段", "20120302", "20120618"),
         ("#1", "20110104", "20130101"),
         ("#2", "20150612", "20170601"),
         ("#3", "20180129", "20190404"),
         ("#4", "20210218", "20240918")]


def seg_ret(nav, lo, hi):
    n = nav[(nav.index >= lo) & (nav.index <= hi)]
    return n.iloc[-1] / n.iloc[0] - 1


def perf(nav):
    yrs = len(nav) / 252
    ann = nav.iloc[-1] ** (1 / yrs) - 1
    mdd = (nav / nav.cummax() - 1).min()
    return ann * 100, mdd * 100


def main():
    conn = sqlite3.connect(DB)
    codes = list(DEFENSIVE) + list(CYCLICAL)
    px = {}
    for c in codes:
        df = pd.read_sql(
            "SELECT trade_date, close FROM sw_industry_daily WHERE ts_code=? AND trade_date>=?",
            conn, params=(c, START))
        s = df.set_index("trade_date")["close"].astype(float)
        s.index = s.index.astype(str)  # 🔴 trade_date 是 INTEGER，必须先转 str
        px[c] = s[s.index <= END]
    conn.close()
    rets = pd.DataFrame(px).pct_change()
    n_nan = rets.isna().sum().sum()
    rets = rets.fillna(0)  # 首行 + 停牌日(6个)按0收益，占比可忽略
    assert rets.notna().all().all()
    print(f"样本 {len(rets)} 日  {rets.index.min()}~{rets.index.max()}  "
          f"停牌/首行缺失 {n_nan} 格 → 按0收益处理")

    r_def = rets[list(DEFENSIVE)].mean(axis=1)   # 必选等权
    r_cyc = rets[list(CYCLICAL)].mean(axis=1)    # 可选等权
    navs = {"必选消费(等权2)": (1 + r_def).cumprod(),
            "可选消费(等权4)": (1 + r_cyc).cumprod()}

    conn = sqlite3.connect(DB)
    for code, label in [("000906.SH", "宽基800(TR)"), ("000922.SH", "红利(TR)")]:
        b, meta = load_benchmark(code, START, END, conn=conn, nav_price_mode="hfq")
        assert b is not None, f"基准 {code} 加载失败"
        s = b.set_index("trade_date")["close"].astype(float)
        s.index = s.index.astype(str)
        navs[label] = s[s.index <= END]
        print(f"基准 {label}: {meta['resolved_code']} {len(s)} 日")
    conn.close()

    idx = navs["必选消费(等权2)"].index
    for k in list(navs):
        navs[k] = navs[k].reindex(idx).ffill()
        navs[k] = navs[k] / navs[k].iloc[0]  # 🔴 基准 close 是绝对点位，必须归一化

    print("\n=== Gate 1 全期（2010-01 ~ 2026-08，价格指数口径 / 基准为全收益）===")
    for k, nav in navs.items():
        ann, mdd = perf(nav)
        print(f"  {k:16s} 年化 {ann:+6.2f}%  回撤 {mdd:7.2f}%")
    print("  （注意：必选/可选是价格指数不含股息，红利类股息高，跨表对比须谨慎）")

    print("\n=== Gate 1 熊市事件研究（视频主张：必选抗跌于可选）===")
    print(f"  {'段':6s} {'区间':26s} {'必选':>8s} {'可选':>8s} {'必-可超额':>9s} "
          f"{'红利':>8s} {'宽基':>8s}")
    wins = 0
    for tag, lo, hi in BEARS:
        d = {k: seg_ret(v, lo, hi) * 100 for k, v in navs.items()}
        ex = d["必选消费(等权2)"] - d["可选消费(等权4)"]
        wins += ex > 0
        print(f"  {tag:6s} {lo}~{hi:18s} {d['必选消费(等权2)']:+8.2f} {d['可选消费(等权4)']:+8.2f} "
              f"{ex:+9.2f} {d['红利(TR)']:+8.2f} {d['宽基800(TR)']:+8.2f}")
    print(f"  → 必选跑赢可选 {wins}/{len(BEARS)} 段")

    print("\n=== 牛市对照（规律若只在熊市成立，牛市应反转）===")
    BULLS = [("2019-2021牛", "20190104", "20210218"),
             ("2024-09起修复", "20240918", "20260828")]
    for tag, lo, hi in BULLS:
        d = {k: seg_ret(v, lo, hi) * 100 for k, v in navs.items()}
        ex = d["必选消费(等权2)"] - d["可选消费(等权4)"]
        print(f"  {tag:12s} 必选 {d['必选消费(等权2)']:+8.2f}  可选 {d['可选消费(等权4)']:+8.2f}  "
              f"必-可 {ex:+8.2f}  红利 {d['红利(TR)']:+8.2f}")

    print("\n=== Gate 5 落地判定素材：防御配置头对头（恒持视角）===")
    print("  问题：如果为了'熊市抗跌'配置消费，为什么不直接恒持红利？")
    for tag, lo, hi in BEARS:
        d = {k: seg_ret(v, lo, hi) * 100 for k, v in navs.items()}
        print(f"  {tag:6s} 红利−必选 = {d['红利(TR)'] - d['必选消费(等权2)']:+8.2f}pp")
    a_def, mdd_def = perf(navs["必选消费(等权2)"])
    a_div, mdd_div = perf(navs["红利(TR)"])
    print(f"  全期：必选 {a_def:+.2f}%/年 vs 红利 {a_div:+.2f}%/年")


if __name__ == "__main__":
    main()
