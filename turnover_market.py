# -*- coding: utf-8 -*-
"""
全市场「换手率」日序列构建 + 四口径对照 + 漂移体检

设计动机
--------
视频 BV1Lghg6NEA 给的是"全市场换手率 5-6%"这类**绝对阈值**。
但换手率的数值高度依赖两件事：

  1. **口径**（分母用什么）
     实测 2024-10-08：流通市值加权 4.33% / 简单平均 8.60% / 中位数 6.72%
     → 同一个市场同一天，口径不同能差 2 倍。**视频的 5-6% 对应"简单平均"口径。**

  2. **结构性漂移**（股票池扩张）
     全 A 股票数 2015 年 2326 只 → 2026 年 5206 只，小盘股占比持续上升，
     而小票换手率天然更高 → **简单平均口径会系统性抬升**（与破净率同款陷阱）。
     流通市值加权对成分扩张稳健得多（大市值主导）。

四口径
------
均剔北交所（.BJ，2021 后才上市且换手率极低，纳入会拉低整体）。
停牌股 turnover_rate=0/NULL，**保留在分母**（市值不为零，剔除会高估换手率）。

  turn_flow   流通市值加权 = Σ(circ_mv × tr/100) / Σ(circ_mv)          【主口径】
  turn_free   自由流通加权 = Σ(circ_mv × tr/100) / Σ(free_mv)
              （free_mv = circ_mv × tr / tr_f，由两换手率定义反推，无需 daily 表）
  turn_mean   简单平均     = AVG(tr)                                    【视频口径】
  turn_total  总市值口径   = Σ(circ_mv × tr/100) / Σ(total_mv)

方向约定
--------
换手率**高** = 市场过热 = 该减仓（与破净率相反：破净率高=便宜=该满仓）。

用法
----
  python turnover_market.py            # 构建全序列（首次约 30-60 秒）
  python turnover_market.py --check    # 只核对锚点日 + 漂移体检
  python turnover_market.py --reuse    # 读缓存 CSV
"""
import os
import sys
import sqlite3
import argparse
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from config import DATA  # noqa: E402

OUT_DIR = os.path.join("data", "results", "turnover")
CACHE = os.path.join(OUT_DIR, "market_turnover.csv")
DRIFT = os.path.join(OUT_DIR, "turnover_drift.csv")

START = "20100101"    # daily_basic 起点
END = "20260831"
ROLL_WIN = 750        # 滚动分位窗口（约 3 年）
ROLL_MIN = 250


def get_conn():
    return sqlite3.connect(DATA["local_db_path"])


def build_series():
    """一次全表 GROUP BY（1320 万行，约 30-60 秒），四口径并列。"""
    con = get_conn()
    print("[换手率] 聚合 daily_basic 全表（约需 30-60 秒）...")

    # 分子统一 = Σ(circ_mv × turnover_rate/100)，即当日总成交额（以流通市值反推）
    # free_mv = circ_mv × turnover_rate / turnover_rate_f（两换手率定义相除反推自由流通市值）
    #   tr   = vol_shares / float_shares
    #   tr_f = vol_shares / free_shares
    #   => free_shares = float_shares × tr / tr_f
    q = """
    SELECT b.trade_date,
           COUNT(*)                                                        AS n_all,
           SUM(b.circ_mv)                                                  AS sum_circ,
           SUM(b.total_mv)                                                 AS sum_total,
           SUM(CASE WHEN b.turnover_rate IS NOT NULL
                    THEN b.circ_mv * b.turnover_rate / 100.0 ELSE 0 END)   AS turnover_amt,
           SUM(CASE WHEN b.turnover_rate IS NOT NULL
                     AND b.turnover_rate_f > 0
                    THEN b.circ_mv * b.turnover_rate / b.turnover_rate_f
                    ELSE 0 END)                                            AS sum_free,
           AVG(b.turnover_rate)                                            AS mean_tr
    FROM daily_basic b
    WHERE b.trade_date >= ? AND b.trade_date <= ?
      AND substr(b.ts_code, -2) <> 'BJ'      /* 剔北交所 */
      AND b.circ_mv > 0
    GROUP BY b.trade_date
    ORDER BY b.trade_date
    """
    df = pd.read_sql_query(q, con, params=(START, END))
    con.close()

    df["turn_flow"] = df["turnover_amt"] / df["sum_circ"] * 100.0
    df["turn_free"] = df["turnover_amt"] / df["sum_free"].replace(0, np.nan) * 100.0
    df["turn_mean"] = df["mean_tr"]
    df["turn_total"] = df["turnover_amt"] / df["sum_total"].replace(0, np.nan) * 100.0

    # 滚动分位（当期在最近 750 个交易日中的百分位，含当期自身）
    for c in ["turn_flow", "turn_free", "turn_mean"]:
        df["pct_" + c.split("_")[1]] = (
            df[c].rolling(ROLL_WIN, min_periods=ROLL_MIN)
                 .apply(lambda w: (w[-1] >= w[:-1]).mean() * 100.0, raw=True)
        )
    return df


def drift_table(df):
    """按年体检：检验各口径是否存在长期水平漂移（破净率踩过的坑）。"""
    d = df.copy()
    d["year"] = d["trade_date"].astype(str).str[:4]
    g = d.groupby("year").agg(
        交易日=("trade_date", "count"),
        股票数=("n_all", "mean"),
        流通加权=("turn_flow", "mean"),
        自由流通加权=("turn_free", "mean"),
        简单平均=("turn_mean", "mean"),
        总市值口径=("turn_total", "mean"),
    ).round(3)
    return g.reset_index()


def check_anchors(df):
    """锚点核对：与公开认知对齐的关键时点。"""
    print("\n[锚点核对] 关键时点全A换手率（剔北交所，停牌股保留在分母）")
    print("%-12s %10s %12s %10s %10s" % ("日期", "流通加权", "自由流通加权", "简单平均", "股票数"))
    anchors = [
        ("20150612", "2015 牛市顶"),
        ("20150826", "2015 股灾后"),
        ("20180102", "2018 年初"),
        ("20181228", "2018 年底地量"),
        ("20210205", "2021.02 核心资产顶"),
        ("20240701", "2024 年中地量"),
        ("20240930", "2024.09 924 行情"),
        ("20241008", "2024.10 顶部"),
        ("20260831", "最新"),
    ]
    idx = df.set_index(df["trade_date"].astype(str))
    for d, label in anchors:
        if d not in idx.index:
            continue
        r = idx.loc[d]
        print("%-12s %9.3f%% %11.3f%% %9.3f%% %10d   %s"
              % (d, r["turn_flow"], r["turn_free"], r["turn_mean"], int(r["n_all"]), label))

    print("\n[漂移体检] 按年均值 —— 看各口径是否长期水平漂移")
    dt = drift_table(df)
    print(dt.to_string(index=False))
    dt.to_csv(DRIFT, index=False, encoding="utf-8-sig")
    print("\n-> 漂移表已存 %s" % DRIFT)

    # 漂移量化：首 3 年 vs 末 3 年
    if len(dt) >= 6:
        head = dt.head(3)[["流通加权", "简单平均"]].mean()
        tail = dt.tail(3)[["流通加权", "简单平均"]].mean()
        print("\n[漂移幅度] 首3年 -> 末3年")
        print("  流通市值加权 %.3f%% -> %.3f%%  (×%.2f)"
              % (head["流通加权"], tail["流通加权"], tail["流通加权"] / head["流通加权"]))
        print("  简单平均     %.3f%% -> %.3f%%  (×%.2f)"
              % (head["简单平均"], tail["简单平均"], tail["简单平均"] / head["简单平均"]))
        print("  同期股票数   %.0f -> %.0f  (×%.2f)"
              % (dt.head(3)["股票数"].mean(), dt.tail(3)["股票数"].mean(),
                 dt.tail(3)["股票数"].mean() / dt.head(3)["股票数"].mean()))


def main():
    ap = argparse.ArgumentParser(description="全市场换手率日序列构建")
    ap.add_argument("--check", action="store_true", help="只核对锚点日，不落盘")
    ap.add_argument("--reuse", action="store_true", help="读已缓存 CSV")
    a = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    if a.reuse or a.check:
        df = pd.read_csv(CACHE, dtype={"trade_date": str})
        print("[复用] %s  %d 行  %s ~ %s"
              % (CACHE, len(df), df["trade_date"].iloc[0], df["trade_date"].iloc[-1]))
    else:
        df = build_series()
        df.to_csv(CACHE, index=False, encoding="utf-8-sig")
        print("[落盘] %s  %d 行  %s ~ %s"
              % (CACHE, len(df), df["trade_date"].iloc[0], df["trade_date"].iloc[-1]))

    check_anchors(df)

    print("\n[全样本分位] 各口径分布")
    print(df[["turn_flow", "turn_free", "turn_mean", "turn_total"]]
          .describe(percentiles=[.05, .25, .5, .75, .95]).round(3).to_string())

    # 视频 5-6% 阈值在各口径下的触发率
    print("\n[视频 5-6%% 阈值触发率]（视频未说明口径，这里四口径并列）")
    for c, name in [("turn_flow", "流通市值加权"), ("turn_free", "自由流通加权"),
                    ("turn_mean", "简单平均"), ("turn_total", "总市值口径")]:
        ge5 = (df[c] >= 5.0).sum()
        ge6 = (df[c] >= 6.0).sum()
        print("  %-14s >=5%%: %4d 天(%5.1f%%)   >=6%%: %4d 天(%5.1f%%)"
              % (name, ge5, ge5 / len(df) * 100, ge6, ge6 / len(df) * 100))


if __name__ == "__main__":
    main()
