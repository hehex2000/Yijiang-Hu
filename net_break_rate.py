# -*- coding: utf-8 -*-
"""
全市场「破净率」日序列构建 + 口径对照

破净 = PB < 1（每股净资产 > 股价）。破净率高 = 市场整体便宜。

口径
----
分母一律取「当日 pb 非空且 pb > 0」的股票（pb<=0 是净资产为负，无法判破净，剔除）。
在此之上给三种 Universe：

  A  all      全 A（含北交所、含 ST）
  B  no_bj    剔除北交所（.BJ）
  C  clean    剔除北交所 + 剔除 ST/退市整理（name 含 ST/*ST/退）

🔴 为什么必须多口径并列：破净率对 Universe 极其敏感。
   2024-09-30 实测全A 与 clean 口径能差出好几个百分点；视频/研报引用"破净率 >12%"
   若口径不同就不可比。**引用任何破净率阈值前必须先锁定口径**（同换手率教训）。

用法
----
  python net_break_rate.py              # 构建全序列（首次约 20-40 秒）
  python net_break_rate.py --check      # 只核对锚点日，不落盘
  python net_break_rate.py --reuse      # 读已缓存 CSV
"""
import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from config import DATA  # noqa: E402

OUT_DIR = os.path.join("data", "results", "net_break")
CACHE = os.path.join(OUT_DIR, "market_net_break.csv")

START = "20100101"   # daily_basic 起点
END = "20260831"


def get_conn():
    return __import__("sqlite3").connect(DATA["local_db_path"])


def build_series():
    """按日聚合三种口径的破净率。一次全表 GROUP BY（1320 万行，约 20-40 秒）。"""
    con = get_conn()
    print("[破净率] 聚合 daily_basic 全表（约需 20-40 秒）...")
    q = """
    SELECT b.trade_date,
           COUNT(*)                                                   AS n_all,
           SUM(CASE WHEN b.pb > 0 AND b.pb < 1 THEN 1 ELSE 0 END)     AS k_all,
           SUM(CASE WHEN substr(b.ts_code, -2) <> 'BJ' THEN 1 ELSE 0 END) AS n_nobj,
           SUM(CASE WHEN substr(b.ts_code, -2) <> 'BJ'
                     AND b.pb > 0 AND b.pb < 1 THEN 1 ELSE 0 END)     AS k_nobj
    FROM daily_basic b
    WHERE b.trade_date >= ? AND b.trade_date <= ?
      AND b.pb IS NOT NULL AND b.pb > 0
    GROUP BY b.trade_date
    ORDER BY b.trade_date
    """
    df = pd.read_sql_query(q, con, params=(START, END))
    con.close()

    df["rate_all"] = df["k_all"] / df["n_all"]
    df["rate_nobj"] = df["k_nobj"] / df["n_nobj"]

    # 口径 C：再剔除 ST/退市整理（需要 stock_basic.name）
    con = get_conn()
    names = pd.read_sql_query("SELECT ts_code, name FROM stock_basic", con)
    con.close()
    st_codes = set(names.loc[
        names["name"].str.contains("ST|退", na=False, regex=True), "ts_code"])
    print(f"[破净率] ST/退市整理标的 {len(st_codes)} 只（用于口径 C）")

    # 口径 C 需要逐日明细（ST 名单是静态表，无法在 SQL 里 join 历史状态），
    # 单独拉一次非北交所 + 有效 pb 的 (date, code, pb) 明细，在 pandas 侧剔除 ST。
    con = get_conn()
    q3 = """
    SELECT b.trade_date, b.ts_code, b.pb
    FROM daily_basic b
    WHERE b.trade_date >= ? AND b.trade_date <= ?
      AND b.pb IS NOT NULL AND b.pb > 0
      AND substr(b.ts_code, -2) <> 'BJ'
    """
    det = pd.read_sql_query(q3, con, params=(START, END))
    con.close()
    det["is_st"] = det["ts_code"].isin(st_codes)
    clean = det[~det["is_st"]]
    gc = clean.groupby("trade_date").apply(
        lambda s: pd.Series({"n_clean": len(s),
                             "k_clean": int((s["pb"] < 1).sum())}),
        include_groups=False).reset_index()
    gc["rate_clean"] = gc["k_clean"] / gc["n_clean"]

    df = df.merge(gc[["trade_date", "n_clean", "k_clean", "rate_clean"]],
                  on="trade_date", how="left")

    # 分位（各自滚动窗口，仅用于后续择时；窗口内自比，无前视）
    for col, out in [("rate_all", "pct_all"), ("rate_nobj", "pct_nobj"),
                     ("rate_clean", "pct_clean")]:
        df[out] = df[col].rolling(750, min_periods=250).apply(
            lambda w: (w[-1] >= w[:-1]).mean() * 100, raw=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(CACHE, index=False, encoding="utf-8-sig")
    print(f"[破净率] 已保存 {CACHE}  ({len(df)} 行)")
    return df


def check_anchors(df):
    """核对复现阶段算过的三个锚点，确认口径没漂移。"""
    print("\n=== 锚点核对（与上一轮复现阶段比对） ===")
    print("上一轮记录：2018底 11.91% / 2021.02峰值 10.21% / 2024.09均值 14.37%(峰值16.09%)")
    print(f"{'日期附近':<14}{'全A':>9}{'剔北交所':>10}{'剔BJ+ST':>10}{'样本数':>9}")
    for d in ["20181228", "20210210", "20240930"]:
        sub = df[df["trade_date"] <= d].tail(1)
        if len(sub) == 0:
            continue
        r = sub.iloc[0]
        print(f"{r['trade_date']:<14}{r['rate_all']*100:>8.2f}%"
              f"{r['rate_nobj']*100:>9.2f}%{r['rate_clean']*100:>9.2f}%"
              f"{int(r['n_all']):>9}")

    # 2021.02 峰值窗口
    w = df[(df["trade_date"] >= "20210101") & (df["trade_date"] <= "20210430")]
    if len(w):
        i = w["rate_all"].idxmax()
        print(f"\n2021 Q1 破净率峰值：{df.loc[i,'trade_date']} "
              f"全A {df.loc[i,'rate_all']*100:.2f}% / "
              f"剔BJ {df.loc[i,'rate_nobj']*100:.2f}% / "
              f"clean {df.loc[i,'rate_clean']*100:.2f}%")

    # 2024.09 区间均值
    w = df[(df["trade_date"] >= "20240901") & (df["trade_date"] <= "20241015")]
    if len(w):
        print(f"2024.09-10 破净率均值：全A {w['rate_all'].mean()*100:.2f}% / "
              f"剔BJ {w['rate_nobj'].mean()*100:.2f}% / "
              f"clean {w['rate_clean'].mean()*100:.2f}%"
              f"  （峰值 全A {w['rate_all'].max()*100:.2f}%）")


def summary(df):
    print("\n=== 全序列概览（全A口径） ===")
    print(df["rate_all"].describe().round(4).to_string())
    hi = df[df["rate_all"] >= 0.10]
    print(f"\n破净率 >= 10% 的交易日：{len(hi)} 天（占比 {len(hi)/len(df)*100:.1f}%）")
    if len(hi):
        # 连续区间归并
        seg, cur = [], [hi.iloc[0]["trade_date"]]
        for d in hi["trade_date"].iloc[1:]:
            prev = df[df["trade_date"] == cur[-1]].index[0]
            now = df[df["trade_date"] == d].index[0]
            if now - prev <= 10:
                cur.append(d)
            else:
                seg.append((cur[0], cur[-1])); cur = [d]
        seg.append((cur[0], cur[-1]))
        print(f"归并成 {len(seg)} 个独立区间（间隔>10个交易日即断开）：")
        for a, b in seg:
            sub = df[(df["trade_date"] >= a) & (df["trade_date"] <= b)]
            print(f"  {a} ~ {b}  峰值 {sub['rate_all'].max()*100:.2f}%  "
                  f"持续 {len(sub)} 日")


def main():
    ap = argparse.ArgumentParser(description="全市场破净率日序列")
    ap.add_argument("--reuse", action="store_true", help="读已缓存 CSV")
    ap.add_argument("--check", action="store_true", help="只核对锚点，不落盘")
    a = ap.parse_args()

    if a.reuse and os.path.exists(CACHE):
        df = pd.read_csv(CACHE, dtype={"trade_date": str})
        print(f"[复用] {CACHE}  {len(df)} 行")
    else:
        df = build_series()

    check_anchors(df)
    summary(df)


if __name__ == "__main__":
    main()
