# -*- coding: utf-8 -*-
"""
⑩ 20日新高个股占比（BV1Lghg6NEA 第⑩条）—— 日频市场状态信号
=============================================================
视频规则：20日新高个股占比 **< 18%** = 历史大底（超卖极值 → 买点）。

定义（本脚本口径，先锁死）：
  - 个股当日"创20日新高" = close_t >= max(close 近20个交易日含当日)
  - 分母 = 当日有成交的个股数（含退市股历史，无幸存者偏差——直接扫 daily）
  - ⚠️ 剔除上市不足 min_hist(60) 交易日的次新股（否则新股上市首日必创新高，虚增占比）
  - 🔴 复权口径：新高判定用 **不复权 close**（创价格新高），与视频口径一致；
    复权价会在除权日人为制造假新高（分红除权 → 价格跳水 → 之后几天全"创新高"），
    这是本信号最重要的口径陷阱。

用法：
    python new_high20.py                      # 全量构建 data/results/new_high20/ratio20.csv
    python new_high20.py --window 20 --min-hist 60

输出列：trade_date, ratio(0~100), n(当日参与统计个股数)
"""
import argparse
import os
import sqlite3
import sys
import time

import numpy as np
import pandas as pd

DB_PATH = "D:/tu-shareData/astock_daily.db"
OUT_DIR = os.path.join("data", "results", "new_high20")


def build(window: int, min_hist: int) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    codes = [r[0] for r in conn.execute(
        "SELECT DISTINCT ts_code FROM daily ORDER BY ts_code")]
    print(f"涉及个股 {len(codes)} 只（daily 全量去重，含已退市）")
    assert len(codes) > 1000, f"样本数异常：{len(codes)}（为 0 就是没查到，先停）"

    counts: dict = {}   # trade_date -> [nh_count, n_count]
    t0 = time.time()
    for i, code in enumerate(codes):
        rows = conn.execute(
            "SELECT trade_date, close FROM daily WHERE ts_code=? ORDER BY trade_date",
            (code,)).fetchall()
        if len(rows) <= min_hist:
            continue
        dates = [r[0] for r in rows]
        close = np.asarray([r[1] for r in rows], dtype=float)
        # 次新剔除：前 min_hist 日不参与统计
        for k in range(min_hist, len(rows)):
            d = dates[k]
            c = close[k]
            if not np.isfinite(c):
                continue
            # 近 window 日含当日：close[k] >= max(close[k-window+1 .. k])
            hi = close[max(0, k - window + 1): k + 1].max()
            nh, n = counts.get(d, (0, 0))
            counts[d] = (nh + (1 if c >= hi else 0), n + 1)
        if (i + 1) % 1000 == 0:
            print(f"  ... {i+1}/{len(codes)}  耗时 {time.time()-t0:.0f}s")
    conn.close()

    df = pd.DataFrame(
        [(d, nh / n * 100, n) for d, (nh, n) in sorted(counts.items())],
        columns=["trade_date", "ratio", "n"])
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--min-hist", type=int, default=60)
    args = ap.parse_args()

    df = build(args.window, args.min_hist)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "ratio20.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n共 {len(df)} 个交易日  {df.trade_date.min()} ~ {df.trade_date.max()}")
    print(f"日均参与统计个股 {int(df['n'].mean())} 只")
    print(f"ratio 分布：min={df.ratio.min():.2f}  p10={df.ratio.quantile(.1):.2f}  "
          f"中位={df.ratio.median():.2f}  p90={df.ratio.quantile(.9):.2f}  max={df.ratio.max():.2f}")
    print(f"ratio<18% 天数：{(df.ratio < 18).sum()}")
    print(f"已写出 {out}")

    # 自证：最近 5 日
    print("\n最近 5 日：")
    print(df.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
