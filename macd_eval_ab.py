# -*- coding: utf-8 -*-
"""MACD 节奏门控 A/B：daily vs monthly vs quarterly 评估频率下的磨损(交易次数)与收益对照。
直接验证「视频样本3：横盘期反复金叉死叉磨损」是否在实盘数据中存在。

NAV 用独立复算（long/flat 在 adj_close 上，close-to-close），避免单只股票 adj_close 缩放下
框架现金记账的失真；交易次数取自真实插件（验证 gate 集成生效）。

用法（平台根目录）：
  venv_ml/Scripts/python.exe macd_eval_ab.py
  venv_ml/Scripts/python.exe macd_eval_ab.py --codes 600036.SH,000001.SZ --zero-line
"""
import argparse
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from backtest.macd_timing_plugin import MacdTimingPlugin

DB = r"D:/tu-shareData/astock_daily.db"


def load_df(code, start, end):
    conn = sqlite3.connect(DB)
    q = (
        "SELECT d.trade_date, d.open, d.close, COALESCE(a.adj_factor, 1.0) AS adj_factor "
        "FROM daily d "
        "LEFT JOIN adj_factor a ON a.ts_code = d.ts_code AND a.trade_date = d.trade_date "
        "WHERE d.ts_code=? AND d.trade_date BETWEEN ? AND ? "
        "ORDER BY d.trade_date"
    )
    df = pd.read_sql_query(q, conn, params=(code, int(start), int(end)))
    conn.close()
    if df.empty:
        return None
    df["adj_close"] = df["close"] * df["adj_factor"]
    return df


def compute_long_state(close, fast, slow, signal, zero_line):
    """与插件 _macd 同口径：DIF>DEA 且(可选)DIF>0 = 多头。返回逐日 long_state。"""
    ema_f = pd.Series(close).ewm(span=fast, adjust=False).mean().values
    ema_s = pd.Series(close).ewm(span=slow, adjust=False).mean().values
    dif = ema_f - ema_s
    dea = pd.Series(dif).ewm(span=signal, adjust=False).mean().values
    n = len(close)
    warm = slow + signal
    long_state = np.zeros(n, dtype=bool)
    prev = False
    for t in range(1, n):
        d, e = dif[t - 1], dea[t - 1]
        if t < warm or np.isnan(d) or np.isnan(e):
            long_state[t] = prev
            continue
        s = bool(d > e)
        if zero_line:
            s = s and (d > 0)
        long_state[t] = s
        prev = s
    return long_state


def clean_nav(close_adj, long_state, eval_mask):
    """long/flat NAV：边界日重估决策，非边界日沿用；close-to-close 近似。返回总收益率。"""
    n = len(close_adj)
    in_mkt = False
    nav = 1.0
    for i in range(1, n):
        if close_adj[i] <= 0 or close_adj[i - 1] <= 0:
            continue  # 停牌/脏数据日，净值不变
        ratio = close_adj[i] / close_adj[i - 1]
        if ratio < 0.5 or ratio > 2.0:
            continue  # 单日超±50%多为除权/拆细未对齐 adj_factor 的断点，跳过
        dec = long_state[i] if eval_mask[i] else in_mkt
        if dec and not in_mkt:
            in_mkt = True
        elif (not dec) and in_mkt:
            in_mkt = False
        if in_mkt:
            nav *= ratio
    return nav - 1.0


def run_one(code, freq, start, end, zero_line):
    df = load_df(code, start, end)
    if df is None or len(df) < 60:
        return None
    close = df["adj_close"].values.astype(float)
    long_state = compute_long_state(close, 12, 26, 9, zero_line)
    mask = MacdTimingPlugin._build_eval_mask(
        df["trade_date"].astype(int).tolist(), freq
    )
    nav_ret = clean_nav(close, long_state, mask)
    # 交易次数用真实插件（验证 gate 集成生效）
    p = MacdTimingPlugin(
        capital=1_000_000, cfg={"eval_freq": freq, "zero_line": zero_line}
    )
    res = p.run(df, start_idx=0)
    trades = len(res["trades"])
    return {
        "freq": freq,
        "trades": trades,
        "rounds": trades // 2,
        "nav_ret": nav_ret,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--codes",
        default="600036.SH,000001.SZ,300750.SZ,002594.SZ,000858.SZ,600519.SH",
    )
    ap.add_argument("--start", default="20190101")
    ap.add_argument("--end", default="20260815")
    ap.add_argument("--zero-line", action="store_true", help="开启零轴过滤(DIF>0)")
    args = ap.parse_args()
    codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    freqs = ["daily", "monthly", "quarterly"]
    print(f"区间 {args.start}~{args.end}  zero_line={args.zero_line}")
    print(f"{'code':<12}{'freq':<11}{'trades':>8}{'rounds':>8}{'NAV_ret':>12}")
    print("-" * 56)
    for code in codes:
        for freq in freqs:
            r = run_one(code, freq, args.start, args.end, args.zero_line)
            if r is None:
                print(f"{code:<12}{freq:<11}{'N/A':>8}")
                continue
            print(
                f"{code:<12}{freq:<11}{r['trades']:>8}{r['rounds']:>8}"
                f"{r['nav_ret'] * 100:>11.1f}%"
            )
        print()


if __name__ == "__main__":
    main()
