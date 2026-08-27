#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
首板低开均值回归回测 (Phase A: 日线近似, 无分钟线)
=================================================
复刻尽职调查: B站 BV1KpGw6JE8E (UP: 上班做量化的鳄鱼)
视频声称: 首板低开策略 3年355% / 最大回撤14% / 年化约66%。

本脚本用平台真实数据独立验证其真伪, 口径严格对齐前期已拍板决策:
  - 涨停判定: 收封板 close>=pre_close*limit_up_ratio (板级±10%/±20%,
              复刻 small_cap_rotation_selector.limit_up_ratio)
  - 首板期限: N=20 (T-1涨停 且 T-2..T-N 全非涨停); 对照 N=1
  - 位置过滤: T-1 close 在 [T-60,T-1] 高低区间位置 < 0.5
  - 低开买入: T日 open 相对 T-1 close 低开 [lo,hi)
  - 出场三变体(日线代理):
      A(主): T+1 开盘盈利(open>buy)即开盘卖, 否则收盘卖  ← 还原视频"早卖锁利"
      B(对照): T+1 收盘无脑卖  ← 剥离出场择时贡献
      C(对照): T+1 开盘无脑卖
  - 宽成本: 佣金万3 + 滑点千3(涨停薄流动性) + 印花(2023-08-28前千1/后千0.5)
  - 退市处理: T+1 无数据 → 以买入价清算(保守近似, 真实退市更差)
  - 前视自检: 所有决策仅用 T-1 及更早数据 + T日 open(已知), 无 T日 close/high 泄露

输出: 控制台表 + data/results/limitup_reversion/*.csv
"""
import sqlite3, os, sys, argparse
from collections import defaultdict
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
DB = r"D:\tu-shareData\astock_daily.db"
EPS = 1e-9
INIT_CAPITAL = 1_000_000.0
RESULT_DIR = os.path.join(BASE, "data", "results", "limitup_reversion")

# ---------- 复刻自 src/small_cap_rotation_selector.limit_up_ratio (板级区分) ----------
def _prefix_of(ts_code):
    return ts_code[:3] if len(ts_code) >= 3 else ts_code

def limit_up_ratio(ts_code, trade_date):
    p = _prefix_of(ts_code)
    if p in ("300", "301"):
        return 1.10 if trade_date < "20200824" else 1.20
    if p == "688":
        return 1.20
    return 1.10

# ---------- 宽成本 ----------
COMM = 0.0003
SLIP = 0.003
STAMP_CUT = "20230828"
def _stamp(d):
    return 0.001 if d < STAMP_CUT else 0.0005
def buy_fee(p, sh, d):
    return p * sh * (COMM + SLIP)
def sell_fee(p, sh, d):
    return p * sh * (COMM + SLIP + _stamp(d))

# ---------- 数据 ----------
def load_daily(start="20140101"):
    con = sqlite3.connect(DB)
    df = pd.read_sql_query(
        f"SELECT ts_code,trade_date,open,high,low,close,pre_close FROM daily WHERE trade_date>='{start}'",
        con)
    con.close()
    for c in ("open", "high", "low", "close", "pre_close"):
        df[c] = df[c].astype(float)
    df["trade_date"] = df["trade_date"].astype(str)
    df["ts_code"] = df["ts_code"].astype(str)
    return df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

def build_events(df, N, lo, hi, pos_thr=0.5, exclude_prefix=("920",)):
    # 排除北交所(涨跌幅±30%不同, 且流动性差异大)
    mask_excl = df["ts_code"].str[:3].isin(exclude_prefix)
    df = df[~mask_excl].copy()
    g = df.groupby("ts_code", sort=False)
    # 板级涨停比例(向量化)
    p = df["ts_code"].str[:3]
    upr = np.where(p.isin(["300", "301"]),
                   np.where(df["trade_date"] < "20200824", 1.10, 1.20),
                   np.where(p == "688", 1.20, 1.10))
    df["upr"] = upr
    df["lu"] = df["close"] >= df["pre_close"] * df["upr"] - EPS
    # 首板: T-1涨停 且 前 N-1 天(到T-2)全非涨停
    # N=1 时只看 T-2(排除连板), 窗口取 max(N-1,1) 避免 rolling(0) 报错
    win = max(N - 1, 1)
    df["prev_lu_sum"] = g["lu"].transform(
        lambda s: s.shift(1).rolling(win, min_periods=win).sum())
    df["first_board"] = df["lu"] & (df["prev_lu_sum"] == 0)
    # 60天相对位置(窗口含T-1当日)
    df["min_low60"] = g["low"].transform(lambda s: s.rolling(60, min_periods=20).min())
    df["max_high60"] = g["high"].transform(lambda s: s.rolling(60, min_periods=20).max())
    rng = (df["max_high60"] - df["min_low60"]).replace(0, 1e-9)
    df["pos60"] = (df["close"] - df["min_low60"]) / rng
    # 移位取 T日 与 T+1日
    df["T_open"] = g["open"].shift(-1)
    df["T_pre"] = g["pre_close"].shift(-1)   # = T-1 close
    df["T_date"] = g["trade_date"].shift(-1)
    df["T1_open"] = g["open"].shift(-2)
    df["T1_close"] = g["close"].shift(-2)
    # 事件筛选(买入条件)
    ev = df[df["first_board"] & (df["pos60"] < pos_thr)].copy()
    low_open = (ev["T_pre"] - ev["T_open"]) / ev["T_pre"].replace(0, np.nan)
    ev = ev[(low_open >= lo) & (low_open < hi) & ev["T_open"].notna()].copy()
    ev["low_open"] = low_open
    events = ev[["ts_code", "T_date", "T_open", "T1_open", "T1_close"]].rename(
        columns={"T_date": "buy_date", "T_open": "buy_price"})
    events = events.dropna(subset=["buy_date", "buy_price"])
    return events

# ---------- 回测引擎 ----------
def backtest(events, exit_rule, start="20160101", end="20260826"):
    ev = events[(events["buy_date"] >= start) & (events["buy_date"] <= end)].copy()
    ev = ev.sort_values("buy_date").reset_index(drop=True)
    # 全交易日轴
    con = sqlite3.connect(DB)
    alld = [r[0] for r in con.execute(
        f"SELECT DISTINCT trade_date FROM daily WHERE trade_date>='{start}' AND trade_date<='{end}' ORDER BY trade_date").fetchall()]
    con.close()
    buys = defaultdict(list)
    for r in ev.itertuples():
        buys[r.buy_date].append(r)

    cash = INIT_CAPITAL
    holdings = []          # {ts,buy_date,buy_price,sh,T1_open,T1_close}
    nav = {}
    trades = []            # (ts,buy_date,sell_date,buy,sell,ret)
    prev_date = None
    for d in alld:
        # 1. 卖出: 持仓 buy_date == 上一交易日(prev_date)
        still = []
        for h in holdings:
            if h["buy_date"] == prev_date:
                T1o, T1c = h["T1_open"], h["T1_close"]
                if exit_rule == "A":
                    if pd.notna(T1o) and T1o > h["buy_price"]:
                        sp = T1o
                    elif pd.notna(T1c):
                        sp = T1c
                    else:
                        sp = h["buy_price"]
                elif exit_rule == "B":
                    sp = T1c if pd.notna(T1c) else h["buy_price"]
                else:  # C
                    sp = T1o if pd.notna(T1o) else (T1c if pd.notna(T1c) else h["buy_price"])
                proceeds = h["sh"] * sp - sell_fee(sp, h["sh"], d)
                cash += proceeds
                gross = (sp - h["buy_price"]) / h["buy_price"]
                net = (proceeds - h["sh"] * h["buy_price"]) / (h["sh"] * h["buy_price"])
                trades.append((h["ts"], h["buy_date"], d, h["buy_price"], sp, net))
            else:
                still.append(h)
        holdings = still
        # 2. 买入(当日候选, 等权, 最多3只)
        cands = buys.get(d, [])
        if cands:
            n = min(len(cands), 3)
            per = cash / n
            for r in cands[:3]:
                bp = r.buy_price
                sh = int(per / bp / 100) * 100
                if sh <= 0:
                    continue
                cost = sh * bp + buy_fee(bp, sh, d)
                if cost <= cash:
                    cash -= cost
                    holdings.append({"ts": r.ts_code, "buy_date": d, "buy_price": bp,
                                    "sh": sh, "T1_open": r.T1_open, "T1_close": r.T1_close})
        # 3. 净值(持仓按成本计价, 卖出日已结算)
        mv = sum(h["sh"] * h["buy_price"] for h in holdings)
        nav[d] = cash + mv
        prev_date = d

    return nav, trades

def metrics(nav, trades):
    dates = sorted(nav)
    arr = np.array([nav[d] for d in dates], dtype=float)
    if len(arr) < 2:
        return (0, 0, 0, 0, 0, 0, 0)
    rets = np.diff(arr) / arr[:-1]
    total = arr[-1] / arr[0] - 1
    years = len(dates) / 252.0
    annual = (arr[-1] / arr[0]) ** (1 / years) - 1 if years > 0 else 0
    peak = np.maximum.accumulate(arr)
    dd = arr / peak - 1
    mdd = dd.min()
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    if trades:
        tr = np.array([t[5] for t in trades])
        win = float((tr > 0).mean())
        per_mean = float(tr.mean())
        ntr = len(tr)
    else:
        win = per_mean = ntr = 0
    return (total, annual, mdd, sharpe, win, per_mean, ntr)

def annual_pnl(nav):
    rows = []
    cur = None
    for d in sorted(nav):
        y = d[:4]
        if y != cur:
            cur = y
        rows.append((y, d, nav[d]))
    df = pd.DataFrame(rows, columns=["year", "date", "nav"])
    out = []
    for y, g in df.groupby("year"):
        v0 = g["nav"].iloc[0]
        v1 = g["nav"].iloc[-1]
        out.append((y, (v1 / v0 - 1) * 100))
    return out

# ---------- HS300 基准 ----------
def hs300_buyhold(start="20160101", end="20260826"):
    con = sqlite3.connect(DB)
    df = pd.read_sql_query(
        f"SELECT trade_date,close FROM index_daily WHERE ts_code='000300.SH' AND trade_date>='{start}' AND trade_date<='{end}' ORDER BY trade_date",
        con)
    con.close()
    if len(df) == 0:
        return None
    arr = df["close"].astype(float).values
    return arr[-1] / arr[0] - 1, (arr[-1] / arr[0]) ** (1 / (len(arr) / 252.0)) - 1

# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20160101")
    ap.add_argument("--end", default="20260826")
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--lo", type=float, default=0.03)
    ap.add_argument("--hi", type=float, default=0.04)
    ap.add_argument("--pos", type=float, default=0.5)
    ap.add_argument("--exit", default="ALL", choices=["A", "B", "C", "ALL"])
    args = ap.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)
    print("=" * 70)
    print("首板低开均值回归 回测 (Phase A, 日线近似)")
    print(f"窗口 {args.start}~{args.end} | N={args.N} | 低开[{args.lo:.0%},{args.hi:.0%}) | 位置<{args.pos}")
    print("=" * 70)

    df = load_daily()
    print(f"[1/3] 加载日线: {len(df):,} 行")

    events = build_events(df, args.N, args.lo, args.hi, args.pos)
    print(f"[2/3] 首板低开事件: {len(events):,} 笔 | 时间跨度 "
          f"{events['buy_date'].min()}~{events['buy_date'].max()}")
    events.to_csv(os.path.join(RESULT_DIR, "events.csv"), index=False)

    print("[3/3] 回测...")
    rules = ["A", "B", "C"] if args.exit == "ALL" else [args.exit]
    res = {}
    for rule in rules:
        nav, trades = backtest(events, rule, args.start, args.end)
        m = metrics(nav, trades)
        res[rule] = (nav, trades, m)
        print(f"\n--- 出场 {rule} ---")
        print(f"  总收益 {m[0]*100:+.2f}% | 年化 {m[1]*100:+.2f}% | MDD {m[2]*100:+.2f}% "
              f"| 夏普 {m[3]:.2f} | 胜率 {m[4]*100:.1f}% | 笔均 {m[5]*100:+.2f}% | 笔数 {m[6]}")

    # 分年度(主跑 A)
    if "A" in res:
        navA = res["A"][0]
        print("\n--- 分年度收益 (出场 A) ---")
        for y, r in annual_pnl(navA):
            print(f"  {y}: {r:+.2f}%")

    # HS300 基准
    h = hs300_buyhold(args.start, args.end)
    if h:
        print(f"\n--- 基准 HS300 买入持有 ---\n  总收益 {h[0]*100:+.2f}% | 年化 {h[1]*100:+.2f}%")

    # 参数敏感性网格(主出场 A): N × 低开区间
    print("\n" + "=" * 70)
    print("参数敏感性网格 (出场 A, 位置<0.5)")
    print("=" * 70)
    grid = []
    for N in (1, 20):
        for (lo, hi) in ((0.02, 0.03), (0.03, 0.04), (0.04, 0.05), (0.02, 0.05)):
            evg = build_events(df, N, lo, hi, args.pos)
            nav, trades = backtest(evg, "A", args.start, args.end)
            m = metrics(nav, trades)
            grid.append((N, lo, hi, m))
            print(f"  N={N:>2} 低开[{lo:.0%},{hi:.0%}) | 总 {m[0]*100:+.2f}% "
                  f"年化 {m[1]*100:+.2f}% MDD {m[2]*100:+.2f}% 胜率 {m[4]*100:.1f}% "
                  f"笔均 {m[5]*100:+.2f}% 笔数 {m[6]}")

    # 保存
    pd.DataFrame(
        [(r, *res[r][2]) for r in res],
        columns=["exit", "total", "annual", "mdd", "sharpe", "win", "per_mean", "ntr"]
    ).to_csv(os.path.join(RESULT_DIR, "summary.csv"), index=False)
    pd.DataFrame(
        [(N, lo, hi, m[0], m[1], m[2], m[3], m[4], m[5], m[6]) for (N, lo, hi, m) in grid],
        columns=["N", "lo", "hi", "total", "annual", "mdd", "sharpe", "win", "per_mean", "ntr"]
    ).to_csv(os.path.join(RESULT_DIR, "grid.csv"), index=False)
    print(f"\n结果已保存至 {RESULT_DIR}")

if __name__ == "__main__":
    main()
