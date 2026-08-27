# -*- coding: utf-8 -*-
"""
run_chan_lun_standalone.py — 缠论量化「真·择时」独立回测（v2，公平检验版）
================================================================
动机：之前的 --consolidation-filter 把"布林带宽盘整探测器"贴"缠论G1"标签嫁接到既有选股上，
  既非测缠论择股也非测缠论择时，且已被 OOS 证伪（见 docs/consolidation_filter_oos_report.md）。
  本脚本才是"测缠论"的本意：用 chan_lun_core 的 分型→笔→线段→中枢→背驰→买卖点 几何结构，
  作为【独立择时信号】驱动单一标的进出场，与同标的买入持有(buy&hold)对照。

v2 相对 v1 的修正（让"回测说话"成立）：
  1. 剔除脏数据：510500(中证500ETF) 的 etf_daily 在 2015-04-15 有 +248.6% 单日跳变（早期价格被约5×缩错），
     其 BH 基准 +681% 是假象。改用干净的中证500指数(000905)/上证50指数(000016)补充。
  2. 策略必须真实交易：v1 入场门槛(b1=下跌末端+底背驰)太苛刻，十几年只交易2-4次、多数年份空仓，
     属无效检验。v2 主模式改为【线段顺势】——HH-HL 链向上才持仓(缠论最经典、且频繁交易的择时用法)，
     并保留【买卖点】模式(一/二/三买进场+顶背驰出场)作对照。
  3. 宽成本：单边 cost_rate(默认0.13%=佣金0.03%+滑点0.1%)，ETF 无印花税；每笔 round-trip ≈0.26%。
     指数标的非直接可交易，成本按同参数建模(已在输出注明)。可选 10% 硬止损。

方法学（诚实约束）：
  1. 无未来函数：信号定型需 n=2 根后确认，故对信号做 signal_lag=3 根滞后，严格只用定型信号；
     信号在 T-1 前已知，T 开盘执行。
  2. long-only（A股多数标的不可裸卖空），空仓(现金)为默认态。
  3. walk-forward：逐日历年拆解收益，暴露非稳健性（不重抽样、不偷看未来）。

运行（本机，venv_ml）：
  venv_ml/Scripts/python.exe run_chan_lun_standalone.py
  venv_ml/Scripts/python.exe run_chan_lun_standalone.py --mode swing --stop 0.10
  venv_ml/Scripts/python.exe run_chan_lun_standalone.py --instruments 510300.SH:etf_daily:沪深300ETF:20100101

诚实预期：缠论结构检测对，但方向不可知 → 择时大概率跑不赢 buy&hold（牛市踏空、熊市抄底被套、
  震荡市线段来回被打脸）。让数据说话，不预设立场。
"""
import os
import argparse
import sqlite3
import numpy as np
import pandas as pd

try:
    import config
    DB_PATH = getattr(config, "DB_PATH", None) or getattr(config, "DATA", None)
    if isinstance(DB_PATH, dict):
        DB_PATH = DB_PATH.get("local_db_path")
except Exception:
    DB_PATH = None
if not DB_PATH or not os.path.exists(DB_PATH):
    DB_PATH = r"D:\tu-shareData\astock_daily.db"

import chan_lun_core as CL


# 默认标的：(code, table, label, start) —— 均为已校验无价格断裂的干净标的
DEFAULT_INSTRUMENTS = [
    ("510300.SH", "etf_daily",   "沪深300ETF",     "20100101"),
    ("510050.SH", "etf_daily",   "上证50ETF",      "20100101"),
    ("159915.SZ", "etf_daily",   "创业板ETF",      "20110101"),
    ("000905.SH", "index_daily", "中证500指数(2005+)", "20050101"),
    ("000016.SH", "index_daily", "上证50指数(2005+)",  "20050101"),
]

# 已发现脏数据（剔除）：510500.SH etf_daily 2015-04-15 +248.6% 单日跳变（早期价格被约5×缩错）


def load_ohlc(code, table, start, end):
    c = sqlite3.connect(DB_PATH)
    rs = c.execute(
        f"SELECT CAST(trade_date AS TEXT), open, high, low, close FROM {table} "
        f"WHERE ts_code=? AND trade_date BETWEEN ? AND ? ORDER BY trade_date",
        (code, start, end)).fetchall()
    c.close()
    if not rs:
        return None
    df = pd.DataFrame(rs, columns=["d", "o", "h", "l", "c"])
    for col in ["o", "h", "l", "c"]:
        df[col] = df[col].ffill()
    if df["c"].isna().any():
        df = df.dropna(subset=["o", "h", "l", "c"])
    # 数据质量护栏：剔除单日收益绝对值 > 25% 的断裂（仅报告，不静默修正）
    cl = df["c"].values.astype(float)
    bad = np.where(np.abs(np.diff(cl) / cl[:-1]) > 0.25)[0]
    if len(bad) > 0:
        print(f"  ⚠️ {code} 检测到 {len(bad)} 处单日>25%断裂(如 {df['d'].iloc[bad[0]+1]})，数据质量存疑，跳过")
        return None
    return df


def compute_signals(highs, lows, closes, params):
    """返回 (trend_up[], buy_trig[], sell_trig[]) 三者均为 len=n 的布尔数组（已做实滞后，无未来函数）。"""
    n = len(closes)
    lag = params["signal_lag"]
    win = params["signal_window"]
    st = CL.compute_states(
        np.asarray(highs, float), np.asarray(lows, float), np.asarray(closes, float),
        zig_th=params["zig_th"],
        consol=dict(win=20, lookback=120, th=params["consol_th"]),
        div_lookback=params["div_lookback"],
    )
    buys = set(i for i, _ in st["buy_points_typed"]) | set(st["bull_div"])
    sells = set(st["bear_div"])
    raw_buy = np.zeros(n, dtype=bool)
    raw_sell = np.zeros(n, dtype=bool)
    for i in buys:
        if 0 <= i < n:
            raw_buy[i] = True
    for i in sells:
        if 0 <= i < n:
            raw_sell[i] = True

    buy_trig = np.zeros(n, dtype=bool)
    sell_trig = np.zeros(n, dtype=bool)
    for i in range(n):
        hi = i - lag
        if hi < 0:
            continue
        lo = max(0, hi - win + 1)
        if raw_buy[lo:hi + 1].any():
            buy_trig[i] = True
        if raw_sell[lo:hi + 1].any():
            sell_trig[i] = True

    # 逐 bar 线段方向(HH-HL 链)：用截至 i-lag 的笔序列判定，避免未来函数
    piv = CL.zigzag(np.asarray(highs, float), np.asarray(lows, float), threshold=params["zig_th"])
    trend_up = np.zeros(n, dtype=bool)
    for i in range(n):
        cut = i - lag
        if cut < 0:
            continue
        sub = [p for p in piv if p["i"] <= cut]
        if len(sub) >= 4:
            trend_up[i] = (CL.swing_trend(sub) == "up")
    return trend_up, buy_trig, sell_trig


def run_strategy(dates, opens, closes, trend_up, buy_trig, sell_trig,
                 cost_rate, stop, mode, init=1.0):
    n = len(closes)
    pos, entry, cash, shares = 0.0, 0.0, init, 0.0
    nav = np.empty(n)
    trades = []
    rts = []
    for i in range(n):
        px = opens[i]
        want_long = trend_up[i] if mode == "swing" else buy_trig[i]
        exit_now = False
        exit_px = px
        if pos == 1:
            if mode == "swing":
                if not trend_up[i]:
                    exit_now = True
            else:
                if sell_trig[i]:
                    exit_now = True
            if stop > 0 and closes[i] <= entry * (1 - stop):
                exit_now = True
                exit_px = closes[i]
        if pos == 0 and want_long:
            shares = (cash * (1 - cost_rate)) / px
            cash = 0.0
            pos = 1.0
            entry = px
            trades.append(("B", dates[i], px))
        elif pos == 1 and exit_now:
            proceed = shares * exit_px * (1 - cost_rate)
            trades.append(("S", dates[i], exit_px))
            if entry > 0:
                rts.append(proceed / (shares * entry / (1 - cost_rate)) - 1.0)
            cash = proceed
            shares = 0.0
            pos = 0.0
            entry = 0.0
        nav[i] = cash + shares * closes[i]
    return nav, trades, rts


def buyhold_nav(opens, closes, cost_rate, init=1.0):
    n = len(closes)
    shares = (init * (1 - cost_rate)) / opens[0]
    nav = shares * closes.copy()
    nav[-1] = shares * closes[-1] * (1 - cost_rate)
    return nav


def nav_stats(nav):
    nav = np.asarray(nav, float)
    rets = nav[1:] / nav[:-1] - 1.0
    rets = rets[np.isfinite(rets)]
    total = nav[-1] / nav[0] - 1.0
    n_years = max(1.0, len(nav) / 252.0)
    annual = (nav[-1] / nav[0]) ** (1.0 / n_years) - 1.0
    peak = np.maximum.accumulate(nav)
    mdd = float((nav / peak - 1.0).min())
    sd = rets.std()
    sharpe = float(rets.mean() / sd * np.sqrt(252)) if sd > 1e-12 else 0.0
    return dict(total=total, annual=annual, mdd=mdd, sharpe=sharpe)


def yearly_returns(dates, nav):
    by_year = {}
    for d, v in zip(dates, nav):
        by_year.setdefault(d[:4], []).append(v)
    return {y: vs[-1] / vs[0] - 1.0 for y, vs in sorted(by_year.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20100101")
    ap.add_argument("--end", default="20251231")
    ap.add_argument("--instruments", default="", help="逗号分隔 code:table:label:start")
    ap.add_argument("--mode", default="swing", choices=["swing", "points", "both"],
                    help="swing=线段顺势(主) | points=买卖点反转 | both=两者都跑")
    ap.add_argument("--zig-th", type=float, default=0.05)
    ap.add_argument("--consol-th", type=float, default=0.25)
    ap.add_argument("--div-lookback", type=int, default=60)
    ap.add_argument("--signal-lag", type=int, default=3)
    ap.add_argument("--signal-window", type=int, default=10)
    ap.add_argument("--cost", type=float, default=0.0013)
    ap.add_argument("--stop", type=float, default=0.0, help="保护性止损(0=不用), 如0.10")
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()

    if a.instruments:
        instrs = []
        for spec in a.instruments.split(","):
            p = spec.split(":")
            instrs.append((p[0], p[1] if len(p) > 1 else "etf_daily",
                           p[2] if len(p) > 2 else p[0], p[3] if len(p) > 3 else a.start))
    else:
        instrs = DEFAULT_INSTRUMENTS

    modes = ["swing", "points"] if a.mode == "both" else [a.mode]
    params = dict(zig_th=a.zig_th, consol_th=a.consol_th, div_lookback=a.div_lookback,
                  signal_lag=a.signal_lag, signal_window=a.signal_window)
    stop_label = ("stop%.0f%%" % (a.stop * 100)) if a.stop > 0 else "nostop"

    for mode in modes:
        print("=" * 100)
        print("  缠论独立择时回测 | 模式=%s | %s~%s | zig_th=%.2f div_lb=%d lag=%d win=%d cost=%.2f%% %s"
              % (mode, a.start, a.end, a.zig_th, a.div_lookback, a.signal_lag, a.signal_window,
                 a.cost * 100, stop_label))
        print("  %s | long-only，空仓为默认态，无未来函数(T-1定型,T开盘执行)"
              % ("线段顺势(HH-HL向上持仓)" if mode == "swing" else "买卖点(一/二/三买进·顶背驰出)"))
        print("=" * 100)
        print("  %-20s%9s%9s%9s%8s%7s | %9s%9s%10s%7s" %
              ("标的", "策略总", "策略年", "策略MDD", "夏普", "笔数", "BH总", "BH年", "超额", "胜率"))
        print("  " + "-" * 96)

        summary = []
        yearly_all = {}
        for code, table, label, start in instrs:
            s = max(start, a.start)
            df = load_ohlc(code, table, s, a.end)
            if df is None or len(df) < 120:
                print("  %-20s 数据不足/脏，跳过" % label)
                continue
            dates = df["d"].tolist()
            opens = df["o"].values.astype(float)
            highs = df["h"].values.astype(float)
            lows = df["l"].values.astype(float)
            closes = df["c"].values.astype(float)

            trend_up, buy_trig, sell_trig = compute_signals(highs, lows, closes, params)
            nav, trades, rts = run_strategy(dates, opens, closes, trend_up, buy_trig, sell_trig,
                                            a.cost, a.stop, mode)
            bh = buyhold_nav(opens, closes, a.cost)
            sc = nav_stats(nav)
            sb = nav_stats(bh)
            n_trades = len([t for t in trades if t[0] == "B"])
            win = (np.mean([r > 0 for r in rts]) if rts else 0.0)
            excess = sc["total"] - sb["total"]
            print("  %-20s%+8.2f%%%+8.2f%%%+8.2f%%%7.2f%6d | %+8.2f%%%+8.2f%%%+9.2fpp%6.1f%%"
                  % (label, sc["total"] * 100, sc["annual"] * 100, sc["mdd"] * 100, sc["sharpe"],
                     n_trades, sb["total"] * 100, sb["annual"] * 100, excess * 100, win * 100))
            summary.append(dict(label=label, code=code, mode=mode, **sc,
                               bh_total=sb["total"], bh_annual=sb["annual"], excess=excess,
                               trades=n_trades, win=win, years=len(dates) / 252.0))
            yearly_all[label] = yearly_returns(dates, nav)
            yearly_all[label + "_BH"] = yearly_returns(dates, bh)

        print("\n  ── 年度收益拆解 (缠论择时 vs buy&hold) ──")
        all_years = sorted({y for v in yearly_all.values() for y in v})
        print("  %-20s" % "标的", end="")
        for y in all_years:
            print("%7s" % y[2:], end="")
        print()
        for label in [x["label"] for x in summary]:
            print("  %-20s" % (label + " CL"), end="")
            for y in all_years:
                v = yearly_all[label].get(y)
                print("%7.1f%%" % (v * 100) if v is not None else "   n/a", end="")
            print()
            print("  %-20s" % (label + " BH"), end="")
            for y in all_years:
                v = yearly_all[label + "_BH"].get(y)
                print("%7.1f%%" % (v * 100) if v is not None else "   n/a", end="")
            print()

        print("\n  ── 诚实判定（不预设立场）──")
        better = [s for s in summary if s["excess"] > 0]
        worse = [s for s in summary if s["excess"] <= 0]
        print("  模式[%s] 择时跑赢 buy&hold：%d/%d；跑输：%d/%d"
              % (mode, len(better), len(summary), len(worse), len(summary)))
        if better:
            print("   跑赢：", ", ".join("%s %+.1fpp(年%.1f%%/MDD%.0f%%)" %
                  (s["label"], s["excess"] * 100, s["annual"] * 100, s["mdd"] * 100) for s in better))
        if worse:
            print("   跑输：", ", ".join("%s %+.1fpp(年%.1f%%/MDD%.0f%%)" %
                  (s["label"], s["excess"] * 100, s["annual"] * 100, s["mdd"] * 100) for s in worse))
        if not a.no_save:
            out = "data/results/chan_lun"
            os.makedirs(out, exist_ok=True)
            pd.DataFrame(summary).to_csv(
                "%s/chan_lun_standalone_%s_%s_%s_%s.csv" % (out, mode, a.start, a.end, stop_label),
                index=False)
            print("\n  CSV 已保存")


if __name__ == "__main__":
    main()
