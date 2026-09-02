# -*- coding: utf-8 -*-
"""
④ M1拐点领先大顶 / ⑲ PPI-CPI轮动 —— 六道闸门（宏观月度信号）
================================================================
视频主张：
  ④ M1 增速拐点领先市场大顶（M1 拐头向下 → 减仓信号；拐头向上 → 加仓）
  ⑲ PPI/CPI 轮动顺序决定仓位/风格（PPI-CPI 剪刀差方向 = 周期位置）

方法：
  - PIT：月度宏观数据 month m 约在 m+1 月 15 日前后发布 → 信号可用日 = m+1 月 15 日
    （保守取 15 日，且严格月度对齐，绝不前视）
  - 信号六口径：M1yoy 水平分位 / M1yoy 3月拐点 / M1-M2 剪刀差 /
                PPI-CPI 剪刀差水平分位 / 剪刀差 3月拐点 / PPI 3月拐点
  - Gate 1：时序 Spearman（信号 vs 未来 20/60 日收益，市场 000906 全收益 + value NAV hfq）
            NW t（lag=H 重叠校正）；分年 IC
  - 事件对账：市场大顶/大底 日 vs M1/PPI 拐点月，领先滞后月数表（视频核心主张直接对账）
  - Gate 6 overlay：规则化择时（满仓/持币）vs 恒持，含切换成本 0.3%/次（加仓0.125%+减仓0.175%）
用法：python m1_ppi_cpi_timing.py
"""
import sys
import sqlite3

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, ".")
from bench_index import load_benchmark

DB = "D:/tu-shareData/astock_daily.db"
NAV = "data/results/value_strategy/backtest_result_hfq_20100104_20260831.csv"
BENCH = "000906.SH"
SWITCH_COST = 0.003  # 一次切换 = 加仓0.125% + 减仓0.175%（与 dividend_bear_excess 同款）


def nw_t(x, y, lag):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 60:
        return np.nan, n
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    e = y - X @ beta
    u = X * e[:, None]
    S = u.T @ u
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)
        S += w * (u[k:].T @ u[: len(u) - k] + u[: len(u) - k].T @ u[k:])
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(V[1, 1])
    return (beta[1] / se if se > 0 else np.nan), n


def month_shift(m, k):
    y, mm = int(m[:4]), int(m[4:])
    tot = y * 12 + mm - 1 + k
    return f"{tot // 12:04d}{tot % 12 + 1:02d}"


def load_macro(conn):
    """宏观月度表（TEXT 存储数值 → 强制 float），只取 2005 后（2010 起参与回测）。"""
    m = pd.read_sql("SELECT month, m1_yoy, m2_yoy FROM cn_m WHERE month>='200501'", conn)
    cpi = pd.read_sql("SELECT month, nt_yoy FROM cn_cpi WHERE month>='200501'", conn)
    ppi = pd.read_sql("SELECT month, ppi_yoy FROM cn_ppi WHERE month>='200501'", conn)
    for df in (m, cpi, ppi):
        for c in df.columns:
            if c != "month":
                df[c] = pd.to_numeric(df[c], errors="coerce")
    df = m.merge(cpi, on="month").merge(ppi, on="month").sort_values("month").reset_index(drop=True)
    df["m1m2"] = df.m1_yoy - df.m2_yoy
    df["ppicpi"] = df.ppi_yoy - df.nt_yoy
    assert len(df) > 200, f"宏观样本不足: {len(df)}"
    return df


def to_daily(macro, nav_dates):
    """PIT 展开：month m → 可用日 = (m+1)月15日起首个交易日 → step 前向填充。"""
    mac = macro.copy()
    mac["avail"] = pd.to_datetime(mac.month.map(lambda x: month_shift(x, 1) + "15"))
    idx = pd.DataFrame({"date": pd.to_datetime(nav_dates)}).sort_values("date")
    out = pd.merge_asof(idx, mac, left_on="date", right_on="avail", direction="backward")
    out = out.drop(columns=["avail"])
    for c in ["m1_yoy", "m2_yoy", "m1m2", "ppicpi", "ppi_yoy", "nt_yoy"]:
        out[c] = out[c].ffill()
    return out


def rolling_pct(s, win=60):
    """60 个月滚动分位（0~100）。"""
    return s.rolling(win, min_periods=36).apply(
        lambda w: (w[-1] >= w[:-1]).mean() * 100, raw=True)


def main():
    conn = sqlite3.connect(DB)
    macro = load_macro(conn)
    print(f"宏观样本：{len(macro)} 个月  {macro.month.min()}~{macro.month.max()} "
          f"(M1缺失{macro.m1_yoy.isna().sum()} / PPI缺失{macro.ppi_yoy.isna().sum()})")

    nav = pd.read_csv(NAV, dtype={"trade_date": str})[["trade_date", "portfolio_value_full"]]
    bench, meta = load_benchmark(BENCH, nav.trade_date.min(), nav.trade_date.max(),
                                 conn=conn, nav_price_mode="hfq")
    assert bench is not None and len(bench) > 3000, "基准加载失败"
    print(f"基准：{meta['resolved_code']}  {len(bench)} 日")

    df = nav.merge(bench[["trade_date", "close"]].rename(columns={"close": "bench"}),
                   on="trade_date").sort_values("trade_date").reset_index(drop=True)
    daily = to_daily(macro, df.trade_date)
    for c in ["m1_yoy", "m2_yoy", "m1m2", "ppicpi", "ppi_yoy", "nt_yoy"]:
        df[c] = daily[c].values

    # 六信号
    df["s_m1_pct"] = rolling_pct(df.m1_yoy)
    df["s_m1_turn"] = df.m1_yoy.diff(3)
    df["s_m1m2"] = df.m1m2
    df["s_ppicpi_pct"] = rolling_pct(df.ppicpi)
    df["s_ppicpi_turn"] = df.ppicpi.diff(3)
    df["s_ppi_turn"] = df.ppi_yoy.diff(3)

    SIGS = [("s_m1_pct", "M1yoy 60月分位", "+"),
            ("s_m1_turn", "M1yoy 3月拐点", "+"),
            ("s_m1m2", "M1-M2 剪刀差", "+"),
            ("s_ppicpi_pct", "PPI-CPI 60月分位", "+"),
            ("s_ppicpi_turn", "PPI-CPI 3月拐点", "+"),
            ("s_ppi_turn", "PPI 3月拐点", "+")]
    print(f"\n合并样本 {len(df)} 日  {df.trade_date.min()}~{df.trade_date.max()}")
    print("\n=== Gate 1 时序 Spearman（视频方向：信号高→未来涨，IC 应为正）===")
    for col, name, _ in SIGS:
        line = f"  {name:16s}"
        for target, px in [("市场", "bench"), ("value", "portfolio_value_full")]:
            for H in [20, 60]:
                fwd = df[px].shift(-H) / df[px] - 1
                ok = np.isfinite(df[col]) & np.isfinite(fwd)
                rho, p = stats.spearmanr(df[col][ok], fwd[ok])
                t, n = nw_t(df[col][ok], fwd[ok], lag=H)
                line += f"  {target}H{H}: {rho:+.3f}/t{t:+.1f}"
        print(line)

    print("\n=== 分年 IC（市场 H=20，s_m1_turn 与 s_ppicpi_turn）===")
    for col, name, _ in SIGS[1:2] + SIGS[4:5]:
        fwd = df.bench.shift(-20) / df.bench - 1
        tmp = pd.DataFrame({"year": df.trade_date.str[:4], "x": df[col], "f": fwd}).dropna()
        ics = {y: stats.spearmanr(s.x, s.f)[0] for y, s in tmp.groupby("year") if len(s) > 100}
        pos = sum(1 for v in ics.values() if v > 0)
        print(f"  {name:16s} 正IC年数 {pos}/{len(ics)}  " +
              " ".join(f"{y}:{v:+.2f}" for y, v in list(ics.items())))

    # ---- 事件对账：市场大顶/大底 vs M1/PPI 拐点 ----
    print("\n=== 事件对账：市场大顶/大底 vs 宏观拐点（视频核心主张：拐点领先大顶）===")
    idx = pd.read_sql("SELECT trade_date, close FROM index_daily WHERE ts_code='000906.SH' "
                      "AND trade_date>='20100101'", conn)
    idx["close"] = pd.to_numeric(idx.close, errors="coerce")
    m_dates = {m: pd.to_datetime(month_shift(m, 1) + "15") for m in macro.month}
    tops = [("2015-06-12", "顶"), ("2018-01-29", "顶"), ("2021-02-18", "顶"),
            ("2016-01-28", "底"), ("2019-01-04", "底"), ("2024-02-05", "底"),
            ("2024-09-18", "底前")]
    myy = macro.set_index("month").m1_yoy
    ppicpi = macro.set_index("month").ppicpi

    def local_peak(series, month):
        """month 前后 3 个月内 series 最大值所在月（拐点定位）。"""
        m2 = month_shift(month, -3)
        win = series[(series.index >= m2) & (series.index <= month_shift(month, 3))].dropna()
        return win.idxmax() if len(win) else None

    def lag_months(top_month, sig_month):
        if sig_month is None:
            return None
        return (int(sig_month[:4]) * 12 + int(sig_month[4:])) - \
               (int(top_month[:4]) * 12 + int(top_month[4:]))

    for d, tag in tops:
        dm = pd.to_datetime(d).strftime("%Y%m")
        pk1 = local_peak(myy, dm)
        pk2 = local_peak(ppicpi, dm)
        l1, l2 = lag_months(dm, pk1), lag_months(dm, pk2)
        f1 = f"{pk1}({l1:+d}月)" if pk1 else "—"
        f2 = f"{pk2}({l2:+d}月)" if pk2 else "—"
        print(f"  {d} {tag}   M1yoy拐点: {f1:22s}  PPI-CPI拐点: {f2}")
    print("  （正数=宏观拐点在市场拐点之后；视频主张应为负数且提前 3~6 月）")

    # ---- Gate 6 overlay ----
    print(f"\n=== Gate 6 overlay：规则化择时 vs 恒持（切换成本 {SWITCH_COST*100:.2f}%/次）===")

    def overlay(sig, label, px, px_name):
        s = (sig > 0).astype(float).shift(1).fillna(0)  # 当日收盘定仓、次日生效
        r = df[px].pct_change().fillna(0)
        sw = s.diff().abs().fillna(0)
        port = (1 + s * r - sw * SWITCH_COST).cumprod()
        hold = (1 + r).cumprod()
        yrs = len(port) / 252

        def perf(n):
            ann = n.iloc[-1] ** (1 / yrs) - 1
            mdd = (n / n.cummax() - 1).min()
            return ann * 100, mdd * 100
        a1, m1 = perf(port)
        a2, m2 = perf(hold)
        print(f"    {label:22s} [{px_name:9s}] 择时: {a1:+6.2f}%/年 回撤{m1:7.2f}% "
              f"(切{int(sw.sum())}次) | 恒持: {a2:+6.2f}%/年 回撤{m2:7.2f}%")

    rules = [(df.s_m1_turn, "R1 M1拐点向上满仓"),
             (df.s_m1_pct - 50, "R2 M1分位>50 满仓"),
             (df.s_m1m2, "R3 M1-M2>0 满仓"),
             (df.s_ppicpi_turn, "R4 剪刀差拐点向上"),
             (df.s_ppi_turn, "R5 PPI拐点向上"),
             (df.s_ppicpi_pct - 50, "R6 剪刀差分位>50")]
    for sig, label in rules:
        overlay(sig, label, "bench", "市场000906")
        overlay(sig, label, "portfolio_value_full", "valueNAV")

    conn.close()


if __name__ == "__main__":
    main()
