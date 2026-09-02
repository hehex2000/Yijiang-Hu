# -*- coding: utf-8 -*-
"""
破净率 overlay 择时回测（在中证800 上先验证频率与阈值，再上真实 value 引擎）

为什么先在指数上跑一遍
----------------------
信号检验（net_break_test.py）给的是 IC，但 IC 有两个盲区：
  1) 日频采样 + 250 日 forward return → 相邻样本重叠 99.6%，t 值仍可能虚高；
  2) IC 只测"排序能力"，不测"仓位映射后扣完成本还剩多少"。
所以必须补一层**真实仓位回测**：按信号给仓位、按调仓频率换仓、扣组合层成本。

方向
----
破净率**高** = 市场便宜 = 未来收益高 → **满仓**（与"PB 分位低→满仓"是同一逻辑的镜像）。

用法
----
  python net_break_overlay.py
"""
import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import bench_index  # noqa: E402
import broker_pe_factor as bpf  # noqa: E402
from broker_pe_redundancy import rolling_residual, timing_backtest, buy_hold_stats  # noqa: E402

NB = os.path.join("data", "results", "net_break", "market_net_break.csv")
MKT = os.path.join("data", "results", "broker_pe", "market_daily.csv")
OUT_DIR = os.path.join("data", "results", "net_break")

BENCH = "000906.SH"
START = "20100101"
END = "20260831"


def load():
    nb = pd.read_csv(NB, dtype={"trade_date": str})
    con = bpf.get_conn()
    bdf, meta = bench_index.load_benchmark(BENCH, START, END, conn=con,
                                           nav_price_mode="hfq")
    con.close()
    print(f"[基准] {BENCH} mode={meta.get('mode')} source={meta.get('source_table')}")
    b = bdf[["trade_date", "close"]].rename(columns={"close": "bench_close"})
    b["trade_date"] = b["trade_date"].astype(str)
    df = nb.merge(b, on="trade_date", how="inner").sort_values("trade_date")
    df["mkt_ret"] = df["bench_close"].pct_change()
    try:
        m = pd.read_csv(MKT, dtype={"trade_date": str})
        df = df.merge(m[["trade_date", "pb_pct750"]], on="trade_date", how="left")
    except Exception as e:
        print(f"[警告] 未载入全市场PB序列: {e}")
    return df.reset_index(drop=True)


def position_from_pct(pct, lo, hi, invert=False):
    """分位 → 目标仓位。

    invert=False: 分位低 → 满仓（PB 类：分位低=便宜）
    invert=True : 分位高 → 满仓（破净率类：分位高=便宜）
    """
    if pct != pct:
        return np.nan
    if invert:
        pct = 100.0 - pct
    if pct <= lo:
        return 1.0
    if pct >= hi:
        return 0.0
    return 0.5


def backtest(ret, sig, lo, hi, rebal_n, invert=False, cost_buy=0.00125,
             cost_sell=0.00175, idle_ret=0.0):
    """按信号给仓位，每 rebal_n 个交易日才允许换仓（其余日仓位不变、净值随涨跌漂移）。"""
    n = len(ret)
    nav = np.ones(n)
    pos = np.zeros(n)
    cur = 0.0
    for i in range(n):
        if i % rebal_n == 0 or i == 0:
            tgt = position_from_pct(sig[i], lo, hi, invert)
            if tgt != tgt:
                # 🔴 信号缺失（滚动窗口未满 / 数据缺口）必须**回退满仓**，
                #    不能 `tgt = cur`：cur 初值是 0.0，会让最初 250 天被当成空仓，
                #    凭空造出一段"躲过下跌"的假收益（我第一版就踩了这个坑，
                #    年化被抬高约 1.2pp、回撤被低估约 14pp）。
                tgt = 1.0
            delta = tgt - cur
            c = delta * cost_buy if delta > 0 else (-delta) * cost_sell
        else:
            tgt, c = cur, 0.0
        w = tgt
        gross = w * ret[i] + (1 - w) * idle_ret
        nav[i] = nav[i - 1] * (1 + gross - c) if i > 0 else (1 + gross - c)
        cur = w
        pos[i] = w
    return nav, pos


def stats_of(nav, dates, label=""):
    nav = np.asarray(nav, dtype=float)
    yrs = len(nav) / 244
    tot = nav[-1] - 1
    ann = (nav[-1]) ** (1 / yrs) - 1 if yrs > 0 and nav[-1] > 0 else np.nan
    dd = nav / np.maximum.accumulate(nav) - 1
    mdd = dd.min()
    r = pd.Series(nav).pct_change().dropna()
    sharpe = (r.mean() / r.std() * np.sqrt(244)) if r.std() > 0 else np.nan
    return {"label": label, "总收益%": round(tot * 100, 2), "年化%": round(ann * 100, 2),
            "最大回撤%": round(mdd * 100, 2), "夏普": round(sharpe, 2)}


def yearly_table(nav, dates, bench_nav):
    d = pd.DataFrame({"trade_date": dates, "nav": nav, "bench": bench_nav})
    d["year"] = d["trade_date"].str[:4]
    rows = []
    for y, s in d.groupby("year"):
        a = s["nav"].iloc[-1] / s["nav"].iloc[0] - 1
        b = s["bench"].iloc[-1] / s["bench"].iloc[0] - 1
        rows.append({"year": y, "策略%": round(a * 100, 2), "基准%": round(b * 100, 2),
                     "超额%": round((a - b) * 100, 2)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="破净率 overlay 择时回测（指数层）")
    ap.add_argument("--freq", default="all", choices=["all", "monthly", "quarterly", "annual"])
    a = ap.parse_args()

    df = load()
    df = df.dropna(subset=["mkt_ret"]).reset_index(drop=True)
    ret = df["mkt_ret"].values
    dates = df["trade_date"].values
    bench_nav = np.cumprod(1 + ret)

    print(f"\n样本 {len(df)} 日：{dates[0]} ~ {dates[-1]}")
    bh = stats_of(bench_nav, dates, "买入持有(中证800全收益)")
    print("基准：" + " | ".join(f"{k}={v}" for k, v in bh.items() if k != "label"))

    # ---- 频率 × 阈值网格
    freqs = {"monthly": 21, "quarterly": 63, "annual": 244}
    if a.freq != "all":
        freqs = {a.freq: freqs[a.freq]}

    sig_nb = df["pct_all"].values
    rows = []
    for fname, rn in freqs.items():
        for lo in (20, 30, 40):
            for hi in (60, 70, 80):
                if lo >= hi:
                    continue
                nav, _ = backtest(ret, sig_nb, lo, hi, rn, invert=True)
                s = stats_of(nav, dates, f"破净率 {fname} lo{lo}/hi{hi}")
                s["freq"] = fname
                s["lo"] = lo
                s["hi"] = hi
                s["跑赢基准"] = "✅" if s["年化%"] > bh["年化%"] else "❌"
                rows.append(s)
    grid = pd.DataFrame(rows)
    print("\n=== 阈值 × 频率网格（破净率分位，invert：分位高=便宜=满仓）===")
    print(grid.to_string(index=False))

    # ---- 三个信号对照（季度调仓，固定 lo30/hi70）
    print("\n=== 信号对照（季度调仓 lo30/hi70）===")
    sigs = [("破净率分位", df["pct_all"].values, True)]
    if "pb_pct750" in df.columns:
        sigs.append(("全市场PB分位", df["pb_pct750"].values, False))
        # 破净率对全市场PB的滚动残差
        m = df[["pct_all", "pb_pct750"]].dropna()
        r = rolling_residual(m["pct_all"].values.astype(float),
                             m["pb_pct750"].values.astype(float), window=750)
        rr = pd.Series(r, index=m.index[-len(r):])
        # 残差分位（滚动 750，无前视）
        resid_pct = rr.rolling(750, min_periods=250).apply(
            lambda w: (w[-1] >= w[:-1]).mean() * 100, raw=True)
        full = pd.Series(np.nan, index=df.index)
        full.loc[resid_pct.index] = resid_pct.values
        sigs.append(("破净率残差分位", full.values, True))

    cmp_rows = [bh]
    for name, s, inv in sigs:
        nav, _ = backtest(ret, s, 30, 70, 63, invert=inv)
        cmp_rows.append(stats_of(nav, dates, name))
    print(pd.DataFrame(cmp_rows).to_string(index=False))

    # ---- 逐年（最佳配置：季度 lo30/hi70）
    nav_q, pos_q = backtest(ret, sig_nb, 30, 70, 63, invert=True)
    yt = yearly_table(nav_q, dates, bench_nav)
    print("\n=== 逐年（破净率 季度 lo30/hi70）===")
    print(yt.to_string(index=False))
    print(f"跑赢基准年份 {(yt['超额%'] > 0).sum()}/{len(yt)}")

    # ---- 🔴 2021.02 反例
    print("\n=== 🔴 2021.02 反例 ===")
    m21 = (dates >= "20210101") & (dates <= "20211231")
    i0 = np.argmax(m21)
    i1 = np.argmax((dates >= "20220101") & (dates <= "20220110"))
    print(f"  2021 年策略收益 {(nav_q[i1]/nav_q[i0]-1)*100:+.2f}%  "
          f"vs 基准 {(bench_nav[i1]/bench_nav[i0]-1)*100:+.2f}%")
    print(f"  2021 年内平均仓位 {pos_q[m21].mean()*100:.1f}%"
          f"（破净率高位→信号让满仓，是本次 overlay 挨打最重的一年）")

    os.makedirs(OUT_DIR, exist_ok=True)
    grid.to_csv(os.path.join(OUT_DIR, "overlay_grid.csv"), index=False,
                encoding="utf-8-sig")
    yt.to_csv(os.path.join(OUT_DIR, "overlay_yearly.csv"), index=False,
              encoding="utf-8-sig")


if __name__ == "__main__":
    main()
