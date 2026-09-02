# -*- coding: utf-8 -*-
"""
在**已有策略 NAV** 上叠加破净率 gate —— 参数网格筛选器

为什么需要它
------------
`run_value_backtest.py --pb-gate` 的真实双账回测要跑几十分钟（每次调仓都要全市场选股），
没法用来扫参数。但 gate 只改**仓位**、不改选股，所以可以在已有的满仓 NAV 上做叠加近似：

    r_gate,t = w_t · r_strat,t − cost_t

其中 w_t 由破净率信号决定，cost 只在 w 变化时发生。这个近似忽略整手约束
（半仓时现金/股数的取整差异），用于**参数筛选**足够；最终结论必须用真实双账回测复核。

支持两种模式
------------
  abs  视频原始规则：破净率 > 10% → 满仓（绝对阈值）
  pct  滚动分位：破净率 750 日分位 > hi → 满仓

用法
----
  python net_break_screen.py --nav data/results/value_strategy/xxx.csv
  python net_break_screen.py --nav <csv> --mode abs --grid
  python net_break_screen.py --nav <csv> --col portfolio_value_full
"""
import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

NB = os.path.join("data", "results", "net_break", "market_net_break.csv")

COST_BUY = 0.00125    # 佣金 0.025% + 滑点 0.10%
COST_SELL = 0.00175   # 再 + 印花税 0.05%
IDLE_RET = 0.0        # 空仓资金按 0% 保守计


def load_nav(path, col):
    df = pd.read_csv(path, dtype={"trade_date": str})
    if col not in df.columns:
        raise SystemExit(f"列 {col} 不存在，可用：{list(df.columns)}")
    df = df[["trade_date", col]].dropna().sort_values("trade_date").reset_index(drop=True)
    df["ret"] = df[col].pct_change()
    return df


def build_signal(df, mode, universe, lag, gate_rebal):
    """给每条 NAV 记录挂上 (signal, exposure_key)。gate_rebal=每几个交易日允许改一次仓位。"""
    nb = pd.read_csv(NB, dtype={"trade_date": str})
    ucol = {"all": "rate_all", "nobj": "rate_nobj", "clean": "rate_clean"}[universe]
    nb = nb[["trade_date", ucol]].dropna()
    nb["pct"] = nb[ucol].rolling(750, min_periods=250).apply(
        lambda w: (w[-1] >= w[:-1]).mean() * 100, raw=True)
    nb = nb.rename(columns={ucol: "rate"})
    m = df.merge(nb, on="trade_date", how="left")
    m["rate"] = m["rate"].ffill()
    m["pct"] = m["pct"].ffill()
    m["sig"] = m["rate"] * 100.0 if mode == "abs" else m["pct"]
    m["sig"] = m["sig"].shift(lag)          # 信号滞后，避免当日收盘价已知
    return m


def exposure(sig, lo, hi):
    """破净率高 = 便宜 = 满仓。信号缺失 → 满仓（不擅自降仓）。"""
    out = np.full(len(sig), 1.0)
    s = np.asarray(sig, dtype=float)
    out[s >= hi] = 1.0
    out[(s > lo) & (s < hi)] = 0.5
    out[s <= lo] = 0.0
    out[~np.isfinite(s)] = 1.0
    return out


def apply_gate(ret, sig, lo, hi, gate_rebal=1):
    """逐日叠加仓位；只有每 gate_rebal 日才允许换仓（中间仓位不变、净值随涨跌漂移）。"""
    n = len(ret)
    w_full = exposure(sig, lo, hi)
    nav = np.ones(n)
    w_act = np.zeros(n)
    cur = 0.0
    for i in range(n):
        if i % gate_rebal == 0:
            tgt = w_full[i]
        else:
            tgt = cur
        delta = tgt - cur
        c = delta * COST_BUY if delta > 0 else (-delta) * COST_SELL
        r = ret[i] if ret[i] == ret[i] else 0.0
        gross = tgt * r + (1 - tgt) * IDLE_RET
        nav[i] = nav[i - 1] * (1 + gross - c) if i > 0 else (1 + gross - c)
        cur = tgt
        w_act[i] = tgt
    return nav, w_act


def stats_of(nav, label=""):
    nav = np.asarray(nav, dtype=float)
    yrs = len(nav) / 244.0
    tot = nav[-1] - 1
    ann = nav[-1] ** (1 / yrs) - 1 if yrs > 0 and nav[-1] > 0 else np.nan
    dd = (nav / np.maximum.accumulate(nav) - 1).min()
    r = pd.Series(nav).pct_change().dropna()
    shp = r.mean() / r.std() * np.sqrt(244) if r.std() > 0 else np.nan
    return {"label": label, "总收益%": round(tot * 100, 2), "年化%": round(ann * 100, 2),
            "最大回撤%": round(dd * 100, 2), "夏普": round(shp, 2)}


def yearly(nav, dates, base_nav):
    d = pd.DataFrame({"trade_date": dates, "nav": nav, "base": base_nav})
    d["year"] = d["trade_date"].str[:4]
    rows = []
    for y, s in d.groupby("year"):
        a = s["nav"].iloc[-1] / s["nav"].iloc[0] - 1
        b = s["base"].iloc[-1] / s["base"].iloc[0] - 1
        rows.append({"year": y, "gate%": round(a * 100, 2), "满仓%": round(b * 100, 2),
                     "超额%": round((a - b) * 100, 2)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="在已有策略 NAV 上叠加破净率 gate")
    ap.add_argument("--nav", required=True, help="策略日频 NAV CSV（需含 trade_date）")
    ap.add_argument("--col", default="portfolio_value", help="NAV 列名")
    ap.add_argument("--mode", default="pct", choices=["pct", "abs"])
    ap.add_argument("--universe", default="all", choices=["all", "nobj", "clean"])
    ap.add_argument("--lag", type=int, default=1)
    ap.add_argument("--gate-rebal", type=int, default=1,
                    help="gate 换仓间隔交易日数(默认1=每天可换; 21=月频; 244=年频)")
    ap.add_argument("--grid", action="store_true", help="跑阈值网格")
    ap.add_argument("--lo", type=float, default=30.0)
    ap.add_argument("--hi", type=float, default=70.0)
    ap.add_argument("--wf-split", default=None,
                    help="walk-forward 分割日(YYYYMMDD): 在此之前选参数, 在此之后做样本外检验。"
                         "全样本网格选出的最优阈值自带选择偏差, 必须有样本外这一刀")
    a = ap.parse_args()

    df = load_nav(a.nav, a.col)
    m = build_signal(df, a.mode, a.universe, a.lag, a.gate_rebal)
    ret = m["ret"].fillna(0.0).values
    dates = m["trade_date"].values
    base_nav = np.cumprod(1 + ret)

    print(f"NAV: {a.nav}  列={a.col}  {len(m)} 日  {dates[0]} ~ {dates[-1]}")
    print(f"模式={a.mode}  口径={a.universe}  lag={a.lag}  gate换仓间隔={a.gate_rebal}日")
    base = stats_of(base_nav, "满仓(对照)")
    print("对照：" + " | ".join(f"{k}={v}" for k, v in base.items() if k != "label"))

    if a.grid:
        los = (5.0, 7.0, 10.0, 12.0) if a.mode == "abs" else (20.0, 30.0, 40.0)
        his = (12.0, 15.0, 20.0) if a.mode == "abs" else (60.0, 70.0, 80.0)
        rows = []
        for lo in los:
            for hi in his:
                if lo >= hi:
                    continue
                nav, w = apply_gate(ret, m["sig"].values, lo, hi, a.gate_rebal)
                s = stats_of(nav, f"{a.mode} lo{lo}/hi{hi}")
                s["lo"], s["hi"] = lo, hi
                s["平均仓位%"] = round(w.mean() * 100, 1)
                s["空仓天数%"] = round((w == 0).mean() * 100, 1)
                s["跑赢满仓"] = "✅" if s["年化%"] > base["年化%"] else "❌"
                rows.append(s)
        g = pd.DataFrame(rows)
        print("\n=== 阈值网格 ===")
        print(g.to_string(index=False))
    else:
        nav, w = apply_gate(ret, m["sig"].values, a.lo, a.hi, a.gate_rebal)
        g = pd.DataFrame([base, stats_of(nav, f"{a.mode} lo{a.lo}/hi{a.hi}")])
        print("\n=== 单组结果 ===")
        print(g.to_string(index=False))
        print(f"平均仓位 {w.mean()*100:.1f}%  空仓天数 {(w==0).mean()*100:.1f}%")

    # 逐年（用网格最优组；非网格模式用 --lo/--hi）
    # ⚠️ 不能用默认 lo=30/hi=70 去跑 abs 模式：abs 的信号量纲是百分比(0~16)，
    #    >=70 永不成立 → 全程空仓 → 逐年全是 0.00%（我第一版就踩了这个坑）。
    if a.grid and len(g):
        best = g.loc[g["年化%"].astype(float).idxmax()]
        lo, hi = float(best["lo"]), float(best["hi"])
        print(f"[逐年采用网格最优组 lo={lo}/hi={hi}]")
    else:
        lo, hi = a.lo, a.hi
    nav, w = apply_gate(ret, m["sig"].values, lo, hi, a.gate_rebal)
    y = yearly(nav, dates, base_nav)
    print(f"\n=== 逐年（{a.mode} lo{lo}/hi{hi}）===")
    print(y.to_string(index=False))
    print(f"跑赢满仓年份 {(y['超额%'] > 0).sum()}/{len(y)}")

    # ---- walk-forward：先选参后样本外，检验阈值是否只是拟合噪音
    if a.wf_split:
        sp = a.wf_split
        tr = dates < sp
        te = dates >= sp
        if tr.sum() < 250 or te.sum() < 250:
            print(f"\n[walk-forward] 分割日 {sp} 导致某侧样本不足 250 日，跳过")
            return
        los = (5.0, 7.0, 10.0, 12.0) if a.mode == "abs" else (20.0, 30.0, 40.0)
        his = (12.0, 15.0, 20.0) if a.mode == "abs" else (60.0, 70.0, 80.0)
        rows = []
        for lo2 in los:
            for hi2 in his:
                if lo2 >= hi2:
                    continue
                n2, _ = apply_gate(ret[tr], m["sig"].values[tr], lo2, hi2, a.gate_rebal)
                s = stats_of(n2, f"lo{lo2}/hi{hi2}")
                s["lo"], s["hi"] = lo2, hi2
                rows.append(s)
        gtr = pd.DataFrame(rows)
        b = gtr.loc[gtr["年化%"].astype(float).idxmax()]
        blo, bhi = float(b["lo"]), float(b["hi"])
        print("\n" + "=" * 78)
        print(f"Walk-forward：{sp} 之前选参（{tr.sum()} 日）→ {sp} 之后样本外（{te.sum()} 日）")
        print("=" * 78)
        print(f"  训练期最优阈值：lo={blo} / hi={bhi}（训练期年化 {b['年化%']}%）")
        nte, wte = apply_gate(ret[te], m["sig"].values[te], blo, bhi, a.gate_rebal)
        ste = stats_of(nte, "样本外 gate")
        bte = stats_of(np.cumprod(1 + ret[te]), "样本外 满仓")
        print(pd.DataFrame([bte, ste]).to_string(index=False))
        d = ste["年化%"] - bte["年化%"]
        print(f"\n  样本外年化差 = {d:+.2f}pp   样本外平均仓位 = {wte.mean()*100:.1f}%")
        print("  判定：" + ("✅ 样本外仍有效" if d > 0 else
                          "❌ 样本外失效 —— 训练期最优阈值很可能是拟合噪音"))
        # 稳健性：训练期所有阈值在样本外的 dispersion
        outs = []
        for _, r in gtr.iterrows():
            n3, _ = apply_gate(ret[te], m["sig"].values[te],
                               float(r["lo"]), float(r["hi"]), a.gate_rebal)
            outs.append(stats_of(n3)["年化%"])
        outs = np.array(outs, dtype=float)
        print(f"  样本外年化跨阈值区间：{outs.min():.2f}% ~ {outs.max():.2f}%"
              f"（极差 {outs.max()-outs.min():.2f}pp，越大说明越依赖挑参数）")


if __name__ == "__main__":
    main()
