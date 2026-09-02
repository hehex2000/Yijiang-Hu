# -*- coding: utf-8 -*-
"""
券商 PB 分位择时策略回测 —— Gate 2（宽成本）+ Gate 4（walk-forward / 阈值敏感性）

为什么做「仓位择时」而不是「多空」：
  A 股做空单一行业（券商）成本极高且工具有限，多空利差在实盘拿不到。
  所以落地形式定为：低估值时满仓券商 / 中性半仓 / 高估值空仓（资金按 0% 计，保守）。

成本模型（组合层，非 per-stock 常数）
--------------------------------------
  加仓 Δ 仓位 → Δ × 0.125%（佣金 0.025% + 滑点 0.10%）
  减仓 Δ 仓位 → Δ × 0.175%（佣金 0.025% + 印花税 0.05% + 滑点 0.10%）
  完整换手 100% → 0.30% round trip
  空仓资金收益率按 0% 计（保守口径，不给策略白占便宜）

用法
----
  python broker_pe_strategy.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

OUT_DIR = os.path.join("data", "results", "broker_pe")
SRC = os.path.join(OUT_DIR, "sector_daily.csv")

COST_BUY = 0.00125
COST_SELL = 0.00175
IDLE_RET = 0.0            # 空仓资金收益（保守）

WINDOW = 750


def load():
    sec = pd.read_csv(SRC, dtype={"trade_date": str}).sort_values("trade_date")
    return sec.reset_index(drop=True)


def position_from_pct(pct, lo=30.0, hi=70.0):
    """分位 → 目标仓位：低估满仓 / 中性半仓 / 高估空仓。"""
    return np.where(pct <= lo, 1.0, np.where(pct >= hi, 0.0, 0.5))


def backtest(sec, col, lo=30.0, hi=70.0, rebal="always"):
    """逐日回测。

    rebal='always'  → 每日按信号调仓（换手高，成本重）
    rebal='quarter' → 每 60 交易日检查一次，其余时间漂移（更接近实盘）
    """
    d = sec[["trade_date", col, "ret", "fwd_bch_1" if "fwd_bch_1" in sec else "ret"]].copy()
    d["sec_ret"] = sec["ret"].fillna(0.0)
    d["target"] = position_from_pct(d[col].values, lo, hi)

    if rebal == "quarter":
        # 每 60 日更新一次目标仓位，其余时间沿用
        tgt = d["target"].values.copy()
        for i in range(1, len(tgt)):
            if i % 60 != 0:
                tgt[i] = tgt[i - 1]
        d["target"] = tgt

    pos = d["target"].values
    sec_ret = d["sec_ret"].values

    # 仓位在 t 日确定，收益从 t+1 开始计（避免用 t 日收盘价同时定仓和计收益）
    nav = np.ones(len(d))
    gross = np.zeros(len(d))
    cost = np.zeros(len(d))
    w = 0.0
    for i in range(1, len(d)):
        # t 日以 pos[i-1] 的仓位持有，获得 t 日收益
        gross[i] = w * sec_ret[i] + (1 - w) * IDLE_RET / 250.0
        # t 日调仓到 pos[i]
        delta = pos[i] - w
        c = delta * COST_BUY if delta > 0 else (-delta) * COST_SELL
        cost[i] = c
        nav[i] = nav[i - 1] * (1 + gross[i] - c)
        w = pos[i]

    d["nav"] = nav
    d["gross"] = gross
    d["cost"] = cost
    d["pos"] = pos

    n = len(d)
    years = n / 250.0
    tot = nav[-1] - 1
    ann = (1 + tot) ** (1 / years) - 1 if tot > -1 else np.nan
    dd = (pd.Series(nav) / pd.Series(nav).cummax() - 1).min()
    r = pd.Series(gross - cost)
    sharpe = r.mean() / r.std() * np.sqrt(250) if r.std() > 0 else np.nan
    turnover = np.abs(np.diff(np.concatenate([[0.0], pos]))).sum()
    total_cost = cost.sum()
    exposure = pos.mean()
    return dict(nav=nav, ann=ann, tot=tot, mdd=dd, sharpe=sharpe,
                turnover=turnover, cost=total_cost, exposure=exposure,
                final_nav=nav[-1])


def buy_hold(sec):
    r = sec["ret"].fillna(0.0)
    nav = (1 + r).cumprod()
    n = len(nav)
    years = n / 250.0
    tot = nav.iloc[-1] - 1
    ann = (1 + tot) ** (1 / years) - 1 if tot > -1 else np.nan
    dd = (nav / nav.cummax() - 1).min()
    sharpe = r.mean() / r.std() * np.sqrt(250) if r.std() > 0 else np.nan
    return dict(nav=nav.values, ann=ann, tot=tot, mdd=dd, sharpe=sharpe)


def bench_stats(sec):
    b = sec["bench_close"].astype(float)
    r = b.pct_change().fillna(0.0)
    nav = (1 + r).cumprod()
    n = len(nav)
    years = n / 250.0
    tot = nav.iloc[-1] - 1
    ann = (1 + tot) ** (1 / years) - 1 if tot > -1 else np.nan
    dd = (nav / nav.cummax() - 1).min()
    sharpe = r.mean() / r.std() * np.sqrt(250) if r.std() > 0 else np.nan
    return dict(nav=nav.values, ann=ann, tot=tot, mdd=dd, sharpe=sharpe)


def main():
    sec = load()
    col = f"pb_pct{WINDOW}"
    d0, d1 = sec.trade_date.iloc[0], sec.trade_date.iloc[-1]
    print(f"[数据] {len(sec)} 行 | {d0} ~ {d1}")
    print(f"[信号] PB 分位（{WINDOW}日滚动窗口）→ 仓位：≤30 满仓 / 30~70 半仓 / ≥70 空仓\n")

    bh = buy_hold(sec)
    bm = bench_stats(sec)

    print("=" * 92)
    print("【Gate 2】策略 vs 买入持有 vs 基准（全样本，已扣组合层成本）")
    print("=" * 92)
    print(f"{'方案':<26}{'总收益%':>10}{'年化%':>9}{'回撤%':>9}{'夏普':>8}"
          f"{'仓位均值':>10}{'累计换手':>10}{'累计成本%':>11}")
    print("-" * 92)

    def show(name, r, cost=None, turn=None, expo=None):
        print(f"{name:<26}{r['tot']*100:>10.2f}{r['ann']*100:>9.2f}{r['mdd']*100:>9.2f}"
              f"{r['sharpe']:>8.2f}"
              f"{(expo*100 if expo is not None else float('nan')):>10.1f}"
              f"{(turn if turn is not None else float('nan')):>10.2f}"
              f"{(cost*100 if cost is not None else float('nan')):>11.2f}")

    show("① 买入持有券商(裸)", bh, 0.0, 0.0, 1.0)
    show("② 基准 中证800全收益", bm, 0.0, 0.0, 1.0)

    for rebal, tag in (("always", "③ PB择时(每日调仓)"),
                       ("quarter", "④ PB择时(每60日调仓)")):
        r = backtest(sec, col, 30, 70, rebal)
        show(tag, r, r["cost"], r["turnover"], r["exposure"])
    print("-" * 92)
    print("  注：买入持有与基准均为【毛】收益（无调仓成本，故成本列记 0）；")
    print("      择时方案的年化/总收益是【已扣成本后】的净收益，可直接与①②对比。")

    # ---------------- 阈值敏感性 ----------------
    print("\n" + "=" * 92)
    print("【Gate 4】阈值敏感性（防跑后调参）—— 网格搜索，看是否只有特定阈值好看")
    print("=" * 92)
    print(f"{'低阈值':>8}{'高阈值':>8}{'年化%':>10}{'回撤%':>10}{'夏普':>9}"
          f"{'仓位均值':>10}{'累计成本%':>11}  相对买入持有")
    print("-" * 92)
    grid = []
    print(f"  (基准 中证800 年化 {bm['ann']*100:.2f}%，买入持有券商 {bh['ann']*100:.2f}%)")
    print("-" * 92)
    for lo in (20, 30, 40, 50):
        for hi in (60, 70, 80):
            if lo >= hi:
                continue
            r = backtest(sec, col, lo, hi, "quarter")
            grid.append((lo, hi, r))
            diff = (r["ann"] - bm["ann"]) * 100     # 对基准，不是对买入持有
            print(f"{lo:>8}{hi:>8}{r['ann']*100:>10.2f}{r['mdd']*100:>10.2f}"
                  f"{r['sharpe']:>9.2f}{r['exposure']*100:>10.1f}{r['cost']*100:>11.2f}"
                  f"   {diff:+.2f}pp")
    print("-" * 92)
    wins = sum(1 for lo, hi, r in grid if r["ann"] > bm["ann"])
    print(f"  {len(grid)} 组阈值中跑赢【基准】的: {wins}/{len(grid)}")
    if wins <= len(grid) * 0.4:
        print("  → ❌ 多数阈值下都跑不赢基准，说明结果依赖阈值挑选（过拟合风险）")
    elif wins >= len(grid) * 0.8:
        print("  → ✅ 对阈值不敏感，不是靠挑参数刷出来的")
    else:
        print("  → 🟡 部分阈值有效，需谨慎，不能宣称稳健")

    # ---------------- 单调性：不设阈值，直接看分档收益 ----------------
    print("\n" + "=" * 92)
    print("【单调性】不设阈值：按 PB 分位分 5 档，各档持有券商的未来 250 日收益")
    print("=" * 92)
    d = sec[[col, "fwd_sec_250", "fwd_bch_250"]].dropna().copy()
    d["q"] = pd.qcut(d[col], 5, labels=False, duplicates="drop")
    g = d.groupby("q").agg(板块=("fwd_sec_250", "mean"),
                           基准=("fwd_bch_250", "mean"),
                           超额=("fwd_exc_250" if "fwd_exc_250" in d else "fwd_sec_250", "mean"),
                           次数=("fwd_sec_250", "count"))
    if "fwd_exc_250" not in d:
        g["超额"] = g["板块"] - g["基准"]
    g.index = [f"Q{i}" for i in g.index]
    g = g.rename(columns={"板块": "板块%", "基准": "基准%", "超额": "超额%"})
    for c in ("板块%", "基准%", "超额%"):
        g[c] = (g[c] * 100).round(2)      # 只把收益率列转成百分数
    print(g.to_string())
    print("  → 若「板块」列随 Q0→Q4 单调递减，说明信号方向单调、阈值只是切点；")
    print("    若不单调（如中间档最高），说明阈值结果可能是碰巧。")

    # ---------------- 逐年 ----------------
    print("\n" + "=" * 92)
    print("【逐年】PB 择时(lo=30,hi=70,每60日调仓) vs 买入持有 vs 基准")
    print("=" * 92)
    r = backtest(sec, col, 30, 70, "quarter")
    nd = r["nav"]
    sec2 = sec.copy()
    sec2["nav_strat"] = nd
    sec2["nav_bh"] = bh["nav"]
    sec2["nav_bm"] = bm["nav"]
    sec2["yr"] = sec2.trade_date.str[:4]
    print(f"{'年份':<8}{'择时%':>10}{'买入持有%':>12}{'基准%':>10}{'择时-持有':>12}{'择时-基准':>12}")
    print("-" * 92)
    first = sec2.groupby("yr").first()
    last = sec2.groupby("yr").last()
    for yr in sorted(sec2.yr.unique()):
        if yr == sec2.yr.iloc[0]:
            continue
        prev = sec2[sec2.yr < yr].iloc[-1]
        cur = last.loc[yr]
        s = cur.nav_strat / prev.nav_strat - 1
        b = cur.nav_bh / prev.nav_bh - 1
        m = cur.nav_bm / prev.nav_bm - 1
        print(f"{yr:<8}{s*100:>10.2f}{b*100:>12.2f}{m*100:>10.2f}"
              f"{(s-b)*100:>12.2f}{(s-m)*100:>12.2f}")
    print("-" * 92)
    yrs = sorted(sec2.yr.unique())[1:]
    win_bh = sum(1 for yr in yrs
                 if last.loc[yr, "nav_strat"] / sec2[sec2.yr < yr].iloc[-1].nav_strat >
                 last.loc[yr, "nav_bh"] / sec2[sec2.yr < yr].iloc[-1].nav_bh)
    print(f"  跑赢买入持有的年份: {win_bh}/{len(yrs)} ({win_bh/len(yrs)*100:.0f}%)")

    # 保存
    out = os.path.join(OUT_DIR, "strategy_nav.csv")
    sec2[["trade_date", "nav_strat", "nav_bh", "nav_bm", col]].to_csv(
        out, index=False, encoding="utf-8-sig")
    print(f"\n[输出] {out}")


if __name__ == "__main__":
    main()
