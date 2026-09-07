# -*- coding: utf-8 -*-
"""
小资金实用性扫描：池子 × 池子大小 N × 资金规模 C × 单笔比例 f
=====================================================

★ 为什么要跑这个
-----------------------------------------------------------------
P1~P3 的全部结论建立在「40 只池 + 100 万本金」上，而平台实际配置是：

    GLOBAL.top_n          = 5           ← 只选 5 只
    per_stock_capital     = 20000        → 5 只 = 10 万
    total_capital         = 200000       → 选股族 20 万
    区间                   = 20200103~20260825
    股票池                 = zz800 (000906.SH)

三个结构性差异，都会改变最优 f：
  1. **池子只有 5 只** ⇒ 并发上限锁死在 5。f 小了资金必然闲置
     （40 只池的 f=0.10 ≈ 分散 6 只；5 只池的 f=0.10 最多投 50%，
      若当日只有 1~2 个信号则只投 10~20%）。
  2. **资金只有 10~20 万** ⇒ 两道小资金特有的约束：
     · 最低 5 元佣金（`max(amount*0.00025, 5.0)`）：单笔 2 万时正好踩线，
       单笔 1 万被抬高 2 倍、2500 被抬高 8 倍 ⇒ 罚小 f
     · 整手约束：单笔金额买不起 1 手高价股时静默 skip
  3. **池子变了** ⇒ 信号密度变了，峰值 f 会漂移（故本脚本跑两个池子做稳健性对照）

用法:  venv_ml/Scripts/python.exe run_small_capital_scan.py
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import config  # noqa: E402
from run_backtest import load_stock_prices, _get_index_constituents_from_db  # noqa: E402
from backtest.mean_reversion_plugin import MeanReversionStrategyPlugin  # noqa: E402
from backtest.portfolio_engine import extract_signals, simulate, resolve_f  # noqa: E402

START, END = "20200103", "20260825"      # 平台 GLOBAL 实际区间
# 同区间下换池子 ⇒ 隔离「池子」这一个变量，用于评估 f 峰值的稳健性
POOLS = [("000906.SH", "zz800"), ("000300.SH", "hs300")]
N_LIST = [5, 10, 20, 40]                  # 池子大小（5 = 平台实际 top_n）
C_LIST = [100_000, 200_000]
F_LIST = [0.0125, 0.025, 0.05, 0.083, 0.10, 0.125, 0.167,
          0.20, 0.25, 0.30, 0.40, 0.50, 1.00]

OUT_DIR = HERE / "data" / "results" / "position_sizing"


def scan_pool(pool: str, pool_name: str) -> pd.DataFrame:
    dfu = _get_index_constituents_from_db(pool, as_of_date=START)
    codes40 = sorted(dfu["code"].tolist())[:40]
    conn = sqlite3.connect(config.DATA["local_db_path"])
    stock_data = {}
    for c in codes40:
        df = load_stock_prices(c, START, END, conn, lookback_days=250)
        if df is None or len(df) < 30:
            continue
        si = df[df["trade_date"] >= START].index.min()
        if pd.isna(si):
            continue
        stock_data[c] = ("", df, int(si))
    conn.close()

    cfg = dict(config.STRATEGIES["mean_reversion"])
    events_all, px_all = extract_signals(stock_data, MeanReversionStrategyPlugin, cfg, START)
    nb = sum(1 for e in events_all if e["action"] == "BUY")
    print(f"  [{pool_name}] 载入 {len(stock_data)} 只｜BUY 信号 {nb} 笔"
          f"（前 N 只截断复用；插件逐票独立 ⇒ 截断等价）")

    rows = []
    for N in N_LIST:
        codes = list(px_all.columns)[:N]
        cset = set(codes)
        px = px_all[codes]
        ev = [e for e in events_all if e["code"] in cset]
        for C in C_LIST:
            for f in F_LIST:
                m, _ = simulate(ev, px, float(C), f=f, cap=0)
                m.update({"pool": pool_name, "N": N, "C": C})
                rows.append(m)
    return pd.DataFrame(rows)


def main():
    print("=" * 104)
    print("小资金实用性扫描：池子 × 池子大小 N × 资金规模 C × 单笔比例 f")
    print("=" * 104)
    print(f"  区间 {START}~{END}（平台 GLOBAL 实际）｜成本=真实分科目"
          f"（含最低 5 元佣金 + 滑点 0.1% + 过户费 + 日期感知印花税）")
    print("  ⚠️ 股票为「按代码排序取前 N 只」，非平台选股结果 ⇒ 看量级与相对排序，不看绝对收益\n")

    frames = [scan_pool(p, n) for p, n in POOLS]
    df = pd.concat(frames, ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "small_capital_scan.csv", index=False, encoding="utf-8-sig")

    # ── 表 1：auto f 的效率损失（每个池子 × N）──
    print("\n" + "=" * 104)
    print("  表 1 · auto f = clamp(2/N, 0.05, 0.30) 的效率（资金 10 万）")
    print("=" * 104)
    print(f"  {'池':<8}{'N':>3}{'auto f':>8}{'auto CAGR%':>12}{'最优f':>8}{'最优CAGR%':>11}"
          f"{'损失pp':>9}{'f=0.10 CAGR%':>14}{'vs最优pp':>10}")
    print("-" * 104)
    for pname in [n for _, n in POOLS]:
        for N in N_LIST:
            g = df[(df["pool"] == pname) & (df["N"] == N) & (df["C"] == 100_000)]
            if g.empty:
                continue
            fa = resolve_f({"portfolio_f": None}, N)
            a = g.iloc[(g["f"] - fa).abs().argmin()]
            b = g.loc[g["cagr_pct"].idxmax()]
            f10 = g.iloc[(g["f"] - 0.10).abs().argmin()]
            print(f"  {pname:<8}{N:>3}{fa:>8.3f}{a['cagr_pct']:>12.2f}{b['f']:>8.3f}"
                  f"{b['cagr_pct']:>11.2f}{a['cagr_pct']-b['cagr_pct']:>+9.2f}"
                  f"{f10['cagr_pct']:>14.2f}{f10['cagr_pct']-b['cagr_pct']:>+10.2f}")
        print()

    # ── 表 2：N=5（平台实际）两个池子的 f 曲线对照 ──
    print("=" * 104)
    print("  表 2 · N=5（平台实际 top_n）· 两池 f 曲线对照（资金 10 万）")
    print("=" * 104)
    p1, p2 = [n for _, n in POOLS]
    g1 = df[(df["pool"] == p1) & (df["N"] == 5) & (df["C"] == 100_000)].set_index("f")
    g2 = df[(df["pool"] == p2) & (df["N"] == 5) & (df["C"] == 100_000)].set_index("f")
    print(f"  {'f':>7}{'单笔元':>10} | {p1+' CAGR%':>13}{'Sharpe':>8}{'MDD%':>9}{'暴露%':>8}"
          f" | {p2+' CAGR%':>13}{'Sharpe':>8}{'MDD%':>9}{'暴露%':>8}")
    print("-" * 104)
    for f in F_LIST:
        a, b = g1.loc[f], g2.loc[f]
        print(f"  {f:>7.3f}{f*100000:>10,.0f} | {a['cagr_pct']:>13.2f}{a['sharpe']:>8.2f}"
              f"{a['mdd_pct']:>9.2f}{a['exposure']*100:>8.1f}"
              f" | {b['cagr_pct']:>13.2f}{b['sharpe']:>8.2f}{b['mdd_pct']:>9.2f}"
              f"{b['exposure']*100:>8.1f}")

    print(f"\n[已保存] {OUT_DIR}/small_capital_scan.csv")
    print("\n  判读：① 看表 1 的『损失pp』——auto f 是否稳定接近最优；")
    print("        ② 看表 2 两池的 CAGR 峰 f 是否一致——不一致说明峰值 f 随池子漂移，"
          "只可取其量级不可精确调优。")


if __name__ == "__main__":
    main()
