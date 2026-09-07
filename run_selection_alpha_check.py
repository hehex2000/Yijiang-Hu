# -*- coding: utf-8 -*-
"""
选股 alpha 检验：真实多因子选股 vs 随机抽样分布
=====================================================================

问题：在 mean_reversion 策略上，「多因子选股选出的 5 只」到底有没有增量？

单个随机对照（run_real_selection_portfolio.py 的 C/D 组）说明不了问题——
一组 5 只是 n=1 样本。故抽 **40 组随机 5 只**做分布，看真实选股落在哪个分位。

同时报两个口径，区分「选股本身有效」与「选股与本策略匹配」：
  · BH      = 买入持有收益（选股选的是不是好股票）
  · MR-CAGR = 均值回归策略组合级收益（选股对【本策略】有没有增量）

用法:  venv_ml/Scripts/python.exe run_selection_alpha_check.py [组数]

注：文件名不能用 *_test.py —— .gitignore 第 59 行会把它忽略掉。
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import config  # noqa: E402
import run_backtest as rb  # noqa: E402
from backtest.mean_reversion_plugin import MeanReversionStrategyPlugin  # noqa: E402
from backtest.portfolio_engine import run_portfolio_mode  # noqa: E402

START, END = config.GLOBAL["backtest_start"], config.GLOBAL["backtest_end"]
TOP_N = config.GLOBAL["top_n"]
TOTAL = float(config.BACKTEST["total_capital"])
N_GROUPS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
SEED = 20260907
OUT = Path("data/results/position_sizing/selection_alpha_test.csv")


def build_pool(sel_date: str):
    """同选股日、同流动性过滤口径的候选池。"""
    pool = rb._get_zz800_from_db(sel_date)
    conn = sqlite3.connect(config.DATA["local_db_path"])
    try:
        pool = rb.prefilter_by_liquidity(conn, pool, sel_date)
    except Exception as e:  # noqa: BLE001
        print(f"  [WARN] 流动性过滤失败，用原池: {e}")
    conn.close()
    return sorted(pool["code"].tolist())


def load(codes: list) -> dict:
    conn = sqlite3.connect(config.DATA["local_db_path"])
    sd = {}
    for c in codes:
        try:
            df = rb.load_stock_prices(c, START, END, conn, lookback_days=250)
        except Exception:  # noqa: BLE001
            continue
        if df is None or len(df) < 30:
            continue
        si = df[df["trade_date"] >= START].index.min()
        if pd.isna(si):
            continue
        sd[c] = ("", df, int(si))
    conn.close()
    return sd


def bh_mean(sd: dict) -> float:
    """买入持有均值（adj_close 首末）。"""
    rets = []
    for _, (_, df, si) in sd.items():
        col = "adj_close" if "adj_close" in df.columns else "close"
        try:
            rets.append(df[col].iloc[-1] / df[col].iloc[si] - 1)
        except Exception:  # noqa: BLE001
            pass
    return float(np.mean(rets) * 100) if rets else float("nan")


def run_one(sd: dict) -> dict:
    cfg = dict(config.STRATEGIES["mean_reversion"])
    cfg["portfolio_shared_pool"] = True
    cfg["portfolio_f"] = None      # auto = clamp(2/N, 0.05, 0.25)
    cfg["portfolio_cap"] = 0
    r = run_portfolio_mode(sd, MeanReversionStrategyPlugin, cfg, TOTAL, START, END)
    if r is None:
        return {}
    m = r["metrics"]
    return {
        "cagr_pct": m["cagr_pct"], "sharpe": m["sharpe"], "mdd_pct": m["mdd_pct"],
        "n_taken": m["n_taken"], "exposure": m["exposure"] * 100,
        "skip_cash": m["skip_cash"], "skip_lot": m["skip_lot"],
        "bh_pct": bh_mean(sd),
    }


def main():
    print("=" * 90)
    print("选股 alpha 检验：真实多因子选股 vs 随机 5 只分布")
    print("=" * 90)

    # ── 真实选股结果（复用上一轮保存的 CSV，避免重跑 40s 因子计算）──
    sel_csv = OUT.parent / "real_selection_top20.csv"
    if not sel_csv.exists():
        print(f"  [FAIL] 缺 {sel_csv}，请先跑 run_real_selection_portfolio.py")
        return 1
    sel = pd.read_csv(sel_csv, dtype={"code": str})
    real = [str(c).zfill(6) for c in sel["code"].tolist()[:TOP_N]]
    sel_date = "20200102"
    print(f"  真实选股({sel_date})：{real}")

    # ── 构造随机组 ──
    pool = build_pool(sel_date)
    print(f"  候选池：{len(pool)} 只（同选股日 + 同流动性过滤）")
    rng = np.random.default_rng(SEED)
    groups = [list(rng.choice(pool, size=TOP_N, replace=False)) for _ in range(N_GROUPS)]

    need = sorted(set(real) | {c for g in groups for c in g})
    print(f"  需加载 {len(need)} 只（真实 {TOP_N} + 随机 {N_GROUPS}×{TOP_N}，去重后）")
    sd_all = load(need)
    print(f"  加载成功 {len(sd_all)} 只")

    sd_real = {c: sd_all[c] for c in real if c in sd_all}
    if len(sd_real) < TOP_N:
        print(f"  [WARN] 真实选股只有 {len(sd_real)}/{TOP_N} 只有数据")

    print("\n  [真实选股] 组合级 mean_reversion ...")
    rr = run_one(sd_real)
    if not rr:
        print("  [FAIL] 真实选股组合级返回空")
        return 1

    print(f"  [随机对照] 跑 {N_GROUPS} 组 ...")
    rrows = []
    for i, g in enumerate(groups, 1):
        sub = {c: sd_all[c] for c in g if c in sd_all}
        if len(sub) < TOP_N:
            continue
        r = run_one(sub)
        if r:
            r["group"] = f"R{i:02d}"
            rrows.append(r)
        if i % 10 == 0:
            print(f"    ... {i}/{N_GROUPS}")

    d = pd.DataFrame(rrows)
    d.to_csv(OUT, index=False, encoding="utf-8-sig")

    def pct(v, col):
        return float((d[col] < v).mean() * 100)

    print("\n" + "=" * 90)
    print("  结果（组合级共享池，f=auto 2/N，总资金 {:,.0f}）".format(TOTAL))
    print("=" * 90)
    print(f"  {'指标':<12}{'真实选股':>12}{'随机中位':>12}{'随机均值':>12}{'随机P25':>11}"
          f"{'随机P75':>11}{'分位':>9}")
    print("-" * 90)
    for col, fmt in [("cagr_pct", "{:.2f}"), ("sharpe", "{:.3f}"),
                     ("mdd_pct", "{:.2f}"), ("exposure", "{:.1f}"),
                     ("bh_pct", "{:.2f}")]:
        v = rr[col]
        print(f"  {col:<12}{fmt.format(v):>12}{fmt.format(d[col].median()):>12}"
              f"{fmt.format(d[col].mean()):>12}"
              f"{fmt.format(d[col].quantile(0.25)):>11}"
              f"{fmt.format(d[col].quantile(0.75)):>11}"
              f"{pct(v, col):>8.0f}%")

    print("-" * 90)
    print(f"  真实选股：成交 {rr['n_taken']:.0f}｜skip_cash {rr['skip_cash']:.0f}"
          f"｜skip_lot {rr['skip_lot']:.0f}")
    print(f"  随机   ：成交中位 {d['n_taken'].median():.0f}"
          f"｜skip_cash 中位 {d['skip_cash'].median():.0f}"
          f"｜skip_lot 中位 {d['skip_lot'].median():.0f}")

    print("\n" + "=" * 90)
    pc, pb = pct(rr["cagr_pct"], "cagr_pct"), pct(rr["bh_pct"], "bh_pct")
    print(f"  判读：")
    print(f"    · MR 策略收益分位 {pc:.0f}%（<25% 说明选股对本策略是【负】增量）")
    print(f"    · 买入持有收益分位 {pb:.0f}%（>75% 说明选股本身选到了好股票）")
    if pb > 70 and pc < 30:
        print("    ⇒ 选股选到了好股票，但【与均值回归策略不匹配】：")
        print("      优质成长股趋势性强，均值回归策略更适合震荡股。")
    elif pc < 30:
        print("    ⇒ 选股对本策略是负增量，且 BH 也无优势 —— 选股本身需重新审视。")
    else:
        print("    ⇒ 选股对本策略未见显著负增量。")
    print("=" * 90)
    print(f"  明细 → {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
