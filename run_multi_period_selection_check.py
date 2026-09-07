# -*- coding: utf-8 -*-
"""
P4 · 多期选股检验：多因子选股到底有没有用？
=====================================================================

§3.7 的检验是 **n=1 选股期**（只在 2020-01-02 选了一次），无论结果多差都不能
断言「多因子选股无效」——单期结果 = 选股能力 + 该期行情，两者无法分离。

本脚本把样本扩到 **7 期**：每期独立选股（次年 1 月第一个交易日选，持有 1 年），
每期都配 **同期同池的随机分布对照**。若选股有真 alpha，则真实选股应在多数期
跑赢随机中位；若只是行情运气，则应在各期分位上随机散布（均值≈50%）。

设计：
  · 每期选股日 = 该期回测开始日的前一交易日（与平台 run_selection 同规则）
  · 真实组：多因子选股 TOP 5（候选 15 取前 5）
  · 对照组：同期同池（zz800 + 流动性过滤）随机抽 K 组 × 5 只
  · 统一走组合级共享池（f=auto 2/N），报 MR-CAGR / Sharpe / BH

用法:  venv_ml/Scripts/python.exe run_multi_period_selection_check.py [K组数]
"""
import sqlite3
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import config  # noqa: E402
import run_backtest as rb  # noqa: E402
from backtest.mean_reversion_plugin import MeanReversionStrategyPlugin  # noqa: E402
from backtest.portfolio_engine import run_portfolio_mode  # noqa: E402

TOP_N = config.GLOBAL["top_n"]
TOTAL = float(config.BACKTEST["total_capital"])
K_GROUPS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
SEED = 20260907
OUT = Path("data/results/position_sizing/multi_period_selection.csv")

# 期定义：(标签, 回测开始, 回测结束)  选股日 = 开始日的前一交易日
PERIODS = [
    ("2020", "20200103", "20201231"),
    ("2021", "20210104", "20211231"),
    ("2022", "20220104", "20221231"),
    ("2023", "20230103", "20231229"),
    ("2024", "20240102", "20241231"),
    ("2025", "20250102", "20251231"),
    ("2026", "20260105", "20260825"),  # 不足一年，单独标注
]


def build_pool(sel_date: str):
    pool = rb._get_zz800_from_db(sel_date)
    conn = sqlite3.connect(config.DATA["local_db_path"])
    try:
        pool = rb.prefilter_by_liquidity(conn, pool, sel_date)
    except Exception as e:  # noqa: BLE001
        print(f"    [WARN] 流动性过滤失败，用原池: {e}")
    conn.close()
    return sorted(pool["code"].tolist())


def load(codes: list, start: str, end: str) -> dict:
    conn = sqlite3.connect(config.DATA["local_db_path"])
    sd = {}
    for c in codes:
        try:
            df = rb.load_stock_prices(c, start, end, conn, lookback_days=250)
        except Exception:  # noqa: BLE001
            continue
        if df is None or len(df) < 30:
            continue
        si = df[df["trade_date"] >= start].index.min()
        if pd.isna(si):
            continue
        sd[c] = ("", df, int(si))
    conn.close()
    return sd


def bh_mean(sd: dict) -> float:
    rets = []
    for _, (_, df, si) in sd.items():
        col = "adj_close" if "adj_close" in df.columns else "close"
        try:
            rets.append(df[col].iloc[-1] / df[col].iloc[si] - 1)
        except Exception:  # noqa: BLE001
            pass
    return float(np.mean(rets) * 100) if rets else float("nan")


def run_one(sd: dict, start: str, end: str) -> dict:
    cfg = dict(config.STRATEGIES["mean_reversion"])
    cfg["portfolio_shared_pool"] = True
    cfg["portfolio_f"] = None      # auto = clamp(2/N, 0.05, 0.25)
    cfg["portfolio_cap"] = 0
    r = run_portfolio_mode(sd, MeanReversionStrategyPlugin, cfg, TOTAL, start, end)
    if r is None:
        return {}
    m = r["metrics"]
    return {
        "cagr_pct": m["cagr_pct"], "sharpe": m["sharpe"], "mdd_pct": m["mdd_pct"],
        "n_taken": m["n_taken"], "exposure": m["exposure"] * 100,
        "bh_pct": bh_mean(sd), "terminal": m["terminal"],
    }


def main():
    print("=" * 92)
    print("P4 · 多期选股检验（每期独立选股 + 同期同池随机分布对照）")
    print("=" * 92)
    print(f"  期数 {len(PERIODS)}｜每期 top_n={TOP_N}｜随机对照 {K_GROUPS} 组/期"
          f"｜总资金 {TOTAL:,.0f}｜组合级 f=auto")

    rng = np.random.default_rng(SEED)
    rows, detail = [], []

    for tag, start, end in PERIODS:
        print("\n" + "=" * 92)
        print(f"  【{tag}】回测 {start} ~ {end}")
        print("=" * 92)
        config.BACKTEST["start_date"] = start
        config.BACKTEST["end_date"] = end

        # ── 1. 本期独立选股 ──
        try:
            sel = rb.run_selection()
        except Exception:  # noqa: BLE001
            print("    [FAIL] 选股抛异常：")
            traceback.print_exc()
            continue
        if sel is None or sel.empty:
            print("    [SKIP] 选股返回空")
            continue
        sel_date = config.SELECTION["date"].replace("-", "")
        real = [str(c).zfill(6) for c in sel["code"].tolist()[:TOP_N]]
        names = dict(zip(sel["code"].astype(str).str.zfill(6),
                         sel.get("name", pd.Series([""] * len(sel)))))
        print(f"    选股日 {sel_date}｜TOP{TOP_N}："
              f"{[f'{c}({names.get(c, '')})' for c in real]}")

        # ── 2. 随机对照（同池同日）──
        pool = build_pool(sel_date)
        if len(pool) < TOP_N * 4:
            print(f"    [SKIP] 池子过小({len(pool)})")
            continue
        groups = [list(rng.choice(pool, size=TOP_N, replace=False))
                  for _ in range(K_GROUPS)]

        need = sorted(set(real) | {c for g in groups for c in g})
        sd_all = load(need, start, end)
        sd_real = {c: sd_all[c] for c in real if c in sd_all}
        if len(sd_real) < TOP_N:
            print(f"    [SKIP] 真实选股只有 {len(sd_real)}/{TOP_N} 只有数据")
            continue

        rr = run_one(sd_real, start, end)
        if not rr:
            print("    [SKIP] 真实选股回测返回空")
            continue

        rrows = []
        for i, g in enumerate(groups, 1):
            sub = {c: sd_all[c] for c in g if c in sd_all}
            if len(sub) < TOP_N:
                continue
            r = run_one(sub, start, end)
            if r:
                r["group"] = f"{tag}_R{i:02d}"
                rrows.append(r)
        if not rrows:
            print("    [SKIP] 随机对照全部失败")
            continue
        d = pd.DataFrame(rrows)
        detail.append(d.assign(period=tag))

        def pct(v, col):
            return float((d[col] < v).mean() * 100)

        row = {
            "period": tag, "sel_date": sel_date, "start": start, "end": end,
            "codes": ",".join(real),
            "real_cagr": rr["cagr_pct"], "real_sharpe": rr["sharpe"],
            "real_bh": rr["bh_pct"], "real_exposure": rr["exposure"],
            "real_n": rr["n_taken"],
            "rand_cagr_med": d["cagr_pct"].median(),
            "rand_sharpe_med": d["sharpe"].median(),
            "rand_bh_med": d["bh_pct"].median(),
            "cagr_pctile": pct(rr["cagr_pct"], "cagr_pct"),
            "sharpe_pctile": pct(rr["sharpe"], "sharpe"),
            "bh_pctile": pct(rr["bh_pct"], "bh_pct"),
            "n_rand": len(d),
        }
        row["win_cagr"] = rr["cagr_pct"] > d["cagr_pct"].median()
        row["win_bh"] = rr["bh_pct"] > d["bh_pct"].median()
        rows.append(row)

        print(f"    真实: CAGR {rr['cagr_pct']:>7.2f}%  Sharpe {rr['sharpe']:>5.2f}"
              f"  BH {rr['bh_pct']:>7.2f}%  暴露 {rr['exposure']:>5.1f}%")
        print(f"    随机: CAGR {d['cagr_pct'].median():>7.2f}%  Sharpe {d['sharpe'].median():>5.2f}"
              f"  BH {d['bh_pct'].median():>7.2f}%  (n={len(d)})")
        print(f"    分位: CAGR {row['cagr_pctile']:>3.0f}%  Sharpe {row['sharpe_pctile']:>3.0f}%"
              f"  BH {row['bh_pctile']:>3.0f}%   → "
              f"{'跑赢随机中位' if row['win_cagr'] else '跑输随机中位'}")

    if not rows:
        print("\n[FAIL] 没有任何一期跑通")
        return 1

    R = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    R.to_csv(OUT, index=False, encoding="utf-8-sig")
    if detail:
        pd.concat(detail, ignore_index=True).to_csv(
            OUT.with_name("multi_period_rand_detail.csv"), index=False,
            encoding="utf-8-sig")

    print("\n" + "=" * 92)
    print("  汇总")
    print("=" * 92)
    print(f"  {'期':<6}{'CAGR%':>9}{'随机中位':>10}{'分位':>7}{'BH%':>9}{'随机中位':>10}"
          f"{'分位':>7}{'胜CAGR':>8}{'胜BH':>7}")
    print("  " + "-" * 80)
    for _, r in R.iterrows():
        print(f"  {r['period']:<6}{r['real_cagr']:>9.2f}{r['rand_cagr_med']:>10.2f}"
              f"{r['cagr_pctile']:>6.0f}%{r['real_bh']:>9.2f}{r['rand_bh_med']:>10.2f}"
              f"{r['bh_pctile']:>6.0f}%{'  WIN' if r['win_cagr'] else '  LOSS':>8}"
              f"{'  WIN' if r['win_bh'] else '  LOSS':>7}")

    n = len(R)
    wc, wb = int(R["win_cagr"].sum()), int(R["win_bh"].sum())
    print("  " + "-" * 80)
    print(f"  胜出期数：MR-CAGR {wc}/{n}｜买入持有 BH {wb}/{n}")
    print(f"  平均分位：MR-CAGR {R['cagr_pctile'].mean():.0f}%｜Sharpe "
          f"{R['sharpe_pctile'].mean():.0f}%｜BH {R['bh_pctile'].mean():.0f}%")

    # 二项检验（双尾，n 小，直接算 p = P(X<=k 或 X>=n-k) 的 2 倍近似）
    from math import comb
    k = min(wc, n - wc)
    p = 2 * sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    print(f"\n  符号检验（H0: 选股与随机无差异，胜出概率 0.5）：p ≈ {min(p, 1.0):.3f}")

    print("\n" + "=" * 92)
    mp = R["cagr_pctile"].mean()
    mb = R["bh_pctile"].mean()
    if mp < 30 and mb < 30:
        print("  ⇒ 选股在本策略上【系统性负增量】：多期平均处随机分布下游，"
              "且买入持有也差 ⇒ 不是策略匹配问题，是选股本身")
    elif mp < 30 and mb > 60:
        print("  ⇒ 选股选到了好股票（BH 分位高），但【与均值回归策略不匹配】")
    elif mp > 60:
        print("  ⇒ 选股在本策略上【有正增量】")
    else:
        print("  ⇒ 选股与随机【无显著差异】：多期平均接近 50% 分位，"
              "§3.7 的单期结果应归因于该期行情运气")
    print("=" * 92)
    print(f"  明细 → {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
