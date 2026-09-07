# -*- coding: utf-8 -*-
"""
组合级引擎自检：验证 backtest/portfolio_engine.py 与 P1 实验引擎口径一致
=====================================================================

P1/P2 的结论数字全部来自 run_position_sizing_ablation.py（独立实验脚本）。
把它平台化后，必须证明**同一输入下两者给出同一结果**，否则：
  · 平台口径漂移 → 已固化的结论（f=0.10 / 8 窗口全胜）不再适用于平台
  · 却没人会发现，因为两个引擎各自跑各自的

本脚本用 P1 完全相同的输入（沪深300 as-of 20190101 前 40 只、20190101~20260731、
初始 100 万、f=0.10、无 cap）跑平台引擎，与 P1 的 sizing_capscan.csv 对照。

用法:  venv_ml/Scripts/python.exe run_portfolio_selftest.py
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
from backtest.portfolio_engine import run_portfolio_mode  # noqa: E402

START, END = "20190101", "20260731"
TOP_N, INIT_CAPITAL, F = 40, 1_000_000.0, 0.10
BENCH = "000300.SH"

# P1 实验引擎实测基准（data/results/position_sizing/sizing_capscan.csv，group=C_nocap）
P1_REF = {
    "n_taken": 419, "exposure": 50.4, "cagr_pct": 13.95,
    "sharpe": 1.05, "mdd_pct": -19.29, "avg_rt_pct": 2.3,
}


def build_stock_data():
    dfu = _get_index_constituents_from_db(BENCH, as_of_date=START)
    codes = sorted(dfu["code"].tolist())[:TOP_N]
    conn = sqlite3.connect(config.DATA["local_db_path"])
    stock_data = {}
    for c in codes:
        df = load_stock_prices(c, START, END, conn, lookback_days=250)
        if df is None or len(df) < 30:
            continue
        si = df[df["trade_date"] >= START].index.min()
        if pd.isna(si):
            continue
        stock_data[c] = ("", df, int(si))
    conn.close()
    return stock_data


def stage_platform(stock_data: dict):
    """第二阶段：走平台主流程 run_backtest()，A/B 对照默认 vs 开启。

    证明两件事：
      1. 默认 False ⇒ 仍是逐票独立输出（行为零变更）
      2. 开启 True  ⇒ 输出一行「组合」记录，且汇总聚合不炸
    """
    import run_backtest as rb

    codes = list(stock_data.keys())
    stocks = pd.DataFrame({"code": codes, "name": [""] * len(codes)})
    config.SELECTION["top_n"] = len(codes)
    config.BACKTEST["start_date"] = START
    config.BACKTEST["end_date"] = END
    for k, v in config.STRATEGIES.items():
        v["enabled"] = (k == "mean_reversion")

    print("\n" + "=" * 92)
    print("  [A] 默认（portfolio_shared_pool=False）→ 逐票独立，应与历史完全一致")
    print("=" * 92)
    config.STRATEGIES["mean_reversion"]["portfolio_shared_pool"] = False
    rb.run_backtest(stocks)

    print("\n" + "=" * 92)
    print("  [B] 开启（portfolio_shared_pool=True）→ 组合级共享池")
    print("=" * 92)
    config.STRATEGIES["mean_reversion"]["portfolio_shared_pool"] = True
    rb.run_backtest(stocks)

    print("\n" + "=" * 92)
    print("  ✅ 平台主流程 A/B 完成：A 应逐票输出 N 行，B 应只输出 1 行「组合」")
    print("=" * 92)


def main():
    print("=" * 92)
    print("组合级引擎自检 —— 平台引擎 vs P1 实验引擎（同输入必须同结果）")
    print("=" * 92)
    print(f"  区间 {START}~{END}｜池 沪深300 as-of {START} 前 {TOP_N} 只｜"
          f"初始 {INIT_CAPITAL:,.0f}｜f={F}｜cap=0(不限)")

    stock_data = build_stock_data()
    print(f"  载入 {len(stock_data)} 只（平台口径：不 reset_index，start_idx 由平台算法给出）")

    if "--platform" in sys.argv:
        stage_platform(stock_data)
        return 0

    cfg = dict(config.STRATEGIES["mean_reversion"])
    cfg["portfolio_shared_pool"] = True
    cfg["portfolio_f"] = F
    cfg["portfolio_cap"] = 0

    r = run_portfolio_mode(stock_data, MeanReversionStrategyPlugin, cfg,
                           INIT_CAPITAL, START, END)
    if r is None:
        print("  [FAIL] 组合级引擎返回 None（无信号或无价格）")
        return 1
    m = r["metrics"]

    print("\n" + "-" * 92)
    print(f"  {'指标':<14}{'平台引擎':>14}{'P1 实验引擎':>14}{'差异':>12}{'判定':>8}")
    print("-" * 92)
    rows = [
        ("成交笔数", m["n_taken"], P1_REF["n_taken"], 0, "{:.0f}"),
        ("暴露度%", m["exposure"] * 100, P1_REF["exposure"], 0.5, "{:.2f}"),
        ("CAGR%", m["cagr_pct"], P1_REF["cagr_pct"], 0.05, "{:.2f}"),
        ("Sharpe", m["sharpe"], P1_REF["sharpe"], 0.02, "{:.3f}"),
        ("MDD%", m["mdd_pct"], P1_REF["mdd_pct"], 0.5, "{:.2f}"),
    ]
    ok_all = True
    for name, got, ref, tol, fmt in rows:
        d = got - ref
        ok = abs(d) <= tol
        ok_all &= ok
        print(f"  {name:<14}{fmt.format(got):>14}{fmt.format(ref):>14}"
              f"{d:>+12.3f}{'  OK' if ok else '  FAIL':>8}")

    print("-" * 92)
    print(f"  均并发 {m['avg_conc']:.2f}｜P90 {m['p90_conc']:.0f}｜最大 {m['max_conc']:.0f}"
          f"｜部署率 {m['deploy']*100:.1f}%｜均笔 {m['avg_rt_pct']:+.3f}%"
          f"｜胜率 {m['win_rate_pct']:.1f}%")
    print(f"  skip: cap={m['skip_cap']} cash={m['skip_cash']} lot={m['skip_lot']}")
    print(f"  终值 {m['terminal']:,.0f}（总收益 {m['total_ret_pct']:+.2f}%）")

    print("\n" + "=" * 92)
    if ok_all:
        print("  ✅ 口径一致：平台组合级引擎 == P1 实验引擎，已固化结论可直接套用")
    else:
        print("  ⚠️ 口径漂移：平台引擎与 P1 不一致，已固化结论【不可】直接套用，先查差异来源")
    print("=" * 92)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
