# -*- coding: utf-8 -*-
"""
RSI+布林带双确认策略 —— 逐改动归因 A/B 完整回测
================================================

目的：把 2026-07-31 的三处改动拆开，量化每一处对完整策略净值的影响：
  改动1：rsi_period 9 → 14
  改动3：kelly_win_rate 0.53 → 0.57（回填事件研究实测胜率）
  改动2：新增 regime gate（市场状态门控）
        - 模式A 'ma'  ：指数站上 MA200 才做多（趋势思路）
        - 模式B 'adx' ：指数 ADX(14) < 25 才做（震荡/均值回归思路）

实验设计（控制变量，只改一项）：
  C0  基线(旧)      : RSI(9)/kelly0.53/无门控
  C1  仅改RSI       : RSI(14)/kelly0.53/无门控
  C2  RSI+凯利      : RSI(14)/kelly0.57/无门控
  C3  +MA200门控    : RSI(14)/kelly0.57/MA200门控
  C4  +ADX门控      : RSI(14)/kelly0.57/ADX(14)<25门控

固定：股票池(沪深300快照前40，剔除.BJ)、期间、每支本金、行情/手续费口径。
"""
import os
import sys
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from run_backtest import (  # noqa: E402
    load_stock_prices,
    calc_win_rate_from_trades,
    max_drawdown_with_dates,
    DB_PATH,
)
from backtest.rsi_bb_dual_plugin import RsiBbDualStrategyPlugin  # noqa: E402
import config  # noqa: E402

START = "20190101"
END = "20260731"
CAPITAL = 100_000
TOP_N = 40
BENCH = "000300.SH"
OUT_DIR = HERE / "data" / "results" / "rsi_bb_dual_ablation"


def build_universe(n: int = TOP_N) -> list[str]:
    conn = sqlite3.connect(DB_PATH)
    snap = conn.execute(
        "SELECT MAX(REPLACE(trade_date,'-','')) FROM index_constituent WHERE index_code=?",
        (BENCH,),
    ).fetchone()[0]
    rows = conn.execute(
        "SELECT ts_code FROM index_constituent "
        "WHERE index_code=? AND REPLACE(trade_date,'-','')=? "
        "ORDER BY ts_code",
        (BENCH, snap),
    ).fetchall()
    conn.close()
    codes = [r[0] for r in rows][:n]
    print(f"[股票池] 沪深300 快照 {snap} → 取 {len(codes)} 只（剔除.BJ）")
    return codes


def load_benchmark_return() -> float:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT trade_date, close FROM index_daily "
        "WHERE ts_code=? AND trade_date BETWEEN ? AND ? ORDER BY trade_date",
        conn, params=(BENCH, START, END),
    )
    conn.close()
    if df.empty or len(df) < 2:
        return 0.0
    return (float(df["close"].iloc[-1]) / float(df["close"].iloc[0]) - 1) * 100


def run_one(code: str, cfg: dict) -> dict | None:
    conn = sqlite3.connect(DB_PATH)
    df = load_stock_prices(code, START, END, conn, lookback_days=250)
    conn.close()
    if df is None or len(df) < 30:
        return None
    start_idx = int(df[df["trade_date"] >= START].index.min())
    if pd.isna(start_idx):
        return None
    try:
        strat = RsiBbDualStrategyPlugin(CAPITAL, cfg)
        res = strat.run(df, start_idx)
    except Exception as e:  # noqa: BLE001
        print(f"    [ERR] {code}: {e}")
        return None
    ret = res.get("returns", 0.0)
    trades = res.get("trades", [])
    win_rate, _, _ = calc_win_rate_from_trades(trades)
    dv = res.get("daily_values", [])
    max_dd = 0.0
    if dv:
        vals = [v["portfolio_value"] for v in dv]
        dates = [v.get("date") for v in dv]
        max_dd, _, _ = max_drawdown_with_dates(vals, dates)
    return {
        "code": code, "ret": ret, "win_rate": win_rate,
        "trades": len(trades), "max_dd": max_dd,
        "final_val": CAPITAL * (1 + ret / 100.0),
    }


def summarize(rows: list[dict], idx_ret: float) -> dict:
    rets = [r["ret"] for r in rows]
    return {
        "n": len(rows),
        "mean": float(np.mean(rets)),
        "median": float(np.median(rets)),
        "best": float(np.max(rets)),
        "worst": float(np.min(rets)),
        "n_pos": int(sum(1 for x in rets if x > 0)),
        "n_beat": int(sum(1 for x in rets if x > idx_ret)),
        "win_rate_mean": float(np.mean([r["win_rate"] for r in rows])),
        "trades_mean": float(np.mean([r["trades"] for r in rows])),
        "max_dd_mean": float(np.mean([r["max_dd"] for r in rows])),
        "total_ret": (sum(r["final_val"] for r in rows) / (CAPITAL * len(rows)) - 1) * 100,
    }


# ── 5 套配置（控制变量）──
BASE = dict(config.STRATEGIES["rsi_bb_dual"])
CONFIGS = {
    "C0 基线(旧)":        dict(BASE, rsi_period=9, kelly_win_rate=0.53, regime_filter=False),
    "C1 仅改RSI(14)":     dict(BASE, kelly_win_rate=0.53, regime_filter=False),
    "C2 RSI+凯利":        dict(BASE, regime_filter=False),
    "C3 +MA200门控":      dict(BASE, regime_filter=True, regime_mode="ma", regime_ma=200),
    "C4 +ADX门控":        dict(BASE, regime_filter=True, regime_mode="adx",
                                regime_adx_period=14, regime_adx_threshold=25.0),
    "C5 RSI9+凯利0.57":   dict(BASE, rsi_period=9, kelly_win_rate=0.57, regime_filter=False),
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    codes = build_universe(TOP_N)
    idx_ret = load_benchmark_return()
    print(f"[基准] {BENCH} {START}~{END} 收益 = {idx_ret:+.2f}%\n")

    results = {}
    for name, cfg in CONFIGS.items():
        print(f"\n### 运行 {name}  (RSI{cfg['rsi_period']} "
              f"kelly{cfg['kelly_win_rate']} "
              f"gate={'off' if not cfg['regime_filter'] else cfg['regime_mode']})")
        rows = []
        for i, code in enumerate(codes, 1):
            r = run_one(code, cfg)
            if r:
                rows.append(r)
            tag = f"{r['ret']:+.1f}%" if r else "  n/a"
            if i % 10 == 0 or i == 1:
                print(f"  ...[{i:>2}/{len(codes)}] done, 累计 {len(rows)} 只")
        results[name] = (rows, summarize(rows, idx_ret))
        s = results[name][1]
        print(f"  → 均值 {s['mean']:+.2f}%  中位 {s['median']:+.2f}%  "
              f"正收益 {s['n_pos']}/{s['n']}  均交易 {s['trades_mean']:.1f}")

    # ── 归因表 ──
    print("\n" + "=" * 88)
    print(f"RSI+布林带双确认 —— 逐改动归因（{START}~{END}，沪深300前40，每支{CAPITAL//10000}万）")
    print("=" * 88)
    hdr = f"{'配置':<16}{'均值%':>9}{'中位%':>9}{'总收益%':>9}{'正收益':>8}{'跑赢指':>8}{'胜率%':>8}{'均交易':>8}{'均回撤%':>9}"
    print(hdr)
    print("-" * 88)
    for name in CONFIGS:
        s = results[name][1]
        print(f"{name:<16}{s['mean']:>+9.2f}{s['median']:>+9.2f}{s['total_ret']:>+9.2f}"
              f"{s['n_pos']:>{6}}/{s['n']:<3}{s['n_beat']:>{6}}/{s['n']:<3}"
              f"{s['win_rate_mean']:>8.2f}{s['trades_mean']:>8.1f}{s['max_dd_mean']:>9.2f}")
    print(f"{'沪深300基准':<16}{idx_ret:>+9.2f}")
    print("-" * 88)
    print("逐项贡献（相对上一档）：")
    prev = None
    for name in CONFIGS:
        s = results[name][1]
        if prev is None:
            print(f"  {name:<16} 基线")
        else:
            d_mean = s['mean'] - prev['mean']
            d_trade = s['trades_mean'] - prev['trades_mean']
            print(f"  {name:<16} Δ均值 {d_mean:+.2f}pp   Δ交易 {d_trade:+.1f}")
        prev = s
    print(f"\n[基准 {BENCH}] {idx_ret:+.2f}%")

    # ── 落盘 ──
    cmp_rows = []
    for name in CONFIGS:
        s = results[name][1]
        cmp_rows.append({"配置": name, **s})
    pd.DataFrame(cmp_rows).to_csv(OUT_DIR / "ablation_compare.csv", index=False, encoding="utf-8-sig")
    # 逐股对照（C0 vs C4 最优对比）
    for name, (rows, _) in results.items():
        pd.DataFrame(rows).to_csv(
            OUT_DIR / f"ablation_{name.split()[0]}.csv", index=False, encoding="utf-8-sig")
    print(f"\n[已保存] {OUT_DIR}/  (ablation_compare.csv + ablation_C0..C4.csv)")


if __name__ == "__main__":
    main()
