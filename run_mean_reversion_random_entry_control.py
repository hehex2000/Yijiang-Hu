# -*- coding: utf-8 -*-
"""
均值回归策略 —— 随机入场控制组（暴露度 / beta 对照实验）
========================================================

问题：OBV 背离把交易数从 26.6 拉到 91.3/只，收益 +10.74% → +15.34%。
      但这个增量有多少是真 alpha，多少只是"交易更多 = 在市时间更长 =
      吃到 2019-2026 长牛(HS300 +54.51%)的 beta"？
      敏感性扫描已发现 Δ 与交易数强正相关（K=5:106笔→+5.51pp, K=20:74笔→+2.95pp），
      高度提示 beta 嫌疑。

方法（随机入场控制组）：
  与 OBV **完全相同的准入条件**（空仓 + 布林带未张开 + 大盘允许），
  仅把"何时入场"换成伯努利随机触发（概率 p）；**出场逻辑与基线完全一致**
  （Z>2 或 RSI>70 + 止损），不做任何改动。
  → 扫描 p 得到"入场笔数 → 收益"响应曲线；
  → 再与 OBV 在**同等入场笔数（38/只）**下比较。

判据（跑前定死）：
  OBV 收益 ≈ 随机控制组（同笔数）  → 增量几乎全是 beta，OBV 无 alpha，应降级
  OBV 收益 显著 > 随机控制组        → OBV 入场时点有真 skill，alpha 成立

对照基准（已由前序实验测得，同口径）：
  C0 基线          : 均值 +10.74%，交易 26.6/只
  CO +OBV背离      : 均值 +15.34%，交易 91.3/只，OBV 入场 38.0/只

随机性处理：每只股票独立随机流（seed = base + 股票序号），
            并跑 N_ROUNDS 轮不同 seed 取均值，压低随机噪声。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from run_backtest import (  # noqa: E402
    load_stock_prices,
    calc_win_rate_from_trades,
    max_drawdown_with_dates,
    _get_index_constituents_from_db,
)
from backtest.mean_reversion_plugin import MeanReversionStrategyPlugin  # noqa: E402
import config  # noqa: E402

START = "20190101"
END = "20260731"
CAPITAL = 100_000
TOP_N = 40
BENCH = "000300.SH"
OUT_DIR = HERE / "data" / "results" / "mean_reversion_random_entry_control"

# 触发概率扫描：覆盖从"接近基线"到"远超 OBV"的暴露度区间
PROBS = [0.01, 0.02, 0.03, 0.045, 0.06, 0.09]
N_ROUNDS = 3          # 每档 p 跑 3 个独立 seed 轮次取均值
REAL_COST = True      # 同时跑真实成本口径

REFERENCE = {
    "C0 基线": {"mean": 10.74, "trades": 26.6, "entries": 0.0},
    "CO +OBV": {"mean": 15.34, "trades": 91.3, "entries": 38.0},
}

_PRICE_CACHE: dict[str, pd.DataFrame] = {}


def build_universe(n: int = TOP_N) -> list[str]:
    df = _get_index_constituents_from_db(BENCH, as_of_date=START)
    if df is None or df.empty:
        return []
    codes = sorted(df["code"].tolist())[:n]
    print(f"[股票池] 沪深300 as-of {START} 快照 → 取 {len(codes)} 只")
    return codes


def run_one(code: str, cfg: dict) -> dict | None:
    import sqlite3
    if code not in _PRICE_CACHE:
        conn = sqlite3.connect(config.DATA["local_db_path"])
        df = load_stock_prices(code, START, END, conn, lookback_days=250)
        conn.close()
        _PRICE_CACHE[code] = df
    df = _PRICE_CACHE[code]
    if df is None or len(df) < 30:
        return None
    df = df.reset_index(drop=True)
    start_idx = int(df[df["trade_date"] >= START].index.min())
    if pd.isna(start_idx):
        return None
    try:
        strat = MeanReversionStrategyPlugin(CAPITAL, cfg)
        res = strat.run(df, start_idx)
    except Exception as e:  # noqa: BLE001
        print(f"    [ERR] {code}: {e}")
        return None
    ret = res.get("returns", 0.0)
    trades = res.get("trades", [])
    win_rate, _, _ = calc_win_rate_from_trades(trades)
    dv = res.get("daily_values", [])
    max_dd = 0.0
    exposure = 0.0
    if dv:
        vals = [v["portfolio_value"] for v in dv]
        dates = [v.get("date") for v in dv]
        max_dd, _, _ = max_drawdown_with_dates(vals, dates)
    return {
        "code": code, "ret": ret, "win_rate": win_rate,
        "trades": len(trades), "max_dd": max_dd,
        "rand_entries": getattr(strat, "_rand_entries", 0),
        "obv_entries": getattr(strat, "_obv_entries", 0),
    }


def summarize(rows: list[dict]) -> dict:
    rets = [r["ret"] for r in rows]
    return {
        "n": len(rows),
        "mean": float(np.mean(rets)),
        "median": float(np.median(rets)),
        "n_pos": int(sum(1 for x in rets if x > 0)),
        "win_rate_mean": float(np.mean([r["win_rate"] for r in rows])),
        "trades_mean": float(np.mean([r["trades"] for r in rows])),
        "max_dd_mean": float(np.mean([r["max_dd"] for r in rows])),
        "rand_entries_mean": float(np.mean([r.get("rand_entries", 0) for r in rows])),
    }


BASE = dict(config.STRATEGIES["mean_reversion"])


def rand_cfg(p: float, seed: int, real: bool) -> dict:
    return dict(BASE, pctb_divergence=False, obv_divergence=False,
                market_regime_gate=False, real_cost=real,
                random_entry=True, random_entry_p=p, random_entry_seed=seed)


def run_group(codes, p, seed, real, label):
    """跑一组：每只股票独立随机流（seed = 轮次基 * 1000 + 股票序号）。"""
    rows = []
    for i, code in enumerate(codes):
        cfg = rand_cfg(p, seed * 1000 + i, real)
        r = run_one(code, cfg)
        if r:
            rows.append(r)
    s = summarize(rows)
    print(f"  p={p:<6} seed轮={seed}  均值 {s['mean']:>+7.2f}%  中位 {s['median']:>+7.2f}%  "
          f"交易 {s['trades_mean']:>6.1f}  随机入场 {s['rand_entries_mean']:>5.1f}/只  "
          f"胜率 {s['win_rate_mean']:>5.2f}  回撤 {s['max_dd_mean']:>5.2f}")
    return s


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    codes = build_universe(TOP_N)

    out_rows = []
    curves = {}

    for real, cost_label in ((False, "简单成本"), (True, "真实成本")):
        print(f"\n\n########## 随机入场控制组 —— {cost_label} ##########")
        agg = {}
        for p in PROBS:
            per_round = []
            for rnd in range(N_ROUNDS):
                s = run_group(codes, p, rnd, real, cost_label)
                per_round.append(s)
            # 多轮 seed 取均值，压低随机噪声
            agg[p] = {
                "mean": float(np.mean([x["mean"] for x in per_round])),
                "median": float(np.mean([x["median"] for x in per_round])),
                "trades": float(np.mean([x["trades_mean"] for x in per_round])),
                "entries": float(np.mean([x["rand_entries_mean"] for x in per_round])),
                "win_rate": float(np.mean([x["win_rate_mean"] for x in per_round])),
                "max_dd": float(np.mean([x["max_dd_mean"] for x in per_round])),
                "spread": float(np.max([x["mean"] for x in per_round])
                                - np.min([x["mean"] for x in per_round])),
            }
            a = agg[p]
            out_rows.append({"成本": cost_label, "p": p,
                             "随机入场/只": a["entries"], "均交易": a["trades"],
                             "均值": a["mean"], "中位": a["median"],
                             "胜率": a["win_rate"], "回撤": a["max_dd"],
                             "轮间极差": a["spread"]})
        curves[cost_label] = agg

        # ── 响应曲线 ──
        print(f"\n{'=' * 104}")
        print(f"暴露度—收益响应曲线（{cost_label}，随机入场，每档 {N_ROUNDS} 个 seed 轮次均值）")
        print("=" * 104)
        print(f"{'p':>7}{'随机入场/只':>13}{'均交易':>10}{'均值%':>10}{'中位%':>10}"
              f"{'胜率%':>8}{'回撤%':>8}{'轮间极差':>10}")
        print("-" * 104)
        for p in PROBS:
            a = agg[p]
            print(f"{p:>7}{a['entries']:>13.1f}{a['trades']:>10.1f}"
                  f"{a['mean']:>+10.2f}{a['median']:>+10.2f}"
                  f"{a['win_rate']:>8.2f}{a['max_dd']:>8.2f}{a['spread']:>10.2f}")
        print("-" * 104)

    # ── 与 OBV 在同等入场笔数下对比（线性插值）──
    print("\n" + "=" * 104)
    print("关键对比：随机控制组 vs OBV（按 OBV 的入场笔数 38.0/只 对齐）")
    print("=" * 104)
    for cost_label, obv_mean in (("简单成本", 15.34), ("真实成本", 13.75)):
        agg = curves[cost_label]
        ps = sorted(agg.keys())
        ents = [agg[p]["entries"] for p in ps]
        means = [agg[p]["mean"] for p in ps]
        # 在 OBV 笔数 38.0 处线性插值
        target = 38.0
        interp = None
        for i in range(len(ps) - 1):
            if ents[i] <= target <= ents[i + 1] and ents[i + 1] > ents[i]:
                w = (target - ents[i]) / (ents[i + 1] - ents[i])
                interp = means[i] + w * (means[i + 1] - means[i])
                break
        base_mean = 10.74 if cost_label == "简单成本" else 10.27
        if interp is None:
            print(f"  {cost_label}: 未能插值到 38.0 笔（实测笔数区间 {min(ents):.1f}~{max(ents):.1f}）")
            continue
        gap = obv_mean - interp
        print(f"\n  ▌ {cost_label}")
        print(f"     基线 C0                    : {base_mean:+.2f}%")
        print(f"     随机控制组(对齐 38.0 笔)   : {interp:+.2f}%   ← beta 成分")
        print(f"     OBV 背离(38.0 笔)          : {obv_mean:+.2f}%")
        print(f"     OBV − 随机控制组            : {gap:+.2f}pp  ← 真 alpha 部分")
        ratio = (obv_mean - base_mean)
        if abs(ratio) > 1e-9:
            print(f"     beta 占比 ≈ {100 * (interp - base_mean) / ratio:.0f}%，"
                  f"alpha 占比 ≈ {100 * gap / ratio:.0f}%")
        if gap >= 1.0:
            print("     判定：✅ OBV 入场时点有真 skill（alpha 成立）")
        elif gap >= 0.3:
            print("     判定：⚠️ OBV 有边际 alpha，但增量大部分来自 beta")
        else:
            print("     判定：❌ 增量几乎全是 beta，OBV 无超越暴露度的 alpha，应降级")

    # ── 落盘 ──
    pd.DataFrame(out_rows).to_csv(OUT_DIR / "random_entry_control.csv",
                                  index=False, encoding="utf-8-sig")
    print(f"\n[已保存] {OUT_DIR}/random_entry_control.csv")


if __name__ == "__main__":
    main()
