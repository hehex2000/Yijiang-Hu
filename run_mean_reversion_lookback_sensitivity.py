# -*- coding: utf-8 -*-
"""
均值回归策略 —— 背离回看窗口 K 参数敏感性扫描
==============================================

目的：确认 %B / OBV 背离的增量不是"K=10 恰好挑在最优点的运气"。
      K=10 目前是插件默认值，从未扫描过（总报告 §11 遗留待办第 1 项）。

判据（跑前预先定死，避免事后找补）：
  稳健(平台型)  : Δ 在多数 K 为正且波动平缓 → 参数不敏感，K=10 非挑参
  过拟合(尖峰型): Δ 仅在 K=10 突高、邻近 K 塌到近零或转负 → 边缘对参数敏感

扫描：
  OBV : obv_lookback  ∈ {5, 8, 10, 12, 15, 20}  （重点，邻域加密）
  %B  : pctb_lookback ∈ {5, 10, 15, 20}
  成本: 简单 / 真实(real_cost) 两种口径各扫一遍
  基线: C0 两种成本各跑一次，被同口径的各 K 共用（隔离单一变量）

固定：股票池(沪深300 as-of 20190101 快照前40)、期间、本金、其余参数全同 config。
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
OUT_DIR = HERE / "data" / "results" / "mean_reversion_lookback_sensitivity"

OBV_KS = [5, 8, 10, 12, 15, 20]
PCTB_KS = [5, 10, 15, 20]

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
    if dv:
        vals = [v["portfolio_value"] for v in dv]
        dates = [v.get("date") for v in dv]
        max_dd, _, _ = max_drawdown_with_dates(vals, dates)
    return {
        "code": code, "ret": ret, "win_rate": win_rate,
        "trades": len(trades), "max_dd": max_dd,
        "pctb_entries": getattr(strat, "_pctb_entries", 0),
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
        "pctb_entries_mean": float(np.mean([r.get("pctb_entries", 0) for r in rows])),
        "obv_entries_mean": float(np.mean([r.get("obv_entries", 0) for r in rows])),
    }


BASE = dict(config.STRATEGIES["mean_reversion"])


def base_cfg(real: bool) -> dict:
    return dict(BASE, pctb_divergence=False, obv_divergence=False,
                market_regime_gate=False, real_cost=real)


def obv_cfg(k: int, real: bool) -> dict:
    return dict(BASE, pctb_divergence=False, obv_divergence=True, obv_lookback=k,
                market_regime_gate=False, real_cost=real)


def pctb_cfg(k: int, real: bool) -> dict:
    return dict(BASE, pctb_divergence=True, pctb_lookback=k, obv_divergence=False,
                market_regime_gate=False, real_cost=real)


def run_group(codes, cfg, label):
    rows = []
    for i, code in enumerate(codes, 1):
        r = run_one(code, cfg)
        if r:
            rows.append(r)
        if i % 20 == 0 or i == 1:
            print(f"  ...[{i:>2}/{len(codes)}] {label}")
    return summarize(rows)


def print_table(ks, res_by_k, base_s, title, entry_key, cost_label):
    print(f"\n{'=' * 100}")
    print(f"{title}（{cost_label}）　基线均值 {base_s['mean']:+.2f}%  中位 {base_s['median']:+.2f}%")
    print("=" * 100)
    print(f"{'K':>5}{'均值%':>10}{'Δ均值':>10}{'中位%':>10}{'Δ中位':>10}"
          f"{'胜率%':>8}{'均交易':>8}{'均回撤%':>9}{'买入/只':>9}")
    print("-" * 100)
    for k in ks:
        s = res_by_k[k]
        print(f"{k:>5}{s['mean']:>+10.2f}{s['mean'] - base_s['mean']:>+10.2f}"
              f"{s['median']:>+10.2f}{s['median'] - base_s['median']:>+10.2f}"
              f"{s['win_rate_mean']:>8.2f}{s['trades_mean']:>8.1f}"
              f"{s['max_dd_mean']:>9.2f}{s[entry_key]:>9.1f}")
    print("-" * 100)
    ds = [res_by_k[k]["mean"] - base_s["mean"] for k in ks]
    n_pos = sum(1 for x in ds if x > 0)
    best_k = ks[int(np.argmax(ds))]
    print(f"  Δ均值>0 的 K: {n_pos}/{len(ks)}   最优 K={best_k} (Δ{max(ds):+.2f}pp)   "
          f"Δ区间 [{min(ds):+.2f}, {max(ds):+.2f}]pp  极差 {max(ds) - min(ds):.2f}pp")
    return ds, best_k


def spike_check(ks, ds, focal, neighbors):
    """焦点 K 是否显著高于邻近 K（尖峰 = 挑参嫌疑）。"""
    d = dict(zip(ks, ds))
    if focal not in d or not all(n in d for n in neighbors):
        return None
    nb = np.mean([d[n] for n in neighbors])
    return d[focal], nb, d[focal] - nb


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    codes = build_universe(TOP_N)

    out_rows = []
    verdicts = []

    for real, cost_label in ((False, "简单成本"), (True, "真实成本")):
        print(f"\n\n########## {cost_label} ##########")
        base_s = run_group(codes, base_cfg(real), f"基线 {cost_label}")
        print(f"  基线 → 均值 {base_s['mean']:+.2f}%  中位 {base_s['median']:+.2f}%  "
              f"交易 {base_s['trades_mean']:.1f}")

        # ── OBV ──
        obv_res = {}
        for k in OBV_KS:
            obv_res[k] = run_group(codes, obv_cfg(k, real), f"OBV K={k} {cost_label}")
        ds, best_k = print_table(OBV_KS, obv_res, base_s,
                                 "OBV 量能背离 —— obv_lookback 敏感性",
                                 "obv_entries_mean", cost_label)
        chk = spike_check(OBV_KS, ds, 10, [8, 12])
        if chk:
            f, nb, gap = chk
            print(f"  尖峰检验: Δ(K=10)={f:+.2f}pp  邻近(K=8,12)均值={nb:+.2f}pp  差={gap:+.2f}pp")
            verdicts.append(("OBV", cost_label, best_k, ds, chk))
        for k in OBV_KS:
            s = obv_res[k]
            out_rows.append({"信号": "OBV", "成本": cost_label, "K": k,
                             "均值": s["mean"], "Δ均值": s["mean"] - base_s["mean"],
                             "中位": s["median"], "Δ中位": s["median"] - base_s["median"],
                             "胜率": s["win_rate_mean"], "交易": s["trades_mean"],
                             "回撤": s["max_dd_mean"], "买入/只": s["obv_entries_mean"]})

        # ── %B ──
        pctb_res = {}
        for k in PCTB_KS:
            pctb_res[k] = run_group(codes, pctb_cfg(k, real), f"%B K={k} {cost_label}")
        ds2, best_k2 = print_table(PCTB_KS, pctb_res, base_s,
                                   "%B 看涨背离 —— pctb_lookback 敏感性",
                                   "pctb_entries_mean", cost_label)
        chk2 = spike_check(PCTB_KS, ds2, 10, [5, 15])
        if chk2:
            f, nb, gap = chk2
            print(f"  尖峰检验: Δ(K=10)={f:+.2f}pp  邻近(K=5,15)均值={nb:+.2f}pp  差={gap:+.2f}pp")
            verdicts.append(("%B", cost_label, best_k2, ds2, chk2))
        for k in PCTB_KS:
            s = pctb_res[k]
            out_rows.append({"信号": "%B", "成本": cost_label, "K": k,
                             "均值": s["mean"], "Δ均值": s["mean"] - base_s["mean"],
                             "中位": s["median"], "Δ中位": s["median"] - base_s["median"],
                             "胜率": s["win_rate_mean"], "交易": s["trades_mean"],
                             "回撤": s["max_dd_mean"], "买入/只": s["pctb_entries_mean"]})

    # ── 总判据 ──
    print("\n" + "=" * 100)
    print("参数敏感性总判据（跑前定死：平台型=稳健 / 尖峰型=挑参嫌疑）")
    print("=" * 100)
    for sig, cost_label, best_k, ds, chk in verdicts:
        f, nb, gap = chk
        n_pos = sum(1 for x in ds if x > 0)
        if n_pos == len(ds) and gap < 1.0:
            v = "✅ 平台型（全 K 为正且无尖峰）→ 参数不敏感，K=10 非挑参"
        elif n_pos >= len(ds) - 1 and gap < 2.0:
            v = "✅ 基本平台型（多数 K 为正、尖峰温和）→ 参数敏感度可接受"
        elif best_k == 10 and gap >= 2.0:
            v = "⚠️ 尖峰型（K=10 为最优点且显著高于邻域）→ 疑似挑参，边缘不可信"
        else:
            v = "⚠️ 混合型（部分 K 转负 / 波动大）→ 参数敏感，需谨慎"
        print(f"  {sig:<4}{cost_label:<6} 最优K={best_k:<3} Δ>0: {n_pos}/{len(ds)}  "
              f"尖峰差={gap:+.2f}pp  {v}")

    # ── 落盘 ──
    pd.DataFrame(out_rows).to_csv(OUT_DIR / "lookback_sensitivity.csv",
                                  index=False, encoding="utf-8-sig")
    print(f"\n[已保存] {OUT_DIR}/lookback_sensitivity.csv")


if __name__ == "__main__":
    main()
