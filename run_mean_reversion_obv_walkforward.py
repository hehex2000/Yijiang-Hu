# -*- coding: utf-8 -*-
"""
均值回归策略 —— OBV 量能背离 walk-forward 稳定性检验（含 regime-gate 组合复核）
==============================================================================

目的：确认 OBV 背离的增量不是"全样本平均掩盖的单期幸运"，即非样本内过拟合；
      并复核"OBV + 大盘趋势门控(regime-gate)"组合是否消除了裸 OBV 的熊市失效。

      固定参数(K=10, 默认口径)，样本切成连续不重叠的 OOS 窗口，逐窗验证
      OBV 增量(Δ=CO−C0)在每窗内是否仍为正。单一变量隔离：仅 obv_divergence，
      且 C0/CO 两侧同等施加 regime-gate（公平对照）。

      每窗 8 组：
        无 gate : C0s(基线/简单) COs(+OBV/简单) C0r(基线/真实) COr(+OBV/真实)
        +gate   : C0gs(基线/简单) COgs(+OBV/简单) C0gr(基线/真实) COgr(+OBV/真实)

固定：股票池(沪深300 as-of 20190101 快照前40，全程不变，消除幸存者偏差与 look-ahead)、
      参数(全同 config，OBV K=10)、行情/手续费口径。
注：每窗用 start-250d warmup 单独切片，指标预热充分、无跨窗信息泄漏；
    regime 掩码仅取 <= 窗口结束日 的指数数据，无未来函数。
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

UNIV_START = "20190101"
FULL_START = "20190101"   # 行情 warmup 从 start-250d 自动补全
FULL_END = "20260731"
CAPITAL = 100_000
TOP_N = 40
BENCH = "000300.SH"
OUT_DIR = HERE / "data" / "results" / "mean_reversion_obv_walkforward"

# 连续不重叠窗口（每窗 ~18 个月），覆盖全样本
WINDOWS = [
    ("W1 2019H1-2020H1", "20190101", "20200630"),
    ("W2 2020H2-2021",   "20200701", "20211231"),
    ("W3 2022H1-2023H1", "20220101", "20230630"),
    ("W4 2023H2-2024",   "20230701", "20241231"),
    ("W5 2025-2026H1",   "20250101", "20260731"),
]

BASE = dict(config.STRATEGIES["mean_reversion"])
# 8 组：无 gate / +gate × C0/+OBV × 简单/真实成本
CONFIGS = [
    ("C0s",  dict(BASE, obv_divergence=False, market_regime_gate=False, real_cost=False)),
    ("COs",  dict(BASE, obv_divergence=True,  market_regime_gate=False, real_cost=False)),
    ("C0r",  dict(BASE, obv_divergence=False, market_regime_gate=False, real_cost=True)),
    ("COr",  dict(BASE, obv_divergence=True,  market_regime_gate=False, real_cost=True)),
    ("C0gs", dict(BASE, obv_divergence=False, market_regime_gate=True,  real_cost=False)),
    ("COgs", dict(BASE, obv_divergence=True,  market_regime_gate=True,  real_cost=False)),
    ("C0gr", dict(BASE, obv_divergence=False, market_regime_gate=True,  real_cost=True)),
    ("COgr", dict(BASE, obv_divergence=True,  market_regime_gate=True,  real_cost=True)),
]

_PRICE_CACHE: dict[str, pd.DataFrame] = {}


def build_universe(n: int = TOP_N) -> list[str]:
    df = _get_index_constituents_from_db(BENCH, as_of_date=UNIV_START)
    if df is None or df.empty:
        return []
    codes = sorted(df["code"].tolist())[:n]
    print(f"[股票池] 沪深300 as-of {UNIV_START} 快照 → 取 {len(codes)} 只（全程不变）")
    return codes


def get_full_df(code: str) -> pd.DataFrame | None:
    if code not in _PRICE_CACHE:
        import sqlite3
        conn = sqlite3.connect(config.DATA["local_db_path"])
        df = load_stock_prices(code, FULL_START, FULL_END, conn, lookback_days=250)
        conn.close()
        _PRICE_CACHE[code] = df
    return _PRICE_CACHE[code]


def run_one(code: str, cfg: dict, ws: str, we: str) -> dict | None:
    full = get_full_df(code)
    if full is None or len(full) < 30:
        return None
    # 切窗：保留 end 之前全部行（含 warmup），仅从 ws 起交易
    sub = full[full["trade_date"] <= we].copy()
    sub = sub.reset_index(drop=True)
    mask = sub["trade_date"] >= ws
    if not mask.any():
        return None
    start_idx = int(mask.idxmax())
    try:
        strat = MeanReversionStrategyPlugin(CAPITAL, cfg)
        res = strat.run(sub, start_idx)
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
        "obv_entries_mean": float(np.mean([r.get("obv_entries", 0) for r in rows])),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    codes = build_universe(TOP_N)
    print(f"[窗口] {len(WINDOWS)} 个不重叠 OOS 窗口；每窗 {len(CONFIGS)} 组×{len(codes)} 只\n")

    # results[win_name][cfg_name] = summary dict
    results: dict[str, dict[str, dict]] = {}
    for wname, ws, we in WINDOWS:
        print(f"\n##### 窗口 {wname}  ({ws}~{we})")
        results[wname] = {}
        for cname, cfg in CONFIGS:
            rows = []
            for code in codes:
                r = run_one(code, cfg, ws, we)
                if r:
                    rows.append(r)
            s = summarize(rows)
            results[wname][cname] = s
            gate = "+gate" if cfg["market_regime_gate"] else "无gate"
            cost = "真实" if cfg["real_cost"] else "简单"
            obv = "+OBV" if cfg["obv_divergence"] else "基线"
            print(f"  {cname:<5} {gate:<7}{cost:<3}{obv:<4} 均值 {s['mean']:>+7.2f}%  "
                  f"中位 {s['median']:>+7.2f}%  正 {s['n_pos']}/{s['n']}  "
                  f"胜率 {s['win_rate_mean']:>5.2f}  交易 {s['trades_mean']:>5.1f}  "
                  f"OBV买 {s['obv_entries_mean']:>5.1f}")

    # ── 逐窗 Δ 表 ──
    print("\n" + "=" * 118)
    print(f"OBV 背离 walk-forward（{UNIV_START}~{FULL_END}，沪深300前{TOP_N}，每支{CAPITAL//10000}万）")
    print("=" * 118)
    hdr = (f"{'窗口':<18}{'Δ简单':>9}{'Δ简单+gate':>11}{'Δ真实':>9}{'Δ真实+gate':>11}"
           f"{'COg胜率':>9}{'COg交易':>9}{'COg回撤':>9}")
    print(hdr)
    print("-" * 118)
    cnt = {"s": 0, "sg": 0, "r": 0, "rg": 0}
    rows_out = []
    for wname, ws, we in WINDOWS:
        R = results[wname]
        ds = R["COs"]["mean"] - R["C0s"]["mean"]
        dsg = R["COgs"]["mean"] - R["C0gs"]["mean"]
        dr = R["COr"]["mean"] - R["C0r"]["mean"]
        drg = R["COgr"]["mean"] - R["C0gr"]["mean"]
        cnt["s"] += 1 if ds > 0 else 0
        cnt["sg"] += 1 if dsg > 0 else 0
        cnt["r"] += 1 if dr > 0 else 0
        cnt["rg"] += 1 if drg > 0 else 0
        print(f"{wname:<18}{ds:>+9.2f}{dsg:>+11.2f}{dr:>+9.2f}{drg:>+11.2f}"
              f"{R['COgs']['win_rate_mean']:>9.2f}{R['COgs']['trades_mean']:>9.1f}"
              f"{R['COgs']['max_dd_mean']:>9.2f}")
        rows_out.append({
            "窗口": wname, "起": ws, "止": we,
            "Δ简单": ds, "Δ简单+gate": dsg, "Δ真实": dr, "Δ真实+gate": drg,
            "COg均值": R["COgs"]["mean"], "C0g均值": R["C0gs"]["mean"],
            "COg中位": R["COgs"]["median"], "COg胜率": R["COgs"]["win_rate_mean"],
            "COg交易": R["COgs"]["trades_mean"], "COg回撤": R["COgs"]["max_dd_mean"],
            "OBV买/只": R["COgs"]["obv_entries_mean"],
        })
    print("-" * 118)

    print(f"\n[稳定性] 正窗口数 / {len(WINDOWS)}：")
    print(f"  裸 OBV       简单 {cnt['s']}/{len(WINDOWS)}   真实成本 {cnt['r']}/{len(WINDOWS)}")
    print(f"  OBV+regime   简单 {cnt['sg']}/{len(WINDOWS)}   真实成本 {cnt['rg']}/{len(WINDOWS)}")
    beat = sum(1 for k in ("s", "sg", "r", "rg") if cnt[k] == len(WINDOWS))
    tag = "✅ 全部窗口为正" if beat == 4 else (
        "✅ 至少一种口径全正" if beat else "⚠️ 无口径达到全正")
    print(f"\n[判据] {tag}")

    # ── 落盘 ──
    pd.DataFrame(rows_out).to_csv(OUT_DIR / "walkforward.csv", index=False, encoding="utf-8-sig")
    detail = []
    for wname, _, _ in WINDOWS:
        for cname, _ in CONFIGS:
            detail.append({"窗口": wname, "配置": cname, **results[wname][cname]})
    pd.DataFrame(detail).to_csv(OUT_DIR / "walkforward_detail.csv", index=False, encoding="utf-8-sig")
    print(f"\n[已保存] {OUT_DIR}/  (walkforward.csv + walkforward_detail.csv)")


if __name__ == "__main__":
    main()
