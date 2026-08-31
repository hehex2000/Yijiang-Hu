# -*- coding: utf-8 -*-
"""
均值回归策略 —— MFI / OBV 量能确认 A/B 归因
============================================

目的：验证"量能确认"能否给现有均值回归策略带来真增量。控制单一变量，
      只开/关 mfi_filter（量能闸门）或 obv_divergence（量能背离），分别隔离：

  MFI 闸门    : 下跌需伴随放量抛压(MFI<超卖)才确认"真超卖"，过滤无量阴跌。
                作用在 buy_signal 上（闸门，减少交易）。
  OBV 背离    : 价格创 K 日新低但 OBV 未同步新低=下跌未获量能确认(抛压枯竭)，
                作为额外买入触发（平行 %B 背离，增加交易）。

  C0   基线(两者均关) : mfi_filter=False, obv_divergence=False   ← 当前默认
  CM   +MFI闸门       : mfi_filter=True
  CO   +OBV背离       : obv_divergence=True

三对照（每组内部仅单变量不同）：
  ① 简单成本(基线口径)  ② 大盘趋势门控  ③ 真实分科目成本

固定：股票池(沪深300 as-of 起始日快照前40，非幸存者偏差口径)、期间、每支本金、
      行情/手续费口径、其余参数全同 config。
注：build_universe 直接用 as_of_date=START 快照，避免幸存者偏差(吸取 headfake 教训)。
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
OUT_DIR = HERE / "data" / "results" / "mean_reversion_mfi_obv_ablation"

_PRICE_CACHE: dict[str, pd.DataFrame] = {}


def build_universe(n: int = TOP_N) -> list[str]:
    """沪深300 成分股（as_of START 快照，消除幸存者偏差）。"""
    df = _get_index_constituents_from_db(BENCH, as_of_date=START)
    if df is None or df.empty:
        print(f"[股票池] ⚠️ 未取到 {BENCH} 成分股")
        return []
    codes = sorted(df["code"].tolist())[:n]  # 按代码排序取前 n，确定性
    print(f"[股票池] 沪深300 as-of {START} 快照 → 取 {len(codes)} 只（非幸存者偏差口径）")
    return codes


def load_benchmark_return() -> float:
    import sqlite3
    conn = sqlite3.connect(config.DATA["local_db_path"])
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
    import sqlite3
    if code not in _PRICE_CACHE:
        conn = sqlite3.connect(config.DATA["local_db_path"])
        df = load_stock_prices(code, START, END, conn, lookback_days=250)
        conn.close()
        _PRICE_CACHE[code] = df
    df = _PRICE_CACHE[code]
    if df is None or len(df) < 30:
        return None
    df = df.reset_index(drop=True)  # 保证索引 0-based 连续，start_idx 计算稳健
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
        "pctb_overlap": getattr(strat, "_pctb_overlap", 0),
        "mfi_suppressed": getattr(strat, "_mfi_suppressed", 0),
        "obv_entries": getattr(strat, "_obv_entries", 0),
        "obv_overlap": getattr(strat, "_obv_overlap", 0),
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
        "mfi_suppressed_mean": float(np.mean([r.get("mfi_suppressed", 0) for r in rows])),
        "obv_entries_mean": float(np.mean([r.get("obv_entries", 0) for r in rows])),
        "obv_overlap_mean": float(np.mean([r.get("obv_overlap", 0) for r in rows])),
    }


BASE = dict(config.STRATEGIES["mean_reversion"])
# 9 组：基线 × 两种量能特征 × 三种成本口径。每组内部仅单变量不同，隔离变量。
CONFIGS = [
    ("C0 基线",       dict(BASE, mfi_filter=False, obv_divergence=False, market_regime_gate=False, real_cost=False)),
    ("CM +MFI闸门",   dict(BASE, mfi_filter=True,  obv_divergence=False, market_regime_gate=False, real_cost=False)),
    ("CO +OBV背离",   dict(BASE, mfi_filter=False, obv_divergence=True,  market_regime_gate=False, real_cost=False)),
    ("C0g regime门控", dict(BASE, mfi_filter=False, obv_divergence=False, market_regime_gate=True,  real_cost=False)),
    ("CMg +MFI+regime", dict(BASE, mfi_filter=True,  obv_divergence=False, market_regime_gate=True,  real_cost=False)),
    ("COg +OBV+regime", dict(BASE, mfi_filter=False, obv_divergence=True,  market_regime_gate=True,  real_cost=False)),
    ("C0r 真实成本",  dict(BASE, mfi_filter=False, obv_divergence=False, market_regime_gate=False, real_cost=True)),
    ("CMr +MFI+成本",  dict(BASE, mfi_filter=True,  obv_divergence=False, market_regime_gate=False, real_cost=True)),
    ("COr +OBV+成本",  dict(BASE, mfi_filter=False, obv_divergence=True,  market_regime_gate=False, real_cost=True)),
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    codes = build_universe(TOP_N)
    idx_ret = load_benchmark_return()
    print(f"[基准] {BENCH} {START}~{END} 收益 = {idx_ret:+.2f}%\n")

    results = {}
    for name, cfg in CONFIGS:
        tag = []
        if cfg.get("market_regime_gate"):
            tag.append("regime门控")
        if cfg.get("real_cost"):
            tag.append("真实成本")
        if cfg.get("mfi_filter"):
            tag.append("MFI闸门")
        if cfg.get("obv_divergence"):
            tag.append("OBV背离")
        print(f"\n### 运行 {name}  ({'/'.join(tag) if tag else '基线'})")
        rows = []
        for i, code in enumerate(codes, 1):
            r = run_one(code, cfg)
            if r:
                rows.append(r)
            if i % 10 == 0 or i == 1:
                print(f"  ...[{i:>2}/{len(codes)}] done, 累计 {len(rows)} 只")
        results[name] = (rows, summarize(rows, idx_ret))
        s = results[name][1]
        print(f"  → 均值 {s['mean']:+.2f}%  中位 {s['median']:+.2f}%  "
              f"正收益 {s['n_pos']}/{s['n']}  均交易 {s['trades_mean']:.1f}")

    # ── 归因表（全部 9 组）──
    print("\n" + "=" * 104)
    print(f"均值回归 MFI/OBV 量能确认 A/B（{START}~{END}，沪深300前{TOP_N}，每支{CAPITAL//10000}万）")
    print("=" * 104)
    hdr = (f"{'配置':<16}{'均值%':>9}{'中位%':>9}{'正收益':>8}{'跑赢指':>8}"
           f"{'胜率%':>8}{'均交易':>8}{'均回撤%':>9}{'MFI弃/只':>9}{'OBV买/只':>9}")
    print(hdr)
    print("-" * 104)
    for name, _ in CONFIGS:
        s = results[name][1]
        print(f"{name:<16}{s['mean']:>+9.2f}{s['median']:>+9.2f}"
              f"{s['n_pos']:>{6}}/{s['n']:<3}{s['n_beat']:>{6}}/{s['n']:<3}"
              f"{s['win_rate_mean']:>8.2f}{s['trades_mean']:>8.1f}{s['max_dd_mean']:>9.2f}"
              f"{s['mfi_suppressed_mean']:>9.1f}{s['obv_entries_mean']:>9.1f}")
    print(f"{'沪深300基准':<16}{idx_ret:>+9.2f}")
    print("-" * 104)

    def delta_block(a, b, title):
        ca, cb = results[a][1], results[b][1]
        print(f"\n▌ {title}（{b} − {a}，隔离单一变量）")
        print(f"  Δ均值 {cb['mean'] - ca['mean']:+.2f}pp   "
              f"Δ中位 {cb['median'] - ca['median']:+.2f}pp")
        print(f"  Δ胜率 {cb['win_rate_mean'] - ca['win_rate_mean']:+.2f}pp   "
              f"Δ交易 {cb['trades_mean'] - ca['trades_mean']:+.1f}   "
              f"Δ回撤 {cb['max_dd_mean'] - ca['max_dd_mean']:+.2f}pp")
        if cb['mfi_suppressed_mean'] > 0 or ca['mfi_suppressed_mean'] > 0:
            print(f"  MFI闸门丢弃 {cb['mfi_suppressed_mean']:.1f}/只 (基线 {ca['mfi_suppressed_mean']:.1f}/只)")
        if cb['obv_entries_mean'] > 0 or ca['obv_entries_mean'] > 0:
            print(f"  OBV买入 {cb['obv_entries_mean']:.1f}/只 (重叠 buy_signal {cb['obv_overlap_mean']:.1f}/只)")

    print("\n======== MFI 闸门（对 buy_signal 加量能确认，减少交易）========")
    delta_block("C0 基线", "CM +MFI闸门", "① 简单成本口径")
    delta_block("C0g regime门控", "CMg +MFI+regime", "② 大盘趋势门控")
    delta_block("C0r 真实成本", "CMr +MFI+成本", "③ 真实分科目成本")

    print("\n======== OBV 量能背离（额外买入触发，平行 %B 背离，增加交易）========")
    delta_block("C0 基线", "CO +OBV背离", "① 简单成本口径")
    delta_block("C0g regime门控", "COg +OBV+regime", "② 大盘趋势门控")
    delta_block("C0r 真实成本", "COr +OBV+成本", "③ 真实分科目成本")
    print(f"\n[基准 {BENCH}] {idx_ret:+.2f}%")

    # ── 落盘 ──
    cmp_rows = []
    for name, _ in CONFIGS:
        s = results[name][1]
        cmp_rows.append({"配置": name, **s})
    pd.DataFrame(cmp_rows).to_csv(OUT_DIR / "ablation_compare.csv", index=False, encoding="utf-8-sig")
    for name, (rows, _) in results.items():
        key = name.split()[0]
        pd.DataFrame(rows).to_csv(
            OUT_DIR / f"ablation_{key}.csv", index=False, encoding="utf-8-sig")
    print(f"\n[已保存] {OUT_DIR}/  (ablation_compare.csv + ablation_C0/CM/CO/C0g/CMg/COg/C0r/CMr/COr.csv)")


if __name__ == "__main__":
    main()
