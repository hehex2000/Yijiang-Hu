# -*- coding: utf-8 -*-
"""
均值回归策略 —— Head-fake 假突破过滤 A/B 归因
============================================

目的：验证 B站进阶视频(BV1FP876bEAo)提出的 Head-fake 过滤能否给现有均值回归策略带来
      真增量。控制单一变量，只开/关 headfake_filter：

  C0  基线(无 headfake) : headfake_filter=False  ← 当前默认
  C1  +headfake 过滤     : headfake_filter=True   ← 候选改进(opt-in)

Head-fake 逻辑（已在 mean_reversion_plugin 实现，opt-in）：
  买入信号触发(价格击穿下轨+RSI<30+未张开)后不立即成交，改等随后 1-2 根 K 线
  重新站回下轨(顺轨站稳)才确认买入；连续贴轨外下行=真破位陷阱，取消信号。
  用 i-1 收盘/下轨判定、i 开盘成交，无未来函数。

固定：股票池(沪深300 as-of START 快照前40、非幸存者偏差口径)、期间、每支本金、行情/手续费口径、其余参数全同 config。
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
    DB_PATH,
)
from backtest.mean_reversion_plugin import MeanReversionStrategyPlugin  # noqa: E402
import config  # noqa: E402

START = "20190101"
END = "20260731"
CAPITAL = 100_000
TOP_N = 40
BENCH = "000300.SH"
OUT_DIR = HERE / "data" / "results" / "mean_reversion_headfake_ablation"


def build_universe(n: int = TOP_N) -> list[str]:
    """沪深300 成分股（as_of START 快照，消除幸存者偏差）。

    注意：必须用回测起始日的快照，而不是数据库最新快照——否则是用 2026 年
    仍在沪深300 的股票去跑 2019 年的回测，系统性剔除被调出/退市的弱势股，
    高估收益。复用 run_backtest._get_index_constituents_from_db（含边界回退）。
    沪深300 不含北交所股票，无需 .BJ 过滤。
    """
    df = _get_index_constituents_from_db(BENCH, as_of_date=START)
    if df is None or df.empty:
        print(f"[股票池] ⚠️ 未取到 {BENCH} 成分股")
        return []
    codes = sorted(df["code"].tolist())[:n]  # 按代码排序取前 n，确定性
    print(f"[股票池] 沪深300 as-of {START} 快照 → 取 {len(codes)} 只（非幸存者偏差口径）")
    return codes


def load_benchmark_return() -> float:
    import sqlite3
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
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    df = load_stock_prices(code, START, END, conn, lookback_days=250)
    conn.close()
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
        "pending_cancelled": getattr(strat, "_cancelled_count", 0),
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
        "cancelled_mean": float(np.mean([r.get("pending_cancelled", 0) for r in rows])),
    }


BASE = dict(config.STRATEGIES["mean_reversion"])
CONFIGS = {
    "C0 基线(无headfake)": dict(BASE, headfake_filter=False),
    "C1 +headfake过滤":    dict(BASE, headfake_filter=True),
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    codes = build_universe(TOP_N)
    idx_ret = load_benchmark_return()
    print(f"[基准] {BENCH} {START}~{END} 收益 = {idx_ret:+.2f}%\n")

    results = {}
    for name, cfg in CONFIGS.items():
        print(f"\n### 运行 {name}  (headfake_filter={cfg['headfake_filter']})")
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

    # ── 归因表 ──
    print("\n" + "=" * 92)
    print(f"均值回归 Head-fake 过滤 A/B（{START}~{END}，沪深300前{TOP_N}，每支{CAPITAL//10000}万）")
    print("=" * 92)
    hdr = f"{'配置':<18}{'均值%':>9}{'中位%':>9}{'正收益':>8}{'跑赢指':>8}{'胜率%':>8}{'均交易':>8}{'均回撤%':>9}{'取消信号':>9}"
    print(hdr)
    print("-" * 92)
    for name in CONFIGS:
        s = results[name][1]
        print(f"{name:<18}{s['mean']:>+9.2f}{s['median']:>+9.2f}"
              f"{s['n_pos']:>{6}}/{s['n']:<3}{s['n_beat']:>{6}}/{s['n']:<3}"
              f"{s['win_rate_mean']:>8.2f}{s['trades_mean']:>8.1f}{s['max_dd_mean']:>9.2f}"
              f"{s['cancelled_mean']:>9.1f}")
    print(f"{'沪深300基准':<18}{idx_ret:>+9.2f}")
    print("-" * 92)
    print("逐项贡献（C1 − C0，隔离单一变量 headfake_filter）：")
    c0 = results["C0 基线(无headfake)"][1]
    c1 = results["C1 +headfake过滤"][1]
    print(f"  Δ均值 {c1['mean'] - c0['mean']:+.2f}pp   "
          f"Δ中位 {c1['median'] - c0['median']:+.2f}pp")
    print(f"  Δ胜率 {c1['win_rate_mean'] - c0['win_rate_mean']:+.2f}pp   "
          f"Δ交易 {c1['trades_mean'] - c0['trades_mean']:+.1f}   "
          f"Δ回撤 {c1['max_dd_mean'] - c0['max_dd_mean']:+.2f}pp")
    print(f"\n[基准 {BENCH}] {idx_ret:+.2f}%")

    # ── 落盘 ──
    cmp_rows = []
    for name in CONFIGS:
        s = results[name][1]
        cmp_rows.append({"配置": name, **s})
    pd.DataFrame(cmp_rows).to_csv(OUT_DIR / "ablation_compare.csv", index=False, encoding="utf-8-sig")
    for name, (rows, _) in results.items():
        pd.DataFrame(rows).to_csv(
            OUT_DIR / f"ablation_{name.split()[0]}.csv", index=False, encoding="utf-8-sig")
    print(f"\n[已保存] {OUT_DIR}/  (ablation_compare.csv + ablation_C0.csv + ablation_C1.csv)")


if __name__ == "__main__":
    main()
