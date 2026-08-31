# -*- coding: utf-8 -*-
"""
均值回归策略 —— 基线单笔期望分析 + 持仓窗口随机对照
====================================================

背景：暴露度对照证明 OBV/%B 的增量全是 beta，但同时暴露了一件更有价值的事——
      随机入场控制组用的是**同一套出场逻辑**（Z>2 或 RSI>70 + 止损），
      却拿到与 OBV 同等的收益 → 说明**正期望来自出场/风控框架，而非入场信号**。

本脚本回答两个问题：

A. 基线单笔期望（描述）—— 一笔交易平均赚多少？
   - FIFO 配对买卖，得到完整 round trip（入场/出场日期、价格、持仓交易日数）
   - 净口径用 buy.cost(含费) 与 sell.revenue(扣费后)，真实反映到手收益
   - 输出：笔数、胜率、平均盈、平均亏、单笔期望、盈亏比、盈亏因子、平均持仓天数

B. 持仓窗口随机对照（决定性）—— 出场/入场时点有没有 skill？
   - 对每一笔真实 round trip（持仓 h 个交易日），在同一只股票的回测区间内
     随机取同样长度 h 的窗口，计算该窗口收益；重复采样取均值
   - 若"策略选中的窗口"平均收益 ≈ "随机窗口"平均收益 → 时点无 skill，
     收益只是"在长牛里持有股票"的自然结果
   - 用 GROSS 口径（均不含费）比较，保证两边同基准

固定：股票池(沪深300 as-of 20190101 快照前40)、区间、本金、参数全同 config 默认
      （即四个候选开关全关 = 平台当前默认行为）。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from run_backtest import load_stock_prices, _get_index_constituents_from_db  # noqa: E402
from backtest.mean_reversion_plugin import MeanReversionStrategyPlugin  # noqa: E402
import config  # noqa: E402

START = "20190101"
END = "20260731"
CAPITAL = 100_000
TOP_N = 40
BENCH = "000300.SH"
N_SAMPLES = 30      # 每笔真实交易对应的随机窗口采样数
OUT_DIR = HERE / "data" / "results" / "mean_reversion_trade_expectancy"


def build_universe(n: int = TOP_N) -> list[str]:
    df = _get_index_constituents_from_db(BENCH, as_of_date=START)
    if df is None or df.empty:
        return []
    codes = sorted(df["code"].tolist())[:n]
    print(f"[股票池] 沪深300 as-of {START} 快照 → 取 {len(codes)} 只")
    return codes


def fifo_round_trips(trades: list[dict]) -> list[dict]:
    """FIFO 配对，得到完整 round trip（含日期与费后口径）。"""
    pending = []   # [{date, shares, cost}]
    rts = []
    for t in trades:
        act = t.get("action", "")
        sh = t.get("shares", 0)
        if act.startswith("BUY"):
            pending.append({"date": t["date"], "shares": sh,
                            "cost": t.get("cost", 0.0), "price": t.get("price", 0.0)})
        elif act.startswith("SELL"):
            rem = sh
            rev_total = t.get("revenue", 0.0)
            unit_rev = rev_total / sh if sh else 0.0
            while rem > 0 and pending:
                f = pending[0]
                m = min(f["shares"], rem)
                unit_cost = f["cost"] / f["shares"] if f["shares"] else 0.0
                gross = (t["price"] - f["price"]) / f["price"] * 100 if f["price"] else 0.0
                net = (unit_rev - unit_cost) / unit_cost * 100 if unit_cost else 0.0
                rts.append({
                    "entry_date": f["date"], "exit_date": t["date"], "shares": m,
                    "entry_price": f["price"], "exit_price": t["price"],
                    "gross_pct": gross, "net_pct": net,
                })
                f["shares"] -= m
                # 已配对的 cost 等比例扣减，保证剩余部分成本正确
                if f["shares"] > 0:
                    f["cost"] -= unit_cost * m
                rem -= m
                if f["shares"] <= 0:
                    pending.pop(0)
    return rts


def analyze_stock(code: str, cfg: dict, rng: np.random.Generator):
    import sqlite3
    conn = sqlite3.connect(config.DATA["local_db_path"])
    df = load_stock_prices(code, START, END, conn, lookback_days=250)
    conn.close()
    if df is None or len(df) < 30:
        return None
    df = df.reset_index(drop=True)
    start_idx = int(df[df["trade_date"] >= START].index.min())
    if pd.isna(start_idx):
        return None

    strat = MeanReversionStrategyPlugin(CAPITAL, cfg)
    res = strat.run(df, start_idx)
    rts = fifo_round_trips(res.get("trades", []))

    closes = df["adj_close"].values
    dates = df["trade_date"].tolist()
    pos_of = {d: i for i, d in enumerate(dates)}
    n = len(closes)

    for r in rts:
        i1 = pos_of.get(r["entry_date"])
        i2 = pos_of.get(r["exit_date"])
        r["hold"] = (i2 - i1) if (i1 is not None and i2 is not None) else 0

        # 随机窗口对照：同长度 h，起点在回测区间内随机
        h = max(int(r["hold"]), 1)
        lo, hi = start_idx, n - 1 - h
        if hi <= lo:
            r["rand_pct"] = np.nan
            continue
        starts = rng.integers(lo, hi + 1, size=N_SAMPLES)
        rets = []
        for s in starts:
            s = int(s)
            if s + h >= n:
                continue
            p0, p1 = closes[s], closes[s + h]
            if p0 and not np.isnan(p0) and not np.isnan(p1):
                rets.append((p1 - p0) / p0 * 100)
        r["rand_pct"] = float(np.mean(rets)) if rets else np.nan

    return {"code": code, "rts": rts, "ret": res.get("returns", 0.0),
            "n_trades": len(res.get("trades", []))}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    codes = build_universe(TOP_N)
    cfg = dict(config.STRATEGIES["mean_reversion"])   # 平台默认（四开关全关）
    print(f"[配置] 使用平台默认 mean_reversion 配置（候选开关全关）")
    print(f"[采样] 每笔真实交易配 {N_SAMPLES} 个等长随机窗口\n")

    rng = np.random.default_rng(20260831)
    all_rts = []
    stock_rows = []
    for i, code in enumerate(codes, 1):
        try:
            out = analyze_stock(code, cfg, rng)
        except Exception as e:  # noqa: BLE001
            print(f"  [ERR] {code}: {e}")
            continue
        if out is None or not out["rts"]:
            continue
        all_rts.extend(out["rts"])
        g = [r["gross_pct"] for r in out["rts"]]
        nn = [r["net_pct"] for r in out["rts"]]
        rr = [r["rand_pct"] for r in out["rts"] if not np.isnan(r["rand_pct"])]
        stock_rows.append({
            "code": code, "strat_ret": out["ret"], "round_trips": len(out["rts"]),
            "gross_mean": float(np.mean(g)), "net_mean": float(np.mean(nn)),
            "rand_mean": float(np.mean(rr)) if rr else np.nan,
            "hold_mean": float(np.mean([r["hold"] for r in out["rts"]])),
        })
        if i % 10 == 0 or i == 1:
            print(f"  ...[{i:>2}/{len(codes)}] {code} 平仓对 {len(out['rts'])}  "
                  f"累计 {len(all_rts)} 笔")

    df_rt = pd.DataFrame(all_rts)
    df_rt.to_csv(OUT_DIR / "round_trips.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(stock_rows).to_csv(OUT_DIR / "per_stock.csv", index=False, encoding="utf-8-sig")

    # ── A. 单笔期望 ──
    net = df_rt["net_pct"].values
    gross = df_rt["gross_pct"].values
    wins = net[net > 0]
    losses = net[net <= 0]
    avg_win = float(np.mean(wins)) if len(wins) else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) else 0.0
    expectancy = float(np.mean(net))
    pf = (abs(float(np.sum(wins))) / abs(float(np.sum(losses)))) if len(losses) and np.sum(losses) != 0 else float("inf")

    print("\n" + "=" * 96)
    print("A. 基线单笔期望（FIFO 配对，净口径含手续费/印花税）")
    print("=" * 96)
    print(f"  平仓交易对数        : {len(net)}")
    print(f"  胜率                : {100 * len(wins) / len(net) if len(net) else 0:.2f}%  "
          f"（{len(wins)} 胜 / {len(losses)} 负）")
    print(f"  平均盈利            : {avg_win:+.2f}%")
    print(f"  平均亏损            : {avg_loss:+.2f}%")
    print(f"  盈亏比(avgW/|avgL|) : {avg_win / abs(avg_loss) if avg_loss else float('inf'):.2f}")
    print(f"  盈亏因子(ΣW/|ΣL|)   : {pf:.2f}")
    print(f"  ★ 单笔期望(净)      : {expectancy:+.3f}%")
    print(f"  单笔期望(毛)        : {float(np.mean(gross)):+.3f}%")
    print(f"  费率拖累            : {float(np.mean(gross)) - expectancy:.3f}pp/笔")
    print(f"  中位(净)            : {float(np.median(net)):+.3f}%")
    print(f"  平均持仓交易日      : {float(np.mean(df_rt['hold'].values)):.1f} 日  "
          f"（中位 {float(np.median(df_rt['hold'].values)):.0f} 日）")

    # ── B. 持仓窗口随机对照 ──
    sub = df_rt.dropna(subset=["rand_pct"])
    a = float(np.mean(sub["gross_pct"]))
    b = float(np.mean(sub["rand_pct"]))
    print("\n" + "=" * 96)
    print(f"B. 持仓窗口随机对照（等长窗口随机起点，{N_SAMPLES} 次采样；GROSS 口径，两边均不含费）")
    print("=" * 96)
    print(f"  参与对照的交易对数  : {len(sub)}")
    print(f"  策略实际窗口平均收益: {a:+.3f}%")
    print(f"  随机窗口平均收益    : {b:+.3f}%")
    print(f"  差（策略 − 随机）   : {a - b:+.3f}pp")
    per = (sub["gross_pct"] > sub["rand_pct"]).mean() * 100
    print(f"  单笔优于随机的比例  : {per:.1f}%  （纯随机预期 ≈ 50%）")

    print("\n[判据]")
    if a - b >= 0.5:
        print("  ✅ 出场/入场时点有正 skill：选中窗口显著优于等长随机窗口")
    elif a - b >= 0.15:
        print("  ⚠️ 时点有边际 skill，但幅度小，需谨慎")
    else:
        print("  ❌ 时点无 skill：收益与『在长牛里随机持有一段』无异，正期望来自 beta 而非框架")
    print(f"\n[已保存] {OUT_DIR}/  (round_trips.csv + per_stock.csv)")


if __name__ == "__main__":
    main()
