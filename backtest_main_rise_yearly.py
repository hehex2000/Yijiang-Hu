"""主升浪战法·分年稳健性：从已存 trades CSV 聚合每年净收益/胜率。

直接复用 backtest_main_rise.trade_net_ret，按 entry_date 的年份分组。
目的：回应视频自证疑虑——"换观察窗口/门槛若结果大变=只适配某段历史"。
这里看的是：负期望是否整段持续（每年都负），还是被某几年熊市撑着。
"""
import os
import argparse
import numpy as np
import pandas as pd
import backtest_main_rise as M

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", default="data/results/main_rise/main_rise_trades.csv")
    ap.add_argument("--out", default="data/results/main_rise")
    ap.add_argument("--notional", type=float, default=100000.0)
    args = ap.parse_args()

    df = pd.read_csv(args.trades, dtype={"ts_code": str, "entry_date": int, "exit_date": int})
    nets, wins, holds = [], [], []
    for _, r in df.iterrows():
        nr = M.trade_net_ret(args.notional, float(r.entry_open), float(r.exit_open),
                             int(r.entry_date), int(r.exit_date))
        if nr is None:
            continue
        nets.append(nr)
        wins.append(1 if nr > 0 else 0)
        holds.append(int(r.hold_days))
    df["net"] = nets
    df["win"] = wins
    df["year"] = df["entry_date"] // 10000

    rows = []
    for y, g in df.groupby("year"):
        rows.append({
            "year": int(y),
            "n": len(g),
            "mean_net_%": round(g["net"].mean(), 4),
            "win_%": round(g["win"].mean() * 100, 2),
            "avg_hold": round(g["hold_days"].mean(), 1),
            "exit_ma20_%": round((g["reason"] == "ma20").mean() * 100, 1),
            "exit_stop_%": round((g["reason"] == "stop").mean() * 100, 1),
        })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(args.out, "main_rise_by_year.csv"), index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))
    print(f"\n[summary] 总年数={len(out)} 每年净收益均负的年数={(out['mean_net_%']<0).sum()} "
          f"年均净收益={out['mean_net_%'].mean():.4f}% 平均胜率={out['win_%'].mean():.1f}%")

if __name__ == "__main__":
    main()
