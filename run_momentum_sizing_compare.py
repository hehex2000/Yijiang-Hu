"""
run_momentum_sizing_compare.py — 动量策略 × 四仓位方案 对比 (resilient)
============================================================

在「真有信号含义」的动量选股上，验证 position_sizing 四种仓位方案
(equal/pyramid/inverted/martingale) 的作用边界。

设计纪律（沿用 pyramid 系列教训）：
  - 用 zz800(000906.SH) 历史成分股快照 → 杜绝幸存者偏差/未来函数
  - 后复权收益 → 动量信号不被分红/送转污染
  - 月度调仓、T-1选股 T开盘执行
  - martingale 自带单票权重上限防爆仓

resilient: 每方案跑完即 upsert 到 compare_partial.csv，支持 --only 断点续跑
（后台任务被杀也不丢已完成方案）。

用法:
  python run_momentum_sizing_compare.py [--start 20140101] [--end 20260630]
                                        [--top-n 15] [--lookback 12]
                                        [--only martingale]   # 只跑指定方案
"""
import os
import sys
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_monthly_rebalance import run_momentum_backtest  # noqa: E402

SCHEMES = ["equal", "pyramid", "inverted", "martingale"]


def load_partial(path):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20140101")
    ap.add_argument("--end", default="20260630")
    ap.add_argument("--top-n", type=int, default=15)
    ap.add_argument("--lookback", type=int, default=12)
    ap.add_argument("--pool", default="000906.SH")
    ap.add_argument("--freq", type=int, default=1)
    ap.add_argument("--only", default=None,
                    help="只跑指定方案(逗号分隔), 用于断点续跑")
    args = ap.parse_args()

    out_dir = "data/results/momentum_sizing"
    os.makedirs(out_dir, exist_ok=True)
    partial_csv = os.path.join(out_dir, "compare_partial.csv")

    schemes = [s.strip() for s in args.only.split(",")] if args.only else SCHEMES
    done = set()
    if os.path.exists(partial_csv):
        try:
            done = set(load_partial(partial_csv)["sizing"].tolist())
        except Exception:
            done = set()

    for sch in schemes:
        if sch in done:
            print(f"[skip] {sch} 已完成 (断点续跑)")
            continue
        print(f"\n\n########## 仓位方案 SIZING = {sch} ##########")
        r = run_momentum_backtest(
            start_date=args.start, end_date=args.end,
            top_n=args.top_n, lookback_months=args.lookback,
            stock_pool=args.pool, rebalance_freq_months=args.freq,
            skip_recent_months=1, sizing=sch,
        )
        row = {
            "sizing": sch,
            "total_return_pct": round(r["total_return"], 2),
            "annual_return_pct": round(r["annual_return"], 2),
            "max_dd_pct": round(r["max_drawdown"], 2),
            "sharpe": round(r["sharpe"], 2),
            "win_rate_pct": round(r.get("win_rate", 0.0), 1),
            "trades": r["trades"],
            "idx_return_pct": round(r["idx_return"], 2),
            "excess_pct": round(r["total_return"] - r["idx_return"], 2),
        }
        df = load_partial(partial_csv)
        if "sizing" in df.columns:
            df = df[df["sizing"] != sch]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(partial_csv, index=False)
        print(f"[saved] {sch} -> {partial_csv}")

    final = load_partial(partial_csv)
    print(f"\n{'=' * 72}")
    print(f"  动量({args.lookback}m) × 四仓位方案 对比  [{args.start}~{args.end}]")
    print(f"{'=' * 72}")
    print(final.to_string(index=False))
    print(f"\n  保存: {partial_csv}")


if __name__ == "__main__":
    main()
