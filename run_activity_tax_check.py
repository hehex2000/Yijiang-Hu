# -*- coding: utf-8 -*-
"""工具②：活跃税 / 处置效应 体检。

对标 Barber & Odean (2000)《Trading Is Hazardous to Your Wealth》核心发现：
最活跃交易者年化收益 11.4% vs 市场 17.9%，"活跃税"约 6.5%/年——频繁交易本身
就在吞噬收益。

本工具扫描一组策略的 trades CSV，用平台真实成本函数（市价/taker 假设）重建 NAV，
计算：
  - 年化换手成本率（活跃税）= 总交易成本 / (平均组合市值 × 年数)
  - 总交易笔数、样本年数、平均组合市值
并对照 6.5%/年 基准，排序标红超阈值策略，提示"该策略的alpha可能被交易摩擦吃光"。

纯分析，不改任何回测引擎。

用法：
  # 默认扫描一组代表性策略（价值/PEG/神奇公式/多因子/EP中性/高股息/日20红利低波）
  python run_activity_tax_check.py

  # 自定义目录 + glob
  python run_activity_tax_check.py --scan data/results --glob "trades_*.csv" --max 40

  # 市场模型用平方根冲击（更真实，小票影响更大）
  MFS_SQRT_IMPACT=1 python run_activity_tax_check.py
"""
import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd

_p = argparse.ArgumentParser(add_help=False)
_p.add_argument("--sqrt", action="store_true")
_args0, _ = _p.parse_known_args()
if _args0.sqrt:
    os.environ["MFS_SQRT_IMPACT"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nav_recon_util as U

# 代表性策略样本（与用户现有 trades 产物对应；非穷举，按研究价值挑选）
DEFAULT_SAMPLE = [
    ("价值选股",        "data/results/monthly_rebalance/trades_blend50_value_20100101_20251231.csv"),
    ("红利低波质量",     "data/results/monthly_rebalance/trades_div_low_vol_20200103_20260815.csv"),
    ("PEG(年度)",       "data/results/peg/trades_n30_c1000000_annual_s3_20140101_20260715.csv"),
    ("神奇公式",        "data/results/magic_formula/trades_n30_c1000000_20230101_20260715.csv"),
    ("多因子Q(长样本)",  "data/results/multifactor/trades_n30_open_Q_m_c1000000_20100101_20260715.csv"),
    ("EP中性",          "data/results/ep_neutral/trades_nG5_open_c5000000_20200101_20260715.csv"),
    ("周度高股息量价",   "data/results/weekly_highdiv_vol/trades_n10_d25_t85_db55_s50_cost_20200103_20260715.csv"),
    ("日20红利低波(月调)", "data/results/daily20_divlow/trades_monthly_20150101_20251231.csv"),
]

BENCH_TAX = 0.065  # Barber&Odean 6.5%/年


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", help="扫描目录（覆盖 DEFAULT_SAMPLE）")
    ap.add_argument("--glob", default="trades_*.csv")
    ap.add_argument("--max", type=int, default=40)
    ap.add_argument("--sqrt", action="store_true", help="市价模型用平方根冲击")
    ap.add_argument("--out", default="data/results/activity_tax_check.csv")
    args = ap.parse_args()

    if args.scan:
        files = sorted(glob.glob(os.path.join(args.scan, "**", args.glob),
                                 recursive=True))
        items = [(os.path.basename(f), f) for f in files[: args.max]]
    else:
        items = DEFAULT_SAMPLE

    rows = []
    for name, path in items:
        if not os.path.exists(path):
            print(f"  ! 缺失 {path}")
            continue
        try:
            trades = U.load_trades(path)
        except Exception as e:
            print(f"  ! {name} 读取失败: {e}")
            continue
        init_cap = U.compute_init_cap(trades)
        res = U.reconstruct(trades, U.slippage_frac_market, init_cap=init_cap)
        if res is None:
            continue
        rows.append(dict(
            name=name, file=os.path.basename(path),
            n_trades=res["n_trades"], years=round(res["years"], 2),
            init_cap=res["init_cap"], total_cost=res["total_cost"],
            total_traded=res["total_traded"],
            active_tax_yr=res["active_tax_yr"],
            round_trip_cost=res["round_trip_cost"],
            total_return=res["total_return"], annualized=res["annualized"],
            max_dd=res["max_dd"],
        ))

    if not rows:
        print("无可用结果")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values("active_tax_yr", ascending=False).reset_index(drop=True)
    df["flag"] = np.where(df["active_tax_yr"] > BENCH_TAX, "⚠️超阈值", "")

    print(f"\n=== 活跃税 / 处置效应 体检（市价/taker 成本，B&O 基准 {BENCH_TAX*100:.1f}%/年）===")
    print(f"{'策略':<14}{'笔数':>6}{'年数':>7}{'建仓资金(万)':>13}"
          f"{'年交易成本(万)':>14}{'活跃税/年':>11}{'单边摩擦':>10}{'标记':>10}")
    for _, r in df.iterrows():
        init_w = r["init_cap"] / 1e4
        cost_yr_w = r["total_cost"] / (r["years"] if r["years"] else 1) / 1e4
        print(f"{r['name']:<14}{int(r['n_trades']):>6}{r['years']:>7.1f}{init_w:>13.0f}"
              f"{cost_yr_w:>14.1f}{r['active_tax_yr']*100:>10.2f}%"
              f"{r['round_trip_cost']*100:>9.2f}%{r['flag']:>10}")

    n_over = int((df["active_tax_yr"] > BENCH_TAX).sum())
    print(f"\n→ {n_over}/{len(df)} 个策略活跃税超过 B&O 6.5%/年基准"
          f"（频繁交易本身在吞噬alpha）。")
    print(f"  注：活跃税=年交易成本/建仓资金（股本代理，口径无关）；单边摩擦=总成本/总成交额（跨策略可比）。")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"明细已写出：{args.out}")


if __name__ == "__main__":
    main()
