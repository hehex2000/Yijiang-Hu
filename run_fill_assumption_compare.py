# -*- coding: utf-8 -*-
"""工具①：限价 vs 市价 成交假设对照器。

同一份策略 trades 流水，分别在两种成交成本假设下重建组合净值，隔离
「成交成本假设」这一单变量：
  - 市价/taker：平台默认（flat 0.1%，或 MFS_SQRT_IMPACT=1 走平方根冲击）
  - 限价/maker：被动挂单，不吃 taker 冲击（maker_slip 默认 0，可传负表示吃价差）

核心用途：左侧/逆向/价值类策略本应挂限价单，平台回测却强制市价成交，
系统性高估其成本——本工具量化「若改用限价成交能挽回多少收益」。

用法：
  # 单文件（默认市价 flat vs 限价 maker=0）
  python run_fill_assumption_compare.py --trades data/results/monthly_rebalance/trades_value_20200102_20251231.csv

  # 市价也用平方根冲击模型（更真实）
  MFS_SQRT_IMPACT=1 python run_fill_assumption_compare.py --trades <csv>

  # 批量扫描一组策略
  python run_fill_assumption_compare.py --scan data/results --glob "trades_*.csv" --max 8

  # 限价模型吃到 half-spread 反向收益（保守上界）
  python run_fill_assumption_compare.py --trades <csv> --maker-slip -0.0005
"""
import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd

# sqrt 市价模型需在导入 nav_recon_util（其会 import run_monthly_rebalance 并
# 在模块加载时读取 USE_SQRT_IMPACT）之前设定环境变量。
_p = argparse.ArgumentParser(add_help=False)
_p.add_argument("--sqrt", action="store_true", help="市价模型用平方根冲击（更真实）")
_args0, _ = _p.parse_known_args()
if _args0.sqrt:
    os.environ["MFS_SQRT_IMPACT"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nav_recon_util as U


def _metrics_row(label, res):
    return dict(
        label=label,
        n_trades=res["n_trades"],
        total_return=res["total_return"],
        annualized=res["annualized"],
        max_dd=res["max_dd"],
        total_cost=res["total_cost"],
        active_tax_yr=res["active_tax_yr"],
    )


def compare_one(path, maker_slip=0.0, sqrt_market=False):
    trades = U.load_trades(path)
    init_cap = U.compute_init_cap(trades)
    # 市价模型
    mkt = U.reconstruct(trades, U.slippage_frac_market, init_cap=init_cap)
    # 限价模型
    lim = U.reconstruct(trades, U.slippage_frac_limit, init_cap=init_cap,
                        maker_slip=maker_slip)
    if mkt is None or lim is None:
        return None
    rm = _metrics_row("market", mkt)
    rl = _metrics_row("limit", lim)
    # 挽回：限价相对市价的收益差（百分点）
    salvage_pp = (rl["total_return"] - rm["total_return"]) * 100
    cost_save = rm["total_cost"] - lim["total_cost"]
    return dict(path=path, mkt=rm, lim=rl, salvage_pp=salvage_pp,
                cost_save=cost_save, init_cap=init_cap,
                n_years=mkt["years"])


def fmt_pct(x):
    return f"{x*100:+.2f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", help="单个 trades CSV")
    ap.add_argument("--scan", help="扫描目录")
    ap.add_argument("--glob", default="trades_*.csv")
    ap.add_argument("--max", type=int, default=12)
    ap.add_argument("--maker-slip", type=float, default=0.0,
                    help="限价模型滑点占比（默认0=仅免taker成本；负数=吃half-spread）")
    ap.add_argument("--sqrt", action="store_true", help="市价模型用平方根冲击（更真实）")
    args = ap.parse_args()

    rows = []
    if args.trades:
        r = compare_one(args.trades, maker_slip=args.maker_slip)
        if r:
            rows.append(r)
    elif args.scan:
        files = sorted(glob.glob(os.path.join(args.scan, "**", args.glob),
                                 recursive=True))
        for f in files[: args.max]:
            try:
                r = compare_one(f, maker_slip=args.maker_slip)
            except Exception as e:
                print(f"  ! 跳过 {f}: {e}")
                continue
            if r:
                rows.append(r)
    else:
        print("需提供 --trades <csv> 或 --scan <dir>")
        return

    if not rows:
        print("无可用结果")
        return

    print(f"\n=== 限价 vs 市价 成交假设对照（maker_slip={args.maker_slip:+.4f}）===")
    print(f"{'策略文件':<62}{'交易数':>7}{'市价年化':>10}{'限价年化':>10}"
          f"{'挽回pp':>9}{'市价活跃税':>11}{'限价活跃税':>11}")
    for r in rows:
        base = os.path.basename(r["path"])
        print(f"{base:<62}{r['mkt']['n_trades']:>7}{fmt_pct(r['mkt']['annualized']):>10}"
              f"{fmt_pct(r['lim']['annualized']):>10}{r['salvage_pp']:>+9.2f}"
              f"{r['mkt']['active_tax_yr']*100:>10.2f}%{r['lim']['active_tax_yr']*100:>10.2f}%")

    # 详细：首个（单文件模式）打印全指标 + 逐笔成本 CSV
    if args.trades and rows:
        r = rows[0]
        print(f"\n--- 明细：{os.path.basename(r['path'])} ---")
        print(f"样本年数={r['n_years']:.2f}  初始资金={r['init_cap']:,.0f}")
        print(f"市价：总收益{fmt_pct(r['mkt']['total_return'])} 年化{fmt_pct(r['mkt']['annualized'])} "
              f"最大回撤{fmt_pct(r['mkt']['max_dd'])} 总成本{r['mkt']['total_cost']:,.0f} "
              f"活跃税{r['mkt']['active_tax_yr']*100:.2f}%/年")
        print(f"限价：总收益{fmt_pct(r['lim']['total_return'])} 年化{fmt_pct(r['lim']['annualized'])} "
              f"最大回撤{fmt_pct(r['lim']['max_dd'])} 总成本{r['lim']['total_cost']:,.0f} "
              f"活跃税{r['lim']['active_tax_yr']*100:.2f}%/年")
        print(f"→ 限价假设相对市价挽回收益 {r['salvage_pp']:+.2f}pp，节省交易成本 {r['cost_save']:,.0f} 元")

        # 逐笔成本明细（前若干笔 + 全量落盘）
        trades = U.load_trades(r["path"])
        recs = []
        for _, t in trades.iterrows():
            a = str(t["action"]).strip().upper()
            amt = t["price"] * t["shares"]
            td = int(t["date"])
            code = t["code"]
            mkt_slip = amt * U.slippage_frac_market(a, amt, code, td)
            lim_slip = amt * U.slippage_frac_limit(a, amt, code, td,
                                                   maker_slip=args.maker_slip)
            mkt_cost = U.cost_of(a, t["price"], t["shares"], td, code,
                                 U.slippage_frac_market)
            lim_cost = U.cost_of(a, t["price"], t["shares"], td, code,
                                U.slippage_frac_limit, maker_slip=args.maker_slip)
            recs.append(dict(date=td, action=a, code=code, amount=amt,
                             mkt_slip=mkt_slip, lim_slip=lim_slip,
                             mkt_cost=mkt_cost, lim_cost=lim_cost,
                             saving=mkt_cost - lim_cost))
        out = pd.DataFrame(recs)
        od = os.path.dirname(os.path.abspath(r["path"]))
        ob = os.path.splitext(os.path.basename(r["path"]))[0]
        out_path = os.path.join(od, f"{ob}_fills_assumption.csv")
        out.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"逐笔成本明细已写出：{out_path}（{len(out)} 笔）")


if __name__ == "__main__":
    main()
