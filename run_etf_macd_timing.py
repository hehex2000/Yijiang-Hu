# -*- coding: utf-8 -*-
"""
可落地 ETF 版 MACD 择时（用户拍板路径②）
==================================================
标的 : 510300(沪深300ETF, 追踪000300.SH) / 515800(中证800ETF, 追踪000906.SH)
信号 : 沪深300 日线 MACD(12/26/9)，金叉满仓、死叉清仓(空仓冻结)
回测 : 用指数净值近似 ETF（真实ETF多跟踪误差+管理费，用 --etf-cost 统一折损）

输出：
  1) 对照表：满仓持有 / MACD择时(纯空仓)
  2) 交易清单 CSV（每次金叉买入/死叉卖出，可复核）
  3) 每日净值 CSV（date, nav_buyhold, nav_macd_cash，
     可喂 macd_plugin_validate --base-nav 继续对照）

用法：
  ./venv_ml/Scripts/python.exe run_etf_macd_timing.py                      # 默认510300
  ./venv_ml/Scripts/python.exe run_etf_macd_timing.py --etf 515800 --etf-cost 0.002
"""
import argparse
import numpy as np
import pandas as pd

import macd_plugin_validate as M
from regime_cash_overlay import load_index_close, apply_overlay, cash_ratio

ETF_MAP = {
    '510300': ('000300.SH', '沪深300ETF(510300.SH, 2012-05-28上市)'),
    '515800': ('000906.SH', '中证800ETF(515800.SH, 2019-01上市)'),
}
SIG = '000300.SH'   # 信号指数固定：沪深300


def main():
    ap = argparse.ArgumentParser(description='ETF版MACD择时回测')
    ap.add_argument('--etf', default='510300', choices=sorted(ETF_MAP))
    ap.add_argument('--start', default=None, help='默认取上市后次月')
    ap.add_argument('--end', default='20251231')
    ap.add_argument('--etf-cost', type=float, default=0.002, help='ETF管理费+跟踪误差年化, 默认0.2%')
    ap.add_argument('--out', default=None, help='净值CSV输出路径(默认 etf_macd_timing_<etf>.csv)')
    args = ap.parse_args()

    base_code, label = ETF_MAP[args.etf]
    if args.start is None:
        args.start = '20130701' if args.etf == '510300' else '20190701'

    # ── 标的净值（指数代理 + ETF 成本拖累）──
    base = M.load_base_index(base_code, args.start, args.end)
    if base is None or len(base) < 30:
        print(f"[ERR] {base_code} 数据不足({args.start}~{args.end})"); return
    dc = 1.0 - (1.0 - args.etf_cost) ** (1.0 / 252.0)          # 每日成本
    r = base.pct_change().fillna(0.0)
    nav_etf = (1.0 + r - dc).cumprod()
    nav_etf = nav_etf / nav_etf.iloc[0]                        # 归一
    nav_etf.index = base.index

    # ── 信号 ──
    hs = load_index_close(SIG, args.start, args.end).reindex(base.index).ffill()
    if len(hs) < 30:
        print("[ERR] 沪深300 信号数据不足"); return
    golden = M.macd_golden(hs.values).values

    # ── 两方案 ──
    nav_bh = nav_etf
    nav_cash = apply_overlay(nav_etf.values, golden, cash_growth=1.0)
    rb, ab, mdb, sb = M.metrics(nav_bh)
    rc, ac, mdc, sc = M.metrics(nav_cash)
    cr = cash_ratio(golden) * 100

    def pct(x): return f"{x*100:+.2f}%"
    print(f"\n{'='*92}")
    print(f"  ETF 版 MACD 择时 | {label} | 追踪{base_code} | 信号={SIG} 沪深300 MACD(12/26/9)")
    print(f"  区间 {args.start}~{args.end} | ETF年成本{args.etf_cost*100:.2f}% | 死叉转空仓(冻结)")
    print(f"{'='*92}")
    print(f"  {'方案':<26}{'总收益':>10}{'年化':>9}{'最大回撤':>10}{'Sharpe':>9}{'持币%':>8}")
    print(f"  {'满仓持有(买入不动)':<22}{pct(rb):>10}{pct(ab):>9}{pct(mdb):>10}{sb:>9.2f}{0.0:>7.1f}%")
    print(f"  {'MACD择时(纯空仓)':<22}{pct(rc):>10}{pct(ac):>9}{pct(mdc):>10}{sc:>9.2f}{cr:>7.1f}%")

    # ── 交易清单 ──
    g = pd.Series(golden, index=base.index)
    gprev = g.shift(1, fill_value=False)
    buys = base.index[(g & ~gprev).values]
    sells = base.index[(~g & gprev).values]
    trades = sorted([(d, 'BUY 金叉满仓', float(hs.loc[d])) for d in buys] +
                    [(d, 'SELL 死叉清仓', float(hs.loc[d])) for d in sells])
    print(f"\n  ── 交易清单（共{len(trades)}笔: 买{len(buys)}/卖{len(sells)}）──")
    for d, act, px in trades:
        print(f"    {d}  {act}  @沪深300收盘 {px:.2f}")

    # ── 输出CSV ──
    out = args.out or f"etf_macd_timing_{args.etf}.csv"
    pd.DataFrame({'date': base.index,
                  'nav_buyhold': nav_bh.values,
                  'nav_macd_cash': nav_cash.values}).to_csv(out, index=False)
    print(f"\n  [输出] 每日净值已存 {out}（可 --base-nav 喂给 macd_plugin_validate 对照）")
    print(f"  [口径] 指数代理近似ETF；真实ETF含上市日限制/折溢价/买卖价差；信号收盘算、")
    print(f"         实盘需次日开盘成交(轻微乐观)。--etf-cost 已扣管理费+跟踪误差。")


if __name__ == '__main__':
    main()
