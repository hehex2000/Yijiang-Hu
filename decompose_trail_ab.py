# -*- coding: utf-8 -*-
"""
移动止盈 vs 现有止损层 —— 组合级 A/B 对照（overlay 模式，不重跑回测）
================================================================
复用 decompose_platform_overlay 的基线/缓存与 macd_plugin_validate 的指标函数。

支持 --base-nav <csv> 用任意自定义基线（如红利低波指数 930955 真实净值）替换等权中证800，
消除「高波动基线 → 锁存止损永久踏空」的失真。overlay 重算秒级，不重跑回测。

核心对照（同一共同基线 = 等权中证800 日净值，或 --base-nav 指定的红利低波净值）：
  - 现有止损层 = 锁存硬止损：组合峰回撤 15% 触止损后永久持币，直到沪深300 MACD 金叉解锁
  - 移动止盈   = 非锁存跟随：base_v >= peak*(1-thr) 持仓，跌破阈值离场、涨回即回场（trailing）

指标（回撤-收益权衡，同 macd_plugin_validate）：
  DD_cut   = 基线MDD − 方法MDD        (pp, 越大=回撤砍得越多=好)
  Ret_cost = 基线收益 − 方法收益       (pp, 越大=牺牲越多=坏)
  Eff      = DD_cut / max(Ret_cost,ε) (越大=每牺牲1pp收益换来的回撤削减越多=省心)

注意（概念边界，见报告）：
  平台现有止损层作用在「组合层」，本身已是回撤触发（移动止盈性质）。
  本脚本量化「组合级移动止盈阈值 thr」vs「现有锁存15%硬止损」的取舍。
  个股级移动止盈（P1① 验证）需回到具体策略个股持仓重跑，不在本脚本范围（超时另排）。
"""
import sys, os
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

import macd_plugin_validate as M
from regime_cash_overlay import load_index_close, BENCH, apply_overlay, cash_ratio

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_decomp_cache.pkl')
START, END = '20100101', '20251231'


def load_base():
    if os.path.exists(CACHE):
        base, hs = pd.read_pickle(CACHE)
        print(f"[cache] 命中基线/沪深300，长度 {len(base)}")
    else:
        base = M.load_base_zz800_eq(START, END)
        hs = load_index_close(BENCH, START, END)
        hs = hs.reindex(base.index).ffill()
        pd.to_pickle((base, hs), CACHE)
        print(f"[load] 已缓存基线")
    return base.values.astype(float), hs.values.astype(float)


def load_hs():
    """沪深300 收盘（现有止损层 MACD 金叉解锁信号源），Series 带 int 日期索引。"""
    return load_index_close(BENCH, START, END)


def load_base_nav(path):
    """从 CSV [date, nav] 读取自定义基线，归一化到首日=1.0，返回与 hs 日期对齐的 (base_v, hs_v)。

    红利低波真实净值（930955）与沪深300 同取自 index_daily 同一窗口 → 交易日集一致，
    位置对齐成立；reindex+ffill 兜底任何微小错位。
    """
    d = pd.read_csv(path)
    date_col = next(c for c in d.columns if 'date' in c.lower())
    nav_col = next(c for c in d.columns if c.lower() in ('nav', 'close', 'value', '净值'))
    s = pd.Series(d[nav_col].astype(float).values, index=d[date_col].astype(int).values)
    s = s.sort_index()
    s = s / s.iloc[0]
    hs = load_hs()
    s = s.reindex(hs.index).ffill()
    return s.values.astype(float), hs.values.astype(float)


def locked_stop(base_v, peak, thr, golden=None):
    """锁存硬止损：峰回撤 thr 触后持币，直到 golden[i] 解锁（None=不解锁=永久离场）。"""
    hit = base_v < peak * (1 - thr)
    stopped = np.zeros(len(base_v), dtype=bool)
    s = False
    for i in range(len(base_v)):
        if hit[i]:
            s = True
        if golden is not None and golden[i]:
            s = False
        stopped[i] = s
    return ~stopped  # True=持仓


def trailing(base_v, peak, thr):
    """非锁存移动止盈跟随：价格 >= 峰值*(1-thr) 持仓，否则离场（涨回即回场）。"""
    return base_v >= peak * (1 - thr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-nav', default=None,
                    help='自定义基线 CSV [date, nav]；省略则用等权中证800')
    args = ap.parse_args()

    if args.base_nav:
        base_v, hs_v = load_base_nav(args.base_nav)
        base_name = f"自定义基线({args.base_nav})"
        print(f"[base-nav] 使用自定义基线: {args.base_nav}  对齐后长度 {len(base_v)}")
    else:
        base_v, hs_v = load_base()
        base_name = "等权中证800"
    peak = pd.Series(base_v).cummax().values
    golden = M.macd_golden(hs_v).values

    def run(mask, name):
        nav = apply_overlay(base_v, mask)
        rb, ab, mdb, sb = M.metrics(nav)
        cr = cash_ratio(mask) * 100
        return dict(name=name, rb=rb, ab=ab, mdb=mdb, sb=sb, cr=cr, nav=nav)

    # ── 方案族 ──
    D = np.ones(len(base_v), dtype=bool)                       # 无控制
    A15 = locked_stop(base_v, peak, 0.15, golden)              # 现有：锁存15%+MACD
    B15 = locked_stop(base_v, peak, 0.15)                     # 现有纯止损：锁存15%硬止损
    trails = {f"移动止盈 thr={int(t*100)}%": trailing(base_v, peak, t)
              for t in (0.10, 0.15, 0.20, 0.25)}

    rows = [run(D, 'D 无控制(满仓等权800)'),
            run(B15, 'B15 现有纯止损(锁存15%)'),
            run(A15, 'A15 现有完整(锁存15%+MACD)')]
    for nm, msk in trails.items():
        rows.append(run(msk, nm))

    rbD, mdbD = rows[0]['rb'], rows[0]['mdb']
    eps = 1e-9

    def pct(x): return f"{x*100:+.2f}%"

    print(f"\n{'='*104}")
    print(f"  组合级 A/B：移动止盈(非锁存跟随) vs 现有锁存硬止损 | 基线={base_name} {START}-{END}")
    print(f"{'='*104}")
    print(f"  {'方案':<24}{'总收益':>9}{'年化':>8}{'最大回撤':>10}{'Sharpe':>8}{'持币%':>7}")
    for r in rows:
        print(f"  {r['name']:<24}{pct(r['rb']):>9}{pct(r['ab']):>8}{pct(r['mdb']):>10}{r['sb']:>8.2f}{r['cr']:>6.1f}%")

    print(f"\n  ── 回撤-收益权衡（相对无控制 D）──")
    print(f"  {'方案':<24}{'DD_cut':>9}{'Ret_cost':>10}{'Eff':>9}")
    for r in rows[1:]:
        dd_cut = (mdbD - r['mdb']) * 100
        ret_cost = (rbD - r['rb']) * 100
        # Eff 仅在「方法收益<基线（有牺牲）」时有意义；收益已超基线则标 N/A
        if ret_cost > 1e-6:
            eff = dd_cut / ret_cost
            eff_s = f"{eff:>9.2f}"
        else:
            eff_s = f"{'N/A':>9}"   # 已超越基线：回撤控制无「牺牲收益」语境
        print(f"  {r['name']:<24}{dd_cut:+8.2f}pp{ret_cost:+9.2f}pp{eff_s}")

    # ── 增量：移动止盈 thr=20% vs 现有纯止损 15% ──
    b15 = next(r for r in rows if r['name'].startswith('B15'))
    t20 = next(r for r in rows if r['name'] == '移动止盈 thr=20%')
    print(f"\n  ── 核心对照：移动止盈20% vs 现有锁存15% ──")
    print(f"    现有15%纯止损 : 回撤 {pct(b15['mdb'])} 收益 {pct(b15['rb'])} 持币 {b15['cr']:.1f}%")
    print(f"    移动止盈20%   : 回撤 {pct(t20['mdb'])} 收益 {pct(t20['rb'])} 持币 {t20['cr']:.1f}%")
    print(f"    Δ回撤={pct(t20['mdb']-b15['mdb'])}  Δ收益={pct(t20['rb']-b15['rb'])}")
    print(f"  结论：Eff 最大者=每牺牲1pp收益砍回撤最多者，即组合级最优移动止盈阈值。")


if __name__ == '__main__':
    main()
