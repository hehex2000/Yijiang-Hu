# -*- coding: utf-8 -*-
"""
MACD 策略 + KDJ-J 确认门 净值消融
=================================
验证 Jim 第4期论点在"组合策略"层面的落地：把 KDJ-J 当确认门（MACD 定结构、
KDJ 做语境确认），是否真的改善 MACD 背离策略的净值。

对照（固定参数、全样本、Jim 规矩③ 范式，不看完结果再改参数）：
  1) 无门        : kdj_gate=False
  2) recover 门  : J 由负区拐头向上 (J_t>0 & J_{t-1}<=0)，N=20  —— 事件研究最强 edge
  3) rising_low门: J 处下半区且当日上行 (J<50 & J_t>J_{t-1})      —— 更宽"回暖"语境

指标：总收益 / 年化 / 最大回撤 / 夏普 + 持仓画像(平均持仓数/活跃月) 防"空仓取巧"。

判定：门版总收益 > 无门 且 (夏普改善 或 回撤降低) → 确认门有效，可接入菜单取代退休位。
"""
from __future__ import annotations
import os, sys, io, contextlib

import run_macd_regime as rm

START, END = "20140301", "20260731"
POOL = "hs300"
CAPITAL = 100000
TOP_N = 10
LOG = "data/results/macd_strategy/ablation_gate.log"
os.makedirs(os.path.dirname(LOG), exist_ok=True)


def run_once(label, **kw):
    logf = open(LOG, "a", encoding="utf-8")
    with contextlib.redirect_stdout(logf):
        try:
            rep = rm.run_strategy(START, END, pool=POOL, capital=CAPITAL,
                                  top_n=TOP_N, **kw)
        finally:
            logf.close()
    return rep


def main():
    open(LOG, "w", encoding="utf-8").close()  # 清空日志
    print("=" * 96)
    print("  MACD 背离策略 + KDJ-J 确认门 净值消融（固定参数、全样本）")
    print(f"  区间 {START}~{END} | 池 {POOL} | 持仓 {TOP_N} | 资金 {CAPITAL:,}")
    print("=" * 96)

    cfgs = [
        ("① 无门 (baseline)", dict(kdj_gate=False)),
        ("② recover门 (J由负拐正,N=20)", dict(kdj_gate=True, kdj_gate_mode="recover", kdj_n=20)),
        ("③ rising_low门 (J下半区上行)", dict(kdj_gate=True, kdj_gate_mode="rising_low", kdj_n=20)),
    ]
    res = {}
    for label, kw in cfgs:
        print(f"\n>>> 运行 {label} ...", flush=True)
        rep = run_once(label, **kw)
        res[label] = rep
        print(f"    {label}: 总收益={rep['total']:+.2%} 年化={rep['annual']:+.2%} "
              f"回撤={rep['mdd']:+.2%} 夏普={rep['sharpe']:.4f} "
              f"均持仓={rep.get('avg_holdings',0):.1f} 活跃月={rep.get('months_active',0)}/{rep.get('total_months',0)}")

    base = res["① 无门 (baseline)"]
    print("\n" + "=" * 96)
    print("  对照表（相对无门 baseline 的增量）")
    print("=" * 96)
    hdr = f"  {'配置':<30}{'总收益':>10}{'Δ总收益':>10}{'年化':>9}{'最大回撤':>10}{'夏普':>8}{'Δ夏普':>8}{'均持仓':>8}"
    print(hdr)
    print("  " + "-" * 88)
    for label, _ in cfgs:
        r = res[label]
        d_t = r["total"] - base["total"]
        d_s = r["sharpe"] - base["sharpe"]
        print(f"  {label:<28}{r['total']:>+9.2%}{d_t:>+9.2%}{r['annual']:>+8.2%}"
              f"{r['mdd']:>+9.2%}{r['sharpe']:>8.4f}{d_s:>+8.4f}{r.get('avg_holdings',0):>8.1f}")

    # 判定
    print("\n" + "=" * 96)
    print("  判定（改善 = 门版总收益>无门 且 (夏普改善 或 回撤降低)）")
    print("=" * 96)
    best_label, best = None, None
    for label, _ in cfgs[1:]:
        r = res[label]
        improved = (r["total"] > base["total"]) and (
            r["sharpe"] > base["sharpe"] or r["mdd"] > base["mdd"])
        verdict = "✅ 改善" if improved else "❌ 未改善"
        print(f"  {label:<28} 总收益Δ={r['total']-base['total']:+.2%} "
              f"夏普Δ={r['sharpe']-base['sharpe']:+.4f} 回撤Δ={r['mdd']-base['mdd']:+.2%}  → {verdict}")
        if improved and (best is None or r["total"] > best["total"]):
            best_label, best = label, r
    if best:
        print(f"\n  >>> 最优确认门: {best_label}  → 可接入择时菜单取代 macd_kdj_RETIRED")
    else:
        print(f"\n  >>> 两种确认门均未通过判定，MACD 策略维持无门，不接入新门。")
    print(f"\n  [日志] 详细输出 → {LOG}")


if __name__ == "__main__":
    main()
