# -*- coding: utf-8 -*-
"""长窗口(2013-2026) 年度 vs 季度 汇总：换手双口径 + 逐年收益配对检验。

用法： venv_ml/Scripts/python.exe -u divlow_longwin_summarize.py
"""
import numpy as np
import pandas as pd
from scipy import stats

import divlow_turnover_ab as T

RES = "data/results/dividend_low_vol"
PAT = {
    "year":    "_official_official_compact_all_12_bk0_rbyear_20130101_20260903_partial.csv",
    "quarter": "_official_official_compact_all_12_bk0_20130101_20260903_partial.csv",
}

print("=" * 78)
print("【1】换手（2013-2026 长窗口，双口径）")
print("=" * 78)
turn = {}
for k, p in PAT.items():
    turn[k] = T.turnover_for(p, top_n=12)

print()
print("=" * 78)
print("【2】逐年收益配对检验（year − quarter，n=14）")
print("=" * 78)
y = pd.read_csv(f"{RES}/bt_quality_yearly_20130101_20260903_official_compact_all_12_rbyear_hfq.csv",
                encoding="utf-8-sig", index_col=0)
q = pd.read_csv(f"{RES}/bt_quality_yearly_20130101_20260903_official_compact_all_12_hfq.csv",
                encoding="utf-8-sig", index_col=0)
# 取策略列（第一列）
yc, qc = y.columns[0], q.columns[0]
common = [i for i in y.index if i in q.index]
a = pd.to_numeric(y.loc[common, yc]).values / 100.0
b = pd.to_numeric(q.loc[common, qc]).values / 100.0
d = a - b
t_stat, p_val = stats.ttest_rel(a, b)
w_stat, w_p = stats.wilcoxon(d)

tbl = pd.DataFrame({"year%": a * 100, "quarter%": b * 100, "diff_pp": d * 100}, index=common)
print(tbl.round(2).to_string())

geo = lambda x: (np.prod(1 + x)) ** (1 / len(x)) - 1
print()
print(f"  算术年收益   year={a.mean()*100:6.2f}%   quarter={b.mean()*100:6.2f}%   diff={(a.mean()-b.mean())*100:+.2f}pp")
print(f"  几何年化     year={geo(a)*100:6.2f}%   quarter={geo(b)*100:6.2f}%   diff={(geo(a)-geo(b))*100:+.2f}pp")
print(f"  逐年 std     year={a.std(ddof=1)*100:6.2f}%   quarter={b.std(ddof=1)*100:6.2f}%   diff={(a.std(ddof=1)-b.std(ddof=1))*100:+.2f}pp")
print(f"  配对 t (逐年收益算术差)  t={t_stat:+.3f}  p={p_val:.3f}   n={len(d)}")
print(f"  Wilcoxon 符号秩                W={w_stat:.1f}  p={w_p:.3f}")
print(f"  胜年数  year 赢 {int((d>0).sum())} / {len(d)}")
print()
print("  判读：|t|<2 → 收益差异**不显著**（n=14，逐年收益噪声大）。")
print("        年度档的几何年化反超来自**波动更小→波动拖累更少**，不是每年都赢。")

if turn and not turn["year"].empty and not turn["quarter"].empty:
    print()
    print("=" * 78)
    print("【3】换手降幅 + 🔴 成本拆解（毛 alpha 是否被改变？）")
    print("=" * 78)
    # 引擎费率（run_monthly_rebalance.py）：
    #   买 = 佣金 0.025% + 滑点 0.1%            = 0.125%
    #   卖 = 佣金 0.025% + 滑点 0.1% + 印花税 0.1% = 0.225%  ← 本次回测走旧税率(引擎告警已提示)
    #   → 每 1 单位「年化单边换手 T」的年成本 = T × (0.125% + 0.225%) = T × 0.35%
    BUY, SELL = 0.00125, 0.00225
    SPAN = 13.67            # 2013-01-10 ~ 2026-09-03
    NET_ANN = {"year": 0.0913, "quarter": 0.0878}   # 引擎输出（复利年化）
    t_ann = {}
    for k in ("year", "quarter"):
        n_per_year = len(turn[k]) / SPAN
        for col, name in (("one_way_w", "资金口径"), ("one_way_n", "只数口径")):
            t_ann.setdefault(k, {})[name] = turn[k][col].mean() * n_per_year
        print(f"  {k:8s} 资金口径年化单边换手 = {t_ann[k]['资金口径']*100:.1f}%"
              f"   只数口径 = {t_ann[k]['只数口径']*100:.1f}%")

    for name in ("资金口径", "只数口径"):
        cy = t_ann["year"][name] * (BUY + SELL)
        cq = t_ann["quarter"][name] * (BUY + SELL)
        net_d = NET_ANN["year"] - NET_ANN["quarter"]
        gross_d = net_d - (cq - cy)
        print(f"\n  —— {name} ——")
        print(f"    年交易成本  year={cy*100:.3f}pp/年   quarter={cq*100:.3f}pp/年   → 省 {(cq-cy)*100:.3f}pp/年")
        print(f"    净年化差    {net_d*100:+.2f}pp  (year {NET_ANN['year']*100:.2f}% vs quarter {NET_ANN['quarter']*100:.2f}%)")
        print(f"    → 毛年化差  {gross_d*100:+.3f}pp   ← 扣掉成本节省后的真实 alpha 差")
        print(f"    成本解释占比 = {(cq-cy)/net_d*100 if net_d else float('nan'):.0f}%")
