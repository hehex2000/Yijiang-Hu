# -*- coding: utf-8 -*-
"""
一票否决阈值敏感性分析（基于 run_industry_sentiment_rotation.py 的逻辑复刻·已修正版）
=========================================================================================
目的：检验视频里"综合分>=85 一票否决、主动放弃过热行业"的风控机制，
      是否真的能降低回撤；阈值到底设在哪最合理。

复用：直接 import run_industry_sentiment_rotation 的 build_pipeline / run_backtest，
      确保与主线回测用的是同一套（已修正 P0-1/P0-2/P1-3/P1-4/P1-5/P1-6 的）逻辑，
      避免两套实现漂移。

方法：因子与综合分（veto 前）只算一次；对一组 VETO 阈值各跑一遍完整回测。
      含一个"不否决"基准(VETO=999)作对照。

输出：
  - 控制台打印对比表（阈值 / 否决次数 / 年化 / 总收益 / 最大回撤 / 年化超额 / 熊市年收益）
  - veto_sensitivity.csv（可追溯的明细表）
"""
import pandas as pd
from run_industry_sentiment_rotation import build_pipeline, run_backtest, metrics

pipe = build_pipeline()
bench_nav = run_backtest(999, pipe)["bench_nav"]
bt, bc, bm = metrics(bench_nav)
print(f"[基准] 31行业等权: 总{bt*100:.1f}% 年化{bc*100:.2f}% 回撤{bm*100:.2f}%")

THRESHOLDS = [999, 95, 90, 85, 80, 75, 70, 60, 50]
rows = []
for V in THRESHOLDS:
    label = "无否决" if V >= 999 else f">={int(V)}"
    res = run_backtest(V, pipe)
    rows.append(dict(
        阈值=label, 否决次数=res["n_veto"], 移除榜首次数=res["n_top_removed"],
        年化=f"{res['cagr']*100:.2f}%", 总收益=f"{res['tot']*100:.1f}%",
        最大回撤=f"{res['mdd']*100:.2f}%", 年化超额=f"{res['excess']*100:.2f}pp",
        熊2018=f"{res['s2018']*100:.1f}%" if pd.notna(res['s2018']) else "-",
        熊2022=f"{res['s2022']*100:.1f}%" if pd.notna(res['s2022']) else "-",
    ))
    print(f"  VETO {label:>6} : 否决{res['n_veto']:>4}次 移除榜首{res['n_top_removed']:>3}次 | "
          f"年化{res['cagr']*100:6.2f}% 总{res['tot']*100:7.1f}% 回撤{res['mdd']*100:7.2f}% "
          f"超额{res['excess']*100:6.2f}pp | 2018={res['s2018']*100:6.1f}% 2022={res['s2022']*100:6.1f}%")

df = pd.DataFrame(rows)
print("\n================ 一票否决阈值敏感性（已修正版） ================")
print(df.to_string(index=False))
print("==============================================================")

base = run_backtest(999, pipe)
v85 = run_backtest(85.0, pipe)
print(f"\n对照：无否决 回撤={base['mdd']*100:.2f}%  年化={base['cagr']*100:.2f}%  超额={base['excess']*100:.2f}pp")
print(f"      VETO>=85 回撤={v85['mdd']*100:.2f}%  年化={v85['cagr']*100:.2f}%  超额={v85['excess']*100:.2f}pp")
print(f"      回撤变化：{(v85['mdd']-base['mdd'])*100:+.2f}pp（负=降回撤）  年化变化：{(v85['cagr']-base['cagr'])*100:+.2f}pp")

df.to_csv("veto_sensitivity.csv", index=False)
print("\n[save] veto_sensitivity.csv")
