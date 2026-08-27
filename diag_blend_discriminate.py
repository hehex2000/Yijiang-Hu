"""诊断：连续 F-score 加权(blend)是否真的改变了持仓集合。

背景：此前 w=0.25/0.5/0.75 三档与 OFF 逐字节相同，根因是 ⑤ 纵向分位默认按
top_n 提前截断候选池，blend 在 ⑤ 之后执行 → 重排一个已全持有集合 → 等权下 no-op。
修复：blend 已移到 ⑤ 之前，在全候选池上重排。

本脚本只跑"选股"不跑"回测"，快速验证：
  - 各调仓日 blend(w=0.5) 的 top5 是否 ≠ OFF 的 top5（区分度）
  - 候选池(⑤前)规模分布（是否经常 > top_n=5，决定 blend 能否区分）
  - F-score 覆盖率（F 数据是否可得）
用法（venv 下）：
  ./venv_ml/Scripts/python.exe diag_blend_discriminate.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

try:
    from run_monthly_rebalance import get_trade_dates
except Exception:
    from run_monthly_rebalance import get_monthly_5th_trading_days as get_trade_dates

from value_stock_selector import select_value_stocks
import pandas as pd

START, END = "20100101", "20251231"
dates = get_trade_dates(START, END)

tot = diff5 = diff999 = empty = 0
pool_sizes = []
f_cov = []
for d in dates:
    off = select_value_stocks(d, top_n=999, stock_pool="zz800")
    bl = select_value_stocks(d, top_n=999, stock_pool="zz800", piotroski_blend=0.5)
    if off.empty:
        empty += 1
        continue
    tot += 1
    off5 = set(off["ts_code"].head(5))
    bl5 = set(bl["ts_code"].head(5))
    if off5 != bl5:
        diff5 += 1
    off999 = set(off["ts_code"].head(999))
    bl999 = set(bl["ts_code"].head(999))
    if off999 != bl999:
        diff999 += 1
    pool_sizes.append(len(off))
    if "fscore" in bl.columns:
        f_cov.append(bl["fscore"].notna().mean())

ps = pd.Series(pool_sizes)
print(f"调仓日总数={len(dates)}  空候选={empty}  有效={tot}")
print(f"blend(w=0.5) top5 与 OFF 不同: {diff5}/{tot} = {diff5/max(tot,1):.1%}")
print(f"blend(w=0.5) 全候选池 与 OFF 不同: {diff999}/{tot} = {diff999/max(tot,1):.1%}")
print(f"候选池(⑤前,top_n=999) 中位={ps.median():.0f} 最大={ps.max()} <=5天数={int((ps<=5).sum())}/{tot}")
if f_cov:
    print(f"F-score 覆盖率(非None占比)均值={pd.Series(f_cov).mean():.1%}")
print("判定：diff5 明显>0 且 候选池常>5 → 修复生效，可重跑完整回测验证收益区分度")
