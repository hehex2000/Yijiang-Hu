# -*- coding: utf-8 -*-
"""再平衡摩擦审计：等权组合"每日再平衡到等权"的真实交易成本。
直接回答：+433%(800只日频再平衡+MACD) 是否被手续费摩擦吃掉大部分？

等权每日再平衡的日单边换手 T = 0.5 * mean(|r_i - r_p|) / (1+r_p)
（把每日漂移买回/卖掉的量，买卖对称）
费率(平台模型): 佣金0.025%/边 + 滑点0.1%/边 + 印花0.05%仅卖出
→ 每日成本 = T * (2*(0.025%+0.1%) + 0.05%) = T * 0.30%
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_retmat_cache.pkl')
ret, codes, hs = pd.read_pickle(CACHE)
print(f"[cache] 收益矩阵 {ret.shape}")

ret_v = ret.values.astype(float)
n_days, n_stk = ret_v.shape

# 沪深300 MACD 信号（对齐持币期）
import macd_plugin_validate as M
from regime_cash_overlay import load_index_close
golden = M.macd_golden(hs.values.astype(float)).values

def friction(subset_cols, mask):
    """subset_cols: 列号; mask: 只在这些天持仓(计算换手)。返回(日单边换手均值, 年化摩擦, 16年复合拖累)"""
    r = ret_v[:, subset_cols]
    rp = np.nanmean(r, axis=1)                       # 等权日收益
    dev = np.abs(r - rp[:, None])                    # |r_i - r_p|
    T = 0.5 * np.nanmean(dev, axis=1) / (1.0 + rp)   # 日单边换手
    T = np.where(mask, T, 0.0)                       # 空仓日不换手
    T_valid = T[mask]
    t_mean = np.nanmean(T_valid) if T_valid.size else 0.0
    daily_cost = t_mean * 0.0030                     # 0.30% 每单位单边换手
    ann_cost = daily_cost * 252
    # 16年复合拖累(按天扣)：1 - prod(1-daily_cost) 用日成本序列
    cost_series = T * 0.0030
    drag = 1.0 - np.prod(1.0 - cost_series[mask])
    return t_mean, ann_cost, drag

all_cols = np.arange(n_stk)
for label, mask in [('全窗口(含空仓日持仓比例)', np.ones(n_days, bool)),
                    ('仅MACD金叉持仓日(择时版)', golden)]:
    t_mean, ann_cost, drag = friction(all_cols, mask)
    print(f"\n[800只全池] {label}:")
    print(f"  日单边换手均值 T={t_mean*100:.3f}%  |  年化摩擦≈{ann_cost*100:.2f}%/年  |  "
          f"16年复合拖累≈{drag*100:.1f}%")

rng = np.random.default_rng(7)
print(f"\n[20只随机样本 x5] 日频再平衡摩擦（新方案规模参照）:")
for k in range(5):
    idx = rng.choice(n_stk, size=20, replace=False)
    t_mean, ann_cost, drag = friction(idx, golden)
    print(f"  样本{k+1}: 日单边换手 T={t_mean*100:.3f}%  |  年化摩擦≈{ann_cost*100:.2f}%/年  |  "
          f"16年复合拖累≈{drag*100:.1f}%")

print(f"\n[对照] 理论等权800+MACD = +433.84%(未扣任何摩擦)；若按上述年化摩擦简单折算：")
print(f"       择时版持仓日~50.5%，年化摩擦 ~= 全窗口年化x持仓比，结论见上方数字。")
print(f"[注] 此模型用比例费率；小资金账户按笔最低佣金¥5会显著放大（尤其800只的微单），未计入。")
