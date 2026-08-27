# -*- coding: utf-8 -*-
"""摩擦全模型 v2：等权800 + MACD择时的毛收益(+433.84%) 扣全量摩擦后剩多少？
摩擦 = ①日频再平衡漂移(比例费率0.30%/单位单边换手, 仅持仓日) + ②每轮MACD清仓/建仓全额单边(0.15%)
overlay 用已验证的 apply_overlay；摩擦以每日乘子施加在 overlay 日收益上。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import macd_plugin_validate as M
from regime_cash_overlay import apply_overlay

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_retmat_cache.pkl')
ret, codes, hs = pd.read_pickle(CACHE)
ret_v = ret.values.astype(float)
n_days, n_stk = ret_v.shape

golden = M.macd_golden(hs.values.astype(float)).values
rp = np.nanmean(ret_v, axis=1)
nav_gross = pd.Series((1.0 + np.nan_to_num(rp, 0.0)).cumprod())

# 日单边换手(漂移)
dev = np.abs(ret_v - rp[:, None])
T = 0.5 * np.nanmean(dev, axis=1) / (1.0 + rp)
T = np.nan_to_num(T, 0.0)

# 毛 overlay（已验证函数）
nav_ov = apply_overlay(nav_gross.values, golden)   # 返回 Series
gross = nav_ov.iloc[-1] / nav_ov.iloc[0] - 1

# 摩擦乘子序列
DRIFT, SW = 0.0030, 0.0015
fric = np.zeros(n_days)
for i in range(1, n_days):
    if golden[i]:
        fric[i] += T[i] * DRIFT          # 持仓日漂移
    if golden[i] != golden[i-1]:
        fric[i] += SW                    # 切换日清仓/建仓全额单边

# 净 overlay：日收益×(1-fric)
ov_ret = nav_ov.values[1:] / nav_ov.values[:-1]
nav_net = np.empty(n_days); nav_net[0] = 1.0
for i in range(1, n_days):
    nav_net[i] = nav_net[i-1] * ov_ret[i-1] * (1.0 - fric[i])
net = nav_net[-1] - 1

def pct(x): return f"{x*100:+.1f}%"
n_sw = int(np.sum(golden[1:] != golden[:-1]))
print("等权800 + MACD 择时 摩擦敏感性(比例费率模型):")
print(f"  毛收益(无摩擦)          : {pct(gross)}")
print(f"  净收益(漂移+切换全模型) : {pct(net)}")
print(f"  摩擦拖累(16年累计)      : {pct(net-gross)}")
print(f"  信号切换次数            : {n_sw} 次 (每轮清仓+建仓各扣0.15%)")
print(f"[注] 小账户按笔最低佣金¥5未计入——800只每日微单会显著放大比例模型；20只方案影响小")
