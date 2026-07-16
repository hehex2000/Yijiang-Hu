# -*- coding: utf-8 -*-
"""
APB5D 因子（示例 / 参考，已证实在 A 股日线数据上无效）

买卖压力因子：5 日 VWAP vs 5 日 TWAP 的偏离。
  买压大 -> VWAP < TWAP -> APB < 0 -> 多头选最小的一档。

作为动态因子平台的"反面教材"保留：同样日线数据、同样严谨回测，
APB 的 RankIC 接近 0，证明因子本身无效，而非框架问题。
"""
import numpy as np
import pandas as pd

NAME = "apb5d"
DIRECTION = "asc"      # 多头 = 因子最小的一档（买压最强）
WINDOW = 5

def compute(df, window=WINDOW):
    """
    参数
    ----
    df : 逐笔 DataFrame，须含 ts_code, vol, amount, adj_factor, close_adj
    返回
    ----
    原 df 新增列 apb5d（逐笔）
    """
    df = df.copy()
    latest_adj = df.groupby('ts_code')['adj_factor'].transform('last')
    ratio = df['adj_factor'] / latest_adj
    df['vwap_adj'] = (df['amount'] / df['vol'] * 10) * ratio
    df['vwap_adj_x_vol'] = df['vwap_adj'] * df['vol']
    df['vwap_adj_x_vol_5d'] = (
        df.groupby('ts_code')['vwap_adj_x_vol']
          .rolling(window=window, min_periods=3).sum().values
    )
    df['vol_5d'] = (
        df.groupby('ts_code')['vol']
          .rolling(window=window, min_periods=3).sum().values
    )
    df['vwap_5d'] = df['vwap_adj_x_vol_5d'] / df['vol_5d']
    df['twap_5d'] = (
        df.groupby('ts_code')['close_adj']
          .rolling(window=window, min_periods=3).mean().values
    )
    df[NAME] = (df['vwap_5d'] - df['twap_5d']) / df['twap_5d']
    df[NAME] = df[NAME].replace([np.inf, -np.inf], np.nan)
    return df
