# -*- coding: utf-8 -*-
"""
低波因子（低波动异象 Low-Volatility Anomaly，A 股长期有效）

逻辑：过去 60 个交易日收益率波动最小的股票，长期风险调整后收益更优。
因子 = 过去 60 日日收益波动率；多头选"波动最小"的一档（因子最小）。
"""
import numpy as np
import pandas as pd

NAME = "low_vol"
DIRECTION = "asc"      # 多头 = 因子最小的一档（波动最低）
WINDOW = 60

def compute(df, window=WINDOW):
    """
    参数
    ----
    df : 逐笔 DataFrame，须含 ts_code, close_adj
    返回
    ----
    原 df 新增列 low_vol（逐笔；NaN = 数据不足）
    """
    df = df.copy()
    daily_ret = df.groupby('ts_code')['close_adj'].pct_change()
    df[NAME] = (
        daily_ret.groupby(df['ts_code'])
                 .rolling(window).std().values
    )
    return df
