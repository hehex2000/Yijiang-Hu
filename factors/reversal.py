# -*- coding: utf-8 -*-
"""
短期反转因子（A 股反转效应显著，学术支撑强）

逻辑：过去 1 个月（约 21 个交易日）跌幅最大的股票，下个月倾向于反弹。
因子 = 过去 1 个月收益率；多头选"跌最多"的一档（因子最小）。
"""
import numpy as np
import pandas as pd

NAME = "reversal"
DIRECTION = "asc"      # 多头 = 因子最小的一档（过去跌最多）
LOOKBACK = 21          # ~1 个月交易日

def compute(df, lookback=LOOKBACK):
    """
    参数
    ----
    df : 逐笔 DataFrame，须含 ts_code, close_adj
    返回
    ----
    原 df 新增列 reversal（逐笔；NaN = 数据不足）
    """
    df = df.copy()
    df[NAME] = df.groupby('ts_code')['close_adj'].pct_change(lookback)
    return df
