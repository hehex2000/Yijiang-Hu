# -*- coding: utf-8 -*-
"""
动量因子（Jegadeesh & Titman 1993，12 个月动量）

逻辑：过去 12 个月涨幅最大的股票，惯性延续。
为规避短期反转污染，用 (t-1月) 到 (t-12月) 区间收益，而非含最近 1 个月。
因子 = 区间收益率；多头选"涨最多"的一档（因子最大）。
"""
import numpy as np
import pandas as pd

NAME = "momentum"
DIRECTION = "desc"     # 多头 = 因子最大的一档（动量最强）
LOOKBACK = 252         # ~12 个月交易日
SKIP = 21              # 跳过最近 1 个月，避免短期反转

def compute(df, lookback=LOOKBACK, skip=SKIP):
    """
    参数
    ----
    df : 逐笔 DataFrame，须含 ts_code, close_adj
    返回
    ----
    原 df 新增列 momentum（逐笔；NaN = 数据不足）
    """
    df = df.copy()
    g = df.groupby('ts_code')['close_adj']
    p_now = g.shift(skip)
    p_past = g.shift(skip + lookback)
    df[NAME] = p_now / p_past - 1
    return df
