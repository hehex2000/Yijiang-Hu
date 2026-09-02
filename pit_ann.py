# -*- coding: utf-8 -*-
"""
PIT(point-in-time) 财务公告日规范化的公共工具。

背景（2026-09-02 全库 36 表扫描确认）
------------------------------------
库里 `fina_indicator.ann_date` 是 **REAL 浮点**（存成 20201029.0，另有 15174 行 NULL），
而 income / balance_sheet / cashflow / express / forecast 的 ann_date 都是 TEXT。
**日期列的类型污染全库只有这一处。**

影响面（已实测，勿凭印象放大）
------------------------------
* SQL 侧 **无影响**：SQLite 的列 affinity 会把 TEXT 参数转成数值再比较。实测
  `ann_date='20201029'`(753 行) / `<= '20200101'`(107,885 行) / 与 income 的 JOIN
  (抽样 300 只 15,563 行) / substr(ann_date,1,4) / LIKE '2020%' —— 结果与数值口径
  **逐行一致**。（此前一度记录为"字符串匹配必然查不到"，已证伪并订正。）
* Python 侧 **有隐患**：pandas 的 `.astype(str)` 会把 20201029.0 变成 **'20201029.0'**。
  实测对 bisect 字典序 PIT 取值**无实质影响**——因为日期的字典序恰好等于时间序，
  且前 8 位就是完整日期，'20201029.0' < '20201030' 仍成立。
  但这是**依赖巧合的脆弱写法**：一旦拿它做 join key、去重 key 或精确匹配，
  '20201029' 与 '20201029.0' 会被当成两个不同值，静默出错且极难排查。

所以统一走 `norm_ann()`：无论输入是 float / int / str / None，一律返回规范的
'YYYYMMDD' 字符串；不可解析的一律置 NaN（保守丢弃，绝不自作聪明猜日期）。

用法
----
    from pit_ann import norm_ann
    df["ann"] = norm_ann(df["ann_date"])
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["norm_ann", "norm_ann_one"]


def norm_ann_one(x):
    """单值版本。返回 'YYYYMMDD' 字符串，不可解析返回 np.nan。"""
    if x is None:
        return np.nan
    # 先判 NaN（float('nan') != 自身）
    if isinstance(x, float) and x != x:
        return np.nan
    s = str(x).strip()
    # 20201029.0 -> 20201029（REAL 列的浮点尾缀）
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s if len(s) == 8 and s.isdigit() else np.nan


def norm_ann(s: pd.Series) -> pd.Series:
    """
    把 ann_date 列规范成 'YYYYMMDD' 字符串 Series。

    - 20201029.0(float) -> '20201029'
    - 20201029  (int)   -> '20201029'
    - '20201029'(str)   -> '20201029'
    - None / nan / 非法 -> NaN（交给下游 dropna 丢弃）
    """
    txt = s.astype(str).str.strip()
    # 去掉 REAL 列的 '.0' 尾缀；TEXT 列本来就没尾缀，不受影响
    txt = txt.str.replace(r"\.0$", "", regex=True)
    ok = txt.str.fullmatch(r"\d{8}")
    return txt.where(ok, other=np.nan)
