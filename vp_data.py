# -*- coding: utf-8 -*-
"""
vp_data: Volume Profile 数据层（计划 M1）
本地 SQLite astock_daily.db 取日线 OHLCV，零在线依赖（tushare 付费数据已落库）。

口径说明：
- 支撑/压力位检测用的是市场实际成交价，因此默认不复权(raw)，
  历史密集成交区就在当时真实交易的价格上，分红跳空不应平移这些"记忆"。
- 若做横截面因子（M2）需要可比净值口径，可在调用处传 adjust='hfq'。
"""
import os
import sqlite3

import numpy as np
import pandas as pd

import config

DB_PATH = config.DATA.get("local_db_path", "")


def _conn():
    return sqlite3.connect(DB_PATH)


def get_daily(ts_code, start=None, end=None, adjust="raw"):
    """取单只日线 OHLCV。adjust='raw' 用 daily 原值；'hfq' 经 adj_factor 还原后复权。
    返回按 trade_date 升序的 DataFrame[date, open, high, low, close, vol]。
    """
    conn = _conn()
    try:
        sql = "SELECT trade_date, open, high, low, close, vol FROM daily WHERE ts_code=?"
        params = [ts_code]
        if start:
            sql += " AND trade_date>=?"
            params.append(start)
        if end:
            sql += " AND trade_date<=?"
            params.append(end)
        sql += " ORDER BY trade_date ASC"
        df = pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()
    if df.empty:
        return df
    if adjust == "hfq":
        df = _apply_hfq(df, ts_code)
    df = df.rename(columns={"trade_date": "date"})
    df["date"] = df["date"].astype(int)
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    df["vol"] = df["vol"].astype(float)
    return df.reset_index(drop=True)


def _apply_hfq(df, ts_code):
    conn = _conn()
    try:
        adj = pd.read_sql_query(
            "SELECT trade_date, adj_factor FROM adj_factor WHERE ts_code=? ORDER BY trade_date ASC",
            conn, params=(ts_code,),
        )
    finally:
        conn.close()
    if adj.empty:
        return df
    m = adj.merge(df, on="trade_date", how="right")
    latest = float(m["adj_factor"].iloc[-1])
    for c in ["open", "high", "low", "close"]:
        m[c] = m[c] * m["adj_factor"] / latest
    return m[["trade_date", "open", "high", "low", "close", "vol"]]


def get_window(ts_code, trade_date, lookback=120, adjust="raw"):
    """取 trade_date 及之前 lookback 个交易日（升序），用于滚动计算密集区。"""
    df = get_daily(ts_code, end=trade_date, adjust=adjust)
    return df.tail(lookback).reset_index(drop=True)
