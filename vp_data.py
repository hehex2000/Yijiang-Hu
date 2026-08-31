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
import threading

import numpy as np
import pandas as pd

import config

DB_PATH = config.DATA.get("local_db_path", "")

# 每线程复用一条长连接
_thread_local = threading.local()


def _conn():
    """返回本线程复用的 SQLite 连接（调用方不要关闭）。

    实测（2026-08-31，7.6GB 库，journal_mode=wal）：
      sqlite3 连接的 close() 约 79ms，一次查询仅约 0.25ms，connect 约 4ms。
    逐票 connect+close 的写法每票约 83ms，其中 95% 是关闭开销；
    全市场 5000 只 ≈ 7 分钟纯浪费。
    复用后 vp_stat_weekly 单周选股 26.9s -> 1.6s（16.8x），结果逐位一致。

    用 thread-local 而非全局单例：sqlite3 连接默认不允许跨线程使用，
    而 Streamlit（vp_dashboard）是多线程的，共享连接会抛
    "SQLite objects created in a thread can only be used in that same thread"。
    """
    conn = getattr(_thread_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(DB_PATH)
        _thread_local.conn = conn
    return conn


def close_conn():
    """显式关闭当前线程的连接（一般不需要；供进程退出或重载前调用）。"""
    conn = getattr(_thread_local, "conn", None)
    if conn is not None:
        try:
            conn.close()
        finally:
            _thread_local.conn = None


def get_daily(ts_code, start=None, end=None, adjust="raw"):
    """取单只日线 OHLCV。adjust='raw' 用 daily 原值；'hfq' 经 adj_factor 还原后复权。
    返回按 trade_date 升序的 DataFrame[date, open, high, low, close, vol]。
    """
    conn = _conn()
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
    adj = pd.read_sql_query(
        "SELECT trade_date, adj_factor FROM adj_factor WHERE ts_code=? ORDER BY trade_date ASC",
        conn, params=(ts_code,),
    )
    if adj.empty:
        return df
    m = adj.merge(df, on="trade_date", how="right")
    latest = float(m["adj_factor"].iloc[-1])
    for c in ["open", "high", "low", "close"]:
        m[c] = m[c] * m["adj_factor"] / latest
    return m[["trade_date", "open", "high", "low", "close", "vol"]]


def get_window(ts_code, trade_date, lookback=120, adjust="raw"):
    """取 trade_date 及之前 lookback 个交易日（升序），用于滚动计算密集区。

    在 SQL 端按 trade_date 倒序只取 lookback 行再翻正为升序，
    避免走 get_daily 把该票全部历史(~4000行)拉进 DataFrame 再 tail。
    行集合与升序结果与旧实现逐位一致（已用数据指纹校验）。
    """
    conn = _conn()
    sql = (
        "SELECT trade_date, open, high, low, close, vol FROM daily "
        "WHERE ts_code=? AND trade_date<=? ORDER BY trade_date DESC LIMIT ?"
    )
    df = pd.read_sql_query(sql, conn, params=(ts_code, trade_date, lookback))
    if df.empty:
        return df
    df = df.iloc[::-1].reset_index(drop=True)  # 翻正为升序
    if adjust == "hfq":
        df = _apply_hfq(df, ts_code)
    df = df.rename(columns={"trade_date": "date"})
    df["date"] = df["date"].astype(int)
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    df["vol"] = df["vol"].astype(float)
    return df.reset_index(drop=True)
