# -*- coding: utf-8 -*-
"""
弱市持币（regime cash overlay）— 独立 overlay 模块
==================================================
从 BV16Dbe6MEsu「板块轮动交易法」抽取出的「弱市持币」风控思想，独立成可复用模块：
  信号：沪深300收盘 < MA(window)  →  BEAR（弱市，持币/空仓，不出手）
        沪深300收盘 >= MA(window) →  BULL（强市，跟随底层策略）
  行为：BEAR 期组合 100% 现金（净值冻结在离场值，仅可选微增货基利息）；BULL 期跟随底层策略净值。

设计要点（防未来函数 / 公平对照）：
  - 信号在「当日收盘」用截至当日的 MA 计算，不做任何前瞻。
  - 纯 NAV 变换：apply_overlay(base_nav, signal) 对任意底层策略净值做 overlay，
    因此可公平叠加到「无风控基线 / 平台15%止损 / 任意策略」做止损层效率对照。
  - 现金期默认收益=0（保守，不假设货基利息；如需可传 cash_growth>1）。

被主线④ (macd_plugin_validate.py) 调用，与「平台15%止损+MACD减仓」做止损层效率对照。
"""
import sqlite3
import numpy as np
import pandas as pd

try:
    from run_monthly_rebalance import get_conn
except Exception:
    def get_conn():
        return sqlite3.connect(r'D:/tu-shareData/astock_daily.db')

BENCH = '000300.SH'     # 大盘基准（默认用沪深300 判弱市）
DEFAULT_MA = 250        # 长周期市场过滤（板块轮动扫描里唯一跑赢BH的组合窗口）


def load_index_close(ts_code=BENCH, start='20100101', end='20251231'):
    """加载单只指数收盘价序列（trade_date 作 int 索引）。"""
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT trade_date,close FROM index_daily WHERE ts_code=? "
        "AND trade_date>=? AND trade_date<=? ORDER BY trade_date",
        conn, params=(ts_code, int(start), int(end)))
    conn.close()
    if len(df) == 0:
        return pd.Series(dtype=float)
    df['trade_date'] = df['trade_date'].astype(int)
    return pd.Series(df['close'].astype(float).values,
                     index=df['trade_date'].values, name=ts_code)


def regime_signal(close, ma_len=DEFAULT_MA, method='trend'):
    """返回与 close 等长、索引相同的布尔序列：True=BULL(持多), False=BEAR(持币)。

    method='trend'：收盘 > MA(ma_len) 为 BULL（默认，与板块轮动弱市持币口径一致）。
    """
    close = pd.Series(close, dtype=float)
    ma = close.rolling(ma_len).mean()
    if method == 'trend':
        sig = close >= ma
    else:
        raise ValueError(f"未知 method: {method}")
    return sig.fillna(False).astype(bool)


def apply_overlay(base_nav, signal, cash_growth=1.0):
    """纯 NAV 变换：BULL 期持有底层，BEAR 期持币（净值冻结）。

    Args:
        base_nav:   与 signal 同索引的净值序列（数值，首值=本金，如 1.0）。
        signal:     同索引布尔序列（True=BULL 持有, False=BEAR 持币）。
        cash_growth:现金期年化微增因子（默认1.0=不动；如货基1.015→日化^(1/252)）。
    Returns:
        overlay 净值序列（同索引）。
    """
    base_nav = np.asarray(base_nav, dtype=float)
    signal = np.asarray(signal, dtype=bool)
    if len(base_nav) != len(signal):
        raise ValueError("base_nav 与 signal 长度必须一致")
    n = len(base_nav)
    out = np.empty(n, dtype=float)
    scale = 1.0          # overlay / base_nav 的比例（BULL 期恒定 → 跟底层同比例）
    cash_val = None      # 离场时冻结的现金值
    in_cash = False
    days_cash = 0
    g = float(cash_growth) ** (1.0 / 252.0)
    for i in range(n):
        if signal[i]:
            if in_cash:
                # 重新入场：把冻结现金按当前底层净值比例转回
                scale = (cash_val / base_nav[i]) if base_nav[i] != 0 else scale
                in_cash = False
                days_cash = 0
            out[i] = scale * base_nav[i]
        else:
            if not in_cash:
                cash_val = scale * base_nav[i]
                in_cash = True
                days_cash = 0
            days_cash += 1
            out[i] = cash_val * (g ** days_cash)
    return pd.Series(out, index=getattr(base_nav, 'index', None))


def cash_ratio(signal):
    """持币期占比（signal 中 False 的比例）。"""
    s = pd.Series(signal, dtype=bool)
    return float((~s).mean()) if len(s) else 0.0
