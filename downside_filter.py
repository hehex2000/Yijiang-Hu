#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
downside_filter.py — 「禁买仍在下跌通道标的」风控筛（可复用模块）

来源：超跌反弹三状态漏斗（BV1Qiuz6WE6Y, Jim）复现后剥离出的风控层。
      原漏斗：超跌 -> 止跌 -> 反弹；本模块取「止跌」条件的反向，
      把"仍处下跌通道"的标的标记为禁买，从候选池剔除（纯剔除、不新增标的）。

信号（后复权价，灭除权假新低；数据截止 trade_date 闭市，PIT 安全）：
  对每只候选标的，在调仓日 t 算：
    ① 仍创新低：close_t <= rolling(N_LOW).min() * (1+EPS_LOW)  低点还在下移/贴 lows
    ② 在中期趋势下：close_t < MA(N_MA)                         未站回短期均线
    ③ 量未缩：vol_5d均值 >= 下跌段均量(前 N_DOWN~N_DOWN_SKIP 日) * Q_VOL  抛压未退
  命中任一 -> 标 禁买（banned）。

参数默认复用超跌反弹漏斗初值（N1=20/N2=5/N3=10/X=15%/P=60%/D=12%/Q=0.8），
本筛只用其中 N_LOW/N_MA/下跌段/Q_VOL，不另调参。

NULL 容忍：历史样本不足（< N_MA 个交易日）或 adj_factor 缺失 -> 该标的跳过判断=保留，
          只剔除"有数据且明确仍在下跌通道"的标的（与价值四道门槛的 NULL 容忍一致）。
"""

import sqlite3
import numpy as np
import pandas as pd

DOWNSIDE_PARAMS = {
    "n_low": 5,        # 近 N_LOW 日最低价窗口（止跌低点不再下移）
    "n_ma": 20,        # MA 窗口（中期趋势）
    "n_down": 15,      # 下跌段起点（往前数 N_DOWN 日）
    "n_down_skip": 5,  # 下跌段跳过最近 N_DOWN_SKIP 日（取前 5~15 日作为下跌段均量）
    "eps_low": 0.005,  # 仍贴 lows 容差（0.5%）
    "q_vol": 0.8,      # 量缩阈值：近期均量 <= 下跌段均量 * Q_VOL 视为已缩
}


def _db_path_fallback():
    try:
        from config import DATA
        p = DATA.get("local_db_path")
    except Exception:
        p = None
    return p or "D:/tu-shareData/astock_daily.db"


def _fetch_window(ts_code, trade_date, db_path, n=25):
    """返回最近 n 个交易日(<=trade_date)的 (trade_date, hfq, vol) 序列，按日期升序。"""
    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT d.trade_date AS td, d.close AS c, d.vol AS v,
                   a.adj_factor AS af
            FROM daily d
            LEFT JOIN adj_factor a
              ON d.ts_code = a.ts_code AND d.trade_date = a.trade_date
            WHERE d.ts_code = ? AND d.trade_date <= ?
            ORDER BY d.trade_date DESC
            LIMIT ?
            """,
            con, params=(ts_code, trade_date, n),
        )
    finally:
        con.close()
    if len(df) == 0:
        return None
    df = df.sort_values("td").reset_index(drop=True)
    # 后复权：close * adj_factor（比例比较，无需归一化到最新）
    df["hfq"] = df["c"] * df["af"]
    return df


def _is_banned(hfq, vol, p):
    """单只标的判断：是否仍在下跌通道（需数据足够）。"""
    n = len(hfq)
    if n < p["n_ma"]:
        return False  # 样本不足 -> 保留（NULL 容忍）
    hfq = np.asarray(hfq, dtype=float)
    vol = np.asarray(vol, dtype=float)
    close_t = hfq[-1]

    # ① 仍贴 lows
    recent_low = hfq[-p["n_low"]:].min()
    banned_low = close_t <= recent_low * (1 + p["eps_low"])

    # ② 在 MA 下
    ma = hfq[-p["n_ma"]:].mean()
    banned_ma = close_t < ma

    # ③ 量未缩
    lo = max(0, n - p["n_down"])
    hi = max(lo, n - p["n_down_skip"])
    down_win = vol[lo:hi]
    down_vol = down_win.mean() if len(down_win) > 0 else 0.0
    vol5 = vol[-p["n_low"]:].mean()
    banned_vol = (down_vol > 0) and (vol5 >= down_vol * p["q_vol"])

    return bool(banned_low or banned_ma or banned_vol)


def compute_downside_banned(codes, trade_date, db_path=None, params=None):
    """返回禁买标的集合（ts_code 列表）。

    Args:
        codes: 候选 ts_code 列表
        trade_date: 调仓日 YYYYMMDD（信号用当日闭市数据，PIT 安全）
        db_path: 数据库路径（None=读 config 默认）
        params: 覆盖 DOWNSIDE_PARAMS 的子集
    Returns:
        set of banned ts_code
    """
    if db_path is None:
        db_path = _db_path_fallback()
    p = dict(DOWNSIDE_PARAMS)
    if params:
        p.update(params)
    need = max(p["n_ma"], p["n_down"]) + 2
    banned = set()
    for tc in codes:
        w = _fetch_window(tc, trade_date, db_path, n=need)
        if w is None or w["af"].isna().all():
            continue  # 无数据/无复权 -> 保留
        hfq = w["hfq"].fillna(w["c"]).values  # adj_factor 缺则退回不复权（极端兜底）
        vol = w["v"].fillna(0).values
        if _is_banned(hfq, vol, p):
            banned.add(tc)
    return banned
