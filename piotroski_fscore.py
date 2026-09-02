# -*- coding: utf-8 -*-
"""
Piotroski F-score 9 项打分（point-in-time，无前视）。

来源：plan_piotroski.md 第 2 步。F-score = 9 个 0/1 二元指标之和（0~9，越高越财务健康）。
经典 Piotroski (2000, JFE)。平台原有 quality_filter 仅 4 项，本模块补上经典 9 项作为
离散质量因子，供 run_piotroski_oos.py 做单因子 OOS 验证（不接入选股引擎，除非 OOS 通过）。

关键设计：
  * 严格 PIT：调仓日 t 只取 ann_date < t 的财报（盘后公告当日不可用），复用 run_kara_factors
    的 build_pit_map / pit_get 范式（此处复制，避免 import 触发该脚本重跑）。
  * 全部年报口径（end_date LIKE '%1231'），防季报 NULL / 老数据坑。
  * 同比项（3/5/6/7/8/9）：取当期 + 去年同期两期，两期都非空才判分，否则记 0（保守，不夸大）。
  * 第 7 项（无增发）：total_share 取自 balance_sheet（daily_basic 无此列），同比增幅 < 阈值判未稀释。
  * 金融股（银行/保险）roa 等常为空 -> 缺失项记 0，自然被排除（经典 F-score 本就不覆盖金融业）。

9 项口径（与 plan_piotroski.md §1 一致）：
  1 盈利  ROA>0                         fina_indicator.roa
  2 盈利  CFO>0  (ocfps>0 代理)         fina_indicator.ocfps
  3 盈利  ΔROA>0                        roa 两期同比
  4 质量  应计低：ocfps>eps             fina_indicator.ocfps vs eps（每股现金流>每股盈余）
  5 杠杆  长期负债率降：Δdebt_to_assets<0  fina_indicator.debt_to_assets 同比近似
  6 流动性 流动比率升：Δcurrent_ratio>0     fina_indicator.current_ratio 同比
  7 稀释  当年无增发：total_share 同比增幅<阈值  balance_sheet.total_share 同比
  8 运营  毛利率升：Δgross_margin>0      fina_indicator.gross_margin 同比
  9 运营  资产周转率升：Δasset_turn>0    fina_indicator.asset_turn 同比

用法：
  from piotroski_fscore import build_fscore_maps, compute_fscore
  con = sqlite3.connect(DB_PATH)
  M = build_fscore_maps(con)
  score, items = compute_fscore("600519.SH", "20260401", M)
"""

import bisect
import math
import sqlite3

import numpy as np
import pandas as pd
from pit_ann import norm_ann          # ann_date 规范化(fina_indicator.ann_date 是 REAL 浮点)

DILUTION_THRESHOLD = 0.05  # 第7项：total_share 同比增幅 < 5% 视为"未显著稀释"（可调）


# ---------- PIT 基础设施（复制自 run_kara_factors.py 的 build_pit_map/pit_get 范式，
#            改为接收 con 参数，避免 import 该脚本触发重跑）----------
def _build_pit_map(con, sql, valcol, denom=None):
    df = pd.read_sql(sql, con)
    df["ann"] = norm_ann(df["ann_date"])
    df[valcol] = pd.to_numeric(df[valcol], errors="coerce")
    if denom is not None:
        df[denom] = pd.to_numeric(df[denom], errors="coerce")
        df["v"] = df[valcol] / df[denom]
    else:
        df["v"] = df[valcol]
    df = df.dropna(subset=["v", "ann"])
    df = df[df["v"] == df["v"]]
    df = df.sort_values(["ts_code", "ann"]).drop_duplicates(["ts_code", "ann"], keep="last")
    out = {}
    for code, g in df.groupby("ts_code"):
        out[code] = (g["ann"].values, g["v"].values)
    return out


def _pit_get(m, code, t):
    """严格 ann_date < t 取最近一期值；无则返回 nan。"""
    if code not in m:
        return np.nan
    anns, vals = m[code]
    i = bisect.bisect_left(anns, t) - 1
    return vals[i] if i >= 0 else np.nan


def _cur_prev(m, code, t):
    """返回 (当期值, 去年同期值)，均用 ann_date<t 严格 PIT。
    当期 = 最近一期；去年同期 = 再往前一期。任一缺失返回 nan。"""
    if code not in m:
        return np.nan, np.nan
    anns, vals = m[code]
    i = bisect.bisect_left(anns, t) - 1
    if i < 0:
        return np.nan, np.nan
    cur = vals[i]
    prev = vals[i - 1] if i - 1 >= 0 else np.nan
    return cur, prev


# ---------- 建图 ----------
def build_fscore_maps(con):
    """构建 F-score 9 项所需的全部 PIT map（年报口径）。返回 dict[name]->pit_map。"""
    YEAR = " AND end_date LIKE '%1231'"
    fin_specs = {
        "roa": "roa",
        "ocfps": "ocfps",
        "eps": "eps",
        "debt_to_assets": "debt_to_assets",
        "current_ratio": "current_ratio",
        "gross_margin": "gross_margin",
        "asset_turn": "asset_turn",
    }
    maps = {}
    for name, col in fin_specs.items():
        sql = (f"SELECT ts_code, ann_date, {col} FROM fina_indicator "
               f"WHERE {col} IS NOT NULL{YEAR}")
        maps[name] = _build_pit_map(con, sql, col)
    # 第7项：balance_sheet.total_share（daily_basic 无此列）
    sql = (f"SELECT ts_code, ann_date, total_share FROM balance_sheet "
           f"WHERE total_share IS NOT NULL{YEAR}")
    maps["total_share"] = _build_pit_map(con, sql, "total_share")
    return maps


# ---------- 9 项打分 ----------
def _gt0(v):
    return v == v and v > 0


def _fall(cur, prev):
    return cur == cur and prev == prev and (cur - prev) < 0


def _rise_ok(cur, prev):
    return cur == cur and prev == prev and (cur - prev) > 0


def compute_fscore(code, t, M):
    """计算单只股票在调仓日 t 的 F-score。

    参数:
      code: ts_code（如 '600519.SH'）
      t:    调仓日字符串（YYYYMMDD），严格 PIT（ann_date < t）
      M:    build_fscore_maps 返回的 map 字典
    返回:
      (score: int 0~9, items: dict{1..9: 0/1})
    """
    roa_c, roa_p = _cur_prev(M["roa"], code, t)
    ocf_c, _ = _cur_prev(M["ocfps"], code, t)
    eps_c, _ = _cur_prev(M["eps"], code, t)
    debt_c, debt_p = _cur_prev(M["debt_to_assets"], code, t)
    cr_c, cr_p = _cur_prev(M["current_ratio"], code, t)
    gm_c, gm_p = _cur_prev(M["gross_margin"], code, t)
    at_c, at_p = _cur_prev(M["asset_turn"], code, t)
    ts_c, ts_p = _cur_prev(M["total_share"], code, t)

    items = {}
    # 1 盈利 ROA>0
    items[1] = 1 if _gt0(roa_c) else 0
    # 2 盈利 CFO>0（ocfps 代理）
    items[2] = 1 if _gt0(ocf_c) else 0
    # 3 盈利 ΔROA>0
    items[3] = 1 if _rise_ok(roa_c, roa_p) else 0
    # 4 质量 应计低：每股现金流 > 每股盈余（ocfps>eps）
    items[4] = 1 if (ocf_c == ocf_c and eps_c == eps_c and ocf_c > eps_c) else 0
    # 5 杠杆 负债率降：Δdebt_to_assets<0
    items[5] = 1 if _fall(debt_c, debt_p) else 0
    # 6 流动性 流动比率升：Δcurrent_ratio>0
    items[6] = 1 if _rise_ok(cr_c, cr_p) else 0
    # 7 稀释 无增发：total_share 同比增幅 < 阈值（buyback 或微增仍算未稀释）
    if ts_c == ts_c and ts_p == ts_p and ts_p > 0:
        items[7] = 1 if (ts_c / ts_p - 1) < DILUTION_THRESHOLD else 0
    else:
        items[7] = 0
    # 8 运营 毛利率升：Δgross_margin>0
    items[8] = 1 if _rise_ok(gm_c, gm_p) else 0
    # 9 运营 资产周转率升：Δasset_turn>0
    items[9] = 1 if _rise_ok(at_c, at_p) else 0

    score = sum(items.values())
    return score, items


# ---------- 自测：抽样本股手工核对 ----------
def _self_test():
    try:
        from config import DATA
        db = DATA.get("local_db_path", "D:/tu-shareData/astock_daily.db")
    except Exception:
        db = "D:/tu-shareData/astock_daily.db"
    con = sqlite3.connect(db)
    M = build_fscore_maps(con)
    print("各 map 覆盖股票数：")
    for k, v in M.items():
        print(f"  {k:14s} {len(v)} 只")

    t = "20260401"  # 可看到 2025 年报（ann_date 多在 2026-03~04）
    samples = ["600519.SH", "300750.SZ", "000001.SZ", "000333.SZ", "601318.SH", "000651.SZ"]
    print(f"\n{t} 调仓日 F-score（年报口径，ann_date<t 严格 PIT）：")
    for c in samples:
        score, items = compute_fscore(c, t, M)
        detail = " ".join(f"{k}:{v}" for k, v in sorted(items.items()))
        print(f"  {c}: F={score}  [{detail}]")
    con.close()


if __name__ == "__main__":
    _self_test()
