# -*- coding: utf-8 -*-
"""红利质量四道门禁(①+ocf豁免 / ②周期缓冲 / ③举债分红 / ④高质押) 递进 A/B 阶梯。
直接复用 select 的预筛选流程构建候选集，再逐道叠加门禁计数（截断前）。"""
import os, sys, io, contextlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_monthly_rebalance as M
import pandas as pd

DATES = ['20230331', '20240329', '20250328']

def build_candidates(trade_date):
    """复刻 select 内部 pre-gate 流程：指数池 + 估值过滤 + 排ST。"""
    conn = M.get_conn()
    actual_date = trade_date
    while True:
        cnt = pd.read_sql_query(
            "SELECT COUNT(*) AS n FROM daily_basic WHERE trade_date = ?",
            conn, params=(actual_date,)).iloc[0]['n']
        if cnt > 0:
            break
        prev = pd.read_sql_query(
            "SELECT MAX(trade_date) AS max_date FROM daily_basic WHERE trade_date < ?",
            conn, params=(actual_date,)).iloc[0, 0]
        if prev is None:
            conn.close(); return pd.DataFrame()
        actual_date = prev
    zz_set = M.get_index_constituents('000906.SH', trade_date=actual_date)
    df = pd.read_sql_query("""
        SELECT ts_code, dv_ttm, total_mv
        FROM daily_basic
        WHERE trade_date = ?
          AND pe_ttm > 0 AND pe_ttm < 50
          AND pb > 0 AND pb < 10
          AND dv_ttm > 0 AND total_mv > 0
    """, conn, params=(actual_date,))
    conn.close()
    if df.empty: return df
    df = df[df['ts_code'].isin(zz_set)]
    conn = M.get_conn()
    st = pd.read_sql_query("SELECT ts_code FROM stock_basic WHERE name LIKE '%ST%' OR name LIKE '%*%'", conn)
    conn.close()
    if len(st): df = df[~df['ts_code'].isin(set(st['ts_code']))]
    return df

for d in DATES:
    df0 = build_candidates(d)
    n0 = len(df0)
    # ①+ocf豁免 (②/③/④ 关)
    a = M.apply_dividend_quality_filters(df0.copy(), d, div_years_min=3,
        require_ocf_cover=True, industry_exempt_ocf=True,
        div_years_min_cyclical=3, require_net_debt_check=False, require_low_pledge=False)
    # +② 周期缓冲
    b = M.apply_dividend_quality_filters(df0.copy(), d, div_years_min=3,
        require_ocf_cover=True, industry_exempt_ocf=True,
        div_years_min_cyclical=10, require_net_debt_check=False, require_low_pledge=False)
    # +③ 举债分红
    c = M.apply_dividend_quality_filters(df0.copy(), d, div_years_min=3,
        require_ocf_cover=True, industry_exempt_ocf=True,
        div_years_min_cyclical=10, require_net_debt_check=True,
        net_debt_ratio_jump=0.20, require_low_pledge=False)
    # +④ 高质押 (全开)
    e = M.apply_dividend_quality_filters(df0.copy(), d, div_years_min=3,
        require_ocf_cover=True, industry_exempt_ocf=True,
        div_years_min_cyclical=10, require_net_debt_check=True,
        net_debt_ratio_jump=0.20, require_low_pledge=True, pledge_ratio_max=30.0)
    print(f"=== {d}  预筛选候选 {n0} ===")
    print(f"  ①+ocf豁免        : {len(a):>4}  (剔除 {n0-len(a)})")
    print(f"  +②周期缓冲       : {len(b):>4}  (②额外 {len(a)-len(b)})")
    print(f"  +③举债分红       : {len(c):>4}  (③额外 {len(b)-len(c)})")
    print(f"  +④高质押(全开)   : {len(e):>4}  (④额外 {len(c)-len(e)})")
    print(f"  → 四门禁累计剔除 {n0-len(e)} 只 ({100*(n0-len(e))/n0:.1f}%)")
    print()
