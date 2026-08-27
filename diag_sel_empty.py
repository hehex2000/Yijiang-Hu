"""诊断 run_backtest 选股为何在 20100101 返回 0 只。
复刻 run_selection 的核心流水线，逐阶段打印长度 + NaN 统计，并用较晚起始日对照。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sqlite3
import pandas as pd
import config as C
from src.data_fetcher import DataFetcher
from src.factor_calculator import FactorCalculator
from src.factor_processor import FactorProcessor
from src.stock_selector import StockSelector

DB_PATH = C.DATA["local_db_path"]

def ts_code(code):
    return code + (".SH" if code.startswith("6") else ".SZ")

def get_prev_day(d):
    return d  # 简化：与 run_backtest 在首日的表现一致

def get_zz800(date_fmt):
    conn = sqlite3.connect(DB_PATH)
    mx = conn.execute("SELECT MAX(trade_date) FROM index_constituent WHERE index_code='000906.SH' AND REPLACE(trade_date,'-','')<=?", (date_fmt,)).fetchone()[0]
    df = pd.read_sql("SELECT ts_code AS code FROM index_constituent WHERE index_code='000906.SH' AND trade_date=?", conn, params=(mx,))
    conn.close()
    return df

def prefilter_listing(pool, sel_date_fmt):
    conn = sqlite3.connect(DB_PATH)
    out = []
    for _, row in pool.iterrows():
        r = conn.execute("SELECT list_date FROM stock_basic WHERE ts_code=?", (ts_code(row['code']),)).fetchone()
        if r is None:
            out.append(row); continue
        ld = r[0].replace("-", "") if "-" in r[0] else r[0]
        if ld <= sel_date_fmt:
            out.append(row)
    conn.close()
    return pd.DataFrame(out).reset_index(drop=True)

def run_one(start_date):
    print("="*70)
    print(f"起始日 {start_date}")
    sel_date = get_prev_day(start_date)
    sel_fmt = sel_date
    pool = get_zz800(sel_fmt)
    print(f"  [pool] zz800 成分(<= {sel_fmt}): {len(pool)} 只")
    pool = prefilter_listing(pool, sel_fmt)
    print(f"  [pool] 上市日过滤后: {len(pool)} 只")
    if pool.empty:
        print("  -> pool 空，无法继续"); return

    fetcher = DataFetcher(primary_source=C.DATA["primary_source"], tushare_token=C.DATA.get("tushare_token",""),
                          local_db_path=DB_PATH, use_akshare_backup=C.DATA["use_akshare_backup"], use_tushare_backup=False)
    calc = FactorCalculator(**C.FACTOR_CALCULATOR)
    proc = FactorProcessor(config=C.FACTOR_PROCESSOR)
    cand = max(C.SELECTION["top_n"]*2, C.SELECTION["top_n"]+10)
    sel = StockSelector(config={"top_n": cand})

    factors = calc.calculate_all_factors(pool["code"].tolist(), fetcher, start_date=None, end_date=sel_date, max_workers=5)
    print(f"  [factors] {len(factors)} 只 x {len([c for c in factors.columns if c[:2] in ('VF','GF','QF','MF','TF','LV','MW')])} 因子")
    if not factors.empty:
        fac_cols = [c for c in factors.columns if c[:2] in ('VF','GF','QF','MF','TF','LV','MW')]
        allna = factors[fac_cols].isna().all(axis=1).sum()
        print(f"  [factors] 全部因子为NaN的股票: {allna} 只 / {len(factors)}")
        print(f"  [factors] current_price NaN: {factors['current_price'].isna().sum() if 'current_price' in factors.columns else 'N/A'}")

    processed = proc.process(factors)
    print(f"  [processed] {len(processed)} 只")
    if not processed.empty and "total_score" in processed.columns:
        print(f"  [processed] total_score NaN: {processed['total_score'].isna().sum()}")

    selected = sel.select(processed, top_n=cand)
    print(f"  [selected] TOP {cand}: {len(selected)} 只")
    if not selected.empty:
        print("    前3:", selected[["code","total_score"]].head(3).to_dict("records"))

if __name__ == "__main__":
    for sd in ["20100101", "20140101", "20150101", "20180101"]:
        run_one(sd)
