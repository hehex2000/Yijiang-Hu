# -*- coding: utf-8 -*-
"""
Backfill fina_indicator (annual) for Piotroski F-score, from a given start year.

Why this script exists
---------------------
piotroski_fscore.py reads ONLY these columns from fina_indicator:
    roa, ocfps, eps, debt_to_assets, current_ratio, gross_margin, asset_turn
plus balance_sheet.total_share (item 7). It does NOT read cashflow/income.

The DB has those columns for 2016+ but they are extremely sparse for 2008-2015
(2009 annual: ~200 rows; 2011/2012: ~20 rows). With missing t-1 data the
year-over-year F-score items all score 0 -> score collapses to 0 -> GATE8
selects nothing -> flat equity curve in early years.

This script fills exactly the 7 columns above (ann_date + end_date) via UPSERT,
leaving every other column in fina_indicator intact. Idempotent: re-run only
overwrites the same (ts_code, end_date) rows.

Usage
-----
    python download_fscore_financials.py                 # all stocks, 2008-2025
    python download_fscore_financials.py --start-year 2008
    python download_fscore_financials.py --pool 000906.SH   # zz800 only (cheaper)
    python download_fscore_financials.py --pool 000906.SH --start-year 2008 --end-year 2015

Notes
-----
* --pool limits to constituents of an index (reads index_constituent). For the
  zz800 backtest use --pool 000906.SH; it is ~7x cheaper than all stocks.
* start-year 2008 (not 2010): F-score at a 2010 selection needs 2009 (current)
  AND 2008 (prior) annuals, so 2008 must exist too.
* Tushare point cost: fina_indicator is a paid endpoint. All-stocks ~5500 calls;
  --pool 000906.SH ~800 calls. Watch your quota.
* Needs network + a valid tushare_token in config.py (same source as the other
  download scripts). Do NOT run in a sandbox without token/network.
"""

import sys
import os
import time
import sqlite3
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from config import DATA
    TOKEN = DATA.get("tushare_token", "")
    DB_PATH = DATA.get("local_db_path", "D:/tu-shareData/astock_daily.db")
except Exception:
    TOKEN = ""
    DB_PATH = "D:/tu-shareData/astock_daily.db"

if not TOKEN:
    print("[ERR] 无法读取 tushare_token，请检查 config.py")
    sys.exit(1)

# Columns piotroski_fscore.py actually consumes (fina_indicator).
FIELDS = "ts_code,ann_date,end_date,roa,ocfps,eps,debt_to_assets,current_ratio,gross_margin,asset_turn"

API_INTERVAL = 0.25  # seconds between calls (paid users ~200/min)


def get_codes(conn, pool):
    if pool:
        rows = conn.execute(
            "SELECT DISTINCT ts_code FROM index_constituent WHERE index_code = ?",
            (pool,),
        ).fetchall()
        codes = [r[0] for r in rows]
        print(f"[pool {pool}] {len(codes)} constituents from index_constituent")
        return codes
    rows = conn.execute("SELECT ts_code FROM stock_basic ORDER BY ts_code").fetchall()
    codes = [r[0] for r in rows]
    print(f"[all] {len(codes)} stocks from stock_basic")
    return codes


def upsert(conn, rows):
    """Insert or update only the F-score columns; never touch other columns."""
    sql = """
        INSERT INTO fina_indicator
            (ts_code, ann_date, end_date, roa, ocfps, eps,
             debt_to_assets, current_ratio, gross_margin, asset_turn)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ts_code, end_date) DO UPDATE SET
            ann_date       = excluded.ann_date,
            roa            = excluded.roa,
            ocfps          = excluded.ocfps,
            eps            = excluded.eps,
            debt_to_assets = excluded.debt_to_assets,
            current_ratio  = excluded.current_ratio,
            gross_margin   = excluded.gross_margin,
            asset_turn     = excluded.asset_turn
    """
    conn.executemany(sql, rows)


def main():
    ap = argparse.ArgumentParser(description="Backfill fina_indicator for F-score")
    ap.add_argument("--start-year", type=int, default=2008)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument("--pool", default="",
                    help="limit to index constituents, e.g. 000906.SH (zz800)")
    ap.add_argument("--api-interval", type=float, default=API_INTERVAL)
    args = ap.parse_args()

    import tushare as ts
    ts.set_token(TOKEN)
    pro = ts.pro_api()

    conn = sqlite3.connect(DB_PATH)
    codes = get_codes(conn, args.pool)
    if not codes:
        print("[ERR] no codes to download")
        conn.close()
        return

    start_d = f"{args.start_year}0101"
    end_d = f"{args.end_year}1231"
    print(f"range: {start_d} .. {end_d}  fields: {FIELDS}")

    total_rows = 0
    fail = 0
    t0 = datetime.now()
    for i, code in enumerate(codes):
        if i > 0 and i % 50 == 0:
            el = (datetime.now() - t0).total_seconds()
            print(f"  progress {i}/{len(codes)}  rows={total_rows}  fail={fail}  {el:.0f}s")

        df = None
        for attempt in range(4):
            try:
                df = pro.fina_indicator(
                    ts_code=code, start_date=start_d, end_date=end_d, fields=FIELDS
                )
                break
            except Exception as e:
                msg = str(e)
                if "频率" in msg or "rate" in msg.lower() or "流量" in msg:
                    print(f"  [rate-limit] {code} wait 60s (attempt {attempt+1})")
                    time.sleep(60)
                else:
                    print(f"  [ERR] {code}: {msg[:120]}")
                    df = None
                    break
        if df is None or len(df) == 0:
            fail += 1
            continue

        rows = []
        for _, r in df.iterrows():
            rows.append((
                r["ts_code"], r.get("ann_date"), r["end_date"],
                r.get("roa"), r.get("ocfps"), r.get("eps"),
                r.get("debt_to_assets"), r.get("current_ratio"),
                r.get("gross_margin"), r.get("asset_turn"),
            ))
        upsert(conn, rows)
        conn.commit()
        total_rows += len(rows)
        time.sleep(args.api_interval)

    conn.close()
    print(f"\nDone. total_rows={total_rows} fail={fail}  elapsed={(datetime.now()-t0).total_seconds():.0f}s")


if __name__ == "__main__":
    main()
