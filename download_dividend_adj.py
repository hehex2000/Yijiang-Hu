# -*- coding: utf-8 -*-
"""
定向补下载 4 只红利类 ETF 的复权因子（Tushare fund_adj）进 etf_adj_factor。

背景：run_dca_etf.py 用 etf_daily(不复权) + etf_adj_factor 算前复权价。
原 download_etf_adj.py 只下载了 12 只宽基，4 只红利 ETF(510880/512890/515080/515100)
缺 adj_factor → fillna(1.0) → 视为不复权 → 现金分红完全不计入 → 高股息品种回报被严重低估。

本脚本只补这 4 只，写入 etf_adj_factor(ts_code, trade_date, adj_factor)，幂等(INSERT OR REPLACE)。
"""
import sys
import os
import time
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA
import tushare as ts

DB = "D:/tu-shareData/astock_daily.db"
TARGETS = ["510880.SH", "512890.SH", "515080.SH", "515100.SH"]


def main():
    ts.set_token(DATA["tushare_token"])
    pro = ts.pro_api()
    con = sqlite3.connect(DB)
    con.execute(
        """CREATE TABLE IF NOT EXISTS etf_adj_factor (
            ts_code     TEXT,
            trade_date  INTEGER,
            adj_factor  REAL,
            PRIMARY KEY (ts_code, trade_date)
        )"""
    )
    con.commit()

    for code in TARGETS:
        total = 0
        years = list(range(2010, 2027))
        for y in years:
            s, e = f"{y}0101", f"{y}1231"
            try:
                df = pro.fund_adj(ts_code=code, start_date=s, end_date=e)
            except Exception as ex:
                print(f"  [WARN] {code} {y} fund_adj 失败: {ex}")
                time.sleep(0.5)
                continue
            if df is not None and not df.empty:
                rows = [
                    (r.ts_code, int(r.trade_date), float(r.adj_factor))
                    for r in df.itertuples()
                ]
                con.executemany(
                    "INSERT OR REPLACE INTO etf_adj_factor VALUES (?, ?, ?)", rows
                )
                total += len(rows)
            time.sleep(0.2)
        con.commit()
        # 校验
        n = con.execute(
            "SELECT COUNT(*) FROM etf_adj_factor WHERE ts_code=?", (code,)
        ).fetchone()[0]
        last = con.execute(
            "SELECT adj_factor FROM etf_adj_factor WHERE ts_code=? ORDER BY trade_date DESC LIMIT 1",
            (code,),
        ).fetchone()[0]
        print(f"✅ {code}: 本次写入 {total} 行, 库内共 {n} 行, 最新 adj_factor={last:.4f}")

    con.close()
    print("ALL_DONE")


if __name__ == "__main__":
    main()
