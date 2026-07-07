# -*- coding: utf-8 -*-
"""
下载 ETF 复权因子（Tushare fund_adj）进本地库，用于修复 etf_daily 未复权导致的拆分断点。

Tushare fund_adj 返回字段：ts_code, trade_date, adj_factor
  - adj_factor 为后复权因子（值随历史缩小，非"最新=1"）。
  - 使用前复权：复权价 = 未复权价 * adj_factor / adj_factor_latest，使最新交易日价格=真实可交易价，历史连续。

写入表：etf_adj_factor(ts_code, trade_date, adj_factor)，主键 (ts_code, trade_date)，INSERT OR REPLACE 幂等。
"""
import sys
import os
import time
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA
import tushare as ts

DB = "D:/tu-shareData/astock_daily.db"

# 网格菜单 + 指数ETF一览实际用到的 ETF
ETF_LIST = [
    "510300.SH",  # 沪深300ETF
    "510050.SH",  # 上证50ETF
    "515800.SH",  # 中证800ETF（汇添富）
    "510500.SH",  # 中证500ETF（南方）
    "512100.SH",  # 中证1000ETF（南方）
    "563300.SH",  # 中证2000ETF
    "510210.SH",  # 上证指数ETF
    "159903.SZ",  # 深成ETF
    "159915.SZ",  # 创业板ETF
    "159949.SZ",  # 创业板50ETF
    "588000.SH",  # 科创50ETF
    "588190.SH",  # 科创100ETF
]


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

    years = list(range(2000, 2027))
    for code in ETF_LIST:
        total = 0
        for y in years:
            s = f"{y}0101"
            e = f"{y}1231"
            try:
                df = pro.fund_adj(ts_code=code, start_date=s, end_date=e)
            except Exception as ex:
                print(f"  [WARN] {code} {y} fund_adj 调用失败: {ex}")
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
            time.sleep(0.2)  # 限流保护
        con.commit()
        print(f"✅ {code}: 累计写入 {total} 行")

    con.close()
    print("ALL_DONE")


if __name__ == "__main__":
    main()
