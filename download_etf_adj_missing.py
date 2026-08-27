# -*- coding: utf-8 -*-
"""
补数脚本：下载缺失的 ETF 复权因子进本地库（etf_adj_factor）。
仅用于修复数据缺口，不修改任何既有策略/下载脚本。

复用 download_etf_adj.py 已验证的限速/重试/写表逻辑（import 复用，不重复造轮子）。
默认清单 = 当前 etf_daily 存在但 etf_adj_factor 缺失的 5 个 ETF：
    159766.SZ（旅游ETF）/ 510880.SH（红利ETF）/ 511990.SH（货币ETF·华宝添益）
    515050.SH（5G通信ETF）/ 518880.SH（黄金ETF）

说明：
  - 511990.SH 是货币ETF，价格恒≈100、无分红拆分，Tushare fund_adj 可能无数据；
    若拉取为空属正常现象，其 adj_factor 本质=1.0，可在下游用 fillna(1.0) 兜底（非绕过真实数据）。
  - 其余 4 个为真实权益/商品 ETF，应能从 fund_adj 补全。
"""
import sys
import os
import sqlite3
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA
import tushare as ts
import download_etf_adj as D   # 复用限速/重试/写表函数

DB = DATA["local_db_path"]

# 默认补数清单：etf_daily 有、etf_adj_factor 缺的 5 个
DEFAULT_MISSING = [
    "159766.SZ", "510880.SH", "511990.SH", "515050.SH", "518880.SH",
]


def main():
    codes = sys.argv[1].split(",") if len(sys.argv) > 1 else DEFAULT_MISSING
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
    for code in codes:
        code = code.strip()
        total, skipped, empties = 0, 0, 0
        for y in years:
            s = f"{y}0101"
            e = f"{y}1231"
            if D._year_covered(con, code, s, e):
                skipped += 1
                continue
            df = D._call_with_retry(pro, code, s, e)
            if df is not None and not df.empty:
                rows = [
                    (r.ts_code, int(r.trade_date), float(r.adj_factor))
                    for r in df.itertuples()
                ]
                con.executemany(
                    "INSERT OR REPLACE INTO etf_adj_factor VALUES (?, ?, ?)", rows
                )
                total += len(rows)
            else:
                empties += 1
        con.commit()
        print(f"✅ {code}: 新增/更新 {total} 行 | 跳过已完整年 {skipped} | 空年(无数据) {empties}")
    con.close()
    print("ALL_DONE")


if __name__ == "__main__":
    main()
