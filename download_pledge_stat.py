# -*- coding: utf-8 -*-
"""
下载股权质押统计 pledge_stat 到本地库（补全 ④ 自利性分红护栏的质押数据）。

数据现实（探针确认）：
  - tushare pledge_stat 接口只能按 ts_code 逐只拉，不能按 end_date 批量全市场拉。
  - 字段仅 ts_code/end_date/pledge_count/unrest_pledge/rest_pledge/total_share/pledge_ratio
    （pledge_ratio = 累计质押股数 / 总股本，整体质押比例，业界通用"高质押"代理）。
  - 不含"控股股东质押占其持股比"，该精度需 pledge_detail（更重，暂不取）。

用法：
  python download_pledge_stat.py            # 全市场（断点续传，跳过已存在）
  python download_pledge_stat.py --limit 30 # 仅前 30 只（测试）
"""
import os
import sys
import time
import sqlite3
import argparse
import pandas as pd
import tushare as ts

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from config import DATA

DB_PATH = DATA["local_db_path"]
ts.set_token(DATA["tushare_token"])
pro = ts.pro_api()

COLS = ["ts_code", "end_date", "pledge_count", "unrest_pledge",
        "rest_pledge", "total_share", "pledge_ratio"]


def ensure_table():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS pledge_stat ("
        "ts_code TEXT, end_date TEXT, pledge_count REAL, unrest_pledge REAL, "
        "rest_pledge REAL, total_share REAL, pledge_ratio REAL, "
        "PRIMARY KEY(ts_code, end_date))")
    conn.commit()
    conn.close()


def get_stock_list():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    df = pd.read_sql_query("SELECT ts_code FROM stock_basic ORDER BY ts_code", conn)
    conn.close()
    return df["ts_code"].tolist()


def done_codes():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        df = pd.read_sql_query("SELECT DISTINCT ts_code FROM pledge_stat", conn)
        s = set(df["ts_code"].tolist())
    except Exception:
        s = set()
    conn.close()
    return s


def fetch_one(code, tries=3):
    for _ in range(tries):
        try:
            df = pro.pledge_stat(ts_code=code)
            if df is None or len(df) == 0:
                return None
            return df
        except Exception as e:
            msg = str(e)
            if "积分" in msg or "每分钟" in msg or "rate" in msg.lower():
                time.sleep(30)
            else:
                time.sleep(2)
    return None


def upsert(df):
    have = [c for c in COLS if c in df.columns]
    data = df[have]
    ph = ",".join(["?"] * len(COLS))
    col_list = ",".join(COLS)
    sql = f"INSERT OR REPLACE INTO pledge_stat ({col_list}) VALUES ({ph})"
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.executemany(sql, [tuple(r) for r in data.itertuples(index=False, name=None)])
    conn.commit()
    conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    ensure_table()
    all_codes = get_stock_list()
    done = done_codes()
    todo = [c for c in all_codes if c not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"待下载 {len(todo)} 只 (已存在 {len(done)})")
    n = 0
    for code in todo:
        df = fetch_one(code)
        if df is not None and len(df):
            upsert(df)
        n += 1
        if n % 50 == 0:
            print(f"  {n}/{len(todo)} done")
        time.sleep(0.3)
    print("完成")


if __name__ == "__main__":
    main()
