# -*- coding: utf-8 -*-
"""
回补 科创100(000698.SH) 的日行情到 index_daily
=========================================================
科创100 本地库的 index_daily 此前为空（仅 ETF 588190 有数据）。
本脚本用 Tushare index_daily 拉取并写入，使网格/指数功能可直接用指数序列。

用法:
    python download_kc100_data.py
    python download_kc100_data.py --start 20190101
依赖: tushare(1.4.29), config.DATA.tushare_token
"""
import sys, os, argparse, sqlite3, time
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import tushare as ts
from config import DATA

DB_PATH = DATA.get("local_db_path", r"D:\tu-shareData\astock_daily.db")
TOKEN = DATA.get("tushare_token", "")

TUSHARE_CODE = "000698.SH"      # Tushare 真实代码（即 .SH，无需转换）
LOCAL_CODE = "000698.SH"        # 本地库存储代码
INDEX_NAME = "科创100"


def get_pro():
    if not TOKEN:
        print("[错误] config.DATA.tushare_token 为空，无法调用 Tushare")
        sys.exit(1)
    ts.set_token(TOKEN)
    return ts.pro_api()


def download_daily(pro, start, end):
    """按年分块拉 index_daily，写入 index_daily 表（INSERT OR REPLACE 幂等）。"""
    print(f"\n[1/1] 下载 index_daily ({TUSHARE_CODE}) {start}~{end}")
    sy, ey = int(start[:4]), int(end[:4])
    frames = []
    for y in range(sy, ey + 1):
        s, e = f"{y}0101", f"{y}1231"
        if e > end:
            e = end
        try:
            df = pro.index_daily(ts_code=TUSHARE_CODE, start_date=s, end_date=e)
        except Exception as ex:
            print(f"    [daily 跳过] {y}: {ex}")
            continue
        if df is None or df.empty:
            continue
        frames.append(df)
        time.sleep(0.3)
    if not frames:
        print("    [空] 无日行情数据")
        return 0
    big = pd.concat(frames, ignore_index=True)
    big["ts_code"] = LOCAL_CODE
    big["trade_date"] = big["trade_date"].astype(int)  # 与现有 index_daily(INTEGER) 一致
    cols = ["ts_code", "trade_date", "close", "open", "high", "low",
            "pre_close", "change", "pct_chg", "vol", "amount"]
    big = big[cols]
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM index_daily WHERE ts_code=?", (LOCAL_CODE,))
    conn.executemany(
        "INSERT OR REPLACE INTO index_daily "
        "(ts_code,trade_date,close,open,high,low,pre_close,change,pct_chg,vol,amount) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        big.values.tolist(),
    )
    conn.commit()
    n = len(big)
    conn.close()
    print(f"    [完成] 写入 index_daily {n} 行 ({big['trade_date'].min()}~{big['trade_date'].max()})")
    return n


def main():
    ap = argparse.ArgumentParser(description="回补 科创100 日行情(index_daily)")
    ap.add_argument("--start", default="20190101", help="起始日期 YYYYMMDD")
    ap.add_argument("--end", default=datetime.now().strftime("%Y%m%d"), help="结束日期 YYYYMMDD(默认今天)")
    args = ap.parse_args()
    pro = get_pro()
    download_daily(pro, args.start, args.end)
    print("\n完成。科创100 现已可直接作为网格/指数标的（000698.SH）。")


if __name__ == "__main__":
    main()
