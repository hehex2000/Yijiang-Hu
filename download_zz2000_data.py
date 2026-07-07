# -*- coding: utf-8 -*-
"""
回补 中证2000(932000.CSI) 的日行情 + 历史成分股时点快照
=========================================================
重要：Tushare 上中证2000 的真实代码是 **932000.CSI**（中证指数公司，后缀 .CSI），
并不是 .SH。本地库的 index_daily / index_constituent 统一用 .SH 体系存储，
故本脚本拉取后改写 index_code/ts_code 为 932000.SH 再写入，与库内其它指数(000300.SH等)保持一致。

用法:
    python download_zz2000_data.py                      # 全量回补
    python download_zz2000_data.py --start 20130101     # 指定起始

依赖: tushare(1.4.29), config.DATA.tushare_token
"""
import sys, os, argparse, sqlite3, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import tushare as ts
from config import DATA

DB_PATH = DATA.get("local_db_path", r"D:\tu-shareData\astock_daily.db")
TOKEN = DATA.get("tushare_token", "")

TUSHARE_CODE = "932000.CSI"      # Tushare 真实代码
LOCAL_CODE = "932000.SH"         # 本地库存储代码（.SH 体系，与 000300.SH 等一致）
INDEX_NAME = "中证2000"


def get_pro():
    if not TOKEN:
        print("[错误] config.DATA.tushare_token 为空，无法调用 Tushare")
        sys.exit(1)
    ts.set_token(TOKEN)
    return ts.pro_api()


def download_daily(pro, start, end):
    """按年分块拉 index_daily，改写 ts_code 后写入 index_daily 表（INSERT OR REPLACE 幂等）。"""
    print(f"\n[1/2] 下载 index_daily ({TUSHARE_CODE} -> {LOCAL_CODE}) {start}~{end}")
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


def download_constituent(pro, start, end):
    """按年分块拉 index_weight（天然时点快照），改写 index_code 后写入 index_constituent。"""
    print(f"\n[2/2] 下载 index_constituent 时点快照 ({TUSHARE_CODE} -> {LOCAL_CODE}) {start}~{end}")
    sy, ey = int(start[:4]), int(end[:4])
    rows = []
    for y in range(sy, ey + 1):
        s, e = f"{y}0101", f"{y}1231"
        if e > end:
            e = end
        try:
            df = pro.index_weight(index_code=TUSHARE_CODE, start_date=s, end_date=e)
        except Exception as ex:
            print(f"    [constituent 跳过] {y}: {ex}")
            continue
        if df is None or df.empty:
            continue
        for _, r in df.iterrows():
            td = str(r["trade_date"])
            code = str(r["con_code"])
            w = float(r["weight"]) if pd.notna(r.get("weight")) else None
            rows.append((LOCAL_CODE, code, td, w, INDEX_NAME))
        time.sleep(0.3)
    if not rows:
        print("    [空] 无成分股快照")
        return 0
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM index_constituent WHERE index_code=?", (LOCAL_CODE,))
    conn.executemany(
        "INSERT INTO index_constituent (index_code, ts_code, trade_date, weight, index_name) "
        "VALUES (?,?,?,?,?)",
        rows,
    )
    conn.commit()
    nsnap = conn.execute(
        "SELECT COUNT(DISTINCT trade_date) FROM index_constituent WHERE index_code=?",
        (LOCAL_CODE,),
    ).fetchone()[0]
    conn.close()
    print(f"    [完成] 写入 index_constituent {len(rows)} 行, 时点快照数={nsnap}")
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description="回补 中证2000 数据(日行情+成分股)")
    ap.add_argument("--start", default="20190101", help="起始日期 YYYYMMDD")
    ap.add_argument("--end", default="20260706", help="结束日期 YYYYMMDD")
    args = ap.parse_args()
    pro = get_pro()
    download_daily(pro, args.start, args.end)
    download_constituent(pro, args.start, args.end)
    print("\n全部完成。接下来在 get_stock_pool_index 与选股器 pool 映射加 "
          "\"zz2000\": \"932000.SH\" 即可使用。")


if __name__ == "__main__":
    main()
