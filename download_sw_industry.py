"""下载 31 个申万一级行业指数全历史日线 → sw_industry_daily 表。

数据源：akshare (sw_index_first_info 取清单, index_hist_sw 取日线)
说明：
- 申万一级行业分类自发布起固定 31 个，无幸存者偏差（分类本身稳定，不随热门主题增减）
- 指数日线自 ~2000 年起，覆盖完整历史，回测时只用调仓日已存在的数据即可
- 表结构与 index_daily 对齐 (ts_code,trade_date,open,high,low,close,vol,amount)，trade_date 为 int YYYYMMDD
"""
import sqlite3
import time
import sys
import akshare as ak
import config

DB = config.DATA["local_db_path"]
COLS = ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"]


def ensure_table(conn):
    conn.execute(
        """CREATE TABLE IF NOT EXISTS sw_industry_daily (
            ts_code TEXT, trade_date INTEGER,
            open REAL, high REAL, low REAL, close REAL,
            vol REAL, amount REAL,
            PRIMARY KEY (ts_code, trade_date))"""
    )
    conn.commit()


def to_intdate(s):
    # akshare 日期形如 '2026-08-21' 或 '20260821'
    s = str(s).replace("-", "").replace("/", "")
    return int(s[:8])


def download_one(conn, code_si):
    sym = code_si.replace(".SI", "")  # index_hist_sw 用 801010
    df = ak.index_hist_sw(symbol=sym)
    if df is None or df.empty:
        return 0
    rows = []
    for _, r in df.iterrows():
        d = to_intdate(r["日期"])
        rows.append((
            code_si, d,
            float(r["开盘"]), float(r["最高"]), float(r["最低"]), float(r["收盘"]),
            float(r.get("成交量", 0) or 0), float(r.get("成交额", 0) or 0),
        ))
    conn.executemany(
        "INSERT OR REPLACE INTO sw_industry_daily "
        "(ts_code,trade_date,open,high,low,close,vol,amount) VALUES (?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    return len(rows)


def main():
    conn = sqlite3.connect(DB)
    ensure_table(conn)
    info = ak.sw_index_first_info()
    codes = info["行业代码"].tolist()  # 801010.SI ...
    print(f"[清单] 申万一级行业 {len(codes)} 个")
    total = 0
    for i, c in enumerate(codes):
        try:
            n = download_one(conn, c)
            total += n
            print(f"  [{i+1}/{len(codes)}] {c} {n} 行")
        except Exception as e:
            print(f"  [{i+1}/{len(codes)}] {c} FAIL {type(e).__name__}: {str(e)[:120]}")
        time.sleep(0.3)  # 轻量限速，避免 akshare 限流
    conn.close()
    print(f"[完成] 累计写入 {total} 行 → sw_industry_daily")


if __name__ == "__main__":
    main()
