# -*- coding: utf-8 -*-
"""增量补 中证2000(932000.CSI -> 本地 932000.SH) 的 2026 年日行情。
不 DELETE 历史：仅 INSERT OR REPLACE 2026 区间，保留 2014~2025 既有数据。
依赖: tushare, config.DATA.tushare_token
"""
import sys, os, sqlite3
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tushare as ts
from config import DATA

DB_PATH = DATA.get("local_db_path", r"D:\tu-shareData\astock_daily.db")
TOKEN = DATA.get("tushare_token", "")
TUSHARE_CODE = "932000.CSI"
LOCAL_CODE = "932000.SH"
START, END = "20260101", "20260713"

def main():
    if not TOKEN:
        print("[错误] tushare_token 为空"); sys.exit(1)
    ts.set_token(TOKEN)
    pro = ts.pro_api()

    print(f"拉取 {TUSHARE_CODE} 日行情 {START}~{END} ...")
    df = pro.index_daily(ts_code=TUSHARE_CODE, start_date=START, end_date=END)
    if df is None or df.empty:
        print("[空] 无数据返回（检查 token 权限 / 网络）"); sys.exit(1)

    df["ts_code"] = LOCAL_CODE
    df["trade_date"] = df["trade_date"].astype(int)
    cols = ["ts_code", "trade_date", "close", "open", "high", "low",
            "pre_close", "change", "pct_chg", "vol", "amount"]
    df = df[cols]

    conn = sqlite3.connect(DB_PATH)
    cur_max = conn.execute("SELECT MAX(trade_date) FROM index_daily WHERE ts_code=?",
                           (LOCAL_CODE,)).fetchone()[0]
    conn.executemany(
        "INSERT OR REPLACE INTO index_daily "
        "(ts_code,trade_date,close,open,high,low,pre_close,change,pct_chg,vol,amount) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        df.values.tolist(),
    )
    conn.commit()
    new_max = conn.execute("SELECT MAX(trade_date) FROM index_daily WHERE ts_code=?",
                           (LOCAL_CODE,)).fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM index_daily WHERE ts_code=?",
                         (LOCAL_CODE,)).fetchone()[0]
    # 校验 2026 区间行数
    n26 = conn.execute("SELECT COUNT(*) FROM index_daily WHERE ts_code=? AND trade_date>=?",
                       (LOCAL_CODE, 20260101)).fetchone()[0]
    conn.close()
    print(f"  写入 {len(df)} 行（2026）。原最大日期={cur_max}，新最大日期={new_max}，"
          f"2026年内行数={n26}，全表总行数={total}")
    if new_max and int(new_max) >= 20260701:
        print("  [OK] 932000.SH 已补齐至 2026-07，菜单[12]可正常回测。")
    else:
        print("  [WARN] 2026 数据仍不完整，请检查。")

if __name__ == "__main__":
    main()
