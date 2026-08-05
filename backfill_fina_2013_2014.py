# -*- coding: utf-8 -*-
"""
补下 fina_indicator 的 2013-12-31 与 2014-12-31 两期年报数据（全字段 108 列）。

背景：
  当前库 fina_indicator 各年报覆盖良好（2012≈3330、2015≈4230、2016≈4477、2017≈4800 只），
  但 2013/2014 两期仅 18/14 只——是价值选股②③④质量门槛的缺口来源
  （盈余质量 ocfps/eps、杠杆 debt_to_assets/ocf_to_debt、应收 ar_turn 全变 NaN）。

  注：Tushare fina_indicator 必须带 ts_code，无法整期批量拉取。
  本脚本对每只股票【一次调用】拉取 2013~2014 区间（含两期年报），
  再按 end_date 筛出目标两期写入，约 5500 次调用完成。

用法：
  python backfill_fina_2013_2014.py
（token 从 config_tushare / config.DATA 自动读取；tushare 已装）
"""

import os
import sys
import time
import sqlite3
import datetime
import pandas as pd
import tushare as ts

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DB_PATH = "D:/tu-shareData/astock_daily.db"
PERIODS = ["20131231", "20141231"]
SLEEP = 0.05  # 调用间隔（秒）；付费token限速宽松

# ── 读取 token ──────────────────────────────────────────────
_TOKEN = ""
try:
    import config_tushare as ct
    _TOKEN = ct.TUSHARE_TOKEN
except Exception:
    pass
if not _TOKEN:
    try:
        from config import DATA
        _TOKEN = DATA.get("tushare_token", "")
    except Exception:
        pass
if not _TOKEN:
    print("[ERR] 无法读取 tushare_token，请检查 config_tushare.py / config.py")
    sys.exit(1)

ts.set_token(_TOKEN)
pro = ts.pro_api()


def get_table_cols(db_path):
    conn = sqlite3.connect(db_path)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(fina_indicator)").fetchall()]
    conn.close()
    return cols


def get_stock_list(db_path):
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT ts_code FROM stock_basic ORDER BY ts_code", conn)
        if len(df) > 0:
            conn.close()
            return df["ts_code"].tolist()
    except Exception:
        pass
    df = pd.read_sql_query(
        "SELECT DISTINCT ts_code FROM daily WHERE trade_date >= '20120101'", conn)
    conn.close()
    return df["ts_code"].tolist()


def upsert(db_path, cols, df):
    """按表列对齐写入（INSERT OR REPLACE）。"""
    ordered = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    data = df[ordered].copy()
    for c in missing:
        data[c] = None
    data = data[cols]
    placeholders = ",".join(["?"] * len(cols))
    col_list = ",".join(cols)
    sql = f"INSERT OR REPLACE INTO fina_indicator ({col_list}) VALUES ({placeholders})"
    conn = sqlite3.connect(db_path)
    conn.executemany(sql, [tuple(r) for r in data.itertuples(index=False, name=None)])
    conn.commit()
    conn.close()


def mark_log(db_path, period):
    try:
        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT OR REPLACE INTO financial_download_log "
            "(data_type, period, status, retry_count, update_time) VALUES (?,?,?,?,?)",
            ("fina_indicator", period, "done", 0,
             datetime.datetime.now().strftime("%Y%m%d%H%M%S")),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"  [提示] 写入下载日志失败(可忽略): {e}")


def main():
    print("=" * 70)
    print("补下 fina_indicator 2013/2014 年报（全字段，逐只一次拉两期）")
    print("=" * 70)
    cols = get_table_cols(DB_PATH)
    print(f"  表列数: {len(cols)}")
    stocks = get_stock_list(DB_PATH)
    print(f"  股票列表: {len(stocks)} 只")

    total_written = 0
    ok = 0
    empty = 0
    fail = 0
    buf = []  # 累积待写入的 DataFrame

    def flush():
        nonlocal total_written, buf
        if buf:
            big = pd.concat(buf, ignore_index=True)
            big = big[big["end_date"].isin(PERIODS)]
            if len(big) > 0:
                upsert(DB_PATH, cols, big)
                total_written += len(big)
            buf = []

    for i, tc in enumerate(stocks):
        if i % 200 == 0:
            print(f"  进度 {i}/{len(stocks)} | 已写 {total_written} 行")
            flush()
        df = None
        try:
            df = pro.fina_indicator(
                ts_code=tc, start_date="20130101", end_date="20141231",
                fields=",".join(cols))
        except Exception as e:
            msg = str(e)
            if "超限" in msg or "rate" in msg.lower():
                print(f"  [限速] {tc} 等待60s...")
                time.sleep(60)
                try:
                    df = pro.fina_indicator(
                        ts_code=tc, start_date="20130101", end_date="20141231",
                        fields=",".join(cols))
                except Exception:
                    df = None
            else:
                df = None
        if df is not None and len(df) > 0:
            buf.append(df)
            ok += 1
        else:
            empty += 1
        time.sleep(SLEEP)

    flush()
    print(f"\n  拉取完成：成功 {ok} 只，空 {empty} 只，失败 {fail} 只，共写入 {total_written} 行(2013+2014)")

    for period in PERIODS:
        mark_log(DB_PATH, period)

    # 校验
    print("\n" + "=" * 70)
    print("校验：")
    conn = sqlite3.connect(DB_PATH)
    for period in PERIODS:
        n = conn.execute(
            "SELECT COUNT(DISTINCT ts_code) FROM fina_indicator WHERE end_date=?",
            (period,)).fetchone()[0]
        print(f"  {period}: {n} 只不同股票")
    conn.close()
    print("完成。")


if __name__ == "__main__":
    main()
