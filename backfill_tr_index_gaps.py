#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
补全 index_tr_official 中【官方全收益/净收益指数】的缺少年份
==============================================================
问题背景：download_tr_index.py 按年下载时, 个别年份因 Tushare 限速/临时失败被
         `continue` 静默跳过, 留下整年缺口。诊断结果(2010+ 异常缺口)：
           H00906.CSI   (中证800全收益)       缺 2021/2024/2025
           H20955.CSI   (红利低波100全收益)    缺 2023
           000922CNY020.CSI (中证红利净收益)   缺 2022/2024
         (2002-2009 缺年为下载起点2010所致, 非异常, 不补)

本脚本：自动探测 2010+ 缺失年份, 仅重下这些年(幂等 INSERT OR REPLACE), 最后重跑交叉验证。
"""
import sys, os, time, sqlite3
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA
import pandas as pd
from download_tr_index import cross_validate, DOWNLOADS

DB = DATA.get("local_db_path", r"D:\tu-shareData\astock_daily.db")


def get_pro():
    import tushare as ts
    tk = DATA.get("tushare_token", "")
    if not tk:
        print("[错误] tushare_token 为空")
        sys.exit(1)
    ts.set_token(tk)
    return ts.pro_api()


def ensure_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS index_tr_official (
            local_code TEXT, tr_code TEXT, idx_name TEXT,
            trade_date TEXT, close REAL,
            PRIMARY KEY (tr_code, trade_date)
        )""")
    conn.commit()


def backfill_one(pro, conn, local, tr, name):
    have = set(r[0] for r in conn.execute(
        "SELECT DISTINCT substr(trade_date,1,4) FROM index_tr_official WHERE tr_code=?", (tr,)))
    cal_years = sorted(set(d[:4] for d in [r[0] for r in conn.execute(
        "SELECT trade_date FROM index_daily WHERE ts_code='000300.SH' AND trade_date>='20100101'")]))
    targets = [y for y in cal_years if y >= "2010" and y not in have]
    if not targets:
        print(f"  {tr:20s} {name:22s} 无缺口(2010+)")
        return 0
    print(f"  {tr:20s} {name:22s} 待补年份: {targets}")
    added = 0
    for y in targets:
        s, e = f"{y}0101", f"{y}1231"
        ok = False
        for _ in range(3):
            try:
                df = pro.index_daily(ts_code=tr, start_date=s, end_date=e)
                if df is not None and not df.empty:
                    rows = [(local, tr, name, str(r["trade_date"]), float(r["close"]))
                            for _, r in df.iterrows()]
                    conn.executemany(
                        "INSERT OR REPLACE INTO index_tr_official "
                        "(local_code, tr_code, idx_name, trade_date, close) VALUES (?,?,?,?,?)", rows)
                    conn.commit()
                    added += len(rows)
                    ok = True
                    break
                else:
                    print(f"    {tr} {y}: 空响应, 重试")
                    time.sleep(1)
            except Exception as ex:
                msg = str(ex)
                if "频率" in msg or "每分钟" in msg:
                    print(f"    {tr} {y}: 限速退避60s")
                    time.sleep(60)
                else:
                    print(f"    {tr} {y}: ERR {msg[:50]}")
                    time.sleep(2)
        if not ok:
            print(f"    [失败] {tr} {y} 三次重试未成功")
        time.sleep(0.3)
    print(f"  {tr:20s} 补录 {added} 行")
    return added


def main():
    conn = sqlite3.connect(DB)
    ensure_table(conn)
    pro = get_pro()
    print("=" * 86)
    print("补全 index_tr_official 缺口年份(2010+)")
    print("=" * 86)
    for local, tr, name in DOWNLOADS:
        backfill_one(pro, conn, local, tr, name)
    print("\n" + "=" * 86)
    print("补全后交叉验证")
    print("=" * 86)
    cross_validate(conn)
    conn.close()


if __name__ == "__main__":
    main()
