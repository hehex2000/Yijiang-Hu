"""将百度文库(D:\BaiduNetdiskDownload\data\astock_daily.db)的财务表
完整导入当前库(D:/tu-shareData/astock_daily.db)。
策略：按百度库的完整 schema 重建当前库四张财务表，再整表 INSERT。
行情表(daily/daily_basic/adj_factor)完全不动。
"""
import sqlite3, time, datetime

SRC = r"D:\BaiduNetdiskDownload\data\astock_daily.db"
DST = r"D:/tu-shareData/astock_daily.db"

# 财务表 -> 下载日志中使用的 data_type 名
TABLES = {
    "income": "income",
    "balance_sheet": "balancesheet",
    "cashflow": "cashflow",
    "fina_indicator": "fina_indicator",
}

def main():
    t0 = time.time()
    dst = sqlite3.connect(DST)
    dst.execute("ATTACH DATABASE ? AS src", (SRC,))
    cur = dst.cursor()

    for tbl, dtype in TABLES.items():
        print(f"\n=== 处理 {tbl} (data_type={dtype}) ===")
        # 1) 取百度库 DDL
        ddl = cur.execute(
            "SELECT sql FROM src.sqlite_master WHERE type='table' AND name=?"
        , (tbl,)).fetchone()
        if not ddl or not ddl[0]:
            print(f"  源库无表 {tbl}，跳过")
            continue
        ddl_sql = ddl[0]

        # 2) 删除当前库旧表（精简 schema 或空表）
        cur.execute(f'DROP TABLE IF EXISTS main."{tbl}"')
        # 3) 用百度 DDL 重建
        cur.execute(ddl_sql)
        print(f"  已按完整 schema 重建（{ddl_sql.count(chr(44))+1} 列）")

        # 4) 整表导入
        before = cur.execute(f'SELECT COUNT(*) FROM main."{tbl}"').fetchone()[0]
        cur.execute(f'INSERT INTO main."{tbl}" SELECT * FROM src."{tbl}"')
        after = cur.execute(f'SELECT COUNT(*) FROM main."{tbl}"').fetchone()[0]
        mx = cur.execute(f'SELECT MAX(end_date) FROM main."{tbl}"').fetchone()[0]
        print(f"  导入行数: {before} -> {after} | MAX(end_date)={mx}")

    dst.commit()
    print(f"\n[1/2] 财务表导入完成，耗时 {time.time()-t0:.1f}s")

    # ---- 同步下载日志 ----
    print("\n=== 同步 financial_download_log ===")
    now = datetime.datetime.now().isoformat()
    for tbl, dtype in TABLES.items():
        # 取该表实际存在的报告期
        periods = [r[0] for r in cur.execute(
            f'SELECT DISTINCT end_date FROM main."{tbl}" ORDER BY end_date'
        ).fetchall()]
        if not periods:
            continue
        # 先清掉该 data_type 的旧记录，再写入 done
        cur.execute(
            'DELETE FROM financial_download_log WHERE data_type=?', (dtype,)
        )
        cur.executemany(
            'INSERT INTO financial_download_log (data_type, period, status, retry_count, update_time) VALUES (?,?,?,?,?)',
            [(dtype, p, "done", 0, now) for p in periods],
        )
        print(f"  {dtype}: 写入 {len(periods)} 期 done 记录")
    dst.commit()

    # ---- 校验 ----
    print("\n=== 校验 ===")
    for tbl, dtype in TABLES.items():
        n = cur.execute(f'SELECT COUNT(*) FROM main."{tbl}"').fetchone()[0]
        cols = len(cur.execute(f'PRAGMA table_info(main."{tbl}")').fetchall())
        log_n = cur.execute(
            'SELECT COUNT(*) FROM financial_download_log WHERE data_type=? AND status=?',
            (dtype, "done")).fetchone()[0]
        print(f"  {tbl:15s} 行数={n:>8,} 列数={cols:>3} 日志done={log_n}")

    # 确认行情表不受影响
    for t in ["daily", "daily_basic", "adj_factor"]:
        n = cur.execute(f'SELECT COUNT(*) FROM main."{t}"').fetchone()[0]
        mx = cur.execute(f'SELECT MAX(trade_date) FROM main."{t}"').fetchone()[0]
        print(f"  {t:15s} 行数={n:>10,} MAX(trade_date)={mx}  (应保持不变)")

    dst.close()
    print(f"\n全部完成，总耗时 {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
