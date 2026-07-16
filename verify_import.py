"""校验财务表导入结果（只读，不修改）"""
import sqlite3

DST = r"D:/tu-shareData/astock_daily.db"
SRC = r"D:\BaiduNetdiskDownload\data\astock_daily.db"

TABLES = {
    "income": "income",
    "balance_sheet": "balancesheet",
    "cashflow": "cashflow",
    "fina_indicator": "fina_indicator",
}

dst = sqlite3.connect(DST)
src = sqlite3.connect(SRC)

print("表 | 当前行数 | 百度行数 | 当前列数 | 百度列数 | MAX(end_date) | 日志done")
print("-" * 100)
for tbl, dtype in TABLES.items():
    dn = dst.execute(f'SELECT COUNT(*) FROM "{tbl}"').fetchone()[0]
    sn = src.execute(f'SELECT COUNT(*) FROM "{tbl}"').fetchone()[0]
    dc = len(dst.execute(f'PRAGMA table_info("{tbl}")').fetchall())
    sc = len(src.execute(f'PRAGMA table_info("{tbl}")').fetchall())
    mx = dst.execute(f'SELECT MAX(end_date) FROM "{tbl}"').fetchone()[0]
    ln = dst.execute(
        'SELECT COUNT(*) FROM financial_download_log WHERE data_type=? AND status=?',
        (dtype, "done")).fetchone()[0]
    print(f"{tbl:15s} | {dn:>8,} | {sn:>8,} | {dc:>3} | {sc:>3} | {mx} | {ln}")

# 抽样：600519.SH 在三张表应有数据
print("\n抽样 600519.SH:")
for tbl in TABLES:
    n = dst.execute(
        f'SELECT COUNT(*) FROM "{tbl}" WHERE ts_code="600519.SH"'
    ).fetchone()[0]
    print(f"  {tbl:15s}: {n} 行")

# 行情表未受影响
print("\n行情表（应保持原样）:")
for t in ["daily", "daily_basic", "adj_factor"]:
    n = dst.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
    mx = dst.execute(f'SELECT MAX(trade_date) FROM "{t}"').fetchone()[0]
    print(f"  {t:15s} 行数={n:>10,} MAX(trade_date)={mx}")

dst.close(); src.close()
print("\n校验完成（只读）")
