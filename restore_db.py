"""
恢复数据库：将 backup.sql（UTF-16LE 格式）导入 astock_daily.db
逐行读取，避免 4.6GB 一次性读入内存
"""
import sqlite3
import os

BACKUP = r"D:\tu-shareData\backup.sql"
DB = r"D:\tu-shareData\astock_daily.db"

# 删除可能存在的空数据库文件
if os.path.exists(DB):
    try:
        os.remove(DB)
        print(f"已删除旧文件: {DB}")
    except PermissionError:
        print(f"无法删除 {DB}，可能正在被使用。请先关闭所有打开该文件的程序。")
        exit(1)

conn = sqlite3.connect(DB)
cur = conn.cursor()

print("正在恢复数据库（逐行读取，分批提交）...")
print(f"备份文件大小: {os.path.getsize(BACKUP) / 1024 / 1024 / 1024:.1f} GB")
print()

line_count = 0
sql_buffer = ""

with open(BACKUP, "r", encoding="utf-16-le") as f:
    for line in f:
        stripped = line.strip()
        if not stripped:
            continue
        # 跳过 PRAGMA 和 BEGIN/COMMIT 指令
        upper = stripped.upper()
        if upper.startswith("PRAGMA") or upper == "BEGIN TRANSACTION;" or upper == "COMMIT;":
            continue
        if upper == "END TRANSACTION;":
            continue

        sql_buffer += line

        # 遇到分号结尾则执行
        if stripped.endswith(";"):
            try:
                cur.execute(sql_buffer)
                line_count += 1
            except sqlite3.Error as e:
                print(f"  [WARN] 第 {line_count + 1} 行执行失败: {e}")
                print(f"     SQL: {sql_buffer[:80]}...")
            sql_buffer = ""

            # 每 10000 行提交一次，并显示进度
            if line_count % 10000 == 0:
                conn.commit()
                if line_count > 0:
                    db_size = os.path.getsize(DB) / 1024 / 1024
                    print(f"  已处理 {line_count} 行，数据库大小: {db_size:.0f} MB")

# 执行剩余的 SQL
if sql_buffer.strip():
    try:
        cur.execute(sql_buffer)
        line_count += 1
    except sqlite3.Error as e:
        print(f"  [WARN] 最后一行执行失败: {e}")

conn.commit()
conn.close()

print()
print(f"✅ 恢复完成！共处理 {line_count} 行 SQL")
db_size = os.path.getsize(DB) / 1024 / 1024
print(f"   数据库大小: {db_size:.0f} MB")

# 验证表是否创建成功
conn = sqlite3.connect(DB)
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print(f"   表列表: {[t[0] for t in tables]}")
row_count = conn.execute("SELECT COUNT(*) FROM daily").fetchone()[0]
print(f"   daily 表行数: {row_count:,}")
conn.close()
