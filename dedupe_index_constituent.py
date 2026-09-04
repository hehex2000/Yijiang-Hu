# -*- coding: utf-8 -*-
"""index_constituent 重复行去重（默认 dry-run，真正删除必须显式 --apply）。

## 背景
(index_code, ts_code, trade_date) 三键存在重复行，重复行之间**只差 index_name**
（一条有值如「中证800」、一条 NULL），weight 完全一致。
→ 整行 DISTINCT 无效，必须按三键分组、保留 index_name 非空的那条。

实测（2026-09-02）：
    index_code   行数      唯一      多余     name 为 NULL
    000906.SH   240,000   156,000   84,000    88,800
    其余 10 个 index：0 多余行（干净）
  重复组中 weight 不一致的组数 = 0 → 去重不丢信息，安全。

## 在哪执行
库：D:\\tu-shareData\\astock_daily.db（平台主库，7.6GB，与所有策略共用一个库）
通道：本脚本用平台 venv 的 python（sqlite3 模块 3.49.1，窗口函数可用）
    cd C:\\Users\\99395\\WorkBuddy\\multi_factor_selection
    ./venv_ml/Scripts/python.exe dedupe_index_constituent.py            # 只看
    ./venv_ml/Scripts/python.exe dedupe_index_constituent.py --apply    # 真删

## 安全设计
- 默认 dry-run：只打印将删除多少、抽样对比、前后计数预览，**不写库**。
- --apply 时先 `VACUUM INTO` 做一致性备份，再在事务内删除 + 校验，
  校验不过自动 ROLLBACK。
"""
import argparse
import os
import sqlite3
import sys
import time

DB_PATH = os.environ.get("LOCAL_DB_PATH", r"D:\tu-shareData\astock_daily.db")

# 默认只处理有重复的 index（实测只有它）；传 --index all 处理全部
DEFAULT_INDEX = "000906.SH"


def dup_rowids_sql(where_idx):
    """选出「应被删掉」的 rowid：三键分组内，保留 index_name 非空的那条。"""
    where = "WHERE index_code = ?" if where_idx else ""
    return f"""
        CREATE TEMP TABLE _dup_rowids AS
        SELECT rowid AS rid
        FROM (
            SELECT rowid,
                   ROW_NUMBER() OVER (
                       PARTITION BY index_code, ts_code, trade_date
                       ORDER BY (index_name IS NOT NULL) DESC, rowid
                   ) AS rn
            FROM index_constituent
            {where}
        )
        WHERE rn > 1
    """


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="真正执行删除（默认 dry-run 只打印）")
    ap.add_argument("--index", default=DEFAULT_INDEX,
                    help=f"要处理的 index_code，默认 {DEFAULT_INDEX}；传 all 处理全部")
    ap.add_argument("--db", default=DB_PATH)
    args = ap.parse_args()

    where_idx = None if args.index.lower() == "all" else args.index
    conn = sqlite3.connect(args.db)

    before = conn.execute("SELECT COUNT(*) FROM index_constituent").fetchone()[0]

    # 1) 选出待删 rowid（临时表，避免 DELETE 子查询自引用）
    t0 = time.time()
    conn.execute("DROP TABLE IF EXISTS temp._dup_rowids")
    sql = dup_rowids_sql(where_idx)
    conn.execute(sql, (where_idx,) if where_idx else ())
    n_del = conn.execute("SELECT COUNT(*) FROM temp._dup_rowids").fetchone()[0]
    conn.commit()
    print(f"[库] {args.db}")
    print(f"[范围] index_code = {where_idx or 'ALL'}")
    print(f"[当前] index_constituent 总行数 {before:,}")
    print(f"[待删] 多余行 {n_del:,}（{time.time()-t0:.1f}s）")

    if n_del == 0:
        print("\n没有重复行，无需处理。")
        conn.close()
        return

    # 2) 抽样：确认待删行确实是「index_name 为 NULL 的那条」，保留行有值
    #    注意：不能用 rowid±1 取邻近行——rowid 相邻的两行可能属于不同 index /
    #    不同日期（实测串出过 000905.SH 的记录）。必须按三键回查配对。
    print("\n=== 抽样 5 组（同一三键下：保留行 vs 待删行）===")
    keys = conn.execute("""
        SELECT ic.index_code, ic.ts_code, ic.trade_date
        FROM index_constituent ic
        WHERE ic.rowid IN (SELECT rid FROM temp._dup_rowids)
        LIMIT 5
    """).fetchall()
    for k in keys:
        rows = conn.execute("""
            SELECT weight, index_name,
                   CASE WHEN rowid IN (SELECT rid FROM temp._dup_rowids)
                        THEN '待删' ELSE '保留' END
            FROM index_constituent
            WHERE index_code = ? AND ts_code = ? AND trade_date = ?
            ORDER BY 3
        """, k).fetchall()
        print(f"  {k[0]} {k[1]} {k[2]}")
        for w, nm, tag in rows:
            print(f"      [{tag}] w={w}  name={nm}")

    # 3) 校验：待删行里是否混有 index_name 非空且是组内唯一的（不该发生）
    bad = conn.execute("""
        SELECT COUNT(*) FROM index_constituent
        WHERE rowid IN (SELECT rid FROM temp._dup_rowids)
          AND index_name IS NOT NULL
    """).fetchone()[0]
    print(f"\n[校验] 待删行中 index_name 非空的行数: {bad:,}"
          f"（应为 0，非 0 说明保留策略反了）")

    if not args.apply:
        conn.execute("DROP TABLE IF EXISTS temp._dup_rowids")
        conn.close()
        print("\n" + "=" * 60)
        print("DRY-RUN 结束，未写库。确认无误后加 --apply 执行：")
        print(f"  ./venv_ml/Scripts/python.exe {os.path.basename(__file__)}"
              f" --index {args.index} --apply")
        return

    if bad:
        print("\n[中止] 校验未通过，拒绝执行删除。")
        conn.close()
        sys.exit(2)

    # 4) 备份（VACUUM INTO 保证一致性快照）
    stamp = time.strftime("%Y%m%d_%H%M%S")
    bak = f"{args.db}.bak_dedup_{stamp}"
    print(f"\n[备份] VACUUM INTO {bak} ...")
    t0 = time.time()
    conn.execute(f"VACUUM INTO '{bak}'")
    print(f"[备份] 完成，{os.path.getsize(bak)/1e9:.2f} GB（{time.time()-t0:.1f}s）")

    # 5) 事务内删除 + 校验
    print("\n[执行] BEGIN ...")
    conn.execute("BEGIN")
    try:
        conn.execute("DELETE FROM index_constituent WHERE rowid IN "
                     "(SELECT rid FROM temp._dup_rowids)")
        after = conn.execute("SELECT COUNT(*) FROM index_constituent").fetchone()[0]
        rest = conn.execute(
            "SELECT COUNT(*) FROM (SELECT 1 FROM index_constituent "
            "GROUP BY index_code, ts_code, trade_date HAVING COUNT(*) > 1)"
        ).fetchone()[0]
        print(f"[执行] 删除 {before - after:,} 行；剩余 {after:,} 行")
        print(f"[校验] 残留重复组 {rest:,}（应为 0）")
        print(f"[校验] 预期 {before:,} - {n_del:,} = {before - n_del:,}，"
              f"实际 {after:,}")
        if rest != 0 or after != before - n_del:
            raise RuntimeError("校验未通过，回滚")
        conn.execute("COMMIT")
        print("[执行] COMMIT 成功")
    except Exception as e:
        conn.execute("ROLLBACK")
        print(f"[中止] {e} → 已 ROLLBACK，库未改动")
        sys.exit(2)
    finally:
        conn.execute("DROP TABLE IF EXISTS temp._dup_rowids")

    conn.close()
    print(f"\n备份保留在：{bak}\n确认无误后可自行删除该备份。")


if __name__ == "__main__":
    main()
