# -*- coding: utf-8 -*-
"""
补全指数「时点成分股」历史快照 —— 消灭幸存者偏差的数据侧解法
=================================================================
背景
-----
`index_constituent` 表里只有少数指数有完整历史快照：
    000300.SH 265 个快照(2010~2026)   000906.SH 114 个
    000905.SH 138 个                  000852.SH  86 个
而创业板指(399006.SZ)、深证成指(399001.SZ)、科创50(000688.SH) 等
**只有 1 个快照(20260706)**。用这唯一的名单去跑 2015 年的回测，
等于提前知道了「谁能活到 2026 年还留在指数里」→ 幸存者偏差 + 未来函数，
回测收益会被系统性夸大（实测创业板动量因此虚增到 +1250%）。

解法
-----
Tushare `index_weight` 接口提供**月度成分权重快照**，实测 399006.SZ
从 2011 年起每月都有 100 只成分的完整名单。本脚本把它拉下来并同步进
`index_constituent`，让 `get_index_constituents(idx, trade_date)` 能
拿到真正的时点名单。

用法
-----
    # 补创业板指 2010 年至今（约 190 次 API 调用，2~4 分钟）
    python backfill_index_constituent.py --index 399006.SZ --start 2010

    # 一次补多个
    python backfill_index_constituent.py --index 399006.SZ,399001.SZ --start 2010

    # 只把库里已有的 index_weight 同步进 index_constituent（不联网）
    python backfill_index_constituent.py --index 399006.SZ --sync-only

    # 看看现在各指数的快照覆盖情况
    python backfill_index_constituent.py --report

说明
-----
  - 幂等：INSERT OR REPLACE，可重复运行补数，不会产生重复行。
  - 与 download_index_weight.py 的区别：那个脚本只为「权重集中度」展示，
    默认从 2023 年起；本脚本面向回测股票池，默认回溯到 2010 年，并且
    **额外同步一份到 index_constituent**（回测实际读的是这张表）。
  - 已知无 index_weight 数据的指数：932000.SH(中证2000)。
  - ⚠️ **000985.SH(中证全指) 曾在此被误列为"无数据"，已于 2026-09-01 证伪**：
    真实原因是**后缀**——`000985.SH` → 0 行，但 **`000985.CSI` → 3400~5200 行**（2019→2026 全可用）。
    与 000922.SH / 930955.SH 是同一个坑（红利类指数在 index_weight 接口必须用 .CSI）。
    已用 `backfill_dividend_constituents.py --only 000985.SH` 补全，可据此自建其全收益指数。
    **判定"Tushare 无数据"前，必须先试 .SH / .SZ / .CSI 三种后缀变体。**
依赖: tushare, config.DATA.tushare_token / local_db_path
"""
import sys
import os
import time
import sqlite3
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA

DB_PATH = DATA.get("local_db_path", r"D:\tu-shareData\astock_daily.db")
TOKEN = DATA.get("tushare_token", "")

INDEX_NAME = {
    "000001.SH": "上证指数", "000016.SH": "上证50", "000300.SH": "沪深300",
    "000688.SH": "科创50", "000698.SH": "科创100", "000852.SH": "中证1000",
    "000905.SH": "中证500", "000906.SH": "中证800", "000985.SH": "中证全指",
    "399001.SZ": "深证成指", "399006.SZ": "创业板指", "399673.SZ": "创业板50",
    "932000.SH": "中证2000",
}

# 默认补这几个「只有 1 个快照」的指数
DEFAULT_INDEXES = ["399006.SZ", "399001.SZ", "399673.SZ", "000688.SH"]


def get_pro():
    import tushare as ts
    if not TOKEN:
        print("[错误] config.DATA.tushare_token 为空，无法调用 Tushare")
        sys.exit(1)
    ts.set_token(TOKEN)
    return ts.pro_api()


def ensure_tables(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS index_weight (
            index_code TEXT, ts_code TEXT, trade_date TEXT, weight REAL,
            PRIMARY KEY (index_code, ts_code, trade_date)
        )""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS index_constituent (
            index_code TEXT, ts_code TEXT, trade_date TEXT,
            weight REAL, index_name TEXT,
            UNIQUE(ts_code, index_code, trade_date)
        )""")
    conn.commit()


def fetch_month(pro, index_code, ms, me, attempt=0):
    """拉某指数某月的权重快照；限速自动退避。返回 list[(index_code, con_code, date, weight)]"""
    try:
        df = pro.index_weight(index_code=index_code, start_date=ms, end_date=me)
    except Exception as e:
        msg = str(e)
        if ("频率" in msg or "每分钟" in msg or "rate" in msg.lower()) and attempt < 4:
            print(f"    [限速] {index_code} {ms}: 退避 60s 重试({attempt + 1})")
            time.sleep(60)
            return fetch_month(pro, index_code, ms, me, attempt + 1)
        print(f"    [跳过] {index_code} {ms}: {msg[:80]}")
        return []
    if df is None or df.empty:
        return []
    rows = []
    for _, r in df.iterrows():
        try:
            rows.append((str(r["index_code"]), str(r["con_code"]),
                         int(r["trade_date"]), float(r["weight"])))
        except Exception:
            continue
    return rows


def download(pro, conn, index_code, start_ym, end_ym):
    """按月拉取并写入 index_weight。返回写入行数。"""
    sy, sm = int(start_ym[:4]), int(start_ym[4:6])
    ey, em = int(end_ym[:4]), int(end_ym[4:6])
    total, snaps, empty = 0, set(), 0
    for y in range(sy, ey + 1):
        for m in range(1, 13):
            if (y == sy and m < sm) or (y == ey and m > em):
                continue
            ms = f"{y}{m:02d}01"
            me = f"{y + 1}0101" if m == 12 else f"{y}{m + 1:02d}01"
            rows = fetch_month(pro, index_code, ms, me)
            if not rows:
                empty += 1
                time.sleep(0.2)
                continue
            conn.executemany(
                "INSERT OR REPLACE INTO index_weight "
                "(index_code, ts_code, trade_date, weight) VALUES (?,?,?,?)", rows)
            conn.commit()
            total += len(rows)
            snaps.update(r[2] for r in rows)
            time.sleep(0.25)
    if snaps:
        print(f"  [下载] {index_code}: {total} 行 / {len(snaps)} 个快照 "
              f"({min(snaps)}~{max(snaps)})，空月 {empty}")
    else:
        print(f"  [下载] {index_code}: 无任何数据（Tushare 可能不提供该指数的 index_weight）")
    return total


def sync_to_constituent(conn, index_code):
    """把 index_weight 的快照同步进 index_constituent（回测实际读的表）。"""
    name = INDEX_NAME.get(index_code, index_code)
    before = conn.execute(
        "SELECT COUNT(DISTINCT trade_date) FROM index_constituent WHERE index_code=?",
        (index_code,)).fetchone()[0]
    cur = conn.execute("""
        INSERT OR REPLACE INTO index_constituent
            (ts_code, index_code, index_name, trade_date, weight)
        SELECT ts_code, index_code, ?, trade_date, weight
        FROM index_weight WHERE index_code = ?
    """, (name, index_code))
    conn.commit()
    after = conn.execute(
        "SELECT COUNT(DISTINCT trade_date) FROM index_constituent WHERE index_code=?",
        (index_code,)).fetchone()[0]
    print(f"  [同步] {index_code} {name}: 写入 {cur.rowcount} 行，"
          f"快照数 {before} → {after}")
    return after


def report(conn):
    print(f"\n{'指数':<12}{'名称':<10}{'快照数':>7}  {'区间':<22}{'时点回测':<8}")
    print("-" * 66)
    rows = conn.execute("""
        SELECT index_code, COUNT(DISTINCT trade_date) nd,
               MIN(trade_date), MAX(trade_date)
        FROM index_constituent GROUP BY index_code
        HAVING index_code NOT LIKE '8%' ORDER BY nd DESC
    """).fetchall()
    for code, nd, d0, d1 in rows:
        ok = "✅ 可用" if nd >= 12 else "❌ 缺数据"
        print(f"{code:<12}{INDEX_NAME.get(code, ''):<10}{nd:>7}  {d0}~{d1:<12}{ok:<8}")
    print("-" * 66)
    print("快照数 < 12 的指数不要用来跑历史回测：会 fallback 失败（空仓）或引入幸存者偏差。")


def main():
    ap = argparse.ArgumentParser(description="补全指数时点成分股历史快照")
    ap.add_argument("--index", default=",".join(DEFAULT_INDEXES),
                    help="指数代码，逗号分隔。默认补 4 个缺历史的指数")
    ap.add_argument("--start", default="2010", help="起始 YYYY 或 YYYYMM")
    ap.add_argument("--end", default="", help="结束 YYYY 或 YYYYMM，默认当月")
    ap.add_argument("--sync-only", action="store_true", help="只同步已有 index_weight，不联网")
    ap.add_argument("--report", action="store_true", help="只打印覆盖报告")
    args = ap.parse_args()

    conn = sqlite3.connect(DB_PATH)
    ensure_tables(conn)

    if args.report:
        report(conn)
        conn.close()
        return

    start_ym = args.start if len(args.start) == 6 else args.start + "01"
    if args.end:
        end_ym = args.end if len(args.end) == 6 else args.end + "12"
    else:
        end_ym = time.strftime("%Y%m")

    codes = [c.strip() for c in args.index.split(",") if c.strip()]
    print(f"[*] 库: {DB_PATH}")
    print(f"[*] 指数: {codes}  区间: {start_ym}~{end_ym}  "
          f"{'(仅同步)' if args.sync_only else ''}")

    pro = None if args.sync_only else get_pro()
    for code in codes:
        print(f"\n=== {code} {INDEX_NAME.get(code, '')} ===")
        if not args.sync_only:
            download(pro, conn, code, start_ym, end_ym)
        sync_to_constituent(conn, code)

    report(conn)
    conn.close()


if __name__ == "__main__":
    main()
