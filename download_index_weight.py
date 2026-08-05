# -*- coding: utf-8 -*-
"""
下载指数成分权重 index_weight 到本地库
=========================================================
用于「指数/ETF 涨跌一览」增加"前十大权重集中度"硬列，回应被动投资
"分散悄悄集中"风险（如科创50/上证50 看似宽基实则高度集中于少数巨头）。

注意：
  - Tushare index_weight 字段为 index_code / con_code / trade_date / weight
    （con_code 是成分股代码，weight 为百分比，如 5.008 表示 5.008%）
  - Tushare 单次接口返回上限约 7000 行。上证指数(000001)单快照 2217 成分，
    整窗会截断 → 本脚本按"月"分页拉取，保证每个快照成分完整。
  - 部分指数在 Tushare 无 index_weight 数据（已知：932000 中证2000、
    000985 中证全指），脚本会跳过并打印提示。注：创业板50(399673) 实际有
    数据（前十大约 68.6%），此前预判断言"无数据"有误，已剔除。
  - 幂等：INSERT OR REPLACE，可重复运行补数。

用法:
    python download_index_weight.py
    python download_index_weight.py --start 20230101 --end 20260706
依赖: tushare, config.DATA.tushare_token / local_db_path
"""
import sys
import os
import time
import sqlite3
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tushare as ts
from config import DATA

DB_PATH = DATA.get("local_db_path", r"D:\tu-shareData\astock_daily.db")
TOKEN = DATA.get("tushare_token", "")

# 需要下载成分权重的指数（与 show_index_etf_changes.INDEX_MAP 对齐）
INDEX_CODES = [
    "000300.SH",  # 沪深300
    "000016.SH",  # 上证50
    "000906.SH",  # 中证800
    "000905.SH",  # 中证500
    "000852.SH",  # 中证1000
    "932000.SH",  # 中证2000（已知无数据，跳过）
    "000001.SH",  # 上证指数
    "399001.SZ",  # 深证成指
    "000985.SH",  # 中证全指（已知无数据，跳过）
    "399006.SZ",  # 创业板指
    "399673.SZ",  # 创业板50（实际有数据，保留下载）
    "000688.SH",  # 科创50
    "000698.SH",  # 科创100
]


def get_pro():
    if not TOKEN:
        print("[错误] config.DATA.tushare_token 为空，无法调用 Tushare")
        sys.exit(1)
    ts.set_token(TOKEN)
    return ts.pro_api()


def ensure_table(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS index_weight (
            index_code TEXT,
            con_code   TEXT,
            trade_date INTEGER,
            weight     REAL,
            PRIMARY KEY (index_code, con_code, trade_date)
        )
        """
    )
    conn.commit()


def fetch_month(pro, index_code, ym_start, ym_end, attempt=0):
    """拉取某指数在某月[start,end]的权重；带限速重试。返回 list of rows。"""
    try:
        df = pro.index_weight(
            index_code=index_code, start_date=ym_start, end_date=ym_end
        )
    except Exception as e:
        msg = str(e)
        # 限速类错误：退避后重试
        if ("频率" in msg or "每分钟" in msg or "rate" in msg.lower()) and attempt < 3:
            print(f"    [限速] {index_code} {ym_start}: 退避 60s 重试({attempt+1})")
            time.sleep(60)
            return fetch_month(pro, index_code, ym_start, ym_end, attempt + 1)
        print(f"    [跳过] {index_code} {ym_start}: {msg[:80]}")
        return []
    if df is None or df.empty:
        return []
    rows = []
    for _, r in df.iterrows():
        try:
            rows.append(
                (str(r["index_code"]), str(r["con_code"]),
                 int(r["trade_date"]), float(r["weight"]))
            )
        except Exception:
            continue
    return rows


def download_index(pro, index_code, start, end, conn):
    """按月分页下载单只指数的全部权重快照，写入 index_weight 表。返回写入行数。"""
    sy, ey = int(start[:4]), int(end[:4])
    sm = int(start[4:6])
    total = 0
    snap_dates = set()
    empty_months = 0
    for y in range(sy, ey + 1):
        for m in range(1, 13):
            if y == sy and m < sm:
                continue
            if y == ey and m > int(end[4:6]):
                break
            ms = f"{y}{m:02d}01"
            # 月末：下月1号减1天
            if m == 12:
                me = f"{y+1}0101"
            else:
                me = f"{y}{(m+1):02d}01"
            rows = fetch_month(pro, index_code, ms, me)
            if not rows:
                empty_months += 1
                time.sleep(0.2)
                continue
            conn.executemany(
                "INSERT OR REPLACE INTO index_weight "
                "(index_code, con_code, trade_date, weight) VALUES (?,?,?,?)",
                rows,
            )
            conn.commit()
            total += len(rows)
            snap_dates.update(r[2] for r in rows)
            time.sleep(0.25)
    # 该指数是否有任何数据
    chk = conn.execute(
        "SELECT 1 FROM index_weight WHERE index_code=? LIMIT 1", (index_code,)
    ).fetchone()
    if not chk:
        print(f"  [无数据] {index_code}：Tushare 未提供 index_weight（集中度列将显示 --）")
    else:
        print(f"  [完成] {index_code}：写入 {total} 行，{len(snap_dates)} 个快照日，"
              f"最新 {max(snap_dates)}")
    return total


def main():
    ap = argparse.ArgumentParser(description="下载指数成分权重 index_weight")
    ap.add_argument("--start", default="20230101", help="起始日期 YYYYMMDD")
    ap.add_argument("--end", default="20260706", help="结束日期 YYYYMMDD")
    args = ap.parse_args()
    pro = get_pro()
    conn = sqlite3.connect(DB_PATH)
    ensure_table(conn)
    print(f"[*] 目标库: {DB_PATH}  区间 {args.start}~{args.end}")
    overall = 0
    for code in INDEX_CODES:
        print(f"\n[下载] {code}")
        overall += download_index(pro, code, args.start, args.end, conn)
    conn.close()
    print(f"\n全部完成，累计写入 index_weight {overall} 行。")


if __name__ == "__main__":
    main()
