# -*- coding: utf-8 -*-
"""
回补 沪深300(000300.SH) 缺失的历史成分股权重快照 -> 本地库 index_constituent 表
=================================================================================
背景:
  index_constituent 中 hs300 仅有 20160129 之后的快照，2010-2015 及 2018 整段缺失。
  用 hs300 池做 2015/2016 回测时，因无快照会触发「前视偏差退化」(当年含未来成分股)。
  本脚本用 Tushare index_weight 接口补回缺失年份。

安全设计(重要):
  - 采用 INSERT OR IGNORE，严格尊重 UNIQUE(ts_code, index_code, trade_date)。
  - **绝不 DELETE 任何现有行**，只补充缺失年份，已有的 2016+/2017/2019+ 数据原样保留。
  - 仅针对 000300.SH(沪深300)；如需其它指数请另写或扩展 YEARS/INDEX_CODE。

用法:
  python backfill_hs300_constituents.py
"""
import sys, os, sqlite3, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from config import DATA
import tushare as ts

DB_PATH = DATA.get("local_db_path", r"D:\tu-shareData\astock_daily.db")
TOKEN = DATA.get("tushare_token", "")
INDEX_CODE = "000300.SH"
INDEX_NAME = "沪深300"
# 缺失年份: 2010-2015 + 2018(已确认整段缺失)。2016/2017/2019+ 已存在, INSERT OR IGNORE 自动跳过。
YEARS = list(range(2010, 2016)) + [2018]


def get_pro():
    if not TOKEN:
        raise SystemExit("[错误] config.DATA.tushare_token 为空")
    ts.set_token(TOKEN)
    return ts.pro_api()


def fetch_year(pro, year):
    """拉取该年 index_weight，返回 list[(index_code, ts_code, trade_date, weight)]。"""
    s, e = f"{year}0101", f"{year}1231"
    df = pro.index_weight(index_code=INDEX_CODE, start_date=s, end_date=e)
    rows = []
    if df is None or df.empty:
        return rows
    for _, r in df.iterrows():
        td = str(r["trade_date"])
        code = str(r["con_code"])
        w = float(r["weight"]) if pd.notna(r.get("weight")) else None
        rows.append((INDEX_CODE, code, td, w))
    return rows


def main():
    pro = get_pro()
    conn = sqlite3.connect(DB_PATH)

    before = conn.execute(
        "SELECT COUNT(DISTINCT trade_date) FROM index_constituent WHERE index_code=?",
        (INDEX_CODE,),
    ).fetchone()[0]
    before_rows = conn.execute(
        "SELECT COUNT(*) FROM index_constituent WHERE index_code=?", (INDEX_CODE,)
    ).fetchone()[0]
    print(f"回补前 hs300: 时点快照 {before} 个, 行数 {before_rows}")

    total_written = 0
    for y in YEARS:
        try:
            rows = fetch_year(pro, y)
        except Exception as ex:
            print(f"  [跳过] {y}: {ex}")
            continue
        if not rows:
            print(f"  [空]   {y} 无数据")
            continue
        n_snap = len({r[2] for r in rows})
        # INSERT OR IGNORE: 已有 (ts_code,index_code,trade_date) 的行不会重复写入
        conn.executemany(
            "INSERT OR IGNORE INTO index_constituent "
            "(index_code, ts_code, trade_date, weight, index_name) VALUES (?,?,?,?,?)",
            [(ic, c, td, w, INDEX_NAME) for (ic, c, td, w) in rows],
        )
        conn.commit()
        # 实际新增行数
        added = conn.execute(
            "SELECT COUNT(*) FROM index_constituent WHERE index_code=? AND trade_date>=? AND trade_date<=?",
            (INDEX_CODE, f"{y}0101", f"{y}1231"),
        ).fetchone()[0]
        print(f"  [年] {y}: 取到 {len(rows)} 行 / {n_snap} 快照, 该年现有 {added} 行")
        total_written += len(rows)
        time.sleep(0.3)

    after = conn.execute(
        "SELECT COUNT(DISTINCT trade_date) FROM index_constituent WHERE index_code=?",
        (INDEX_CODE,),
    ).fetchone()[0]
    after_rows = conn.execute(
        "SELECT COUNT(*) FROM index_constituent WHERE index_code=?", (INDEX_CODE,)
    ).fetchone()[0]
    conn.close()

    print(f"\n完成: 尝试写入 {total_written} 行(重复已忽略); hs300 时点快照 {before} -> {after}, 行数 {before_rows} -> {after_rows}")

    # 验证: 列出各年快照数
    import sqlite3 as _sq
    c = _sq.connect(DB_PATH)
    print("hs300 各年快照数:")
    for r in c.execute(
        "SELECT substr(trade_date,1,4) yr, COUNT(DISTINCT trade_date) n "
        "FROM index_constituent WHERE index_code=? GROUP BY yr ORDER BY yr",
        (INDEX_CODE,),
    ):
        print("  ", r[0], r[1])
    c.close()


if __name__ == "__main__":
    main()
