"""回补 2015-01-05 之前的 adj_factor（后复权因子），写入本地 adj_factor 表。

背景:
- 本库 adj_factor 最早仅 2015-01-05，导致 run_dogs_annual.py 等回测在 pre-2015
  的 hfq 轨退化为「锚定 2015 基准因子」(get_hfq_price 后向填充)，造成
  pre-2015→2015 的虚假台阶、且 pre-2015 不含真实分红。
- 用 Tushare pro.adj_factor 逐股回补 2000-01-01~2014-12-31（其实只取 <20150105）。
- INSERT OR IGNORE：不覆盖现有 2015+ 数据；与 2015-01-05 基准因子连续衔接。
- 可重复运行：启动时自动跳过已有 pre-2015 因子的 ts_code（断点续传）。
"""
import sqlite3
import time
import sys
import config
import tushare as ts
import pandas as pd

DB = config.DATA["local_db_path"]
TOKEN = config.DATA["tushare_token"]
CUTOFF = "20150105"          # 只补该日期之前的因子
START = "20000101"          # 向前最多补到 2000 年（Tushare adj_factor 一般支持到上市日起）

pro = ts.pro_api(TOKEN)


def already_done(conn):
    """返回已有 pre-2015 因子的 ts_code 集合。"""
    rows = conn.execute(
        "SELECT DISTINCT ts_code FROM adj_factor WHERE trade_date < ?", (CUTOFF,)
    ).fetchall()
    return {r[0] for r in rows}


def need_backfill(conn):
    """返回在 daily 里有 pre-2015 数据、但 adj_factor 还缺 pre-2015 的 ts_code。"""
    rows = conn.execute(
        "SELECT DISTINCT ts_code FROM daily WHERE trade_date < ?", (CUTOFF,)
    ).fetchall()
    return [r[0] for r in rows]


def backfill_one(ts_code):
    df = pro.adj_factor(ts_code=ts_code, start_date=START, end_date="20141231")
    if df is None or len(df) == 0:
        return 0
    df = df[df["trade_date"] < CUTOFF]
    if len(df) == 0:
        return 0
    rows = [
        (ts_code, str(r["trade_date"]), float(r["adj_factor"]))
        for _, r in df.iterrows()
    ]
    c = sqlite3.connect(DB)
    c.executemany(
        "INSERT OR IGNORE INTO adj_factor (ts_code, trade_date, adj_factor) VALUES (?,?,?)",
        rows,
    )
    n = c.total_changes
    c.commit()
    c.close()
    return n


def main():
    conn = sqlite3.connect(DB)
    done = already_done(conn)
    todo = [t for t in need_backfill(conn) if t not in done]
    conn.close()
    print(f"[start] 已补 {len(done)} 只，待补 {len(todo)} 只", flush=True)

    ok = 0
    fail = 0
    tot_rows = 0
    for i, t in enumerate(todo):
        try:
            n = backfill_one(t)
        except Exception as e:
            time.sleep(3)
            try:
                n = backfill_one(t)
            except Exception as e2:
                print(f"[FAIL] {t}: {e2}", flush=True)
                fail += 1
                n = 0
        if n > 0:
            ok += 1
            tot_rows += n
        if (i + 1) % 100 == 0 or (i + 1) == len(todo):
            print(f"[{i+1}/{len(todo)}] ok={ok} fail={fail} rows+={tot_rows}", flush=True)
        time.sleep(0.08)   # 轻量限速，避免触发 Tushare 频率限制

    print(f"[done] 成功 {ok} 只，失败 {fail} 只，新增因子行 {tot_rows}", flush=True)


if __name__ == "__main__":
    main()
