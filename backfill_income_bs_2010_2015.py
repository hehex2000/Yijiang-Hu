# -*- coding: utf-8 -*-
"""
回填 income + balance_sheet 的 2010~2015 年报（逐只股票从 Tushare 拉取并 upsert）
============================================================================
背景：
  神奇公式 run_magic_formula.py 只用 end_date LIKE '%1231' 的年报，且要
  income.ebit + balance_sheet(total_cur_assets/total_cur_liab/fix_assets/
  total_liab/money_cap) 五字段齐全。
  2010~2013 年报在库中极度稀疏（2012 income 仅 796 行、balance 1399 行），
  导致 2014（需 2012 年报）/2015（需 2013 年报）选股落空，报"无有效财务指标"。

做法：
  1. 预载 income/balance_sheet 现有 (ts_code,end_date) 集合（2 次查询，非逐只）
  2. 在内存里算每只股票缺哪些 2010~2015 年报
  3. 仅对缺失股票：pro.income / pro.balancesheet 拉 2010~2015 全历史 →
     过滤到年报 → upsert（DELETE 旧 + INSERT 新），只动 2010~2015 年报，
     不碰 2016+ 已完整的数据
  4. 写到正确库 D:/tu-shareData/astock_daily.db

注意：当前 Tushare 1.4.29 的 income/balancesheet 接口强制要求 ts_code，
      不能只传 end_date 批量拉，因此只能逐只。已跳过全覆盖股票以省调用。
"""
import sqlite3
import time
import sys
import os
from collections import defaultdict

import tushare as ts

# ── 配置 ───────────────────────────────────────────────────────
DB = r"D:/tu-shareData/astock_daily.db"
TOKEN = "761165a821532fe625262d6b33e144b9859a887c004acbcb981c319b"
ANNUAL = ['20101231', '20111231', '20121231', '20131231', '20141231', '20151231']
ANNUAL_SET = set(ANNUAL)
INTERVAL = 0.3          # 每次股票调用后休眠（秒），避免触发限速
MAX_RETRY = 3


def db_cols(conn, table):
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def load_have(conn, table):
    """返回 {ts_code: set(end_date)} —— 一次查询全表，避免逐只慢查询。"""
    d = defaultdict(set)
    for code, ed in conn.execute(f"SELECT ts_code, end_date FROM {table}"):
        d[code].add(ed)
    return d


def upsert(conn, table, df, cols):
    """把 df 中属于 2010~2015 年报的行 upsert 进 table（按 ts_code+end_date 覆盖）。"""
    df = df[[c for c in df.columns if c in cols]]
    df = df[df['end_date'].isin(ANNUAL_SET)]
    # Tushare balancesheet 偶发同一 end_date 返回重复行，先去重避免表里留重复年报
    df = df.drop_duplicates(subset=['ts_code', 'end_date'])
    if len(df) == 0:
        return 0
    for _, row in df.iterrows():
        conn.execute(f"DELETE FROM {table} WHERE ts_code=? AND end_date=?",
                     (row['ts_code'], row['end_date']))
    df.to_sql(table, conn, if_exists="append", index=False, method="multi")
    return len(df)


def fetch_with_retry(pro, api_func, ts_code):
    for attempt in range(MAX_RETRY):
        try:
            df = api_func(ts_code=ts_code, start_date='20100101', end_date='20151231')
            return df
        except Exception as e:
            msg = str(e)
            if '必填' in msg:
                return None
            print(f"    [重试 {attempt+1}/{MAX_RETRY}] {ts_code}: {msg[:80]}")
            time.sleep(3)
    return None


def main():
    ts.set_token(TOKEN)
    pro = ts.pro_api()

    conn = sqlite3.connect(DB)
    inc_cols = db_cols(conn, 'income')
    bs_cols = db_cols(conn, 'balance_sheet')

    print("[1/4] 预载现有年报集合 ...")
    inc_have = load_have(conn, 'income')
    bs_have = load_have(conn, 'balance_sheet')

    codes = [r[0] for r in conn.execute(
        "SELECT DISTINCT ts_code FROM daily WHERE trade_date <= ?", ('20151231',))]

    # 精准回填：--pool hs300 只补沪深300成分股（2010~2015 期间曾入选者）
    if len(sys.argv) > 1 and '--pool' in sys.argv:
        pi = sys.argv.index('--pool') + 1
        pool = sys.argv[pi] if pi < len(sys.argv) else 'hs300'
        idx_map = {'hs300': '000300.SH', 'zz500': '000905.SH',
                   'zz800': '000906.SH', 'zz1000': '000852.SH',
                   'zz2000': '932000.SH'}
        idx = idx_map.get(pool, '000300.SH')
        pool_codes = [r[0] for r in conn.execute(
            "SELECT DISTINCT ts_code FROM index_constituent "
            "WHERE index_code=? AND CAST(trade_date AS INTEGER)<=20151231", (idx,))]
        codes = pool_codes
        print(f"  [--pool {pool}] 限制为 {idx} 成分股 {len(codes)} 只")

    # 算待补清单
    todo = []
    for c in codes:
        mi = [p for p in ANNUAL if p not in inc_have.get(c, set())]
        mb = [p for p in ANNUAL if p not in bs_have.get(c, set())]
        if mi or mb:
            todo.append((c, mi, mb))

    # 冒烟测试用：仅处理前 N 只
    if len(sys.argv) > 1 and sys.argv[1] == '--limit':
        try:
            lim = int(sys.argv[2])
            todo = todo[:lim]
            print(f"  [--limit] 仅处理前 {lim} 只")
        except (IndexError, ValueError):
            pass
    print(f"[2/4] 待补股票: {len(todo)} / {len(codes)}")
    conn.close()

    if not todo:
        print("无需补充，退出。")
        return

    conn = sqlite3.connect(DB)
    inc_total = bs_total = 0
    done = 0
    t0 = time.time()

    print("[3/4] 开始逐只回填 ...")
    for c, mi, mb in todo:
        if mi:
            df = fetch_with_retry(pro, pro.income, c)
            if df is not None and len(df) > 0:
                inc_total += upsert(conn, 'income', df, inc_cols)
            conn.commit()
            time.sleep(INTERVAL)
        if mb:
            df = fetch_with_retry(pro, pro.balancesheet, c)
            if df is not None and len(df) > 0:
                bs_total += upsert(conn, 'balance_sheet', df, bs_cols)
            conn.commit()
            time.sleep(INTERVAL)
        done += 1
        if done % 50 == 0:
            print(f"  [{done}/{len(todo)}] 用时{time.time()-t0:.0f}s "
                  f"新增income={inc_total} balance={bs_total}")

    conn.commit()
    conn.close()
    print(f"[4/4] 完成：处理 {done} 只，新增 income={inc_total} 行、"
          f"balance_sheet={bs_total} 行，总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
