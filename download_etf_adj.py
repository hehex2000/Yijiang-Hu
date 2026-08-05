# -*- coding: utf-8 -*-
"""
下载 ETF 复权因子（Tushare fund_adj）进本地库，用于修复 etf_daily 未复权导致的拆分断点。

Tushare fund_adj 返回字段：ts_code, trade_date, adj_factor
  - adj_factor 为后复权因子（值随历史缩小，非"最新=1"）。
  - 使用前复权：复权价 = 未复权价 * adj_factor / adj_factor_latest，使最新交易日价格=真实可交易价，历史连续。

写入表：etf_adj_factor(ts_code, trade_date, adj_factor)，主键 (ts_code, trade_date)，INSERT OR REPLACE 幂等。

限速：滑动窗口，每分钟最多 MAX_CALLS 次（默认 175，低于 Tushare 的 200/分钟上限）。
断点续传：若某标的某自然年已在本地库覆盖到年末，则跳过该年，避免重复拉取与浪费配额。
重试：调用失败（含限流）自动退避重试，不再静默丢年。
"""
import sys
import os
import time
import sqlite3
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA
import tushare as ts

DB = "D:/tu-shareData/astock_daily.db"

# ── 限速配置：滑动窗口，每分钟最多调用次数（务必 < 200）──
MAX_CALLS = 175          # 每分钟上限（留 25 次余量，绝不超 200）
WINDOW = 60.0            # 滑动窗口长度（秒）
_call_times = deque()


def _ratelimit():
    """确保任意滚动 60s 窗口内调用次数不超过 MAX_CALLS。"""
    now = time.time()
    while _call_times and now - _call_times[0] >= WINDOW:
        _call_times.popleft()
    if len(_call_times) >= MAX_CALLS:
        sleep_for = WINDOW - (now - _call_times[0]) + 0.2
        if sleep_for > 0:
            time.sleep(sleep_for)
        _ratelimit()
    _call_times.append(time.time())


def _year_covered(con, code, s, e):
    """该自然年是否已在本地库完整覆盖（存在 >= 年末交易日的数据点）。"""
    r = con.execute(
        "SELECT MAX(trade_date) FROM etf_adj_factor "
        "WHERE ts_code=? AND trade_date>=? AND trade_date<=?",
        (code, int(s), int(e)),
    ).fetchone()
    mx = r[0] if r and r[0] else 0
    return mx >= int(e)


def _is_ratelimit(msg):
    m = msg.lower()
    return any(k in m for k in [
        "每分钟", "频率", "frequency", "rate", "exceed", "limit",
        "too many", "200", "访问该接口",
    ])


def _call_with_retry(pro, code, s, e, max_retries=6):
    """带限速 + 退避重试的 fund_adj 调用；全部失败返回 None。"""
    for attempt in range(max_retries):
        _ratelimit()
        try:
            df = pro.fund_adj(ts_code=code, start_date=s, end_date=e)
            return df
        except Exception as ex:
            msg = str(ex)
            wait = 5 * (attempt + 1)
            if attempt < max_retries - 1:
                tag = "限流" if _is_ratelimit(msg) else "异常"
                print(f"  [{tag}] {code} {s[:4]} 第{attempt+1}次重试，等待 {wait}s: {msg[:70]}")
                time.sleep(wait)
                continue
            print(f"  [WARN] {code} {s[:4]} fund_adj 放弃(已达重试上限): {msg[:90]}")
            return None
    return None


# 网格菜单 + 指数ETF一览实际用到的 ETF
ETF_LIST = [
    "510300.SH",  # 沪深300ETF
    "510050.SH",  # 上证50ETF
    "515800.SH",  # 中证800ETF（汇添富）
    "510500.SH",  # 中证500ETF（南方）
    "512100.SH",  # 中证1000ETF（南方）
    "563300.SH",  # 中证2000ETF
    "510210.SH",  # 上证指数ETF
    "159903.SZ",  # 深成ETF
    "159915.SZ",  # 创业板ETF
    "159949.SZ",  # 创业板50ETF
    "588000.SH",  # 科创50ETF
    "588190.SH",  # 科创100ETF
    # ── 红利类（高股息，分红必须计入，否则回报严重低估）──
    "510880.SH",  # 红利ETF(上证红利)
    "512890.SH",  # 红利低波ETF
    "515080.SH",  # 中证红利ETF
    "515100.SH",  # 红利低波100ETF
]


def main():
    ts.set_token(DATA["tushare_token"])
    pro = ts.pro_api()

    con = sqlite3.connect(DB)
    con.execute(
        """CREATE TABLE IF NOT EXISTS etf_adj_factor (
            ts_code     TEXT,
            trade_date  INTEGER,
            adj_factor  REAL,
            PRIMARY KEY (ts_code, trade_date)
        )"""
    )
    con.commit()

    years = list(range(2000, 2027))
    for code in ETF_LIST:
        total = 0
        skipped = 0
        for y in years:
            s = f"{y}0101"
            e = f"{y}1231"
            # 断点续传：若该年已被本地库完整覆盖，跳过（避免重复拉取、浪费配额）
            if _year_covered(con, code, s, e):
                skipped += 1
                continue
            df = _call_with_retry(pro, code, s, e)
            if df is not None and not df.empty:
                rows = [
                    (r.ts_code, int(r.trade_date), float(r.adj_factor))
                    for r in df.itertuples()
                ]
                con.executemany(
                    "INSERT OR REPLACE INTO etf_adj_factor VALUES (?, ?, ?)", rows
                )
                total += len(rows)
        con.commit()
        print(f"✅ {code}: 新增/更新 {total} 行，跳过已完整年份 {skipped} 个")

    con.close()
    print("ALL_DONE")


if __name__ == "__main__":
    main()
