# -*- coding: utf-8 -*-
"""
补齐轮动池中滞后的 7 只 ETF/LOF 行情，使其追上 etf_daily 当前最新交易日。

背景：用户增量更新 ETF 数据时，轮动池 20 只里只有 13 只到了 20260803，
其余 7 只（上证指数/半导体/恒生/纳指/黄金/原油LOF/华宝添益）仍停在 20260703。
本脚本只补齐这 7 只 20260704~20260803 的缺口，绝不动其它标的、也不会过头。

复用 download_etf_data.download_fund_daily（同一 tushare 接口、同一 11 字段），
写入上限钉在 etf_daily 当前 MAX(trade_date)（=前沿），保证补齐后 20 只对齐。

用法:
    python refresh_etf_gap.py
"""
import os
import sys
import sqlite3
import datetime
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from download_etf_data import download_fund_daily, get_conn, ensure_table

DB_PATH = r"D:/tu-shareData/astock_daily.db"

# 轮动池中仍停在 20260703 的 7 只（其余 13 只已到 20260803）
STALE = [
    ("510980.SH", "上证指数ETF"),
    ("512480.SH", "半导体ETF"),
    ("159920.SZ", "恒生ETF"),
    ("513100.SH", "纳指ETF"),
    ("518880.SH", "黄金ETF"),
    ("501018.SH", "原油LOF"),
    ("511990.SH", "华宝添益"),
]


def next_day(date8):
    d = datetime.datetime.strptime(date8, "%Y%m%d")
    return (d + datetime.timedelta(days=1)).strftime("%Y%m%d")


def frontier(conn):
    return conn.execute("SELECT MAX(trade_date) FROM etf_daily").fetchone()[0]


def main():
    conn = get_conn()
    ensure_table(conn)
    fr = frontier(conn)
    print(f"etf_daily 当前前沿(最新交易日): {fr}")
    print(f"将为以下 7 只补齐 {next_day('20260703')} ~ {fr} 的行情:\n")

    total = 0
    for code, name in STALE:
        cur_max = conn.execute(
            "SELECT MAX(trade_date) FROM etf_daily WHERE ts_code=?", (code,)
        ).fetchone()[0] or "00000000"
        start = next_day(cur_max) if cur_max != "00000000" else "20100101"
        end = fr
        print(f"[{code}] {name}: 本地最新={cur_max}  拉取区间 {start}~{end}")
        df = download_fund_daily(code, start, end)
        if df is None or len(df) == 0:
            print(f"    [WARN] tushare 未返回数据（该区间可能无交易/接口限频），跳过")
            continue
        existing = {r[0] for r in conn.execute(
            "SELECT trade_date FROM etf_daily WHERE ts_code=?", (code,)).fetchall()}
        ins = 0
        for _, row in df.iterrows():
            ds = str(row["trade_date"])
            if ds in existing:
                continue
            conn.execute("""
                INSERT OR IGNORE INTO etf_daily
                (ts_code, trade_date, open, high, low, close,
                 pre_close, change, pct_chg, vol, amount)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(row["ts_code"]), ds,
                float(row["open"]) if pd.notna(row["open"]) else None,
                float(row["high"]) if pd.notna(row["high"]) else None,
                float(row["low"]) if pd.notna(row["low"]) else None,
                float(row["close"]) if pd.notna(row["close"]) else None,
                float(row["pre_close"]) if pd.notna(row["pre_close"]) else None,
                float(row["change"]) if pd.notna(row["change"]) else None,
                float(row["pct_chg"]) if pd.notna(row["pct_chg"]) else None,
                float(row["vol"]) if pd.notna(row["vol"]) else None,
                float(row["amount"]) if pd.notna(row["amount"]) else None,
            ))
            ins += 1
        conn.commit()
        new_max = conn.execute(
            "SELECT MAX(trade_date) FROM etf_daily WHERE ts_code=?", (code,)
        ).fetchone()[0]
        total += ins
        ok = "✅已对齐" if new_max == fr else "⚠️仍落后"
        print(f"    新增 {ins} 条 → 本地最新={new_max} {ok}")
        if new_max != fr:
            print(f"    [提示] 该标的最新仅到 {new_max}（tushare 可能尚未发布更晚数据）")

    print(f"\n合计新增 {total} 条。\n最终核对（应全部 == 前沿 {fr}）:")
    all_ok = True
    for code, name in STALE:
        m = conn.execute(
            "SELECT MAX(trade_date) FROM etf_daily WHERE ts_code=?", (code,)
        ).fetchone()[0]
        flag = "✅" if m == fr else "❌"
        if m != fr:
            all_ok = False
        print(f"  {code:12} {name:10} 最新={m} {flag}")
    conn.close()
    print("\n结果:", "全部对齐 ✅" if all_ok else "存在落后标的，见上 ⚠️")


if __name__ == "__main__":
    main()
