# -*- coding: utf-8 -*-
"""
补齐 沪深300(000300.SH) 全历史日线到 index_daily（平台现有数据仅 2019 起）。
数据源：akshare 新浪源 stock_zh_index_daily(symbol="sh000300")，覆盖 2002-至今。
写入：INSERT OR REPLACE（主键 ts_code,trade_date）幂等，不删已有 2019+ 数据。
用法：./venv_ml/Scripts/python.exe download_hs300.py
"""
import sqlite3
import akshare as ak
import pandas as pd

DB = r'D:/tu-shareData/astock_daily.db'
TS = '000300.SH'


def main():
    print(f"抓取 {TS} 全历史（新浪源）...")
    df = ak.stock_zh_index_daily(symbol='sh000300')
    if len(df) == 0:
        print("[ERR] 抓取为空")
        return
    df['trade_date'] = df['date'].astype(str).str.replace('-', '')
    df['ts_code'] = TS
    df['vol'] = df['volume']
    cols = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close',
            'pre_close', 'change', 'pct_chg', 'vol', 'amount']
    out = pd.DataFrame({
        'ts_code': df['ts_code'],
        'trade_date': df['trade_date'].astype(int),
        'open': df['open'].astype(float),
        'high': df['high'].astype(float),
        'low': df['low'].astype(float),
        'close': df['close'].astype(float),
        'pre_close': float('nan'),
        'change': float('nan'),
        'pct_chg': float('nan'),
        'vol': df['vol'].astype(float),
        'amount': float('nan'),
    })[cols]

    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    n_before = cur.execute(
        "SELECT COUNT(*) FROM index_daily WHERE ts_code=?", (TS,)).fetchone()[0]
    rows = [tuple(r) for r in out.itertuples(index=False, name=None)]
    cur.executemany(
        "INSERT OR REPLACE INTO index_daily"
        "(ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    n_after = cur.execute(
        "SELECT COUNT(*) FROM index_daily WHERE ts_code=?", (TS,)).fetchone()[0]
    rg = cur.execute(
        "SELECT MIN(trade_date),MAX(trade_date) FROM index_daily WHERE ts_code=?",
        (TS,)).fetchone()
    conn.close()
    print(f"[OK] {TS} 写入 {len(rows)} 行 | 库内 {n_before}→{n_after} 行 | 区间 {rg[0]}~{rg[1]}")


if __name__ == '__main__':
    main()
