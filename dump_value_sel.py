# -*- coding: utf-8 -*-
# 仅 dump 价值选股(value, pobreak, hs300, top5) 每年入选清单，验证选股真实性。
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from loguru import logger
    logger.remove()
except Exception:
    pass

import sqlite3, pandas as pd
import run_dogs_annual as M
from src.value_stock_selector import select_value_stocks

DB = "D:/tu-shareData/astock_daily.db"
def prev_td(td):
    conn = sqlite3.connect(DB)
    r = pd.read_sql_query("SELECT MAX(trade_date) FROM daily WHERE trade_date < ?", conn, params=(td,))
    conn.close()
    return str(r.iloc[0, 0]) if r.iloc[0, 0] else td

start, end = "20190301", "20260727"
tds = M.get_trade_dates(start, end)
yf = M.get_first_trading_days(tds)

for y in sorted(yf):
    td = yf[y]
    prev = prev_td(td)
    try:
        sel = select_value_stocks(prev, top_n=5, stock_pool="hs300", mode="pobreak")
    except Exception as e:
        print(f"{y} ERROR {e}")
        continue
    if sel is None or len(sel) == 0:
        print(f"{y} (选股日 {prev}) -> 空选")
        continue
    codes = "、".join(f"{r.ts_code}({r['name']},PB={r.pb:.2f},PE={r.pe_ttm:.1f},ROE={r.roe if r.roe==r.roe else 'NA'})"
                      for _, r in sel.iterrows())
    print(f"{y} (选股日 {prev}) 选 {len(sel)} 只: {codes}")
