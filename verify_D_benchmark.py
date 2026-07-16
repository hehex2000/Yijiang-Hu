import sqlite3, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_monthly_rebalance import get_conn
from show_index_etf_changes import _latest_etf_close, _query_index_series

conn = get_conn()
S, E = "20220102", "20251231"
etf, idx = "510300.SH", "000300.SH"

etf_end = _latest_etf_close(conn, etf, E)
idx_rows = _query_index_series(conn, idx, S, E)
c0, c1 = idx_rows[0][1], idx_rows[-1][1]
ratio = c1 / c0
d_pct = (ratio - 1.0) * 100.0

print("=== [D] 算法核心 (compute_one 情形A) ===")
print("etf_end (锚定现价) =", etf_end)
print("index 000300:", c0, "->", c1, " ratio=", round(ratio, 4))
print("D_reported_pct = (ratio-1)*100 =", round(d_pct, 2), "  <-- 沪深300[指数]价格回报")

rows = conn.execute("SELECT close FROM etf_daily WHERE ts_code=? AND trade_date>=? AND trade_date<=? ORDER BY trade_date",(etf,S,E)).fetchall()
af = conn.execute("SELECT adj_factor FROM etf_adj_factor WHERE ts_code=? AND trade_date>=? AND trade_date<=? ORDER BY trade_date",(etf,S,E)).fetchall()
fc, lc = rows[0][0], rows[-1][0]
a0, a1 = af[0][0], af[-1][0]
etf_total = lc / (fc * a0 / a1) - 1

print("ETF total return (前复权) =", round(etf_total*100, 2), "%")
print("Strategy = +4.07%")
print("REAL excess (vs ETF total) =", round((0.0407 - etf_total)*100, 2), "%")
print("SURFACE excess (vs [D] index price) =", round((0.0407 - d_pct/100)*100, 2), "%  [inflated by dividends + tracking]")
conn.close()
