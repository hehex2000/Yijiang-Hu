import sqlite3, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_monthly_rebalance import get_conn

conn = get_conn()
cur = conn.cursor()

S, E = "20220102", "20251231"
code = "510300.SH"

# 未复权价（etf_daily）
rows = cur.execute(
    "SELECT trade_date, close FROM etf_daily WHERE ts_code=? AND trade_date>=? AND trade_date<=? ORDER BY trade_date",
    (code, S, E)).fetchall()
first_dt, first_close = rows[0]
last_dt, last_close = rows[-1]
print(f"未复权(裸价): {first_dt} 收={first_close}  ->  {last_dt} 收={last_close}")
nominal_ret = last_close / first_close - 1
print(f"  裸价回报(买入持有·价格口径): {nominal_ret:+.2%}")

# 复权因子
af = cur.execute(
    "SELECT trade_date, adj_factor FROM etf_adj_factor WHERE ts_code=? AND trade_date>=? AND trade_date<=? ORDER BY trade_date",
    (code, S, E)).fetchall()
f_dt, f0 = af[0]
l_dt, f1 = af[-1]
print(f"复权因子: {f_dt}={f0}  ->  {l_dt}={f1}  因子比(末/初)={f1/f0:.4f}")

# 前复权口径的首日价 = 裸价 × f0/f1 ; 末日价=裸价(锚定最新)
adj_first = first_close * f0 / f1
total_ret = last_close / adj_first - 1
print(f"前复权首日价={adj_first:.4f}, 末日价={last_close}")
print(f"  前复权回报(买入持有·总回报=价格+股息): {total_ret:+.2%}")

# 股息贡献
div_contrib = total_ret - nominal_ret
print(f"  其中股息贡献 ≈ {div_contrib:+.2%}")

# 策略超额(用前复权口径比较)
strat = 0.0407
print(f"\n策略回报 +4.07% (前复权/总回报口径)")
print(f"  vs ETF总回报 +{total_ret:.2%}  -> 真实超额 = {strat-total_ret:+.2%}")
print(f"  vs ETF裸价  {nominal_ret:+.2%}  -> 表面'超额' = {strat-nominal_ret:+.2%}  (其中股息 {div_contrib:+.2%} 二者共有)")
conn.close()
