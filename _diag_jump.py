# -*- coding: utf-8 -*-
"""追查 20210610 单日比值跳变 +5.29% 的成因

若来自「送转股/大比例分红」→ 合法；
若来自「整股下单现金残留差异」或「持仓不一致」→ 需修复。
"""
import pandas as pd
import sqlite3
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

import run_monthly_rebalance as _m
con0 = _m.get_conn()
db = con0.execute("PRAGMA database_list").fetchone()[2]
con0.close()
con = sqlite3.connect(db)

TARGET = "20210610"
PREV = "20210609"

# 1) 当日 adj_factor 变动最大的股票
q = """
SELECT a.ts_code, a.trade_date, a.adj_factor
FROM adj_factor a
WHERE a.trade_date IN (?, ?)
"""
df = pd.read_sql_query(q, con, params=(PREV, TARGET))
print(f"adj_factor 在 {PREV}/{TARGET} 的行数: {len(df)}")

p = df.pivot_table(index="ts_code", columns="trade_date", values="adj_factor")
if len(p.columns) == 2:
    p["chg"] = p[TARGET] / p[PREV] - 1
    big = p[p["chg"] > 0.03].sort_values("chg", ascending=False)
    print(f"\n当日因子上调 >3% 的股票: {len(big)} 只")
    print(big.head(20).to_string())
else:
    print("  两日数据不齐，列出当日全部变动:")
    print(p.head(20).to_string())

# 2) 当日 adj_factor 有行的股票总数（判断是否缺行日）
n_target = pd.read_sql_query(
    "SELECT COUNT(DISTINCT ts_code) c FROM adj_factor WHERE trade_date=?", con, params=(TARGET,))["c"].iloc[0]
n_prev = pd.read_sql_query(
    "SELECT COUNT(DISTINCT ts_code) c FROM adj_factor WHERE trade_date=?", con, params=(PREV,))["c"].iloc[0]
print(f"\n{TARGET}: {n_target} 只有因子行 | {PREV}: {n_prev} 只")

# 3) 当日 NAV 上下文
BASE = "data/results/monthly_rebalance"
raw = pd.read_csv(f"{BASE}/backtest_20200102_20260723.csv")
hfq = pd.read_csv(f"{BASE}/backtest_hfq_20200102_20260723.csv")
m = raw.merge(hfq, on="date", suffixes=("_raw", "_hfq")).sort_values("date")
m["date"] = m["date"].astype(str)
m["ratio"] = m["value_hfq"] / m["value_raw"]
i = m.index[m["date"] == TARGET][0]
print(f"\nNAV 上下文 ({TARGET} 前后 3 天):")
print(m.iloc[i-3:i+4][["date", "value_raw", "value_hfq", "ratio"]].to_string(index=False))

# 4) 当日 raw NAV 涨跌 vs hfq NAV 涨跌
r_chg = m["value_raw"].pct_change()
h_chg = m["value_hfq"].pct_change()
print(f"\n当日 raw NAV 涨跌 {r_chg[i]*100:+.3f}%   hfq NAV 涨跌 {h_chg[i]*100:+.3f}%   "
      f"差 {(h_chg[i]-r_chg[i])*100:+.3f}pp")

# 5) 检查当日是否为调仓日附近 + 持仓快照
tr = pd.read_csv(f"{BASE}/trades_value_20200102_20260723.csv")
tr["date"] = tr["date"].astype(str)
near = tr[(tr["date"] >= "20210601") & (tr["date"] <= "20210620")]
print(f"\n2021-06-01~06-20 成交 {len(near)} 笔:")
print(near.to_string(index=False))

con.close()
