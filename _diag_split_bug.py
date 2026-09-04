# -*- coding: utf-8 -*-
"""验证：raw 口径回测是否处理「送转股」导致的持股数变化

假设：回测持仓 shares 固定，从不因送转而增加 → raw 模式下每次送转都凭空蒸发
      (1 - 1/送转比例) 的持仓市值。hfq 模式因 hfq 价不变而天然免疫。

验证方法：取当日因子跳涨的股票，比较 daily.close 跌幅 vs 因子涨幅。
         若 close 跌幅 ≈ 1 - 1/因子比 → 确为送转 → raw 模式必受损。
"""
import sqlite3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_monthly_rebalance as m
import pandas as pd

con0 = m.get_conn()
db = con0.execute("PRAGMA database_list").fetchone()[2]
con0.close()
con = sqlite3.connect(db)

TARGET = "20210610"
PREV = "20210609"

# 当日因子跳涨 >3% 的股票 + 其 close 变化
q = """
SELECT d.ts_code, d.trade_date, d.close
FROM daily d WHERE d.trade_date IN (?,?) AND d.ts_code IN (
    SELECT ts_code FROM adj_factor WHERE trade_date=?
)
"""
px = pd.read_sql_query(q, con, params=(PREV, TARGET, TARGET))
adj = pd.read_sql_query(
    "SELECT ts_code, trade_date, adj_factor FROM adj_factor WHERE trade_date IN (?,?)",
    con, params=(PREV, TARGET))

p = px.pivot_table(index="ts_code", columns="trade_date", values="close")
a = adj.pivot_table(index="ts_code", columns="trade_date", values="adj_factor")
j = p.join(a, lsuffix="_px", rsuffix="_adj").dropna()
j = j.rename(columns={f"{PREV}_px": "px0", f"{TARGET}_px": "px1",
                      f"{PREV}_adj": "f0", f"{TARGET}_adj": "f1"})
j["fchg"] = j["f1"] / j["f0"] - 1
j["pxchg"] = j["px1"] / j["px0"] - 1
# 送转的特征：价格跌幅 ≈ 1 - 1/因子比（纯送转），分红则跌幅远小于此
j["送转理论跌幅"] = 1 - 1 / (j["f1"] / j["f0"])
j["偏差"] = (j["pxchg"] + j["送转理论跌幅"]).abs()

big = j[j["fchg"] > 0.03].sort_values("fchg", ascending=False)
print(f"当日因子跳涨 >3% 共 {len(big)} 只 —— 价格跌幅 vs 送转理论跌幅")
print(big[["px0", "px1", "f0", "f1", "fchg", "pxchg", "送转理论跌幅", "偏差"]].head(12).to_string(
    index=False, float_format=lambda x: f"{x:,.4f}"))

n_split = int((big["偏差"] < 0.02).sum())
print(f"\n其中 {n_split}/{len(big)} 只的价格跌幅与「纯送转理论跌幅」偏差 <2pp → 确认为送转/拆股事件")

# ── 直接演示 raw 模式的凭空亏损 ──
print()
print("=" * 74)
print("raw 模式送转损失演示（持仓 shares 不变，价格按除权后下跌）")
print("=" * 74)
print(f"{'代码':<12}{'送转比例':>10}{'持仓市值损失':>14}{'hfq价变动':>12}")
tot = 0.0
for code, r in big.head(8).iterrows():
    ratio = r["f1"] / r["f0"]
    loss = 1 - 1 / ratio
    hfq_chg = r["px1"] * r["f1"] / (r["px0"] * r["f0"]) - 1
    print(f"{code:<12}{ratio:>10.4f}{loss*100:>13.2f}%{hfq_chg*100:>11.2f}%")

con.close()

print()
print("=" * 74)
print("结论判断")
print("=" * 74)
print("若上表「hfq价变动」接近 0 而「持仓市值损失」显著为负 → 证明 raw 口径回测")
print("在每次送转事件上凭空蒸发持仓市值，且 shares 从不随送转增加。")
print("这是比「漏计分红」更严重的 bug：送转单次可致 -17%~-45%。")
