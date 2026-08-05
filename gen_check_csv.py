# -*- coding: utf-8 -*-
# 生成"价值选股年度回测真实性核查"CSV：逐年选股 + 独立持有收益 + 报告收益 + 指数对照
import csv, sqlite3, sys, os, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from loguru import logger; logger.remove()
except Exception: pass
import run_dogs_annual as M
from src.value_stock_selector import select_value_stocks

DB = "D:/tu-shareData/astock_daily.db"
def prev_td(td):
    c = sqlite3.connect(DB)
    r = pd.read_sql_query("SELECT MAX(trade_date) FROM daily WHERE trade_date < ?", c, params=(td,))
    c.close(); return str(r.iloc[0, 0]) if r.iloc[0, 0] else td

start, end = "20190301", "20260727"
tds = M.get_trade_dates(start, end)
yf = M.get_first_trading_days(tds)
year_last = {}
for td in tds:
    year_last[td[:4]] = td

report = {"2019":-6.97,"2020":-0.01,"2021":45.68,"2022":7.79,"2023":17.92,"2024":32.14,"2025":53.40,"2026":2.23}
rows = []
for y in sorted(yf):
    td0, td1 = yf[y], year_last[y]
    sel = select_value_stocks(prev_td(td0), top_n=5, stock_pool="hs300", mode="pobreak")
    if sel is None or len(sel) == 0:
        rows.append([y, 0, "空选(保持持仓)", 0, 0, report.get(y, 0), 0]); continue
    N = len(sel)
    names = "、".join(f"{r['name']}" for _, r in sel.iterrows())
    rets = []
    for _, r in sel.iterrows():
        bo = M.get_hfq_price(r.ts_code, td0, "open"); bc = M.get_hfq_price(r.ts_code, td1, "close")
        if bo and bc and bo > 0: rets.append(bc/bo - 1)
    mean = sum(rets)/len(rets) if rets else 0
    port = 0.196 * N * mean          # 复现 run_dogs_annual 的"选N只只投 N/5 仓位"现金闲置
    rep = report[y]
    rows.append([y, N, names, round(mean*100, 2), round(port*100, 2), rep, round(port*100 - rep, 2)])

conn = sqlite3.connect(DB)
def idx(code):
    d = pd.read_sql_query(f"SELECT close FROM index_daily WHERE ts_code='{code}' AND trade_date BETWEEN '20190301' AND '20260727' ORDER BY trade_date", conn)
    return (d['close'].iloc[-1]/d['close'].iloc[0]-1)*100 if len(d) > 1 else None
rows.append(["对照", "", f"沪深300 +{idx('000300.SH'):.1f}% | 中证红利 +{idx('000922.SH'):.1f}% | 中证500 +{idx('000905.SH'):.1f}% (同期价格指数)", "", "", "", ""])

os.makedirs("data", exist_ok=True)
out = "data/价值选股年度回测真实性核查.csv"
with open(out, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["年份","选股数","选股清单","独立持有等权收益%","组合收益%(现金拖累)","报告年度收益%","差额%"])
    for r in rows: w.writerow(r)
print("written ->", out)
