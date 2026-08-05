# -*- coding: utf-8 -*-
# 验证：每年选中的股票按等权持有一年(年初开盘买->年末收盘卖, hfq后复权) 的真实收益，
# 并复现 run_dogs_annual 的"选 N 只只投 N/5 仓位"现金闲置逻辑，对比报告年度收益。
import sys, os, sqlite3, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from loguru import logger; logger.remove()
except Exception: pass
import run_dogs_annual as M
from src.value_stock_selector import select_value_stocks

DB = "D:/tu-shareData/astock_daily.db"
def prev_td(td):
    conn = sqlite3.connect(DB)
    r = pd.read_sql_query("SELECT MAX(trade_date) FROM daily WHERE trade_date < ?", conn, params=(td,))
    conn.close(); return str(r.iloc[0,0]) if r.iloc[0,0] else td
def hfq(code, date, pt):
    return M.get_hfq_price(code, date, pt)   # 全局后复权价

start, end = "20190301", "20260727"
tds = M.get_trade_dates(start, end)
yf = M.get_first_trading_days(tds)
# 每年最后一个交易日（覆盖赋值，保留最后一条）
year_last = {}
for td in tds:
    year_last[td[:4]] = td
years = sorted(yf.keys())

report = {"2019":-6.97,"2020":-0.01,"2021":45.68,"2022":7.79,"2023":17.92,"2024":32.14,"2025":53.40,"2026":2.23}

print(f"{'年':<6}{'N':>3}{'等权股收益':>12}{'现金拖累后组合':>16}{'报告年度':>12}{'差'}")
cum = 1.0
for y in years:
    td0 = yf[y]; td1 = year_last[y]
    sel = select_value_stocks(prev_td(td0), top_n=5, stock_pool="hs300", mode="pobreak")
    if sel is None or len(sel)==0:
        print(f"{y:<6}{0:>3}{'-- 空选(保持持仓)':>12}"); continue
    N = len(sel)
    stock_rets = []
    for _, r in sel.iterrows():
        bo = hfq(r.ts_code, td0, "open"); bc = hfq(r.ts_code, td1, "close")
        if bo and bc and bo>0:
            stock_rets.append(bc/bo - 1)
    if not stock_rets:
        print(f"{y:<6}{N:>3}{'价缺失':>12}"); continue
    mean_ret = sum(stock_rets)/len(stock_rets)
    # 复现现金闲置: 每只投 总资金*0.98/5, 选N只 -> 组合收益 = 0.196*N*mean_ret
    port = 0.196 * N * mean_ret
    rep = report.get(y, float('nan'))
    print(f"{y:<6}{N:>3}{mean_ret*100:>11.2f}%{port*100:>15.2f}%{rep:>11.2f}%{(port*100-rep):>+6.2f}")
    cum *= (1+port)
print(f"\n现金拖累后组合累计: {(cum-1)*100:.2f}%  (报告全程 +251.29%)")
