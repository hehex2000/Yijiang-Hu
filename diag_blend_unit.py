"""单元验证 apply_piotroski_blend：候选>5 时 w 改变排序，且 fscore 列存在。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from src.value_stock_selector import apply_piotroski_blend

TD = "20240508"
# 取该日 zz800 时点快照前 12 只作候选
import sqlite3
con = sqlite3.connect(r"D:/tu-shareData/astock_daily.db")
snap = pd.read_sql_query(
    "SELECT MAX(trade_date) AS d FROM index_constituent WHERE index_code='000906.SH' "
    "AND CAST(trade_date AS INTEGER)<=20240508", con)
code = snap.iloc[0, 0]
pool = pd.read_sql_query(
    "SELECT ts_code FROM index_constituent WHERE index_code='000906.SH' AND trade_date=?",
    con, params=(code,))
con.close()
cands = pool["ts_code"].head(12).tolist()
chosen = pd.DataFrame({"ts_code": cands, "pb": [0.5]*len(cands)})

w0 = apply_piotroski_blend(chosen, TD, w=0.0)
w50 = apply_piotroski_blend(chosen, TD, w=0.5)
w100 = apply_piotroski_blend(chosen, TD, w=1.0)

print("候选数:", len(cands))
print("w=0.0 序:", w0[0]["ts_code"].tolist(), "fscore=", w0[0]["fscore"].tolist())
print("w=0.5 序:", w50[0]["ts_code"].tolist(), "fscore=", w50[0]["fscore"].tolist())
print("w=1.0 序:", w100[0]["ts_code"].tolist(), "fscore=", w100[0]["fscore"].tolist())

same05 = w0[0]["ts_code"].tolist() == w50[0]["ts_code"].tolist()
same10 = w0[0]["ts_code"].tolist() == w100[0]["ts_code"].tolist()
# w=1 应与 w=0 不同(纯F排序)，除非 fscore 全相等
fss = w100[0]["fscore"].tolist()
print("\n[校验] w=0.5 重排生效:", "PASS" if not same05 else "FAIL")
print("[校验] w=1.0 与 w=0 不同:", "PASS" if (not same10 and len(set(fss)) > 1) else "FAIL/全相等")
print("[校验] fscore 列存在:", "PASS" if "fscore" in w50[0].columns else "FAIL")
print("[校验] 不空仓(保留全部候选):", "PASS" if len(w50[0]) == len(cands) else "FAIL")
