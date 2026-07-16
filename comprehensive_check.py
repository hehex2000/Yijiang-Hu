"""全面复核网格菜单全部 12 个标的的数据口径：
  - ETF(510300/510500/515800): 裸价 vs 前复权 vs 基准指数，确认无异常缩放、股息贡献合理
  - 指数(4-12): 序列连续性，确认无单日异常断点(>±15%)"""
import sqlite3, os

BASE = r"C:\Users\99395\WorkBuddy\multi_factor_selection"
DATA_DB = r"D:\tu-shareData\astock_daily.db"
REL_DB = os.path.join(BASE, "data", "tu-sharedata", "astock_daily.db")
db = DATA_DB if os.path.exists(DATA_DB) else REL_DB
print(f"数据库: {db}\n")

conn = sqlite3.connect(db)
def get_raw(ts, s, e):
    return conn.execute("SELECT trade_date,close FROM etf_daily WHERE ts_code=? AND trade_date>=? AND trade_date<=? ORDER BY trade_date", (ts, s, e)).fetchall()
def get_adj(ts, s, e):
    return conn.execute("SELECT trade_date,adj_factor FROM etf_adj_factor WHERE ts_code=? AND trade_date>=? AND trade_date<=? ORDER BY trade_date", (ts, s, e)).fetchall()
def get_idx(code, s, e):
    return conn.execute("SELECT trade_date,close FROM index_daily WHERE ts_code=? AND trade_date>=? AND trade_date<=? ORDER BY trade_date", (code, s, e)).fetchall()

s, e = "20260102", "20260703"
etfs = {"510300.SH": ("沪深300ETF", "000300.SH"),
        "510500.SH": ("中证500ETF", "000905.SH"),
        "515800.SH": ("中证800ETF", "000906.SH")}

def ffill(seq):
    out = list(seq)
    for i in range(len(out)):
        if out[i] is None or out[i] == 0:
            j = i
            while j < len(out) and (out[j] is None or out[j] == 0): j += 1
            fill = out[j] if j < len(out) else (out[i-1] if i > 0 else 1)
            for k in range(i, min(j, len(out))): out[k] = fill
    return out

print("===== ETF 标的：裸价 / 前复权 / 基准指数 三方核对 =====")
for ts, (name, idx) in etfs.items():
    raw = get_raw(ts, s, e); adj = get_adj(ts, s, e); idxr = get_idx(idx, s, e)
    if not raw:
        print(f"{ts} {name}: 无 etf_daily 数据 ⚠️"); continue
    p0, p1 = raw[0][1], raw[-1][1]; raw_ret = (p1/p0-1)*100
    if adj:
        seq = ffill([a[1] for a in adj]); base = seq[-1]
        pf_ret = ((p1*seq[-1]/base) / (p0*seq[0]/base) - 1) * 100
        af_ratio = seq[-1] / seq[0]
    else:
        pf_ret, af_ratio = raw_ret, 1.0
    idx_ret = (idxr[-1][1]/idxr[0][1]-1)*100 if idxr else None
    print(f"\n{ts} {name}:")
    print(f"  裸价回报={raw_ret:+.2f}%  前复权回报={pf_ret:+.2f}%  股息贡献(前-裸)={pf_ret-raw_ret:+.2f}%")
    print(f"  adj_factor 末/首={af_ratio:.4f}  {'（区间内无复权事件）' if abs(af_ratio-1)<1e-6 else '（区间内发生过分红/拆分）'}")
    if idx_ret is not None:
        print(f"  基准指数 {idx}={idx_ret:+.2f}%  ETF前复权-指数={pf_ret-idx_ret:+.2f}%")
    else:
        print(f"  基准指数 {idx}=无数据 ⚠️")

print("\n===== 指数标的：序列连续性（最大单日涨跌幅应<±15%，否则疑数据断点）=====")
indexes = ["000001.SH","399001.SZ","000016.SH","000852.SH","000688.SH","000698.SH","399006.SZ","399673.SZ","932000.SH"]
for code in indexes:
    r = get_idx(code, s, e)
    if not r:
        print(f"{code}: 无数据 ⚠️"); continue
    c = [x[1] for x in r]
    rets = [c[i]/c[i-1]-1 for i in range(1, len(c))]
    mu, md = max(rets)*100, min(rets)*100
    flag = "  ⚠️ 异常单日波动!" if (mu > 15 or md < -15) else ""
    print(f"{code}: 行数={len(r)} 区间回报={(c[-1]/c[0]-1)*100:+.2f}%  最大日涨={mu:.2f}% 最大日跌={md:.2f}%{flag}")
conn.close()
