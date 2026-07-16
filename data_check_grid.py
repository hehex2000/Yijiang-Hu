"""独立核查：515800 中证800ETF 的裸价 / 前复权 / 指数口径一致性。
不依赖 pandas，纯标准库 sqlite3，避免环境差异。"""
import sqlite3, os

BASE = r"C:\Users\99395\WorkBuddy\multi_factor_selection"
DATA_DB = r"D:\tu-shareData\astock_daily.db"
REL_DB = os.path.join(BASE, "data", "tu-sharedata", "astock_daily.db")

# 复现 run_monthly_rebalance.get_conn 的库选择逻辑
db = DATA_DB if os.path.exists(DATA_DB) else REL_DB
print("=" * 60)
print(f"选用数据库: {db}  存在: {os.path.exists(db)}")
print("=" * 60)

conn = sqlite3.connect(db)
def q(sql, params=()):
    return conn.execute(sql, params).fetchall()

ts = "515800.SH"
idx_code = "000906.SH"
start, end = "20260103", "20260703"

# 1) etf_daily 裸价
raw = q("SELECT trade_date, close FROM etf_daily WHERE ts_code=? AND trade_date>=? AND trade_date<=? ORDER BY trade_date", (ts, start, end))
print(f"\n[1] etf_daily 裸价 {ts}: 行数={len(raw)}")
if raw:
    p0, p1 = raw[0][1], raw[-1][1]
    print(f"    首 {raw[0][0]}={p0}  末 {raw[-1][0]}={p1}")
    print(f"    裸价(未复权)回报: {(p1/p0-1)*100:+.2f}%")

# 2) etf_adj_factor
adj = q("SELECT trade_date, adj_factor FROM etf_adj_factor WHERE ts_code=? AND trade_date>=? AND trade_date<=? ORDER BY trade_date", (ts, start, end))
print(f"\n[2] etf_adj_factor {ts}: 行数={len(adj)}")
if adj:
    af0, af1 = adj[0][1], adj[-1][1]
    afmin = min(a[1] for a in adj); afmax = max(a[1] for a in adj)
    # 复现 _load_etf_adjusted：bfill/ffill 后 base=末值
    af = [a[1] for a in adj]
    # 简单 bfill/ffill（两端填充）
    seq = list(af)
    for i in range(len(seq)):
        if seq[i] is None or seq[i] == 0:
            j = i
            while j < len(seq) and (seq[j] is None or seq[j] == 0): j += 1
            fill = seq[j] if j < len(seq) else (seq[i-1] if i>0 else 1)
            for k in range(i, min(j, len(seq))): seq[k] = fill
    base = seq[-1]
    pf0 = p0 * seq[0] / base
    pf1 = p1 * seq[-1] / base
    print(f"    首因子={af0} 末因子={af1} min={afmin} max={afmax}")
    print(f"    前复权首价={pf0:.3f} 前复权末价={pf1:.3f}")
    print(f"    前复权回报(=run_grid口径): {(pf1/pf0-1)*100:+.2f}%")
    print(f"    前复权相对裸价的缩放倍数(末因子/首因子)={base/seq[0]:.4f}")
else:
    print("    无数据 -> 回退裸价")

# 3) 指数 000906
idx = q("SELECT trade_date, close FROM index_daily WHERE ts_code=? AND trade_date>=? AND trade_date<=? ORDER BY trade_date", (idx_code, start, end))
print(f"\n[3] index_daily {idx_code}: 行数={len(idx)}")
if idx:
    i0, i1 = idx[0][1], idx[-1][1]
    print(f"    首 {idx[0][0]}={i0}  末 {idx[-1][0]}={i1}")
    print(f"    指数价格回报: {(i1/i0-1)*100:+.2f}%")

# 4) 差异分解
print("\n[4] 差异分解（以 run_grid 的 idx_return=前复权回报 为基准）")
if adj and raw and idx:
    grid_ret = (pf1/pf0-1)*100
    raw_ret = (p1/p0-1)*100
    idx_ret = (i1/i0-1)*100
    print(f"    网格基准对比里展示的'买入持有' = 前复权回报 = {grid_ret:+.2f}%")
    print(f"    同区间裸价回报(未复权)        = {raw_ret:+.2f}%")
    print(f"    中证800指数价格回报(000906)   = {idx_ret:+.2f}%")
    print(f"    前复权 - 裸价 = {grid_ret-raw_ret:+.2f}%  <- 这被 run_grid 标为'股息贡献'")
    print(f"    前复权 - 指数 = {grid_ret-idx_ret:+.2f}%  <- 基准对比的'超额收益'")
    print(f"    指数 - 裸价   = {idx_ret-raw_ret:+.2f}%  <- 指数 vs ETF裸价(应≈0，因跟踪同一指数)")

conn.close()
