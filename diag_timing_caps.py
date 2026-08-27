import sys, datetime, collections
import sqlite3, pandas as pd
sys.path.insert(0, r"C:\Users\99395\WorkBuddy\multi_factor_selection")
from market_timing_overlay import compute_breadth_oscillator, position_cap
from run_monthly_rebalance import get_monthly_5th_trading_days

DB = "D:/tu-shareData/astock_daily.db"

def load_close(start, end):
    sy, sm, sd = int(start[:4]), int(start[4:6]), int(start[6:8])
    lo = (datetime.date(sy, sm, sd) - datetime.timedelta(days=300)).strftime("%Y%m%d")
    con = sqlite3.connect(f"file:{DB}?immutable=1", uri=True)
    try:
        df = pd.read_sql("SELECT trade_date,ts_code,close FROM daily WHERE trade_date>=? AND trade_date<=?",
                         con, params=(lo, end))
    finally:
        con.close()
    if df.empty:
        return pd.DataFrame()
    px = df.pivot(index="trade_date", columns="ts_code", values="close").sort_index()
    px.index = pd.to_numeric(px.index).astype(int)
    return px

def get_trade_dates(start, end):
    con = sqlite3.connect(f"file:{DB}?immutable=1", uri=True)
    try:
        ds = pd.read_sql("SELECT trade_date FROM daily WHERE trade_date>=? AND trade_date<=? "
                         "GROUP BY trade_date ORDER BY trade_date",
                         con, params=(start, end))["trade_date"].tolist()
    finally:
        con.close()
    return [int(x) for x in ds]

def caps_for(start, end, boil=80, ice=20, floor=0.0):
    tdates = get_trade_dates(start, end)
    rbs = get_monthly_5th_trading_days(tdates)
    rbs = [d for d in rbs if start <= str(d) <= end]
    px = load_close(start, end)
    osc_all = compute_breadth_oscillator(px).dropna()
    caps = {}
    miss = 0
    for d in sorted(rbs):
        if d in osc_all.index:
            caps[d] = float(position_cap(float(osc_all[d]), boil, ice, floor))
        else:
            caps[d] = 1.0
            miss += 1
    return caps, osc_all, miss

# ── 先读一次 ETF 窗口 osc，再扫不同 ice 标定看 cap 分布 ──
print("=" * 70)
print("扫描 ice 标定（ETF窗口 2020-2025，boil=80，floor=0）")
tdates = get_trade_dates("20200101", "20251231")
rbs = [d for d in get_monthly_5th_trading_days(tdates) if "20200101" <= str(d) <= "20251231"]
px = load_close("20200101", "20251231")
osc_all = compute_breadth_oscillator(px).dropna()
for ice in [20, 40, 50, 55, 60, 65, 70]:
    caps = {d: float(position_cap(float(osc_all[d]), 80, ice, 0.0)) if d in osc_all.index else 1.0
            for d in rbs}
    n_full = sum(1 for c in caps.values() if c >= 0.999)
    n_zero = sum(1 for c in caps.values() if c <= 1e-9)
    n_part = len(caps) - n_full - n_zero
    avg = sum(caps.values()) / len(caps)
    print(f"  ice={ice:>2}: 满仓={n_full:>2} 半仓={n_part:>2} 清仓={n_zero:>2}  平均cap={avg:.3f}")

print()
print("=" * 70)
print("扫描 ice 标定（Kara全样本 2014-2026，boil=80，floor=0）")
tdates_k = get_trade_dates("20140101", "20261231")
rbs_k = [d for d in get_monthly_5th_trading_days(tdates_k) if "20140101" <= str(d) <= "20261231"]
px_k = load_close("20140101", "20261231")
osc_k = compute_breadth_oscillator(px_k).dropna()
for ice in [20, 40, 50, 55, 60, 65, 70]:
    caps = {d: float(position_cap(float(osc_k[d]), 80, ice, 0.0)) if d in osc_k.index else 1.0
            for d in rbs_k}
    n_full = sum(1 for c in caps.values() if c >= 0.999)
    n_zero = sum(1 for c in caps.values() if c <= 1e-9)
    n_part = len(caps) - n_full - n_zero
    avg = sum(caps.values()) / len(caps)
    print(f"  ice={ice:>2}: 满仓={n_full:>2} 半仓={n_part:>2} 清仓={n_zero:>2}  平均cap={avg:.3f}")

print()
print("=" * 70)
for label, (s, e) in [("ETF窗口 2020-2025", ("20200101", "20251231")),
                       ("Kara全样本 2014-2026", ("20140101", "20261231"))]:
    print(label)
    caps, osc_all, miss = caps_for(s, e)
    n_full = sum(1 for c in caps.values() if c >= 0.999)
    n_zero = sum(1 for c in caps.values() if c <= 1e-9)
    n_part = len(caps) - n_full - n_zero
    avg = sum(caps.values()) / len(caps)
    print(f"osc 范围: {osc_all.min():.1f}~{osc_all.max():.1f}  调仓月数: {len(caps)}  对齐miss: {miss}")
    print(f"cap==1(满仓): {n_full}  0<cap<1(半仓): {n_part}  cap==0(清仓): {n_zero}  平均cap: {avg:.3f}")
