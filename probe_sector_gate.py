# -*- coding: utf-8 -*-
"""聚合验证：对完整 20 只混合池，在关键日期统计三态分布，检测是否过度过滤(全drop=变货币基金)。"""
import numpy as np, pandas as pd, sqlite3, sys
sys.path.insert(0, ".")
from sector_state_machine import classify_state, STATE_CN

DB = "D:/tu-shareData/astock_daily.db"

# 混合池 20 只（与 run_etf_rotation_v6_merged.ETF_UNIVERSE 一致）
POOL = ["510300.SH","510050.SH","515800.SH","510980.SH","510500.SH",
        "512100.SH","159915.SZ","159949.SZ","588000.SH","512480.SH",
        "515030.SH","512010.SH","159928.SZ","512880.SH","159920.SZ",
        "513100.SH","518880.SH","501018.SH","511010.SH","511990.SH"]

def table_of(code):
    if code.endswith('.SH') and len(code) == 9 and code[:2] in ('51','58'):
        return 'etf_daily'
    if code.endswith('.SZ') and len(code) == 9 and code[:3] == '159':
        return 'etf_daily'
    return 'index_daily'

def hist(code, date, limit=90):
    tbl = table_of(code)
    con = sqlite3.connect(f"file:{DB}?immutable=1", uri=True)
    df = pd.read_sql_query(
        f"SELECT close FROM {tbl} WHERE ts_code=? AND trade_date<=? "
        f"ORDER BY trade_date DESC LIMIT ?", con, params=(code, date, limit))
    con.close()
    if len(df) < 90:
        return None
    return np.array(df['close'].values[::-1], dtype=float)  # 升序

dates = ["20190131","20191231","20200331","20210131","20220131",
         "20240131","20240628","20240930","20250331"]

print("=== 板块三态闸门 · 20只混合池聚合分布（keep=右侧+趋势加速, drop=加速见底）===")
print(f"{'日期':<10}{'池中数':>6}{'有价':>6}{'keep':>6}{'drop':>6}{'drop%':>7}")
for d in dates:
    have, keep, drop = 0, 0, 0
    for code in POOL:
        asc = hist(code, d, 90)
        if asc is None:
            continue
        have += 1
        st, _ = classify_state(asc)
        if st in ("RIGHT_TREND","TREND_ACCEL"):
            keep += 1
        else:
            drop += 1
    dpct = (drop / have * 100) if have else 0
    print(f"{d:<10}{len(POOL):>6}{have:>6}{keep:>6}{drop:>6}{dpct:>6.0f}%")
print("\n解读：drop% 过高(如>80%)→过度过滤变货币基金；过低(<20%)→闸门形同虚设。")
print("      目标：熊市/早期恢复 drop 高(避险)，牛市 drop 低(吃趋势)。")
