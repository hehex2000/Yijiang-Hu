"""拉取 31 个申万一级行业指数的真实日线，落库 sw_l1_index_daily。
用于：① 基准=31行业指数等权（对标视频"28行业等权"） ② 因子信号源用真实指数数据。
"""
import sqlite3, time
import pandas as pd
import tushare as ts
import config

DB = config.DATA["local_db_path"]
ts.set_token(config.DATA["tushare_token"])
pro = ts.pro_api()

SW_L1 = [
    "801010","801030","801040","801050","801080","801110","801120","801130",
    "801140","801150","801160","801170","801180","801200","801210","801230",
    "801710","801720","801730","801740","801750","801760","801770","801780",
    "801790","801880","801890","801950","801960","801970","801980",
]
START, END = "20150101", "20260831"

frames = []
seen_cols = None
for c in SW_L1:
    code = c + ".SI"
    try:
        d = pro.index_daily(ts_code=code, start_date=START, end_date=END)
    except Exception as e:
        print(f"  [ERR] {code}: {e}")
        time.sleep(1)
        continue
    if d is None or len(d) == 0:
        print(f"  [skip] {code}: 0 行")
        continue
    if seen_cols is None:
        seen_cols = list(d.columns)
        print("index_daily 字段:", seen_cols)
    frames.append(d)
    print(f"  [ok] {code}: {len(d)} 行, {d['trade_date'].min()}~{d['trade_date'].max()}")
    time.sleep(0.05)

if not frames:
    raise SystemExit("没有任何数据拉到，终止")

alld = pd.concat(frames, ignore_index=True)
print(f"\n合计: {len(alld):,} 行 / {alld['ts_code'].nunique()} 个指数")

con = sqlite3.connect(DB)
con.execute("DROP TABLE IF EXISTS sw_l1_index_daily")
# 按实际字段建表（不写死列名）
cols = [f'"{c}"' for c in seen_cols]
con.execute(f"CREATE TABLE sw_l1_index_daily ({', '.join(cols)})")
alld.to_sql("sw_l1_index_daily", con, if_exists="append", index=False)
# 校验
chk = pd.read_sql("SELECT ts_code, COUNT(*) n, MIN(trade_date) a, MAX(trade_date) b FROM sw_l1_index_daily GROUP BY ts_code ORDER BY ts_code", con)
print("落库校验:")
print(chk.to_string())
con.close()
print("[OK] sw_l1_index_daily 已落库")
