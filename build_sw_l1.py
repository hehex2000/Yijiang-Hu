"""
用硬编码的 31 个申万一级(SW2021)指数代码，从 tushare 拉真实名称+当前成分，
落库 stock_ind_sw_l1。比 keyword 代理和 endswith 启发式都干净。
"""
import sqlite3, time
import pandas as pd
import tushare as ts
import config

DB = config.DATA["local_db_path"]
ts.set_token(config.DATA["tushare_token"])
pro = ts.pro_api()

# 公认的申万一级(2021修订) 31 个指数代码
SW_L1 = [
    "801010","801030","801040","801050","801080","801110","801120","801130",
    "801140","801150","801160","801170","801180","801200","801210","801230",
    "801710","801720","801730","801740","801750","801760","801770","801780",
    "801790","801880","801890","801950","801960","801970","801980",
]

rows = []
names = {}
for c in SW_L1:
    code = c + ".SI"
    try:
        cl = pro.index_classify(index_code=code)
        nm = cl.iloc[0]["industry_name"] if cl is not None and len(cl) else None
    except Exception:
        nm = None
    if not nm:
        try:
            nm = pro.index_basic(ts_code=code).iloc[0]["name"]
        except Exception:
            nm = c
    names[code] = nm
    try:
        m = pro.index_member(index_code=code)
    except Exception as e:
        print(f"  [WARN] {code} member err: {e}")
        continue
    if m is None or len(m) == 0:
        print(f"  [skip] {code} {nm}: 0 成分")
        continue
    cur = m[m["out_date"].isnull()] if "out_date" in m.columns else m
    for _, s in cur.iterrows():
        rows.append((s["con_code"], code, nm))
    print(f"  [ok] {code} {nm}: {len(cur)} 只")
    time.sleep(0.05)

df = pd.DataFrame(rows, columns=["ts_code", "sw_code", "sw_name"]).drop_duplicates("ts_code")
print(f"\n映射完成: {len(df):,} 只个股 / {df['sw_name'].nunique()} 个申万一级行业")

con = sqlite3.connect(DB)
con.execute("DROP TABLE IF EXISTS stock_ind_sw_l1")
con.execute("CREATE TABLE stock_ind_sw_l1 (ts_code TEXT, sw_code TEXT, sw_name TEXT, PRIMARY KEY(ts_code))")
df.to_sql("stock_ind_sw_l1", con, if_exists="append", index=False)
daily = pd.read_sql("SELECT DISTINCT ts_code FROM daily", con)
cov = daily["ts_code"].isin(df["ts_code"]).mean()
print(f"本地 daily 个股={len(daily):,}, 命中SW映射={cov*100:.1f}%")
print("行业清单:", sorted(df["sw_name"].unique()))
con.close()
print("[OK] 已落库 stock_ind_sw_l1")
