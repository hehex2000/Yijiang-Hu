# -*- coding: utf-8 -*-
"""--price-mode hfq 单元测试：验证价格空间三类不变量

不变量 1：首日 hfq 价格必须严格等于 raw（归一化基准正确）
不变量 2：含分红个股的 hfq 区间涨幅 > raw 区间涨幅（分红贡献为正）
不变量 3：指数代码（000906.SH 等）不被乘个股因子（保持 raw）
不变量 4：adj_factor 缺行日 ffill 生效，不出现回落到 raw 的假跳空
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_monthly_rebalance as m

FAIL = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name}  {detail}")
    if not cond:
        FAIL.append(name)


print("=" * 70)
print("1) 默认口径必须是 raw（保持历史结果可复现）")
print("=" * 70)
check("默认 PRICE_MODE == raw", m.PRICE_MODE == "raw", f"实际={m.PRICE_MODE}")

# 用代表性高股息蓝筹（因子增长温和但真实，北交所次新因子畸高不具代表性）
SAMPLES = ["600000.SH", "000651.SZ", "600036.SH", "601398.SH", "000895.SZ"]
conn = m.get_conn()
qmarks = ",".join("?" * len(SAMPLES))
df_all = m.pd.read_sql_query(
    f"""SELECT ts_code, MIN(adj_factor) f0, MAX(adj_factor) f1, COUNT(*) n
        FROM adj_factor WHERE ts_code IN ({qmarks})
        AND trade_date BETWEEN '20200101' AND '20260723'
        GROUP BY ts_code ORDER BY f1/f0 DESC""", conn, params=SAMPLES)
conn.close()
print(df_all.to_string(index=False))
code = str(df_all["ts_code"].iloc[0])

conn = m.get_conn()
df = m.pd.read_sql_query(
    "SELECT trade_date, adj_factor FROM adj_factor WHERE ts_code=? "
    "AND trade_date BETWEEN '20200101' AND '20260723' ORDER BY trade_date",
    conn, params=(code,))
conn.close()
d0, d1 = str(df["trade_date"].iloc[0]), str(df["trade_date"].iloc[-1])
print(f"  选定样本: {code}  {d0} → {d1}  factor {df['adj_factor'].iloc[0]:.4f} → {df['adj_factor'].iloc[-1]:.4f}")

# ── 不变量 1 & 2 ──
print()
print("=" * 70)
print("2) 切到 hfq 后：首日 == raw，区间涨幅 > raw")
print("=" * 70)
raw_p0 = m.get_price(code, d0)
raw_p1 = m.get_price(code, d1)
raw_ret = raw_p1 / raw_p0 - 1

m.PRICE_MODE = "hfq"
m._ADJ_CACHE.clear()
m._ADJ_REF.clear()

hfq_p0 = m.get_price(code, d0)
hfq_p1 = m.get_price(code, d1)
hfq_ret = hfq_p1 / hfq_p0 - 1

check("不变量1 首日 hfq == raw", abs(hfq_p0 - raw_p0) < 1e-9,
      f"raw={raw_p0:.4f}  hfq={hfq_p0:.4f}")
check("不变量2 hfq 涨幅 > raw 涨幅", hfq_ret > raw_ret,
      f"raw={raw_ret*100:.2f}%  hfq={hfq_ret*100:.2f}%  diff={(hfq_ret-raw_ret)*100:.2f}pp")

# 交叉验证：hfq_ret 应约等于 (p1*f1)/(p0*f0) - 1，即因子增长正是分红贡献
f0 = float(df["adj_factor"].iloc[0])
f1 = float(df["adj_factor"].iloc[-1])
theo = (1 + raw_ret) * (f1 / f0) - 1
check("不变量2b hfq 涨幅 ≈ (1+raw_ret)×(f1/f0)-1", abs(hfq_ret - theo) < 1e-6,
      f"实测={hfq_ret*100:.4f}%  理论={theo*100:.4f}%")

# ── 不变量 3 ──
print()
print("=" * 70)
print("3) 指数代码不被乘个股因子")
print("=" * 70)
idx = "000906.SH"
idx_hfq = m.get_price(idx, d1)
idx_raw_probe = None
m.PRICE_MODE = "raw"
idx_raw = m.get_price(idx, d1)
m.PRICE_MODE = "hfq"
check("不变量3 指数 hfq == raw（未被污染）",
      idx_raw is not None and abs(idx_hfq - idx_raw) < 1e-9,
      f"raw={idx_raw}  hfq={idx_hfq}")

# ── 不变量 4：缺行 ffill ──
print()
print("=" * 70)
print("4) adj_factor 缺行日：必须 ffill，不得回落 raw")
print("=" * 70)
# 找一个全市场缺失的交易日
conn = m.get_conn()
cur = conn.cursor()
cur.execute("""
    SELECT d.trade_date
    FROM (SELECT DISTINCT trade_date FROM daily WHERE ts_code=?) d
    LEFT JOIN (SELECT DISTINCT trade_date FROM adj_factor WHERE ts_code=?) a
      ON d.trade_date = a.trade_date
    WHERE a.trade_date IS NULL AND d.trade_date BETWEEN '20200101' AND '20260723'
    ORDER BY d.trade_date
    LIMIT 1
""", (code, code))
miss = cur.fetchone()
conn.close()
if miss is None:
    # 该股无个股级缺行 → 退回全市场同缺日
    conn = m.get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT d.trade_date
        FROM (SELECT DISTINCT trade_date FROM daily WHERE trade_date BETWEEN '20200101' AND '20260723') d
        LEFT JOIN (SELECT DISTINCT trade_date FROM adj_factor WHERE trade_date BETWEEN '20200101' AND '20260723') a
          ON d.trade_date = a.trade_date
        WHERE a.trade_date IS NULL
        ORDER BY d.trade_date DESC
        LIMIT 1
    """)
    miss = cur.fetchone()
    conn.close()

if miss:
    md = str(miss[0])
    # 该日 daily 有价格但 adj_factor 无行
    conn = m.get_conn()
    px = m.pd.read_sql_query(
        "SELECT close FROM daily WHERE ts_code=? AND trade_date=?", conn, params=(code, md))
    conn.close()
    if not px.empty:
        hfq_miss = m.get_price(code, md)
        # 手工算：应等于 close × ffill(adj) / ref
        sub = df[df["trade_date"].astype(str) <= md]
        adj_ffill = float(sub["adj_factor"].iloc[-1])
        ref = m._ADJ_REF[code]
        expect = float(px["close"].iloc[0]) * adj_ffill / ref
        naive = float(px["close"].iloc[0]) * 1.0 / ref  # 若误用 fillna(1.0)
        check("不变量4 缺行日 hfq == close×ffill(adj)/ref",
              abs(hfq_miss - expect) < 1e-9,
              f"缺行日={md}  实测={hfq_miss:.4f}  ffill期望={expect:.4f}  误用fillna(1.0)会得到={naive:.4f}")
        check("不变量4b 缺行日未掉回 raw 量级", abs(hfq_miss - naive) > 1e-6,
              f"差异={abs(hfq_miss-naive):.4f}（>0 说明 ffill 生效）")
    else:
        print(f"  跳过：{code} 在缺行日 {md} 无 daily 价格")
else:
    print("  跳过：该区间无全市场缺行日")

# ── 不变量 5：成交价与估值价同空间 ──
print()
print("=" * 70)
print("5) get_open_price 与 get_price 必须同空间")
print("=" * 70)
m._ADJ_CACHE.clear()
m._ADJ_REF.clear()
o0 = m.get_open_price(code, d0)
c0 = m.get_price(code, d0)
m.PRICE_MODE = "raw"
o0r = m.get_open_price(code, d0)
c0r = m.get_price(code, d0)
m.PRICE_MODE = "hfq"
# 比值应保持一致（同乘同一因子）
check("不变量5 open/close 比值 raw 与 hfq 一致",
      abs((o0 / c0) - (o0r / c0r)) < 1e-9,
      f"hfq o/c={o0/c0:.6f}  raw o/c={o0r/c0r:.6f}")
check("不变量5b 首日 open 也归一化到 raw", abs(o0 - o0r) < 1e-9,
      f"raw={o0r:.4f}  hfq={o0:.4f}")

print()
print("=" * 70)
if FAIL:
    print(f"结果：{len(FAIL)} 项失败 -> {FAIL}")
    sys.exit(1)
print("结果：全部通过")
print("=" * 70)
