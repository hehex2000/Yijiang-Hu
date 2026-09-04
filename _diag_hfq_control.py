# -*- coding: utf-8 -*-
"""受控实验：固定篮子买入持有，隔离价格空间逻辑 vs 路径效应

目的：若 hfq 与 raw 的差 = 篮子真实股息贡献（预期 4-6%/年），则价格空间实现正确；
      若差显著更大（如回测里的 11.5%/年），说明组合构建路径里还有第二个 bug。

两组样本：
  A) 高股息蓝筹（股息率 4-6%）
  B) 中证800 全池随机 100 只（股息率 ~2.3%，应与基准 2.23%/年 对齐）
"""
import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_monthly_rebalance as m
import pandas as pd

D0, D1 = "20200102", "20260723"
conn = m.get_conn()

# ── 取交易日序列 ──
dates = [str(x) for x in pd.read_sql_query(
    "SELECT trade_date FROM daily WHERE ts_code='000001.SZ' AND trade_date BETWEEN ? AND ? ORDER BY trade_date",
    conn, params=(D0, D1))["trade_date"].tolist()]
print(f"交易日 {len(dates)} 天: {dates[0]} → {dates[-1]}")

# ── 样本 A：2020 年股息率最高的蓝筹 ──
dv = pd.read_sql_query("""
    SELECT b.ts_code, AVG(b.dv_ttm) dv
    FROM daily_basic b
    WHERE b.trade_date BETWEEN '20200101' AND '20201231'
      AND b.dv_ttm IS NOT NULL AND b.dv_ttm > 0
    GROUP BY b.ts_code
    HAVING COUNT(*) > 200
    ORDER BY dv DESC LIMIT 30
""", conn)
sample_a = dv["ts_code"].tolist()

# ── 样本 B：中证800 成分（2020 快照）随机 100 ──
try:
    zz800 = pd.read_sql_query(
        "SELECT DISTINCT con_code FROM index_weight WHERE index_code IN ('000906.SH','000906.CSI') "
        "AND trade_date LIKE '2020%' LIMIT 1", conn)
    pool = pd.read_sql_query(
        "SELECT DISTINCT con_code FROM index_weight WHERE index_code LIKE '000906%' "
        "AND trade_date BETWEEN '20200101' AND '20201231'", conn)["con_code"].tolist()
except Exception:
    pool = []
if not pool:
    pool = pd.read_sql_query(
        "SELECT DISTINCT ts_code FROM daily WHERE trade_date='20200102' LIMIT 2000",
        conn)["ts_code"].tolist()
random.seed(42)
sample_b = random.sample(pool, min(100, len(pool)))
conn.close()

YEARS = len(dates) / 252.0


def run_basket(codes, mode, init=200000.0):
    """等权买入持有，整股下单，与回测引擎同逻辑。返回 (终值, 年化)。"""
    m.PRICE_MODE = mode
    m._ADJ_CACHE.clear()
    m._ADJ_REF.clear()

    px0 = {}
    ok = []
    for c in codes:
        p = m.get_price(c, dates[0])
        if p and p > 0:
            px0[c] = p
            ok.append(c)
    if not ok:
        return None
    per = init / len(ok)
    shares = {}
    cash = 0.0
    for c in ok:
        n = int(per // px0[c])
        shares[c] = n
        cash += per - n * px0[c]
    for c in list(ok):
        if shares[c] == 0:
            ok.remove(c)
    # 末日估值
    val = cash
    for c in ok:
        p1 = m.get_price(c, dates[-1])
        val += shares[c] * (p1 if p1 else px0[c])
    ann = (val / init) ** (1 / YEARS) - 1
    return val, ann, len(ok)


print()
print("=" * 78)
print(f"受控买入持有  {D0} → {D1}  ({YEARS:.2f} 年)")
print("=" * 78)

results = {}
for label, codes in [("A 高股息蓝筹 Top30", sample_a), ("B 中证800 随机100", sample_b)]:
    r_raw = run_basket(codes, "raw")
    r_hfq = run_basket(codes, "hfq")
    if not r_raw or not r_hfq:
        print(f"{label}: 数据不足，跳过")
        continue
    v0, a0, n0 = r_raw
    v1, a1, n1 = r_hfq
    # 年化差 = 复利意义下的"股息贡献"
    div_ann = (v1 / v0) ** (1 / YEARS) - 1 if (v0 > 0) else float("nan")
    results[label] = (a0, a1, div_ann)
    print(f"\n【{label}】持仓 {n0} 只")
    print(f"  raw : 终值 {v0:>12,.0f}   年化 {a0*100:>7.2f}%")
    print(f"  hfq : 终值 {v1:>12,.0f}   年化 {a1*100:>7.2f}%")
    print(f"  隐含股息贡献(复利): {div_ann*100:>7.2f}%/年")

# ── 逐股收益分布：验证不是少数极端股拉高 ──
print()
print("=" * 78)
print("逐股对照：样本 A 的 raw vs hfq 收益")
print("=" * 78)
rows = []
for c in sample_a[:15]:
    m.PRICE_MODE = "raw"
    m._ADJ_CACHE.clear(); m._ADJ_REF.clear()
    p0r, p1r = m.get_price(c, dates[0]), m.get_price(c, dates[-1])
    m.PRICE_MODE = "hfq"
    m._ADJ_CACHE.clear(); m._ADJ_REF.clear()
    p0h, p1h = m.get_price(c, dates[0]), m.get_price(c, dates[-1])
    if not all([p0r, p1r, p0h, p1h]):
        continue
    rr = p1r / p0r - 1
    rh = p1h / p0h - 1
    rows.append((c, rr, rh, ((1 + rh) / (1 + rr)) ** (1 / YEARS) - 1))
df = pd.DataFrame(rows, columns=["ts_code", "raw_ret", "hfq_ret", "隐含股息/年"])
df = df.sort_values("隐含股息/年", ascending=False)
print(df.to_string(index=False, formatters={
    "raw_ret": lambda x: f"{x*100:7.2f}%",
    "hfq_ret": lambda x: f"{x*100:7.2f}%",
    "隐含股息/年": lambda x: f"{x*100:7.2f}%"}))
print()
print(f"  样本 A 隐含股息/年  中位数 {df['隐含股息/年'].median()*100:.2f}%  "
      f"均值 {df['隐含股息/年'].mean()*100:.2f}%  "
      f"最大 {df['隐含股息/年'].max()*100:.2f}%")

print()
print("=" * 78)
print("判定")
print("=" * 78)
for label, (a0, a1, d) in results.items():
    flag = "OK" if d < 0.08 else "⚠️ 偏高"
    print(f"  {label}: 隐含股息 {d*100:.2f}%/年  [{flag}]")
print()
print("  回测引擎口径下策略隐含股息 = 11.54%/年（2020-2026 value top20）")
print("  若受控实验显示 4-6%/年，则回测里的超额来自【路径效应】(假止损→换股→锁定亏损)")
print("  而非价格空间实现错误；反之则价格空间仍有 bug。")
