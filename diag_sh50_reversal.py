# -*- coding: utf-8 -*-
"""
diag_sh50_reversal.py — 诊断「上证50 ETF vs 上证50 指数」同底层反向红旗
=====================================================================
现象：run_chan_lun_faithful.py 里 510050.SH(etf_daily) 缠论 +84pp 跑赢 BH，
而 000016.SH(index_daily) 缠论 -24pp 跑输 BH —— 同一篮 50 只股票结论反转。
本脚本定位根因，输出：
  1) 两表 schema（index_daily 是否只有 close，缺 open/high/low）
  2) 各列 NULL 数（index_daily 的 OHLC 是否真实存在）
  3) 价格比例 ETF/IDX（归一化）的稳定性 + 单日跳变
  4) 各序列单日 >10% 断裂（疑似缩放/数据错误）
  5) 日收益相关系数（同底层应 ~0.99，否则形状被改）
  6) 忠实缠论内核买卖点日期重叠（同底层应高度重合）
结论用于决定「修数据 / 统一基准 / 剔除指数」。
"""
import sqlite3
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from chan_lun_core_faithful import compute_states

DB = r"D:\tu-shareData\astock_daily.db"


def schema(table):
    c = sqlite3.connect(DB)
    rows = c.execute(f"PRAGMA table_info({table})").fetchall()
    c.close()
    return [r[1] for r in rows]


def load(code, table, start="20100101"):
    c = sqlite3.connect(DB)
    rows = c.execute(
        f"SELECT trade_date, open, high, low, close FROM {table} "
        f"WHERE ts_code=? AND trade_date>=? ORDER BY trade_date",
        (code, start)).fetchall()
    c.close()
    return pd.DataFrame(rows, columns=["date", "open", "high", "low", "close"])


print("== schema etf_daily ==", schema("etf_daily"))
print("== schema index_daily ==", schema("index_daily"))

etf = load("510050.SH", "etf_daily")
idx = load("000016.SH", "index_daily")
print(f"\nETF 510050 rows={len(etf)} range {etf['date'].iloc[0]}~{etf['date'].iloc[-1]}")
print(f"IDX 000016 rows={len(idx)} range {idx['date'].iloc[0]}~{idx['date'].iloc[-1]}")

print("\n--- 前3行快照 ---")
print("ETF:", etf.head(3).to_string(index=False))
print("IDX:", idx.head(3).to_string(index=False))

# NULL 检查
print("\n--- 各列 NULL 数 ---")
for col in ["open", "high", "low", "close"]:
    print(f"  ETF.{col} NULL={etf[col].isna().sum()} | IDX.{col} NULL={idx[col].isna().sum()}")

# 合并对齐
m = etf.merge(idx, on="date", suffixes=("_e", "_i"))
print(f"\ncommon dates={len(m)}")
for col in ["open_e", "high_e", "low_e", "open_i", "high_i", "low_i"]:
    nnull = m[col].isna().sum()
    if nnull:
        print(f"  ⚠ {col} NULL={nnull}")

# 归一化比例
m["ne"] = m["close_e"].astype(float) / m["close_e"].astype(float).iloc[0] * 100
m["ni"] = m["close_i"].astype(float) / m["close_i"].astype(float).iloc[0] * 100
m["ratio"] = m["ne"] / m["ni"]
print(f"\nratio(ETF归一/IDX归一) min={m['ratio'].min():.3f} max={m['ratio'].max():.3f} "
      f"mean={m['ratio'].mean():.3f} std={m['ratio'].std():.3f}")

# ratio 单日跳变
m["rjump"] = m["ratio"].pct_change().abs()
jb = m[m["rjump"] > 0.05]
print(f"ratio 单日跳变>5% 次数={len(jb)}")
for _, row in jb.head(10).iterrows():
    er = (row["close_e"].astype(float).pct_change()) * 100
    ir = (row["close_i"].astype(float).pct_change()) * 100
    print(f"   {row['date']} ratio {row['ratio']:.3f} jump {row['rjump']*100:.1f}%  "
          f"ETFret {er:.1f}% IDXret {ir:.1f}%")

# 单日 >10% 断裂
for name, df in [("ETF", etf), ("IDX", idx)]:
    df["r"] = df["close"].astype(float).pct_change().abs()
    big = df[df["r"] > 0.10]
    print(f"{name} 单日>10%跳变 次数={len(big)} -> {list(big['date'])[:5]}")

# 日收益相关性
re = etf.set_index("date")["close"].astype(float).pct_change()
ri = idx.set_index("date")["close"].astype(float).pct_change()
corr = re.corr(ri)
print(f"\n日收益相关系数 = {corr:.4f}")

# 忠实内核买卖点日期重叠
def sigs(df, label):
    h = df["high"].astype(float).values
    l = df["low"].astype(float).values
    cl = df["close"].astype(float).values
    st = compute_states(h, l, cl)
    bm = {i for i, _ in st["buys"]}
    sm = {i for i, _ in st["sells"]}
    bd = {df["date"].iloc[i] for i in bm}
    sd = {df["date"].iloc[i] for i in sm}
    return bd, sd, len(st["buys"]), len(st["sells"]), len(st["bi"]), len(st["segments"])

be, se, nb_e, ns_e, bi_e, seg_e = sigs(etf, "ETF")
bi_, si_, nb_i, ns_i, bi_i, seg_i = sigs(idx, "IDX")
print(f"\nETF 买点{nb_e} 卖点{ns_e} | 笔{bi_e} 线段{seg_e}")
print(f"IDX 买点{nb_i} 卖点{ns_i} | 笔{bi_i} 线段{seg_i}")
print(f"买点日期重叠(同日期)={len(be & bi_)} (ETF买{len(be)} IDX买{len(bi_)})")
print(f"卖点日期重叠(同日期)={len(se & si_)} (ETF卖{len(se)} IDX卖{len(si_)})")

print("\n== 诊断结论提示 ==")
if corr < 0.9:
    print("  ✦ 日收益相关性偏低(<0.9) → 两序列形状已被数据差异改掉，反转很可能源于数据口径/错误。")
if m["ratio"].std() > 0.05:
    print("  ✦ ratio 波动较大 → 两序列非简单缩放关系，存在结构性差异(断点/复权/价格vs全收益)。")
if idx["open"].isna().sum() or idx["high"].isna().sum() or idx["low"].isna().sum():
    print("  ✦ index_daily(000016) 缺 open/high/low → 缠论需 OHLC，指数仅收盘价无法忠实分析。")
print("  → 见上，据此决定修数据 / 统一为单一可比基准(如上证50仅用510050 ETF)。")
