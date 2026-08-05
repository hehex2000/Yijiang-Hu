# -*- coding: utf-8 -*-
"""
神奇公式 V2 「利润回落护栏」调参对比
────────────────────────────────────────────────────────────
背景：V2 用 EBIT 近3年均值压"周期顶单年虚高"，但副作用是
      「一次性暴利会在均值里赖满3年」→ 又变成价值陷阱。
      典型：九安医疗 2022年EBIT 185.01亿（前两年仅 9.29/4.24亿），
      2023年崩回 7.56亿。V2 在 2024 年跑输基准 21.95pp 主因。

关键诊断（point-in-time）：
  2023-12-29 时点（2024调仓）可用最新年报=2022年报=185亿峰值本身，
    最新/均值 = 2.80  → ⑤回落护栏抓不到（那是暴涨不是回落）
  2024-12-31 时点（2025调仓）可用最新年报=2023年报=7.56亿，
    最新/均值 = 0.11  → ⑤才触发，但钱已经亏完了
  ∴ 必须补 ⑧暴涨护栏 才能在"暴利当年"就拦住。

口径：全部 zz800 / 5只 / 20万 / 20140101~20260730，与用户手动跑的 V2 基线一致。
基线 A 复用已落盘 CSV，不重跑。
"""
import os
import sys
import time
import io
import contextlib
import pandas as pd
import numpy as np

import run_magic_v2 as v2

START, END = "20140101", "20260730"
POOL, TOPN, CAP = "zz800", 5, 200000
OUT_DIR = "data/results/magic_v2"
BASE_CSV = f"{OUT_DIR}/backtest_v2_{START}_{END}.csv"

# 变体：(标签, run_backtest_v2 额外参数, 预期落盘 CSV 后缀 tag)
VARIANTS = [
    ("C ⑧暴涨2.0",        {"spike_guard": 2.0},                                    "_sg2"),
    ("D ⑤0.5+⑧2.0",      {"profit_guard": 0.5, "spike_guard": 2.0},               "_pg05_sg2"),
    ("F ⑦中位数+⑤+⑧",    {"ebit_stat": "median", "profit_guard": 0.5,
                            "spike_guard": 2.0},                                   "_median_pg05_sg2"),
]


def yearly_from_csv(path, col="value_real"):
    """从日净值 CSV 算年度收益率(%)，口径同 v2._yearly（年内首末净值比）。"""
    df = pd.read_csv(path, dtype={"date": str})
    df = df.dropna(subset=[col])
    out = {}
    for y, g in df.groupby(df["date"].str[:4]):
        s0, s1 = float(g[col].iloc[0]), float(g[col].iloc[-1])
        if s0 > 0:
            out[y] = (s1 / s0 - 1) * 100
    return out


def metrics_from_csv(path):
    """总收益/年化/最大回撤/夏普——真实趴账口径(value_real)，回撤用 raw 主口径。"""
    df = pd.read_csv(path, dtype={"date": str})
    real = df["value_real"].astype(float).values
    raw = df["value_raw"].astype(float).values
    dates = df["date"].tolist()

    total = (real[-1] / real[0] - 1) * 100
    yrs = (pd.to_datetime(dates[-1]) - pd.to_datetime(dates[0])).days / 365.25
    annual = ((real[-1] / real[0]) ** (1 / yrs) - 1) * 100 if yrs > 0 else 0.0

    peak, mdd = raw[0], 0.0
    for v in raw:
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1)
    rets = np.diff(real) / real[:-1]
    sharpe = ((rets.mean() * 252 - 0.025) / (rets.std() * np.sqrt(252))
              if rets.std() > 0 else 0.0)
    return {"total": total, "annual": annual, "mdd": mdd * 100,
            "sharpe": sharpe, "final": real[-1]}


def bench_yearly():
    """基准 000906.SH 价格指数年度收益(%)。"""
    c = v2._conn()
    rows = c.execute(
        "SELECT CAST(trade_date AS TEXT), close FROM index_daily "
        "WHERE ts_code='000906.SH' AND trade_date BETWEEN ? AND ? "
        "ORDER BY trade_date", (START, END)).fetchall()
    c.close()
    df = pd.DataFrame(rows, columns=["date", "close"])
    out = {}
    for y, g in df.groupby(df["date"].str[:4]):
        s0, s1 = float(g["close"].iloc[0]), float(g["close"].iloc[-1])
        if s0 > 0:
            out[y] = (s1 / s0 - 1) * 100
    total = (float(df["close"].iloc[-1]) / float(df["close"].iloc[0]) - 1) * 100
    return out, total


def main():
    if not os.path.exists(BASE_CSV):
        print(f"[ERROR] 缺少 v2 基线 CSV: {BASE_CSV}")
        return 1

    results = {"A 基线(原v2)": BASE_CSV}
    for label, kw, tag in VARIANTS:
        csv_path = f"{OUT_DIR}/backtest_v2{tag}_{START}_{END}.csv"
        if os.path.exists(csv_path):
            print(f"  [skip] {label} 已存在 → {csv_path}", flush=True)
            results[label] = csv_path
            continue
        print(f"\n{'='*74}\n  ▶ 开始跑 {label}  {kw}\n{'='*74}", flush=True)
        t0 = time.time()
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):     # 吞掉逐年明细，只留摘要
                v2.run_backtest_v2(START, END, top_n=TOPN, stock_pool=POOL,
                                   capital=CAP, industry_cap=2, ebit_years=3,
                                   trend_filter=True, **kw)
        except Exception as e:
            print(f"  [FAIL] {label}: {type(e).__name__}: {e}", flush=True)
            tail = "\n".join(buf.getvalue().splitlines()[-15:])
            print(tail, flush=True)
            continue
        print(f"  [done] {label}  用时 {time.time()-t0:.0f}s", flush=True)
        if os.path.exists(csv_path):
            results[label] = csv_path
        else:
            print(f"  [WARN] 未找到预期产物 {csv_path}", flush=True)

    # ── 汇总 ──
    byr, btotal = bench_yearly()
    rows, yr_tbl = [], {}
    for label, path in results.items():
        m = metrics_from_csv(path)
        rows.append({"变体": label,
                     "总收益%": round(m["total"], 2),
                     "年化%": round(m["annual"], 2),
                     "最大回撤%": round(m["mdd"], 2),
                     "夏普": round(m["sharpe"], 4),
                     "终值": round(m["final"], 0),
                     "超额基准pp": round(m["total"] - btotal, 2)})
        yr_tbl[label] = yearly_from_csv(path)

    summary = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print("  📊 V2 利润护栏调参对比（真实趴账口径 | zz800 | 5只 | 20万 | "
          f"基准000906 {btotal:+.2f}%）")
    print("=" * 90)
    print(summary.to_string(index=False))

    years = sorted(set().union(*[set(v) for v in yr_tbl.values()]))
    ydf = pd.DataFrame({"年份": years, "基准%": [round(byr.get(y, np.nan), 2) for y in years]})
    for label in results:
        ydf[label] = [round(yr_tbl[label].get(y, np.nan), 2) for y in years]
    print("\n" + "=" * 90)
    print("  📅 逐年收益对比（%，真实趴账口径）")
    print("=" * 90)
    print(ydf.to_string(index=False))

    os.makedirs("data/results", exist_ok=True)
    summary.to_csv("data/results/magic_v2_guard_summary.csv",
                   index=False, encoding="utf-8-sig")
    ydf.to_csv("data/results/magic_v2_guard_yearly.csv",
               index=False, encoding="utf-8-sig")
    print("\n  已保存 → data/results/magic_v2_guard_summary.csv")
    print("  已保存 → data/results/magic_v2_guard_yearly.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
