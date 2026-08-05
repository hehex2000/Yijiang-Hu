# -*- coding: utf-8 -*-
"""
RSI 参数 / 市场状态(regime) 事件研究
=====================================
目的：用自己的 A 股数据验证两个论点
  论点1（参数）: Wilder 的 14+70/30 不是最优，且最优参数随品种(市值组)变化
  论点2（状态）: RSI 超卖只在震荡市有效；趋势下跌市中会钝化，
                且"碰布林带下轨"这个确认条件挡不住钝化

方法：事件研究(event study)，不做完整回测，只统计信号后 N 日收益分布
  - 样本：按 total_mv 分大/中/小三组，每组随机 120 只
  - regime：ADX(14) <20 震荡 / >=25 趋势，趋势再按 close vs MA60 分上下
  - 信号：RSI(p) < oversold  （另可叠加 close <= BB_lower*1.02）
  - 观测：未来 20 日收益（后复权收盘价）

输出：控制台文本表格 + CSV 明细
"""
from __future__ import annotations

import os
import sqlite3
import random
import numpy as np
import pandas as pd
import talib as ta

DB = r"D:\tu-shareData\astock_daily.db"
START = "20160101"
END = "20260730"
N_PER_GROUP = 120
FWD = 20          # 前瞻天数
SEED = 42
OUT_DIR = "data/results/rsi_regime"


# ────────────────────────────────────────────────
# 1. 取样本股票（按市值分三组）
# ────────────────────────────────────────────────
def pick_universe(conn) -> dict[str, list[str]]:
    # 用一个较近的交易日的市值做分组（避免用最新日期导致新股偏差，取 2023 年中）
    ref_date = conn.execute(
        "select max(trade_date) from daily_basic where trade_date<='20230630'"
    ).fetchone()[0]
    df = pd.read_sql(
        "select ts_code, total_mv from daily_basic where trade_date=? and total_mv>0",
        conn, params=(ref_date,),
    )
    # 排除上市不足 3 年的
    basic = pd.read_sql("select ts_code, list_date, name from stock_basic", conn)
    df = df.merge(basic, on="ts_code", how="inner")
    df = df[df["list_date"] < "20150101"]
    df = df[~df["name"].str.contains("ST", na=False)]

    df = df.sort_values("total_mv")
    n = len(df)
    groups = {
        "小盘": df.iloc[: n // 3]["ts_code"].tolist(),
        "中盘": df.iloc[n // 3: 2 * n // 3]["ts_code"].tolist(),
        "大盘": df.iloc[2 * n // 3:]["ts_code"].tolist(),
    }
    rng = random.Random(SEED)
    out = {}
    for k, v in groups.items():
        out[k] = rng.sample(v, min(N_PER_GROUP, len(v)))
    print(f"[样本] 参考日={ref_date}  可选池={n}  每组抽取={N_PER_GROUP}")
    return out


# ────────────────────────────────────────────────
# 2. 读行情 + 后复权
# ────────────────────────────────────────────────
def load_prices(conn, codes: list[str]) -> pd.DataFrame:
    ph = ",".join("?" * len(codes))
    sql = f"""
        select d.ts_code, d.trade_date, d.open, d.high, d.low, d.close,
               a.adj_factor as adj
        from daily d
        left join adj_factor a
               on a.ts_code=d.ts_code and a.trade_date=d.trade_date
        where d.ts_code in ({ph}) and d.trade_date between ? and ?
        order by d.ts_code, d.trade_date
    """
    df = pd.read_sql(sql, conn, params=codes + [START, END])
    # 复权因子缺失必须前后填充，绝不能填 1.0（会造成价格跳变→虚假巨额收益）
    df["adj"] = df.groupby("ts_code")["adj"].ffill().bfill()
    df = df[df["adj"].notna()]
    for c in ("open", "high", "low", "close"):
        df[c] = df[c] * df["adj"]
    return df


# ────────────────────────────────────────────────
# 3. 单只股票：算指标 + 收集事件
# ────────────────────────────────────────────────
PERIODS = [7, 9, 14, 21]
THRESHOLDS = [(20, 80), (25, 75), (30, 70), (35, 65)]


def classify_regime(adx: np.ndarray, close: np.ndarray, ma60: np.ndarray) -> np.ndarray:
    """0=震荡 1=上升趋势 2=下降趋势 3=过渡"""
    reg = np.full(len(close), 3, dtype=np.int8)
    reg[adx < 20] = 0
    trend = adx >= 25
    reg[trend & (close >= ma60)] = 1
    reg[trend & (close < ma60)] = 2
    return reg


MAX_ABS_RET = 1.0   # 剔除 20 日涨跌幅超过 ±100% 的样本（多为数据异常）


def process_stock(g: pd.DataFrame):
    """返回 (信号事件列表, 基准数组[regime, fwd_ret])"""
    if len(g) < 300:
        return [], np.empty((0, 2))
    high = g["high"].to_numpy(dtype=float)
    low = g["low"].to_numpy(dtype=float)
    close = g["close"].to_numpy(dtype=float)
    dates = g["trade_date"].to_numpy()

    adx = ta.ADX(high, low, close, timeperiod=14)
    ma60 = ta.SMA(close, timeperiod=60)
    upper, mid, lower = ta.BBANDS(close, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
    regime = classify_regime(np.nan_to_num(adx, nan=-1), close, np.nan_to_num(ma60, nan=1e18))

    # 前瞻收益
    fwd = np.full(len(close), np.nan)
    fwd[:-FWD] = close[FWD:] / close[:-FWD] - 1.0

    rsis = {p: ta.RSI(close, timeperiod=p) for p in PERIODS}
    near_lower = close <= lower * 1.02

    rows = []
    valid = (
        ~np.isnan(fwd) & ~np.isnan(mid) & (adx > 0)
        & (np.abs(np.nan_to_num(fwd, nan=9.9)) <= MAX_ABS_RET)
    )
    # ── 基准：同一 regime 下"任意一天买入"的前瞻收益（无条件对照组）──
    bidx = np.flatnonzero(valid)
    baseline = np.column_stack([regime[bidx].astype(float), fwd[bidx]])

    for p in PERIODS:
        r = rsis[p]
        r_prev = np.roll(r, 1); r_prev[0] = np.nan
        for os_, ob_ in THRESHOLDS:
            sig = valid & ~np.isnan(r_prev) & (r_prev < os_)
            idx = np.flatnonzero(sig)
            for i in idx:
                rows.append({
                    "period": p, "oversold": os_,
                    "regime": int(regime[i]),
                    "bb_confirm": bool(near_lower[i - 1]) if i >= 1 else False,
                    "fwd_ret": float(fwd[i]),
                    "date": str(dates[i]),
                })
    return rows, baseline


# ────────────────────────────────────────────────
# 4. 主流程
# ────────────────────────────────────────────────
REG_NAME = {0: "震荡市", 1: "上升趋势", 2: "下降趋势", 3: "过渡"}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    conn = sqlite3.connect(DB)
    universe = pick_universe(conn)

    all_rows = []
    base_parts = []
    for grp, codes in universe.items():
        print(f"[加载] {grp} {len(codes)} 只 ...", flush=True)
        px = load_prices(conn, codes)
        for code, g in px.groupby("ts_code", sort=False):
            rows, base = process_stock(g.reset_index(drop=True))
            for r in rows:
                r["group"] = grp
            all_rows.extend(rows)
            if len(base):
                bdf = pd.DataFrame(base, columns=["regime", "fwd_ret"])
                bdf["group"] = grp
                base_parts.append(bdf)
        print(f"       {grp} 完成，累计事件 {len(all_rows):,}", flush=True)
    conn.close()

    ev = pd.DataFrame(all_rows)
    ev["regime_name"] = ev["regime"].map(REG_NAME)
    ev.to_csv(f"{OUT_DIR}/events_raw.csv", index=False, encoding="utf-8-sig")

    bl = pd.concat(base_parts, ignore_index=True)
    bl["regime"] = bl["regime"].astype(int)
    bl["regime_name"] = bl["regime"].map(REG_NAME)
    print(f"\n[事件总数] {len(ev):,}   [基准样本] {len(bl):,}")

    def agg(df):
        return pd.Series({
            "信号数": len(df),
            "胜率%": (df["fwd_ret"] > 0).mean() * 100,
            "均值%": df["fwd_ret"].mean() * 100,
            "中位%": df["fwd_ret"].median() * 100,
        })

    pd.set_option("display.width", 200)
    pd.set_option("display.unicode.east_asian_width", True)

    # ── 表0：基准（无条件买入）──
    print("\n" + "=" * 88)
    print(f"表0  【对照组】不看任何指标，该状态下任意一天买入，未来{FWD}日收益")
    print("=" * 88)
    t0 = bl.groupby("regime_name", group_keys=False).apply(agg, include_groups=False)
    print(t0.round(2).to_string())

    # ── 表1：论点2 —— 同一套参数(14/30)在不同 regime 的表现 ──
    print("\n" + "=" * 88)
    print(f"表1  Wilder 默认 RSI(14)<30，未来{FWD}日收益 —— 按市场状态拆分（含超额）")
    print("=" * 88)
    base = ev[(ev["period"] == 14) & (ev["oversold"] == 30)]
    t1 = base.groupby("regime_name", group_keys=False).apply(agg, include_groups=False)
    t1["基准胜率%"] = t0["胜率%"]
    t1["超额胜率pp"] = t1["胜率%"] - t0["胜率%"]
    t1["基准中位%"] = t0["中位%"]
    t1["超额中位pp"] = t1["中位%"] - t0["中位%"]
    print(t1.round(2).to_string())

    # ── 表2：布林带下轨确认能否救趋势市 ──
    print("\n" + "=" * 78)
    print("表2  加上『价格<=布林下轨*1.02』二次确认后，各 regime 表现变化")
    print("=" * 78)
    t2 = base.groupby(["regime_name", "bb_confirm"], group_keys=False).apply(agg, include_groups=False)
    print(t2.round(2).to_string())

    # ── 表3：论点1 —— 参数网格 × 市值组（仅震荡市）──
    print("\n" + "=" * 88)
    print("表3  【震荡市】参数网格 × 市值组，超额 = 相对该市值组震荡市基准")
    print("=" * 88)
    osc = ev[ev["regime"] == 0]
    bl_osc = bl[bl["regime"] == 0].groupby("group", group_keys=False).apply(agg, include_groups=False)
    print("\n[各市值组 震荡市基准]")
    print(bl_osc.round(2).to_string())

    t3 = osc.groupby(["group", "period", "oversold"], group_keys=False).apply(agg, include_groups=False)
    t3r = t3.reset_index()
    t3r["超额胜率pp"] = t3r["胜率%"] - t3r["group"].map(bl_osc["胜率%"])
    t3r["超额中位pp"] = t3r["中位%"] - t3r["group"].map(bl_osc["中位%"])
    print("\n" + t3r.round(2).to_string(index=False))

    # 每组最优参数
    print("\n" + "-" * 88)
    print("各市值组在震荡市下的最优参数（按超额胜率，信号数>=300）")
    print("-" * 88)
    cand = t3r[t3r["信号数"] >= 300]
    best = cand.loc[cand.groupby("group")["超额胜率pp"].idxmax()]
    print(best.round(2).to_string(index=False))
    print("\n[对照] Wilder 默认 14/30 在各组的表现：")
    print(t3r[(t3r["period"] == 14) & (t3r["oversold"] == 30)].round(2).to_string(index=False))

    # ── 表4：全市场（不分组）参数网格，含所有 regime ──
    print("\n" + "=" * 78)
    print("表4  全样本（不分 regime）参数网格 —— 对照组")
    print("=" * 78)
    t4 = ev.groupby(["period", "oversold"], group_keys=False).apply(agg, include_groups=False)
    print(t4.round(2).to_string())

    t0.to_csv(f"{OUT_DIR}/t0_baseline.csv", encoding="utf-8-sig")
    t1.to_csv(f"{OUT_DIR}/t1_regime.csv", encoding="utf-8-sig")
    t2.to_csv(f"{OUT_DIR}/t2_bb_confirm.csv", encoding="utf-8-sig")
    t3r.to_csv(f"{OUT_DIR}/t3_grid_by_group.csv", index=False, encoding="utf-8-sig")
    t4.to_csv(f"{OUT_DIR}/t4_grid_all.csv", encoding="utf-8-sig")
    print(f"\n[完成] 明细已写入 {OUT_DIR}/")


if __name__ == "__main__":
    main()
