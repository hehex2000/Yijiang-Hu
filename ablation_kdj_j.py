# -*- coding: utf-8 -*-
"""
KDJ-J 事件研究（消融）— 验证 Jim 第4期论点 + 平台 TF7_kdj_j 独立增量
====================================================================
目的：
  1) 验证 Jim 论点：把 KDJ"金叉"当买卖按钮无 edge；KDJ 作为"价格位置+变化速度"描述器才有意义。
  2) 测 TF7_kdj_j（仅取 J 线）作为独立描述器，到底有没有"独立增量"。

方法（严格遵循 Jim 规矩③ + 平台 analyze_rsi_regime.py 范式）：
  - 固定参数（不看完结果再改）：N∈{9,14,20}，M1=M2=3。
  - 全样本、按市值分三组各随机 120 只，后复权价。
  - 统计信号后 N 日收益分布（含无条件对照组）。
  - 信号（全是描述器语境，绝不把交叉当按钮）：
      j_recover    : J 由负区拐头向上 (J_t>0 & J_{t-1}<=0) —— Jim"看交叉前的位置变化"
      j_oversold   : J_t < 0   （极端低位描述器）
      j_overbought : J_t > 100 （极端高位描述器）
      golden_cross : K 上穿 D  （"金叉当按钮"——对比组，预期无 edge）
  - 输出：控制台表格 + CSV。

结论判据：若 j_recover / j_oversold 相对对照组无显著超额 → KDJ-J 作为独立信号无增量，
         其价值仅在"作为 MACD 策略的语境确认门"（另测），与 Jim 立场一致。
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
HORIZONS = [10, 20, 60]
N_GRID = [9, 14, 20]
PRIMARY = 20
SEED = 42
OUT_DIR = "data/results/kdj_j_ablation"
MAX_ABS_RET = 1.0


# ────────────────────────────────────────────────
# 1. 取样本股票（按市值分三组，同 analyze_rsi_regime 范式）
# ────────────────────────────────────────────────
def pick_universe(conn) -> dict:
    ref_date = conn.execute(
        "select max(trade_date) from daily_basic where trade_date<='20230630'"
    ).fetchone()[0]
    df = pd.read_sql(
        "select ts_code, total_mv from daily_basic where trade_date=? and total_mv>0",
        conn, params=(ref_date,),
    )
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
def load_prices(conn, codes: list) -> pd.DataFrame:
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
# 3. KDJ 计算 + 事件收集
# ────────────────────────────────────────────────
def kdj(high, low, close, n):
    k, d = ta.STOCH(
        high, low, close,
        fastk_period=n, slowk_period=3, slowk_matype=0,
        slowd_period=3, slowd_matype=0,
    )
    j = 3.0 * k - 2.0 * d
    return k, d, j


def process_stock(g: pd.DataFrame):
    if len(g) < 300:
        return [], np.empty((0, len(HORIZONS)))
    high = g["high"].to_numpy(dtype=float)
    low = g["low"].to_numpy(dtype=float)
    close = g["close"].to_numpy(dtype=float)
    dates = g["trade_date"].to_numpy()

    # 前瞻收益（多 horizon）
    fwd = {}
    for h in HORIZONS:
        f = np.full(len(close), np.nan)
        if len(close) > h:
            f[:-h] = close[h:] / close[:-h] - 1.0
        fwd[h] = f

    rows = []
    base_rows = []
    kd = {n: kdj(high, low, close, n) for n in N_GRID}

    # 有效掩码（以 PRIMARY horizon 为准，剔除异常收益）
    valid = ~np.isnan(fwd[PRIMARY])
    valid &= (np.abs(np.nan_to_num(fwd[PRIMARY], nan=9.9)) <= MAX_ABS_RET)

    # 对照组：有效日（无条件"任意一天"）
    for i in np.flatnonzero(valid):
        base_rows.append(tuple(float(fwd[h][i]) for h in HORIZONS))

    def collect(mask, sig_name, n_val):
        idx = np.flatnonzero(mask & valid)
        for i in idx:
            rec = {"signal": sig_name, "n_param": n_val, "date": str(dates[i])}
            for h in HORIZONS:
                rec[f"fwd{h}"] = float(fwd[h][i])
            rows.append(rec)

    for n in N_GRID:
        k, d, j = kd[n]
        j_prev = np.roll(j, 1); j_prev[0] = np.nan
        k_prev = np.roll(k, 1); k_prev[0] = np.nan
        d_prev = np.roll(d, 1); d_prev[0] = np.nan
        # j_recover：J 由负区拐头向上（描述器"恢复中"，Jim 看交叉前位置）
        collect((j > 0) & (j_prev <= 0), "j_recover", n)
        # j_oversold：极端低位描述器
        collect(j < 0, "j_oversold", n)
        # j_overbought：极端高位描述器
        collect(j > 100, "j_overbought", n)
        # golden_cross：K 上穿 D（"金叉当按钮"——对比组）
        collect((k > d) & (k_prev <= d_prev), "golden_cross", n)

    return rows, np.array(base_rows, dtype=float)


# ────────────────────────────────────────────────
# 4. 主流程
# ────────────────────────────────────────────────
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
                bdf = pd.DataFrame(base, columns=[f"fwd{h}" for h in HORIZONS])
                bdf["group"] = grp
                base_parts.append(bdf)
        print(f"       {grp} 完成，累计事件 {len(all_rows):,}", flush=True)
    conn.close()

    ev = pd.DataFrame(all_rows)
    bl = pd.concat(base_parts, ignore_index=True)
    print(f"\n[事件总数] {len(ev):,}   [基准样本] {len(bl):,}")

    def agg(df):
        f = df[f"fwd{PRIMARY}"]
        return pd.Series({
            "信号数": len(df),
            "胜率%": (f > 0).mean() * 100,
            "均值%": f.mean() * 100,
            "中位%": f.median() * 100,
            "t值": (f.mean() / (f.std() / np.sqrt(len(f)))) if len(f) > 2 and f.std() > 0 else 0.0,
        })

    base_mean = bl[f"fwd{PRIMARY}"].mean() * 100
    base_win = (bl[f"fwd{PRIMARY}"] > 0).mean() * 100
    print(f"\n[基准 未来{PRIMARY}日] 均值={base_mean:.2f}% 胜率={base_win:.2f}%")

    pd.set_option("display.width", 240)
    pd.set_option("display.unicode.east_asian_width", True)

    for sig in ["j_recover", "j_oversold", "j_overbought", "golden_cross"]:
        sub = ev[ev["signal"] == sig]
        print("\n" + "=" * 100)
        print(f"信号【{sig}】未来{PRIMARY}日收益（按 N 参数）  | 基准均值={base_mean:.2f}% 基准胜率={base_win:.2f}%")
        print("=" * 100)
        if len(sub) == 0:
            print("  (无事件)")
            continue
        t = sub.groupby("n_param", group_keys=False).apply(agg, include_groups=False)
        t["超额均值pp"] = t["均值%"] - base_mean
        t["超额胜率pp"] = t["胜率%"] - base_win
        print(t.round(2).to_string())

    # 多 horizon 小结：描述器 vs 按钮
    print("\n" + "=" * 100)
    print("多 horizon 小结：j_recover(描述器语境) vs golden_cross(金叉当按钮)  均值% / 超额pp(相对基准)")
    print("=" * 100)
    summ = []
    for sig in ["j_recover", "golden_cross"]:
        sub = ev[ev["signal"] == sig]
        for h in HORIZONS:
            f = sub[f"fwd{h}"]
            bm = bl[f"fwd{h}"].mean() * 100
            summ.append({
                "signal": sig, "horizon": h, "n": len(f),
                "均值%": f.mean() * 100, "超额pp": f.mean() * 100 - bm,
                "胜率%": (f > 0).mean() * 100,
            })
    print(pd.DataFrame(summ).round(2).to_string(index=False))

    # 保存
    ev.to_csv(f"{OUT_DIR}/events_raw.csv", index=False, encoding="utf-8-sig")
    bl.to_csv(f"{OUT_DIR}/baseline.csv", index=False, encoding="utf-8-sig")
    print(f"\n[完成] 明细已写入 {OUT_DIR}/")


if __name__ == "__main__":
    main()
