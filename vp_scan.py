# -*- coding: utf-8 -*-
"""
vp_scan: Volume Profile 市场扫描（计划 M3，对标 Kara BV1tgVs6SE4C Tab4 市场扫描）
对全 A 批量跑快照式 VP 检测，输出 Kara Tab4 三张排序表：
  - 支撑位附近（按距支撑由近到远排）
  - 压力位附近（按距压力由近到远排）
  - 盈亏比(R)最优（按 R 从高到低排）
以及顶部五张统计卡片：总扫描数 / 附近支撑数 / 压力附近数 / R最优数 / 平均盈亏比。

设计要点：
- 快照：每只票用最近 WINDOW 日线算当前密集区，取最新收盘价为"今日价"。
- 无前视：profile 只用 [t-WINDOW, t] 历史（因子层已验证）。
- 口径：raw（市场实际成交价，分红跳空不平移记忆位）。
- 不强依赖逐票 rolling_backtest（那要 ~5000×120 次循环，过慢）；扫描是截面快照，
  历史胜率(回测置信度)属 Tab2 决策支持，按需另跑 sample。
"""
import os
import sqlite3

import numpy as np
import pandas as pd

import config
import vp_data
import vp_core

# ---- 配置 ----
WINDOW = 250          # 密集区回看交易日
N_BINS = 80           # 分箱数
MIN_ROWS = 250        # 最少历史行数（不满足则跳过）
NEAR_PCT = 0.03       # "附近"阈值（3%，对应 Kara 黄色预警）
R_MIN = 2.0           # "R最优"阈值（reward>=2×risk）
RISK_FLOOR = 0.005    # 风险分母下限（0.5%）：价贴支撑时防止 R 除零吹爆
SMOOTH = 2.0

DB_PATH = config.DATA.get("local_db_path", "")
OUT_DIR = os.path.join("data", "results", "volume_profile")


def _load_basic():
    """ts_code -> (name, industry)"""
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            "SELECT ts_code, name, industry FROM stock_basic", conn
        )
    finally:
        conn.close()
    return {
        r["ts_code"]: (r["name"], r["industry"]) for _, r in df.iterrows()
    }


def _universe(min_rows=MIN_ROWS):
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = pd.read_sql_query(
            "SELECT ts_code, COUNT(*) n FROM daily GROUP BY ts_code", conn
        )
    finally:
        conn.close()
    return rows[rows["n"] >= min_rows]["ts_code"].tolist()


def scan_one(ts_code):
    """返回该票快照 dict；无效返回 None。"""
    df = vp_data.get_daily(ts_code)  # raw, 升序
    if df is None or len(df) < MIN_ROWS:
        return None
    price = float(df["close"].iloc[-1])
    if price <= 0:
        return None
    wdf = df.tail(WINDOW)
    res = vp_core.volume_profile(wdf, n_bins=N_BINS, smooth_sigma=SMOOTH)
    if res is None:
        return None
    centers, raw, sm = res
    zones, poc = vp_core.detect_zones(centers, sm)
    if not zones:
        return None
    supports = [(p, s) for p, s in zones if p < price]
    resists = [(p, s) for p, s in zones if p > price]
    if not supports:
        return None  # 跌破（下方无支撑）
    ns = max(supports)            # (price, strength) 最近支撑
    nr = min(resists) if resists else None  # 最近压力
    sup_dist = (price - ns[0]) / price
    sup_str = ns[1]
    if nr is not None:
        res_dist = (nr[0] - price) / price
        res_str = nr[1]
        risk = price - ns[0]
        if risk < price * RISK_FLOOR:
            risk = price * RISK_FLOOR  # 价贴支撑时给最小止损距离，R 有界
        R = (nr[0] - price) / risk
    else:
        res_dist = None
        res_str = None
        R = None
    # 信号
    if sup_dist <= NEAR_PCT:
        sig = "支撑附近"
    elif nr is not None and res_dist <= NEAR_PCT:
        sig = "压力附近"
    elif nr is None:
        sig = "突破(上方无压力)"
    else:
        sig = "中性"
    return {
        "ts_code": ts_code,
        "price": price,
        "poc": poc,
        "poc_dist_pct": (price - poc) / price,
        "support": ns[0],
        "support_dist_pct": sup_dist,
        "support_strength": sup_str,
        "resistance": nr[0] if nr else np.nan,
        "resistance_dist_pct": res_dist if nr else np.nan,
        "resistance_strength": res_str if nr else np.nan,
        "R": R if R is not None else np.nan,
        "n_zones": len(zones),
        "signal": sig,
    }


def run():
    basic = _load_basic()
    universe = _universe()
    print("宇宙: %d 只 (日线>=%d 行)" % (len(universe), MIN_ROWS))
    recs = []
    for i, code in enumerate(universe):
        r = scan_one(code)
        if r is not None:
            name, ind = basic.get(code, ("", ""))
            r["name"] = name
            r["industry"] = ind
            recs.append(r)
        if (i + 1) % 500 == 0:
            print("  ... %d/%d 处理, 有效 %d" % (i + 1, len(universe), len(recs)))
    full = pd.DataFrame(recs)
    print("有效扫描: %d 只" % len(full))

    # ---- 顶部卡片 ----
    near_sup = full[full["support_dist_pct"] <= NEAR_PCT]
    near_res = full[full["resistance_dist_pct"] <= NEAR_PCT]
    valid_R = full[full["R"].notna()]
    best_R = valid_R[valid_R["R"] >= R_MIN]
    avg_R = valid_R["R"].mean()
    print("\n===== Tab4 顶部统计卡片 =====")
    print("总扫描数      : %d" % len(full))
    print("附近支撑(<=%.0f%%) : %d" % (NEAR_PCT * 100, len(near_sup)))
    print("压力附近(<=%.0f%%) : %d" % (NEAR_PCT * 100, len(near_res)))
    print("R最优(>=%.1f)   : %d" % (R_MIN, len(best_R)))
    print("平均盈亏比 R    : %.2f" % avg_R)

    # ---- 三张表 ----
    t_sup = near_sup.sort_values("support_dist_pct").reset_index(drop=True)
    t_res = near_res.sort_values("resistance_dist_pct").reset_index(drop=True)
    t_R = best_R.sort_values("R", ascending=False).reset_index(drop=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = pd.Timestamp.now().strftime("%Y%m%d")
    full.to_csv(os.path.join(OUT_DIR, "scan_%s.csv" % stamp), index=False,
                encoding="utf-8-sig")
    t_sup.to_csv(os.path.join(OUT_DIR, "scan_near_support_%s.csv" % stamp),
                 index=False, encoding="utf-8-sig")
    t_res.to_csv(os.path.join(OUT_DIR, "scan_near_resistance_%s.csv" % stamp),
                 index=False, encoding="utf-8-sig")
    t_R.to_csv(os.path.join(OUT_DIR, "scan_best_R_%s.csv" % stamp),
               index=False, encoding="utf-8-sig")

    print("\n===== 支撑位附近 Top15 (按距支撑由近到远) =====")
    _show(t_sup.head(15))
    print("\n===== 压力位附近 Top15 (按距压力由近到远) =====")
    _show(t_res.head(15))
    print("\n===== 盈亏比(R)最优 Top15 (按 R 从高到低) =====")
    _show(t_R.head(15))
    print("\n输出目录: %s" % os.path.abspath(OUT_DIR))
    return full, t_sup, t_res, t_R


def _show(df):
    cols = ["ts_code", "name", "price", "support", "support_dist_pct",
            "resistance", "resistance_dist_pct", "R", "signal"]
    cols = [c for c in cols if c in df.columns]
    with pd.option_context("display.max_columns", None,
                           "display.width", 200, "display.float_format",
                           lambda x: "%.4f" % x):
        print(df[cols].to_string(index=False))


if __name__ == "__main__":
    run()
