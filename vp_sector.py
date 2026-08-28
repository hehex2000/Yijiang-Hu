# -*- coding: utf-8 -*-
"""
vp_sector: Volume Profile 板块热力（计划 M4，对标 Kara BV1tgVs6SE4C Tab5 板块热力）

实现：对 31 个申万一级行业指数（sw_industry_daily）各自算 Volume Profile，
      把"行业技术位置"聚合成热力。复用平台已有 sw_industry_daily（无需额外下载映射）。

指标语义（对齐 M2 结论：VP 预测短期均值回复，非方向突破）：
  - 价量密集区(value area) VA=[va_low, va_high]，POC=最大密集价。
  - 价位在 VA 内的相对位置 pos = (price-va_low)/(va_high-va_low) ∈ (0,1)。
  - 支撑触战比(绿) = 1 - pos  = 价位下方价值区质量占比（"支撑质量"）
  - 压力触战比(红) = pos       = 价位上方价值区质量占比（"压力质量"）
  - 状态: pos<0.5 -> 偏多(价在价值区下半, 均值回复向上); pos>0.5 -> 偏空; ≈0.5 -> 中性
  - 另给最近支撑/压力价位与偏离%供核对。

无前视：profile 只用 <=t 历史（同因子层口径）。
输出：data/results/volume_profile/sector_heatmap_<date>.csv + 排序表。
"""
import os
import sqlite3

import numpy as np
import pandas as pd

import config
import vp_core

# ---- 配置 ----
WINDOW = 250          # 密集区回看交易日（与 M3 扫描一致）
N_BINS = 80
SMOOTH = 2.0
DB_PATH = config.DATA.get("local_db_path", "")
OUT_DIR = os.path.join("data", "results", "volume_profile")

# 申万一级指数代码 -> 名称（2021 版，31 个）
SW_L1 = {
    "801010.SI": "农林牧渔", "801030.SI": "基础化工", "801040.SI": "钢铁",
    "801050.SI": "有色金属", "801080.SI": "电子", "801110.SI": "家用电器",
    "801120.SI": "食品饮料", "801130.SI": "纺织服饰", "801140.SI": "轻工制造",
    "801150.SI": "医药生物", "801160.SI": "公用事业", "801170.SI": "交通运输",
    "801180.SI": "房地产", "801200.SI": "商贸零售", "801210.SI": "社会服务",
    "801230.SI": "综合", "801710.SI": "建筑材料", "801720.SI": "建筑装饰",
    "801730.SI": "电力设备", "801740.SI": "国防军工", "801750.SI": "计算机",
    "801760.SI": "传媒", "801770.SI": "通信", "801780.SI": "银行",
    "801790.SI": "非银金融", "801880.SI": "汽车", "801890.SI": "机械设备",
    "801950.SI": "煤炭", "801960.SI": "石油石化", "801970.SI": "环保",
    "801980.SI": "美容护理",
}


def load_index(ts_code):
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            "SELECT trade_date, open, high, low, close, vol FROM sw_industry_daily "
            "WHERE ts_code=? ORDER BY trade_date ASC", conn, params=(ts_code,)
        )
    finally:
        conn.close()
    return df


def _value_area(centers, sm, coverage=0.70):
    """从平滑密度 sm（索引=centers 价）算价值区 [va_low, va_high]（覆盖 70% 量）。"""
    tot = sm.sum()
    if tot <= 0:
        return np.nan, np.nan
    order = np.argsort(sm)[::-1]
    cum = 0.0
    idxs = []
    for i in order:
        cum += sm[i]
        idxs.append(i)
        if cum >= coverage * tot:
            break
    idxs = np.array(sorted(idxs))
    return centers[idxs].min(), centers[idxs].max()


def profile_one(ts_code):
    df = load_index(ts_code)
    if df is None or len(df) < WINDOW:
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
    va_low, va_high = _value_area(centers, sm)
    if not np.isfinite(va_low) or not np.isfinite(va_high) or va_high <= va_low:
        return None
    supports = [(p, s) for p, s in zones if p < price]
    resists = [(p, s) for p, s in zones if p > price]
    ns = max(supports) if supports else None
    nr = min(resists) if resists else None
    pos = (price - va_low) / (va_high - va_low)        # 0..1
    pos = min(1.0, max(0.0, pos))
    green = 1 - pos     # 支撑触战比
    red = pos           # 压力触战比
    status = "偏多" if pos < 0.45 else ("偏空" if pos > 0.55 else "中性")
    return {
        "ts_code": ts_code,
        "name": SW_L1.get(ts_code, ts_code),
        "price": price,
        "poc": poc,
        "va_low": va_low,
        "va_high": va_high,
        "nearest_support": ns[0] if ns else np.nan,
        "support_dist_pct": (price - ns[0]) / price if ns else np.nan,
        "nearest_resistance": nr[0] if nr else np.nan,
        "resistance_dist_pct": (nr[0] - price) / price if nr else np.nan,
        "pos_in_va": pos,
        "support_ratio": green,     # 绿
        "resistance_ratio": red,    # 红
        "status": status,
    }


def run():
    recs = []
    for code in SW_L1:
        r = profile_one(code)
        if r is not None:
            recs.append(r)
    df = pd.DataFrame(recs)
    print("有效行业指数: %d / 31" % len(df))

    # 排序：偏多优先（green 高），其次按 support_ratio 降序
    df = df.sort_values(["support_ratio"], ascending=False).reset_index(drop=True)

    # ---- 概览 ----
    n_long = (df["status"] == "偏多").sum()
    n_short = (df["status"] == "偏空").sum()
    n_mid = (df["status"] == "中性").sum()
    print("\n===== Tab5 板块热力概览 =====")
    print("偏多行业: %d | 偏空: %d | 中性: %d" % (n_long, n_short, n_mid))

    # ---- 明细表（带绿/红条文字化）----
    print("\n===== 行业技术位置（按支撑触战比降序）=====")
    with pd.option_context("display.max_rows", None, "display.width", 200,
                           "display.float_format", lambda x: "%.4f" % x):
        show = df[["name", "price", "poc", "va_low", "va_high",
                   "support_ratio", "resistance_ratio", "status",
                   "support_dist_pct", "resistance_dist_pct"]].copy()
        show["green_bar"] = (df["support_ratio"] * 20).round().astype(int)
        show["red_bar"] = (df["resistance_ratio"] * 20).round().astype(int)
        for _, row in show.iterrows():
            g = "█" * int(row["green_bar"])
            r = "█" * int(row["red_bar"])
            print("%-6s %8.2f | 支撑触战%s(%.2f) 压力触战%s(%.2f) | %s"
                  % (row["name"], row["price"], g, row["support_ratio"],
                     r, row["resistance_ratio"], row["status"]))

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = pd.Timestamp.now().strftime("%Y%m%d")
    out = os.path.join(OUT_DIR, "sector_heatmap_%s.csv" % stamp)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print("\n输出: %s" % os.path.abspath(out))
    return df


if __name__ == "__main__":
    run()
