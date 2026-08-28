# -*- coding: utf-8 -*-
"""
vp_dashboard: Volume Profile Streamlit 仪表盘（计划 P4 / M6）
对标 Kara 视频 6 Tab 展示层，但按我们实际产物落地，强调诚实红线：
  - 日线级近似，非精确支撑压力
  - 因子仅为弱描述工具，非已验证 alpha（见因子验证总览红牌）
四个 Tab：
  1) 个股 Volume Profile 分析（实时算，vp_data+vp_core）
  2) 因子验证总览（IC / 宽成本净LS / walk-forward / 冗余 / overlay + 红牌结论）
  3) 板块热力（31 申万一级）
  4) 市场扫描（全A 支撑/压力/R 最优）
运行：streamlit run vp_dashboard.py  （需 managed venv 装 streamlit+plotly）
"""
import os
import sqlite3

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import config
import vp_data
import vp_core
import vp_factor

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "data", "results", "volume_profile")
DB_PATH = config.DATA.get("local_db_path", "")

st.set_page_config(page_title="Volume Profile 仪表盘", layout="wide")
st.title("Volume Profile（密集成交区）分析仪表盘")

st.markdown(
    "> **诚实红线**：日线高低价区间宽 → VPVR 为**日线级近似**（峰值偏软、箱宽敏感），"
    "非精确支撑压力位；因子验证结论显示 dist_to_poc 为**弱描述因子**（|IC|≈0.06，64% 冗余于 MA 距离），"
    "仅作反转信号廉价确认 overlay（+2.6pp/年），**非独立 alpha**。下方数字均为已跑完验证的产物，非实时重算策略收益。"
)


@st.cache_data
def load_stock_list():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT ts_code, name FROM stock_basic WHERE ts_code LIKE '%.SH' OR ts_code LIKE '%.SZ'",
        conn,
    )
    conn.close()
    df["label"] = df["ts_code"] + " " + df["name"]
    return df.sort_values("ts_code").reset_index(drop=True)


def read_csv(name):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


# ───────────────────────── Tab 1: 个股 VP 实时分析 ─────────────────────────
def tab_single():
    st.header("① 个股 Volume Profile 分析（实时计算）")
    sl = load_stock_list()
    sel = st.selectbox("选择标的", sl["label"].tolist(), index=0)
    ts_code = sel.split(" ")[0]
    c1, c2, c3 = st.columns(3)
    window = c1.slider("回看交易日", 60, 250, 120, 10)
    n_bins = c2.slider("分箱数", 40, 160, 80, 10)
    sigma = c3.slider("平滑 σ", 0.5, 4.0, 2.0, 0.5)
    adjust = st.radio("复权口径", ["raw", "hfq"], horizontal=True,
                      help="S/R 检测用 raw（真实成交价记忆）；因子可比用 hfq")
    df = vp_data.get_daily(ts_code, adjust=adjust)
    if df is None or len(df) < window + 5:
        st.warning("数据不足，无法计算")
        return
    w = df.tail(window).reset_index(drop=True)
    res = vp_core.volume_profile(w, n_bins=n_bins, smooth_sigma=sigma)
    if res is None:
        st.warning("volume_profile 返回 None")
        return
    centers, raw, sm = res
    zones, poc = vp_core.detect_zones(centers, sm)
    va_low, va_high = vp_factor.value_area_band(raw, centers)
    last_close = float(w["close"].iloc[-1])
    dist_to_poc = (last_close - poc) / poc if poc > 0 else np.nan
    supports = [p for p, _ in zones if p < last_close]
    resist = [p for p, _ in zones if p > last_close]
    near_sup = max(supports) if supports else np.nan
    near_res = min(resist) if resist else np.nan
    sup_dist = (last_close - near_sup) / last_close * 100 if near_sup == near_sup else np.nan
    va_pass = 1.0 if last_close >= va_low else 0.0

    m = st.columns(4)
    m[0].metric("dist_to_poc", f"{dist_to_poc*100:.2f}%")
    m[1].metric("最近支撑距离%", f"{sup_dist:.2f}%" if sup_dist == sup_dist else "NA")
    m[2].metric("VA 通过(1=在VA上沿上)", f"{va_pass:.0f}")
    m[3].metric("POC 价", f"{poc:.2f}")

    st.markdown(f"**POC**={poc:.2f} ｜ **价值区**=[{va_low:.2f}, {va_high:.2f}] ｜ "
                f"**最近支撑**={near_sup:.2f} ｜ **最近压力**={near_res:.2f} ｜ 显著区数={len(zones)}")

    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=w["date"], y=w["close"], mode="lines",
                                   name="close", line=dict(color="#1f77b4", width=1.2)))
    for lbl, val, col in [("POC", poc, "red"), ("VA高", va_high, "green"),
                          ("VA低", va_low, "green"), ("最近支撑", near_sup, "orange"),
                          ("最近压力", near_res, "purple")]:
        if val == val:
            fig_price.add_hline(y=val, line_dash="dash", line_color=col,
                                annotation_text=lbl, annotation_font_size=10)
    fig_price.update_layout(height=360, title="价格 + 关键价位",
                            xaxis_title="trade_date", yaxis_title="price", margin=dict(t=30))
    st.plotly_chart(fig_price, use_container_width=True)

    fig_vp = go.Figure()
    fig_vp.add_trace(go.Bar(x=sm, y=centers, orientation="h",
                            marker=dict(color=sm, colorscale="Blues"), name="成交量分布"))
    fig_vp.add_vline(x=sm.max(), line_color="red", annotation_text="POC")
    fig_vp.update_layout(height=420, title="Volume Profile（水平直方图，价 vs 成交量）",
                         xaxis_title="成交量(平滑)", yaxis_title="price", margin=dict(t=30))
    st.plotly_chart(fig_vp, use_container_width=True)
    st.caption("算法四步：range-weighted 分箱 → 加权 → scipy 高斯平滑 → find_peaks 检测峰值。"
               "POC=成交量最大箱中心；价值区=从 POC 向两侧覆盖 70% 成交量的区间。")


# ───────────────────────── Tab 2: 因子验证总览 ─────────────────────────
def tab_validation():
    st.header("② 因子验证总览（anti-overfitting 六闸产物）")
    st.error("🔴 **红牌结论**：dist_to_poc 真实但弱（|IC|≈0.06）+ regime-dependent + "
             "64% 冗余于 MA距离 → **不独立入库**，仅作 rev_21 反转信号的确认 overlay（+2.6pp/年）。", icon="🚫")

    st.subheader("IC / 分层（Gate1，top-N 小宇宙，含幸存者偏差）")
    ic = read_csv("ic_report.csv")
    if ic is not None:
        st.dataframe(ic, use_container_width=True)
    else:
        st.warning("ic_report.csv 缺失")

    st.subheader("宽成本组合层净 LS（Gate2，全市场面板，扣 0.60%/月）")
    net = read_csv("m2_wide_net_portfolio.csv")
    if net is not None:
        st.dataframe(net, use_container_width=True)
        alive = net[net["alive"] == "Y"]["factor"].tolist()
        st.success("存活因子(alive=Y)：" + (", ".join(alive) if alive else "无"))
    else:
        st.warning("m2_wide_net_portfolio.csv 缺失")

    st.subheader("Walk-forward OOS（Gate4）")
    eq = read_csv("wf_equity.csv")
    yt = read_csv("wf_year_table.csv")
    wf = read_csv("wf_rolling_folds.csv")
    if eq is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eq["date"], y=eq["cum_eq"], mode="lines", name="累计净净值"))
        fig.update_layout(height=320, title="dist_to_poc 净LS 累计净值(滚动OOS)",
                          yaxis_title="cum_eq", margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)
    if yt is not None:
        st.markdown("**逐年净LS**")
        st.dataframe(yt, use_container_width=True)
    if wf is not None:
        st.markdown("**滚动折叠（train60/test12，仅test期OOS）**")
        st.dataframe(wf, use_container_width=True)

    st.subheader("冗余检验（Gate5）：与 rev/ma_dist/mom 横截面相关 + 残差增量")
    corr = read_csv("redundancy_corr_raw.csv")
    ric = read_csv("redundancy_ic.csv")
    inc = read_csv("redundancy_incremental.csv")
    if corr is not None:
        cm = corr.set_index(corr.columns[0])
        fig = go.Figure(go.Heatmap(z=cm.values, x=cm.columns.tolist(),
                                   y=cm.index.tolist(), colorscale="RdBu", zmid=0))
        fig.update_layout(height=360, title="Spearman 相关矩阵", margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)
    if ric is not None:
        st.dataframe(ric, use_container_width=True)
    if inc is not None:
        st.dataframe(inc, use_container_width=True)
        st.info("残差 IC≈0 → 不提供超出 MA距离/长反转 的增量（冗余）")

    st.subheader("Overlay 确认（Gate6）：dist_to_poc 确认 rev_21")
    ov = read_csv("overlay_confirm.csv")
    if ov is not None:
        st.dataframe(ov, use_container_width=True)


# ───────────────────────── Tab 3: 板块热力 ─────────────────────────
def tab_sector():
    st.header("③ 板块热力（31 申万一级，VP 视角）")
    df = read_csv("sector_heatmap_20260828.csv")
    if df is None:
        st.warning("sector_heatmap_20260828.csv 缺失")
        return
    st.dataframe(df, use_container_width=True)
    if "status" in df.columns and "support_dist_pct" in df.columns:
        colmap = {"bullish": "red", "bearish": "green", "neutral": "gray"}
        cols = df["status"].map(lambda s: colmap.get(str(s), "gray"))
        fig = go.Figure(go.Bar(x=df["support_dist_pct"], y=df["name"],
                               orientation="h", marker=dict(color=cols),
                               text=df["status"], textposition="auto"))
        fig.update_layout(height=720, title="距最近支撑距离%（红=偏多/近支撑, 绿=偏空/远支撑）",
                          margin=dict(t=30), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)


# ───────────────────────── Tab 4: 市场扫描 ─────────────────────────
def tab_scan():
    st.header("④ 市场扫描（全A，2026-08-28 快照）")
    near_s = read_csv("scan_near_support_20260828.csv")
    near_r = read_csv("scan_near_resistance_20260828.csv")
    best_r = read_csv("scan_best_R_20260828.csv")
    kind = st.radio("扫描类型", ["近支撑", "近压力", "R最优"], horizontal=True)
    d = {"近支撑": near_s, "近压力": near_r, "R最优": best_r}[kind]
    if d is None:
        st.warning("对应 CSV 缺失")
        return
    st.dataframe(d, use_container_width=True)
    st.caption("signal/support_strength/resistance_strength/R 为 VPVR 描述性指标，"
               "非买卖建议；R=潜在收益/风险比（带 RISK_FLOOR 防除零）。")


tabs = st.tabs(["① 个股分析", "② 因子验证总览", "③ 板块热力", "④ 市场扫描"])
with tabs[0]:
    tab_single()
with tabs[1]:
    tab_validation()
with tabs[2]:
    tab_sector()
with tabs[3]:
    tab_scan()
