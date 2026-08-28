# -*- coding: utf-8 -*-
"""
vp_dashboard: Volume Profile 最简看板
只做一件事：把"量价密集区"翻译成人话——
  - 个股：当前价跌到成交量密集的"支撑位"附近？还是顶在"压力位/见顶"附近？
  - 全市场：哪些票在支撑位（潜在机会）、哪些在压力位（潜在见顶）？
不堆指标、不装专业。前视/成本等严谨问题在代码与 CSV 里，看板只给结论。
"""
import os
import sqlite3

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
import vp_data
import vp_core

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "data", "results", "volume_profile")
DB_PATH = config.DATA.get("local_db_path", "")
NEAR_PCT = 0.03

st.set_page_config(page_title="Volume Profile 量价支撑/见顶看板", layout="wide")
st.title("Volume Profile 量价支撑 / 见顶看板")
st.caption(
    "原理一句话：把一段时间每根 K 线的成交量按价位堆起来，堆得最高的价位就是"
    "「大多数人在这成交」的地方。价格跌回这种密集区=下方有量撑（支撑位）；"
    "涨到这种密集区=上方有量压（压力位/容易见顶）。本看板只告诉你「现在在哪」。"
)


@st.cache_data
def load_stock_list():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT ts_code, name, industry FROM stock_basic", conn)
    finally:
        conn.close()
    df["label"] = df["name"] + " (" + df["ts_code"] + ")"
    return df.sort_values("label")


@st.cache_data
def load_scan(kind):
    if kind == "support":
        fs = [f for f in os.listdir(OUT_DIR) if f.startswith("scan_near_support_")]
    else:
        fs = [f for f in os.listdir(OUT_DIR) if f.startswith("scan_near_resistance_")]
    if not fs:
        return pd.DataFrame()
    df = pd.read_csv(os.path.join(OUT_DIR, sorted(fs)[-1]), encoding="utf-8-sig")
    return df


def diagnose(ts_code, window):
    df = vp_data.get_daily(ts_code)
    if df is None or len(df) < window:
        return None
    price = float(df["close"].iloc[-1])
    w = df.tail(window).reset_index(drop=True)
    res = vp_core.volume_profile(w, n_bins=80, smooth_sigma=2.0)
    if res is None:
        return None
    centers, raw, sm = res
    zones, poc = vp_core.detect_zones(centers, sm)
    if not zones:
        return None
    supports = [p for p, _ in zones if p < price]
    resists = [p for p, _ in zones if p > price]
    support = max(supports) if supports else None
    resistance = min(resists) if resists else None
    sup_dist = (price - support) / price if support else None
    res_dist = (resistance - price) / price if resistance else None

    if support and sup_dist <= NEAR_PCT:
        verdict, color = "支撑位附近", "green"
        tip = "价格跌到成交量密集区，下方有「量撑」，潜在机会区"
    elif resistance and res_dist <= NEAR_PCT:
        verdict, color = "见顶 / 压力位附近", "red"
        tip = "价格顶在成交量密集区，上方有「量压」，容易见顶/回调"
    elif resistance is None:
        verdict, color = "突破（上方无压力）", "blue"
        tip = "现价已在所有密集区之上，上方暂无量压"
    else:
        verdict, color = "中性", "gray"
        tip = "既不在支撑也不在压力附近"
    return dict(
        ts_code=ts_code, price=price, poc=poc, support=support,
        sup_dist=sup_dist, resistance=resistance, res_dist=res_dist,
        verdict=verdict, color=color, tip=tip,
    )


def price_chart(ts_code, window, d):
    df = vp_data.get_daily(ts_code).tail(window).reset_index(drop=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["close"], name="收盘价",
        line=dict(color="#1f77b4", width=1.5),
    ))
    lines = []
    if d["support"] is not None:
        lines.append(("支撑位", d["support"], "green"))
    if d["resistance"] is not None:
        lines.append(("压力位/见顶", d["resistance"], "red"))
    if d["poc"] is not None:
        lines.append(("POC(最密成交价)", d["poc"], "orange"))
    for name, lvl, c in lines:
        fig.add_hline(
            y=lvl, line_color=c, line_dash="dash",
            annotation_text=name, annotation_position="right",
        )
    fig.update_layout(
        height=380, margin=dict(l=30, r=30, t=20, b=20),
        yaxis_title="价格", xaxis_title="交易日(近%d根)" % window,
        showlegend=False,
    )
    return fig


def tab_single():
    st.header("① 个股：现在在支撑位还是见顶压力位？")
    slist = load_stock_list()
    sel = st.selectbox("选一只股票", slist["label"].tolist(), index=0)
    code = slist.loc[slist["label"] == sel, "ts_code"].iloc[0]
    window = st.slider("看多长（交易日）", 60, 500, 250, 10)
    d = diagnose(code, window)
    if d is None:
        st.warning("数据不足，算不出来。")
        return
    badge = (
        "<span style='background:%s;color:white;padding:4px 12px;"
        "border-radius:6px;font-size:20px;font-weight:bold'>%s</span>"
        % (d["color"], d["verdict"])
    )
    st.markdown(badge, unsafe_allow_html=True)
    st.caption(d["tip"])

    c1, c2, c3 = st.columns(3)
    c1.metric("当前价", "%.2f" % d["price"])
    c2.metric(
        "支撑位",
        "%.2f" % d["support"] if d["support"] else "—",
        "%.1f%%" % (d["sup_dist"] * 100) if d["sup_dist"] is not None else "",
    )
    c3.metric(
        "压力位/见顶",
        "%.2f" % d["resistance"] if d["resistance"] else "—（突破）",
        "%.1f%%" % (d["res_dist"] * 100) if d["res_dist"] is not None else "",
    )

    st.plotly_chart(price_chart(code, window, d), use_container_width=True)

    parts = ["当前价 %.2f。" % d["price"]]
    if d["support"]:
        parts.append(
            "下方支撑位 %.2f（差 %.1f%%），跌到这里大概率是「量撑」区。"
            % (d["support"], d["sup_dist"] * 100)
        )
    if d["resistance"]:
        parts.append(
            "上方压力位 %.2f（差 %.1f%%），涨到这里容易见顶。"
            % (d["resistance"], d["res_dist"] * 100)
        )
    elif d["support"]:
        parts.append("上方已无密集压力，属突破状态。")
    st.info("".join(parts))


def tab_scan():
    st.header("② 全市场扫描：谁在支撑位（机会）？谁在见顶（风险）？")
    sup = load_scan("support")
    res = load_scan("resistance")

    m1, m2, m3 = st.columns(3)
    m1.metric("已扫描（有量价区）", "%d 只" % (len(sup) + len(res)) if not sup.empty else "—")
    m2.metric("支撑位附近（潜在机会）", "%d 只" % len(sup) if not sup.empty else "0")
    m3.metric("见顶压力附近（潜在风险）", "%d 只" % len(res) if not res.empty else "0")

    maxd = st.slider("最大距离(%)——只看离得最近的", 1, 10, 5, 1)
    only_a = st.checkbox("剔除 ST / *", value=True)
    st.caption("列表按「离支撑/压力由近到远」排序，越靠前越贴合。")

    def filt(df, dist_col):
        if df.empty:
            return df
        df = df[df[dist_col] <= maxd / 100]
        if only_a:
            df = df[~df["name"].astype(str).str.contains("ST|\\*", na=False)]
        return df

    st.subheader("🟢 支撑位附近（跌到量架上，潜在机会）")
    s = filt(sup, "support_dist_pct")
    if s.empty:
        st.write("暂无")
    else:
        show = s[["name", "ts_code", "industry", "price", "support",
                  "support_dist_pct", "signal"]].copy()
        show.columns = ["名称", "代码", "行业", "当前价", "支撑位", "距支撑%", "信号"]
        show["距支撑%"] = (show["距支撑%"] * 100).round(2)
        show["当前价"] = show["当前价"].round(2)
        show["支撑位"] = show["支撑位"].round(2)
        st.dataframe(show, use_container_width=True, height=360)

    st.subheader("🔴 见顶 / 压力位附近（顶在量顶下，潜在风险）")
    r = filt(res, "resistance_dist_pct")
    if r.empty:
        st.write("暂无")
    else:
        show = r[["name", "ts_code", "industry", "price", "resistance",
                  "resistance_dist_pct", "signal"]].copy()
        show.columns = ["名称", "代码", "行业", "当前价", "压力位", "距压力%", "信号"]
        show["距压力%"] = (show["距压力%"] * 100).round(2)
        show["当前价"] = show["当前价"].round(2)
        show["压力位"] = show["压力位"].round(2)
        st.dataframe(show, use_container_width=True, height=360)


def tab_help():
    st.header("③ 这东西怎么看（大白话）")
    st.markdown(
        "**它在算什么**\n"
        "- 把最近 N 天每根 K 线的成交量，按它走过的价位区间分摊，堆成一条"
        "「成交量 vs 价位」的直方图。\n"
        "- 堆得最高的几个价位 = 大多数人曾经在这成交，叫「量价密集区」。\n"
        "- 现价**下方**最近的密集区 = **支撑位**（跌到这里有人接）；现价**上方**"
        "最近的密集区 = **压力位**（涨到这里有人砸）。\n\n"
        "**两种结论**\n"
        "- 🟢 **支撑位附近**：价格已跌回密集区，下方有「量撑」——传统上是观察低吸的区域。\n"
        "- 🔴 **见顶 / 压力位附近**：价格涨到密集区，上方有「量压」——传统上是容易回落/见顶的区域。\n\n"
        "**诚实提醒（不是免责，是事实）**\n"
        "- 它只回答「**现在价格在哪**」，不保证「一定会弹 / 一定会跌」。量是历史成交，不是未来托单。\n"
        "- 这是**看板 / 选股漏斗**：先圈出「在支撑、在压力」的票，再拿你自己的其它信号"
        "（基本面、趋势、我们做好的反转 overlay）去筛，别单独照它买卖。\n"
        "- 全市场「支撑附近」远多于「压力附近」时，说明大盘普遍趴在量架上"
        "（可能超卖、也可能要破位），结合大势看。\n"
        "- 扫描用的是本地日线（不复权），分红跳空不平移记忆位；截面快照，未做前视。\n"
    )


t1, t2, t3 = st.tabs(["① 个股诊断", "② 全市场扫描", "③ 怎么看"])
with t1:
    tab_single()
with t2:
    tab_scan()
with t3:
    tab_help()
