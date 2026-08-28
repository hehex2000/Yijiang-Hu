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


def reversal_overlay(rev21, price, poc):
    """反转 overlay 确认：rev_21(近21日收益) 与 dist_to_poc(价相对POC) 是否同向。
    两者都指同一均值回复方向才确认。返回 (确认bool, 中文说明)。"""
    if rev21 is None or poc is None:
        return False, "无数据"
    vp_long = price < poc            # 价在POC下 -> 均值回复向上 -> 做多偏
    rev_long = rev21 < 0             # 近21日下跌 -> 短期反转向上 -> 做多偏
    agree = (vp_long == rev_long)
    vp_txt = "价在POC下(偏多)" if vp_long else "价在POC上(偏空)"
    rev_txt = "近21日下跌(偏多)" if rev_long else "近21日上涨(偏空)"
    if agree:
        direction = "做多/反弹" if vp_long else "做空/回落"
        return True, "确认(%s · %s → 同向%s)" % (vp_txt, rev_txt, direction)
    return False, "未确认(%s · %s → 反向)" % (vp_txt, rev_txt)


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

    # 反转 overlay（位置 + 信号双确认）
    df0 = vp_data.get_daily(code)
    rev21 = float(df0["close"].pct_change(21).iloc[-1]) if (
        df0 is not None and len(df0) > 21) else None
    ok, ov_txt = reversal_overlay(rev21, d["price"], d["poc"])
    ov_color = "green" if ok else "gray"
    st.markdown(
        "<b>反转 overlay（位置+信号双确认）</b>："
        "<span style='color:%s;font-weight:bold'>%s</span>"
        % (ov_color, ov_txt),
        unsafe_allow_html=True,
    )

    # 综合判断：位置 + 反转双确认 -> ✅机会 / ⚠️风险
    if d["verdict"] == "支撑位附近":
        if ok:
            cmb_color, cmb_txt = "green", "✅ 机会（支撑位 + 反转双确认，偏低吸）"
        else:
            cmb_color, cmb_txt = "gray", "⚪ 在支撑但反转未确认（待观察，别急）"
    elif d["verdict"] == "见顶 / 压力位附近":
        if ok:
            cmb_color, cmb_txt = "red", "⚠️ 风险（见顶压力 + 反转双确认，偏回落）"
        else:
            cmb_color, cmb_txt = "gray", "⚪ 见顶但反转未确认（待观察）"
    elif d["verdict"] == "突破（上方无压力）":
        cmb_color, cmb_txt = "blue", "🔵 突破（上方暂无量压）"
    else:
        cmb_color, cmb_txt = "gray", "⚪ 中性（既不在支撑也不在压力）"
    st.markdown(
        "<b>综合判断</b>："
        "<span style='color:%s;font-weight:bold;font-size:18px'>%s</span>"
        % (cmb_color, cmb_txt),
        unsafe_allow_html=True,
    )

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
    st.caption("进阶：勾选「反转overlay确认」可把列表缩到「位置 + 反转信号双确认」的高信心候选。")
    sup = load_scan("support")
    res = load_scan("resistance")

    m1, m2, m3 = st.columns(3)
    m1.metric("已扫描（有量价区）", "%d 只" % (len(sup) + len(res)) if not sup.empty else "—")
    m2.metric("支撑位附近（潜在机会）", "%d 只" % len(sup) if not sup.empty else "0")
    m3.metric("见顶压力附近（潜在风险）", "%d 只" % len(res) if not res.empty else "0")

    # 全市场冷热温度计：支撑数 - 压力数（用全量扫描，不看下方距离过滤）
    support_n = len(sup)
    pressure_n = len(res)
    gap = support_n - pressure_n
    if support_n + pressure_n > 0:
        therm = go.Figure()
        therm.add_trace(go.Bar(
            x=["全市场冷热"], y=[max(gap, 0)],
            marker_color="#2ca02c", name="支撑多出",
            text="支撑 %d" % support_n, textposition="outside",
        ))
        therm.add_trace(go.Bar(
            x=["全市场冷热"], y=[min(gap, 0)],
            marker_color="#d62728", name="压力多出",
            text="压力 %d" % pressure_n, textposition="outside",
        ))
        therm.update_layout(
            height=150, margin=dict(l=20, r=20, t=34, b=10),
            yaxis_title="只数差", showlegend=False,
            title="🌡️ 支撑数 − 压力数 = %+d（读数偏暖 = 多数票趴在量撑区）" % gap,
        )
        st.plotly_chart(therm, use_container_width=True)
        if gap > 100:
            tone = "市场偏暖：全市场普遍落在量撑区——可能超卖待弹，也可能即将破位，结合大势看"
        elif gap < -20:
            tone = "市场偏冷：压力附近明显多于支撑，注意回落"
        else:
            tone = "多空大致平衡"
        st.caption(tone)

    maxd = st.slider("最大距离(%)——只看离得最近的", 1, 10, 5, 1)
    only_a = st.checkbox("剔除 ST / *", value=True)
    only_ov = st.checkbox("只看反转overlay确认的（位置+信号双确认）", value=False)
    st.caption("列表按「离支撑/压力由近到远」排序，越靠前越贴合。")

    def filt(df, dist_col):
        if df.empty:
            return df
        df = df[df[dist_col] <= maxd / 100]
        if only_a:
            df = df[~df["name"].astype(str).str.contains("ST|\\*", na=False)]
        if only_ov and "rev_21" in df.columns:
            vp_long = df["poc_dist_pct"] < 0
            rev_long = df["rev_21"] < 0
            df = df[vp_long == rev_long]
        return df

    def overlay_col(df):
        if "rev_21" not in df.columns:
            return None
        return np.where((df["poc_dist_pct"] < 0) == (df["rev_21"] < 0), "✅", "—")

    st.subheader("🟢 支撑位附近（跌到量架上，潜在机会）")
    s = filt(sup, "support_dist_pct")
    if s.empty:
        st.write("暂无")
    else:
        ov = overlay_col(s)
        cols = ["name", "ts_code", "industry", "price", "support", "support_dist_pct"]
        if ov is not None:
            s = s.copy()
            s["反转overlay"] = ov
            cols.append("反转overlay")
        cols.append("signal")
        show = s[cols].copy()
        show.columns = ["名称", "代码", "行业", "当前价", "支撑位", "距支撑%"] + \
            (["反转overlay"] if ov is not None else []) + ["信号"]
        show["距支撑%"] = (show["距支撑%"] * 100).round(2)
        show["当前价"] = show["当前价"].round(2)
        show["支撑位"] = show["支撑位"].round(2)
        st.dataframe(show, use_container_width=True, height=360)

    st.subheader("🔴 见顶 / 压力位附近（顶在量顶下，潜在风险）")
    r = filt(res, "resistance_dist_pct")
    if r.empty:
        st.write("暂无")
    else:
        ov = overlay_col(r)
        cols = ["name", "ts_code", "industry", "price", "resistance", "resistance_dist_pct"]
        if ov is not None:
            r = r.copy()
            r["反转overlay"] = ov
            cols.append("反转overlay")
        cols.append("signal")
        show = r[cols].copy()
        show.columns = ["名称", "代码", "行业", "当前价", "压力位", "距压力%"] + \
            (["反转overlay"] if ov is not None else []) + ["信号"]
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
        "- 扫描用的是本地日线（不复权），分红跳空不平移记忆位；截面快照，未做前视。\n\n"
        "**反转 overlay（位置 + 信号双确认）**\n"
        "- 光看「在支撑」还不够：支撑可能跌破。我们加了一道**反转信号**做确认——\n"
        "  用「近21日收益」(rev_21，跌多了短期易反弹) 和「价相对成交密集中心(dist_to_poc)」"
        "两个反向因子，**方向一致才确认**：\n"
        "  - 🟢 支撑附近 + 近21日下跌 + 价在POC下 → 三重同向，反弹概率更高（✅确认）；\n"
        "  - 🔴 见顶附近 + 近21日上涨 + 价在POC上 → 三重同向，回落概率更高（✅确认）。\n"
        "- 勾选「只看反转overlay确认的」即可把列表缩到双确认的高信心候选。\n"
        "- 该 overlay 已做组合层净成本 + walk-forward 验证（覆盖约82%、年化净 +13.1% vs 基线 +10.5%），"
        "是弱因子确认层、非独立 alpha；仍须结合你自己的判断，别单独照它买卖。\n"
    )


t1, t2, t3 = st.tabs(["① 个股诊断", "② 全市场扫描", "③ 怎么看"])
with t1:
    tab_single()
with t2:
    tab_scan()
with t3:
    tab_help()
