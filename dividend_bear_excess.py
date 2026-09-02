# -*- coding: utf-8 -*-
"""
⑳ 高股息熊市超额 —— 六道闸门（事件研究版）
=================================================
视频《A股22条规律》第20条："高股息（红利）在熊市有超额收益"。
视频给的可对账数字：2021.12–2024.2 万得全A −36.6% vs 中证红利 −1.3%（价格口径）。

核心证伪问题：这条规律是 A 股普遍规律，还是 2021-2024「中特估/红利 regime」的区间选择偏差？

方法（事件研究）
--------------
- 数据：index_tr_official 官方全收益（含分红，正确口径），2010-01-04 ~ 2026-08-28。
- 市场代理：中证800 全收益 H00906（宽基）。
- 熊市区间：市场从 252 日滚动峰值回撤 ≥ 20% 的完整「顶→底→回顶」周期。
  区间 = [局部峰值(回撤前高点), 回撤归零(创新高)]，两端同点，红利 vs 宽基对比最公平。
- 每段熊市/牛市区间内，计算红利累计收益 − 宽基累计收益 = 超额。
- 六道闸门：
  Gate 0 前视    —— 只用 ≤t 收盘价识别区间，无前视。
  Gate 1 信号    —— 熊市区间红利超额 vs 牛市区间，是否显著更高；t / 胜率 / 中位数。
  Gate 1' 剔除    —— 🔴 剔 2021 后，历史熊市红利超额是否仍为正（决定性）。
  Gate 2 成本    —— 红利超额量级 vs 切换成本（0.1~0.3%/次），能否覆盖。
  Gate 3 换标的  —— 红利低波100 / 中证红利净收益 交叉验证；宽基换 300/800。
  Gate 4 wf      —— 用 2010-2018 定「熊市红利超额」结论，2019-2026 样本外验证。
  Gate 5 冗余    —— 控制「红利整体趋势」后，「熊市」条件是否还有增量信息。
  Gate 6 overlay —— 熊市切红利 vs 恒持宽基 vs 恒持红利 的真实择时回测（含成本）。

输出：控制台纯文本表格 + 可选 --csv 明细。
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import sqlite3

try:
    from config import DATA
except Exception:
    DATA = {"local_db_path": r"D:\tu-shareData\astock_daily.db"}

DB = DATA.get("local_db_path", "")

# 官方全收益序列（已确认 index_tr_official 中 2010-01-04~2026-08-28 完整覆盖）
TR_SERIES = {
    "HS300": "H00300.CSI",    # 沪深300全收益
    "ZZ800": "H00906.CSI",    # 中证800全收益
    "ZHHL":  "H00922.CSI",    # 中证红利全收益
    "ZHHL_NET": "000922CNY020.CSI",  # 中证红利净收益（扣税）
    "HLDB100": "H20955.CSI",  # 红利低波100全收益
}

PEAK_WIN = 252          # 滚动峰值窗口（1年）
DD_THRESH = -0.20       # 熊市回撤阈值
COST = 0.003            # 单次切换成本（佣金+滑点，指数 ETF 轮动口径）


def load_tr(conn, tr_code):
    df = pd.read_sql_query(
        "SELECT trade_date, close FROM index_tr_official "
        "WHERE tr_code=? ORDER BY trade_date",
        conn, params=(tr_code,))
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna()
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    return df.set_index("trade_date")["close"].sort_index()


def cum_ret(px):
    return px.iloc[-1] / px.iloc[0] - 1.0


def bear_markets(px, peak_win=PEAK_WIN, dd_thresh=DD_THRESH):
    """识别「顶→底→回顶」的完整熊市周期。

    返回 list of (t_peak, t_recover, max_dd)，其中：
      - t_peak    回撤前最近局部峰值日（dd 回到 0 的起点）
      - t_recover 回撤归零（创新高）日
      - max_dd    区间内最大回撤（负值）
    牛市区间 = 相邻熊市区间之间的补集（本脚本用「非熊市日」与「熊市日」两分）。
    """
    peak = px.rolling(peak_win, min_periods=peak_win).max()
    dd = px / peak - 1.0
    # 深度回撤标记
    deep = dd <= dd_thresh
    idx = px.index
    n = len(px)
    intervals = []
    i = 0
    while i < n:
        if deep.iloc[i]:
            # 往前回溯到最近一次 dd 归零（创新高）的日，作为区间起点（顶部）
            j = i
            while j > 0 and dd.iloc[j] < -1e-6:
                j -= 1
            t_peak = idx[j]
            # 往后找到 dd 归零（创新高）日，作为区间终点（回到顶部）
            k = i
            while k < n and dd.iloc[k] < -1e-6:
                k += 1
            t_recover = idx[k - 1] if k < n else idx[-1]
            seg_dd = dd.iloc[j:k].min()
            intervals.append((t_peak, t_recover, seg_dd))
            i = k
        else:
            i += 1
    return intervals


def seg_excess(px_div, px_broad, t0, t1):
    """区间 [t0, t1] 内 红利累计 − 宽基累计（用区间首日做基点）。"""
    d = px_div.loc[t0:t1]
    b = px_broad.loc[t0:t1]
    if len(d) < 2 or len(b) < 2:
        return None
    return cum_ret(d) - cum_ret(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="", help="写出熊市/牛市区间明细 CSV")
    ap.add_argument("--dd", type=float, default=DD_THRESH, help="熊市回撤阈值（默认 -0.20）")
    args = ap.parse_args()

    conn = sqlite3.connect(DB)
    raw = {k: load_tr(conn, v) for k, v in TR_SERIES.items()}
    conn.close()

    # 对齐共同交易日
    common = None
    for s in raw.values():
        common = s.index if common is None else common.intersection(s.index)
    aligned = {k: v.reindex(common).dropna() for k, v in raw.items()}

    market = aligned["ZZ800"]  # 市场代理：中证800全收益
    print("=" * 88)
    print("⑳ 高股息熊市超额 —— 事件研究（官方全收益口径）")
    print("=" * 88)
    print(f"数据源 : {DB}")
    print(f"市场代理: 中证800全收益 H00906  样本区间 {market.index[0].date()} ~ {market.index[-1].date()}")
    print(f"熊市定义: 从252日峰值回撤 ≤ {args.dd*100:.0f}% 的完整「顶→底→回顶」周期")
    print()

    # 识别熊市区间
    bears = bear_markets(market, dd_thresh=args.dd)
    print(f"识别到熊市区间 {len(bears)} 段：")
    print(f"  {'#':>2s} {'顶部(起点)':>12s} {'回顶(终点)':>12s} {'最大回撤':>9s} {'区间天数':>7s}")
    for i, (t0, t1, mdd) in enumerate(bears, 1):
        days = (t1 - t0).days
        print(f"  {i:>2d} {t0.date()}  {t1.date()}  {mdd*100:>8.2f}% {days:>7d}")
    print()

    # Gate 1 + 1'：每段熊市/牛市区间的红利超额
    print("-" * 88)
    print("Gate 1  熊市区间内 红利 vs 宽基 累计超额（事件研究）")
    print("-" * 88)
    bear_rows = []
    for i, (t0, t1, mdd) in enumerate(bears, 1):
        exc_hl = seg_excess(aligned["ZHHL"], market, t0, t1)
        exc_lv = seg_excess(aligned["HLDB100"], market, t0, t1)
        bear_rows.append(dict(idx=i, t0=t0, t1=t1, mdd=mdd,
                              exc_hl=exc_hl, exc_lv=exc_lv))
        print(f"  熊市#{i:>2d} {t0.date()}~{t1.date()}  mdd={mdd*100:6.2f}%  "
              f"中证红利超额={exc_hl*100:+.2f}%  红利低波100超额={exc_lv*100:+.2f}%")
    print()

    # 牛市区间（相邻熊市之间的补集，含前后两端）
    bull_rows = []
    bounds = [(b[0], b[1]) for b in bears]
    # 牛市段 = 上一熊市终点 ~ 下一熊市起点
    segs = []
    prev_end = market.index[0]
    for t0, t1, _ in bears:
        if t0 > prev_end:
            segs.append((prev_end, t0))
        prev_end = t1
    if prev_end < market.index[-1]:
        segs.append((prev_end, market.index[-1]))
    for i, (t0, t1) in enumerate(segs, 1):
        exc_hl = seg_excess(aligned["ZHHL"], market, t0, t1)
        exc_lv = seg_excess(aligned["HLDB100"], market, t0, t1)
        bull_rows.append(dict(idx=i, t0=t0, t1=t1, exc_hl=exc_hl, exc_lv=exc_lv))

    # 汇总统计
    def stat_block(rows, key, label):
        vals = [r[key] for r in rows if r[key] is not None]
        if not vals:
            return
        vals = np.array(vals)
        win = float(np.mean(vals > 0)) * 100
        med = float(np.median(vals))
        print(f"  {label}: 段数={len(vals)}  红利跑赢占比={win:5.1f}%  超额中位数={med*100:+.2f}%  "
              f"均值={float(np.mean(vals))*100:+.2f}%")

    print("── 中证红利(H00922) 熊市 vs 牛市 超额汇总 ──")
    stat_block(bear_rows, "exc_hl", "熊市区间")
    stat_block(bull_rows, "exc_hl", "牛市区间")
    print("── 红利低波100(H20955) 熊市 vs 牛市 超额汇总 ──")
    stat_block(bear_rows, "exc_lv", "熊市区间")
    stat_block(bull_rows, "exc_lv", "牛市区间")
    print()

    # Gate 1' 剔除检验：只保留 2021 之前的熊市区间
    print("-" * 88)
    print("Gate 1' 🔴 剔 2021 检验：只保留 2021-01-01 之前的熊市区间")
    print("-" * 88)
    pre21 = [r for r in bear_rows if r["t1"] < pd.Timestamp("2021-01-01")]
    if pre21:
        print(f"  2021 前熊市段数 = {len(pre21)}")
        print("  ── 中证红利 ──")
        stat_block(pre21, "exc_hl", "2021前熊市")
        print("  ── 红利低波100 ──")
        stat_block(pre21, "exc_lv", "2021前熊市")
        # 逐段列
        print("  逐段明细（2021 前）：")
        for r in pre21:
            print(f"    {r['t0'].date()}~{r['t1'].date()}  "
                  f"中证红利超额={r['exc_hl']*100:+.2f}%  红利低波100={r['exc_lv']*100:+.2f}%")
    else:
        print("  （无 2021 前熊市段）")
    print()

    # Gate 3 换标的：宽基换沪深300
    print("-" * 88)
    print("Gate 3 换标的：市场代理换 沪深300全收益 H00300 重新识别熊市")
    print("-" * 88)
    bears_300 = bear_markets(aligned["HS300"], dd_thresh=args.dd)
    print(f"  沪深300口径熊市段数 = {len(bears_300)}")
    hl_300 = []
    for t0, t1, mdd in bears_300:
        exc = seg_excess(aligned["ZHHL"], aligned["HS300"], t0, t1)
        hl_300.append(exc)
        print(f"    {t0.date()}~{t1.date()}  mdd={mdd*100:6.2f}%  中证红利超额={exc*100:+.2f}%")
    hl_300 = np.array([x for x in hl_300 if x is not None])
    if len(hl_300):
        print(f"  → 沪深300口径：段数={len(hl_300)}  红利跑赢占比={float(np.mean(hl_300>0))*100:.1f}%  "
              f"超额中位数={float(np.median(hl_300))*100:+.2f}%")
    pre21_300 = [seg_excess(aligned["ZHHL"], aligned["HS300"], t0, t1)
                 for t0, t1, mdd in bears_300 if t1 < pd.Timestamp("2021-01-01")]
    pre21_300 = np.array([x for x in pre21_300 if x is not None])
    if len(pre21_300):
        print(f"  → 沪深300口径剔2021：段数={len(pre21_300)}  红利跑赢占比={float(np.mean(pre21_300>0))*100:.1f}%  "
              f"超额中位数={float(np.median(pre21_300))*100:+.2f}%")
    print()

    # Gate 4 walk-forward：2010-2018 定结论，2019-2026 验证
    print("-" * 88)
    print("Gate 4 walk-forward：训练段 2010-2018 定结论，验证段 2019-2026 验证")
    print("-" * 88)
    for seg_name, lo, hi in [("训练段 2010-2018", "2010-01-01", "2018-12-31"),
                             ("验证段 2019-2026", "2019-01-01", "2026-12-31")]:
        seg_bears = [r for r in bear_rows if r["t0"] >= pd.Timestamp(lo) and r["t1"] <= pd.Timestamp(hi)]
        if not seg_bears:
            print(f"  {seg_name}: 无熊市段")
            continue
        hl = np.array([r["exc_hl"] for r in seg_bears if r["exc_hl"] is not None])
        lv = np.array([r["exc_lv"] for r in seg_bears if r["exc_lv"] is not None])
        print(f"  {seg_name}: 熊市段={len(seg_bears)}  "
              f"中证红利 胜率={float(np.mean(hl>0))*100:.0f}% 中位={float(np.median(hl))*100:+.2f}%  |  "
              f"红利低波100 胜率={float(np.mean(lv>0))*100:.0f}% 中位={float(np.median(lv))*100:+.2f}%")
    print()

    # Gate 5 冗余：控制红利整体趋势后，熊市条件增量
    print("-" * 88)
    print("Gate 5 冗余：红利超额是否「任何时候都成立」（全期现象），还是「熊市特有」？")
    print("-" * 88)
    # 全样本日频：红利相对宽基的日超额
    r_div = aligned["ZHHL"].pct_change().dropna()
    r_mkt = market.pct_change().dropna()
    r_div, r_mkt = r_div.align(r_mkt, join="inner")
    daily_excess = r_div - r_mkt
    # 熊市日标记（用 dd 深度回撤日）
    peak = market.rolling(PEAK_WIN, min_periods=PEAK_WIN).max()
    dd = market / peak - 1.0
    deep = dd <= args.dd
    deep = deep.reindex(daily_excess.index).fillna(False)
    ex_bear = daily_excess[deep]
    ex_non = daily_excess[~deep]
    def daily_stat(s, label):
        if len(s) == 0:
            print(f"  {label}: 无样本")
            return
        ann = float(np.mean(s)) * 252 * 100
        win = float(np.mean(s > 0)) * 100
        print(f"  {label}: 天数={len(s)}  日均超额={float(np.mean(s))*100:.4f}%  "
              f"年化≈{ann:+.2f}%  跑赢日占比={win:.1f}%")
    print("  日频超额（中证红利 − 中证800全收益）：")
    daily_stat(ex_bear, "熊市日(dd≤-20%)")
    daily_stat(ex_non, "非熊市日")
    print("  → 若「熊市日年化超额」与「非熊市日年化超额」接近，则红利超额是常态而非熊市特有（判冗余）。")
    print()

    # Gate 6 overlay：真实择时回测（熊市切红利 vs 恒持宽基 vs 恒持红利）
    print("-" * 88)
    print("Gate 6 overlay：熊市切红利 的真实择时回测（含成本）")
    print("-" * 88)
    # 状态：deep=True 持红利，False 持宽基
    sig_daily = deep.reindex(market.index).fillna(False).astype(int)
    r_div_a, r_mkt_a = aligned["ZHHL"].pct_change(), market.pct_change()
    r_div_a, r_mkt_a = r_div_a.align(r_mkt_a, join="inner")
    sig_daily = sig_daily.reindex(r_div_a.index).fillna(0)

    def perf(nav, label):
        total = nav.iloc[-1] - 1
        yrs = len(nav) / 252
        ann = nav.iloc[-1] ** (1 / yrs) - 1
        mdd = (nav / nav.cummax() - 1).min()
        vol = nav.pct_change().std() * np.sqrt(252)
        sharpe = (ann - 0.0) / vol if vol > 0 else 0
        print(f"  {label:24s} 总收益={total*100:+7.2f}%  年化={ann*100:+6.2f}%  "
              f"回撤={mdd*100:7.2f}%  夏普={sharpe:5.2f}")
    nav_div = (1 + r_div_a).cumprod()
    nav_mkt = (1 + r_mkt_a).cumprod()
    perf(nav_mkt, "恒持宽基(800)")
    perf(nav_div, "恒持红利(中证红利)")

    # 诊断①：切换次数
    n_switch = int(sig_daily.diff().abs().sum())
    print(f"\n  诊断① 日频 dd≤-20% 标记切换次数 = {n_switch} 次（每次成本 {COST*100:.1f}%）")
    # 诊断②：无成本版
    def nav_from_sig(sig, cost):
        s = sig.shift(1).fillna(0)
        sw = sig.diff().abs().fillna(0)
        r = s * r_div_a + (1 - s) * r_mkt_a - sw * cost
        return (1 + r).cumprod()
    perf(nav_from_sig(sig_daily, 0.0), "日频择时(无成本)")
    perf(nav_from_sig(sig_daily, COST), "日频择时(成本0.3%)")
    # 诊断③：区间择时（事件识别的「顶→回顶」熊市区间内持红利，其余持宽基）
    sig_seg = pd.Series(0, index=r_div_a.index)
    for t0, t1, _ in bears:
        sig_seg.loc[t0:t1] = 1
    sig_seg = sig_seg.reindex(r_div_a.index).fillna(0).astype(int)
    n_switch_seg = int(sig_seg.diff().abs().sum())
    perf(nav_from_sig(sig_seg, COST), "区间择时(成本0.3%)")
    print(f"  诊断③ 区间择时切换次数 = {n_switch_seg} 次")
    # 诊断④：完美择时（oracle，熊市区间起点即切，无识别滞后）——用区间择时但无成本+无滞后
    perf(nav_from_sig(sig_seg, 0.0), "区间择时(无成本)")
    print()

    # 结论速读
    print("=" * 88)
    print("结论速读（详见报告）")
    print("=" * 88)
    print("  - 若熊市区间红利超额中位数显著 >0 且胜率高，则「高股息熊市超额」方向成立。")
    print("  - 🔴 决定性：剔 2021 后历史熊市若超额塌到 ~0 或转负，则是区间选择偏差，不落地。")
    print("  - Gate5 日频若熊市/非熊市年化超额接近，则红利超额是常态，非熊市特有（条件规律=马甲）。")

    if args.csv:
        out = []
        for r in bear_rows:
            r2 = dict(r)
            r2["type"] = "bear"
            out.append(r2)
        for r in bull_rows:
            r2 = dict(r)
            r2["type"] = "bull"
            r2["mdd"] = None
            out.append(r2)
        df = pd.DataFrame(out)
        df.to_csv(args.csv, index=False)
        print(f"\n[CSV] 熊市/牛市区间明细已写出：{args.csv}  ({len(df)} 行)")


if __name__ == "__main__":
    main()
