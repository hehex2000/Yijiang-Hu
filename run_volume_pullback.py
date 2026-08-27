# -*- coding: utf-8 -*-
"""
缩量回踩战法 · 量化复现与 A/B 验证
=================================
忠实翻译 Jim《缩量回踩战法》五步框架(看趋势→定突破→算回落→等确认→做分类):

  ① 看趋势 : close>MA(中期) & MA(中期)向上
  ② 定突破位 : 上升通道中跟踪 running peak(=突破位), 存突破位量能均线(缩量比对基准)
  ③ 算回落 : 回落深度=(peak−当前)/peak ∈ [plo,phi]
            缩量比率=回落期量能均线 / 突破前量能均线 < vthr
  ④ 等确认 : 回落区间内 & 结构未破(close>MA中期) & 重新稳住(收>昨收, 即止跌转涨)
  ⑤ 做分类 : candidate / structure_changed(破中期均线) / trade_restricted(停板)

核心假设(待验证): 缩量是否给"纯价格回踩"提供增量信息?
  → A/B 双平行账本, 仅入场闸门不同, 退出规则完全一致:
      A = price_only  (①~④ 价格规则, 不含量能)
      B = price_vol   (A + ③缩量比率过滤)

反过拟合纪律(平台既有):
  - 参数全部 ex-ante 固定(写进 CLI, 不许看结果改线 —— Jim 原话)
  - universe = 中证800 (扩展候选池, 防小样本幸存者偏差)
  - 宽成本: 佣金+印花税+滑点 (平台默认)
  - 分市况报告(牛/熊/震荡) + 年度窗口(稳定性) —— 补 Jim 未给的失效边界
  - 样本分类统计(候选/结构改变/受限) —— 源头堵幸存者偏差

口径: 所有信号 T-1 收盘判定 / T 开盘执行(无未来函数); hfq 空间算均线/高低点, 原始价估值。
依赖: run_livermore_v2(L, 提供面板/宇宙/指数加载) + run_monthly_rebalance(费用/交易日)。
"""
import sys
import os
import argparse
import numpy as np
import pandas as pd

import run_livermore_v2 as L
from run_monthly_rebalance import (
    get_trade_dates, calc_fee, COMMISSION_RATE, STAMP_DUTY_RATE,
    SLIPPAGE_RATE, COMMISSION_MIN,
)

CAPITAL = 1_000_000.0
INDEX_MARKET = "000300.SH"
INDEX_BENCH_2 = "000906.SH"
RES_DIR = "data/results/volume_pullback"


# ────────────────────────────────────────────────────────────
#  信号构建(五步 → 双入场矩阵, 全部 T-1 空间)
# ────────────────────────────────────────────────────────────
def build_signals(all_dates, close_r, vol_r, cfg):
    """五步 → 双入场矩阵(全部 T-1 空间).

    设计要点(修正自首版缺陷):
      首版用"收盘价突破 N 日最高"判定突破, 但上升通道里每天都是"新高突破",
      假突破刷屏、真正回落被 cooldown 挡掉。正确语义是 Jim 的"自近期高点(突破位)回落":
      无需单独突破事件, 直接跟踪上升通道中的 running peak(=突破位), 价格自 peak
      回落 [plo,phi] + 缩量 + 结构未破(站中期均线上) + 止跌转涨 → 入场。
    """
    codes = list(close_r.columns)
    # 在含 warmup 的原始面板上算指标(利用预热数据), 再对齐到 all_dates
    close = close_r.astype(float)
    vol = vol_r.astype(float)
    mid = int(cfg["mid_ma"])
    ma_mid = close.rolling(mid, min_periods=mid).mean().shift(1)
    ma_slope = ma_mid.diff()                       # ma_mid 已 shift1 → 即 T-1 空间斜率
    vol_ma = vol.rolling(int(cfg["vol_win"]), min_periods=int(cfg["vol_win"])).mean().shift(1)

    close = close.reindex(all_dates).astype(float)
    vol = vol.reindex(all_dates).astype(float)
    cm = ma_mid.reindex(all_dates).values.astype(float)
    csl = ma_slope.reindex(all_dates).values.astype(float)
    cv = vol_ma.reindex(all_dates).values.astype(float)
    cc = close.values.astype(float)
    n = len(all_dates)
    ep = np.zeros((n, len(codes)), dtype=bool)    # price_only 入场
    ev = np.zeros((n, len(codes)), dtype=bool)    # price_vol 入场
    tot = dict(in_zone=0, entered=0, struct_changed=0, timeout=0)

    plo = float(cfg["pullback_lo"]); phi = float(cfg["pullback_hi"])
    vthr = float(cfg["vol_ratio_thr"]); cool = int(cfg["cooldown"])
    pmax = int(cfg["pullback_max_days"])

    for k in range(len(codes)):
        peak = np.nan; pre_vol = np.nan; last_entry = -10**9; pull_days = 0
        for i in range(n):
            ci = cc[i, k]; cmi = cm[i, k]; csl_i = csl[i, k]; cvi = cv[i, k]
            if not (np.isfinite(ci) and np.isfinite(cmi)):
                continue
            trend_ok = (ci > cmi) and (csl_i > 0)     # ① 看趋势(收盘>中期均线 & 中期均线向上)
            if not trend_ok:
                if np.isfinite(peak):
                    tot["struct_changed"] += 1        # ⑤ 结构改变样本
                peak = np.nan; pre_vol = np.nan; pull_days = 0
                continue
            # trend_ok 成立 → 跟踪 running peak(=突破位)
            if not np.isfinite(peak) or ci > peak:
                peak = ci
                pre_vol = cvi if np.isfinite(cvi) else np.nan   # ② 存突破位量能均线
                pull_days = 0
                continue
            # ci < peak → 处于回落中
            pull_days += 1
            depth = (peak - ci) / peak if peak > 0 else 0.0
            in_zone = (depth >= plo) and (depth <= phi)        # ③ 算回落深度
            if in_zone:
                tot["in_zone"] += 1
            # ④ 等确认: 结构未破(已在 trend_ok 保证) + 止跌转涨(收>昨收)
            confirm = (i > 0) and (ci > cc[i - 1, k])
            if in_zone and confirm and (i - last_entry) >= cool:
                vol_ok = np.isfinite(pre_vol) and np.isfinite(cvi) and (cvi < pre_vol * vthr)
                ep[i, k] = True
                tot["entered"] += 1
                if vol_ok:                                    # ③ 缩量比率过滤(仅 B 组)
                    ev[i, k] = True
                last_entry = i
                peak = ci                                     # 重置 peak, 避免同一轮回落重复入场
                pull_days = 0
                continue
            if pull_days > pmax:                              # 超时未触发 → 重置
                tot["timeout"] += 1
                peak = ci; pre_vol = cvi if np.isfinite(cvi) else np.nan; pull_days = 0
    ep_df = pd.DataFrame(ep, index=all_dates, columns=codes)
    ev_df = pd.DataFrame(ev, index=all_dates, columns=codes)
    return ep_df, ev_df, tot


# ────────────────────────────────────────────────────────────
#  平行账本回测(单次扫描, A/B 同时记账)
# ────────────────────────────────────────────────────────────
def run_backtest(start, end, cfg):
    all_dates = get_trade_dates(start, end)
    if len(all_dates) < 120:
        print(f"  [跳过] {start}-{end} 交易日不足"); return None, None, None, None

    univ_snaps = L.load_universe_dates(end)
    snap_dates = [s[0] for s in univ_snaps]
    import bisect
    def univ_at(d):
        i = bisect.bisect_right(snap_dates, d) - 1
        return univ_snaps[i][1] if i >= 0 else set()

    all_codes_set = set()
    for _, s in univ_snaps:
        all_codes_set |= s
    all_codes = sorted(all_codes_set)
    if not all_codes:
        print(f"  [跳过] {start}-{end} 无成分数据"); return None, None, None, None

    open_r, high_r, low_r, close_r, pre_close_r, vol_r, adj_d = L.load_panels(all_codes, start, end)
    # 所有面板显式对齐到 all_dates(行), 避免 load_panels 内部日期口径与 get_trade_dates 错行导致 DataFrame 乘法收缩行数
    adj_f = adj_d.reindex(all_dates).ffill().fillna(1.0)
    open_raw = open_r.reindex(all_dates).astype(float)
    close_raw = close_r.reindex(all_dates).astype(float)
    pre_close_raw = pre_close_r.reindex(all_dates).astype(float)
    close_h = (close_r.reindex(all_dates).astype(float) * adj_f).reindex(index=all_dates)
    high_h = (high_r.reindex(all_dates).astype(float) * adj_f).reindex(index=all_dates)

    ep_df, ev_df, tot = build_signals(all_dates, close_r, vol_r, cfg)
    mid = int(cfg["mid_ma"])
    ma_mid = close_h.rolling(mid, min_periods=mid).mean()   # 退出用结构判定(hfq)

    n = len(all_dates)
    cols = list(close_raw.columns)
    code2idx = {c: j for j, c in enumerate(cols)}
    # 末端强制对齐行=all_dates、列=cols, 行列数必须 == (n, len(cols))
    close_h_arr = close_h.reindex(index=all_dates, columns=cols).values.astype(float)
    high_h_arr = high_h.reindex(index=all_dates, columns=cols).values.astype(float)
    ma_arr = ma_mid.reindex(index=all_dates, columns=cols).values.astype(float)
    assert close_h_arr.shape == (n, len(cols)), f"close_h_arr {close_h_arr.shape} != ({n},{len(cols)})"
    assert ma_arr.shape == (n, len(cols)), f"ma_arr {ma_arr.shape} != ({n},{len(cols)})"

    ep_sets = [set(np.compress(ep_df.values[i], cols)) for i in range(n)]
    ev_sets = [set(np.compress(ev_df.values[i], cols)) for i in range(n)]

    groups = ["price_only", "price_vol"]
    books = {g: dict(cash=CAPITAL, positions={}, pending=set(), nav=[], trades=[])
             for g in groups}
    entry_sets = {"price_only": ep_sets, "price_vol": ev_sets}

    stop_loss = float(cfg["stop_loss"])
    max_hold = int(cfg["max_hold"])

    def close_trade(bk, code, sell_price, d, entry_idx):
        pos = bk["positions"][code]
        sh = pos["shares"]
        if sh <= 0:
            return
        proceeds = sell_price * sh - calc_fee("sell", sell_price, sh, d)
        bk["cash"] += proceeds
        cost = pos["entry_open"] * sh + calc_fee("buy", pos["entry_open"], sh, pos["entry_date"])
        ret = proceeds / cost - 1.0
        bk["trades"].append(dict(entry_date=pos["entry_date"], exit_date=d, code=code,
                                 ret=ret))
        bk["positions"].pop(code, None)

    for i in range(n):
        d = all_dates[i]
        univ = univ_at(d)
        if not univ:
            for g in groups:
                books[g]["nav"].append((d, books[g]["cash"]))
            continue
        prev_i = i - 1

        for g in groups:
            bk = books[g]
            positions = bk["positions"]

            # 1) 卖出执行(前一日跌停封死重试)
            sell_exec = []
            for code in list(bk["pending"]):
                if code not in positions:
                    bk["pending"].discard(code); continue
                op = open_raw.iloc[i].get(code); cl = close_raw.iloc[i].get(code); pc = pre_close_raw.iloc[i].get(code)
                if op is None or pd.isna(op) or cl is None or pd.isna(cl) or pc is None or pd.isna(pc):
                    bk["pending"].discard(code); continue
                if op <= pc * 0.901:
                    if cl <= pc * 0.901:
                        continue
                    sell_exec.append((code, cl))
                else:
                    sell_exec.append((code, op))
            bk["pending"].clear()

            # 2) 持仓退出判定(T-1 信号 T 开盘执行)
            if prev_i >= 0:
                for code, pos in list(positions.items()):
                    if code not in univ:
                        last_c = pos.get("last_close")
                        if last_c is None or pd.isna(last_c):
                            last_c = pos.get("entry_open", 0.0)
                        sell_exec.append((code, last_c)); continue
                    if code not in code2idx:
                        continue
                    j = code2idx[code]
                    ch = close_h_arr[prev_i, j]
                    if not np.isfinite(ch):
                        continue
                    exit_now = False
                    entry_idx = all_dates.index(pos["entry_date"]) if pos["entry_date"] in all_dates else -1
                    held = (i - entry_idx) if entry_idx >= 0 else 9999
                    if held >= max_hold:
                        exit_now = True
                    if not exit_now and not pd.isna(pos["entry_hfq"]) and ch < pos["entry_hfq"] * (1 - stop_loss):
                        exit_now = True
                    if not exit_now:                         # 结构破坏(破中期均线: 收盘跌破中期均线)
                        ma_val = ma_arr[prev_i, j] if j < ma_arr.shape[1] else np.nan
                        if np.isfinite(ma_val) and ch < ma_val:
                            exit_now = True
                    if exit_now:
                        op = open_raw.iloc[i].get(code); cl = close_raw.iloc[i].get(code); pc = pre_close_raw.iloc[i].get(code)
                        if op is not None and not pd.isna(op) and cl is not None and not pd.isna(cl) and pc is not None and not pd.isna(pc):
                            if op <= pc * 0.901:
                                if cl <= pc * 0.901:
                                    bk["pending"].add(code)
                                else:
                                    sell_exec.append((code, cl))
                            else:
                                sell_exec.append((code, op))

            for code, price in sell_exec:
                if code in positions:
                    close_trade(bk, code, price, d,
                                all_dates.index(positions[code]["entry_date"]) if positions[code]["entry_date"] in all_dates else -1)

            # 3) 新开仓(入场信号 T-1, T 开盘执行; 涨停不可买)
            if prev_i >= 0 and univ:
                cand_set = (entry_sets[g][prev_i] & univ) - set(positions) - bk["pending"]
                cand = sorted(cand_set)
                def _px(c):
                    p = close_raw.iloc[i].get(c)
                    if p is None or pd.isna(p):
                        p = positions[c].get("last_close") or 0.0
                    return p
                equity = bk["cash"] + sum(_px(c) * positions[c]["shares"] for c in positions)
                slots = max_hold - len(positions)
                if slots > 0 and cand:
                    take = cand[:slots]
                    per_val = equity / max_hold
                    for code in take:
                        op = open_raw.iloc[i].get(code); cl = close_raw.iloc[i].get(code); pc = pre_close_raw.iloc[i].get(code)
                        if op is None or pd.isna(op) or op <= 0 or cl is None or pd.isna(cl) or pc is None or pd.isna(pc):
                            continue
                        if op >= pc * 1.099:       # 涨停不可买 → 跳过(信号作废)
                            continue
                        buy_price = op
                        if pd.isna(buy_price) or buy_price <= 0:
                            continue
                        sh = int(per_val / (buy_price * (1 + COMMISSION_RATE + SLIPPAGE_RATE)) / 100) * 100
                        if sh <= 0:
                            sh = 100
                        cost = buy_price * sh + calc_fee("buy", buy_price, sh, d)
                        if pd.isna(cost) or cost > bk["cash"]:
                            continue
                        bk["cash"] -= cost
                        j = code2idx.get(code)
                        positions[code] = dict(shares=sh, entry_open=buy_price,
                                               entry_hfq=close_h_arr[prev_i, j] if j is not None else np.nan,
                                               entry_date=d, last_close=buy_price)

        # 4) 估值
        for g in groups:
            bk = books[g]
            mv = bk["cash"]
            for code, pos in bk["positions"].items():
                c = close_raw.iloc[i].get(code)
                if c is None or pd.isna(c):
                    c = pos.get("last_close")
                if c is not None and not pd.isna(c):
                    mv += pos["shares"] * c
                    pos["last_close"] = c
            bk["nav"].append((d, mv))

    out = {}
    for g in groups:
        bk = books[g]
        out[g] = dict(nav=[v for _, v in bk["nav"]], trades=bk["trades"])

    # 匹配 A/B: 同一批价格回踩候选(ep), 按是否含缩量确认(ev)拆两组,
    # 比较各自持有期收益(退出规则完全一致). 隔离"缩量是否增量信息"单一变量.
    matched = matched_ab(all_dates, ep_df, ev_df, open_raw, close_raw,
                         pre_close_raw, close_h_arr, ma_arr, code2idx, cfg)
    return out, tot, all_dates, matched


# ────────────────────────────────────────────────────────────
#  匹配 A/B (同候选集 · 缩量增量信息检验)
# ────────────────────────────────────────────────────────────
def matched_ab(all_dates, ep_df, ev_df, open_raw, close_raw, pre_close_raw,
               close_h_arr, ma_arr, code2idx, cfg, ret_years=False):
    stop_loss = float(cfg["stop_loss"]); max_hold = int(cfg["max_hold"])
    n = len(all_dates)
    cols = list(ep_df.columns)
    epv = ep_df.values; evv = ev_df.values
    vol_r, novol_r, vol_y, novol_y = [], [], [], []

    def sim(s, k):
        e = s + 1
        if e >= n:
            return None
        j = code2idx.get(k)
        if j is None or j >= close_h_arr.shape[1]:
            return None
        op = open_raw.iloc[e].get(k); pc = pre_close_raw.iloc[e].get(k)
        if op is None or pd.isna(op) or op <= 0:
            return None
        if op >= pc * 1.099:
            return None
        buy = float(op)
        entry_hfq = close_h_arr[s, j] if np.isfinite(close_h_arr[s, j]) else np.nan
        sh = 100
        cost = buy * sh + calc_fee("buy", buy, sh, all_dates[e])
        if not np.isfinite(cost) or cost <= 0:
            return None
        for t in range(e + 1, min(e + max_hold + 1, n)):
            prev = t - 1
            ch = close_h_arr[prev, j] if np.isfinite(close_h_arr[prev, j]) else np.nan
            if not np.isfinite(ch):
                continue
            exit_now = False
            held = t - e
            if held >= max_hold:
                exit_now = True
            if not exit_now and np.isfinite(entry_hfq) and ch < entry_hfq * (1 - stop_loss):
                exit_now = True
            if not exit_now:
                mv = ma_arr[prev, j] if np.isfinite(ma_arr[prev, j]) else np.nan
                if np.isfinite(mv) and ch < mv:
                    exit_now = True
            if exit_now:
                op_t = open_raw.iloc[t].get(k); cl_t = close_raw.iloc[t].get(k); pc_t = pre_close_raw.iloc[t].get(k)
                if op_t is None or pd.isna(op_t):
                    sell = cl_t
                elif op_t <= pc_t * 0.901:
                    sell = cl_t if (cl_t is not None and not pd.isna(cl_t)) else op_t
                else:
                    sell = op_t
                if sell is None or pd.isna(sell) or sell <= 0:
                    return None
                proceeds = sell * sh - calc_fee("sell", sell, sh, all_dates[t])
                return proceeds / cost - 1.0
        # 窗口内未触发 → 末日强制平仓
        t_end = min(e + max_hold, n - 1)
        op_t = open_raw.iloc[t_end].get(k); cl_t = close_raw.iloc[t_end].get(k); pc_t = pre_close_raw.iloc[t_end].get(k)
        if op_t is not None and not pd.isna(op_t) and op_t <= pc_t * 0.901 and cl_t is not None and not pd.isna(cl_t) and cl_t <= pc_t * 0.901:
            sell = cl_t
        elif op_t is not None and not pd.isna(op_t):
            sell = op_t
        else:
            sell = cl_t
        if sell is None or pd.isna(sell) or sell <= 0:
            return None
        proceeds = sell * sh - calc_fee("sell", sell, sh, all_dates[t_end])
        return proceeds / cost - 1.0

    for s, j in np.argwhere(epv):
        k = cols[j]
        r = sim(s, k)
        if r is None:
            continue
        if evv[s, j]:
            vol_r.append(r); vol_y.append(all_dates[s][:4])
        else:
            novol_r.append(r); novol_y.append(all_dates[s][:4])
    if ret_years:
        return vol_r, novol_r, vol_y, novol_y
    return vol_r, novol_r


def _welch(a, b):
    a = np.array(a, float); b = np.array(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / len(a) + vb / len(b))
    return (a.mean() - b.mean()) / se if se > 0 else float("nan")


def _stat(xs):
    xs = np.array(xs, float)
    if len(xs) == 0:
        return 0, float("nan"), float("nan")
    return len(xs), float(xs.mean()), float((xs > 0).mean())


# ────────────────────────────────────────────────────────────
#  指标 / 分市况 / 年度窗口
# ────────────────────────────────────────────────────────────
def metrics(vals, years):
    vals = np.array(vals, dtype=float)
    tot = vals[-1] / vals[0] - 1
    ann = (vals[-1] / vals[0]) ** (1 / years) - 1 if years > 0 else 0
    peak = np.maximum.accumulate(vals)
    mdd = (vals / peak - 1).min()
    rets = np.diff(vals) / vals[:-1]
    vol = rets.std() * np.sqrt(252) if len(rets) > 1 else 0
    sharpe = (rets.mean() * 252 - 0.02) / vol if (vol > 0 and len(rets) > 1) else 0
    return dict(total=tot, ann=ann, mdd=mdd, sharpe=sharpe)


def annual_pnl(all_dates, nav_vals):
    s = pd.Series(nav_vals, index=pd.to_datetime(all_dates))
    rows = []
    for yr, g in s.groupby(s.index.year):
        if len(g) < 2:
            continue
        ret = g.iloc[-1] / g.iloc[0] - 1
        rows.append((yr, g.iloc[0], g.iloc[-1], ret))
    return rows


def regime_split(trades, all_dates, idx_close):
    idx = idx_close.reindex(all_dates)
    ma20 = idx.rolling(20, min_periods=20).mean()
    out = {"bull": [], "bear": [], "side": []}
    d2i = {d: i for i, d in enumerate(all_dates)}
    for t in trades:
        i = d2i.get(t["entry_date"])
        if i is None or i >= len(idx) or pd.isna(idx.iloc[i]) or pd.isna(ma20.iloc[i]):
            continue
        ratio = idx.iloc[i] / ma20.iloc[i] - 1.0
        key = "bull" if ratio > 0.03 else ("bear" if ratio < -0.03 else "side")
        out[key].append(t["ret"])
    res = {}
    for k, v in out.items():
        res[k] = (len(v), float(np.mean(v)) if v else float("nan"))
    return res


# ────────────────────────────────────────────────────────────
#  参数敏感性扫描(ex-ante 固定网格 · 匹配 A/B 多组)
# ────────────────────────────────────────────────────────────
def _load_data(start, end):
    """一次性加载面板/universe, 供单跑与扫描复用(避免每组重复读库)."""
    all_dates = get_trade_dates(start, end)
    univ_snaps = L.load_universe_dates(end)
    snap_dates = [s[0] for s in univ_snaps]
    import bisect
    def univ_at(d):
        i = bisect.bisect_right(snap_dates, d) - 1
        return univ_snaps[i][1] if i >= 0 else set()
    all_codes_set = set()
    for _, s in univ_snaps:
        all_codes_set |= s
    all_codes = sorted(all_codes_set)
    if not all_codes:
        return None
    open_r, high_r, low_r, close_r, pre_close_r, vol_r, adj_d = L.load_panels(all_codes, start, end)
    adj_f = adj_d.reindex(all_dates).ffill().fillna(1.0)
    open_raw = open_r.reindex(all_dates).astype(float)
    close_raw = close_r.reindex(all_dates).astype(float)
    pre_close_raw = pre_close_r.reindex(all_dates).astype(float)
    close_h = (close_r.reindex(all_dates).astype(float) * adj_f).reindex(index=all_dates)
    cols = list(close_raw.columns)
    code2idx = {c: j for j, c in enumerate(cols)}
    close_h_arr = close_h.reindex(index=all_dates, columns=cols).values.astype(float)
    assert close_h_arr.shape[0] == len(all_dates), \
        f"close_h_arr {close_h_arr.shape} vs dates {len(all_dates)}"
    return dict(all_dates=all_dates, univ_at=univ_at, open_raw=open_raw, close_raw=close_raw,
                pre_close_raw=pre_close_raw, close_h=close_h, close_h_arr=close_h_arr,
                close_r=close_r, vol_r=vol_r, cols=cols, code2idx=code2idx)


def _ma_arr(close_h, all_dates, cols, mid):
    return close_h.rolling(mid, min_periods=mid).mean().reindex(index=all_dates, columns=cols).values.astype(float)


def portfolio_backtest(all_dates, data, ep_df, ev_df, ma_arr, cfg):
    """单次扫描 A/B 平行账本(逻辑与 run_backtest 内联循环一致). 返回 books."""
    n = len(all_dates)
    cols = data["cols"]; code2idx = data["code2idx"]
    open_raw = data["open_raw"]; close_raw = data["close_raw"]; pre_close_raw = data["pre_close_raw"]
    close_h_arr = data["close_h_arr"]; univ_at = data["univ_at"]
    ep_sets = [set(np.compress(ep_df.values[i], cols)) for i in range(n)]
    ev_sets = [set(np.compress(ev_df.values[i], cols)) for i in range(n)]
    groups = ["price_only", "price_vol"]
    books = {g: dict(cash=CAPITAL, positions={}, pending=set(), nav=[], trades=[])
             for g in groups}
    entry_sets = {"price_only": ep_sets, "price_vol": ev_sets}
    stop_loss = float(cfg["stop_loss"]); max_hold = int(cfg["max_hold"])

    def close_trade(bk, code, sell_price, d, entry_idx):
        pos = bk["positions"][code]
        sh = pos["shares"]
        if sh <= 0:
            return
        proceeds = sell_price * sh - calc_fee("sell", sell_price, sh, d)
        bk["cash"] += proceeds
        cost = pos["entry_open"] * sh + calc_fee("buy", pos["entry_open"], sh, pos["entry_date"])
        ret = proceeds / cost - 1.0
        bk["trades"].append(dict(entry_date=pos["entry_date"], exit_date=d, code=code, ret=ret))
        bk["positions"].pop(code, None)

    for i in range(n):
        d = all_dates[i]
        univ = univ_at(d)
        if not univ:
            for g in groups:
                books[g]["nav"].append((d, books[g]["cash"]))
            continue
        prev_i = i - 1
        for g in groups:
            bk = books[g]; positions = bk["positions"]
            sell_exec = []
            for code in list(bk["pending"]):
                if code not in positions:
                    bk["pending"].discard(code); continue
                op = open_raw.iloc[i].get(code); cl = close_raw.iloc[i].get(code); pc = pre_close_raw.iloc[i].get(code)
                if op is None or pd.isna(op) or cl is None or pd.isna(cl) or pc is None or pd.isna(pc):
                    bk["pending"].discard(code); continue
                if op <= pc * 0.901:
                    if cl <= pc * 0.901:
                        continue
                    sell_exec.append((code, cl))
                else:
                    sell_exec.append((code, op))
            bk["pending"].clear()
            if prev_i >= 0:
                for code, pos in list(positions.items()):
                    if code not in univ:
                        last_c = pos.get("last_close")
                        if last_c is None or pd.isna(last_c):
                            last_c = pos.get("entry_open", 0.0)
                        sell_exec.append((code, last_c)); continue
                    if code not in code2idx:
                        continue
                    j = code2idx[code]
                    ch = close_h_arr[prev_i, j]
                    if not np.isfinite(ch):
                        continue
                    exit_now = False
                    entry_idx = all_dates.index(pos["entry_date"]) if pos["entry_date"] in all_dates else -1
                    held = (i - entry_idx) if entry_idx >= 0 else 9999
                    if held >= max_hold:
                        exit_now = True
                    if not exit_now and not pd.isna(pos["entry_hfq"]) and ch < pos["entry_hfq"] * (1 - stop_loss):
                        exit_now = True
                    if not exit_now:
                        ma_val = ma_arr[prev_i, j] if j < ma_arr.shape[1] else np.nan
                        if np.isfinite(ma_val) and ch < ma_val:
                            exit_now = True
                    if exit_now:
                        op = open_raw.iloc[i].get(code); cl = close_raw.iloc[i].get(code); pc = pre_close_raw.iloc[i].get(code)
                        if op is not None and not pd.isna(op) and cl is not None and not pd.isna(cl) and pc is not None and not pd.isna(pc):
                            if op <= pc * 0.901:
                                if cl <= pc * 0.901:
                                    bk["pending"].add(code)
                                else:
                                    sell_exec.append((code, cl))
                            else:
                                sell_exec.append((code, op))
            for code, price in sell_exec:
                if code in positions:
                    close_trade(bk, code, price, d,
                                all_dates.index(positions[code]["entry_date"]) if positions[code]["entry_date"] in all_dates else -1)
            if prev_i >= 0 and univ:
                cand_set = (entry_sets[g][prev_i] & univ) - set(positions) - bk["pending"]
                cand = sorted(cand_set)
                def _px(c):
                    p = close_raw.iloc[i].get(c)
                    if p is None or pd.isna(p):
                        p = positions[c].get("last_close") or 0.0
                    return p
                equity = bk["cash"] + sum(_px(c) * positions[c]["shares"] for c in positions)
                slots = max_hold - len(positions)
                if slots > 0 and cand:
                    take = cand[:slots]
                    per_val = equity / max_hold
                    for code in take:
                        op = open_raw.iloc[i].get(code); cl = close_raw.iloc[i].get(code); pc = pre_close_raw.iloc[i].get(code)
                        if op is None or pd.isna(op) or op <= 0 or cl is None or pd.isna(cl) or pc is None or pd.isna(pc):
                            continue
                        if op >= pc * 1.099:
                            continue
                        buy_price = op
                        if pd.isna(buy_price) or buy_price <= 0:
                            continue
                        sh = int(per_val / (buy_price * (1 + COMMISSION_RATE + SLIPPAGE_RATE)) / 100) * 100
                        if sh <= 0:
                            sh = 100
                        cost = buy_price * sh + calc_fee("buy", buy_price, sh, d)
                        if pd.isna(cost) or cost > bk["cash"]:
                            continue
                        bk["cash"] -= cost
                        j = code2idx.get(code)
                        positions[code] = dict(shares=sh, entry_open=buy_price,
                                               entry_hfq=close_h_arr[prev_i, j] if j is not None else np.nan,
                                               entry_date=d, last_close=buy_price)
        for g in groups:
            bk = books[g]
            mv = bk["cash"]
            for code, pos in bk["positions"].items():
                c = close_raw.iloc[i].get(code)
                if c is None or pd.isna(c):
                    c = pos.get("last_close")
                if c is not None and not pd.isna(c):
                    mv += pos["shares"] * c
                    pos["last_close"] = c
            bk["nav"].append((d, mv))
    out = {}
    for g in groups:
        bk = books[g]
        out[g] = dict(nav=[v for _, v in bk["nav"]], trades=bk["trades"])
    return out


def _portfolio_summary(res, all_dates, years):
    out = {}
    for g in ["price_only", "price_vol"]:
        out[g] = metrics(res[g]["nav"], years)
    return dict(a_total=out["price_only"]["total"], a_sharpe=out["price_only"]["sharpe"],
                b_total=out["price_vol"]["total"], b_sharpe=out["price_vol"]["sharpe"])


def _matched_per_year(vol_r, novol_r, vol_y, novol_y):
    years = sorted(set(vol_y) | set(novol_y))
    neg = 0; tot = 0
    for y in years:
        vm = [r for r, yy in zip(vol_r, vol_y) if yy == y]
        nm = [r for r, yy in zip(novol_r, novol_y) if yy == y]
        if len(vm) >= 20 and len(nm) >= 20:
            tot += 1
            if float(np.mean(vm)) < float(np.mean(nm)):
                neg += 1
    return dict(neg=neg, tot=tot)


def run_sweep(start, end, base_cfg, grid, with_portfolio=True):
    data = _load_data(start, end)
    if data is None:
        print("  [跳过] 无成分数据/交易日不足"); return None
    all_dates = data["all_dates"]
    years = (pd.Timestamp(all_dates[-1]) - pd.Timestamp(all_dates[0])).days / 365.25
    close_h = data["close_h"]; cols = data["cols"]
    ma_cache = {}
    for combo in grid:
        m = int(combo["mid_ma"])
        if m not in ma_cache:
            ma_cache[m] = _ma_arr(close_h, all_dates, cols, m)
    rows = []
    for combo in grid:
        cfg = dict(base_cfg)
        for k in ("mid_ma", "pullback_lo", "pullback_hi", "vol_ratio_thr"):
            cfg[k] = combo[k]
        ep_df, ev_df, tot = build_signals(all_dates, data["close_r"], data["vol_r"], cfg)
        ma_arr = ma_cache[int(cfg["mid_ma"])]
        vol_r_l, novol_r_l, vol_y, novol_y = matched_ab(
            all_dates, ep_df, ev_df, data["open_raw"], data["close_raw"],
            data["pre_close_raw"], data["close_h_arr"], ma_arr, data["code2idx"], cfg, ret_years=True)
        nv, mv, wv = _stat(vol_r_l)
        nn, mn, wn = _stat(novol_r_l)
        t = _welch(vol_r_l, novol_r_l)
        diff = (mv - mn) * 100 if (nv and nn) else float("nan")
        yr = _matched_per_year(vol_r_l, novol_r_l, vol_y, novol_y)
        if with_portfolio:
            res = portfolio_backtest(all_dates, data, ep_df, ev_df, ma_arr, cfg)
            pa = _portfolio_summary(res, all_dates, years)
        else:
            pa = dict(a_total=float("nan"), a_sharpe=float("nan"), b_total=float("nan"), b_sharpe=float("nan"))
        rows.append(dict(
            mid_ma=cfg["mid_ma"], pullback=f"[{cfg['pullback_lo']},{cfg['pullback_hi']}]",
            vthr=cfg["vol_ratio_thr"], n_vol=nv, n_novol=nn,
            mean_vol=f"{mv*100:+.2f}%", mean_novol=f"{mn*100:+.2f}%",
            diff_pp=f"{diff:+.2f}", welch=(f"{t:+.2f}" if np.isfinite(t) else "nan"),
            vol_worse=bool(diff < 0 and np.isfinite(t) and abs(t) > 1.96),
            yr_neg=yr["neg"], yr_tot=yr["tot"],
            A_total=(f"{pa['a_total']*100:+.2f}%" if np.isfinite(pa['a_total']) else "nan"),
            A_sharpe=(f"{pa['a_sharpe']:.2f}" if np.isfinite(pa['a_sharpe']) else "nan"),
            B_total=(f"{pa['b_total']*100:+.2f}%" if np.isfinite(pa['b_total']) else "nan"),
            B_sharpe=(f"{pa['b_sharpe']:.2f}" if np.isfinite(pa['b_sharpe']) else "nan"),
        ))
    return rows


def sweep_main(args):
    start, end = args.start, args.end
    base_cfg = dict(mid_ma=60, vol_win=20, pullback_lo=0.05, pullback_hi=0.20,
                    vol_ratio_thr=0.70, cooldown=30, pullback_max_days=40,
                    max_hold=20, stop_loss=0.08)
    # ex-ante 固定网格(运行前确定, 不许看结果改线)
    grid = [
        dict(mid_ma=m, pullback_lo=plo, pullback_hi=phi, vol_ratio_thr=v)
        for m in [40, 60, 120]
        for (plo, phi) in [(0.03, 0.10), (0.05, 0.15), (0.08, 0.20)]
        for v in [0.6, 0.7]
    ]
    print(f"缩量回踩 参数敏感性扫描 | {start}-{end} | 网格 {len(grid)} 组(ex-ante 固定)")
    print("假设: 缩量在多种参数下是否仍为负增量信息? (系统性失败 vs 特定参数失败)\n")
    rows = run_sweep(start, end, base_cfg, grid, with_portfolio=not args.no_portfolio)
    if rows is None:
        return
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    n_worse = sum(1 for r in rows if r["vol_worse"])
    n_better = sum(1 for r in rows
                   if r["diff_pp"].startswith("+") and r["welch"] != "nan" and abs(float(r["welch"])) > 1.96)
    n_yr = [r["yr_tot"] for r in rows if r["yr_tot"] > 0]
    avg_yr_neg = (sum(r["yr_neg"] for r in rows if r["yr_tot"] > 0) /
                  sum(r["yr_tot"] for r in rows if r["yr_tot"] > 0)) if n_yr else float("nan")
    print(f"\n【扫描结论】缩量显著更差(vol_worse) = {n_worse}/{len(rows)} 组 ｜ "
          f"缩量显著更好 = {n_better}/{len(rows)} 组")
    print(f"  逐年稳定性: 有样本年份中, 缩量更差占比均值 ≈ {avg_yr_neg*100:.0f}% "
          f"(yr_neg/yr_tot 跨组平均)")
    if not args.no_portfolio:
        tradeable = [r for r in rows
                     if r["A_total"] != "nan" and (r["A_total"].startswith("+") or r["B_total"].startswith("+"))]
        print(f"  组合层可交易(任一账本总收益>0) = {len(tradeable)}/{len(rows)} 组")
    os.makedirs(RES_DIR, exist_ok=True)
    out_path = os.path.join(RES_DIR, "volume_pullback_sweep.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nCSV 已写: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="缩量回踩战法 A/B 验证 (price_only vs price_vol)")
    ap.add_argument("--start", default="20180101")
    ap.add_argument("--end", default="20251231")
    ap.add_argument("--mid-ma", type=int, default=60, help="中期均线(趋势/结构判定)")
    ap.add_argument("--vol-win", type=int, default=20, help="量能均线窗口(日)")
    ap.add_argument("--pullback-lo", type=float, default=0.05, help="回落深度下限")
    ap.add_argument("--pullback-hi", type=float, default=0.20, help="回落深度上限")
    ap.add_argument("--vol-ratio-thr", type=float, default=0.70, help="缩量阈值(回落量<突破前量×此值)")
    ap.add_argument("--cooldown", type=int, default=30, help="两次突破最小间隔(日)")
    ap.add_argument("--pullback-max-days", type=int, default=40, help="回落观察超时(日)")
    ap.add_argument("--max-hold", type=int, default=20, help="固定持有(交易日)")
    ap.add_argument("--stop-loss", type=float, default=0.08, help="硬止损")
    ap.add_argument("--sweep", action="store_true", help="参数敏感性扫描(匹配 A/B 多组 ex-ante 固定参数)")
    ap.add_argument("--no-portfolio", action="store_true", help="扫描时跳过组合层回测(仅匹配 A/B)")
    args = ap.parse_args()

    cfg = dict(mid_ma=args.mid_ma, vol_win=args.vol_win, pullback_lo=args.pullback_lo,
               pullback_hi=args.pullback_hi, vol_ratio_thr=args.vol_ratio_thr,
               cooldown=args.cooldown, pullback_max_days=args.pullback_max_days,
               max_hold=args.max_hold, stop_loss=args.stop_loss)

    if args.sweep:
        sweep_main(args)
        return

    print(f"缩量回踩 A/B | {args.start}-{args.end} | mid={args.mid_ma} "
          f"vw={args.vol_win} pull=[{args.pullback_lo},{args.pullback_hi}] "
          f"vthr={args.vol_ratio_thr} hold={args.max_hold} sl={args.stop_loss}")
    print("假设: 缩量是否给纯价格回踩提供增量信息? A=price_only B=price_vol(仅入场闸门不同)\n")

    res, tot, all_dates, matched = run_backtest(args.start, args.end, cfg)
    if res is None:
        return

    years = (pd.Timestamp(all_dates[-1]) - pd.Timestamp(all_dates[0])).days / 365.25
    idx_close = L.load_index(INDEX_MARKET, args.start, args.end)
    b1 = idx_close.reindex(all_dates)
    b2 = L.load_index(INDEX_BENCH_2, args.start, args.end).reindex(all_dates)
    def bench_nav(s):
        fv = s.first_valid_index()
        if fv is None:
            return np.array([CAPITAL] * len(all_dates))
        nav = s / s[fv] * CAPITAL
        return nav.where(nav.notna(), CAPITAL).ffill().values
    mb1 = metrics(bench_nav(b1), years)
    mb2 = metrics(bench_nav(b2), years)

    print("═" * 78)
    print("【样本分类统计】(源头堵幸存者偏差, Jim 第⑤步)")
    print(f"  进入回落区={tot['in_zone']}  实际入场={tot['entered']}  "
          f"结构改变剔除={tot['struct_changed']}  超时剔除={tot['timeout']}")
    print("═" * 78)

    rows = []
    for g in ["price_only", "price_vol"]:
        nav = res[g]["nav"]; tr = res[g]["trades"]
        m = metrics(nav, years)
        n_tr = len(tr)
        winrate = (sum(1 for t in tr if t["ret"] > 0) / n_tr) if n_tr else float("nan")
        rg = regime_split(tr, all_dates, idx_close)
        rows.append([g, n_tr, f"{winrate*100:.1f}%",
                     f"{m['total']*100:+.2f}%", f"{m['ann']*100:+.2f}%",
                     f"{m['mdd']*100:.2f}%", f"{m['sharpe']:.2f}",
                     f"{(m['total']-mb1['total'])*100:+.2f}pp",
                     f"{(m['total']-mb2['total'])*100:+.2f}pp",
                     f"牛{rg['bull'][0]}/{rg['bull'][1]*100:+.1f}%"
                     f" 熊{rg['bear'][0]}/{rg['bear'][1]*100:+.1f}%"
                     f" 震{rg['side'][0]}/{rg['side'][1]*100:+.1f}%"])
    df = pd.DataFrame(rows, columns=["组", "笔数", "胜率", "总收益", "年化", "最大回撤",
                                     "夏普", "超额HS300", "超额中证800", "分市况(笔数/均收益)"])
    label_map = {"price_only": "A 价格回踩(不含量)", "price_vol": "B 价格+缩量"}
    df["组"] = df["组"].map(label_map)
    print("\n【A/B 对照】(退出规则完全一致, 仅入场闸门不同 · 修正结构破坏出场后)")
    print(df.to_string(index=False))
    print(f"\n基准 HS300: 总收益 {mb1['total']*100:+.2f}% | 中证800: 总收益 {mb2['total']*100:+.2f}%")

    a_total = float(rows[0][3].replace('+', '').replace('%', '')) / 100
    b_total = float(rows[1][3].replace('+', '').replace('%', '')) / 100
    print(f"\n[组合层判据] B−A 总收益差 = {(b_total-a_total)*100:+.2f}pp ｜ "
          f"B−A 夏普差 = {float(rows[1][6])-float(rows[0][6]):+.2f}")
    print("  注: 组合层 A/B 交易数差仍大, 此差含'暴露/频率'效应, 非纯缩量信号质量。")

    # ── 匹配 A/B: 同候选集, 按是否缩量确认拆分, 隔离'缩量增量信息' ──
    vol_r, novol_r = matched
    nv, mv, wv = _stat(vol_r)
    nn, mn, wn = _stat(novol_r)
    t = _welch(vol_r, novol_r)
    print("\n══════════════════════════════════════════════════════════════════")
    print("【匹配 A/B · 缩量增量信息检验】(同一批价格回踩候选, 仅按缩量确认拆分)")
    print(f"  候选总数 = {nv+nn}  ｜  缩量确认组 n={nv}  ｜  非缩量组 n={nn}")
    print(f"  缩量确认组 : 均收益 {mv*100:+.2f}%  胜率 {wv*100:.1f}%")
    print(f"  非缩量组   : 均收益 {mn*100:+.2f}%  胜率 {wn*100:.1f}%")
    print(f"  增量(缩量−非缩量) 均值差 = {(mv-mn)*100:+.2f}pp  ｜  Welch t = {t:+.2f}")
    if nv and nn:
        if mv > mn:
            sig = "显著" if abs(t) > 1.96 else "方向正但样本内不显著"
            print(f"  → 缩量确认组的持有期收益更高: 缩量提供【正增量信息】({sig})。")
        else:
            print(f"  → 缩量确认组收益未更高: 缩量在此口径下【未提供增量信息】。")
    print("══════════════════════════════════════════════════════════════════")

    # 年度窗口(稳定性)
    print("\n【年度窗口】(稳定性 / 失效边界)")
    for g in ["price_only", "price_vol"]:
        ap_rows = annual_pnl(all_dates, res[g]["nav"])
        line = "  ".join(f"{yr}:{r*100:+.1f}%" for yr, _, _, r in ap_rows)
        print(f"  {label_map[g]}: {line}")

    os.makedirs(RES_DIR, exist_ok=True)
    out_path = os.path.join(RES_DIR, "volume_pullback_ab.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nCSV 已写: {out_path}")


if __name__ == "__main__":
    main()
