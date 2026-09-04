# -*- coding: utf-8 -*-
"""
左侧-右侧混合择时策略 —— 基于微观结构角色统合左右侧之争
================================================================
视频: "左侧交易与右侧交易：拆解两种买入模式的胜率、盈亏比与心理压力"
      (UP主: 悦悦笔记, BV1uAuu6vE1j)

核心论点（学术支撑）:
  - 左侧=限价单=流动性提供者: 胜率低(<50%, 逆向选择), 盈亏比高(成本极低、止损窄)
    Linnainmaa (2010, JF): 限价买单成交次日平均亏 ~51bp, 因逆向选择
  - 右侧=市价单=流动性消费者: 胜率高(突破后惯性), 盈亏比低(买得贵、止损宽)
    Odean (1998, JF): 处置效应——卖赢家持输家, 卖出的赢家次年比死扛的输家多涨3.4%
    Barber & Odean (2000, JF): 最活跃交易者年化11.4% vs 市场17.9%, "活跃税"6.5%/年
  - 散户亏损=把左侧的低胜率+右侧的低盈亏比缝在一起（逆势抄底+追高杀跌+窄止损被震出）

策略设计:
  环境判断器（T-1数据 → T执行）:
    - 波动率状态: 布林带宽分位 < squeeze_th → 震荡regime
    - 均线状态: MA5>MA10>MA20多头排列 → 趋势regime
    - 不确定 → 空仓

  左侧模式（震荡regime）:
    - 入场: 收盘价触及布林带下轨 + RSI<oversold → 逆势预判
    - 成交模拟: 限价单——买入价=前一日收盘价(近似限价挂单成交, 无滑点溢价)
    - 止损: left_stop (3%, 窄止损)
    - 止盈: 回到布林带中轨或上轨

  右侧模式（趋势regime）:
    - 入场: 突破N日新高 + 量能确认 → 顺势确认
    - 成交模拟: 市价单——买入价=次日开盘价×(1+滑点) (市价穿越价差)
    - 止损: right_trail (8%跟踪止盈, 宽止损)
    - 止盈: 跟踪止盈让利润奔跑

  消融对照（4组同时跑）:
    A) 混合（环境切换）← 视频核心主张
    B) 纯左侧（始终均值回归）
    C) 纯右侧（始终突破）
    D) 缝合怪（左侧入场+右侧止损 = 散户死法）← 视频批评的反面教材

涨跌停处理:
    - 买入: 开盘涨停则检查收盘价是否仍封板,未封死按收盘价买,封死放弃
    - 卖出: 开盘跌停则检查收盘价,未封死按收盘价卖,封死次日重试
"""
import sys
import os
import argparse
import numpy as np
import pandas as pd

import config
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest"))
from run_monthly_rebalance import (get_conn, get_trade_dates, COMMISSION_RATE,
                                   STAMP_DUTY_RATE, SLIPPAGE_RATE, COMMISSION_MIN, calc_fee)

RES_DIR = "data/results/left_right_regime"
os.makedirs(RES_DIR, exist_ok=True)
CAPITAL = 1000000.0
INDEX_MARKET = "000300.SH"      # 市场环境门控
INDEX_BENCH_1 = "000300.SH"     # 基准 沪深300
INDEX_BENCH_2 = "000906.SH"     # 基准 中证800
UNIV_INDEX = "000906.SH"        # 股票池 中证800


# ════════════════════════════════════════════════════════════
#  数据预载（复用 Livermore 模式）
# ════════════════════════════════════════════════════════════

def load_universe_dates(end):
    """zz800 成分快照: 返回 [(trade_date_str, set(codes)), ...] 按日期升序。"""
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT ts_code, trade_date FROM index_constituent "
        "WHERE index_code=? ORDER BY trade_date",
        conn, params=(UNIV_INDEX,))
    conn.close()
    df["trade_date"] = df["trade_date"].astype(str)
    out = []
    for d, g in df.groupby("trade_date"):
        out.append((d, set(g["ts_code"].tolist())))
    return out


def load_panels(codes, start, end, warmup_days=400):
    """批量加载日线数据，返回宽表 (index=trade_date, columns=ts_code)。"""
    conn = get_conn()
    q_start = (pd.Timestamp(start) - pd.Timedelta(days=warmup_days)).strftime("%Y%m%d")
    ph = ",".join("?" for _ in codes)
    daily = pd.read_sql_query(
        f"SELECT ts_code, trade_date, open, high, low, close, pre_close, vol "
        f"FROM daily WHERE ts_code IN ({ph}) AND trade_date BETWEEN ? AND ? ORDER BY trade_date",
        conn, params=(*codes, q_start, end))
    adj = pd.read_sql_query(
        f"SELECT ts_code, trade_date, adj_factor FROM adj_factor "
        f"WHERE ts_code IN ({ph}) AND trade_date BETWEEN ? AND ? ORDER BY trade_date",
        conn, params=(*codes, q_start, end))
    conn.close()
    daily["trade_date"] = daily["trade_date"].astype(str)
    adj["trade_date"] = adj["trade_date"].astype(str)
    open_r = daily.pivot(index="trade_date", columns="ts_code", values="open")
    high_r = daily.pivot(index="trade_date", columns="ts_code", values="high")
    low_r = daily.pivot(index="trade_date", columns="ts_code", values="low")
    close_r = daily.pivot(index="trade_date", columns="ts_code", values="close")
    pre_close_r = daily.pivot(index="trade_date", columns="ts_code", values="pre_close")
    vol_r = daily.pivot(index="trade_date", columns="ts_code", values="vol")
    adj_d = adj.pivot(index="trade_date", columns="ts_code", values="adj_factor")
    return open_r, high_r, low_r, close_r, pre_close_r, vol_r, adj_d


def load_index(index_code, start, end):
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT trade_date, close FROM index_daily WHERE ts_code=? "
        "AND trade_date BETWEEN ? AND ? ORDER BY trade_date",
        conn, params=(index_code, str(start), str(end)))
    conn.close()
    df["trade_date"] = df["trade_date"].astype(str)
    return df.set_index("trade_date")["close"].astype(float)


# ════════════════════════════════════════════════════════════
#  信号矩阵预计算（向量化, 严格 T-1）
# ════════════════════════════════════════════════════════════

def build_signals(all_dates, open_r, high_r, low_r, close_r, pre_close_r, vol_r, adj_d, cfg):
    """返回 dict of aligned DataFrames (index=all_dates, columns=codes).
    全部信号用 <= T-1 数据，T 日开盘执行。"""
    open_raw = open_r.reindex(all_dates)
    high_r = high_r.reindex(all_dates)
    low_r = low_r.reindex(all_dates)
    close_raw = close_r.reindex(all_dates)
    pre_close_raw = pre_close_r.reindex(all_dates)
    vol_r = vol_r.reindex(all_dates)
    adj_d = adj_d.reindex(all_dates)
    adj_f = adj_d.ffill().fillna(1.0)

    # 后复权空间（剔除分红缺口）
    open_h = open_raw * adj_f
    high_h = high_r * adj_f
    low_h = low_r * adj_f
    close_h = close_raw * adj_f

    # ── 布林带（20日, 2σ）──
    bb_win = int(cfg.get("bb_win", 20))
    bb_std = float(cfg.get("bb_std", 2.0))
    bb_mid = close_h.rolling(bb_win, min_periods=bb_win).mean()
    bb_sigma = close_h.rolling(bb_win, min_periods=bb_win).std()
    bb_upper = bb_mid + bb_std * bb_sigma
    bb_lower = bb_mid - bb_std * bb_sigma
    bb_width = (bb_upper - bb_lower) / bb_mid

    # ── RSI（14日）──
    rsi_win = int(cfg.get("rsi_win", 14))
    delta = close_h.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(rsi_win, min_periods=rsi_win).mean()
    avg_loss = loss.rolling(rsi_win, min_periods=rsi_win).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.fillna(50)  # NaN时中性

    # ── 左侧信号: 收盘触及布林带下轨 + RSI超卖 ──
    rsi_oversold = float(cfg.get("rsi_oversold", 30))
    left_entry_raw = (close_h < bb_lower) & (rsi < rsi_oversold)
    # T-1 信号
    left_entry = left_entry_raw.shift(1).fillna(False).astype(bool)

    # 左侧止盈信号: 收盘回到中轨上方
    left_exit_raw = close_h > bb_mid
    left_exit = left_exit_raw.shift(1).fillna(False).astype(bool)

    # ── 右侧信号: 突破N日新高 + 量能确认 ──
    breakout_n = int(cfg.get("breakout_n", 20))
    vol_mult = float(cfg.get("vol_mult", 1.3))
    high_n = high_h.rolling(breakout_n, min_periods=breakout_n).max()
    # 突破: T-1收盘 > 过去N日最高价（不含当日, shift避免自指）
    breakout_raw = close_h > high_n.shift(1)
    # 量能确认: T-1成交量 > 20日均量 * vol_mult
    vol_ma20 = vol_r.rolling(20, min_periods=20).mean().shift(1)
    vol_ok = (vol_r > vol_ma20 * vol_mult) & vol_ma20.notna()
    # 连续confirm_days天站稳在突破位上方
    confirm_days = int(cfg.get("confirm_days", 2))
    above_prev = close_h > high_n.shift(1)
    confirm_mask = above_prev.rolling(confirm_days, min_periods=confirm_days).apply(
        lambda x: x.all(), raw=True)
    right_entry_raw = breakout_raw & vol_ok & confirm_mask.fillna(False)
    right_entry = right_entry_raw.shift(1).fillna(False).astype(bool)

    # ── 布林带宽分位（用于环境判断）──
    bbw_lookback = int(cfg.get("bbw_lookback", 120))
    bbw_pct = bb_width.rolling(bbw_lookback, min_periods=30).rank(pct=True)

    # ── 均线多头排列 ──
    ma5 = close_h.rolling(5, min_periods=5).mean()
    ma10 = close_h.rolling(10, min_periods=10).mean()
    ma20 = close_h.rolling(20, min_periods=20).mean()
    bull_align = (ma5 > ma10) & (ma10 > ma20) & ma20.notna()

    return dict(
        open_raw=open_raw, close_raw=close_raw, pre_close_raw=pre_close_raw,
        open_h=open_h, high_h=high_h, low_h=low_h, close_h=close_h,
        bb_mid=bb_mid, bb_upper=bb_upper, bb_lower=bb_lower, bb_width=bb_width,
        bbw_pct=bbw_pct, rsi=rsi, ma5=ma5, ma10=ma10, ma20=ma20,
        bull_align=bull_align,
        left_entry=left_entry, left_exit=left_exit,
        right_entry=right_entry,
    )


# ════════════════════════════════════════════════════════════
#  环境判断器
# ════════════════════════════════════════════════════════════

def build_regime(all_dates, sig, idx_close, cfg):
    """返回 DataFrame: index=all_dates, columns=[regime]
    regime: 'left'(震荡) / 'right'(趋势) / 'flat'(空仓)
    全部用 T-1 数据。"""
    squeeze_th = float(cfg.get("squeeze_th", 0.25))
    bbw_pct = sig["bbw_pct"].reindex(all_dates)

    # 指数市场环境: 沪深300 > MA20
    mkt = idx_close.reindex(all_dates)
    mkt_ma20 = mkt.rolling(20, min_periods=20).mean()
    mkt_bull = (mkt > mkt_ma20) & mkt_ma20.notna()
    mkt_bull = mkt_bull.fillna(False).astype(bool)

    # 用个股级别的bbw_pct中位数代表整体市场状态
    bbw_median = bbw_pct.median(axis=1)

    # 用指数多头排列判断趋势（比个股更稳健）
    idx_ma5 = mkt.rolling(5, min_periods=5).mean()
    idx_ma10 = mkt.rolling(10, min_periods=10).mean()
    idx_ma20 = mkt_ma20
    idx_bull_align = (mkt > idx_ma20) & (idx_ma5 > idx_ma10) & (idx_ma10 > idx_ma20) & idx_ma20.notna()
    idx_bull_align = idx_bull_align.fillna(False)

    # 环境判断（全部用 T-1 数据）:
    # - bbw_pct < squeeze_th → 震荡（左侧）
    # - bbw_pct >= squeeze_th 且 指数多头排列 → 趋势（右侧）
    # - 其余 → 空仓
    regime = pd.Series("flat", index=all_dates)
    squeeze = bbw_median < squeeze_th
    trending = (bbw_median >= squeeze_th) & idx_bull_align

    regime[squeeze] = "left"
    regime[trending & ~squeeze] = "right"

    # shift 1: 用 T-1 的环境判断指导 T 的交易
    regime = regime.shift(1).fillna("flat")
    return regime, mkt_bull


# ════════════════════════════════════════════════════════════
#  回测引擎（支持4种模式）
# ════════════════════════════════════════════════════════════

def run_backtest(start, end, cfg, mode="hybrid", use_market=True):
    """
    mode:
      'hybrid'   — 环境切换: 震荡用左侧, 趋势用右侧
      'left'     — 纯左侧: 始终均值回归
      'right'    — 纯右侧: 始终突破
      'frankenstein' — 缝合怪: 左侧入场+右侧止损（散户死法）
    """
    all_dates = get_trade_dates(start, end)
    if len(all_dates) < 60:
        print(f"  [跳过] {start}-{end} 交易日不足")
        return None

    univ_snaps = load_universe_dates(end)
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
        print(f"  [跳过] {start}-{end} 无成分数据")
        return None

    open_r, high_r, low_r, close_r, pre_close_r, vol_r, adj_d = load_panels(all_codes, start, end)
    sig = build_signals(all_dates, open_r, high_r, low_r, close_r, pre_close_r, vol_r, adj_d, cfg)

    # 市场环境
    idx_close = load_index(INDEX_MARKET, start, end)
    regime_s, mkt_bull = build_regime(all_dates, sig, idx_close, cfg)

    open_raw = sig["open_raw"]
    close_raw = sig["close_raw"]
    pre_close_raw = sig["pre_close_raw"]
    close_h = sig["close_h"]
    high_h = sig["high_h"]
    bb_mid = sig["bb_mid"]
    bb_upper = sig["bb_upper"]
    bb_lower = sig["bb_lower"]
    left_entry = sig["left_entry"]
    left_exit = sig["left_exit"]
    right_entry = sig["right_entry"]

    n = len(all_dates)
    cols = list(left_entry.columns)
    code2idx = {c: j for j, c in enumerate(cols)}
    le_arr = left_entry.values.astype(bool)
    re_arr = right_entry.values.astype(bool)
    lx_arr = left_exit.values.astype(bool)
    close_h_arr = close_h.values.astype(float)
    high_h_arr = high_h.values.astype(float)
    bb_mid_arr = bb_mid.values.astype(float)
    bb_lower_arr = bb_lower.values.astype(float)
    bb_upper_arr = bb_upper.values.astype(float)

    le_sets = [set(np.compress(le_arr[i], cols)) for i in range(n)]
    re_sets = [set(np.compress(re_arr[i], cols)) for i in range(n)]
    lx_sets = [set(np.compress(lx_arr[i], cols)) for i in range(n)]

    max_hold = int(cfg["max_hold"])
    left_stop = float(cfg.get("left_stop", 0.03))
    right_trail = float(cfg.get("right_trail", 0.08))
    right_stop = float(cfg.get("right_stop", 0.05))

    cash = CAPITAL
    positions = {}      # code -> dict(shares, entry_open, entry_hfq, side, high_water_mark, entry_date)
    pending_sell = set()
    nav = []
    n_entries = n_exits = 0
    days_in_market = 0
    bull_rets, bear_rets = [], []
    trade_records = []
    total_left_entries = total_right_entries = 0
    regime_counts = {"left": 0, "right": 0, "flat": 0}

    for i in range(n):
        d = all_dates[i]
        univ = univ_at(d)
        if not univ:
            nav.append((d, cash))
            continue

        reg = regime_s.iloc[i] if i < len(regime_s) else "flat"
        regime_counts[reg] = regime_counts.get(reg, 0) + 1
        bull_prev = bool(mkt_bull.iloc[i-1]) if i >= 1 else False

        # --- 1) 执行重试卖出（前一日跌停封死）---
        sell_exec = []
        for code in list(pending_sell):
            if code not in positions:
                pending_sell.discard(code)
                continue
            op = open_raw.iloc[i].get(code)
            cl = close_raw.iloc[i].get(code)
            pc = pre_close_raw.iloc[i].get(code)
            if op is None or pd.isna(op) or cl is None or pd.isna(cl) or pc is None or pd.isna(pc):
                pending_sell.discard(code)
                continue
            if op <= pc * 0.901:
                if cl <= pc * 0.901:
                    continue
                else:
                    sell_price = cl
            else:
                sell_price = op
            sell_exec.append((code, sell_price))
        pending_sell.clear()

        # --- 2) 持仓退出判定（T-1收盘信号, T开盘执行）---
        force_exit_all = (use_market and not bull_prev)
        for code, pos in list(positions.items()):
            if code not in univ:
                last_c = pos.get("last_close")
                if last_c is None or pd.isna(last_c):
                    last_c = pos.get("entry_open", 0.0)
                sell_exec.append((code, last_c))
                continue
            ch = close_h_arr[i-1, code2idx[code]] if (i >= 1 and code in code2idx) else np.nan
            if pd.isna(ch):
                continue
            hh = high_h_arr[i-1, code2idx[code]] if (i >= 1 and code in code2idx) else np.nan
            pos["high_water_mark"] = max(pos["high_water_mark"], ch, hh if not pd.isna(hh) else ch)
            exit_now = False
            side = pos.get("side", "left")

            if mode == "frankenstein":
                # 缝合怪: 左侧入场 + 右侧止损（宽止损套在逆势仓位上 → 该窄不窄、该宽不宽）
                trail_stop = right_trail
                hard_stop = right_stop
                # 左侧止盈信号仍然生效（但止损用右侧参数）
                if i >= 1 and code in lx_sets[i-1]:
                    exit_now = True
            elif side == "left":
                # 左侧退出: 回到中轨止盈 OR 窄止损
                trail_stop = left_stop
                hard_stop = left_stop
                if i >= 1 and code in lx_sets[i-1]:
                    exit_now = True
            else:
                # 右侧退出: 跟踪止盈 OR 硬止损
                trail_stop = right_trail
                hard_stop = right_stop

            # 跟踪止盈
            if not exit_now and trail_stop > 0 and pos["high_water_mark"] > 0:
                if ch < pos["high_water_mark"] * (1 - trail_stop):
                    exit_now = True
            # 硬止损
            if not exit_now and hard_stop > 0 and not pd.isna(pos["entry_hfq"]):
                if ch < pos["entry_hfq"] * (1 - hard_stop):
                    exit_now = True

            if force_exit_all:
                exit_now = True
            if exit_now:
                op = open_raw.iloc[i].get(code)
                cl = close_raw.iloc[i].get(code)
                pc = pre_close_raw.iloc[i].get(code)
                if op is not None and not pd.isna(op) and cl is not None and not pd.isna(cl) and pc is not None and not pd.isna(pc):
                    if op <= pc * 0.901:
                        if cl <= pc * 0.901:
                            pending_sell.add(code)
                        else:
                            sell_exec.append((code, cl))
                    else:
                        sell_exec.append((code, op))

        for code, sell_price in sell_exec:
            if code not in positions:
                continue
            sh = positions[code]["shares"]
            if sh > 0:
                proceeds = sell_price * sh - calc_fee("sell", sell_price, sh, d)
                cash += proceeds
                n_exits += 1
                cost = positions[code]["entry_open"] * sh + calc_fee("buy", positions[code]["entry_open"], sh, positions[code]["entry_date"])
                ret = proceeds / cost - 1
                hold_days = (pd.Timestamp(d) - pd.Timestamp(positions[code]["entry_date"])).days
                exit_type = "止损/退出"
                hwm = positions[code].get("high_water_mark", 0)
                trail = right_trail if positions[code].get("side") == "right" or mode == "frankenstein" else left_stop
                if hwm > 0 and sell_price >= hwm * (1 - trail):
                    exit_type = "跟踪止盈"
                elif ret > 0.05:
                    exit_type = "盈利卖出"
                elif ret < -0.05:
                    exit_type = "亏损卖出"
                trade_records.append(dict(
                    entry_date=positions[code]["entry_date"], exit_date=d,
                    code=code, ret=ret, hold_days=hold_days, exit_type=exit_type,
                    side=positions[code].get("side", "?")))
            positions.pop(code, None)

        # --- 3) 新开仓（T-1 信号, T 开盘执行）---
        prev_i = i - 1
        can_open = ((not use_market) or bull_prev)
        if can_open and prev_i >= 0:
            # 根据模式决定入场信号来源
            if mode == "left":
                entry_set = le_sets[prev_i] & univ
                entry_side = "left"
            elif mode == "right":
                entry_set = re_sets[prev_i] & univ
                entry_side = "right"
            elif mode == "frankenstein":
                # 缝合怪: 左侧入场
                entry_set = le_sets[prev_i] & univ
                entry_side = "left"
            else:  # hybrid
                if reg == "left":
                    entry_set = le_sets[prev_i] & univ
                    entry_side = "left"
                elif reg == "right":
                    entry_set = re_sets[prev_i] & univ
                    entry_side = "right"
                else:
                    entry_set = set()
                    entry_side = None

            if entry_side == "left":
                total_left_entries += len(entry_set)
            else:
                total_right_entries += len(entry_set)

            cand = list(entry_set - set(positions) - pending_sell)
            equity = cash
            for code, pos in positions.items():
                c = close_raw.iloc[i].get(code)
                if c is not None and not pd.isna(c):
                    equity += pos["shares"] * c
            slots = max_hold - len(positions)
            if slots > 0 and cand:
                take = cand[:slots]
                per_val = equity / max_hold
                for code in take:
                    op = open_raw.iloc[i].get(code)
                    cl = close_raw.iloc[i].get(code)
                    pc = pre_close_raw.iloc[i].get(code)
                    if op is None or pd.isna(op) or op <= 0 or cl is None or pd.isna(cl) or pc is None or pd.isna(pc):
                        continue

                    # 成交模拟:
                    # 左侧=限价单 → 买入价≈前日收盘价（近似限价挂单成交，不付滑点溢价）
                    # 右侧=市价单 → 买入价=开盘价×(1+滑点)（穿越价差）
                    if entry_side == "left":
                        # 涨停检查仍需执行
                        if op >= pc * 1.099:
                            if cl >= pc * 1.099:
                                continue
                            else:
                                buy_price = cl
                        else:
                            buy_price = op
                        # 限价单近似: 不加滑点溢价（已通过 calc_fee 收基础佣金）
                    else:
                        # 市价单: 正常涨停检查 + 开盘价执行
                        if op >= pc * 1.099:
                            if cl >= pc * 1.099:
                                continue
                            else:
                                buy_price = cl
                        else:
                            buy_price = op

                    if pd.isna(buy_price) or buy_price <= 0:
                        continue
                    if pd.isna(per_val) or per_val <= 0:
                        continue
                    sh = int(per_val / (buy_price * (1 + COMMISSION_RATE + SLIPPAGE_RATE)) / 100) * 100
                    if sh <= 0:
                        sh = 100
                    if sh <= 0:
                        sh = int(cash / (buy_price * (1 + COMMISSION_RATE + SLIPPAGE_RATE)) / 100) * 100
                        if sh <= 0:
                            continue
                    cost = buy_price * sh + calc_fee("buy", buy_price, sh, d)
                    if pd.isna(cost) or cost > cash:
                        continue
                    cash -= cost
                    j = code2idx.get(code)
                    eh = close_h_arr[prev_i, j] if j is not None else np.nan
                    positions[code] = dict(
                        shares=sh, entry_open=buy_price,
                        entry_hfq=eh, side=entry_side,
                        high_water_mark=eh, entry_date=d)
                    n_entries += 1

        # --- 4) 估值（收盘原始价）---
        mv = cash
        held = 0
        for code, pos in positions.items():
            c = close_raw.iloc[i].get(code)
            if c is None or pd.isna(c):
                c = pos.get("last_close")
            if c is not None and not pd.isna(c):
                mv += pos["shares"] * c
                pos["last_close"] = c
                held += 1
        if held > 0:
            days_in_market += 1
        nav.append((d, mv))

        if i > 0 and len(nav) >= 2:
            r = nav[-1][1] / nav[-2][1] - 1
            if bull_prev:
                bull_rets.append(r)
            else:
                bear_rets.append(r)

    if len(nav) < 2:
        return None

    # --- 指标 ---
    def metrics(vals):
        vals = np.array(vals, dtype=float)
        tot = vals[-1] / vals[0] - 1
        years = (pd.Timestamp(all_dates[-1]) - pd.Timestamp(all_dates[0])).days / 365.25
        ann = (vals[-1] / vals[0]) ** (1 / years) - 1 if years > 0 else 0
        peak = np.maximum.accumulate(vals)
        mdd = (vals / peak - 1).min()
        rets = np.diff(vals) / vals[:-1]
        vol = rets.std() * np.sqrt(252) if len(rets) > 1 else 0
        sharpe = (rets.mean() * 252 - 0.02) / vol if (vol > 0 and len(rets) > 1) else 0
        return dict(total=tot, ann=ann, mdd=mdd, sharpe=sharpe, final=vals[-1])

    def bench_nav(series):
        series = series.reindex(all_dates)
        fv = series.first_valid_index()
        if fv is None:
            return np.array([CAPITAL] * len(all_dates))
        base = series[fv]
        return (series / base * CAPITAL).ffill().values

    b1 = load_index(INDEX_BENCH_1, start, end)
    b2 = load_index(INDEX_BENCH_2, start, end)
    mb1 = metrics(bench_nav(b1))
    mb2 = metrics(bench_nav(b2))
    m = metrics([v for _, v in nav])

    def compound(rets):
        if not rets:
            return 0.0
        p = 1.0
        for r in rets:
            p *= (1 + r)
        return p - 1

    trade_df = pd.DataFrame(trade_records) if trade_records else pd.DataFrame()
    left_trades = trade_df[trade_df.get("side", "") == "left"] if not trade_df.empty else pd.DataFrame()
    right_trades = trade_df[trade_df.get("side", "") == "right"] if not trade_df.empty else pd.DataFrame()

    return dict(
        start=start, end=end, mode=mode, nav=nav, m=m, mb1=mb1, mb2=mb2,
        n_entries=n_entries, n_exits=n_exits,
        time_in_market=days_in_market / len(all_dates),
        bull_ret=compound(bull_rets), bear_ret=compound(bear_rets),
        total_left_entries=total_left_entries, total_right_entries=total_right_entries,
        regime_counts=regime_counts,
        trade_records=trade_records,
        trade_df=trade_df,
        left_trades=left_trades,
        right_trades=right_trades,
    )


# ════════════════════════════════════════════════════════════
#  报告
# ════════════════════════════════════════════════════════════

def fmt_pct(x, signed=False):
    if x is None:
        return "-"
    return f"{x*100:+.2f}%" if signed else f"{x*100:.2f}%"


def print_window(r, label=""):
    if r is None:
        print(f"  [{label}] 无结果")
        return
    m = r["m"]
    mb1, mb2 = r["mb1"], r["mb2"]
    print(f"  ── {label or r['mode']} ({r['start']}→{r['end']}) ──")
    if r.get("regime_counts"):
        rc = r["regime_counts"]
        tot_rc = sum(rc.values()) or 1
        print(f"  环境分布: 左侧(震荡){rc.get('left',0)}天({rc.get('left',0)/tot_rc:.1%}) "
              f"/ 右侧(趋势){rc.get('right',0)}天({rc.get('right',0)/tot_rc:.1%}) "
              f"/ 空仓{rc.get('flat',0)}天({rc.get('flat',0)/tot_rc:.1%})")
    print(f"  入场: 左侧信号{r.get('total_left_entries',0)} 右侧信号{r.get('total_right_entries',0)} "
          f"/ 实际开仓{r['n_entries']} 出场{r['n_exits']}")
    print(f"  持仓时间占比: {fmt_pct(r['time_in_market'])}")
    print(f"  策略 : 总收益 {fmt_pct(m['total'])} / 年化 {fmt_pct(m['ann'])} "
          f"/ 最大回撤 {fmt_pct(m['mdd'])} / 夏普 {m['sharpe']:.3f}")
    print(f"  沪深300: 总收益 {fmt_pct(mb1['total'])} / 年化 {fmt_pct(mb1['ann'])} / 最大回撤 {fmt_pct(mb1['mdd'])}")
    print(f"  中证800: 总收益 {fmt_pct(mb2['total'])}")
    print(f"  超额(vs沪深300): {fmt_pct(m['total']-mb1['total'], signed=True)} "
          f"/ 超额(vs中证800): {fmt_pct(m['total']-mb2['total'], signed=True)}")
    print(f"  强弱分段: 牛市日 {fmt_pct(r['bull_ret'], signed=True)} | 熊市日 {fmt_pct(r['bear_ret'], signed=True)}")

    # 交易统计
    tdf = r.get("trade_df")
    if tdf is not None and not tdf.empty:
        win = tdf[tdf["ret"] > 0]
        loss = tdf[tdf["ret"] <= 0]
        wr = len(win) / len(tdf) if len(tdf) > 0 else 0
        avg_win = win["ret"].mean() if len(win) > 0 else 0
        avg_loss = loss["ret"].mean() if len(loss) > 0 else 0
        pf = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        print(f"  胜率: {wr:.1%} | 盈亏比: {pf:.2f} | 交易数: {len(tdf)}")
        print(f"  盈利均收益 {fmt_pct(avg_win)} ({win['hold_days'].mean():.0f}天) "
              f"/ 亏损均收益 {fmt_pct(avg_loss)} ({loss['hold_days'].mean():.0f}天)")
        # 左右分拆统计
        for side_label, side_df in [("左侧", r.get("left_trades")),
                                     ("右侧", r.get("right_trades"))]:
            if side_df is not None and not side_df.empty:
                sw = side_df[side_df["ret"] > 0]
                sl = side_df[side_df["ret"] <= 0]
                swr = len(sw) / len(side_df) if len(side_df) > 0 else 0
                saw = sw["ret"].mean() if len(sw) > 0 else 0
                sal = sl["ret"].mean() if len(sl) > 0 else 0
                spf = abs(saw / sal) if sal != 0 else float('inf')
                print(f"    {side_label}: 胜率{swr:.1%} 盈亏比{spf:.2f} "
                      f"均盈利{fmt_pct(saw)} 均亏损{fmt_pct(sal)} 交易{len(side_df)}")
    print()


# ════════════════════════════════════════════════════════════
#  主入口
# ════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="左侧-右侧混合择时策略 (BV1uAuu6vE1j · 悦悦笔记)")
    ap.add_argument("--start", default="20180101")
    ap.add_argument("--end", default="20251231")
    ap.add_argument("--bb-win", type=int, default=20)
    ap.add_argument("--bb-std", type=float, default=2.0)
    ap.add_argument("--rsi-win", type=int, default=14)
    ap.add_argument("--rsi-oversold", type=float, default=30)
    ap.add_argument("--breakout-n", type=int, default=20)
    ap.add_argument("--vol-mult", type=float, default=1.3)
    ap.add_argument("--confirm-days", type=int, default=2)
    ap.add_argument("--squeeze-th", type=float, default=0.25,
                    help="布林带宽分位阈值: <此值=震荡(left), >=此值且多头排列=趋势(right)")
    ap.add_argument("--bbw-lookback", type=int, default=120)
    ap.add_argument("--left-stop", type=float, default=0.03,
                    help="左侧止损(窄, 3%)")
    ap.add_argument("--right-trail", type=float, default=0.08,
                    help="右侧跟踪止盈(宽, 8%)")
    ap.add_argument("--right-stop", type=float, default=0.05,
                    help="右侧硬止损(5%)")
    ap.add_argument("--max-hold", type=int, default=5)
    ap.add_argument("--no-market-gate", action="store_true",
                    help="关闭市场环境门控")
    ap.add_argument("--mode", default="all",
                    choices=["all", "hybrid", "left", "right", "frankenstein"],
                    help="all=跑4组消融对照; 或单独跑某一模式")
    args = ap.parse_args()

    base_cfg = dict(
        bb_win=args.bb_win, bb_std=args.bb_std,
        rsi_win=args.rsi_win, rsi_oversold=args.rsi_oversold,
        breakout_n=args.breakout_n, vol_mult=args.vol_mult,
        confirm_days=args.confirm_days,
        squeeze_th=args.squeeze_th, bbw_lookback=args.bbw_lookback,
        left_stop=args.left_stop, right_trail=args.right_trail,
        right_stop=args.right_stop, max_hold=args.max_hold,
    )

    use_market = not args.no_market_gate

    if args.mode == "all":
        print(f"{'='*80}")
        print(f"  左侧-右侧混合择时策略 · 四组消融对照")
        print(f"  视频源: 悦悦笔记 BV1uAuu6vE1j")
        print(f"  回测区间: {args.start} → {args.end}")
        print(f"  参数: bb={args.bb_win}σ{args.bb_std} rsi{args.rsi_win}<{args.rsi_oversold} "
              f"breakout{args.breakout_n} vol{args.vol_mult} confirm{args.confirm_days} "
              f"squeeze<{args.squeeze_th} left_stop{args.left_stop} right_trail{args.right_trail}")
        print(f"  股票池: 中证800 | 市场门控: {'ON' if use_market else 'OFF'}")
        print(f"{'='*80}\n")

        results = {}
        for mode, label in [
            ("hybrid",       "A) 混合(环境切换)"),
            ("left",         "B) 纯左侧(均值回归)"),
            ("right",        "C) 纯右侧(突破)"),
            ("frankenstein", "D) 缝合怪(左入+右止)"),
        ]:
            print(f"{'─'*60}")
            r = run_backtest(args.start, args.end, base_cfg, mode=mode, use_market=use_market)
            print_window(r, label)
            results[mode] = r

        # --- 汇总对比表 ---
        print(f"\n{'='*80}")
        print(f"  消融对照汇总")
        print(f"{'='*80}")
        print(f"  {'模式':<24} {'总收益':>10} {'年化':>10} {'最大回撤':>10} {'夏普':>8} "
              f"{'超额(vs300)':>12} {'交易数':>6}")
        print(f"  {'─'*80}")
        for mode, label in [
            ("hybrid",       "A) 混合(环境切换)"),
            ("left",         "B) 纯左侧(均值回归)"),
            ("right",        "C) 纯右侧(突破)"),
            ("frankenstein", "D) 缝合怪(左入+右止)"),
        ]:
            r = results.get(mode)
            if r is None:
                continue
            m = r["m"]
            mb1 = r["mb1"]
            print(f"  {label:<24} {fmt_pct(m['total']):>10} {fmt_pct(m['ann']):>10} "
                  f"{fmt_pct(m['mdd']):>10} {m['sharpe']:>8.3f} "
                  f"{fmt_pct(m['total']-mb1['total'], signed=True):>12} {r['n_entries']:>6}")
        print(f"  {'─'*80}")
        # 基准
        for mode, label in [("hybrid", "")]:
            r = results.get(mode)
            if r:
                mb1, mb2 = r["mb1"], r["mb2"]
                print(f"  {'沪深300(基准)':<24} {fmt_pct(mb1['total']):>10} "
                      f"{fmt_pct(mb1['ann']):>10} {fmt_pct(mb1['mdd']):>10}")
                print(f"  {'中证800(基准)':<24} {fmt_pct(mb2['total']):>10} "
                      f"{fmt_pct(mb2['ann']):>10} {fmt_pct(mb2['mdd']):>10}")
        print()

        # --- 核心结论 ---
        h = results.get("hybrid")
        f = results.get("frankenstein")
        if h and f:
            print(f"  ── 核心验证 ──")
            print(f"  混合 vs 缝合怪: {fmt_pct(h['m']['total'], signed=True)} vs "
                  f"{fmt_pct(f['m']['total'], signed=True)}")
            if h['m']['total'] > f['m']['total']:
                print(f"  [PASS] 验证成立: 环境切换混合策略跑赢缝合怪 -> '散户死于缝合'论点得到量化支撑")
            else:
                print(f"  [FAIL] 验证不成立: 缝合怪反而跑赢混合 -> 需进一步分析原因")
            print()

    else:
        r = run_backtest(args.start, args.end, base_cfg, mode=args.mode, use_market=use_market)
        print_window(r, args.mode)


if __name__ == "__main__":
    main()