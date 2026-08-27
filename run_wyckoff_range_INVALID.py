# -*- coding: utf-8 -*-
# ============================================================================
# ❌ INVALIDATED — 威科夫区间突破策略 (BV1Zg876WE8V · 跟着Jim学量化 · 白名单)
# ----------------------------------------------------------------------------
# 判定: 作为字面买卖规则**无正期望**, 经 2018-2025 全样本严谨回测证伪。
#   - 主配置: 总收益 −19.31% / 年化 −2.65% / 最大回撤 −41.30% / 夏普 −0.308
#   - 沪深300基准 +13.27% / 中证800 +14.37% → 超额 −32.6pp (跑输基准)
#   - 消融: 市场门控仅减亏(ON −16.6% / OFF −33.2%), 不扭转负期望
#   - 18组参数敏感性网格**全组净负、超额全负、无一组跑赢基准**(−23.4%~−71.9%)
# 结论: 与Jim系列(隔夜/主升浪/利弗莫尔/达瓦斯/缩量回踩)同族——信号层无alpha,
#       仅"市场环境门控"常识风控有用。Jim 白名单/零荐股/主动免责, 不主张是骗子。
# 保留作证据归档(可复跑复核), 不作为可实盘策略。详见 bilibili-critical-summary §5.17。
# ============================================================================
"""
威科夫(Wyckoff)横盘区间突破策略 —— 量化复现版
=============================================
视频: "威科夫交易法——横盘之后方向怎么看"
      (UP主: 跟着Jim学量化, BV1Zg876WE8V, 白名单第4条)

策略规则（完全数字化，规避所有回测作弊逻辑）:
  区间识别(找区间):
    - 取近 N 日 close 包络: support=窗口最低收, resistance=窗口最高收, mid=(sup+res)/2
    - 区间有效条件(同时成立):
        1) 相对宽度 width=(res-sup)/mid ∈ [min_w, max_w]  —— 真横盘,非趋势非直线
        2) 窗口内至少 touch_min>=2 天收在支撑附近、touch_max>=2 天收在阻力附近 —— 反复震荡、多次在相近位置止跌/遇压
        3) 非趋势: 窗口首尾收盘差/mid 小于 trend_tol —— 横着走
  突破(看出走) + 试探(看放量后价前进还是被推回):
    - 突破预备: T-1 收盘(hfq) > resistance*(1+break_thr)
    - 量能确认: T-1 成交量 > 近20日均量*vol_mult
    - 价前进(非被推回): T-1 收盘在当日[hfq开, hfq高]上半区(close >= (open+high)/2)
    - 假突破过滤: 上述条件连续 confirm_days 天成立(滚动), 才触发入场
  退出:
    - 跟踪止盈: 从持仓最高价(hfq)回撤 trailing_stop 触发
    - 硬止损: 跌破入场价 hfoq 的 stop_loss
    - 跌破区间下沿(失败离场): 收盘 hfoq < 入场时支撑*(1-break_thr) —— 突破失败/派发确认
    - 市场转熊(可选): 沪深300跌破MA20 → 整批清仓
  涨跌停处理:
    - 买入: 开盘涨停则检查收盘价是否仍封板,未封死按收盘价买,封死放弃
    - 卖出: 开盘跌停则检查收盘价,未封死按收盘价卖,封死次日重试

注: 威科夫名义含吸筹(横盘后向上)与派发(横盘后向下)。A股多头策略只做吸筹→向上突破一侧,
    派发侧作为"跌破下沿=离场/规避"信号使用,不裸做空。
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

RES_DIR = "data/results/wyckoff"
os.makedirs(RES_DIR, exist_ok=True)
CAPITAL = 1000000.0
INDEX_MARKET = "000300.SH"      # 市场环境门控 / 熊市清仓
INDEX_BENCH_1 = "000300.SH"     # 基准 沪深300
INDEX_BENCH_2 = "000906.SH"     # 基准 中证800 (股票池)
UNIV_INDEX = "000906.SH"        # 股票池 中证800


# ════════════════════════════════════════════════════════════
#  数据预载
# ════════════════════════════════════════════════════════════

def load_universe_dates(end):
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
#  矩阵预计算（向量化信号, 严格 T-1）
# ════════════════════════════════════════════════════════════

def build_signals(all_dates, open_r, high_r, low_r, close_r, pre_close_r, vol_r, adj_d, cfg):
    """返回 dict of aligned DataFrames (index=all_dates, columns=codes). 全部用 <= T-1 数据。"""
    # 截断到 all_dates 范围（warmup 只用于预热）
    open_raw = open_r.reindex(all_dates)
    high_r = high_r.reindex(all_dates)
    low_r = low_r.reindex(all_dates)
    close_raw = close_r.reindex(all_dates)
    pre_close_raw = pre_close_r.reindex(all_dates)
    vol_r = vol_r.reindex(all_dates)
    adj_d = adj_d.reindex(all_dates)
    adj_f = adj_d.ffill().fillna(1.0)

    # hfq 空间（剔除分红缺口）
    open_h = open_raw * adj_f
    high_h = high_r * adj_f
    close_h = close_raw * adj_f

    N = int(cfg["range_n"])
    min_w = float(cfg["min_w"])
    max_w = float(cfg["max_w"])
    touch_tol = float(cfg["touch_tol"])
    touch_min = int(cfg["touch_min"])
    trend_tol = float(cfg["trend_tol"])
    break_thr = float(cfg["break_thr"])
    vol_mult = float(cfg["vol_mult"])
    confirm_days = int(cfg["confirm_days"])

    # —— 区间识别 ——
    # 关键修正: 用"排除突破判定窗口"的固定包络(pivot envelope), 窗口止于 t-(confirm_days+1)。
    # 原实现用包含当天的滚动窗口当上沿, 导致"收盘突破区间上沿"在数学上恒不成立(上沿已含该收盘)→ 0 信号。
    # pivot envelope 不含突破判定窗口, "收盘站稳上沿上方"才成为可能。
    rmin = close_h.rolling(N, min_periods=N).min()
    rmax = close_h.rolling(N, min_periods=N).max()
    pivot_high = rmax.shift(confirm_days + 1)   # 截至 t-(confirm_days+1) 的区间上沿
    pivot_low = rmin.shift(confirm_days + 1)    # 区间下沿
    mid_p = (pivot_high + pivot_low) / 2.0
    width_p = (pivot_high - pivot_low) / mid_p

    # 触边计数: pivot 窗口内收在支撑/阻力附近的交易日数
    near_low = (close_h <= rmin * (1 + touch_tol))
    near_high = (close_h >= rmax * (1 - touch_tol))
    cnt_low = near_low.shift(confirm_days + 1).rolling(N, min_periods=N).sum()
    cnt_high = near_high.shift(confirm_days + 1).rolling(N, min_periods=N).sum()

    # 非趋势: pivot 窗口首尾收盘差/mid 小
    trend_p = (close_h.shift(confirm_days + 1) - close_h.shift(confirm_days + 1 + N)) / mid_p

    range_valid = (
        (width_p >= min_w) & (width_p <= max_w) &
        (cnt_low >= touch_min) & (cnt_high >= touch_min) &
        (trend_p.abs() < trend_tol) &
        mid_p.notna()
    )
    range_valid = range_valid.fillna(False)

    # —— 突破确认: 最近 confirm_days 天收盘全部站稳在区间上沿上方(回踩不掉回) ——
    above_cols = [
        (close_h.shift(k) > pivot_high * (1 + break_thr))
        for k in range(1, confirm_days + 1)
    ]
    breakout_ready = above_cols[0]
    for c in above_cols[1:]:
        breakout_ready = breakout_ready & c
    breakout_ready = breakout_ready & range_valid

    # 量能确认 + 价前进(放量后价格继续前进而非被推回), 仅作用于触发日 t-1
    vol_ma20 = vol_r.rolling(20, min_periods=20).mean().shift(1)
    vol_ok = (vol_r.shift(1) > vol_ma20 * vol_mult) & vol_ma20.notna()
    o_t = open_h.shift(1)
    h_t = high_h.shift(1)
    price_advanced = (close_h.shift(1) >= (o_t + h_t) / 2.0) & h_t.notna() & (h_t > o_t)
    breakout_ready = breakout_ready & vol_ok & price_advanced
    breakout_ready = breakout_ready.fillna(False)

    # 入场信号 = 突破确认(已含 confirm_days 连续站稳)
    entry_ready = breakout_ready

    support_at_entry = pivot_low   # 入场时区间下沿(跌破=失败离场)

    return dict(
        open_raw=open_raw, close_raw=close_raw, pre_close_raw=pre_close_raw,
        open_h=open_h, high_h=high_h, close_h=close_h,
        entry_ready=entry_ready, support_at_entry=support_at_entry,
    )


# ════════════════════════════════════════════════════════════
#  回测主循环
# ════════════════════════════════════════════════════════════

def run_window(start, end, cfg, use_market=True):
    all_dates = get_trade_dates(start, end)
    if len(all_dates) < 60:
        print(f"  [跳过] {start}-{end} 交易日不足"); return None

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
        print(f"  [跳过] {start}-{end} 无成分数据"); return None

    open_r, high_r, low_r, close_r, pre_close_r, vol_r, adj_d = load_panels(all_codes, start, end)
    sig = build_signals(all_dates, open_r, high_r, low_r, close_r, pre_close_r, vol_r, adj_d, cfg)

    # 市场门控: 沪深300 > MA20
    idx_close = load_index(INDEX_MARKET, start, end)
    mkt = idx_close.reindex(all_dates)
    mkt_ma20 = mkt.rolling(20, min_periods=20).mean()
    bull = (mkt > mkt_ma20) & mkt_ma20.notna()
    bull = bull.fillna(False).astype(bool)

    open_raw = sig["open_raw"]
    close_raw = sig["close_raw"]
    pre_close_raw = sig["pre_close_raw"]
    er = sig["entry_ready"]
    sup = sig["support_at_entry"]
    close_h = sig["close_h"]
    high_h = sig["high_h"]

    n = len(all_dates)
    cols = list(er.columns)
    code2idx = {c: j for j, c in enumerate(cols)}
    er_arr = er.values.astype(bool)
    close_h_arr = close_h.values.astype(float)
    high_h_arr = high_h.values.astype(float)
    sup_arr = sup.values.astype(float)
    er_sets = [set(np.compress(er_arr[i], cols)) for i in range(n)]

    max_hold = int(cfg["max_hold"])
    trailing_stop = float(cfg["trailing_stop"])
    stop_loss = float(cfg["stop_loss"])
    break_thr = float(cfg["break_thr"])
    market_exit = bool(cfg.get("market_exit", True)) and use_market

    cash = CAPITAL
    positions = {}      # code -> dict(shares, entry_open, entry_hfq, support_hfq, high_water_mark, entry_date)
    pending_sell = set()
    nav = []
    n_entries = n_exits = 0
    days_in_market = 0
    bull_rets, bear_rets = [], []
    trade_records = []
    total_entry_signal = total_open = 0

    for i in range(n):
        d = all_dates[i]
        univ = univ_at(d)
        if not univ:
            nav.append((d, cash)); continue
        bull_prev = bool(bull.iloc[i-1]) if i >= 1 else False

        # --- 1) 执行重试卖出（前一日跌停封死）---
        sell_exec = []
        for code in list(pending_sell):
            if code not in positions:
                pending_sell.discard(code); continue
            op = open_raw.iloc[i].get(code); cl = close_raw.iloc[i].get(code); pc = pre_close_raw.iloc[i].get(code)
            if op is None or pd.isna(op) or cl is None or pd.isna(cl) or pc is None or pd.isna(pc):
                pending_sell.discard(code); continue
            if op <= pc * 0.901:
                if cl <= pc * 0.901:
                    continue
                else:
                    sell_price = cl
            else:
                sell_price = op
            sell_exec.append((code, sell_price))
        pending_sell.clear()

        # --- 2) 持仓退出判定（T 收盘信号, 次日开盘执行）---
        force_exit_all = (market_exit and use_market and not bull_prev)
        for code, pos in list(positions.items()):
            if code not in univ:
                last_c = pos.get("last_close")
                if last_c is None or pd.isna(last_c):
                    last_c = pos.get("entry_open", 0.0)
                sell_exec.append((code, last_c)); continue
            ch = close_h_arr[i-1, code2idx[code]] if (i >= 1 and code in code2idx) else np.nan
            if pd.isna(ch):
                continue
            hh = high_h_arr[i-1, code2idx[code]] if (i >= 1 and code in code2idx) else np.nan
            pos["high_water_mark"] = max(pos["high_water_mark"], ch, hh if not pd.isna(hh) else ch)
            exit_now = False
            if trailing_stop > 0 and pos["high_water_mark"] > 0 and ch < pos["high_water_mark"] * (1 - trailing_stop):
                exit_now = True
            if not exit_now and stop_loss > 0 and not pd.isna(pos["entry_hfq"]) and ch < pos["entry_hfq"] * (1 - stop_loss):
                exit_now = True
            # 跌破区间下沿(失败离场): 收盘 hfq < 入场支撑*(1-break_thr)
            if not exit_now and not pd.isna(pos["support_hfq"]) and ch < pos["support_hfq"] * (1 - break_thr):
                exit_now = True
            if force_exit_all:
                exit_now = True
            if exit_now:
                op = open_raw.iloc[i].get(code); cl = close_raw.iloc[i].get(code); pc = pre_close_raw.iloc[i].get(code)
                if op is not None and not pd.isna(op) and cl is not None and not pd.isna(cl) and pc is not None and not pd.isna(pc):
                    if op <= pc * 0.901:
                        if cl <= pc * 0.901:
                            pending_sell.add(code)
                        else:
                            sell_exec.append((code, cl))
                    else:
                        sell_exec.append((code, op))

        for code, sell_price in sell_exec:
            if code not in positions: continue
            sh = positions[code]["shares"]
            if sh > 0:
                proceeds = sell_price * sh - calc_fee("sell", sell_price, sh, d)
                cash += proceeds
                n_exits += 1
                cost = positions[code]["entry_open"] * sh + calc_fee("buy", positions[code]["entry_open"], sh, positions[code]["entry_date"])
                ret = proceeds / cost - 1
                hold_days = (pd.Timestamp(d) - pd.Timestamp(positions[code]["entry_date"])).days
                exit_type = "跌破下沿/失败" if ret < 0 and (positions[code]["support_hfq"] is not None) else "退出"
                if pos_hwm(positions[code]) > 0 and sell_price >= pos_hwm(positions[code]) * (1 - trailing_stop):
                    exit_type = "跟踪止盈"
                elif ret > 0.05:
                    exit_type = "盈利卖出"
                elif ret < -0.05:
                    exit_type = "亏损卖出"
                trade_records.append(dict(entry_date=positions[code]["entry_date"], exit_date=d,
                                          code=code, ret=ret, hold_days=hold_days, exit_type=exit_type))
            positions.pop(code, None)

        # --- 3) 新开仓（T-1 信号, T 开盘执行）---
        can_open = ((not use_market) or bull_prev)
        prev_i = i - 1
        if can_open and prev_i >= 0:
            prev_entry = er_sets[prev_i] & univ
            total_entry_signal += len(prev_entry)
            cand = list(prev_entry - set(positions) - pending_sell)
            equity = cash
            for code, pos in positions.items():
                c = close_raw.iloc[i].get(code)
                if c is not None and not pd.isna(c):
                    equity += pos["shares"] * c
            slots = max_hold - len(positions)
            if slots > 0 and cand:
                take = cand[:slots]
                per_vals = [equity / max_hold] * len(take)
                for k, code in enumerate(take):
                    per_val = per_vals[k]
                    op = open_raw.iloc[i].get(code); cl = close_raw.iloc[i].get(code); pc = pre_close_raw.iloc[i].get(code)
                    if op is None or pd.isna(op) or op <= 0 or cl is None or pd.isna(cl) or pc is None or pd.isna(pc):
                        continue
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
                    sh0 = sup_arr[prev_i, j] if j is not None else np.nan
                    positions[code] = dict(shares=sh, entry_open=buy_price,
                                           entry_hfq=eh, support_hfq=sh0, high_water_mark=eh,
                                           entry_date=d)
                    n_entries += 1
                    total_open += 1

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
        if not rets: return 0.0
        p = 1.0
        for r in rets: p *= (1 + r)
        return p - 1
    bull_ret = compound(bull_rets)
    bear_ret = compound(bear_rets)

    return dict(start=start, end=end, nav=nav, m=m, mb1=mb1, mb2=mb2,
                n_entries=n_entries, n_exits=n_exits,
                time_in_market=days_in_market / len(all_dates),
                bull_ret=bull_ret, bear_ret=bear_ret,
                total_entry_signal=total_entry_signal, total_open=total_open,
                trade_records=trade_records)


def pos_hwm(pos):
    return pos.get("high_water_mark", 0.0)


# ════════════════════════════════════════════════════════════
#  报告
# ════════════════════════════════════════════════════════════

def fmt_pct(x, signed=False):
    if x is None: return "-"
    return f"{x*100:+.2f}%" if signed else f"{x*100:.2f}%"


def print_window(r):
    if r is None: return
    m, mb1, mb2 = r["m"], r["mb1"], r["mb2"]
    print(f"  入场信号数(累计触发){r['total_entry_signal']} → 实际开仓{r['total_open']}")
    print(f"  交易: 入场 {r['n_entries']} / 出场 {r['n_exits']} / 持仓时间占比 {fmt_pct(r['time_in_market'])}")
    print(f"  策略 : 总收益 {fmt_pct(m['total'])} / 年化 {fmt_pct(m['ann'])} / 最大回撤 {fmt_pct(m['mdd'])} / 夏普 {m['sharpe']:.3f}")
    print(f"  沪深300: 总收益 {fmt_pct(mb1['total'])} / 年化 {fmt_pct(mb1['ann'])} / 最大回撤 {fmt_pct(mb1['mdd'])}")
    print(f"  中证800: 总收益 {fmt_pct(mb2['total'])}")
    print(f"  超额(策略-沪深300): {fmt_pct(m['total']-mb1['total'], signed=True)} | 超额(策略-中证800): {fmt_pct(m['total']-mb2['total'], signed=True)}")
    print(f"  强弱分段: 牛市日收益 {fmt_pct(r['bull_ret'], signed=True)} | 熊市日收益 {fmt_pct(r['bear_ret'], signed=True)}")
    trade_df = pd.DataFrame(r.get("trade_records", []))
    if not trade_df.empty:
        win_df = trade_df[trade_df["ret"] > 0]; loss_df = trade_df[trade_df["ret"] <= 0]
        print(f"  胜率: {len(win_df)/len(trade_df):.2%} | 盈亏比: {abs(win_df['ret'].mean()/loss_df['ret'].mean()):.2f}" if not loss_df.empty and loss_df['ret'].mean()!=0 else "  盈亏比: 无穷大")
        print(f"  盈利均收益 {win_df['ret'].mean():.2%}({win_df['hold_days'].mean():.1f}天) | 亏损均收益 {loss_df['ret'].mean():.2%}({loss_df['hold_days'].mean():.1f}天)")
        for p in [0.1, 0.5, 0.9]:
            print(f"    {int(p*100)}分位收益: {trade_df['ret'].quantile(p):.2%}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20180101")
    ap.add_argument("--end", default="20251231")
    ap.add_argument("--range-n", type=int, default=60)
    ap.add_argument("--min-w", type=float, default=0.12)
    ap.add_argument("--max-w", type=float, default=0.45)
    ap.add_argument("--touch-tol", type=float, default=0.02)
    ap.add_argument("--touch-min", type=int, default=2)
    ap.add_argument("--trend-tol", type=float, default=0.15)
    ap.add_argument("--break-thr", type=float, default=0.01)
    ap.add_argument("--vol-mult", type=float, default=1.3)
    ap.add_argument("--confirm-days", type=int, default=3)
    ap.add_argument("--trailing-stop", type=float, default=0.10)
    ap.add_argument("--stop-loss", type=float, default=0.05)
    ap.add_argument("--max-hold", type=int, default=5)
    ap.add_argument("--no-market-gate", action="store_true", help="关闭市场环境门控(仅关新开,保留熊市清仓需另设)")
    ap.add_argument("--sensitivity", action="store_true", help="跑参数敏感性网格(range_n x break_thr x confirm_days)")
    ap.add_argument("--ablation", action="store_true", help="消融: 市场门控开/关")
    args = ap.parse_args()

    base_cfg = dict(range_n=args.range_n, min_w=args.min_w, max_w=args.max_w,
                    touch_tol=args.touch_tol, touch_min=args.touch_min, trend_tol=args.trend_tol,
                    break_thr=args.break_thr, vol_mult=args.vol_mult, confirm_days=args.confirm_days,
                    trailing_stop=args.trailing_stop, stop_loss=args.stop_loss, max_hold=args.max_hold,
                    market_exit=not args.no_market_gate)

    if args.ablation:
        print(f"=== 消融: 市场门控 开/关 (区间突破纯逻辑) ===\n")
        for label, um in [("市场门控ON", True), ("市场门控OFF", False)]:
            print(f"----- {label} -----")
            r = run_window(args.start, args.end, base_cfg, use_market=um)
            print_window(r)
        return

    if args.sensitivity:
        print(f"=== 敏感性网格 (zz800池, {args.start}-{args.end}) ===\n")
        rows = []
        for rn in [40, 60, 90]:
            for bthr in [0.005, 0.01, 0.02]:
                for cd in [2, 3]:
                    cfg = dict(base_cfg, range_n=rn, break_thr=bthr, confirm_days=cd)
                    r = run_window(args.start, args.end, cfg, use_market=True)
                    if r is None: continue
                    rows.append(dict(range_n=rn, break_thr=bthr, confirm=cd,
                                     total=fmt_pct(r["m"]["total"]), ann=fmt_pct(r["m"]["ann"]),
                                     mdd=fmt_pct(r["m"]["mdd"]), sharpe=f"{r['m']['sharpe']:.2f}",
                                     ex_hs300=fmt_pct(r["m"]["total"]-r["mb1"]["total"], signed=True),
                                     open=r["total_open"]))
        print(pd.DataFrame(rows).to_string(index=False))
        return

    r = run_window(args.start, args.end, base_cfg, use_market=not args.no_market_gate)
    print(f"===== 威科夫区间突破 {args.start}→{args.end} | 配置 range_n={args.range_n} min_w={args.min_w} max_w={args.max_w} break_thr={args.break_thr} confirm={args.confirm_days} vol_mult={args.vol_mult} trail={args.trailing_stop} stop={args.stop_loss} max_hold={args.max_hold} =====\n")
    print_window(r)


if __name__ == "__main__":
    main()
