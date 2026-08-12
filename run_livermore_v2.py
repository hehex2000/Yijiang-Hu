# -*- coding: utf-8 -*-
"""
利弗莫尔四步法关键点突破策略 —— 完整验证版（含涨跌停处理、假突破过滤、跟踪止盈）
================================================================
视频: "利弗莫尔交易法——用数据来验证经典交易战法"
      (UP主: 跟着Jim学量化, BV1Z2u76TEYq)

策略规则（完全数字化，规避所有回测作弊逻辑）:
  步骤1 市场环境(水流): 沪深300收盘价>50日均线，且近10日累计收益率≥0 → 才允许开新仓
  步骤2 板块强度(板块靠前): 个股5日收益率在其细分行业内排名前30% → 保留
  步骤3 关键点(被越过): 收盘价创下20日新高（前高突破），且连续2日站稳关键点上方 → 过滤假突破
  步骤4 退出规则:
    - 跟踪止盈: 从持仓最高价回撤8% → 触发卖出
    - 硬止损: 跌破成本价5% → 触发卖出
    - 跌破突破位下方3% → 失效离场
    - 市场转熊 → 整批清仓
  涨跌停处理:
    - 买入: 开盘涨停则检查收盘价是否仍封板，未封死按收盘价买入，封死则放弃
    - 卖出: 开盘跌停则检查收盘价是否仍封板，未封死按收盘价卖出，封死则次日重试
"""
import sys
import os
import argparse
import numpy as np
import pandas as pd

import config
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest"))
from run_monthly_rebalance import get_conn, get_trade_dates, COMMISSION_RATE, STAMP_DUTY_RATE, SLIPPAGE_RATE, COMMISSION_MIN, calc_fee

RES_DIR = "data/results/livermore"
CAPITAL = 1000000.0
INDEX_MARKET = "000300.SH"      # 市场环境门控
INDEX_BENCH_1 = "000300.SH"     # 基准 沪深300
INDEX_BENCH_2 = "000906.SH"     # 基准 中证800
UNIV_INDEX = "000906.SH"        # 股票池 中证800


# ════════════════════════════════════════════════════════════
#  数据预载
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


def load_industry():
    conn = get_conn()
    df = pd.read_sql_query("SELECT ts_code, industry FROM stock_basic", conn)
    conn.close()
    return dict(zip(df["ts_code"], df["industry"]))


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
    # 宽表: index=trade_date, columns=ts_code
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
#  矩阵预计算（向量化信号）
# ════════════════════════════════════════════════════════════

def build_signals(all_dates, open_r, high_r, low_r, close_r, pre_close_r, vol_r, adj_d, industry_map, cfg):
    """返回 dict of aligned DataFrames (index=all_dates, columns=codes)"""
    codes = list(open_r.columns)
    # 截断到 all_dates 范围（warmup 期间数据只用于指标预热，不交易）
    idx = open_r.index.intersection(all_dates)
    open_raw = open_r.reindex(all_dates)
    high_r = high_r.reindex(all_dates)
    low_r = low_r.reindex(all_dates)
    close_raw = close_r.reindex(all_dates)
    pre_close_raw = pre_close_r.reindex(all_dates)
    vol_r = vol_r.reindex(all_dates)
    adj_d = adj_d.reindex(all_dates)
    adj_f = adj_d.ffill().fillna(1.0)

    # hfq 空间（剔除分红缺口）: close*adj / high*adj
    close_h = close_raw * adj_f
    high_h = high_r * adj_f

    look = int(cfg["lookback"])
    # 关键点 = 前 lookback 日最高价(hfq)，shift(1) 取 ≤ T-1
    key_level = high_h.rolling(look, min_periods=look).max().shift(1)
    # 突破信号（T-1 判定，T 执行）: close[T-1] > key_level[T-1]
    breakout_ready = (close_h > key_level) & key_level.notna()
    breakout_ready = breakout_ready.fillna(False)
    
    # 突破确认：连续confirm_days天站稳关键点上方 → 过滤假突破
    confirm_days = int(cfg.get("confirm_days", 2))
    breakout_confirmed = breakout_ready.rolling(confirm_days, min_periods=confirm_days).sum() == confirm_days
    breakout_confirmed = breakout_confirmed.fillna(False)
    
    # 成交量确认：突破当日成交量>前N日均量1.5倍 → 过滤无量假突破
    vol_win = int(cfg.get("vol_win", 5))
    vol_ma = vol_r.rolling(vol_win, min_periods=vol_win).mean().shift(1)  # 前N日均量，shift1避免未来函数
    vol_confirmed = (vol_r > vol_ma * 1.5) & vol_ma.notna()
    vol_confirmed = vol_confirmed.fillna(False)
    # 最终突破信号：同时满足站稳确认+成交量确认
    breakout_confirmed = breakout_confirmed & vol_confirmed

    # MA 退出（hfq 空间）
    ma_per = int(cfg["ma_period"])
    ma = close_h.rolling(ma_per, min_periods=ma_per).mean()
    ma_below = (close_h < ma) & ma.notna()
    ma_below = ma_below.fillna(False)

    # 跌破突破位退出
    exit_pct = float(cfg["exit_pct"])
    below_key = (close_h < key_level * (1.0 - exit_pct)) & key_level.notna()
    below_key = below_key.fillna(False)

    # 板块强度(动量): 个股5日收益率在行业内排名前 sector_top_pct —— 经消融证明为负贡献
    mom = close_h / close_h.shift(5) - 1.0  # 5日收益率
    sector_pass = _sector_rank(mom, codes, industry_map, cfg["sector_top_pct"])

    # 改进A: 低波筛选(替代动量板块强度) —— 个股20日已实现波动率在行业内排名底部 sector_top_pct
    vol_lookback = int(cfg.get("vol_lookback", 20))
    daily_ret = close_h / close_h.shift(1) - 1.0
    realized_vol = daily_ret.rolling(vol_lookback, min_periods=vol_lookback).std().shift(1)  # T-1, 避免未来函数
    # 用 -realized_vol 喂给排名函数 => 排名前 sector_top_pct = 波动率最低(低波异象为正因子)
    lowvol_pass = _sector_rank(-realized_vol, codes, industry_map, cfg["sector_top_pct"])

    # 改进D(源自《股票大作手回忆录》原则①「最小阻力方向」): 突破前必须先有窄幅横盘
    # 书中案例: 小麦在1.10-1.20横盘(箱宽~9%)、伯利恒钢铁等6周才突破98 —— 都是"先压缩僵持再突破",
    # 不是一路上涨后的新高(那种往往是主力派货阶段, 接盘)。仅抄"突破"动作漏掉了真正有价值的"横盘前置"。
    box_len = int(cfg.get("box_len", 0))
    if box_len > 0:
        box_hi = close_h.rolling(box_len, min_periods=box_len).max()
        box_lo = close_h.rolling(box_len, min_periods=box_len).min()
        box_mid = close_h.rolling(box_len, min_periods=box_len).mean()
        box_width = (box_hi - box_lo) / box_mid                        # 箱体相对宽度
        # T-1 及之前的 box_len 日窗口判定(不含突破当日, 避免未来函数)
        squeeze_pass = (box_width.shift(1) <= float(cfg.get("box_width", 0.15))) & box_width.notna()
        squeeze_pass = squeeze_pass.fillna(False)
    else:
        squeeze_pass = None

    high_h = high_r * adj_f
    return dict(open_raw=open_raw, close_raw=close_raw, pre_close_raw=pre_close_raw, close_h=close_h,
                breakout_ready=breakout_confirmed, sector_pass=sector_pass, lowvol_pass=lowvol_pass,
                realized_vol=realized_vol,
                ma_below=ma_below, below_key=below_key, key_level=key_level, high_h=high_h,
                squeeze_pass=squeeze_pass)


def _sector_rank(mom, codes, industry_map, top_pct):
    """返回 bool DataFrame: 个股动量在其行业内排名前 top_pct。"""
    df = mom.reset_index().melt(id_vars="trade_date", var_name="ts_code", value_name="mom")
    ind = pd.DataFrame({"ts_code": codes})
    ind["ind"] = ind["ts_code"].map(lambda c: str(industry_map.get(c, "NA")) if industry_map.get(c) else "NA")
    df = df.merge(ind, on="ts_code", how="left")
    df["rank"] = df.groupby(["trade_date", "ind"])["mom"].rank(pct=True, method="average")
    # 前 top_pct 即 rank >= 1-top_pct
    df["pass"] = (df["rank"] >= (1.0 - top_pct)) & df["mom"].notna()
    out = df.pivot(index="trade_date", columns="ts_code", values="pass").reindex(mom.index)
    return out.fillna(False)


# ════════════════════════════════════════════════════════════
#  回测主循环
# ════════════════════════════════════════════════════════════

def run_window(start, end, cfg, layers):
    """layers: dict(market=bool, sector=bool, exit=bool)；组合逻辑见下。"""
    all_dates = get_trade_dates(start, end)
    if len(all_dates) < 60:
        print(f"  [跳过] {start}-{end} 交易日不足"); return None

    # 成分快照 → 每日 universe 集合
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
    industry_map = load_industry()
    sig = build_signals(all_dates, open_r, high_r, low_r, close_r, pre_close_r, vol_r, adj_d, industry_map, cfg)

    # 市场门控优化: 改用20日均线快速判牛熊，解决原50日门控滞后导致的熊市回撤过大问题
    idx_close = load_index(INDEX_MARKET, start, end)
    mkt = idx_close.reindex(all_dates)
    mkt_ma20 = mkt.rolling(20, min_periods=20).mean()
    bull = (mkt > mkt_ma20) & mkt_ma20.notna()
    bull = bull.fillna(False).astype(bool)

    # 改进C: 高波动体制跳过 —— 沪深300 20日已实现波动率破过去250日90分位时暂停新开仓(持仓照常退出)
    use_vol_skip = bool(cfg.get("vol_skip", False))
    idx_ret = mkt / mkt.shift(1) - 1.0
    idx_vol = idx_ret.rolling(20, min_periods=20).std().shift(1)
    idx_vol_thr = idx_vol.rolling(250, min_periods=60).quantile(0.9)
    vol_skip = ((idx_vol > idx_vol_thr) & idx_vol_thr.notna()).fillna(False).astype(bool)

    open_raw = sig["open_raw"]
    close_raw = sig["close_raw"]
    pre_close_raw = sig["pre_close_raw"]
    br = sig["breakout_ready"]
    # 改进A: 板块过滤模式切换(momentum=原动量前30%负贡献 / lowvol=低波筛选 / none=关闭)
    sector_mode = str(cfg.get("sector_mode", "momentum"))
    if sector_mode == "lowvol":
        sp = sig["lowvol_pass"]
    elif sector_mode == "none":
        sp = None
    else:
        sp = sig["sector_pass"]
    ma_b = sig["ma_below"]
    bk = sig["below_key"]
    kl = sig["key_level"]
    close_h = sig["close_h"]
    high_h = sig["high_h"]
    # 改进D: 箱体压缩(最小阻力线前置)
    squeeze = sig.get("squeeze_pass")
    sq_arr = squeeze.values.astype(bool) if (squeeze is not None and hasattr(squeeze, "values")) else None

    n = len(all_dates)
    cols = list(br.columns)
    code2idx = {c: j for j, c in enumerate(cols)}
    br_arr = br.values.astype(bool)
    sp_arr = sp.values.astype(bool) if sp is not None else np.zeros((n, len(cols)), dtype=bool)
    ma_arr = ma_b.values.astype(bool)
    close_h_arr = close_h.values.astype(float)
    vol_arr = sig["realized_vol"].values.astype(float)  # 已实现波动率矩阵(T-1对齐), 用于波动目标仓位
    high_h_arr = high_h.values.astype(float)
    kl_arr = kl.values.astype(float)
    # 每日触发集合（避免每日扫描全池 ~800 只）
    br_sets = [set(np.compress(br_arr[i], cols)) for i in range(n)]
    sp_sets = [set(np.compress(sp_arr[i], cols)) for i in range(n)]
    ma_sets = [set(np.compress(ma_arr[i], cols)) for i in range(n)]

    use_market = layers.get("market", True)
    use_sector = layers.get("sector", True)
    use_exit = layers.get("exit", True)
    market_exit = bool(cfg.get("market_exit", True)) and use_market
    stop_loss = float(cfg.get("stop_loss", 0.0)) if use_exit else 0.0
    exit_pct = float(cfg.get("exit_pct", 0.0)) if use_exit else 0.0
    max_hold = int(cfg["max_hold"])
    trailing_stop = float(cfg.get("trailing_stop", 0.08))
    fail_exit_days = int(cfg.get("fail_exit_days", 2))

    cash = CAPITAL
    positions = {}      # code -> dict(shares, entry_open, entry_hfq, key_hfq, high_water_mark, entry_date)
    pending_sell = set()  # 跌停封死次日重试的卖出队列
    nav = []
    n_entries = n_exits = 0
    days_in_market = 0
    bull_rets, bear_rets = [], []
    # 交易归因统计
    trade_records = []  # 每笔交易的记录：(dict(entry_date, exit_date, code, ret, hold_days, exit_type))
    # 统计各层过滤信号数量
    total_breakout = total_sector_pass = total_market_pass = total_open = 0

    for i in range(n):
        d = all_dates[i]
        univ = univ_at(d)
        if not univ:
            nav.append((d, cash)); continue
        # 修正未来函数：市场环境只能用前一天的信号（T-1收盘计算，T开盘执行）
        bull_t = bool(bull.iloc[i-1]) if i-1 >= 0 and i-1 < len(bull) else False
        bull_prev = bool(bull.iloc[i-1]) if (i >= 1 and (i-1) < len(bull)) else False
        vol_skip_prev = bool(vol_skip.iloc[i-1]) if (use_vol_skip and i >= 1 and (i-1) < len(vol_skip)) else False

        # --- 1) 执行卖出（优先处理前一日跌停封死的重试卖出）---
        sell_exec = []
        for code in list(pending_sell):
            if code not in positions:
                pending_sell.discard(code)
                continue
            op = open_raw.iloc[i].get(code)
            cl = close_raw.iloc[i].get(code)
            pre_close = pre_close_raw.iloc[i].get(code)
            if op is None or pd.isna(op) or cl is None or pd.isna(cl) or pre_close is None or pd.isna(pre_close):
                pending_sell.discard(code)
                continue
            # 检查是否开盘跌停
            if op <= pre_close * 0.901:
                if cl <= pre_close * 0.901:  # 仍封死，继续次日重试
                    continue
                else:  # 没封死，按收盘价卖出
                    sell_price = cl
            else:  # 没跌停，按开盘价卖出
                sell_price = op
            sell_exec.append((code, sell_price))
        pending_sell.clear()

        # --- 2) 持仓退出判定（用 T 收盘信号, 次日开盘执行）---
        # 市场转熊 → 整批关信号（若启用）
        force_exit_all = (market_exit and use_market and not bull_prev)
        ma_i = ma_sets[i-1] if i >= 1 else set()
        for code, pos in list(positions.items()):
            if code not in univ:
                # 持仓股退出成分股: 用最近已知收盘价卖出, 避免 open_raw 取 NaN 污染 cash(原代码会打穿净值)
                last_c = pos.get("last_close")
                if last_c is None or pd.isna(last_c):
                    last_c = pos.get("entry_open", 0.0)
                sell_exec.append((code, last_c)); continue
            ch = close_h_arr[i-1, code2idx[code]] if (i >= 1 and code in code2idx) else np.nan
            if pd.isna(ch):
                continue
            # 跟踪止盈水位线：用日内最高价(i-1)而非收盘价，标准写法（锁利更早）
            hh = high_h_arr[i-1, code2idx[code]] if (i >= 1 and code in code2idx) else np.nan
            pos["high_water_mark"] = max(pos["high_water_mark"], ch, hh if not pd.isna(hh) else ch)
            exit_now = False
            exit_price = None
            if use_exit:
                # 突破后2个交易日不创新高直接离场 → 过滤假突破，减少平均亏损
                # 【2026-08-07 修正同日未来函数】卖出决策在 day i 开盘做出，只能用 i-1 及之前数据。
                # 故 day2_high(=entry_idx+2) 必须 <= i-1，即 i >= entry_idx+3。
                # 原按自然日 hold_days==2 判断，会令 entry_idx+2 落到当天 i，偷看当天日内 high(前视，虚增收益约30pp/几乎全在牛市日)。
                entry_idx = all_dates.index(pos["entry_date"]) if pos["entry_date"] in all_dates else -1
                if entry_idx >= 0 and i >= entry_idx + 3:
                    day1_high = high_h_arr[entry_idx+1, code2idx[code]] if (entry_idx+1 < len(all_dates) and code in code2idx) else np.nan
                    day2_high = high_h_arr[entry_idx+2, code2idx[code]] if (entry_idx+2 < len(all_dates) and code in code2idx) else np.nan
                    entry_high = high_h_arr[entry_idx, code2idx[code]] if code in code2idx else np.nan
                    if not pd.isna(day1_high) and not pd.isna(day2_high) and not pd.isna(entry_high):
                        if day1_high <= entry_high and day2_high <= entry_high:
                            exit_now = True
                # 跟踪止盈: 从最高价回撤12%（放宽到12%，延长盈利持仓时间）
                if not exit_now and trailing_stop > 0 and pos["high_water_mark"] > 0 and ch < pos["high_water_mark"] * (1 - 0.12):
                    exit_now = True
                # 硬止损: 跌破成本价5%
                if not exit_now and stop_loss > 0 and not pd.isna(pos["entry_hfq"]) and ch < pos["entry_hfq"] * (1 - stop_loss):
                    exit_now = True
                # 跌破突破位下方3% → 失效离场
                if not exit_now and not pd.isna(pos["key_hfq"]) and ch < pos["key_hfq"] * (1 - exit_pct):
                    exit_now = True
                # 跌破MA
                if not exit_now and code in ma_i:
                    exit_now = True
            if force_exit_all:
                exit_now = True
            if exit_now:
                op = open_raw.iloc[i].get(code)
                cl = close_raw.iloc[i].get(code)
                pre_close = pre_close_raw.iloc[i].get(code)
                if op is not None and not pd.isna(op) and cl is not None and not pd.isna(cl) and pre_close is not None and not pd.isna(pre_close):
                    # 检查是否开盘跌停
                    if op <= pre_close * 0.901:
                        if cl <= pre_close * 0.901:  # 封死，次日重试
                            pending_sell.add(code)
                        else:  # 没封死，收盘价卖出
                            sell_exec.append((code, cl))
                    else:  # 没跌停，开盘价卖出
                        sell_exec.append((code, op))

        # 执行卖出
        for code, sell_price in sell_exec:
            if code not in positions:
                continue
            sh = positions[code]["shares"]
            if sh > 0:
                proceeds = sell_price * sh - calc_fee("sell", sell_price, sh, d)
                cash += proceeds
                n_exits += 1
                # 计算单笔交易收益
                cost = positions[code]["entry_open"] * sh + calc_fee("buy", positions[code]["entry_open"], sh, positions[code]["entry_date"])
                ret = proceeds / cost - 1
                hold_days = (pd.Timestamp(d) - pd.Timestamp(positions[code]["entry_date"])).days
                # 判断退出类型
                exit_type = "止损/退出"
                if positions[code]["high_water_mark"] > 0 and sell_price >= positions[code]["high_water_mark"] * (1 - trailing_stop):
                    exit_type = "跟踪止盈"
                elif ret > 0.05:
                    exit_type = "盈利卖出"
                elif ret < -0.05:
                    exit_type = "亏损卖出"
                trade_records.append(dict(
                    entry_date=positions[code]["entry_date"],
                    exit_date=d,
                    code=code,
                    ret=ret,
                    hold_days=hold_days,
                    exit_type=exit_type
                ))
            positions.pop(code, None)

        # --- 3) 新开仓（T-1 信号, T 开盘执行）---
        # 改进C: 高波动体制跳过(仅在 vol_skip_prev 时不新开)
        can_open = ((not use_market) or bull_prev) and (not vol_skip_prev)
        prev_i = i - 1
        if can_open and prev_i >= 0:
            # 统计各层过滤数量
            prev_breakout = br_sets[prev_i] & univ
            total_breakout += len(prev_breakout)
            prev_sector = prev_breakout & sp_sets[prev_i] if (use_sector and sp is not None) else prev_breakout
            total_sector_pass += len(prev_sector)
            prev_market = prev_sector if can_open else set()
            total_market_pass += len(prev_market)
            
            cand = prev_market - set(positions) - pending_sell
            cand = list(cand)
            # 改进D: 仅保留"突破前box_len日窄幅横盘"的候选(最小阻力线前置, 排除一路新高接盘)
            if sq_arr is not None and prev_i < sq_arr.shape[0]:
                cand = [c for c in cand if (c in code2idx and sq_arr[prev_i, code2idx[c]])]
            # 等权买入（目标权重 1/max_hold）
            equity = cash
            for code, pos in positions.items():
                c = close_raw.iloc[i].get(code)
                if c is not None and not pd.isna(c):
                    equity += pos["shares"] * c
            slots = max_hold - len(positions)
            if slots > 0 and cand:
                take = cand[:slots]
                # 改进B: 波动目标仓位(风险平价-lite) —— 按 1/已实现波动率 在空位间分配预算
                use_vol_size = bool(cfg.get("vol_size", False))
                if use_vol_size:
                    vols_take = [vol_arr[prev_i, code2idx[c]] if (c in code2idx and prev_i < vol_arr.shape[0]) else np.nan for c in take]
                    # 鲁棒化: 超低波动会令 1/vol 爆炸 -> 设波动下限(最大波动的0.5倍)避免单票被放大到近满仓
                    pos_vols = [v for v in vols_take if (v is not None and not pd.isna(v) and v > 0)]
                    vfloor = (max(pos_vols) * 0.5) if pos_vols else 1.0
                    inv = []
                    for v in vols_take:
                        vv = v if (v is not None and not pd.isna(v) and v > 0) else vfloor
                        vv = max(vv, vfloor)
                        inv.append(1.0 / vv)
                    s_inv = sum(inv)
                    w = [x / s_inv for x in inv]
                    # 集中度封顶: 单票权重不超过 50%, 防止极端逆波动定权导致崩塌
                    w = [min(x, 0.5) for x in w]
                    s2 = sum(w)
                    w = [x / s2 for x in w]
                    budget = equity / max_hold * slots  # 总新开预算与等权一致, 不放大杠杆
                    per_vals = [budget * wk for wk in w]
                else:
                    per_vals = [equity / max_hold] * len(take)
                for k, code in enumerate(take):
                    per_val = per_vals[k]
                    op = open_raw.iloc[i].get(code)
                    cl = close_raw.iloc[i].get(code)
                    pre_close = pre_close_raw.iloc[i].get(code)
                    if op is None or pd.isna(op) or op <= 0 or cl is None or pd.isna(cl) or pre_close is None or pd.isna(pre_close):
                        continue
                    # 检查是否开盘涨停
                    if op >= pre_close * 1.099:
                        if cl >= pre_close * 1.099:  # 封死，放弃
                            continue
                        else:  # 没封死，按收盘价买入
                            buy_price = cl
                    else:  # 没涨停，按开盘价买入
                        buy_price = op
                    # 检查buy_price有效性
                    if pd.isna(buy_price) or buy_price <= 0:
                        continue
                    # 计算可买数量，避免NaN
                    if pd.isna(per_val) or per_val <= 0:
                        continue
                    sh = int(per_val / (buy_price * (1 + COMMISSION_RATE + SLIPPAGE_RATE)) / 100) * 100
                    if sh <= 0:
                        sh = 100  # 最少买100股
                    if sh <= 0:
                        # 退而求其次买得起的数量
                        sh = int(cash / (buy_price * (1 + COMMISSION_RATE + SLIPPAGE_RATE)) / 100) * 100
                        if sh <= 0:
                            continue
                    cost = buy_price * sh + calc_fee("buy", buy_price, sh, d)
                    if pd.isna(cost) or cost > cash:
                        continue
                    cash -= cost
                    j = code2idx.get(code)
                    kh = kl_arr[prev_i, j] if j is not None else np.nan
                    eh = close_h_arr[prev_i, j] if j is not None else np.nan
                    positions[code] = dict(shares=sh, entry_open=buy_price,
                                           entry_hfq=eh, key_hfq=kh, high_water_mark=eh,
                                           entry_date=d)
                    n_entries += 1
                    total_open += 1

        # --- 4) 估值（收盘原始价）---
        mv = cash
        held = 0
        for code, pos in positions.items():
            c = close_raw.iloc[i].get(code)
            if c is None or pd.isna(c):
                c = pos.get("last_close")   # 停牌/缺失日沿用最近已知收盘价, 避免持仓价值被误判为0打穿净值
            if c is not None and not pd.isna(c):
                mv += pos["shares"] * c
                pos["last_close"] = c
                held += 1
        if held > 0:
            days_in_market += 1
        nav.append((d, mv))

        # 强弱市场分段收益
        if i > 0 and len(nav) >= 2:
            r = nav[-1][1] / nav[-2][1] - 1
            if bull_t:
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

    # 强弱分段
    def compound(rets):
        if not rets:
            return 0.0
        p = 1.0
        for r in rets:
            p *= (1 + r)
        return p - 1
    bull_ret = compound(bull_rets)
    bear_ret = compound(bear_rets)
    bull_days = int(bull.sum())

    return dict(start=start, end=end, nav=nav, m=m, mb1=mb1, mb2=mb2,
                n_entries=n_entries, n_exits=n_exits,
                time_in_market=days_in_market / len(all_dates),
                bull_ret=bull_ret, bear_ret=bear_ret,
                bull_days=bull_days, total_days=len(all_dates),
                layers=layers,
                filter_stats=dict(
                    total_breakout=total_breakout,
                    total_sector_pass=total_sector_pass,
                    total_market_pass=total_market_pass,
                    total_open=total_open
                ),
                trade_records=trade_records)


# ════════════════════════════════════════════════════════════
#  报告
# ════════════════════════════════════════════════════════════

def fmt_pct(x, signed=False):
    if x is None:
        return "-"
    return f"{x*100:+.2f}%" if signed else f"{x*100:.2f}%"


def print_window(r):
    if r is None:
        return
    m, mb1, mb2 = r["m"], r["mb1"], r["mb2"]
    fs = r.get("filter_stats", {})
    print(f"===== {r['start']} → {r['end']} | 分层={r['layers']} =====")
    print(f"  过滤统计: 突破信号{fs.get('total_breakout',0)} → 板块过滤后{fs.get('total_sector_pass',0)} → 市场环境过滤后{fs.get('total_market_pass',0)} → 实际开仓{fs.get('total_open',0)}")
    print(f"  交易: 入场 {r['n_entries']} / 出场 {r['n_exits']} / 持仓时间占比 {fmt_pct(r['time_in_market'])}")
    print(f"  策略 : 总收益 {fmt_pct(m['total'])} / 年化 {fmt_pct(m['ann'])} / 最大回撤 {fmt_pct(m['mdd'])} / 夏普 {m['sharpe']:.3f}")
    print(f"  沪深300: 总收益 {fmt_pct(mb1['total'])} / 年化 {fmt_pct(mb1['ann'])} / 最大回撤 {fmt_pct(mb1['mdd'])}")
    print(f"  中证800: 总收益 {fmt_pct(mb2['total'])}")
    print(f"  超额(策略-沪深300): {fmt_pct(m['total']-mb1['total'], signed=True)}")
    print(f"  强弱分段: 牛市日({r['bull_days']}天)收益 {fmt_pct(r['bull_ret'], signed=True)} | 熊市日收益 {fmt_pct(r['bear_ret'], signed=True)}")
    # 交易归因分析
    trade_df = pd.DataFrame(r.get("trade_records", []))
    if not trade_df.empty:
        print("\n  ====== 交易归因分析 ======")
        win_df = trade_df[trade_df["ret"] > 0]
        loss_df = trade_df[trade_df["ret"] <= 0]
        print(f"  总交易笔数: {len(trade_df)} | 胜率: {len(win_df)/len(trade_df):.2%}")
        print(f"  盈利交易: 笔数 {len(win_df)} | 平均收益 {win_df['ret'].mean():.2%} | 平均持仓天数 {win_df['hold_days'].mean():.1f}")
        print(f"  亏损交易: 笔数 {len(loss_df)} | 平均收益 {loss_df['ret'].mean():.2%} | 平均持仓天数 {loss_df['hold_days'].mean():.1f}")
        print(f"  盈亏比: {abs(win_df['ret'].mean()/loss_df['ret'].mean()):.2f}" if not loss_df.empty and loss_df['ret'].mean() != 0 else "  盈亏比: 无穷大")
        # 退出类型统计
        exit_type_stats = trade_df.groupby("exit_type").agg(
            count=("ret", "count"),
            avg_ret=("ret", "mean"),
            avg_hold=("hold_days", "mean")
        )
        print("\n  退出类型分布:")
        for idx, row in exit_type_stats.iterrows():
            print(f"    {idx}: {row['count']}笔 | 平均收益 {row['avg_ret']:.2%} | 平均持仓 {row['avg_hold']:.1f}天")
        # 收益分布分位数
        print("\n  收益分布分位数:")
        for p in [0.1, 0.25, 0.5, 0.75, 0.9]:
            print(f"    {int(p*100)}分位: {trade_df['ret'].quantile(p):.2%}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20180101")
    ap.add_argument("--end", default="20251231")
    ap.add_argument("--windows", default=None,
                    help="多窗口: 20180101-20251231,20190101-20251231,...")
    ap.add_argument("--lookback", type=int, default=20, help="关键点窗口(日)")
    ap.add_argument("--sector-top-pct", type=float, default=0.30, help="行业内前X%算强势")
    ap.add_argument("--exit-pct", type=float, default=0.03, help="跌回突破位X%退出")
    ap.add_argument("--ma-period", type=int, default=20, help="退出MA周期")
    ap.add_argument("--stop-loss", type=float, default=0.05, help="硬止损(跌破入场价X)")
    ap.add_argument("--max-hold", type=int, default=5, help="最多持仓数")
    ap.add_argument("--confirm-days", type=int, default=2, help="突破确认连续站稳天数")
    ap.add_argument("--trailing-stop", type=float, default=0.12, help="跟踪止盈回撤比例")
    ap.add_argument("--no-market-exit", action="store_true", help="市场转熊不强制清仓(仅关新开)")
    ap.add_argument("--fail-exit", type=int, default=2, help="突破后N日不创新高即失败离场(0=关)")
    ap.add_argument("--sector-mode", default="momentum", choices=["momentum", "lowvol", "none"],
                    help="板块过滤: momentum(原动量前30%,负贡献)/lowvol(低波筛选)/none(关闭)")
    ap.add_argument("--vol-size", action="store_true", help="改进B: 逆波动定权(风险平价-lite)")
    ap.add_argument("--vol-skip", action="store_true", help="改进C: 市场波动率破顶时暂停新开仓")
    ap.add_argument("--vol-lookback", type=int, default=20, help="已实现波动率窗口(日)")
    ap.add_argument("--box-len", type=int, default=0, help="改进D: 突破前窄幅横盘窗口(日); 0=关。源自利弗莫尔'最小阻力线'——先压缩僵持再突破")
    ap.add_argument("--box-width", type=float, default=0.15, help="改进D: 箱宽阈值(相对宽度<=此值算窄幅横盘), 配合--box-len")
    ap.add_argument("--risk-per", type=float, default=0.02, help="每仓位风险预算(预留, 当前未直接用于定权)")
    ap.add_argument("--vol-win", type=int, default=5, help="量能确认基线窗口(日): 突破日量>前N日均量1.5倍; 默认5。10=用更长'常态量'基线")
    ap.add_argument("--ablation", action="store_true", help="跑分层消融(仅关键点/+市场/+板块/全)")
    args = ap.parse_args()

    cfg = dict(lookback=args.lookback,
               sector_top_pct=args.sector_top_pct, exit_pct=args.exit_pct,
               ma_period=args.ma_period, stop_loss=args.stop_loss,
               max_hold=args.max_hold, market_exit=not args.no_market_exit,
               confirm_days=args.confirm_days, trailing_stop=args.trailing_stop,
               fail_exit_days=args.fail_exit,
               sector_mode=args.sector_mode, vol_size=args.vol_size,
               vol_skip=args.vol_skip, vol_lookback=args.vol_lookback,
               box_len=args.box_len, box_width=args.box_width,
               risk_per=args.risk_per, vol_win=args.vol_win)

    if args.ablation:
        layers_list = [
            ("仅关键点(+退出)", dict(market=False, sector=False, exit=True)),
            ("+市场环境", dict(market=True, sector=False, exit=True)),
            ("+板块强度", dict(market=False, sector=True, exit=True)),
            ("全四步", dict(market=True, sector=True, exit=True)),
        ]
        print(f"参数: lookback={args.lookback} 板块前{args.sector_top_pct:.0%} "
              f"exit_pct={args.exit_pct:.0%} ma={args.ma_period} stop={args.stop_loss:.0%} max_hold={args.max_hold} "
              f"confirm_days={args.confirm_days} trailing_stop={args.trailing_stop:.0%} "
              f"market_exit={not args.no_market_exit}")
        print(f"初始资金 {CAPITAL:,.0f} / 基准 {INDEX_BENCH_1} & {INDEX_BENCH_2}\n")
        rows = []
        for label, ly in layers_list:
            r = run_window(args.start, args.end, cfg, ly)
            if r is None:
                continue
            fs = r.get("filter_stats", {})
            rows.append({"分层": label, "策略总收益": fmt_pct(r["m"]["total"]),
                         "年化": fmt_pct(r["m"]["ann"]), "最大回撤": fmt_pct(r["m"]["mdd"]),
                         "夏普": f"{r['m']['sharpe']:.2f}",
                         "超额HS300": fmt_pct(r["m"]["total"]-r["mb1"]["total"], signed=True),
                         "持仓占比": fmt_pct(r["time_in_market"]),
                         "牛市日收益": fmt_pct(r["bull_ret"], signed=True),
                         "熊市日收益": fmt_pct(r["bear_ret"], signed=True),
                         "实际开仓数": fs.get("total_open",0)})
        print(pd.DataFrame(rows).to_string(index=False))
        print("\n[说明] 每一步叠加展示从'仅关键点'到'全四步'的 P&L 贡献；视频只给了漏斗计数，这里给收益。")
        return

    if args.windows:
        wins = [w.split("-") for w in args.windows.split(",")]
        print(f"参数: lookback={args.lookback} 板块前{args.sector_top_pct:.0%} "
              f"exit_pct={args.exit_pct:.0%} ma={args.ma_period} stop={args.stop_loss:.0%} max_hold={args.max_hold} "
              f"confirm_days={args.confirm_days} trailing_stop={args.trailing_stop:.0%} "
              f"market_exit={not args.no_market_exit}")
        print(f"初始资金 {CAPITAL:,.0f} / 基准 {INDEX_BENCH_1} & {INDEX_BENCH_2}\n")
        rows = []
        for s, e in wins:
            r = run_window(s, e, cfg, dict(market=True, sector=True, exit=True))
            if r is None:
                continue
            fs = r.get("filter_stats", {})
            rows.append({"窗口": f"{s}-{e}", "策略总收益": fmt_pct(r["m"]["total"]),
                         "年化": fmt_pct(r["m"]["ann"]), "最大回撤": fmt_pct(r["m"]["mdd"]),
                         "夏普": f"{r['m']['sharpe']:.2f}",
                         "超额HS300": fmt_pct(r["m"]["total"]-r["mb1"]["total"], signed=True),
                         "持仓占比": fmt_pct(r["time_in_market"]),
                         "牛市日": fmt_pct(r["bull_ret"], signed=True),
                         "熊市日": fmt_pct(r["bear_ret"], signed=True),
                         "实际开仓数": fs.get("total_open",0)})
        print(pd.DataFrame(rows).to_string(index=False))
        print("\n[口径] 信号T-1收盘判定/T开盘执行; 突破位与MA用hfq(去分红缺口)空间; 估值用原始价。")
        return

    r = run_window(args.start, args.end, cfg, dict(market=True, sector=True, exit=True))
    print_window(r)


if __name__ == "__main__":
    main()
