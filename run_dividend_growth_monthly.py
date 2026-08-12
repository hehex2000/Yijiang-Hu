# -*- coding: utf-8 -*-
"""
高股息 + 基本面成长 双因子月度调仓策略 —— B站视频复刻验证版
============================================================
视频: "不看K线、不懂MACD，这个笨策略凭什么跑赢基准16个百分点？"
      (UP主: 上班族量化的鳄鱼, BV1CzMf6FENh)

策略规则（照视频原话实现）:
  筛1 股息率排名 : 全市场 三年总分红÷市值 → 股息率 取前 10%
       └ 数据口径: daily_basic.dv_ttm (TTM股息率, 全市场可用; 视频为3年均值, 代理并标注)
  筛2 基本面五关 : PE∈(0,20] | PEG∈[0.08,2] | ROE>3% | 营收同比>5% | 净利同比>11%
       └ 数据: pe_ttm(选股日T-1) + fina_indicator最新报告(ann_date≤T-1, 防未来函数)
  筛3 交易层     : 排除停牌 / 排除昨日涨停 → 取前 10 名等权持有
  执行           : 每月第5交易日调仓(T-1选股, T开盘执行) + 日规则:
                   持仓昨涨停今未封 → 当日收盘卖出

估值双轨:
  value_raw : 原始价 NAV（不含分红, 偏低）
  value_hfq : 后复权 NAV = Σ shares×close×adj(t)/adj(买入)（含分红再投, 主口径）

基准: 000300.SH 沪深300（视频对照口径）; 932000.SH 中证红利（同赛道参照）
"""
import sys
import os
import re
import argparse
import pandas as pd
import numpy as np

import config
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest"))
from atr_stop_loss import ATRStopLoss
from run_monthly_rebalance import (
    get_conn, get_trade_dates, get_monthly_5th_trading_days, calc_fee,
    get_open_price, COMMISSION_RATE, SLIPPAGE_RATE,
)

RES_DIR = "data/results/dividend_growth"
CAPITAL = 200000.0
INDEX_PRIMARY = "000300.SH"
INDEX_DIV = "932000.SH"
A_SHARE_RE = re.compile(r"^(60|00|30|68)\d{4}\.(SH|SZ)$")


def limit_pct(code: str, date_int: int) -> float:
    """各板块涨停阈值（%）。ST 已被剔除。"""
    if code.startswith("688"):
        return 19.9
    if code.startswith("300"):
        return 19.9 if date_int >= 20200824 else 9.9
    return 9.9


# ════════════════════════════════════════════════════════════
#  数据预载（批量, 一次查询）
# ════════════════════════════════════════════════════════════

def load_stock_basic():
    conn = get_conn()
    df = pd.read_sql_query("SELECT ts_code, name FROM stock_basic", conn)
    conn.close()
    return df


def load_daily_basic_sel(sel_dates):
    conn = get_conn()
    ph = ",".join("?" for _ in sel_dates)
    df = pd.read_sql_query(
        f"SELECT ts_code, trade_date, dv_ttm, pe_ttm, total_mv "
        f"FROM daily_basic WHERE trade_date IN ({ph})",
        conn, params=sel_dates)
    conn.close()
    return df


def load_daily_pct_sel(sel_dates):
    """选股日(T-1)全市场涨跌幅 → 昨涨停过滤"""
    conn = get_conn()
    ph = ",".join("?" for _ in sel_dates)
    df = pd.read_sql_query(
        f"SELECT ts_code, trade_date, close, pre_close FROM daily "
        f"WHERE trade_date IN ({ph})",
        conn, params=sel_dates)
    conn.close()
    df["pct"] = (df["close"] - df["pre_close"]) / df["pre_close"] * 100
    return df


def load_fina(end):
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT ts_code, end_date, ann_date, roe, netprofit_yoy, tr_yoy "
        "FROM fina_indicator WHERE ann_date IS NOT NULL AND ann_date != '' "
        "AND ann_date <= ?",
        conn, params=(str(end),))
    conn.close()
    df["ann_date"] = df["ann_date"].apply(lambda x: str(int(float(x))))
    return df


def load_pick_daily(codes, start, end, warmup_days=0):
    """持仓标的行情。warmup_days>0 时查询起点前移（供 ATR 计算 warmup）。"""
    conn = get_conn()
    ph = ",".join("?" for _ in codes)
    q_start = start
    if warmup_days > 0:
        q_start = (pd.Timestamp(start) - pd.Timedelta(days=warmup_days)).strftime("%Y%m%d")
    df = pd.read_sql_query(
        f"SELECT ts_code, trade_date, high, low, close, pre_close FROM daily "
        f"WHERE ts_code IN ({ph}) AND trade_date BETWEEN ? AND ? ORDER BY trade_date",
        conn, params=(*codes, q_start, end))
    conn.close()
    df["pct"] = (df["close"] - df["pre_close"]) / df["pre_close"] * 100
    return df


def load_adj(codes, start, end):
    conn = get_conn()
    ph = ",".join("?" for _ in codes)
    df = pd.read_sql_query(
        f"SELECT ts_code, trade_date, adj_factor FROM adj_factor "
        f"WHERE ts_code IN ({ph}) AND trade_date BETWEEN ? AND ? ORDER BY trade_date",
        conn, params=(*codes, start, end))
    conn.close()
    return df


def load_index(index_code, start, end):
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT trade_date, close FROM index_daily WHERE ts_code=? "
        "AND trade_date BETWEEN ? AND ? ORDER BY trade_date",
        conn, params=(index_code, start, end))
    conn.close()
    return dict(zip(df["trade_date"].astype(str), df["close"].astype(float)))


# ════════════════════════════════════════════════════════════
#  选股（三筛）
# ════════════════════════════════════════════════════════════

def select_picks(rebal_dates, all_dates, cfg, basic_map, pct_map, fina):
    """返回 [(rebal_date, [ts_code...]), ...] + 选股明细日志"""
    st_map = dict(zip(fina["ts_code"] + "@" + fina["ann_date"], fina[["roe", "netprofit_yoy", "tr_yoy"]].values))
    # 按 ann_date 排序后逐期推进 latest
    fina_sorted = fina.sort_values("ann_date")
    latest = {}          # ts_code -> (roe, np_yoy, tr_yoy)
    picks = []
    log = []
    prev_ann = ""
    for rb in rebal_dates:
        rb_idx = all_dates.index(rb)
        sel_date = all_dates[max(0, rb_idx - 1)]   # T-1
        ann_cut = str(int(sel_date))
        # 推进最新财报（ann_date <= T-1）
        sub = fina_sorted[(fina_sorted["ann_date"] > prev_ann) & (fina_sorted["ann_date"] <= ann_cut)]
        for _, r in sub.iterrows():
            latest[r["ts_code"]] = (r["roe"], r["netprofit_yoy"], r["tr_yoy"])
        prev_ann = ann_cut

        b = basic_map[sel_date]
        if b is None or len(b) == 0:
            picks.append((rb, [])); continue
        b = b[(b["dv_ttm"] > 0) & (b["pe_ttm"] > 0) & (b["total_mv"] > 0)].copy()
        b = b[b["ts_code"].str.match(A_SHARE_RE)]
        if len(b) == 0:
            picks.append((rb, [])); continue

        # ── 筛1：股息率前10% ──
        n_top = max(int(round(len(b) * cfg["top_pct"])), 1)
        b = b.sort_values("dv_ttm", ascending=False).head(n_top).copy()

        # ── 筛2：基本面五关（最新财报, ann_date<=T-1）──
        f = b["ts_code"].map(latest)
        roe = f.map(lambda x: x[0] if isinstance(x, tuple) else np.nan)
        np_yoy = f.map(lambda x: x[1] if isinstance(x, tuple) else np.nan)
        tr_yoy = f.map(lambda x: x[2] if isinstance(x, tuple) else np.nan)
        pe = b["pe_ttm"]
        peg = pe / np_yoy
        m = (
            (pe > 0) & (pe <= cfg["pe_max"]) &
            (peg >= cfg["peg_min"]) & (peg <= cfg["peg_max"]) &
            (roe > cfg["roe_min"]) &
            (tr_yoy > cfg["rev_min"]) &
            (np_yoy > cfg["np_min"])
        )
        b = b[m]
        peg_by_code = dict(zip(b["ts_code"], peg[m]))

        # ── 筛3：交易层（T-1 停牌 / 昨涨停）──
        pct_row = pct_map.get(sel_date, {})
        keep = []
        for _, r in b.iterrows():
            code = r["ts_code"]
            p = pct_row.get(code)
            if p is None:          # T-1 无行情 → 停牌
                continue
            if p >= limit_pct(code, int(sel_date)) - 0.1:   # 昨涨停
                continue
            keep.append(code)
        if not keep:
            picks.append((rb, [])); continue

        # 取前 top_n（按股息率降序）
        top = b[b["ts_code"].isin(keep)].sort_values("dv_ttm", ascending=False).head(cfg["top_n"])
        codes = top["ts_code"].tolist()
        picks.append((rb, codes))
        for _, r in top.iterrows():
            f_ = latest.get(r["ts_code"])
            pv = peg_by_code.get(r["ts_code"], np.nan)
            log.append((rb, sel_date, r["ts_code"], r["dv_ttm"], r["pe_ttm"],
                        round(float(pv), 3) if pd.notna(pv) else None,
                        round(float(f_[0]), 2) if f_ and pd.notna(f_[0]) else None,
                        round(float(f_[1]), 1) if f_ and pd.notna(f_[1]) else None,
                        round(float(f_[2]), 1) if f_ and pd.notna(f_[2]) else None))
    return picks, log


# ════════════════════════════════════════════════════════════
#  回测（等权 + 日规则 + 双轨估值）
# ════════════════════════════════════════════════════════════

def _ffill(dmap, code, date, all_dates, idx):
    d = all_dates[idx]
    v = dmap.get(code, {}).get(d)
    if v is not None:
        return v
    j = idx - 1
    while j >= 0:
        v = dmap.get(code, {}).get(all_dates[j])
        if v is not None:
            return v
        j -= 1
    return None


def run_window(start, end, cfg):
    all_dates = get_trade_dates(start, end)
    if len(all_dates) < 30:
        print(f"  [跳过] {start}-{end} 交易日不足"); return None
    rebal_all = [d for d in get_monthly_5th_trading_days(all_dates) if start <= d <= end]
    if not rebal_all:
        print(f"  [跳过] {start}-{end} 无调仓日"); return None

    # 选股（T-1 数据）
    sel_dates = []
    for rb in rebal_all:
        rb_idx = all_dates.index(rb)
        sel_dates.append(all_dates[max(0, rb_idx - 1)])

    basic_df = load_daily_basic_sel(sel_dates)
    pct_df = load_daily_pct_sel(sel_dates)
    basic_map = {d: g for d, g in basic_df.groupby("trade_date")}
    pct_map = {d: dict(zip(g["ts_code"], g["pct"])) for d, g in pct_df.groupby("trade_date")}
    # daily_basic 个别交易日整日缺失 → 回退到最近可用日（≤ T-1，无未来函数）
    import bisect
    basic_dates = sorted(basic_map.keys())
    pct_dates = sorted(pct_map.keys())
    for i, sd in enumerate(sel_dates):
        bi = bisect.bisect_right(basic_dates, sd)
        if bi > 0 and basic_dates[bi - 1] != sd:
            basic_map[sd] = basic_map[basic_dates[bi - 1]]
        if sd not in pct_map:
            pi = bisect.bisect_right(pct_dates, sd)
            if pi > 0:
                pct_map[sd] = pct_map[pct_dates[pi - 1]]
    fina = load_fina(end)
    picks, log = select_picks(rebal_all, all_dates, cfg, basic_map, pct_map, fina)

    codes = sorted(set(c for _, cs in picks for c in cs))
    if not codes:
        print(f"  [跳过] {start}-{end} 全程无选股"); return None

    stop_loss = float(cfg.get("stop_loss", 0.0))   # 0=无止损; 0.15=跌破买入价15%卖出
    atr_stop = float(cfg.get("atr_stop", 0.0))     # 0=关闭; >0=ATR动态止损(倍数, period14)
    atr_period = int(cfg.get("atr_period", 14))

    # 持仓标的行情/复权因子（warmup 400 自然日供 ATR 计算）
    daily_df = load_pick_daily(codes, start, end, warmup_days=400)
    close_map, pct_map2, high_map, atr_map = {}, {}, {}, {}
    for c in codes:
        close_map[c] = {}
        pct_map2[c] = {}
        high_map[c] = {}
        atr_map[c] = {}
    for _, r in daily_df.iterrows():
        ck = str(r["ts_code"])
        close_map[ck][str(r["trade_date"])] = float(r["close"])
        pct_map2[ck][str(r["trade_date"])] = float(r["pct"])
        if atr_stop > 0:
            high_map[ck][str(r["trade_date"])] = float(r["high"])
    if atr_stop > 0:
        for c in codes:
            sub = daily_df[daily_df["ts_code"] == c].sort_values("trade_date")
            if len(sub) < 3:
                continue
            h = sub["high"].astype(float).to_numpy()
            l = sub["low"].astype(float).to_numpy()
            cl = sub["close"].astype(float).to_numpy()
            tmp = ATRStopLoss(atr_period=atr_period)
            arr = tmp.calc_atr(h, l, cl)
            ds = sub["trade_date"].astype(str).tolist()
            for d_, a in zip(ds, arr):
                if a > 0:
                    atr_map[c][d_] = float(a)
    adj_df = load_adj(codes, start, end)
    adj_map = {c: {} for c in codes}
    for _, r in adj_df.iterrows():
        adj_map[str(r["ts_code"])][str(r["trade_date"])] = float(r["adj_factor"])

    # 基准
    b300 = load_index(INDEX_PRIMARY, start, end)
    b932 = load_index(INDEX_DIV, start, end)

    # ── 主循环 ──
    cash = CAPITAL
    positions = {}     # code -> shares
    buy_adj = {}       # code -> adj_factor(买入日)
    entry_price = {}   # code -> 首次买入价（硬止损基准, setdefault 语义）
    sl_dict = {}       # code -> ATRStopLoss 实例（ATR 动态止损模式）
    pending_stops = {}     # code -> True: 当日收盘触发止损, 次日成交价(get_open_price)执行卖出
    recently_stopped = set()  # 当日止损减持, 避免同一次调仓日立刻回补
    nav_raw, nav_hfq = [], []
    n_rebal_sells = n_daily_sells = n_stop_sells = 0
    rebal_dict = dict(picks)

    for idx, d in enumerate(all_dates):
        d_int = int(d)
        # 0.5) 执行上一交易日触发的止损: 以"当日成交价"(2026-07-06前=开盘价, 后=收盘价)
        #      卖出。信号来自 T 日收盘, 成交落在 T+1 日, 与引擎约定一致, 杜绝同日开盘进/收盘出
        if (stop_loss > 0 or atr_stop > 0) and pending_stops:
            for code in list(pending_stops.keys()):
                px = get_open_price(code, d)
                if px:
                    sh = positions.get(code)
                    if sh:
                        proceeds = px * sh - calc_fee("sell", px, sh)
                        cash += proceeds
                        del positions[code]; buy_adj.pop(code, None)
                        entry_price.pop(code, None); sl_dict.pop(code, None)
                        n_stop_sells += 1
                        recently_stopped.add(code)
                pending_stops.pop(code, None)
        # 1) 调仓日: 开盘价执行等权再平衡
        if d in rebal_dict:
            target = [c for c in rebal_dict[d] if c not in recently_stopped]
            def epx(code):
                return get_open_price(code, d)
            mv = cash
            for code, sh in positions.items():
                px = epx(code)
                if px:
                    mv += sh * px
            n_tgt = max(len(target), 1)
            per = mv / n_tgt
            # ⚠️ set 迭代顺序受 PYTHONHASHSEED 影响 → 必须 sorted() 保证可复现
            for code in sorted(set(positions) | set(target)):
                px = epx(code)
                if px is None:
                    continue
                cur = positions.get(code, 0)
                desired = int(per // (px * (1 + COMMISSION_RATE + SLIPPAGE_RATE))) if code in target else 0
                diff = desired - cur
                if diff > 0:
                    cost = px * diff + calc_fee("buy", px, diff)
                    if cost <= cash:
                        cash -= cost
                        positions[code] = cur + diff
                        if code not in buy_adj:
                            a = _ffill(adj_map, code, d, all_dates, idx)
                            buy_adj[code] = a if a else 1.0
                        # 止损状态：硬止损记录首次买入价；ATR 初始防线用 T-1 ATR
                        if stop_loss > 0:
                            entry_price.setdefault(code, px)
                        if atr_stop > 0 and code not in sl_dict:
                            atr_t1 = atr_map.get(code, {}).get(all_dates[max(0, idx - 1)])
                            if atr_t1 and atr_t1 > 0:
                                sl_dict[code] = ATRStopLoss(atr_period=atr_period,
                                                            atr_mult=atr_stop, trail_mult=atr_stop)
                                sl_dict[code].on_entry(px, atr_t1)
                elif diff < 0:
                    sell = -diff
                    proceeds = px * sell - calc_fee("sell", px, sell)
                    cash += proceeds
                    positions[code] = cur - sell
                    n_rebal_sells += 1
                    if positions[code] == 0:
                        del positions[code]; buy_adj.pop(code, None)
                        entry_price.pop(code, None); sl_dict.pop(code, None)
            recently_stopped.clear()   # 止损冷却仅覆盖本次调仓, 下次调仓可重新入选

        # 2) 日规则: 昨涨停 且 今未封 → 当日收盘卖出
        if d != all_dates[0]:
            for code in list(positions.keys()):
                pc = pct_map2.get(code, {})
                p_prev = pc.get(all_dates[idx - 1])
                p_cur = pc.get(d)
                if p_prev is None or p_cur is None:
                    continue
                lim = limit_pct(code, d_int)
                if p_prev >= lim - 0.1 and p_cur < lim - 0.1:
                    px = close_map.get(code, {}).get(d)
                    if px:
                        sell = positions[code]
                        proceeds = px * sell - calc_fee("sell", px, sell)
                        cash += proceeds
                        del positions[code]; buy_adj.pop(code, None)
                        entry_price.pop(code, None); sl_dict.pop(code, None)
                        n_daily_sells += 1

        # 2.5) 止损信号: 仅"当日收盘"触发判定; 实际卖出在次日成交价(get_open_price)执行,
        #      由步骤 0.5 完成 → 与引擎成交价约定完全一致, 且杜绝同日开盘进/收盘出的未来函数式偏差
        #   硬止损 = 收盘 < 首次买入价 × (1 - stop_loss)
        #   ATR止损 = 收盘 < 防线（入场价-ATR×mult 起步, 最高价-ATR×trail 追踪, 只升不降）
        if (stop_loss > 0 or atr_stop > 0) and positions:
            for code in list(positions.keys()):
                cur_close = close_map.get(code, {}).get(d)
                if cur_close is None:
                    continue
                should = False
                if atr_stop > 0:
                    sl = sl_dict.get(code)
                    if sl is not None:
                        hi = high_map.get(code, {}).get(d)
                        atr_t = atr_map.get(code, {}).get(d)
                        if hi is not None and atr_t and atr_t > 0:
                            sl.update(hi, atr_t)
                        should, stop_px, _ = sl.check_stop(cur_close)
                else:
                    ep = entry_price.get(code)
                    if ep is not None:
                        stop_px = ep * (1.0 - stop_loss)
                        should = cur_close < stop_px
                if should and code not in pending_stops:
                    pending_stops[code] = True   # 次日成交价卖出, 见步骤 0.5

        # 3) 估值（收盘）
        mv_r, mv_h = cash, cash
        for code, sh in positions.items():
            c = _ffill(close_map, code, d, all_dates, idx)
            if c is None:
                continue
            mv_r += sh * c
            a_t = _ffill(adj_map, code, d, all_dates, idx) or 1.0
            bf = buy_adj.get(code, 1.0) or 1.0
            mv_h += sh * c * a_t / bf
        nav_raw.append((d, mv_r))
        nav_hfq.append((d, mv_h))

    # ── 指标 ──
    def metrics(nav):
        vals = np.array([v for _, v in nav], dtype=float)
        dates = [d for d, _ in nav]
        n = len(vals)
        tot = vals[-1] / vals[0] - 1
        years = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days / 365.25
        ann = (vals[-1] / vals[0]) ** (1 / years) - 1 if years > 0 else 0
        peak = np.maximum.accumulate(vals)
        mdd = (vals / peak - 1).min()
        rets = np.diff(vals) / vals[:-1]
        vol = rets.std() * np.sqrt(252) if len(rets) > 1 else 0
        sharpe = (ann - 0.02) / vol if vol > 0 else 0
        return dict(total=tot, ann=ann, mdd=mdd, sharpe=sharpe, final=vals[-1])

    def bench_nav(bmap):
        levels = [bmap.get(d) for d in all_dates]
        fv = next((i for i, v in enumerate(levels) if v is not None), None)
        if fv is None:
            return [(d, CAPITAL) for d in all_dates]
        base = levels[fv]
        out, last = [], None
        for i, v in enumerate(levels):
            if v is None:
                if last is None:
                    last = levels[fv] / base * CAPITAL
                out.append(last)
            else:
                last = v / base * CAPITAL
                out.append(last)
        return list(zip(all_dates, out))

    nav_b300 = bench_nav(b300)
    nav_b932 = bench_nav(b932) if b932 else None

    m_raw = metrics(nav_raw)
    m_hfq = metrics(nav_hfq)
    m_b300 = metrics(nav_b300)
    m_b932 = metrics(nav_b932) if nav_b932 else None

    def yearly(nav):
        df = pd.DataFrame(nav, columns=["date", "v"])
        df["year"] = df["date"].str[:4]
        out = {}
        for y, g in df.groupby("year"):
            out[y] = g["v"].iloc[-1] / g["v"].iloc[0] - 1
        return out

    y_raw, y_hfq, y_b = yearly(nav_raw), yearly(nav_hfq), yearly(nav_b300)

    # ── 保存 ──
    os.makedirs(RES_DIR, exist_ok=True)
    nav_df = pd.DataFrame({"date": all_dates, "value_raw": [v for _, v in nav_raw],
                           "value_hfq": [v for _, v in nav_hfq],
                           "bench300": [v for _, v in nav_b300]})
    nav_df.to_csv(f"{RES_DIR}/nav_{start}_{end}.csv", index=False)
    if log:
        pd.DataFrame(log, columns=["rebal_date", "sel_date", "ts_code", "dv_ttm", "pe",
                                   "peg", "roe", "np_yoy", "tr_yoy"]).to_csv(
            f"{RES_DIR}/picks_{start}_{end}.csv", index=False)
    if picks:
        pd.DataFrame(picks, columns=["rebal_date", "codes"]).to_csv(
            f"{RES_DIR}/targets_{start}_{end}.csv", index=False)

    return dict(start=start, end=end, picks=picks, n_rebal_sells=n_rebal_sells,
                n_daily_sells=n_daily_sells, n_stop_sells=n_stop_sells,
                m_raw=m_raw, m_hfq=m_hfq,
                m_b300=m_b300, m_b932=m_b932, y_raw=y_raw, y_hfq=y_hfq, y_b=y_b,
                nav_raw=nav_raw, nav_hfq=nav_hfq, nav_b300=nav_b300)


def fmt_pct(x, signed=False):
    if x is None:
        return "-"
    return f"{x*100:+.2f}%" if signed else f"{x*100:.2f}%"


def print_window(r, show_yearly=False):
    if r is None:
        return
    print(f"===== {r['start']} → {r['end']} =====")
    print(f"  选股 {len(r['picks'])} 期 / 调仓卖出 {r['n_rebal_sells']} 笔 / 日规则卖出 {r['n_daily_sells']} 笔 / 止损卖出 {r.get('n_stop_sells', 0)} 笔")
    m = r["m_raw"]; mh = r["m_hfq"]; mb = r["m_b300"]
    print(f"  NAV(raw, 纯价): 总收益 {fmt_pct(m['total'])} / 年化 {fmt_pct(m['ann'])} / 最大回撤 {fmt_pct(m['mdd'])} / 夏普 {m['sharpe']:.3f}")
    print(f"  NAV(hfq, 含分红): 总收益 {fmt_pct(mh['total'])} / 年化 {fmt_pct(mh['ann'])} / 最大回撤 {fmt_pct(mh['mdd'])} / 夏普 {mh['sharpe']:.3f}")
    print(f"  基准 沪深300:  总收益 {fmt_pct(mb['total'])} / 年化 {fmt_pct(mb['ann'])} / 最大回撤 {fmt_pct(mb['mdd'])}")
    print(f"  超额(hfq-沪深300): {fmt_pct(mh['total']-mb['total'], signed=True)}")
    if r["m_b932"]:
        print(f"  中证红利(932000) 总收益: {fmt_pct(r['m_b932']['total'])} (基准之一)")
    if show_yearly:
        yrs = sorted(set(r["y_hfq"].keys()))
        rows = []
        for y in yrs:
            rows.append({"年份": y, "策略hfq": fmt_pct(r["y_hfq"].get(y, 0), signed=True),
                         "策略raw": fmt_pct(r["y_raw"].get(y, 0), signed=True),
                         "沪深300": fmt_pct(r["y_b"].get(y, 0), signed=True)})
        print(pd.DataFrame(rows).to_string(index=False))
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20140101")
    ap.add_argument("--end", default="20260720")
    ap.add_argument("--windows", default=None,
                    help="多窗口逗号分隔: 20140101-20261231,20140101-20161231,...")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--top-pct", type=float, default=0.10)
    ap.add_argument("--pe-max", type=float, default=20.0)
    ap.add_argument("--peg-min", type=float, default=0.08)
    ap.add_argument("--peg-max", type=float, default=2.0)
    ap.add_argument("--roe-min", type=float, default=3.0)
    ap.add_argument("--rev-min", type=float, default=5.0)
    ap.add_argument("--np-min", type=float, default=11.0)
    ap.add_argument("--stop-loss", type=float, default=0.0,
                    help="硬止损：收盘 < 买入价×(1-x) 卖出，如 0.15=15%%；0=关闭(默认)")
    ap.add_argument("--atr-stop", type=float, default=0.0,
                    help="ATR动态止损倍数(period14)：入场价-ATR×x 起步，最高价-ATR×x 追踪；0=关闭(默认)")
    ap.add_argument("--atr-period", type=int, default=14)
    ap.add_argument("--yearly", action="store_true")
    args = ap.parse_args()

    cfg = dict(top_n=args.top_n, top_pct=args.top_pct, pe_max=args.pe_max,
               peg_min=args.peg_min, peg_max=args.peg_max, roe_min=args.roe_min,
               rev_min=args.rev_min, np_min=args.np_min,
               stop_loss=args.stop_loss, atr_stop=args.atr_stop, atr_period=args.atr_period)

    if args.windows:
        wins = [w.split("-") for w in args.windows.split(",")]
        print("参数: top_n=%d top_pct=%.0f%% PE≤%.0f PEG[%.2f,%.2f] ROE>%.0f%% 营收>%.0f%% 净利>%.0f%%"
              % (cfg["top_n"], cfg["top_pct"]*100, cfg["pe_max"], cfg["peg_min"], cfg["peg_max"],
                 cfg["roe_min"], cfg["rev_min"], cfg["np_min"]))
        print(f"初始资金 {CAPITAL:,.0f} / 基准 {INDEX_PRIMARY}\n")
        rows = []
        for s, e in wins:
            r = run_window(s, e, cfg)
            if r is None:
                continue
            rows.append({"窗口": f"{s}-{e}",
                         "策略hfq": fmt_pct(r["m_hfq"]["total"]),
                         "策略raw": fmt_pct(r["m_raw"]["total"]),
                         "沪深300": fmt_pct(r["m_b300"]["total"]),
                         "超额(hfq)": fmt_pct(r["m_hfq"]["total"] - r["m_b300"]["total"], signed=True),
                         "年化hfq": fmt_pct(r["m_hfq"]["ann"]),
                         "回撤raw": fmt_pct(r["m_raw"]["mdd"]),
                         "夏普hfq": f"{r['m_hfq']['sharpe']:.2f}"})
            if r["m_b932"]:
                rows[-1]["中证红利"] = fmt_pct(r["m_b932"]["total"])
        print(pd.DataFrame(rows).to_string(index=False))
        print("\n[口径] hfq=后复权含分红(主口径); raw=原始价不含分红(偏低); 超额=hfq-沪深300价格指数(不含分红)。")
    else:
        r = run_window(args.start, args.end, cfg)
        print_window(r, show_yearly=args.yearly)


if __name__ == "__main__":
    main()
