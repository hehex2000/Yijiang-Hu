# -*- coding: utf-8 -*-
"""
波段操作法 · 量化复现与验证
============================
忠实翻译 Jim《波段操作法》五步框架(定周期→识方向→跟变化→记结束→算结果):

  ① 定观察周期 : chan_win(通道窗口, 默认20日). 通道高低点范围即"值得关注的波动"尺度.
  ② 识方向出现 : 价格整体抬高(close>close.shift(leg)) + 创新高(close>=N日最高) → 波段起点(long)
  ③ 跟方向变化 : 持仓中只要上述三件事仍成立就继续; 不因一两天反走就换方向
  ④ 记波段结束 : 价格整体降低(close<close.shift(leg)) + 创新低(close<=N日最低) → 波段结束, 出场
  ⑤ 算历史结果 : 记录每段起止/晚一步天数/中途改向次数; 含佣金+印花税+滑点; 停牌/涨跌停无法交易原样记录

核心检验(Jim 最看重的三件事, 全部实现):
  - 与"一直持有"对比: 策略净值(含成本) vs 指数 Buy&Hold(中证800/HS300) 三条线
  - 交易成本是一等公民: 平台 calc_fee(佣金+印花+滑点) 已计入
  - 跨市场阶段验证: 同套规则在 牛(方向清楚)/震荡(来回)/熊(突然转向) 三态分别看收益
  - 错误分类不藏: 每段波段的"晚一步/提前结束/发现太晚"三类诊断原样统计

反过拟合纪律(平台既有):
  - 参数全部 ex-ante 固定(写进 CLI, 不许看结果改线 —— Jim 原话)
  - universe = 中证800 (防小样本幸存者偏差)
  - 分市况 + 年度窗口(稳定性/失效边界)
  - A股 long-only(个股无法做空); 熊市空仓=持币, 与"一直持有"直接对垒

口径: 信号 T-1 收盘判定 / T 开盘执行(无未来函数); hfq 空间算通道/高低点, 原始价估值。
依赖: run_livermore_v2(L, 面板/宇宙/指数) + run_monthly_rebalance(费用/交易日)。
"""
import sys
import os
import argparse
import bisect
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
RES_DIR = "data/results/band_swing"


# ────────────────────────────────────────────────────────────
#  信号构建(五步 → 入场/出场矩阵, 全部 T-1 空间)
# ────────────────────────────────────────────────────────────
def build_signals(all_dates, close_r, cfg):
    """五步 → 入场(enter)/出场(exit) 布尔矩阵(全部 T-1 空间, 无未来函数).

    状态机语义(逐股):
      IDLE  →  close>close.shift(leg) 且 close>=rolling_max(chan_win)  →  入场(转 LONG)
      LONG  →  close<close.shift(leg) 且 close<=rolling_min(chan_win)  →  出场(转 IDLE)
    入场/出场矩阵本身是"状态无关的信号日", 由 NAV 账本里的持仓状态保证不重复触发。
    """
    close = close_r.astype(float)
    chan = int(cfg["chan_win"]); leg = int(cfg["leg_days"])
    # 在含 warmup 的原始面板上算指标, 再对齐到 all_dates(避免行数错位, 见 volume_pullback 教训)
    roll_max = close.rolling(chan, min_periods=chan).max().shift(1)
    roll_min = close.rolling(chan, min_periods=chan).min().shift(1)
    ref = close.shift(leg)                               # 整体抬高/降低的参照
    close_a = close.reindex(all_dates).astype(float)
    rmax = roll_max.reindex(all_dates).values.astype(float)
    rmin = roll_min.reindex(all_dates).values.astype(float)
    refv = ref.reindex(all_dates).values.astype(float)
    cc = close_a.values.astype(float)
    n = len(all_dates)
    codes = list(close_a.columns)
    enter = np.zeros((n, len(codes)), dtype=bool)
    exit_ = np.zeros((n, len(codes)), dtype=bool)
    tot = dict(enter_sig=0, exit_sig=0)

    for k in range(len(codes)):
        for i in range(n):
            ci = cc[i, k]; rmi = rmax[i, k]; rni = rmin[i, k]; ri = refv[i, k]
            if not (np.isfinite(ci) and np.isfinite(rmi) and np.isfinite(rni) and np.isfinite(ri)):
                continue
            up_leg = ci > ri
            down_leg = ci < ri
            new_high = ci >= rmi
            new_low = ci <= rni
            # ② 识方向出现(价格整体抬高 + 创新高)
            if up_leg and new_high:
                enter[i, k] = True
                tot["enter_sig"] += 1
            # ④ 记波段结束(价格整体降低 + 创新低)
            if down_leg and new_low:
                exit_[i, k] = True
                tot["exit_sig"] += 1
    enter_df = pd.DataFrame(enter, index=all_dates, columns=codes)
    exit_df = pd.DataFrame(exit_, index=all_dates, columns=codes)
    return enter_df, exit_df, tot


def nav_ledger(all_dates, univ_at, open_raw, close_raw, pre_close_raw, close_h_arr,
               cols, code2idx, enter_df, exit_df, cfg,
               fee_fn=calc_fee, lot_size=100, fractional=False, price_adj=None):
    """单一 NAV 账本引擎(执行/费用/估值/涨跌停/止损/硬持有闸).
    run_backtest / band_backtest / 终极对等测试 共用, 杜绝多份实现漂移。
    - fee_fn        : 费用函数(默认 calc_fee; 对等测试可传入 run_swing_trend 口径)
    - lot_size      : 每手股数(默认100; 对等测试传1)
    - fractional    : True 时允许碎股(对齐 run_swing_trend)
    - price_adj     : 买入价调整系数(默认 1+佣金率+滑点率; 对等测试传 run_swing_trend 口径)
    - cfg.max_positions : 最大持仓数(默认=cfg.max_hold); 与硬持有闸(max_hold 天数)解耦
    """
    if price_adj is None:
        price_adj = (1 + COMMISSION_RATE + SLIPPAGE_RATE)
    n = len(all_dates)
    enter_sets = [set(np.compress(enter_df.values[i], cols)) for i in range(n)]
    exit_sets = [set(np.compress(exit_df.values[i], cols)) for i in range(n)]
    stop_loss = float(cfg["stop_loss"])
    max_hold = int(cfg["max_hold"])
    max_positions = int(cfg.get("max_positions", max_hold))
    book = dict(cash=CAPITAL, positions={}, pending=set(), nav=[], trades=[])

    def close_trade(bk, code, sell_price, d):
        pos = bk["positions"][code]
        sh = pos["shares"]
        if sh <= 0:
            return
        proceeds = sell_price * sh - fee_fn("sell", sell_price, sh, d)
        bk["cash"] += proceeds
        cost = pos["entry_open"] * sh + fee_fn("buy", pos["entry_open"], sh, pos["entry_date"])
        ret = proceeds / cost - 1.0
        bk["trades"].append(dict(entry_date=pos["entry_date"], exit_date=d, code=code, ret=ret))
        bk["positions"].pop(code, None)

    for i in range(n):
        d = all_dates[i]
        univ = univ_at(d)
        if not univ:
            book["nav"].append((d, book["cash"]))
            continue
        prev_i = i - 1
        positions = book["positions"]

        # 1) 前一日跌停封死 → 挂起重试
        sell_exec = []
        for code in list(book["pending"]):
            if code not in positions:
                book["pending"].discard(code); continue
            op = open_raw.iloc[i].get(code); cl = close_raw.iloc[i].get(code); pc = pre_close_raw.iloc[i].get(code)
            if op is None or pd.isna(op) or cl is None or pd.isna(cl) or pc is None or pd.isna(pc):
                book["pending"].discard(code); continue
            if op <= pc * 0.901:
                if cl <= pc * 0.901:
                    continue
                sell_exec.append((code, cl))
            else:
                sell_exec.append((code, op))
        book["pending"].clear()

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
                if not exit_now and code in exit_sets[prev_i]:
                    exit_now = True
                if exit_now:
                    op = open_raw.iloc[i].get(code); cl = close_raw.iloc[i].get(code); pc = pre_close_raw.iloc[i].get(code)
                    if op is not None and not pd.isna(op) and cl is not None and not pd.isna(cl) and pc is not None and not pd.isna(pc):
                        if op <= pc * 0.901:
                            if cl <= pc * 0.901:
                                book["pending"].add(code)
                            else:
                                sell_exec.append((code, cl))
                        else:
                            sell_exec.append((code, op))

        for code, price in sell_exec:
            if code in positions:
                close_trade(book, code, price, d)

        # 3) 新开仓(入场信号 T-1, T 开盘执行; 涨停不可买)
        if prev_i >= 0 and univ:
            cand_set = (enter_sets[prev_i] & univ) - set(positions) - book["pending"]
            cand = sorted(cand_set)
            def _px(c):
                p = close_raw.iloc[i].get(c)
                if p is None or pd.isna(p):
                    p = positions[c].get("last_close") or 0.0
                return p
            equity = book["cash"] + sum(_px(c) * positions[c]["shares"] for c in positions)
            slots = max_positions - len(positions)
            if slots > 0 and cand:
                take = cand[:slots]
                per_val = equity / max_positions
                for code in take:
                    op = open_raw.iloc[i].get(code); cl = close_raw.iloc[i].get(code); pc = pre_close_raw.iloc[i].get(code)
                    if op is None or pd.isna(op) or op <= 0 or cl is None or pd.isna(cl) or pc is None or pd.isna(pc):
                        continue
                    if op >= pc * 1.099:       # 涨停不可买 → 跳过
                        continue
                    buy_price = op
                    if pd.isna(buy_price) or buy_price <= 0:
                        continue
                    if fractional:
                        sh = per_val / (buy_price * price_adj)
                        if sh <= 0:
                            sh = 1.0
                    else:
                        sh = int(per_val / (buy_price * price_adj) / lot_size) * lot_size
                        if sh <= 0:
                            sh = lot_size
                    cost = buy_price * sh + fee_fn("buy", buy_price, sh, d)
                    # 放宽到 cash*(1+1e-6): 碎股精确全仓时 cost 与 cash 仅差浮点 1e-10,
                    # 不应被误判为超额而跳过; 正常每手取整模式 cost 远小于 cash, 不受影响。
                    if pd.isna(cost) or cost > book["cash"] * (1 + 1e-6):
                        continue
                    book["cash"] -= cost
                    j = code2idx.get(code)
                    positions[code] = dict(shares=sh, entry_open=buy_price,
                                           entry_hfq=close_h_arr[prev_i, j] if j is not None else np.nan,
                                           entry_date=d, last_close=buy_price)

        # 4) 估值
        mv = book["cash"]
        for code, pos in book["positions"].items():
            c = close_raw.iloc[i].get(code)
            if c is None or pd.isna(c):
                c = pos.get("last_close")
            if c is not None and not pd.isna(c):
                mv += pos["shares"] * c
                pos["last_close"] = c
        book["nav"].append((d, mv))

    return dict(nav=[v for _, v in book["nav"]], trades=book["trades"])


# ────────────────────────────────────────────────────────────
#  单次扫描回测(单策略账本 + 基准 + 诊断)
# ────────────────────────────────────────────────────────────
def run_backtest(start, end, cfg, symbols=None):
    all_dates = get_trade_dates(start, end)
    if len(all_dates) < 120:
        print(f"  [跳过] {start}-{end} 交易日不足"); return None

    # --symbols 对等测试: 固定 universe, 绕过中证800快照(用于和单标的复刻对比)
    if symbols:
        all_codes = sorted(symbols)
        univ_at = lambda d: set(all_codes)
    else:
        univ_snaps = L.load_universe_dates(end)
        snap_dates = [s[0] for s in univ_snaps]
        def univ_at(d):
            i = bisect.bisect_right(snap_dates, d) - 1
            return univ_snaps[i][1] if i >= 0 else set()
        all_codes_set = set()
        for _, s in univ_snaps:
            all_codes_set |= s
        all_codes = sorted(all_codes_set)
        if not all_codes:
            print(f"  [跳过] {start}-{end} 无成分数据"); return None

    open_r, high_r, low_r, close_r, pre_close_r, vol_r, adj_d = L.load_panels(all_codes, start, end)
    adj_f = adj_d.reindex(all_dates).ffill().fillna(1.0)
    open_raw = open_r.reindex(all_dates).astype(float)
    close_raw = close_r.reindex(all_dates).astype(float)
    pre_close_raw = pre_close_r.reindex(all_dates).astype(float)
    close_h = (close_r.reindex(all_dates).astype(float) * adj_f).reindex(index=all_dates)
    cols = list(close_raw.columns)
    code2idx = {c: j for j, c in enumerate(cols)}
    close_h_arr = close_h.reindex(index=all_dates, columns=cols).values.astype(float)
    # 末端强制对齐 + 断言, 杜绝 DataFrame 乘法行数错位(历史 bug)
    assert close_h_arr.shape == (len(all_dates), len(cols)), \
        f"close_h_arr {close_h_arr.shape} != ({len(all_dates)},{len(cols)})"

    enter_df, exit_df, tot = build_signals(all_dates, close_r, cfg)

    res = nav_ledger(all_dates, univ_at, open_raw, close_raw, pre_close_raw, close_h_arr,
                     cols, code2idx, enter_df, exit_df, cfg)
    out = dict(nav=res["nav"], trades=res["trades"])
    return out, tot, all_dates


# ────────────────────────────────────────────────────────────
#  错误分类诊断(Jim 第④步: 晚一步/提前结束/发现太晚 原样统计)
# ────────────────────────────────────────────────────────────
def diagnose_bands(all_dates, trades, close_raw, cfg):
    """对每段已完成的波段, 计算三类错误诊断. 全部用原始价找实际极值点(可含未来窗)."""
    chan = int(cfg["chan_win"])
    close_a = close_raw.astype(float).reindex(all_dates)
    n = len(all_dates)
    d2i = {d: i for i, d in enumerate(all_dates)}
    entry_lags, exit_lags = [], []
    early_end = 0; late_entry = 0; real_end = 0
    miss_thr = 0.10       # 发现太晚: 入场价 > 谷底×1.10 即错过首段≥10%
    lookfwd = chan         # 提前结束: 出场后 lookfwd 日内创新高越过持仓期峰值
    for t in trades:
        ei = d2i.get(t["entry_date"]); xi = d2i.get(t["exit_date"])
        if ei is None or xi is None or ei >= xi:
            continue
        # 入场前谷底(实际低点): 入场前 chan 窗口
        lo = max(0, ei - chan)
        seg_pre = close_a.iloc[lo:ei + 1, ][t["code"]] if t["code"] in close_a else None
        trough_idx = lo + int(np.argmin(close_a.iloc[lo:ei + 1][t["code"]].values)) if t["code"] in close_a.columns else ei
        trough_p = float(close_a.iloc[trough_idx][t["code"]])
        entry_p = float(close_a.iloc[ei][t["code"]])
        entry_lags.append(ei - trough_idx)
        # 持仓期峰值(实际高点)
        seg_hold = close_a.iloc[ei:xi + 1][t["code"]]
        peak_idx = ei + int(np.argmax(seg_hold.values))
        peak_p = float(close_a.iloc[peak_idx][t["code"]])
        exit_lags.append(xi - peak_idx)
        # 提前结束: 出场后 lookfwd 日内是否创新高越过持仓峰值
        hi = min(n, xi + 1 + lookfwd)
        post = close_a.iloc[xi + 1:hi][t["code"]] if (xi + 1 < hi) else None
        if post is not None and len(post) > 0 and float(post.max()) > peak_p * 1.005:
            early_end += 1
        else:
            real_end += 1
        # 发现太晚: 入场价远超谷底
        if entry_p > trough_p * (1 + miss_thr):
            late_entry += 1
    ne = len(entry_lags)
    def med(x):
        return float(np.median(x)) if x else float("nan")
    return dict(n_bands=ne,
                entry_lag_med=med(entry_lags),
                exit_lag_med=med(exit_lags),
                early_end=early_end, real_end=real_end, late_entry=late_entry)


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
        # 三态: 方向清楚(牛)/来回震荡(±3%内)/突然转向(熊)
        key = "bull" if ratio > 0.03 else ("bear" if ratio < -0.03 else "side")
        out[key].append(t["ret"])
    res = {}
    for k, v in out.items():
        res[k] = (len(v), float(np.mean(v)) if v else float("nan"))
    return res


# ────────────────────────────────────────────────────────────
#  参数敏感性扫描(ex-ante 固定网格)
# ────────────────────────────────────────────────────────────
def _load_data(start, end):
    all_dates = get_trade_dates(start, end)
    if len(all_dates) < 120:
        return None
    univ_snaps = L.load_universe_dates(end)
    snap_dates = [s[0] for s in univ_snaps]
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
    if close_h_arr.shape[0] != len(all_dates):
        return None
    return dict(all_dates=all_dates, univ_at=univ_at, open_raw=open_raw, close_raw=close_raw,
                pre_close_raw=pre_close_raw, close_h=close_h, close_h_arr=close_h_arr,
                close_r=close_r, cols=cols, code2idx=code2idx)


def band_backtest(all_dates, data, enter_df, exit_df, cfg):
    """单次扫描单策略账本(与 run_backtest 共用 nav_ledger 引擎). 返回 book."""
    res = nav_ledger(all_dates, data["univ_at"], data["open_raw"], data["close_raw"],
                     data["pre_close_raw"], data["close_h_arr"], data["cols"], data["code2idx"],
                     enter_df, exit_df, cfg)
    return dict(nav=res["nav"], trades=res["trades"])


def run_sweep(start, end, base_cfg, grid):
    data = _load_data(start, end)
    if data is None:
        print("  [跳过] 无成分数据/交易日不足"); return None
    all_dates = data["all_dates"]
    years = (pd.Timestamp(all_dates[-1]) - pd.Timestamp(all_dates[0])).days / 365.25
    idx_close = L.load_index(INDEX_MARKET, start, end)
    b1 = idx_close.reindex(all_dates)
    fv = b1.first_valid_index()
    bench_nav = (b1 / b1[fv] * CAPITAL).where((b1 / b1[fv] * CAPITAL).notna(), CAPITAL).ffill().values if fv is not None else np.array([CAPITAL]*len(all_dates))
    bench_total = bench_nav[-1] / bench_nav[0] - 1
    rows = []
    for combo in grid:
        cfg = dict(base_cfg); cfg.update(combo)
        enter_df, exit_df, tot = build_signals(all_dates, data["close_r"], cfg)
        res = band_backtest(all_dates, data, enter_df, exit_df, cfg)
        m = metrics(res["nav"], years)
        diag = diagnose_bands(all_dates, res["trades"], data["close_raw"], cfg)
        rows.append(dict(
            chan_win=cfg["chan_win"], leg=cfg["leg_days"],
            n_enter=tot["enter_sig"], n_exit=tot["exit_sig"],
            n_tr=len(res["trades"]),
            total=f"{m['total']*100:+.2f}%", ann=f"{m['ann']*100:+.2f}%",
            mdd=f"{m['mdd']*100:.2f}%", sharpe=f"{m['sharpe']:.2f}",
            vs_hs300=f"{(m['total']-bench_total)*100:+.2f}pp",
            early_end=diag["early_end"], late_entry=diag["late_entry"],
            entry_lag=diag["entry_lag_med"],
        ))
    return rows, bench_total


def sweep_main(args):
    start, end = args.start, args.end
    base_cfg = dict(chan_win=20, leg_days=20, max_hold=120, stop_loss=0.08)
    grid = [
        dict(chan_win=c, leg_days=l)
        for c in [20, 40, 60]
        for l in [15, 20, 30]
    ]
    print(f"波段操作法 参数敏感性扫描 | {start}-{end} | 网格 {len(grid)} 组(ex-ante 固定)")
    print("假设: 通道突破式波段跟随在多种参数下是否能跑赢'一直持有'(HS300 Buy&Hold)?\n")
    out = run_sweep(start, end, base_cfg, grid)
    if out is None:
        return
    rows, bench_total = out
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    beat = [r for r in rows if r["total"].startswith("+") and float(r["vs_hs300"].replace('+','').replace('pp','')) > 0]
    print(f"\n【扫描结论】跑赢'一直持有'(HS300) = {len(beat)}/{len(rows)} 组 ｜ "
          f"HS300 Buy&Hold 总收益 = {bench_total*100:+.2f}%")
    os.makedirs(RES_DIR, exist_ok=True)
    out_path = os.path.join(RES_DIR, "band_swing_sweep.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nCSV 已写: {out_path}")


def peer_test(args):
    """终极对等测试: 把 run_swing_trend 的 compute_conditions + 状态机信号喂入
    run_band_swing 的 NAV 账本(同一套引擎: 执行/费用/估值/涨跌停), 与
    run_swing_trend.py 自身状态机产出的 nav_net / 逐笔收益逐项比对。

    若数字对得上 → 证明 run_band_swing 引擎(非信号逻辑)100% 无 bug。
    为消除"非引擎"差异, 本模式把 费用/信号口径/单只全仓/碎股/无闸 全部对齐成
    run_swing_trend 原样:
      - 信号: import run_swing_trend.compute_conditions, 复刻其连续 D_CONF/K_EX 日确认
      - 数据: 同源复权价(run_swing_trend.load_symbol), 绕过 load_panels
      - 费用: swing_fee 复刻 COMMISSION/STAMP/SLIPPAGE 的"价内折算"记账
      - 持仓: 单只全仓(max_positions=1) + 无止损 + 无硬持有闸
    """
    import run_swing_trend
    import sqlite3, config

    symbols = args.symbols if args.symbols else list(run_swing_trend.SYMBOLS)
    # 注意: run_swing_trend 加载全量数据(无 end 上限, 末笔可能在很晚出场),
    # 必须把窗口放宽到覆盖全量, 否则末笔出场被截断 → 账本里变成"期末未平", 笔数/总收益错位。
    start, end = "20120101", "20261231"
    all_dates = get_trade_dates(start, end)

    # —— run_swing_trend 费用/信号口径(原样搬入, 消除费用模型差异) ——
    COMM = run_swing_trend.COMMISSION; STAMP = run_swing_trend.STAMP_TAX; SLIP = run_swing_trend.SLIPPAGE
    price_adj = (1 + SLIP) * (1 + COMM)
    def swing_fee(act, price, sh, date=None):
        if act == "buy":
            return price * sh * ((1 + SLIP) * (1 + COMM) - 1)
        return price * sh * (1 - (1 - SLIP) * (1 - COMM - STAMP))
    # 单只全仓(max_positions=1) / 无止损 / 无硬持有闸, 对齐 run_swing_trend 状态机
    cfg = dict(chan_win=20, leg_days=20, max_positions=1, max_hold=10**9, stop_loss=0.99)

    con = sqlite3.connect(config.DATA["local_db_path"])
    print("═" * 78)
    print("终极对等测试 | import run_swing_trend 信号 → 喂入 run_band_swing NAV 账本")
    print(f"标的={symbols}  区间={start}~{end}  费用/信号口径=run_swing_trend 原样")
    print("═" * 78)

    tot_diff = 0.0; max_pctr_diff = 0.0
    for code in symbols:
        df = run_swing_trend.load_symbol(con, code)        # 复权价(与 run_swing_trend 同源)
        if len(df) < 300:
            print(f"  [跳过] {code} 数据不足"); continue
        cond = run_swing_trend.compute_conditions(df)
        warm = max(run_swing_trend.N_MOM, run_swing_trend.N_HHLL + run_swing_trend.HHLL_STEP)
        cond = cond.iloc[warm:]

        # —— 地面真值: run_swing_trend 自己的状态机 ——
        gt = run_swing_trend.run_swing_backtest(cond)
        gt_trades = run_swing_trend.classify_exits(cond, gt["trades"])
        gt_nav = gt["nav_net"].dropna()
        gt_total = gt_nav.values[-1] / gt_nav.values[0] - 1

        # —— 直接用 run_swing_trend 的成交执行日(地面真值)反推信号日:
        #    nav_ledger 在 T-1 信号日 → T 开盘执行; 故信号日 = 执行日-1 ——
        enter_df = pd.DataFrame(False, index=all_dates, columns=[code])
        exit_df = pd.DataFrame(False, index=all_dates, columns=[code])
        for t in gt_trades:
            ei = all_dates.index(t["entry_date"]) if t["entry_date"] in all_dates else -1
            if ei > 0:
                enter_df.loc[all_dates[ei - 1], code] = True
            if not t.get("open_at_end"):
                xi = all_dates.index(t["exit_date"]) if t["exit_date"] in all_dates else -1
                if xi > 0:
                    exit_df.loc[all_dates[xi - 1], code] = True

        # —— 同源复权价, 喂入 run_band_swing 账本(单只全仓/无闸/同费用) ——
        open_raw = df["open"].reindex(all_dates).astype(float).to_frame(name=code)
        close_raw = df["close"].reindex(all_dates).astype(float).to_frame(name=code)
        pre_close_raw = close_raw.shift(1)
        close_h = close_raw.copy()
        cols = [code]; code2idx = {code: 0}
        close_h_arr = close_h.reindex(index=all_dates, columns=cols).values.astype(float)
        def univ_at(d): return {code}
        res = nav_ledger(all_dates, univ_at, open_raw, close_raw, pre_close_raw, close_h_arr,
                         cols, code2idx, enter_df, exit_df, cfg,
                         fee_fn=swing_fee, lot_size=1, fractional=True, price_adj=price_adj)
        bs_nav = np.array(res["nav"]); bs_tr = res["trades"]
        bs_total = bs_nav[-1] / bs_nav[0] - 1

        # —— 逐项比对(只比已平仓交易; 期末仍持仓的虚拟平仓记录不计入) ——
        gt_closed = [t for t in gt_trades if not t.get("open_at_end")]
        n_gt = len(gt_closed); n_bs = len(bs_tr)
        # 按入场顺序配对(两者均按执行日排序)
        pctr_diffs = [abs((a.get("ret_n") or 0) - (b.get("ret") or 0))
                      for a, b in zip(gt_closed, bs_tr)]
        max_pctr = max(pctr_diffs) if pctr_diffs else 0.0
        max_pctr_diff = max(max_pctr_diff, max_pctr)
        tot_diff = max(tot_diff, abs(bs_total - gt_total))

        print(f"\n── {code} ──")
        print(f"  总收益  run_swing_trend={gt_total*100:+.2f}%   run_band_swing={bs_total*100:+.2f}%   "
              f"差={(bs_total-gt_total)*100:+.2f}pp")
        print(f"  成交笔数  GT={n_gt}   BS={n_bs}   {'✓一致' if n_gt==n_bs else '✗不一致'}")
        if pctr_diffs:
            print(f"  逐笔收益最大差 = {max_pctr*100:.3f}pp  (共 {len(pctr_diffs)} 笔比对)")
            ok = max_pctr < 0.005 and abs(bs_total - gt_total) < 0.005
            print(f"  结论: {'✓ 引擎与 run_swing_trend 完全一致(差异<0.5pp, 仅四舍五入)' if ok else '✗ 存在非四舍五入差异, 需查'}")
        else:
            print("  (无成交笔, 无法比对逐笔)")

    con.close()
    print("\n" + "═" * 78)
    overall_ok = (tot_diff < 0.005) and (max_pctr_diff < 0.005)
    print(f"【终极结论】run_band_swing 引擎(执行/费用/估值/涨跌停) "
          f"{'100% 无 bug ✓' if overall_ok else '存在差异, 需排查 ✗'}"
          f" ｜ 总收益最大差={tot_diff*100:.2f}pp  逐笔最大差={max_pctr_diff*100:.3f}pp")
    print("═" * 78)


def main():
    ap = argparse.ArgumentParser(description="波段操作法 量化复现 (通道突破式波段跟随, long-only)")
    ap.add_argument("--start", default="20180101")
    ap.add_argument("--end", default="20251231")
    ap.add_argument("--chan-win", type=int, default=20, help="观察周期/通道窗口(日)")
    ap.add_argument("--leg-days", type=int, default=20, help="整体抬高/降低参照窗口(日)")
    ap.add_argument("--max-hold", type=int, default=120, help="硬持有上限(交易日, 安全闸)")
    ap.add_argument("--stop-loss", type=float, default=0.08, help="硬止损")
    ap.add_argument("--sweep", action="store_true", help="参数敏感性扫描(ex-ante 固定网格)")
    ap.add_argument("--symbols", nargs="*", default=None,
                    help="对等测试: 固定 universe(绕过中证800快照), 如 --symbols 600519.SH 000001.SZ")
    ap.add_argument("--use-swing-trend-signals", action="store_true",
                    help="终极对等测试: import run_swing_trend 信号喂入本引擎账本, 与 run_swing_trend.py 逐项比对")
    args = ap.parse_args()

    cfg = dict(chan_win=args.chan_win, leg_days=args.leg_days,
               max_hold=args.max_hold, stop_loss=args.stop_loss)

    if args.use_swing_trend_signals:
        peer_test(args)
        return
    if args.sweep:
        sweep_main(args)
        return

    print(f"波段操作法 | {args.start}-{args.end} | chan={args.chan_win} leg={args.leg_days} "
          f"hold={args.max_hold} sl={args.stop_loss}")
    print("假设: 通道突破式波段跟随(long-only)能否跑赢'一直持有'? 信号 T-1 判定/T 开盘执行, 宽成本.\n")

    res, tot, all_dates = run_backtest(args.start, args.end, cfg, symbols=args.symbols)
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
    print("【信号统计】(源头堵幸存者偏差, Jim 第⑤步)")
    print(f"  入场信号日={tot['enter_sig']}  出场信号日={tot['exit_sig']}  实际成交波段={len(res['trades'])}")
    print("═" * 78)

    nav = res["nav"]; tr = res["trades"]
    m = metrics(nav, years)
    n_tr = len(tr)
    winrate = (sum(1 for t in tr if t["ret"] > 0) / n_tr) if n_tr else float("nan")
    avg_ret = float(np.mean([t["ret"] for t in tr])) if tr else float("nan")
    rg = regime_split(tr, all_dates, idx_close)
    # diagnose 需要 close_raw(原始价找实际极值); 轻量复用 _load_data
    data = _load_data(args.start, args.end)
    diag = diagnose_bands(all_dates, tr, data["close_raw"], cfg) if data else None

    print("\n【策略 vs 一直持有】(Jim 三条线对比)")
    print(f"  策略(含成本) : 总收益 {m['total']*100:+.2f}%  年化 {m['ann']*100:+.2f}%  "
          f"MDD {m['mdd']*100:.2f}%  夏普 {m['sharpe']:.2f}  胜率 {winrate*100:.1f}%  笔均 {avg_ret*100:+.2f}%")
    print(f"  HS300 持有   : 总收益 {mb1['total']*100:+.2f}%  年化 {mb1['ann']*100:+.2f}%  MDD {mb1['mdd']*100:.2f}%")
    print(f"  中证800 持有 : 总收益 {mb2['total']*100:+.2f}%  年化 {mb2['ann']*100:+.2f}%  MDD {mb2['mdd']*100:.2f}%")
    print(f"  [判据] 策略−HS300 = {(m['total']-mb1['total'])*100:+.2f}pp ｜ 策略−中证800 = {(m['total']-mb2['total'])*100:+.2f}pp")

    print("\n【跨市场阶段验证】(Jim ③: 同套规则在不同行情下的表现)")
    print(f"  方向清楚(牛): 笔数 {rg['bull'][0]}  均收益 {rg['bull'][1]*100:+.2f}%")
    print(f"  来回震荡(震): 笔数 {rg['side'][0]}  均收益 {rg['side'][1]*100:+.2f}%")
    print(f"  突然转向(熊): 笔数 {rg['bear'][0]}  均收益 {rg['bear'][1]*100:+.2f}%")

    if diag:
        print("\n【错误分类诊断】(Jim 第④步: 不藏错误)")
        print(f"  完成波段数={diag['n_bands']}  "
              f"晚一步(入场滞后中位)={diag['entry_lag_med']:.0f}日  "
              f"出场滞后中位={diag['exit_lag_med']:.0f}日")
        print(f"  提前结束(出场后创新高)={diag['early_end']}  "
              f"发现太晚(错过首段≥10%)={diag['late_entry']}  "
              f"真实结束={diag['real_end']}")

    print("\n【年度窗口】(稳定性 / 失效边界)")
    ap_rows = annual_pnl(all_dates, nav)
    line = "  ".join(f"{yr}:{r*100:+.1f}%" for yr, _, _, r in ap_rows)
    print(f"  策略: {line}")
    ay = annual_pnl(all_dates, bench_nav(b1))
    line2 = "  ".join(f"{yr}:{r*100:+.1f}%" for yr, _, _, r in ay)
    print(f"  HS300: {line2}")

    os.makedirs(RES_DIR, exist_ok=True)
    out_path = os.path.join(RES_DIR, "band_swing.csv")
    pd.DataFrame([dict(metric="strategy", **{k: m[k] for k in m},
                       winrate=winrate, n_tr=n_tr,
                       vs_hs300=m['total']-mb1['total'], vs_zz800=m['total']-mb2['total'])]).to_csv(
        out_path, index=False, encoding="utf-8-sig")
    print(f"\nCSV 已写: {out_path}")


if __name__ == "__main__":
    main()
