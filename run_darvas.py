# -*- coding: utf-8 -*-
"""
达瓦斯箱体 · 个股选股策略（方向 B · 学以致用 · v2 改进版）
==================================================
把白名单 UP 主 Jim(BV1jTM16yEoU) 讲的达瓦斯箱体精神落地为一只 A 股个股策略。

v2 相对 v1(首版)的四处改进（对应首版回测暴露的三大结构问题）：
  ① 箱体收窄为参数化分档（5%~15%，默认 10%；首版硬编码 20% 太宽→信号稀疏+单笔大亏）
  ② 趋势过滤：仅当 收盘 > MA200（处于上升趋势）才允许突破信号，压低震荡市假突破
  ③ 止损触发口径：改为「收盘跌破箱底」才离场（首版用日内最低价，被下影线噪声打出大量微亏）
  ④ 金字塔加仓：持仓股走出更高箱体(新高箱顶)时加仓（≤原仓 0.6 且总仓≤2x），贴合 Jim「走势延续才加仓」

基本面预筛（结合平台已有的 动量/价值/成长 三因子）：
  - 动量：过去 6 个月收益率（跳过最近 1 个月）> 0
  - 价值：0 < pe_ttm <= 40 且 0 < pb <= 6
  - 成长：营业利润同比 op_yoy > 0 或 basic_eps_yoy > 0
  - 流动性：total_mv >= 30 亿；上市 >= 1 年；排除 ST / *ST；排除 科创板(688)

费用：完全复用引擎 calc_fee（佣金万2.5最低5元 + 印花税历史分段 + 滑点0.1%损失向）。
基准：股票池对应指数（默认 中证800 000906.SH）。

运行（venv_ml python）：
  run_darvas.py --box_pct 0.10 --variant imp          # 改进版 10% 箱体
  run_darvas.py --box_pct 0.05 --variant imp          # 改进版 5% 箱体
  run_darvas.py --box_pct 0.20                        # 首版基线（无改进）
开关：--no-trend / --stop-intraday / --no-pyramid
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_monthly_rebalance import (
    get_conn, get_price, get_open_price, calc_fee, get_trade_dates,
    get_monthly_5th_trading_days, get_index_constituents, get_stock_name,
    INIT_CAPITAL, INDEX_DISPLAY_NAME,
)

# ── 策略参数（顶部可调）──
BOX_WIN = 60          # 箱体识别窗口（交易日）
BOX_PCT = 0.20        # 箱体宽度容忍度（默认模块值；改进运行传 0.05~0.15）
TOP_N = 10            # 持仓数量（等权重）
MOM_LOOKBACK = 126    # 动量回看交易日（≈6个月）
MOM_SKIP = 21         # 跳过最近交易日（≈1个月，避短期反转）
MIN_BOX_AGE = 3       # 箱顶至少 N 日前形成（确保有整理）
PE_MAX = 40.0
PB_MAX = 6.0
MV_MIN_WAN = 300000.0 # 总市值下限（万元），30亿
LIST_MIN_YEARS = 1    # 上市至少年数

# ── v2 改进开关（默认全开）──
TREND_FILTER = True   # ② 趋势过滤：收盘 > MA200 才允许突破
TREND_MA = 200        #   趋势过滤均线窗口
STOP_ON_CLOSE = True  # ③ 收盘跌破箱底才止损（False=首版日内最低价）
PYRAMID = True        # ④ 金字塔加仓（更高箱体才加）

PRE_START_YEARS = 2   # 预加载向前多取 N 年（MA200 需 200 日，留足余量）


def _nearest_le(dates, target):
    """返回 dates 中 <= target 的最后一个下标；无则 -1。dates 须升序。"""
    i = int(np.searchsorted(dates, target, side="right")) - 1
    return i


def _asof_arr(dates, vals, target):
    """在 (dates,vals) 数组上取 target 之前最近的非空值；无则返回 None。"""
    i = _nearest_le(dates, target)
    if i < 0:
        return None
    v = vals[i]
    if v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)):
        return None
    return float(v)


def _ma(close, end, window):
    """收盘价的 window 日简单移动平均（截至 end 含）。不足返回 None。"""
    if end < window - 1:
        return None
    return float(np.mean(close[end - window + 1:end + 1]))


def load_universe_data(pool, start_date, end_date):
    """预加载股票池的 OHLC / 估值 / 成长 数据为紧凑 numpy 数组（低内存）。"""
    cons = get_index_constituents(pool, trade_date=start_date)
    if cons is None or len(cons) == 0:
        # 无时点快照时显式回退到最新名单（会打印幸存者偏差告警）。
        # 达尔瓦斯用「回测起点的固定股票池」，本就不做期中换池，故可接受；
        # 但若该指数缺历史快照，结论同样带幸存者偏差，需按告警提示补数。
        cons = get_index_constituents(pool, trade_date=start_date,
                                      allow_stale_fallback=True)
    pre_start = str(int(start_date) - PRE_START_YEARS * 10000)
    codes = sorted(c for c in cons if not str(c).startswith("688"))
    print(f"股票池 {pool} {INDEX_DISPLAY_NAME.get(pool, '')} 成分股(剔除688): {len(codes)} 只")

    conn = get_conn()
    meta = pd.read_sql_query(
        "SELECT ts_code, name, list_date FROM stock_basic WHERE ts_code IN (%s)"
        % ",".join("?" * len(codes)), conn, params=codes)
    st_set = set(meta.loc[meta["name"].str.contains("ST", na=False), "ts_code"])
    list_date = dict(zip(meta["ts_code"], meta["list_date"]))

    ohlc = pd.read_sql_query(
        "SELECT ts_code, trade_date, high, low, close FROM daily "
        "WHERE ts_code IN (%s) AND trade_date>=? AND trade_date<=?"
        % ",".join("?" * len(codes)),
        conn, params=codes + [pre_start, end_date])
    basic = pd.read_sql_query(
        "SELECT ts_code, trade_date, pe_ttm, pb, total_mv FROM daily_basic "
        "WHERE ts_code IN (%s) AND trade_date>=? AND trade_date<=?"
        % ",".join("?" * len(codes)),
        conn, params=codes + [pre_start, end_date])
    fin = pd.read_sql_query(
        "SELECT ts_code, ann_date, op_yoy, basic_eps_yoy FROM fina_indicator "
        "WHERE ts_code IN (%s)" % ",".join("?" * len(codes)),
        conn, params=codes)
    conn.close()

    price, fund, grow = {}, {}, {}
    for c, g in ohlc.groupby("ts_code"):
        g = g.sort_values("trade_date")
        price[c] = {
            "date": g["trade_date"].to_numpy(dtype=np.int64),
            "high": g["high"].to_numpy(dtype=np.float64),
            "low": g["low"].to_numpy(dtype=np.float64),
            "close": g["close"].to_numpy(dtype=np.float64),
        }
    for c, g in basic.groupby("ts_code"):
        g = g.dropna(subset=["trade_date"]).sort_values("trade_date")
        fund[c] = {
            "date": g["trade_date"].to_numpy(dtype=np.int64),
            "pe": g["pe_ttm"].to_numpy(dtype=np.float64),
            "pb": g["pb"].to_numpy(dtype=np.float64),
            "mv": g["total_mv"].to_numpy(dtype=np.float64),
        }
    for c, g in fin.groupby("ts_code"):
        g = g.dropna(subset=["ann_date"]).sort_values("ann_date")
        grow[c] = {
            "date": g["ann_date"].to_numpy(dtype=np.int64),
            "op": g["op_yoy"].to_numpy(dtype=np.float64),
            "eps": g["basic_eps_yoy"].to_numpy(dtype=np.float64),
        }
    return codes, st_set, list_date, price, fund, grow


def momentum_return(pc, prev_int, skip_days, lookback_days):
    """本地动量：close[prev_int] / close[prev_int前 skip+lookback 交易日] - 1。"""
    dates = pc["date"]
    end = _nearest_le(dates, prev_int)
    if end < skip_days + lookback_days:
        return None
    c_now = float(pc["close"][end])
    c_past = float(pc["close"][end - skip_days - lookback_days])
    if c_past <= 0:
        return None
    return c_now / c_past - 1.0


def darvas_box_arrays(high, low, close, box_pct, win, min_box_age):
    """在 high/low/close（升序，尾部即最新，含"今日"）上识别箱体突破。
    返回 (box_top, box_bottom, width) 或 None。

    关键：箱顶取【前 win 日(不含今日)】的最高高 —— 不能用含今日的窗口最高高，
    否则 box_top>=今日最高>=今日收盘，close>box_top 永不成立（初版 0 突破根因）。
      - 箱顶 = 前 win 日最高高；箱底 = 箱顶出现后的最低低
      - 要求箱顶至少 min_box_age 日前形成（确有整理）
      - 箱体宽度 <= box_pct 且 今日收盘 > 箱顶 → 向上突破信号
    """
    n = len(close)
    if n < win + 1:  # 需前 win 日 + 今日
        return None
    sub_h = high[-(win + 1):-1]   # 前 win 日最高高
    sub_l = low[-(win + 1):-1]
    top_idx = int(np.argmax(sub_h))
    if top_idx > len(sub_l) - 1 - min_box_age:
        return None  # 箱顶太新，尚无整理
    box_top = sub_h[top_idx]
    box_bottom = sub_l[top_idx + 1:].min() if top_idx + 1 < len(sub_l) else sub_l.min()
    if box_bottom <= 0 or box_top <= 0:
        return None
    width = box_top / box_bottom - 1.0
    if width > box_pct:
        return None
    if close[-1] > box_top:  # 今日收盘突破前 win 日最高高
        return (float(box_top), float(box_bottom), float(width))
    return None


def stock_ok(code, prev_int, codes, st_set, list_date, price, fund, grow,
             trend_filter=TREND_FILTER, trend_ma=TREND_MA):
    """基本面+动量预筛（价值/成长/动量/流动性/上市/ST）。通过返回 True。"""
    if code in st_set:
        return False
    ld = list_date.get(code)
    if ld is None or int(str(ld)) > prev_int - LIST_MIN_YEARS * 10000:
        return False
    pc = price.get(code)
    if pc is None:
        return False
    end = _nearest_le(pc["date"], prev_int)
    need = MOM_LOOKBACK + MOM_SKIP
    if trend_filter:
        need = max(need, trend_ma)
    if end < need:
        return False
    # ② 趋势过滤：收盘须在 MA(trend_ma) 上方（上升趋势）
    if trend_filter:
        ma = _ma(pc["close"], end, trend_ma)
        if ma is None or pc["close"][end] <= ma:
            return False
    pe = _asof_arr(fund[code]["date"], fund[code]["pe"], prev_int) if code in fund else None
    pb = _asof_arr(fund[code]["date"], fund[code]["pb"], prev_int) if code in fund else None
    mv = _asof_arr(fund[code]["date"], fund[code]["mv"], prev_int) if code in fund else None
    if pe is None or pb is None or mv is None:
        return False
    if not (0 < pe <= PE_MAX) or not (0 < pb <= PB_MAX) or mv < MV_MIN_WAN:
        return False
    op = _asof_arr(grow[code]["date"], grow[code]["op"], prev_int) if code in grow else None
    eps = _asof_arr(grow[code]["date"], grow[code]["eps"], prev_int) if code in grow else None
    if (op is None or op <= 0) and (eps is None or eps <= 0):
        return False
    mom = momentum_return(pc, prev_int, MOM_SKIP, MOM_LOOKBACK)
    if mom is None or mom <= 0:
        return False
    return True


def darvas_breakout(code, prev_int, price):
    """返回该股的达瓦斯箱体突破信号 (box_top, box_bottom, width) 或 None。"""
    pc = price.get(code)
    if pc is None:
        return None
    end = _nearest_le(pc["date"], prev_int)
    if end < BOX_WIN:
        return None
    hi = pc["high"][:end + 1]
    lo = pc["low"][:end + 1]
    cl = pc["close"][:end + 1]
    return darvas_box_arrays(hi, lo, cl, BOX_PCT, BOX_WIN, MIN_BOX_AGE)


def select_darvas_stocks(prev_td, top_n, codes, st_set, list_date, price, fund, grow,
                         trend_filter=TREND_FILTER):
    """返回 基本面合格 + 达瓦斯箱体突破 的候选 (code, mom, box_bottom) 列表（按动量降序，取 top_n）。"""
    prev_int = int(prev_td)
    cands = []
    for c in codes:
        if not stock_ok(c, prev_int, codes, st_set, list_date, price, fund, grow, trend_filter):
            continue
        sig = darvas_breakout(c, prev_int, price)
        if sig is None:
            continue
        pc = price[c]
        mom = momentum_return(pc, prev_int, MOM_SKIP, MOM_LOOKBACK)
        cands.append((c, mom, sig[1]))
    if not cands:
        return []
    cands.sort(key=lambda x: x[1], reverse=True)
    return cands[:top_n]


def run_backtest(start_date="20180101", end_date="20251231", top_n=TOP_N,
                 box_pct=BOX_PCT, pool="000906.SH",
                 trend_filter=TREND_FILTER, stop_on_close=STOP_ON_CLOSE,
                 pyramid=PYRAMID, variant="base"):
    codes, st_set, list_date, price, fund, grow = load_universe_data(pool, start_date, end_date)
    trade_dates = get_trade_dates(start_date, end_date)
    rebalance_set = set(get_monthly_5th_trading_days(trade_dates))
    date_idx = {d: i for i, d in enumerate(trade_dates)}
    print(f"交易日 {len(trade_dates)}，调仓 {len(rebalance_set)} 次；"
          f"参数 box_pct={box_pct:.0%} top_n={top_n} "
          f"trend={trend_filter}(MA{TREND_MA}) stop_on_close={stop_on_close} pyramid={pyramid}")

    positions = {}
    cash = INIT_CAPITAL
    daily_vals = []
    trades = []
    stop_count = 0
    name_cache = {}

    def get_name(code):
        if code not in name_cache:
            name_cache[code] = get_stock_name(code)
        return name_cache[code]

    for i, td in enumerate(trade_dates):
        td_int = int(td)
        # ── 移动止损检查（开盘执行昨日标记的止损）──
        for code in list(positions.keys()):
            pos = positions[code]
            if pos.get("stop_triggered", False):
                op = get_open_price(code, td)
                if op is not None:
                    fee = calc_fee("sell", op, pos["shares"])
                    cash += pos["shares"] * op - fee
                    stop_count += 1
                    trades.append({"date": td, "action": "SELL", "code": code,
                                   "name": get_name(code), "price": op,
                                   "shares": pos["shares"], "reason": "darvas_stop"})
                    del positions[code]
                continue
            pc = price.get(code)
            if pc is None:
                continue
            end = _nearest_le(pc["date"], td_int)
            if end < 0:
                continue
            hi = float(pc["high"][end])
            lo = float(pc["low"][end])
            cl = float(pc["close"][end])
            if hi > pos["box_top"]:
                pos["box_top"] = hi
                pos["box_top_pos"] = end
            bs = pos["box_top_pos"]
            if bs + 1 <= end:
                new_bottom = float(pc["low"][bs + 1:end + 1].min())
                if pos["box_top"] > 0 and (pos["box_top"] / new_bottom - 1) <= box_pct:
                    pos["stop"] = max(pos["stop"], new_bottom)
            # ③ 止损触发口径：收盘跌破箱底（首版用日内最低价）
            trigger_price = cl if stop_on_close else lo
            if trigger_price <= pos["stop"]:
                pos["stop_triggered"] = True

        # ── 调仓日 ──
        if td in rebalance_set:
            prev_td = trade_dates[i - 1] if i > 0 else td
            prev_int = int(prev_td)
            # 1) 持有股若基本面/动量/趋势恶化（达瓦斯"判断失效"）则卖出；趋势中赢家继续持有
            for code in list(positions.keys()):
                if not stock_ok(code, prev_int, codes, st_set, list_date, price, fund, grow, trend_filter):
                    op = get_open_price(code, td)
                    if op is None:
                        continue
                    pos = positions[code]
                    fee = calc_fee("sell", op, pos["shares"])
                    cash += pos["shares"] * op - fee
                    trades.append({"date": td, "action": "SELL", "code": code,
                                   "name": get_name(code), "price": op,
                                   "shares": pos["shares"], "reason": "fund_deteriorate"})
                    del positions[code]
            # 2) 选突破候选（基本面合格 + 达瓦斯箱体突破），买入至 top_n
            selected = select_darvas_stocks(prev_td, top_n, codes, st_set, list_date,
                                            price, fund, grow, trend_filter)
            new_to_buy = [c for c, _, _ in selected if c not in positions]
            if new_to_buy and len(positions) < top_n:
                room = top_n - len(positions)
                new_to_buy = new_to_buy[:room]
                cash_per = cash / len(new_to_buy)
                for c in new_to_buy:
                    op = get_open_price(c, td)
                    if op is None:
                        continue
                    shares = int(cash_per / op / 100) * 100
                    if shares < 100:
                        continue
                    cost = shares * op
                    fee = calc_fee("buy", op, shares)
                    if cost + fee <= cash:
                        cash -= cost + fee
                        sig = darvas_breakout(c, prev_int, price)
                        pc = price[c]
                        end = _nearest_le(pc["date"], prev_int)
                        box_top = float(sig[0]) if sig else float(op)
                        box_bottom = float(sig[1]) if sig else float(op)
                        positions[c] = {
                            "shares": shares, "entry_shares": shares, "buy_price": op,
                            "buy_idx": i, "box_top": box_top, "box_top_pos": end,
                            "stop": box_bottom, "stop_triggered": False,
                        }
                        trades.append({"date": td, "action": "BUY", "code": c,
                                       "name": get_name(c), "price": op,
                                       "shares": shares, "reason": "darvas_buy"})
            # 3) ④ 金字塔加仓：持仓股走出更高箱体(新高箱顶)时加仓（≤原仓0.6，总仓≤2x）
            if pyramid:
                for code in list(positions.keys()):
                    pos = positions[code]
                    if pos.get("stop_triggered", False):
                        continue
                    sig = darvas_breakout(code, prev_int, price)
                    if sig is None:
                        continue
                    new_top, new_bottom, _ = sig
                    if new_top <= pos["box_top"] * 1.002:
                        continue  # 未形成更高箱体
                    op = get_open_price(code, td)
                    if op is None:
                        continue
                    entry = pos.get("entry_shares", pos["shares"])
                    allowed = int(entry * 2) - pos["shares"]      # 总仓上限 2x
                    max_add = int(entry * 0.6 / 100) * 100        # 单次≤0.6原仓
                    add_shares = int(min(allowed, max_add) / 100) * 100
                    if add_shares < 100:
                        continue
                    cost = add_shares * op
                    fee = calc_fee("buy", op, add_shares)
                    if cost + fee <= cash:
                        cash -= cost + fee
                        pc = price[code]
                        end = _nearest_le(pc["date"], prev_int)
                        pos["shares"] += add_shares
                        pos["box_top"] = new_top
                        pos["box_top_pos"] = end
                        pos["stop"] = max(pos["stop"], new_bottom)
                        trades.append({"date": td, "action": "BUY", "code": code,
                                       "name": get_name(code), "price": op,
                                       "shares": add_shares, "reason": "darvas_pyramid"})
            if not positions and not new_to_buy:
                print(f"  调仓日 {td}：无候选，空仓等待")

        # ── 每日市值 ──
        total = cash
        for code, pos in list(positions.items()):
            p = get_price(code, td)
            if p is not None:
                total += pos["shares"] * p
        daily_vals.append({"date": td, "value": total})

    # 结束平仓
    if trade_dates:
        last = trade_dates[-1]
        for code in list(positions.keys()):
            p = get_price(code, last)
            if p is not None:
                pos = positions[code]
                fee = calc_fee("sell", p, pos["shares"])
                cash += pos["shares"] * p - fee
                trades.append({"date": last, "action": "SELL", "code": code,
                               "name": get_name(code), "price": p,
                               "shares": pos["shares"], "reason": "end"})
                del positions[code]

    # ── 绩效 ──
    final = cash
    total_ret = (final / INIT_CAPITAL - 1) * 100
    years = len(trade_dates) / 252
    ann = ((final / INIT_CAPITAL) ** (1 / years) - 1) * 100 if years > 0 else 0
    vals = np.array([d["value"] for d in daily_vals])
    cummax = np.maximum.accumulate(vals)
    safe = np.where(cummax == 0, 1, cummax)
    mdd = float(np.min((vals - cummax) / safe)) * 100
    rets = np.diff(vals) / np.where(vals[:-1] == 0, 1, vals[:-1])
    sharpe = (np.mean(rets) * 252 - 0.025) / (np.std(rets) * np.sqrt(252)) if np.std(rets) > 0 else 0

    conn = get_conn()
    bs = pd.read_sql_query("SELECT close FROM index_daily WHERE ts_code=? AND trade_date>=? "
                           "ORDER BY trade_date ASC LIMIT 1", conn, params=(pool, trade_dates[0]))
    be = pd.read_sql_query("SELECT close FROM index_daily WHERE ts_code=? AND trade_date<=? "
                           "ORDER BY trade_date DESC LIMIT 1", conn, params=(pool, trade_dates[-1]))
    conn.close()
    idx_ret = 0.0
    if len(bs) > 0 and len(be) > 0:
        idx_ret = (float(be.iloc[0]["close"]) / float(bs.iloc[0]["close"]) - 1) * 100

    tag = variant.replace(" ", "_") or "base"
    print("\n" + "=" * 64)
    print(f"达瓦斯箱体个股策略(v2 {tag})  区间 {start_date}~{end_date}  池={INDEX_DISPLAY_NAME.get(pool, pool)}")
    print("=" * 64)
    print(f"  初始/最终: {INIT_CAPITAL:,.0f} -> {final:,.0f}  盈亏 {final-INIT_CAPITAL:+,.0f}")
    print(f"  总收益: {total_ret:+.2f}%   年化: {ann:+.2f}%   最大回撤: {mdd:.2f}%   夏普: {sharpe:.2f}")
    print(f"  基准({INDEX_DISPLAY_NAME.get(pool, pool)}): {idx_ret:+.2f}%   超额: {total_ret-idx_ret:+.2f}%")
    print(f"  交易次数: {len(trades)}   止损次数: {stop_count}")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "results", "darvas")
    os.makedirs(out_dir, exist_ok=True)
    summ = os.path.join(out_dir, f"darvas_{pool}_{start_date}_{end_date}_box{int(box_pct*100)}_{tag}.csv")
    with open(summ, "w", encoding="utf-8") as f:
        f.write("metric,value\n")
        f.write(f"start,{start_date}\nend,{end_date}\npool,{pool}\n")
        f.write(f"box_pct,{box_pct}\ntop_n,{top_n}\nvariant,{tag}\n")
        f.write(f"trend_filter,{int(trend_filter)}\nstop_on_close,{int(stop_on_close)}\npyramid,{int(pyramid)}\n")
        f.write(f"init_capital,{INIT_CAPITAL}\nfinal,{final:.2f}\n")
        f.write(f"total_return,{total_ret:.4f}\nannual,{ann:.4f}\nmax_drawdown,{mdd:.4f}\n")
        f.write(f"sharpe,{sharpe:.4f}\nindex_return,{idx_ret:.4f}\n")
        f.write(f"excess,{total_ret-idx_ret:.4f}\ntrades,{len(trades)}\nstop_count,{stop_count}\n")
    tlog = os.path.join(out_dir, f"darvas_trades_{pool}_{start_date}_{end_date}_box{int(box_pct*100)}_{tag}.csv")
    pd.DataFrame(trades).to_csv(tlog, index=False, encoding="utf-8")
    print(f"  结果: {summ}\n  成交: {tlog}")
    return {"total_return": total_ret, "annual": ann, "mdd": mdd, "sharpe": sharpe,
            "index_return": idx_ret, "excess": total_ret - idx_ret, "trades": len(trades),
            "stop_count": stop_count}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20180101")
    ap.add_argument("--end", default="20251231")
    ap.add_argument("--top_n", type=int, default=TOP_N)
    ap.add_argument("--box_pct", type=float, default=BOX_PCT)
    ap.add_argument("--pool", default="000906.SH")
    ap.add_argument("--variant", default="base")
    ap.add_argument("--no-trend", dest="trend_filter", action="store_false",
                    help="关闭 MA200 趋势过滤")
    ap.add_argument("--stop-intraday", dest="stop_on_close", action="store_false",
                    help="用日内最低价触发止损(首版口径)")
    ap.add_argument("--no-pyramid", dest="pyramid", action="store_false",
                    help="关闭金字塔加仓")
    args = ap.parse_args()
    run_backtest(args.start, args.end, args.top_n, args.box_pct, args.pool,
                 trend_filter=args.trend_filter, stop_on_close=args.stop_on_close,
                 pyramid=args.pyramid, variant=args.variant)
