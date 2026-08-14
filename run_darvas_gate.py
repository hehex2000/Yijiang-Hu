# -*- coding: utf-8 -*-
"""
达瓦斯箱体 · 作为强势因子的「退出/减仓闸门」（方向 B · 风控复用版）
================================================================
把白名单 UP 主 Jim(BV1jTM16yEoU) 的达瓦斯箱体**只做风控、不做选股**：

  · 基础策略（神奇公式 value / 动量 momentum）按月选股 → 负责「进入」
  · 达瓦斯闸门（日内检查）只负责「退出」：
        GATE_OPEN（继续持有）  ⇔  close > MA(gate_ma)  且  close >= 箱底(stop)
        GATE_CLOSED（减仓到现金）⇔  close <= MA  或  close < 箱底

设计要点（来自方向B v2 消融教训）：
  · 闸门是**非对称**的——只卖不买，因此不会像 v2「MA200 进入过滤」那样「迟入踏空」，
    只会在下跌/破位时减仓，理论上净降回撤。
  · 箱底移动止损沿用 v2 最佳结构（宽箱≈20% 才有效）：新高刷新箱顶、箱底随之上移锁利。
  · 进场只由基础策略决定；闸门在进场时只做「是否当下处于突破持仓」的二次过滤
    （close>MA 且 close>=箱底 才允许建仓，否则等下一调仓日）。

费用：完全复用引擎 calc_fee（佣金万2.5最低5元 + 印花税历史分段 + 滑点0.1%损失向）。
基准：中证800 000906.SH。对比口径：同一引擎 `--no-gate`（纯基础策略） vs 加闸门。

运行（venv_ml python）：
  run_darvas_gate.py --base value   --no-gate            # 神奇公式 纯基线
  run_darvas_gate.py --base value                       # 神奇公式 + 达瓦斯闸门
  run_darvas_gate.py --base momentum --no-gate           # 动量 纯基线
  run_darvas_gate.py --base momentum                     # 动量 + 达瓦斯闸门
  run_darvas_gate.py --base value --gate_ma 20           # 闸门均线 20 日
  run_darvas_gate.py --base value --no-ma-gate           # 仅箱底止损(关MA闸门)
  run_darvas_gate.py --base value --no-box-gate          # 仅MA闸门(关箱底止损)
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
    INIT_CAPITAL, INDEX_DISPLAY_NAME, select_by_method,
)

# ── 基础策略 + 闸门参数 ──
BASE = "value"          # "value"(神奇公式) / "momentum"(动量)
TOP_N = 10              # 基础策略持仓数（等权重）
GATE_MA = 60            # 趋势闸门均线窗口（close>MA 才持有）
BOX_WIN = 60            # 箱底移动止损识别窗口（交易日）
BOX_PCT = 0.20          # 箱体宽度容忍（v2 最佳≈20%）
PRE_START_YEARS = 2     # 预加载向前多取 N 年（MA 需历史）


def _nearest_le(dates, target):
    """返回 dates 中 <= target 的最后一个下标；无则 -1。dates 须升序。"""
    i = int(np.searchsorted(dates, target, side="right")) - 1
    return i


def _ma(close, end, window):
    """收盘价的 window 日简单移动平均（截至 end 含）。不足返回 None。"""
    if end < window - 1:
        return None
    return float(np.mean(close[end - window + 1:end + 1]))


def _load_ohlc(codes, pre_start, end_date):
    """预加载指定股票的 OHLC（high/low/close）为紧凑 numpy 数组（低内存）。"""
    conn = get_conn()
    ohlc = pd.read_sql_query(
        "SELECT ts_code, trade_date, high, low, close FROM daily "
        "WHERE ts_code IN (%s) AND trade_date>=? AND trade_date<=?"
        % ",".join("?" * len(codes)),
        conn, params=codes + [pre_start, end_date])
    conn.close()
    price = {}
    for c, g in ohlc.groupby("ts_code"):
        g = g.sort_values("trade_date")
        price[c] = {
            "date": g["trade_date"].to_numpy(dtype=np.int64),
            "high": g["high"].to_numpy(dtype=np.float64),
            "low": g["low"].to_numpy(dtype=np.float64),
            "close": g["close"].to_numpy(dtype=np.float64),
        }
    return price


def init_box(pc, prev_int, box_win):
    """用进场前 box_win 日数据初始化箱体 (box_top, box_bottom, box_top_pos)。"""
    end = _nearest_le(pc["date"], prev_int)
    if end < box_win:
        return None
    sub_h = pc["high"][end - box_win + 1:end + 1]
    sub_l = pc["low"][end - box_win + 1:end + 1]
    top_idx = int(np.argmax(sub_h))
    box_top = float(sub_h[top_idx])
    box_bottom = float(sub_l[top_idx + 1:].min()) if top_idx + 1 < len(sub_l) else float(sub_l.min())
    box_top_pos = end - box_win + 1 + top_idx
    return box_top, box_bottom, box_top_pos


def gate_open_today(pc, end, close, gate_ma, box_pct, box_win,
                    use_ma_gate, use_box_gate, box_top=None, box_top_pos=None, stop=None):
    """判断当日闸门是否「开」（可持有/可建仓）。返回 (open_bool, box_top, box_top_pos, stop)。"""
    hi = float(pc["high"][end]); lo = float(pc["low"][end]); cl = float(pc["close"][end])
    # 箱底移动止损（新高刷新箱顶，箱底随上移）
    if use_box_gate and box_top is not None:
        if hi > box_top:
            box_top = hi
            box_top_pos = end
        if box_top_pos + 1 <= end:
            nb = float(pc["low"][box_top_pos + 1:end + 1].min())
            if box_top > 0 and (box_top / nb - 1) <= box_pct:
                stop = max(stop, nb)
    open_flag = True
    if use_ma_gate:
        ma = _ma(close, end, gate_ma)
        if ma is None or cl <= ma:
            open_flag = False
    if use_box_gate and open_flag and stop is not None:
        if cl < stop:
            open_flag = False
    return open_flag, box_top, box_top_pos, stop


def get_base_selection(base, td_int, top_n):
    """基础策略选股：返回 ts_code 列表。"""
    td = str(td_int)
    if base == "momentum":
        df = select_by_method("momentum", td, top_n=top_n, lookback_months=6)
    elif base == "div_low_vol_quality":
        # 红利低波 + 质量过滤（B站策略口径）
        df = select_by_method("div_low_vol", td, top_n=top_n, div_quality_filter=True)
    else:
        # value / div_low_vol / div_growth 都直接走 select_by_method
        df = select_by_method(base, td, top_n=top_n)
    if df is None or len(df) == 0:
        return []
    return df["ts_code"].tolist()


def run_backtest(start_date="20180101", end_date="20251231", base=BASE, top_n=TOP_N,
                 gate_ma=GATE_MA, box_win=BOX_WIN, box_pct=BOX_PCT,
                 use_ma_gate=True, use_box_gate=True, no_gate=False,
                 reentry=False, pool="000906.SH"):
    if no_gate:
        use_ma_gate = use_box_gate = False

    trade_dates = get_trade_dates(start_date, end_date)
    rebalance_set = set(get_monthly_5th_trading_days(trade_dates))
    pre_start = str(int(start_date) - PRE_START_YEARS * 10000)

    # ── 1) 收集基础策略选过的全部股票（用于预加载 OHLC）──
    all_codes = set()
    for td in trade_dates:
        if td in rebalance_set:
            prev_int = int(trade_dates[trade_dates.index(td) - 1]) if trade_dates.index(td) > 0 else int(td)
            all_codes.update(get_base_selection(base, prev_int, top_n))
    all_codes = sorted(all_codes)
    print(f"基础策略={base} 调仓 {len(rebalance_set)} 次；出现过 {len(all_codes)} 只候选股")
    price = _load_ohlc(all_codes, pre_start, end_date)

    # ── 2) 模拟 ──
    positions = {}          # ts_code -> {shares,buy_price,box_top,box_top_pos,stop}
    pending_exits = set()   # 当日闸门关闭、次日开盘执行的减仓
    pending_entries = set() # 方向C：闸门减仓后价格重回 MA，次日开盘回补
    watch = {}              # code -> 箱底状态（被闸门减仓后等待回补）
    last_target_set = set() # 最近一次调仓基础策略选中的股票（避免与 watch 重复建仓）
    cash = INIT_CAPITAL
    daily_vals = []
    trades = []
    gate_exit_count = 0
    reentry_count = 0
    name_cache = {}

    def get_name(code):
        if code not in name_cache:
            name_cache[code] = get_stock_name(code)
        return name_cache[code]

    def cur_equity():
        eq = cash
        for code, pos in positions.items():
            pc = price.get(code)
            if pc is None:
                continue
            end = _nearest_le(pc["date"], _td_int_cur)
            if end >= 0:
                eq += pos["shares"] * float(pc["close"][end])
        return eq

    _td_int_cur = 0
    for i, td in enumerate(trade_dates):
        _td_int_cur = int(td)

        # (a) 执行上一日标记的闸门减仓（次日开盘）
        for code in list(pending_exits):
            op = get_open_price(code, td)
            if op is not None and code in positions:
                pos = positions[code]
                fee = calc_fee("sell", op, pos["shares"])
                cash += pos["shares"] * op - fee
                gate_exit_count += 1
                trades.append({"date": td, "action": "SELL", "code": code,
                               "name": get_name(code), "price": op,
                               "shares": pos["shares"], "reason": "gate_exit"})
                del positions[code]
            pending_exits.discard(code)

        # (a2) 执行上一日标记的闸门回补（方向C：次日开盘买回）
        for code in list(pending_entries):
            op = get_open_price(code, td)
            pc = price.get(code)
            if op is not None and code not in positions and pc is not None:
                end = _nearest_le(pc["date"], _td_int_cur)
                eq = cur_equity()
                target_val = eq / top_n
                shares = int(target_val / op / 100) * 100
                if shares >= 100:
                    cost = shares * op
                    fee = calc_fee("buy", op, shares)
                    if cost + fee <= cash:
                        cash -= cost + fee
                        w = watch.get(code, {})
                        bt = w.get("box_top"); btp = w.get("box_top_pos"); st = w.get("stop")
                        if bt is None:
                            ib = init_box(pc, int(trade_dates[i - 1]), box_win)
                            if ib is not None:
                                bt, btp, st = ib
                            else:
                                bt, btp, st = op, end, op * 0.9
                        positions[code] = {"shares": shares, "buy_price": op,
                                           "box_top": bt, "box_top_pos": btp, "stop": st}
                        reentry_count += 1
                        trades.append({"date": td, "action": "BUY", "code": code,
                                       "name": get_name(code), "price": op,
                                       "shares": shares, "reason": "gate_reentry"})
            watch.pop(code, None)
            pending_entries.discard(code)

        # (b) 每日闸门检查（用当日收盘，次日开盘执行）
        for code in list(positions.keys()):
            pc = price.get(code)
            if pc is None:
                continue
            end = _nearest_le(pc["date"], _td_int_cur)
            if end < 0:
                continue
            pos = positions[code]
            open_flag, bt, btp, st = gate_open_today(
                pc, end, pc["close"], gate_ma, box_pct, box_win,
                use_ma_gate, use_box_gate, pos["box_top"], pos["box_top_pos"], pos["stop"])
            pos["box_top"], pos["box_top_pos"], pos["stop"] = bt, btp, st
            if not open_flag and code not in pending_exits:
                pending_exits.add(code)
                watch[code] = {"box_top": bt, "box_top_pos": btp, "stop": st}

        # (b2) 方向C：被闸门减仓的股票，若价格重回 MA 则标记次日回补
        if reentry:
            for code in list(watch.keys()):
                if code in positions or code in pending_entries or code in pending_exits:
                    continue
                if code in last_target_set:  # 交给月度调仓处理，避免重复建仓
                    continue
                pc = price.get(code)
                if pc is None:
                    continue
                end = _nearest_le(pc["date"], _td_int_cur)
                if end < 0:
                    continue
                w = watch[code]
                open_flag, bt, btp, st = gate_open_today(
                    pc, end, pc["close"], gate_ma, box_pct, box_win,
                    use_ma_gate, use_box_gate, w["box_top"], w["box_top_pos"], w["stop"])
                w["box_top"], w["box_top_pos"], w["stop"] = bt, btp, st
                if open_flag and code not in pending_entries:
                    pending_entries.add(code)

        # (c) 调仓日：基础策略进入 / 退出
        if td in rebalance_set:
            prev_int = int(trade_dates[i - 1]) if i > 0 else _td_int_cur
            target = get_base_selection(base, prev_int, top_n)
            target_set = set(target)
            last_target_set = target_set
            # 基础策略剔除的 → 卖出
            for code in list(positions.keys()):
                if code not in target_set:
                    op = get_open_price(code, td)
                    if op is not None:
                        pos = positions[code]
                        fee = calc_fee("sell", op, pos["shares"])
                        cash += pos["shares"] * op - fee
                        trades.append({"date": td, "action": "SELL", "code": code,
                                       "name": get_name(code), "price": op,
                                       "shares": pos["shares"], "reason": "rebalance_out"})
                    del positions[code]
                    pending_exits.discard(code)
            # 基础策略新选的 → 闸门开才建仓
            new_codes = [c for c in target if c not in positions]
            for code in new_codes:
                pc = price.get(code)
                if pc is None:
                    continue
                end = _nearest_le(pc["date"], prev_int)
                if end < 0:
                    continue
                open_flag, bt, btp, st = gate_open_today(
                    pc, end, pc["close"], gate_ma, box_pct, box_win,
                    use_ma_gate, use_box_gate)
                if not open_flag:
                    continue  # 闸门关：等下一调仓日更好入场
                op = get_open_price(code, td)
                if op is None:
                    continue
                eq = cur_equity()
                target_val = eq / top_n
                shares = int(target_val / op / 100) * 100
                if shares < 100:
                    continue
                cost = shares * op
                fee = calc_fee("buy", op, shares)
                if cost + fee <= cash:
                    cash -= cost + fee
                    if bt is None:  # 无前置箱体，用保守止损
                        bt, btp, st = op, end, op * 0.9
                    positions[code] = {"shares": shares, "buy_price": op,
                                       "box_top": bt, "box_top_pos": btp, "stop": st}
                    trades.append({"date": td, "action": "BUY", "code": code,
                                   "name": get_name(code), "price": op,
                                   "shares": shares, "reason": "base_select"})
                    watch.pop(code, None)

        # (d) 每日市值
        total = cash
        for code, pos in list(positions.items()):
            pc = price.get(code)
            if pc is None:
                p = get_price(code, td)
                if p is not None:
                    total += pos["shares"] * p
            else:
                end = _nearest_le(pc["date"], _td_int_cur)
                if end >= 0:
                    total += pos["shares"] * float(pc["close"][end])
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

    # ── 3) 绩效 ──
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

    # 闸门配置标签
    if no_gate:
        tag = "nogate"
    else:
        parts = []
        parts.append(f"ma{gate_ma}" if use_ma_gate else "noma")
        parts.append(f"box{int(box_pct*100)}" if use_box_gate else "nobox")
        if reentry:
            parts.append("re")
        tag = "_".join(parts)

    print("\n" + "=" * 64)
    print(f"达瓦斯闸门 · 基础={base} · {tag}   区间 {start_date}~{end_date}")
    print("=" * 64)
    print(f"  初始/最终: {INIT_CAPITAL:,.0f} -> {final:,.0f}  盈亏 {final-INIT_CAPITAL:+,.0f}")
    print(f"  总收益: {total_ret:+.2f}%   年化: {ann:+.2f}%   最大回撤: {mdd:.2f}%   夏普: {sharpe:.2f}")
    print(f"  基准(中证800): {idx_ret:+.2f}%   超额: {total_ret-idx_ret:+.2f}%")
    print(f"  交易次数: {len(trades)}   闸门减仓次数: {gate_exit_count}   闸門回补次数: {reentry_count}")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "results", "darvas")
    os.makedirs(out_dir, exist_ok=True)
    summ = os.path.join(out_dir, f"gate_{base}_{pool}_{start_date}_{end_date}_{tag}.csv")
    with open(summ, "w", encoding="utf-8") as f:
        f.write("metric,value\n")
        f.write(f"start,{start_date}\nend,{end_date}\npool,{pool}\nbase,{base}\n")
        f.write(f"top_n,{top_n}\ngate_ma,{gate_ma}\nbox_win,{box_win}\nbox_pct,{box_pct}\n")
        f.write(f"use_ma_gate,{int(use_ma_gate)}\nuse_box_gate,{int(use_box_gate)}\ntag,{tag}\n")
        f.write(f"init_capital,{INIT_CAPITAL}\nfinal,{final:.2f}\n")
        f.write(f"total_return,{total_ret:.4f}\nannual,{ann:.4f}\nmax_drawdown,{mdd:.4f}\n")
        f.write(f"sharpe,{sharpe:.4f}\nindex_return,{idx_ret:.4f}\n")
        f.write(f"excess,{total_ret-idx_ret:.4f}\ntrades,{len(trades)}\ngate_exit_count,{gate_exit_count}\n")
        f.write(f"reentry_count,{reentry_count}\n")
    tlog = os.path.join(out_dir, f"gate_trades_{base}_{pool}_{start_date}_{end_date}_{tag}.csv")
    pd.DataFrame(trades).to_csv(tlog, index=False, encoding="utf-8")
    print(f"  结果: {summ}\n  成交: {tlog}")
    return {"base": base, "tag": tag, "total_return": total_ret, "annual": ann,
            "mdd": mdd, "sharpe": sharpe, "index_return": idx_ret,
            "excess": total_ret - idx_ret, "trades": len(trades),
            "gate_exit_count": gate_exit_count}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20180101")
    ap.add_argument("--end", default="20251231")
    ap.add_argument("--base", default=BASE,
                    choices=["value", "momentum", "div_low_vol", "div_growth", "div_low_vol_quality"])
    ap.add_argument("--top_n", type=int, default=TOP_N)
    ap.add_argument("--gate_ma", type=int, default=GATE_MA)
    ap.add_argument("--box_win", type=int, default=BOX_WIN)
    ap.add_argument("--box_pct", type=float, default=BOX_PCT)
    ap.add_argument("--no-ma-gate", dest="use_ma_gate", action="store_false",
                    help="关闭 MA 趋势闸门（仅箱底止损）")
    ap.add_argument("--no-box-gate", dest="use_box_gate", action="store_false",
                    help="关闭箱底移动止损（仅 MA 闸门）")
    ap.add_argument("--no-gate", action="store_true",
                    help="纯基础策略（无达瓦斯闸门），作为对比基线")
    ap.add_argument("--reentry", action="store_true",
                    help="方向C：闸门减仓后价格重回 MA 即次日回补（消除卖低买高时滞）")
    args = ap.parse_args()
    run_backtest(args.start, args.end, args.base, args.top_n, args.gate_ma,
                 args.box_win, args.box_pct, args.use_ma_gate, args.use_box_gate,
                 args.no_gate, args.reentry)
