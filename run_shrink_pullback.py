# -*- coding: utf-8 -*-
"""
缩量回踩战法 · 事件驱动回测（跟 Jim 学量化 BV1PfuR69EdY 方法论落地）
=================================================================
把白名单 UP 主 Jim 的「缩量回踩五步识别框架」落地为 A 股个股策略：
  ① 看趋势  ② 定突破  ③ 算回落  ④ 等确认  ⑤ 做分类
视频只给"识别框架"不给买卖/回测结果 -> 本脚本补全：
  入场：④确认日 T 收盘信号 -> T+1 开盘买入
  离场：有效跌破突破位(R) / 上升结构改变(破中期均线或中期均线拐头) /
        回到阶段高点止盈 / 超最大持有期；涨停买不进·跌停卖不出则次日重试
  费用：复用引擎 calc_fee（佣金万2.5+印花历史分段+滑点0.1%损失向）

⚠️ 前视纪律（吸取 ma5 策略教训）：
  - 突破位 R 在突破日 T_b 即用 T_b 之前数据固定，之后不再移动。
  - 所有信号条件在 T 日收盘评估，买入/卖出一律 T+1 开盘执行。
  - 缩量比率只用 T 日及之前量能，绝不用 T+1 量能排序/决策。

基准：股票池对应指数（默认 中证1000 000852.SH）。
运行：
  run_shrink_pullback.py --start 20180101 --end 20260630 --pool 000852.SH
  run_shrink_pullback.py --confirm range --pool 000852.SH
  run_shrink_pullback.py --confirm vol  --pool 000852.SH
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_monthly_rebalance import (
    get_conn, get_price, get_open_price, calc_fee, get_trade_dates,
    get_index_constituents, get_stock_name, INIT_CAPITAL, INDEX_DISPLAY_NAME,
)

# ── 参数（顶部可调，全部"实验前固定"，对应视频要求）──
MA_SHORT = 20
MA_MID = 60
MID_SLOPE_LOOK = 20          # 中期均线拐头判定：MA_MID[T] > MA_MID[T-MID_SLOPE_LOOK]
RESIST_WIN = 60              # ② 突破=收盘创 RESIST_WIN 日新高（越过前期未突破位）
VBASE_WIN = 20              # 突破前基准量能窗口（T_b 之前 VBASE_WIN 日）
PULLBACK_MIN = 0.03         # ③ 最小回踩深度（太浅=没真回踩）
PULLBACK_MAX = 0.18         # ③ 最大回踩深度（超过=结构改变/反转）
VOL_RATIO_MAX = 0.70        # ③ 缩量比率上限（回落量能/突破前量能）
BREAKOUT_MAX_GAP = 60       # ② 突破日到信号日最大间隔（交易日）
CONFIRM = "ma"              # ④ 确认方式：ma=回短期均线上方 / range=超前日波动范围 / vol=量能恢复
TP_AT_STAGE_HIGH = True     # 回到阶段高点止盈
MAX_HOLD = 120              # 最大持有交易日
TOP_N = 10                  # 等权持仓上限
MV_MIN_WAN = 300000.0       # 流动性下限（总市值，万元）30亿
LIST_MIN_YEARS = 1
PRE_START_YEARS = 2


def _ma_array(close, window):
    """window 日简单移动平均（向量化），不足返回 nan。"""
    n = len(close)
    out = np.full(n, np.nan)
    if n < window:
        return out
    csum = np.concatenate(([0.0], np.cumsum(close.astype(np.float64))))
    ma = (csum[window:] - csum[:-window]) / window
    out[window - 1:] = ma
    return out


def prep_arrays(dates, high, low, close, vol):
    """预计算 MA / 中期拐头 / 突破检测 / 冻结的 R 与 V_base。全部只用"当日及之前"数据。"""
    n = len(close)
    ma_s = _ma_array(close, MA_SHORT)
    ma_m = _ma_array(close, MA_MID)
    mid_up = np.zeros(n, dtype=bool)
    for i in range(MA_MID + MID_SLOPE_LOOK - 1, n):
        if not np.isnan(ma_m[i]) and not np.isnan(ma_m[i - MID_SLOPE_LOOK]):
            mid_up[i] = ma_m[i] > ma_m[i - MID_SLOPE_LOOK]

    # 前期 RESIST_WIN 日最高收（不含当日）
    prior_resist = np.full(n, np.nan)
    if n >= RESIST_WIN + 1:
        s = pd.Series(close.astype(np.float64))
        pr = s.rolling(RESIST_WIN).max().shift(1)
        prior_resist = pr.to_numpy()

    # 突破日：close 创 RESIST_WIN 日新高（越过前期未突破位）
    is_breakout = np.zeros(n, dtype=bool)
    for i in range(RESIST_WIN, n):
        if not np.isnan(prior_resist[i]) and close[i] > prior_resist[i]:
            is_breakout[i] = True

    # 每个信号日 T 对应"之前最近的突破日"（冻结构突破位/基准量能）
    btb = np.full(n, -1, dtype=np.int64)
    last_b = -1
    R_at = np.full(n, np.nan)
    Vbase_at = np.full(n, np.nan)
    for i in range(n):
        btb[i] = last_b
        if is_breakout[i]:
            R_at[i] = prior_resist[i]
            lo = max(0, i - VBASE_WIN)
            Vbase_at[i] = float(np.mean(vol[lo:i])) if i > 0 else np.nan
            last_b = i

    return {
        "ma_s": ma_s, "ma_m": ma_m, "mid_up": mid_up,
        "btb": btb, "R_at": R_at, "Vbase_at": Vbase_at,
        "high": high, "low": low, "close": close, "vol": vol,
        "date": dates,
    }


def load_universe_data(pool, start_date, end_date):
    """预加载股票池 OHLCV + 市值，并预计算 MA/突破数组。"""
    cons = get_index_constituents(pool, trade_date=start_date)
    if cons is None or len(cons) == 0:
        cons = get_index_constituents(pool, trade_date=start_date, allow_stale_fallback=True)
    pre_start = str(int(start_date) - PRE_START_YEARS * 10000)
    codes = sorted(cons)
    print(f"股票池 {pool} {INDEX_DISPLAY_NAME.get(pool, '')} 成分股(剔除.BJ): {len(codes)} 只")

    conn = get_conn()
    meta = pd.read_sql_query(
        "SELECT ts_code, name, list_date FROM stock_basic WHERE ts_code IN (%s)"
        % ",".join("?" * len(codes)), conn, params=codes)
    st_set = set(meta.loc[meta["name"].str.contains("ST", na=False), "ts_code"])
    list_date = dict(zip(meta["ts_code"], meta["list_date"]))
    ohlc = pd.read_sql_query(
        "SELECT ts_code, trade_date, high, low, close, vol FROM daily "
        "WHERE ts_code IN (%s) AND trade_date>=? AND trade_date<=?"
        % ",".join("?" * len(codes)),
        conn, params=codes + [pre_start, end_date])
    basic = pd.read_sql_query(
        "SELECT ts_code, trade_date, total_mv FROM daily_basic "
        "WHERE ts_code IN (%s) AND trade_date>=? AND trade_date<=?"
        % ",".join("?" * len(codes)),
        conn, params=codes + [pre_start, end_date])
    conn.close()

    price, fund = {}, {}
    for c, g in ohlc.groupby("ts_code"):
        g = g.sort_values("trade_date")
        price[c] = {
            "date": g["trade_date"].to_numpy(dtype=np.int64),
            "high": g["high"].to_numpy(dtype=np.float64),
            "low": g["low"].to_numpy(dtype=np.float64),
            "close": g["close"].to_numpy(dtype=np.float64),
            "vol": g["vol"].to_numpy(dtype=np.float64),
        }
    for c, g in basic.groupby("ts_code"):
        g = g.dropna(subset=["trade_date"]).sort_values("trade_date")
        fund[c] = {
            "date": g["trade_date"].to_numpy(dtype=np.int64),
            "mv": g["total_mv"].to_numpy(dtype=np.float64),
        }

    prep = {}
    for c in codes:
        if c not in price or c not in fund:
            continue
        pc, pf = price[c], fund[c]
        # 对齐日期
        di = np.searchsorted(pc["date"], pf["date"])
        di = di[di < len(pc["date"])]
        if len(di) == 0:
            continue
        mv = pf["mv"][np.searchsorted(pf["date"], pc["date"][di])]
        # 仅保留有市值且达流动性下限的交易日段
        prep[c] = {"pc": pc, "mv": mv, "arr": prep_arrays(
            pc["date"], pc["high"], pc["low"], pc["close"], pc["vol"])}

    return codes, st_set, list_date, prep


def _nearest_le(dates, target):
    i = int(np.searchsorted(dates, target, side="right")) - 1
    return i


def signal_ok(code, prev_int, prep, st_set, list_date):
    """在 prev_int(=T 日) 收盘评估完整五步信号；通过返回 (R, stage_high, vol_ratio) 或 None。"""
    if code in st_set:
        return None
    ld = list_date.get(code)
    if ld is not None and int(str(ld)) > prev_int - LIST_MIN_YEARS * 10000:
        return None
    d = prep[code]
    pc, arr = d["pc"], d["arr"]
    i = _nearest_le(pc["date"], prev_int)
    if i < MA_MID + MID_SLOPE_LOOK:
        return None
    # 流动性
    if i >= len(d["mv"]) or np.isnan(d["mv"][i]) or d["mv"][i] < MV_MIN_WAN:
        return None
    close = arr["close"]
    # ① 看趋势
    if np.isnan(arr["ma_m"][i]) or np.isnan(arr["ma_s"][i]):
        return None
    if not (close[i] > arr["ma_m"][i] and arr["ma_s"][i] > arr["ma_m"][i] and arr["mid_up"][i]):
        return None
    # ② 定突破：取 T 之前最近突破日
    tb = int(arr["btb"][i])
    if tb < 0:
        return None
    gap = i - tb
    if gap < 1 or gap > BREAKOUT_MAX_GAP:
        return None
    R = arr["R_at"][tb]
    Vbase = arr["Vbase_at"][tb]
    if np.isnan(R) or np.isnan(Vbase) or Vbase <= 0:
        return None
    # ③ 算回落
    seg = close[tb:i + 1]
    stage_high = float(np.max(seg))
    if stage_high <= 0:
        return None
    depth = (stage_high - close[i]) / stage_high
    if depth < PULLBACK_MIN or depth > PULLBACK_MAX:
        return None
    if close[i] <= R:           # 已跌破突破位 -> 结构改变，非回踩
        return None
    vol_seg = arr["vol"][tb:i + 1]
    vol_ratio = float(np.mean(vol_seg)) / Vbase
    if vol_ratio > VOL_RATIO_MAX:
        return None
    # ④ 等确认
    if CONFIRM == "ma":
        ok = close[i] > arr["ma_s"][i]
    elif CONFIRM == "range":
        ok = (i >= 1) and (close[i] > arr["high"][i - 1])
    elif CONFIRM == "vol":
        ok = arr["vol"][i] > float(np.mean(vol_seg))
    else:
        ok = False
    if not ok:
        return None
    return (R, stage_high, vol_ratio)


def run_backtest(start_date="20180101", end_date="20260630", top_n=TOP_N,
                 pool="000852.SH", confirm=CONFIRM,
                 tp_at_stage_high=TP_AT_STAGE_HIGH, max_hold=MAX_HOLD,
                 variant="base"):
    global CONFIRM
    CONFIRM = confirm
    codes, st_set, list_date, prep = load_universe_data(pool, start_date, end_date)
    trade_dates = get_trade_dates(start_date, end_date)
    date_idx = {d: i for i, d in enumerate(trade_dates)}
    print(f"交易日 {len(trade_dates)}；参数 confirm={confirm} top_n={top_n} "
          f"MA{MA_SHORT}/{MA_MID} resist{RESIST_WIN} volr{VOL_RATIO_MAX} "
          f"pb[{PULLBACK_MIN},{PULLBACK_MAX}] gap{BREAKOUT_MAX_GAP} maxhold{max_hold}")

    positions = {}          # code -> dict
    pending_buy = {}        # code -> (R, stage_high, vol_ratio)
    pending_exit = set()    # code
    cash = INIT_CAPITAL
    daily_vals = []
    trades = []
    name_cache = {}

    # ⑤ 分类统计（视频核心产物：一批可比较样本）
    stat = {"candidates": 0, "entered": 0, "stop_breakout": 0,
            "stop_structure": 0, "take_profit": 0, "max_hold": 0,
            "trade_restricted": 0}

    def get_name(code):
        if code not in name_cache:
            name_cache[code] = get_stock_name(code)
        return name_cache[code]

    for i, td in enumerate(trade_dates):
        td_int = int(td)
        # 1) 执行挂单卖出（T+1 开盘）
        for code in list(pending_exit):
            op = get_open_price(code, td)
            if op is None:
                stat["trade_restricted"] += 1
                continue  # 涨跌停/停牌：次日重试
            pos = positions.get(code)
            if pos is None:
                pending_exit.discard(code)
                continue
            fee = calc_fee("sell", op, pos["shares"])
            cash += pos["shares"] * op - fee
            trades.append({"date": td, "action": "SELL", "code": code,
                           "name": get_name(code), "price": op,
                           "shares": pos["shares"], "reason": pos["exit_reason"]})
            if pos["exit_reason"] == "stop_breakout":
                stat["stop_breakout"] += 1
            elif pos["exit_reason"] == "stop_structure":
                stat["stop_structure"] += 1
            elif pos["exit_reason"] == "take_profit":
                stat["take_profit"] += 1
            elif pos["exit_reason"] == "max_hold":
                stat["max_hold"] += 1
            del positions[code]
            pending_exit.discard(code)
        # 2) 执行挂单买入（T+1 开盘）
        for code in list(pending_buy.keys()):
            op = get_open_price(code, td)
            if op is None:
                pending_buy.pop(code, None)
                continue  # 买不进：放弃该信号
            if code in positions or len(positions) >= top_n:
                pending_buy.pop(code, None)
                continue
            R, stage_high, vol_ratio = pending_buy.pop(code)
            cash_per = cash / (top_n - len(positions)) if (top_n - len(positions)) > 0 else cash
            shares = int(cash_per / op / 100) * 100
            if shares < 100:
                continue
            cost = shares * op
            fee = calc_fee("buy", op, shares)
            if cost + fee <= cash:
                cash -= cost + fee
                positions[code] = {
                    "shares": shares, "buy_price": op, "buy_idx": i,
                    "R": R, "stage_high": stage_high, "vol_ratio": vol_ratio,
                    "exit_reason": None,
                }
                trades.append({"date": td, "action": "BUY", "code": code,
                               "name": get_name(code), "price": op,
                               "shares": shares, "reason": "confirm_entry"})
                stat["entered"] += 1
        # 3) 持仓退出判定（T 收盘）
        for code in list(positions.keys()):
            pos = positions[code]
            if code in pending_exit:
                continue
            d = prep.get(code)
            if d is None:
                continue
            arr = d["arr"]
            ii = _nearest_le(arr["date"], td_int)
            if ii < 0:
                continue
            close = arr["close"][ii]
            # 止盈优先
            if tp_at_stage_high and close >= pos["stage_high"]:
                pos["exit_reason"] = "take_profit"
                pending_exit.add(code)
                continue
            if close < pos["R"]:                 # 有效跌破突破位
                pos["exit_reason"] = "stop_breakout"
                pending_exit.add(code)
                continue
            if close < arr["ma_m"][ii] or not arr["mid_up"][ii]:  # 结构改变
                pos["exit_reason"] = "stop_structure"
                pending_exit.add(code)
                continue
            if i - pos["buy_idx"] >= max_hold:
                pos["exit_reason"] = "max_hold"
                pending_exit.add(code)
        # 4) 新信号（T 收盘评估，T+1 开盘执行）
        if len(positions) + len(pending_buy) < top_n:
            for code in codes:
                if code in positions or code in pending_buy:
                    continue
                if code not in prep:
                    continue
                sig = signal_ok(code, td_int, prep, st_set, list_date)
                if sig is None:
                    continue
                stat["candidates"] += 1
                pending_buy[code] = sig
                if len(positions) + len(pending_buy) >= top_n:
                    break
        # 5) 每日市值
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

    print("\n" + "=" * 64)
    print(f"缩量回踩战法({confirm})  区间 {start_date}~{end_date}  池={INDEX_DISPLAY_NAME.get(pool, pool)}")
    print("=" * 64)
    print(f"  初始/最终: {INIT_CAPITAL:,.0f} -> {final:,.0f}  盈亏 {final-INIT_CAPITAL:+,.0f}")
    print(f"  总收益: {total_ret:+.2f}%   年化: {ann:+.2f}%   最大回撤: {mdd:.2f}%   夏普: {sharpe:.2f}")
    print(f"  基准({INDEX_DISPLAY_NAME.get(pool, pool)}): {idx_ret:+.2f}%   超额: {total_ret-idx_ret:+.2f}%")
    print(f"  交易次数: {len(trades)}")
    print(f"  样本分类: 候选观察={stat['candidates']} 入场={stat['entered']} "
          f"止盈={stat['take_profit']} 破位止损={stat['stop_breakout']} "
          f"结构止损={stat['stop_structure']} 超时={stat['max_hold']} 交易受限={stat['trade_restricted']}")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "results", "shrink_pullback")
    os.makedirs(out_dir, exist_ok=True)
    tag = variant
    summ = os.path.join(out_dir, f"spb_{pool}_{start_date}_{end_date}_{confirm}_{tag}.csv")
    with open(summ, "w", encoding="utf-8") as f:
        f.write("metric,value\n")
        f.write(f"start,{start_date}\nend,{end_date}\npool,{pool}\nconfirm,{confirm}\n")
        f.write(f"total_return,{total_ret:.4f}\nannual,{ann:.4f}\nmax_drawdown,{mdd:.4f}\n")
        f.write(f"sharpe,{sharpe:.4f}\nindex_return,{idx_ret:.4f}\n")
        f.write(f"excess,{total_ret-idx_ret:.4f}\ntrades,{len(trades)}\n")
        for k, v in stat.items():
            f.write(f"stat_{k},{v}\n")
    tlog = os.path.join(out_dir, f"spb_trades_{pool}_{start_date}_{end_date}_{confirm}_{tag}.csv")
    pd.DataFrame(trades).to_csv(tlog, index=False, encoding="utf-8")
    print(f"  结果: {summ}\n  成交: {tlog}")
    return {"total_return": total_ret, "annual": ann, "mdd": mdd, "sharpe": sharpe,
            "index_return": idx_ret, "excess": total_ret - idx_ret, "trades": len(trades),
            "stat": stat}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20180101")
    ap.add_argument("--end", default="20260630")
    ap.add_argument("--top_n", type=int, default=TOP_N)
    ap.add_argument("--pool", default="000852.SH")
    ap.add_argument("--confirm", default=CONFIRM, choices=["ma", "range", "vol"])
    ap.add_argument("--no-tp", dest="tp", action="store_false", help="关闭回到阶段高点止盈")
    ap.add_argument("--max_hold", type=int, default=MAX_HOLD)
    ap.add_argument("--variant", default="base")
    args = ap.parse_args()
    run_backtest(args.start, args.end, args.top_n, args.pool, confirm=args.confirm,
                 tp_at_stage_high=args.tp, max_hold=args.max_hold, variant=args.variant)
