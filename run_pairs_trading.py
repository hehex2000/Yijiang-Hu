# -*- coding: utf-8 -*-
"""
配对套利策略 - 多头配对轮动（A股适配版）
======================================
核心原理：两只高相关ETF的价差会均值回归。
传统套利=做多便宜的+做空贵的，A股改为：
  始终只持有一只ETF，价差极端时从贵的切换到便宜的。
  叠加市场趋势过滤（沪深300ETF > MA60时才开仓）。

支持的配对（--pair 编号）：
  [1] 沪深300ETF vs 上证50ETF    — 宽基轮动
  [2] 中证500ETF vs 中证1000ETF   — 中小盘序列
  [3] 创业板ETF vs 创业板50ETF   — 创业板系列

回测频率（--check-freq）：
  daily   每日检查
  weekly  每周一检查（默认）

数据来源：etf_daily 表（真实ETF价格）
"""

import sys, os, argparse, numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 复用现有 ETF 工具函数
from run_etf_rotation import get_etf_price, get_etf_open, calc_etf_fee
from run_etf_rotation import get_conn, get_trade_dates, COMMISSION_RATE, COMMISSION_MIN, SLIPPAGE_RATE

# ── 常量 ──────────────────────────────────────────────────
INITIAL_CAPITAL = 100000
BENCHMARK = "510300.SH"    # 沪深300ETF 用作市场过滤器
CASH_NAME = "货币基金"

# ── 配对定义 ──────────────────────────────────────────────
PAIRS = [
    {
        "id": 1,
        "name": "沪深300-上证50",
        "a": "510300.SH", "a_name": "沪深300ETF",
        "b": "510050.SH", "b_name": "上证50ETF",
    },
    {
        "id": 2,
        "name": "中证500-中证1000",
        "a": "510500.SH", "a_name": "中证500ETF",
        "b": "512100.SH", "b_name": "中证1000ETF",
    },
    {
        "id": 3,
        "name": "创业板-创业板50",
        "a": "159915.SZ", "a_name": "创业板ETF",
        "b": "159949.SZ", "b_name": "创业板50ETF",
    },
]

PAIR_BY_ID = {p["id"]: p for p in PAIRS}


# ── 统计计算 ──────────────────────────────────────────────

def _query_history(code, trade_date, limit):
    """获取历史收盘价序列（降序）"""
    conn = get_conn()
    rows = pd.read_sql_query(
        "SELECT close FROM etf_daily WHERE ts_code = ? AND trade_date <= ? "
        "ORDER BY trade_date DESC LIMIT ?",
        conn, params=(code, trade_date, limit)
    )
    conn.close()
    if len(rows) < 1:
        return None
    return [float(r) for r in rows["close"].values]


def calc_price_ratio(code_a, code_b, trade_date, window=60):
    """计算配对的价格比和 Z-score

    返回:
        {"ratio": float, "zscore": float, "is_valid": bool}
        或 None（数据不足）
    """
    prices_a = _query_history(code_a, trade_date, window + 5)
    prices_b = _query_history(code_b, trade_date, window + 5)
    if prices_a is None or prices_b is None:
        return None

    n = min(len(prices_a), len(prices_b))
    if n < window:
        return None

    # 计算价格比序列（降序，最新在前）
    ratios = [prices_a[i] / prices_b[i] for i in range(n)]

    current = ratios[0]
    mean = np.mean(ratios[:window])
    std = np.std(ratios[:window])

    if std == 0:
        return None

    zscore = (current - mean) / std
    return {"ratio": current, "mean": mean, "std": std, "zscore": zscore}


def calc_correlation(code_a, code_b, start_date, end_date):
    """计算两只ETF的相关系数（用区间内日收益率）"""
    conn = get_conn()
    rows_a = pd.read_sql_query(
        "SELECT trade_date, pct_chg FROM etf_daily WHERE ts_code = ? "
        "AND trade_date BETWEEN ? AND ? ORDER BY trade_date",
        conn, params=(code_a, start_date, end_date)
    )
    rows_b = pd.read_sql_query(
        "SELECT trade_date, pct_chg FROM etf_daily WHERE ts_code = ? "
        "AND trade_date BETWEEN ? AND ? ORDER BY trade_date",
        conn, params=(code_b, start_date, end_date)
    )
    conn.close()

    if len(rows_a) < 20 or len(rows_b) < 20:
        return 0.0

    merged = pd.merge(rows_a, rows_b, on="trade_date", suffixes=("_a", "_b"))
    if len(merged) < 20:
        return 0.0

    corr = merged["pct_chg_a"].corr(merged["pct_chg_b"])
    return float(corr)


# ── 信号判断 ──────────────────────────────────────────────

def signal_decision(zscore, threshold=1.8, exit_z=0.5):
    """根据 Z-score 判断动作

    返回:
        "buy_a"     → A相对便宜，买A
        "buy_b"     → B相对便宜，买B
        "exit"      → 回归均值，平仓
        "hold"      → 继续等待
    """
    if zscore > threshold:
        # A相对B贵 → 买B（便宜的）
        return "buy_b"
    elif zscore < -threshold:
        # B相对A贵 → 买A（便宜的）
        return "buy_a"
    elif abs(zscore) < exit_z:
        # 已回归均值
        return "exit"
    else:
        return "hold"


def is_market_safe(trade_date, ma_period=60):
    """市场过滤器：沪深300ETF > MA60 时才可开仓"""
    price = get_etf_price(BENCHMARK, trade_date)
    if price is None:
        return False
    conn = get_conn()
    rows = pd.read_sql_query(
        "SELECT close FROM etf_daily WHERE ts_code = ? AND trade_date <= ? "
        "ORDER BY trade_date DESC LIMIT ?",
        conn, params=(BENCHMARK, trade_date, ma_period + 5)
    )
    conn.close()
    if len(rows) < ma_period:
        return False
    closes = [float(r) for r in rows["close"].values]
    ma = np.mean(closes[:ma_period])
    return price > ma


# ── 主回测循环 ────────────────────────────────────────────

def run_pairs_backtest(start_date="20200101", end_date="20251231",
                       pair_id=1, threshold=1.8, window=60,
                       check_freq="weekly", capital=INITIAL_CAPITAL,
                       verbose=True):
    """配对套利回测

    Args:
        start_date:  回测开始
        end_date:    回测结束
        pair_id:     配对编号 1/2/3
        threshold:   开仓Z-score阈值（默认1.8）
        window:      滚动Z-score窗口（默认60）
        check_freq:  "daily" / "weekly"
        capital:     初始资金
        verbose:     是否打印日志
    """
    pair = PAIR_BY_ID.get(pair_id)
    if pair is None:
        valid = [p["id"] for p in PAIRS]
        print(f"[ERR] 无效配对: {pair_id}，可选 {valid}")
        return None

    code_a, name_a = pair["a"], pair["a_name"]
    code_b, name_b = pair["b"], pair["b_name"]

    # ── 获取交易日 ──
    trade_dates = get_trade_dates(start_date, end_date)
    if len(trade_dates) < 120:
        print(f"[ERR] 交易日数据不足：{len(trade_dates)} 天")
        return None

    # ── 计算相关系数（用户可见） ──
    corr = calc_correlation(code_a, code_b, start_date, end_date)
    if verbose:
        method_names = {1: "沪深300-上证50", 2: "中证500-中证1000", 3: "创业板-创业板50"}
        check_names = {"daily": "每日", "weekly": "每周"}
        print(f"\n{'=' * 70}")
        print(f"  配对套利回测")
        print(f"{'=' * 70}")
        print(f"  配对：{name_a} vs {name_b}")
        print(f"  相关系数：{corr:.4f}")
        print(f"  开仓阈值：{threshold}σ | Z窗口：{window}天")
        print(f"  检查频率：{check_names.get(check_freq, check_freq)}")
        print(f"  市场过滤：沪深300ETF > MA60 才开仓")
        print(f"  回测区间：{start_date} ~ {end_date}")
        print(f"  初始资金：{capital:,.2f}")
        print(f"  交易日：{len(trade_dates)} 天")
        print()

    # ── 初始化 ──
    cash = float(capital)
    position = None       # "A" 或 "B" 或 None
    pos_shares = 0
    pos_buy_price = 0.0
    pos_day = 0
    trades = []
    daily_vals = []

    # ── 检查日设置 ──
    if check_freq == "daily":
        check_dates = set(trade_dates)
    else:
        # 每周：取每周第一个交易日
        check_dates = set()
        for i, d_str in enumerate(trade_dates):
            if i == 0:
                check_dates.add(int(d_str))
            else:
                prev_ymd = str(trade_dates[i - 1])
                curr_ymd = str(d_str)
                # 检查是否跨周
                from datetime import datetime as dt
                prev_w = dt.strptime(prev_ymd, "%Y%m%d").weekday()
                curr_w = dt.strptime(curr_ymd, "%Y%m%d").weekday()
                if curr_w < prev_w:  # 跨周
                    check_dates.add(int(d_str))
        # 也加第一个交易日
        check_dates.add(int(trade_dates[0]))

    # ── 逐日循环 ──
    for i, td_str in enumerate(trade_dates):
        td = int(td_str)
        prev_td = int(trade_dates[i - 1]) if i > 0 else td

        # ── 检查日：判断信号 ──
        if td in check_dates:
            stats = calc_price_ratio(code_a, code_b, prev_td, window=window)
            if stats is None:
                pass  # 数据不足，跳过
            else:
                zscore = stats["zscore"]
                action = signal_decision(zscore, threshold=threshold)

                # 市场过滤器：沪深300ETF跌破MA60时强制空仓
                market_ok = is_market_safe(prev_td)
                if position and not market_ok:
                    # 市场走坏，强制平仓
                    action = "exit_market"
                if not market_ok and action in ("buy_a", "buy_b"):
                    action = "hold"  # 不清仓也不开仓

                if verbose and action != "hold":
                    pos_status = f"当前持仓={position or '空仓'}"
                    mkt = f" 市场={'OK' if market_ok else '空头'}"
                    print(f"  [{td}] Z={zscore:.2f}  ratio={stats['ratio']:.4f}  "
                          f"信号={action}  {mkt}  {pos_status}")

                # ── 平仓（exit / exit_market / 需要换仓） ──
                if position is not None:
                    should_exit = False
                    if action in ("exit", "exit_market"):
                        should_exit = True
                    elif action == "buy_a" and position != "A":
                        should_exit = True  # 需要换仓
                    elif action == "buy_b" and position != "B":
                        should_exit = True  # 需要换仓

                    if should_exit:
                        # 按开盘价卖出
                        sell_code = code_a if position == "A" else code_b
                        sell_name = name_a if position == "A" else name_b
                        sell_price = get_etf_open(sell_code, td)
                        if sell_price and sell_price > 0:
                            proceeds = pos_shares * sell_price
                            fee = calc_etf_fee('sell', sell_price, pos_shares)
                            cash += proceeds - fee
                            trades.append({
                                "date": td, "action": "SELL", "code": sell_code,
                                "name": sell_name, "price": sell_price,
                                "shares": pos_shares,
                                "pnl": proceeds - fee - pos_shares * pos_buy_price,
                                "reason": action
                            })
                            if verbose:
                                print(f"    → 卖出 {sell_name}：{pos_shares}份 @ {sell_price:.3f}")
                        position = None
                        pos_shares = 0

                # ── 开仓（exit后或新开） ──
                if action in ("buy_a", "buy_b"):
                    buy_code = code_a if action == "buy_a" else code_b
                    buy_name = name_a if action == "buy_a" else name_b
                    open_price = get_etf_open(buy_code, td)
                    if open_price and open_price > 0:
                        # 预留费用空间
                        alloc = cash * 0.998
                        max_shares = int(alloc / open_price)
                        if max_shares >= 1:
                            cost = max_shares * open_price
                            fee = calc_etf_fee('buy', open_price, max_shares)
                            if cost + fee <= cash:
                                cash -= cost + fee
                                position = "A" if action == "buy_a" else "B"
                                pos_shares = max_shares
                                pos_buy_price = open_price
                                trades.append({
                                    "date": td, "action": "BUY", "code": buy_code,
                                    "name": buy_name, "price": open_price,
                                    "shares": max_shares, "pnl": None,
                                    "reason": "open"
                                })
                                if verbose:
                                    print(f"    → 买入 {buy_name}：{max_shares}份 @ {open_price:.3f}")

        # ── 每日净值 ──
        total_value = cash
        if position == "A":
            p = get_etf_price(code_a, td)
            if p: total_value += pos_shares * p
        elif position == "B":
            p = get_etf_price(code_b, td)
            if p: total_value += pos_shares * p
        daily_vals.append({"date": td, "value": total_value})

    # ── 回测结束：强制平仓 ──
    if trade_dates:
        last_date = trade_dates[-1]
        if position:
            sell_code = code_a if position == "A" else code_b
            sell_name = name_a if position == "A" else name_b
            price = get_etf_price(sell_code, last_date)
            if price and price > 0:
                proceeds = pos_shares * price
                fee = calc_etf_fee('sell', price, pos_shares)
                cash += proceeds - fee
                trades.append({
                    "date": last_date, "action": "SELL", "code": sell_code,
                    "name": sell_name, "price": price, "shares": pos_shares,
                    "pnl": proceeds - fee - pos_shares * pos_buy_price,
                    "reason": "backtest_end"
                })
            position = None
            pos_shares = 0

    # ── 绩效计算 ──
    final_value = cash
    total_return = (final_value / capital - 1) * 100
    days = len(trade_dates)
    years = days / 252.0
    annual_return = ((final_value / capital) ** (1 / years) - 1) * 100 if years > 0 else 0

    vals = np.array([d["value"] for d in daily_vals])
    cummax = np.maximum.accumulate(vals)
    safe_cummax = np.where(cummax == 0, 1, cummax)
    drawdowns = (vals - cummax) / safe_cummax
    max_dd = float(np.min(drawdowns)) * 100

    # 胜率
    closed_trades = [t for t in trades if t["action"] == "SELL" and t.get("pnl") is not None]
    wins = sum(1 for t in closed_trades if t["pnl"] > 0)
    win_rate = wins / len(closed_trades) * 100 if closed_trades else 0
    total_pnl = sum(t["pnl"] for t in closed_trades)

    # ── 输出 ──
    print(f"\n{'=' * 70}")
    print(f"  配对套利回测结果 — {pair['name']}")
    print(f"{'=' * 70}")
    print(f"  初始资金：{capital:,.2f}")
    print(f"  最终资产：{final_value:,.2f}")
    print(f"  总盈亏：{total_pnl:+,.0f} 元")
    print(f"  总收益率：{total_return:+.2f}%")
    print(f"  年化收益率：{annual_return:+.2f}%")
    print(f"  最大回撤：{max_dd:.2f}%")
    print(f"  交易次数：{len(closed_trades)} 次")
    print(f"  胜率：{win_rate:.1f}%（{wins}/{len(closed_trades)}）")
    print(f"  总手续费+滑点：扣除已完成")

    return {
        "pair": pair["name"],
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "trades": len(closed_trades),
        "final_value": final_value,
    }


# ── CLI 入口 ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="配对套利策略回测（A股多头轮动版）")
    parser.add_argument("start_date", nargs="?", default="20200101")
    parser.add_argument("end_date", nargs="?", default="20251231")
    parser.add_argument("--pair", type=int, default=1, choices=[1, 2, 3],
                        help="配对编号 1=沪深300-上证50, 2=中证500-中证1000, 3=创业板-创业板50")
    parser.add_argument("--threshold", type=float, default=2.0,
                        help="Z-score开仓阈值（默认2.0）")
    parser.add_argument("--window", type=int, default=60,
                        help="Z-score滚动窗口（默认60天）")
    parser.add_argument("--check-freq", default="weekly",
                        choices=["daily", "weekly"],
                        help="检查频率 daily/weekly（默认weekly）")
    parser.add_argument("--capital", type=int, default=INITIAL_CAPITAL)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    run_pairs_backtest(
        start_date=args.start_date,
        end_date=args.end_date,
        pair_id=args.pair,
        threshold=args.threshold,
        window=args.window,
        check_freq=args.check_freq,
        capital=args.capital,
        verbose=not args.quiet,
    )
