# -*- coding: utf-8 -*-
"""
配对套利策略 - 多头配对轮动（A股适配版）
======================================
核心原理：两只高相关ETF的价差会均值回归。
传统套利=做多便宜的+做空贵的，A股改为：
  始终只持有一只ETF，价差极端时从贵的切换到便宜的。
  叠加市场趋势过滤（沪深300ETF > MA60时才开仓）。

支持的配对（--pair 编号）：
  [1]  沪深300 vs 上证50       — 宽基·大盘内部
  [2]  中证500 vs 中证800      — 宽基·中大盘
  [3]  创业板   vs 创业板50     — 宽基·创业板系列
  [4]  科创50   vs 创业板50    — 宽基·科技成长
  [5]  半导体   vs 新能源车    — 行业·科技制造
  [6]  沪深300 vs 中证800      — 宽基·高度相关
  [7]  恒生ETF vs 沪深300     — 跨境·AH溢价
  [8]  黄金ETF vs 国债ETF      — 避险·股债轮动
  [9]  红利ETF vs 红利低波ETF  — 红利·同主题高相关

回测频率（--check-freq）：
  daily   每日检查
  weekly  每周一检查（默认）

数据来源：etf_daily 表（真实ETF价格）

注：2026-07-06 起适用交易新规——当日收盘价触发的"卖出信号"可在当日收盘价成交
（非未来函数）；此前版本仅在检查日按上一日收盘信号、于次日开盘成交。
"""

import sys, os, argparse, numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 复用现有 ETF 工具函数
from run_etf_rotation import get_etf_price, get_etf_open, calc_etf_fee, PremiumGate
from run_etf_rotation import get_conn, get_trade_dates, COMMISSION_RATE, COMMISSION_MIN, SLIPPAGE_RATE

# ── 常量 ──────────────────────────────────────────────────
INITIAL_CAPITAL = 500000   # 默认初始资金 50 万（与 bat 的 P_PAIRS_CAPITAL 对齐）
BENCHMARK = "510300.SH"    # 沪深300ETF 用作市场过滤器
CASH_NAME = "货币基金"

# 2026-07-06 起交易新规：当日收盘价触发的"卖出信号"可于当日收盘价成交（非未来函数）。
# 此前版本仅在检查日按上一日收盘信号、于次日开盘成交。
NEW_RULE_START = 20260706

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
        "name": "中证500-中证800",
        "a": "510500.SH", "a_name": "中证500ETF",
        "b": "515800.SH", "b_name": "中证800ETF",
    },
    {
        "id": 3,
        "name": "创业板-创业板50",
        "a": "159915.SZ", "a_name": "创业板ETF",
        "b": "159949.SZ", "b_name": "创业板50ETF",
    },
    # ── 新增配对 ──────────────────────────────────────────
    {
        "id": 4,
        "name": "科创50-创业板50",
        "a": "588000.SH", "a_name": "科创50ETF",
        "b": "159949.SZ", "b_name": "创业板50ETF",
    },
    {
        "id": 5,
        "name": "半导体-新能源车",
        "a": "512480.SH", "a_name": "半导体ETF",
        "b": "515030.SH", "b_name": "新能源车ETF",
    },
    {
        "id": 6,
        "name": "沪深300-中证800",
        "a": "510300.SH", "a_name": "沪深300ETF",
        "b": "515800.SH", "b_name": "中证800ETF",
    },
    {
        "id": 7,
        "name": "恒生-沪深300",
        "a": "159920.SZ", "a_name": "恒生ETF",
        "b": "510300.SH", "b_name": "沪深300ETF",
    },
    {
        "id": 8,
        "name": "黄金-国债",
        "a": "518880.SH", "a_name": "黄金ETF",
        "b": "511010.SH", "b_name": "国债ETF",
    },
    # ── 红利主题配对（2026-07-23 新增）─────────────────────
    {
        "id": 9,
        "name": "红利-红利低波",
        "a": "515080.SH", "a_name": "红利ETF",
        "b": "512890.SH", "b_name": "红利低波ETF",
    },
]

PAIR_BY_ID = {p["id"]: p for p in PAIRS}


# ── 统计计算 ──────────────────────────────────────────────

def _query_history(code, trade_date, limit):
    """获取历史 (交易日, 收盘价) 序列（降序，最新在前）。

    返回 [(trade_date, close), ...]。按日期对齐能避免两只 ETF 交易日历不同
    （如恒生ETF跟港股假期、黄金/国债ETF偶发缺口）导致的价格比错位。
    """
    conn = get_conn()
    rows = pd.read_sql_query(
        "SELECT trade_date, close FROM etf_daily WHERE ts_code = ? AND trade_date <= ? "
        "ORDER BY trade_date DESC LIMIT ?",
        conn, params=(code, trade_date, limit)
    )
    conn.close()
    if len(rows) < 1:
        return None
    return [(int(r[0]), float(r[1])) for r in rows.itertuples(index=False)]


def calc_price_ratio(code_a, code_b, trade_date, window=60):
    """计算配对的价格比和 Z-score（按交易日对齐，修复跨日历错位 bug）

    返回:
        {"ratio": float, "mean": float, "std": float, "zscore": float}
        或 None（数据不足 / 对齐后样本不够）

    修复点：旧实现用 prices_a[i]/prices_b[i] 按"位置索引"配对，假定两 ETF
    交易日历完全一致；一旦任一 ETF 有数据缺口或跨市场假期不同，i 位置对应的
    日期就不一致，比值建立在错位的日期上 → 虚假的极端 Z-score → 错误信号。
    现改为先取两者共同的交易日（交集）再算比，彻底消除错位。
    """
    hist_a = _query_history(code_a, trade_date, window + 5)
    hist_b = _query_history(code_b, trade_date, window + 5)
    if hist_a is None or hist_b is None:
        return None

    # 按交易日对齐：只取两者都有的共同交易日，避免跨市场/数据缺口导致的错位
    map_a = {d: c for d, c in hist_a}
    map_b = {d: c for d, c in hist_b}
    common = sorted(set(map_a) & set(map_b), reverse=True)  # 降序，最新在前
    if len(common) < window:
        return None

    # 价格比序列（共同交易日，最新在前）
    ratios = [map_a[d] / map_b[d] for d in common[:window]]

    current = ratios[0]
    mean = np.mean(ratios)
    std = np.std(ratios)

    if std == 0:
        return None

    zscore = (current - mean) / std
    return {"ratio": current, "mean": mean, "std": std, "zscore": zscore}


def calc_correlation(code_a, code_b, start_date, lookback=120):
    """配对相关性（去除"后见之明"误导）。

    改为计算【进入回测时已知】的相关性：取 start_date 之前 lookback 个交易日的
    日收益率做相关系数。这样反映的是策略开始运行时"已经看到"的配对稳定性，
    而非用整个回测区间（含未来）的事后相关性误导读者。

    返回 float 或 None（样本不足 <20 日）。
    """
    conn = get_conn()
    rows_a = pd.read_sql_query(
        "SELECT trade_date, pct_chg FROM etf_daily WHERE ts_code = ? "
        "AND trade_date < ? ORDER BY trade_date DESC LIMIT ?",
        conn, params=(code_a, start_date, lookback)
    )
    rows_b = pd.read_sql_query(
        "SELECT trade_date, pct_chg FROM etf_daily WHERE ts_code = ? "
        "AND trade_date < ? ORDER BY trade_date DESC LIMIT ?",
        conn, params=(code_b, start_date, lookback)
    )
    conn.close()

    if len(rows_a) < 20 or len(rows_b) < 20:
        return None

    merged = pd.merge(rows_a, rows_b, on="trade_date", suffixes=("_a", "_b"))
    if len(merged) < 20:
        return None

    corr = merged["pct_chg_a"].corr(merged["pct_chg_b"])
    return float(corr)


# ── 信号判断 ──────────────────────────────────────────────

def signal_decision(zscore, threshold=2.0, exit_z=0.5):
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
                       pair_id=1, threshold=2.0, window=60,
                       check_freq="weekly", capital=INITIAL_CAPITAL,
                       verbose=True, premium_filter="qdii"):
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

    # ── 折溢价闸门（默认 qdii：跨境ETF≥8%溢价拦截；515080/512890等国内ETF不拦）──
    gate = PremiumGate([{"code": code_a, "name": name_a}, {"code": code_b, "name": name_b}],
                       mode=premium_filter)
    if gate.enabled and verbose:
        print(f"  折溢价过滤：开启 [{premium_filter}]（跨境≥8%溢价拦截，国内放行）")

    # ── 获取交易日 ──
    trade_dates = get_trade_dates(start_date, end_date)
    # 至少需要 window+10 个交易日才能算出有意义的 Z-score
    min_days = window + 10
    if len(trade_dates) < min_days:
        print(f"[ERR] 交易日数据不足：{len(trade_dates)} 天")
        return None

    # ── 计算相关系数（进入回测时已知，非事后全样本） ──
    corr = calc_correlation(code_a, code_b, start_date, lookback=max(window * 2, 120))
    if verbose:
        method_names = {1: "沪深300-上证50", 2: "中证500-中证800", 3: "创业板-创业板50"}
        check_names = {"daily": "每日", "weekly": "每周"}
        print(f"\n{'=' * 70}")
        print(f"  配对套利回测")
        print(f"{'=' * 70}")
        print(f"  配对：{name_a} vs {name_b}")
        if corr is None:
            print(f"  参考相关系数：数据不足（回测起始前样本<20日）")
        else:
            lb = max(window * 2, 120)
            print(f"  参考相关系数（回测起始前{lb}日·进入时已知，非事后全样本）：{corr:.4f}")
        print(f"  开仓阈值：{threshold}σ | Z窗口：{window}天")
        print(f"  检查频率：{check_names.get(check_freq, check_freq)}")
        print(f"  市场过滤：沪深300ETF > MA60 才开仓")
        print(f"  回测区间：{start_date} ~ {end_date}")
        print(f"  收盘卖出新规：{'适用（2026-07-06 起，当日收盘卖出信号可当日收盘价成交）' if int(end_date) >= NEW_RULE_START else '不适用（仅次日开盘成交）'}")
        print(f"  初始资金：{capital:,.2f}")
        print(f"  交易日：{len(trade_dates)} 天")
        print()

    # ── 初始化 ──
    cash = float(capital)
    position = None       # "A" 或 "B" 或 None
    pos_shares = 0
    pos_buy_price = 0.0
    pos_buy_fee = 0.0     # 建仓时支付的佣金+滑点（平仓核算真实盈亏需扣减）
    pos_day = 0           # 建仓交易日（区分"前日持仓"与"当日开盘新买"）
    trades = []
    daily_vals = []

    # 内部辅助：统一费用与真实盈亏核算（修复 Bug3 后集中管理）
    def _sell(td, sell_price, reason):
        nonlocal cash, position, pos_shares, pos_buy_price, pos_buy_fee
        sell_code = code_a if position == "A" else code_b
        sell_name = name_a if position == "A" else name_b
        if sell_price and sell_price > 0:
            proceeds = pos_shares * sell_price
            fee = calc_etf_fee('sell', sell_price, pos_shares)
            cash += proceeds - fee
            trades.append({
                "date": td, "action": "SELL", "code": sell_code,
                "name": sell_name, "price": sell_price,
                "shares": pos_shares,
                "pnl": proceeds - fee - pos_shares * pos_buy_price - pos_buy_fee,
                "reason": reason
            })
            if verbose:
                print(f"    → 卖出 {sell_name}：{pos_shares}份 @ {sell_price:.3f}（{reason}）")
        position = None
        pos_shares = 0
        pos_buy_price = 0.0
        pos_buy_fee = 0.0

    def _buy(td, buy_price, action):
        nonlocal cash, position, pos_shares, pos_buy_price, pos_buy_fee, pos_day
        buy_code = code_a if action == "buy_a" else code_b
        buy_name = name_a if action == "buy_a" else name_b
        # ── 折溢价闸门（默认 qdii）：高溢价标的跳过开仓，防追高溢价收敛亏损 ──
        if gate.enabled:
            allow, prem = gate.check(buy_code, td)
            if not allow:
                if verbose:
                    print(f"    → [折溢价拦截] {buy_name} 溢价 {prem:+.2%} ≥ 阈值，跳过开仓")
                return
        if not (buy_price and buy_price > 0):
            return
        alloc = cash * 0.998
        max_shares = int(alloc / buy_price / 100) * 100  # 取整到100份(1手)
        if max_shares < 100:
            return
        cost = max_shares * buy_price
        fee = calc_etf_fee('buy', buy_price, max_shares)
        if cost + fee > cash:
            return
        cash -= cost + fee
        position = "A" if action == "buy_a" else "B"
        pos_shares = max_shares
        pos_buy_price = buy_price
        pos_buy_fee = fee
        pos_day = td
        trades.append({
            "date": td, "action": "BUY", "code": buy_code,
            "name": buy_name, "price": buy_price,
            "shares": max_shares, "pnl": None,
            "reason": "open"
        })
        if verbose:
            print(f"    → 买入 {buy_name}：{max_shares}份 @ {buy_price:.3f}")

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

    # 新规判定：回测区间覆盖 2026-07-06 起，才启用"当日收盘卖出"路径
    new_rule_active = int(end_date) >= NEW_RULE_START

    # ── 逐日循环 ──
    for i, td_str in enumerate(trade_dates):
        td = int(td_str)
        prev_td = int(trade_dates[i - 1]) if i > 0 else None
        new_rule = new_rule_active and td >= NEW_RULE_START

        # ── 每日市场过滤（修复 hint1：原仅在"检查日"触发，周频下暴跌要等下周一）──
        # 关键：用"上一交易日收盘"判断（td 开盘时已知，非未来函数）→ 触发 td 开盘强平；
        # 仅在新规下才用 td 收盘判断并当日收盘价强平（td 收盘成交，同样非未来函数）。
        prev_market_ok = is_market_safe(prev_td) if prev_td is not None else True
        td_market_ok = is_market_safe(td) if new_rule else True

        if position is not None:
            if new_rule:
                if not td_market_ok:
                    sell_code = code_a if position == "A" else code_b
                    _sell(td, get_etf_price(sell_code, td), "exit_market")
            else:
                if prev_td is not None and not prev_market_ok:
                    sell_code = code_a if position == "A" else code_b
                    _sell(td, get_etf_open(sell_code, td), "exit_market")

        # ── 检查日：基于 prev_td 收盘信号，开盘成交（正常配对轮动）──
        if td in check_dates and prev_td is not None:
            stats = calc_price_ratio(code_a, code_b, prev_td, window=window)
            if stats is not None:
                zscore = stats["zscore"]
                action = signal_decision(zscore, threshold=threshold)

                # 市场走坏（用上一日收盘判定，td 开盘时已知）时不新开仓
                if not prev_market_ok and action in ("buy_a", "buy_b"):
                    action = "hold"

                if verbose and action != "hold":
                    pos_status = f"当前持仓={position or '空仓'}"
                    mkt = f" 市场={'OK' if prev_market_ok else '空头'}"
                    print(f"  [{td}] Z={zscore:.2f}  ratio={stats['ratio']:.4f}  "
                          f"信号={action}  {mkt}  {pos_status}")

                # 平仓/换仓：基于 prev 信号的开盘成交
                if position is not None:
                    should_exit = False
                    if action in ("exit", "exit_market"):
                        should_exit = True
                    elif action == "buy_a" and position != "A":
                        should_exit = True
                    elif action == "buy_b" and position != "B":
                        should_exit = True
                    if should_exit:
                        sell_code = code_a if position == "A" else code_b
                        _sell(td, get_etf_open(sell_code, td), action)

                # 开仓
                if action in ("buy_a", "buy_b"):
                    if not ((action == "buy_a" and position == "A") or (action == "buy_b" and position == "B")):
                        buy_code = code_a if action == "buy_a" else code_b
                        _buy(td, get_etf_open(buy_code, td), action)

        # ── 收盘阶段（新规 2026-07-06 起）：基于 td 当日收盘信号，可当日收盘价卖出 ──
        # 仅处理"卖出类"信号（新规放开的是收盘卖出，未放开收盘买入）。
        # 仅对"前日持仓"(pos_day<td)生效，避免与当日开盘买入形成人为日内往返。
        if new_rule and position is not None and pos_day < td and prev_td is not None:
            stats_c = calc_price_ratio(code_a, code_b, td, window=window)
            if stats_c is not None:
                zscore_c = stats_c["zscore"]
                action_c = signal_decision(zscore_c, threshold=threshold)
                should_exit_close = (
                    action_c == "exit"
                    or (action_c == "buy_a" and position == "B")
                    or (action_c == "buy_b" and position == "A")
                )
                if should_exit_close:
                    if verbose:
                        print(f"  [{td}] 收盘信号 Z={zscore_c:.2f} → 当日收盘价卖出（新规）")
                    sell_code = code_a if position == "A" else code_b
                    _sell(td, get_etf_price(sell_code, td), "exit_close")

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
                    "pnl": proceeds - fee - pos_shares * pos_buy_price - pos_buy_fee,
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
    parser.add_argument("--pair", type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                        help="配对编号 1~9（见文件顶部 PAIRS 定义；默认1=沪深300-上证50）")
    parser.add_argument("--threshold", type=float, default=2.0,
                        help="Z-score开仓阈值（默认2.0，与函数/菜单一致）")
    parser.add_argument("--window", type=int, default=60,
                        help="Z-score滚动窗口（默认60天）")
    parser.add_argument("--check-freq", default="weekly",
                        choices=["daily", "weekly"],
                        help="检查频率 daily/weekly（默认weekly）")
    parser.add_argument("--capital", type=int, default=INITIAL_CAPITAL)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--premium-filter", type=str, default="qdii",
                        choices=["off", "uniform", "strict", "qdii", "rolling"],
                        help="折溢价闸门(默认qdii)：跨境ETF≥8%%溢价拦截；off=关闭。依据BV1YP326jE7S(用NAV非IOPV)")
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
        premium_filter=args.premium_filter,
    )
