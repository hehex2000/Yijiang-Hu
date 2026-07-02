# -*- coding: utf-8 -*-
"""
ETF轮动策略 - 国内资产动量轮动
============================
支持的调仓方法（--method）：
  single      单动量法：每期满仓第1名，全部走弱时转现金
  dual        双动量法（默认）：前2名等权，MA60过滤
  ma_filter   均线过滤法：MA60之上才买，全部跌破MA60时空仓

标的池（5个国内指数，0黄金/0海外资产）：
  000016.SH 上证50    — 超大盘蓝筹
  000300.SH 沪深300   — 大盘价值（基准指数）
  000905.SH 中证500   — 中盘成长
  000852.SH 中证1000  — 小盘
  399006.SZ 创业板指  — 科技成长

数据来自 index_daily 表，价格 ÷1000 模拟ETF净值。
"""

import sys, os, argparse, numpy as np, pandas as pd

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── 导入现有工具函数 ──────────────────────────────────────
from run_monthly_rebalance import get_conn, get_monthly_5th_trading_days, COMMISSION_RATE, COMMISSION_MIN, SLIPPAGE_RATE

# ── 常量 ──────────────────────────────────────────────────
STAMP_DUTY_RATE_ETF = 0.0       # ETF 免印花税
INITIAL_CAPITAL = 100000        # 初始总资金
PRICE_SCALE = 0.001             # 指数价格缩放因子（÷1000 → ~元/份）

# ── 标的池定义 ────────────────────────────────────────────
ETF_UNIVERSE = [
    {"code": "000016.SH", "name": "上证50",    "desc": "超大盘蓝筹"},
    {"code": "000300.SH", "name": "沪深300",   "desc": "大盘价值"},
    {"code": "000905.SH", "name": "中证500",   "desc": "中盘成长"},
    {"code": "000852.SH", "name": "中证1000",  "desc": "小盘"},
    {"code": "399006.SZ", "name": "创业板指",  "desc": "科技成长"},
]

BENCHMARK_CODE = "000300.SH"   # 基准指数代码
CASH_NAME = "货币基金"          # 现金避险名称


# ── 价格辅助函数 ──────────────────────────────────────────

def get_etf_price(ts_code, trade_date, scaled=True):
    """获取指数模拟ETF价格

    从 index_daily 表查询收盘价，除以 PRICE_SCALE 模拟ETF净值。
    支持精确日期匹配 + fallback 到最近交易日。
    """
    conn = get_conn()
    row = pd.read_sql_query(
        "SELECT close FROM index_daily WHERE ts_code = ? AND trade_date = ?",
        conn, params=(ts_code, trade_date)
    )
    if len(row) > 0:
        conn.close()
        price = float(row.iloc[0]["close"])
        return price * PRICE_SCALE if scaled else price

    row2 = pd.read_sql_query(
        "SELECT close FROM index_daily WHERE ts_code = ? AND trade_date < ? ORDER BY trade_date DESC LIMIT 1",
        conn, params=(ts_code, trade_date)
    )
    conn.close()
    if len(row2) > 0:
        price = float(row2.iloc[0]["close"])
        return price * PRICE_SCALE if scaled else price
    return None


def get_etf_open(ts_code, trade_date, scaled=True):
    """获取指数模拟ETF开盘价

    指数没有真正的开盘价概念，用前一日收盘价代替（与网格策略一致）。
    """
    return get_etf_price(ts_code, trade_date, scaled=scaled)


# ── ETF 费用计算（免印花税）────────────────────────────────

def calc_etf_fee(buy_or_sell, price, shares):
    """计算ETF交易费用（免印花税）"""
    amount = price * shares
    commission = max(amount * COMMISSION_RATE, COMMISSION_MIN)
    slippage = amount * SLIPPAGE_RATE
    return commission + slippage


# ── 技术指标函数 ──────────────────────────────────────────

def calc_roc(ts_code, trade_date, period=20):
    """ROC涨跌幅：close / close_N_days_ago - 1

    使用缩放后价格计算（不影响ROC百分比值，因为缩放因子约掉了）。
    但为了统一，直接用原始指数价格计算。
    """
    conn = get_conn()
    rows = pd.read_sql_query(
        "SELECT close FROM index_daily WHERE ts_code = ? AND trade_date <= ? ORDER BY trade_date DESC LIMIT ?",
        conn, params=(ts_code, trade_date, period + 5)
    )
    conn.close()
    if len(rows) < period + 1:
        return None
    closes = [float(r) for r in rows["close"].values]
    return closes[0] / closes[period] - 1.0


def calc_ma(ts_code, trade_date, period=60):
    """计算指数移动平均线"""
    conn = get_conn()
    rows = pd.read_sql_query(
        "SELECT close FROM index_daily WHERE ts_code = ? AND trade_date <= ? ORDER BY trade_date DESC LIMIT ?",
        conn, params=(ts_code, trade_date, period + 5)
    )
    conn.close()
    if len(rows) < period:
        return None
    closes = [float(r) for r in rows["close"].values]
    return np.mean(closes[:period])


def calc_volatility(ts_code, trade_date, window=20):
    """计算指数年化波动率

    基于 window 个交易日的对数收益率计算年化标准差。
    """
    conn = get_conn()
    rows = pd.read_sql_query(
        "SELECT close FROM index_daily WHERE ts_code = ? AND trade_date <= ? ORDER BY trade_date DESC LIMIT ?",
        conn, params=(ts_code, trade_date, window + 5)
    )
    conn.close()
    if len(rows) < window + 1:
        return None
    closes = [float(r) for r in rows["close"].values]
    closes = closes[::-1]  # 升序
    returns = np.diff(np.log(closes))
    return float(np.std(returns) * np.sqrt(252))


# ── 交易日期 ──────────────────────────────────────────────

def get_trade_dates(start_date, end_date):
    """获取回测区间的所有交易日（从 index_daily 表获取）"""
    conn = get_conn()
    rows = pd.read_sql_query(
        "SELECT DISTINCT trade_date FROM index_daily WHERE "
        "trade_date BETWEEN ? AND ? ORDER BY trade_date",
        conn, params=(start_date, end_date)
    )
    conn.close()
    return [int(r) for r in rows["trade_date"].values]


# ── 打分与选标逻辑 ────────────────────────────────────────

def score_assets(trade_date, method="dual", roc_period=20, ma_period=60):
    """对 ETF_UNIVERSE 中所有标的打分，返回排序后的列表

    返回:
        [ (code, name, score), ... ] 按得分降序
        或 [] 表示全部走弱应空仓
    """
    results = []
    for etf in ETF_UNIVERSE:
        code = etf["code"]
        close_price = get_etf_price(code, trade_date, scaled=False)
        if close_price is None:
            continue

        # ── 计算 MA60 位置 ──
        ma60 = calc_ma(code, trade_date, period=ma_period)

        # ── 计算 ROC（双动量用两个周期） ──
        if method == "single":
            roc_val = calc_roc(code, trade_date, roc_period)
            if roc_val is None:
                continue
            # 单动量：MA60 非必须（无过滤），但可用于标记
            score = roc_val

        elif method == "dual":
            roc20 = calc_roc(code, trade_date, 20)
            roc60 = calc_roc(code, trade_date, 60)
            vol = calc_volatility(code, trade_date, 20)
            if roc20 is None or roc60 is None or vol is None:
                continue
            # MA60 过滤：跌破MA60的不参与排名
            if ma60 is not None and close_price < ma60:
                continue
            score = roc20 * 0.5 + roc60 * 0.3 - vol * 0.2

        elif method == "ma_filter":
            roc_val = calc_roc(code, trade_date, roc_period)
            if roc_val is None:
                continue
            # MA60 过滤：跌破MA60的不买入
            if ma60 is not None and close_price < ma60:
                continue
            score = roc_val

        else:
            raise ValueError(f"未知的调仓方法: {method}")

        results.append((code, etf["name"], score))

    # 按得分降序排列
    results.sort(key=lambda x: x[2], reverse=True)
    return results


def select_targets(scored_list, method="dual"):
    """根据调仓方法从打分列表中选择目标持仓

    返回:
        [ (code, name, weight_pct), ... ]
        weight_pct 为分配权重（小数，如 0.5=50%）
        空列表表示空仓（全部转现金）
    """
    if not scored_list:
        return []

    if method == "single":
        return [(scored_list[0][0], scored_list[0][1], 1.0)]

    elif method == "dual":
        n = min(2, len(scored_list))
        return [(scored_list[i][0], scored_list[i][1], 1.0 / n) for i in range(n)]

    elif method == "ma_filter":
        # 全部排在 MA60 过滤后，剩下的才买入
        n = max(1, min(len(scored_list) // 2 or 1, len(scored_list)))
        return [(c[0], c[1], 1.0 / n) for c in scored_list[:n]]

    return []


# ── 主回测循环 ────────────────────────────────────────────

def run_etf_rotation(start_date="20200101", end_date="20251231",
                     method="dual", roc_period=20, ma_period=60,
                     capital=INITIAL_CAPITAL, verbose=True):
    """ETF轮动策略主回测

    Args:
        start_date: 回测开始日期 YYYYMMDD
        end_date:   回测结束日期 YYYYMMDD
        method:     调仓方法 "single" / "dual" / "ma_filter"
        roc_period: ROC 计算周期
        ma_period:  MA 计算周期
        capital:    初始资金
        verbose:    是否打印日志

    Returns:
        dict: 回测结果
    """
    # ── 获取交易日和调仓日 ──
    trade_dates = get_trade_dates(start_date, end_date)
    if len(trade_dates) < 60:
        print(f"[ERR] 交易日数据不足：{len(trade_dates)} 天 (需至少 60 天)")
        return None

    rebalance_dates = get_monthly_5th_trading_days(trade_dates)
    # 调仓日集合，只包含回测区间内的
    rebalance_dates = {d for d in rebalance_dates if start_date <= str(d) <= end_date}

    method_names = {"single": "单动量法", "dual": "双动量法", "ma_filter": "均线过滤法"}

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  ETF轮动策略回测")
        print(f"{'=' * 70}")
        print(f"  标的池：{' | '.join(e['name'] for e in ETF_UNIVERSE)}")
        print(f"  调仓方法：{method_names.get(method, method)}")
        print(f"  回测区间：{start_date} ~ {end_date}")
        print(f"  初始资金：{capital:,.2f}")
        print(f"  交易日：{len(trade_dates)} 天 | 调仓日：{len(rebalance_dates)} 次")
        print()

    # ── 初始化 ──
    cash = float(capital)
    positions = {}  # { code: {"shares": int, "buy_price": float} }
    trades = []
    daily_vals = []

    # ── 逐日循环 ──
    for i, td_str in enumerate(trade_dates):
        td = int(td_str)

        # 前一天日期
        prev_td = int(trade_dates[i - 1]) if i > 0 else td

        # ── 每日市值记录（用于计算回撤） ──
        total_value = cash
        for code, pos in list(positions.items()):
            price = get_etf_price(code, td)
            if price is not None:
                total_value += pos["shares"] * price
        daily_vals.append({"date": td, "value": total_value})

        # ── 调仓日：执行轮动 ──
        if td in rebalance_dates:
            # 用前一天收盘价打分
            scored = score_assets(prev_td, method=method,
                                  roc_period=roc_period, ma_period=ma_period)
            targets = select_targets(scored, method=method)  # [(code, name, weight), ...]

            if verbose:
                score_str = ", ".join(f"{n}({s:.1%})" for _, n, s in scored[:3])
                target_str = ", ".join(f"{n}({w:.0%})" for _, n, w in targets) if targets else CASH_NAME
                print(f"  调仓日 {td}：得分前三={score_str} | 目标={target_str}")

            # 获取目标代码集合
            target_codes = {c for c, _, _ in targets}
            current_codes = set(positions.keys())

            # ── 卖出不在目标中的旧持仓（按开盘价） ──
            for code in list(positions.keys()):
                if code not in target_codes:
                    open_price = get_etf_open(code, td)
                    if open_price is None or open_price <= 0:
                        continue
                    pos = positions[code]
                    proceeds = pos["shares"] * open_price
                    fee = calc_etf_fee('sell', open_price, pos["shares"])
                    cash += proceeds - fee
                    trades.append({
                        "date": td, "action": "SELL", "code": code,
                        "name": next((e["name"] for e in ETF_UNIVERSE if e["code"] == code), code),
                        "price": open_price, "shares": pos["shares"], "reason": "rotation"
                    })
                    if verbose:
                        etf_name = next((e["name"] for e in ETF_UNIVERSE if e["code"] == code), code)
                        print(f"    → 卖出 {etf_name}：{pos['shares']}份 @ {open_price:.3f}")
                    del positions[code]

            # ── 买入新的目标持仓（等权分配） ──
            if targets:
                # 筛选需要买入的（尚未持仓的）
                new_to_buy = [(c, n, w) for c, n, w in targets if c not in positions]
                if new_to_buy:
                    cash_per_target = cash / len(new_to_buy)
                    for code, name, weight in new_to_buy:
                        open_price = get_etf_open(code, td)
                        if open_price is None or open_price <= 0:
                            continue
                        # 分配资金
                        alloc = cash_per_target * weight * len(new_to_buy)  # 实际分配金额
                        alloc = min(alloc, cash)  # 不超过剩余现金
                        max_shares = int(alloc / open_price)
                        if max_shares < 1:
                            continue
                        cost = max_shares * open_price
                        fee = calc_etf_fee('buy', open_price, max_shares)
                        if cost + fee <= cash:
                            cash -= cost + fee
                            positions[code] = {"shares": max_shares, "buy_price": open_price}
                            trades.append({
                                "date": td, "action": "BUY", "code": code,
                                "name": name, "price": open_price,
                                "shares": max_shares, "reason": "rotation"
                            })
                            if verbose:
                                print(f"    → 买入 {name}：{max_shares}份 @ {open_price:.3f}")

            # 如果 targets 为空，全部转现金（已实现：现金自然保留）

    # ── 回测结束：平仓 ──
    if trade_dates:
        last_date = trade_dates[-1]
        for code in list(positions.keys()):
            price = get_etf_price(code, last_date)
            if price is not None:
                pos = positions[code]
                proceeds = pos["shares"] * price
                fee = calc_etf_fee('sell', price, pos["shares"])
                cash += proceeds - fee
                trades.append({
                    "date": last_date, "action": "SELL", "code": code,
                    "name": next((e["name"] for e in ETF_UNIVERSE if e["code"] == code), code),
                    "price": price, "shares": pos["shares"], "reason": "backtest_end"
                })
                del positions[code]

    # ── 计算绩效 ──
    final_value = cash
    total_return = (final_value / capital - 1) * 100
    days = len(trade_dates)
    years = days / 252.0
    annual_return = ((final_value / capital) ** (1 / years) - 1) * 100 if years > 0 else 0

    vals = np.array([d["value"] for d in daily_vals])
    cummax = np.maximum.accumulate(vals)
    safe_cummax = np.where(cummax == 0, 1, np.array(cummax, dtype=float))
    drawdowns = (vals - cummax) / safe_cummax
    max_dd = float(np.min(drawdowns)) * 100

    rets = np.diff(vals) / np.where(vals[:-1] == 0, 1, vals[:-1])
    if len(rets) > 1 and np.std(rets) > 0:
        sharpe = (np.mean(rets) * 252 - 0.025) / (np.std(rets) * np.sqrt(252))
    else:
        sharpe = 0.0

    # ── 基准指数收益 ──
    b_start = get_etf_price(BENCHMARK_CODE, trade_dates[0], scaled=False)
    b_end = get_etf_price(BENCHMARK_CODE, trade_dates[-1], scaled=False)
    idx_return = (b_end / b_start - 1) * 100 if b_start and b_end else 0

    # ── 输出 ──
    profit_amount = final_value - capital
    print(f"\n{'=' * 70}")
    print(f"  ETF轮动策略回测结果（{method_names.get(method, method)}）")
    print(f"{'=' * 70}")
    print(f"  初始资金：{capital:,.2f}")
    print(f"  最终资产：{final_value:,.2f}")
    print(f"  总盈亏：{profit_amount:+,.0f} 元")
    print(f"  总收益率：{total_return:+.2f}%")
    print(f"  年化收益率：{annual_return:+.2f}%")
    print(f"  最大回撤：{max_dd:.2f}%")
    print(f"  夏普比率：{sharpe:.2f}")
    print(f"  交易次数：{len(trades)}（买{sum(1 for t in trades if t['action']=='BUY')}次 / "
          f"卖{sum(1 for t in trades if t['action']=='SELL')}次）")
    print(f"  基准（沪深300）涨幅：{idx_return:+.2f}%")
    print(f"  超额收益：{total_return - idx_return:+.2f}%")

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "trades": len(trades),
        "idx_return": idx_return,
        "final_value": final_value,
    }


# ── CLI 入口 ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETF轮动策略回测")
    parser.add_argument("start_date", nargs="?", default="20200101", help="开始日期 YYYYMMDD")
    parser.add_argument("end_date", nargs="?", default="20251231", help="结束日期 YYYYMMDD")
    parser.add_argument("--method", default="dual",
                        choices=["single", "dual", "ma_filter"],
                        help="调仓方法（默认 dual=双动量法）")
    parser.add_argument("--roc-period", type=int, default=20, help="ROC计算周期（默认20）")
    parser.add_argument("--ma-period", type=int, default=60, help="MA计算周期（默认60）")
    parser.add_argument("--capital", type=int, default=INITIAL_CAPITAL, help="初始资金（默认100000）")
    parser.add_argument("--quiet", action="store_true", help="静默模式")
    args = parser.parse_args()

    run_etf_rotation(
        start_date=args.start_date,
        end_date=args.end_date,
        method=args.method,
        roc_period=args.roc_period,
        ma_period=args.ma_period,
        capital=args.capital,
        verbose=not args.quiet,
    )
