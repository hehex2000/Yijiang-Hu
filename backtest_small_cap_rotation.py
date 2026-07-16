# -*- coding: utf-8 -*-
"""
小市值轮动策略 —— 回测引擎 (v2, 手册对齐版)
==========================================
周频（每周二）等权换仓，满仓持有全市场（沪深，剔除北交所）流通市值最小的 N 只股票（中证2000 风格宇宙）。

相对 MVP(v1) 的核心改进（均来自《小市值量化策略投研手册》）：
  · 无前视：选股用「成交日前一交易日」快照排序，次日开盘成交（手册陷阱1）
  · 流动性过滤：选股剔除日均成交额<3000万；下单限制单票<=当日成交额5%（陷阱7）
  · 幸存者偏差：选股器 LEFT JOIN 含退市股；持仓股连续缺失>30交易日→退市归零（陷阱2）
  · 三层止损（enable_stop_loss 开关）：
      层1 单票自买入价回撤>12% → 清仓该股且一段时间内不再买回
      层2 中证2000单日跌幅>6.6% → 清仓全部、当周空仓
      层3 昨涨停今炸板(周二开盘低于昨涨停价) → 清仓保利润
  · 成本/滑点：佣金万2.5(最低5)+印花千1(卖)；流动性自适应滑点（小盘股顶到上限）
  · 涨跌停：开盘涨停买不进、开盘跌停卖不出（日线策略只看开盘/收盘，不参考盘中高低）
  · 交易统计：轮动胜率（每笔买卖往返盈亏）
"""

import os
import sys
import sqlite3
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA, BACKTEST

DB_PATH = DATA.get("local_db_path", "D:/tu-shareData/astock_daily.db")

# ───────────────────────── 成本与滑点模型 ─────────────────────────
COMMISSION_RATE = 0.00025
COMMISSION_MIN = 5.0
STAMP_RATE = 0.001

# 流动性自适应滑点：base + 冲击系数×参与度(订单额/20日日均成交额)，封顶 MAX_SLIP。
BASE_SLIP = 0.0008
SLIP_IMPACT = 0.10
MAX_SLIP = 0.02
MIN_SLIP = BASE_SLIP

# 流动性限仓：单笔委托额 <= 当日成交额的比例
MAX_PARTICIPATION = 0.05

# 退市判定：持仓股连续无数据交易日数超过该值 → 视为退市，归零清仓
DELIST_MISSING_DAYS = 30
# 层1/层3 止损后禁止买回的交易日数
STOP_EXCLUDE_DAYS = 20

from src.small_cap_rotation_selector import limit_up_ratio, limit_down_ratio, MIN_AVG_AMOUNT_K, POOL_DESC


def trade_cost(side, price, shares):
    """单笔交易成本（元），不含滑点（滑点已并入成交价）。"""
    amt = price * shares
    comm = max(amt * COMMISSION_RATE, COMMISSION_MIN)
    if side == "sell":
        return comm + amt * STAMP_RATE
    return comm


def get_raw_bar(ts_code, td):
    conn = sqlite3.connect(DB_PATH)
    r = conn.execute(
        "SELECT open, high, low, close, pre_close, amount FROM daily WHERE ts_code=? AND trade_date=?",
        (ts_code, td),
    ).fetchone()
    conn.close()
    return r


def get_slippage_rate(ts_code, td, order_amount):
    """流动性自适应滑点（比例）。order_amount 为该笔委托金额（元）。"""
    conn = sqlite3.connect(DB_PATH)
    amts = [
        r[0]
        for r in conn.execute(
            "SELECT amount FROM daily WHERE ts_code=? AND trade_date<=? ORDER BY trade_date DESC LIMIT 20",
            (ts_code, td),
        )
    ]
    conn.close()
    amts = [a for a in amts if a and a > 0]
    if not amts:
        return MAX_SLIP
    avg_amt_yuan = (sum(amts) / len(amts)) * 1000.0
    participation = order_amount / avg_amt_yuan
    rate = BASE_SLIP + SLIP_IMPACT * participation
    return min(max(rate, MIN_SLIP), MAX_SLIP)


def get_hfq_price(ts_code, trade_date, price_type="close"):
    conn = sqlite3.connect(DB_PATH)
    px = pd_read_sql(
        f"SELECT {price_type} AS p FROM daily WHERE ts_code=? AND trade_date<=? ORDER BY trade_date DESC LIMIT 1",
        conn, (ts_code, trade_date),
    )
    fac = pd_read_sql(
        "SELECT adj_factor FROM adj_factor WHERE ts_code=? AND trade_date<=? ORDER BY trade_date DESC LIMIT 1",
        conn, (ts_code, trade_date),
    )
    conn.close()
    if len(px) == 0 or px.iloc[0]["p"] is None:
        return None
    p = float(px.iloc[0]["p"])
    f = fac.iloc[0]["adj_factor"] if (len(fac) > 0 and fac.iloc[0]["adj_factor"] is not None) else None
    if f is None or f == 0:
        return p
    return p * f


def get_index_close(index_code, trade_date):
    conn = sqlite3.connect(DB_PATH)
    df = pd_read_sql(
        "SELECT close FROM index_daily WHERE ts_code=? AND trade_date<=? ORDER BY trade_date DESC LIMIT 1",
        conn, (index_code, trade_date),
    )
    conn.close()
    return float(df.iloc[0]["close"]) if len(df) > 0 and df.iloc[0]["close"] is not None else None


def get_index_drop(index_code, td):
    """指数当日涨跌幅（close/pre_close - 1）。无数据返回 None。"""
    bar = get_raw_bar_index(index_code, td)
    if not bar or bar[0] is None or bar[1] is None or bar[1] == 0:
        return None
    return bar[0] / bar[1] - 1.0


def get_raw_bar_index(index_code, td):
    conn = sqlite3.connect(DB_PATH)
    r = conn.execute(
        "SELECT close, pre_close FROM index_daily WHERE ts_code=? AND trade_date=?",
        (index_code, td),
    ).fetchone()
    conn.close()
    return r


def pd_read_sql(sql, conn, params=()):
    import pandas as pd
    return pd.read_sql_query(sql, conn, params=params)


def get_trade_dates(start_date, end_date):
    conn = sqlite3.connect(DB_PATH)
    df = pd_read_sql(
        "SELECT DISTINCT trade_date FROM daily WHERE trade_date BETWEEN ? AND ? ORDER BY trade_date",
        conn, (start_date, end_date),
    )
    conn.close()
    return df["trade_date"].tolist()


# ───────────────────────── 回测主流程 ─────────────────────────
def run_backtest(
    start_date="20200102",
    end_date="20251231",
    hold_count=7,
    capital=None,
    benchmark="932000.SH",
    empty_jan_apr=False,
    enable_stop_loss=False,
    fundamental_filter=False,
    exclude_delisted=False,
    min_avg_amount_k=None,
    pool_mode="zz2000",
    quiet=False,
):
    """执行小市值轮动回测（v2：无前视 + 流动性 + 三层止损 + 退市清仓）。
    quiet=True 时抑制逐周/年度明细打印，仅保留最终汇总（供对照/敏感性批量调用）。"""
    from src.small_cap_rotation_selector import SmallCapRotationSelector, POOL_DESC

    if capital is None:
        capital = BACKTEST.get("total_capital", 500000)
    TOTAL = capital
    selector = SmallCapRotationSelector(
        hold_count=hold_count,
        fundamental_filter=fundamental_filter,
        exclude_delisted=exclude_delisted,
        min_avg_amount_k=min_avg_amount_k if min_avg_amount_k else MIN_AVG_AMOUNT_K,
        pool_mode=pool_mode,
    )

    trade_dates = get_trade_dates(start_date, end_date)
    if not trade_dates:
        print("[ERROR] 无交易日数据")
        return None

    print(f"\n{'='*70}")
    print(f"  小市值轮动策略 · 回测 (v2 手册对齐)")
    print(f"  区间: {start_date} ~ {end_date}  |  持有: {hold_count} 只 (选股宇宙: {POOL_DESC.get(pool_mode, pool_mode)})")
    print(f"  总资金: {TOTAL:,} 元 (等权满仓)  |  基准: {benchmark}")
    print(f"  空仓1/4月: {'开' if empty_jan_apr else '关'}  |  三层止损: {'开' if enable_stop_loss else '关'}"
          + (f"  |  基本面过滤: {'开' if fundamental_filter else '关'}")
          + (f"  |  含退市股: {'是(LEFT JOIN)' if not exclude_delisted else '否(INNER JOIN)'}"))
    print(f"  成本: 佣金万2.5(最低5) + 印花税千1(卖)")
    print(f"  滑点: 流动性自适应 = base{int(BASE_SLIP*1e4)}bp + {int(SLIP_IMPACT*100)}%×参与度, 上限{int(MAX_SLIP*1e4)}bp")
    print(f"  流动性: 选股日均成交额>=3000万; 单票<=当日成交额{int(MAX_PARTICIPATION*100)}%")
    print(f"  无前视: 选股用成交日前一交易日快照")
    print(f"  涨跌停: 涨停买不进 / 跌停卖不出 (按板块前缀区分: 主板±10%, 创业板/科创板±20%)")
    print(f"{'='*70}\n")

    positions = {}          # {ts_code: {"shares": N, "entry": 买入价(hfq)}}
    cash = TOTAL
    daily_vals = []
    rebalance_count = 0
    n_limit_up_skip = 0
    n_limit_down_hold = 0
    n_delist = 0
    missing_days = {}       # {code: 连续无数据天数}
    stop_exclude = {}       # {code: 剩余禁止买回天数}
    # 交易统计
    trade_wins = 0
    trade_total = 0

    def record_sell(code, revenue, cost):
        nonlocal trade_wins, trade_total
        trade_total += 1
        if revenue > cost:
            trade_wins += 1

    for idx, td in enumerate(trade_dates):
        year = int(td[:4])
        month = int(td[4:6])
        is_tuesday = datetime.strptime(td, "%Y%m%d").weekday() == 1

        # ── 退市判定：持仓股连续无数据 > 阈值 → 归零清仓 ──
        for code in list(positions.keys()):
            px = get_hfq_price(code, td, "close")
            if px is None:
                missing_days[code] = missing_days.get(code, 0) + 1
                if missing_days[code] > DELIST_MISSING_DAYS:
                    cost = positions[code]["shares"] * positions[code]["entry"]
                    record_sell(code, 0.0, cost)   # 归零，记为亏损
                    del positions[code]
                    del missing_days[code]
                    n_delist += 1
            else:
                missing_days[code] = 0

        # ── 当日总市值 ──
        total_value = cash
        for code, info in positions.items():
            close = get_hfq_price(code, td, "close")
            if close:
                total_value += info["shares"] * close
        bench_px = get_index_close(benchmark, td)
        daily_vals.append({"date": td, "value": total_value, "bench": bench_px})

        # ── 换仓日：每周二 ──
        if is_tuesday:
            in_empty_month = empty_jan_apr and month in (1, 4)

            # 三层止损（若开启）：用本周二前一日数据评估
            systemic_clear = False
            if enable_stop_loss:
                prev_td = trade_dates[idx - 1] if idx > 0 else td
                # 层2：中证2000 单日跌幅 > 6.6%
                drop = get_index_drop(benchmark, prev_td)
                if drop is not None and drop <= -0.066:
                    systemic_clear = True
                # 层1/层3：单票回撤>12% / 昨涨停今炸板 → 禁止买回
                for code, info in list(positions.items()):
                    prev_bar = get_raw_bar(code, prev_td)
                    if not prev_bar or prev_bar[0] is None or prev_bar[4] is None:
                        continue
                    prev_close = prev_bar[0]
                    prev_pre = prev_bar[4]
                    prev_up = limit_up_ratio(code, prev_td)
                    # 层1：自买入价回撤>12%
                    if info["entry"] and prev_close < info["entry"] * (1 - 0.12):
                        stop_exclude[code] = STOP_EXCLUDE_DAYS
                    # 层3：昨(=prev_td)涨停，今(周二)开盘低于昨涨停价 → 炸板
                    prev_limit_price = prev_pre * prev_up
                    if prev_close >= prev_limit_price * 0.999:
                        tue_bar = get_raw_bar(code, td)
                        if tue_bar and tue_bar[0] is not None and tue_bar[0] < prev_limit_price:
                            stop_exclude[code] = STOP_EXCLUDE_DAYS

            # 卖出全部旧持仓（开盘价，带滑点；跌停卖不出则续持）
            for code in list(positions.keys()):
                info = positions.pop(code)
                sh = info["shares"]
                entry = info["entry"]
                bar = get_raw_bar(code, td)
                if not bar or bar[0] is None or bar[4] is None:
                    positions[code] = info
                    continue
                open_p = bar[0]
                pre_close = bar[4]
                if open_p / pre_close <= limit_down_ratio(code, td):
                    positions[code] = info
                    n_limit_down_hold += 1
                    continue
                hfq_open = get_hfq_price(code, td, "open")
                if hfq_open is None or hfq_open <= 0:
                    positions[code] = info
                    continue
                slip = get_slippage_rate(code, td, sh * hfq_open)
                eff = hfq_open * (1 - slip)
                revenue = sh * eff - trade_cost("sell", eff, sh)
                cash += revenue
                record_sell(code, revenue, sh * entry)

            if in_empty_month or systemic_clear:
                reason = f"空仓月{month}月" if in_empty_month else "系统性风险(中证2000单日>-6.6%)"
                print(f"  {td} [周二·{reason}] 清仓，保持现金")
                rebalance_count += 1
                # 递减禁止买回计数
                _decay(stop_exclude)
                continue

            # 选股（无前视：用前一交易日快照）
            snapshot = trade_dates[idx - 1] if idx > 0 else td
            codes = selector.select_stocks(snapshot)
            if not codes:
                print(f"  {td} [周二] 选股为空，跳过买入")
                rebalance_count += 1
                _decay(stop_exclude)
                continue

            cash_per = cash / hold_count
            bought = 0
            for code in codes:
                if stop_exclude.get(code, 0) > 0:
                    continue
                bar = get_raw_bar(code, td)
                if not bar or bar[0] is None or bar[4] is None:
                    continue
                if bar[5] is None or bar[5] <= 0:   # 当日停牌
                    continue
                if bar[0] / bar[4] >= limit_up_ratio(code, td):   # 涨停买不进
                    n_limit_up_skip += 1
                    continue
                hfq_open = get_hfq_price(code, td, "open")
                if hfq_open is None or hfq_open <= 0:
                    continue
                slip = get_slippage_rate(code, td, cash_per)
                eff = hfq_open * (1 + slip)
                # 流动性限仓：单笔 <= 当日成交额 5%
                day_amt_yuan = bar[5] * 1000.0
                max_by_liq = int(MAX_PARTICIPATION * day_amt_yuan / eff / 100) * 100
                max_by_cash = int(cash_per / eff / 100) * 100
                max_shares = min(max_by_cash, max_by_liq)
                if max_shares <= 0:
                    continue
                cost = max_shares * eff + trade_cost("buy", eff, max_shares)
                if cost > cash:
                    continue
                positions[code] = {"shares": max_shares, "entry": eff}
                cash -= cost
                missing_days[code] = 0
                bought += 1

            rebalance_count += 1
            _decay(stop_exclude)
            if not quiet:
                print(f"  {td} [周二换仓] 选 {len(codes)} 只, 买入 {bought} 只, 持仓 {len(positions)} 只, 现金 {cash:,.0f}"
                      + (f"  [涨停跳过{n_limit_up_skip}]" if n_limit_up_skip else "")
                      + (f"  [跌停续持{n_limit_down_hold}]" if n_limit_down_hold else "")
                      + (f"  [退市{n_delist}]" if n_delist else ""))

    # ── 回测结束：强制平仓（带卖滑点）──
    if positions:
        last_td = trade_dates[-1]
        for code in list(positions.keys()):
            close = get_hfq_price(code, last_td, "close")
            if close:
                info = positions.pop(code)
                slip = get_slippage_rate(code, last_td, info["shares"] * close)
                eff = close * (1 - slip)
                revenue = info["shares"] * eff - trade_cost("sell", eff, info["shares"])
                cash += revenue
                record_sell(code, revenue, info["shares"] * info["entry"])
        if daily_vals:
            daily_vals[-1]["value"] = cash

    # ── 年报 ──
    year_groups = {}
    for d in daily_vals:
        y = d["date"][:4]
        year_groups.setdefault(y, {"first": d["value"], "first_date": d["date"]})
        year_groups[y]["last"] = d["value"]
        year_groups[y]["last_date"] = d["date"]

    if not quiet:
        print(f"\n{'='*70}")
        print(f"  📊 年度收益对比")
        print(f"{'='*70}")
        print(f"  {'年份':<8}{'策略收益':>10}{'基准收益':>10}{'超额收益':>10}")
        print(f"  {'─'*44}")
        for y in sorted(year_groups.keys()):
            yg = year_groups[y]
            strat_ret = (yg["last"] / yg["first"] - 1) * 100 if yg["first"] > 0 else 0
            b_start = get_index_close(benchmark, yg["first_date"])
            b_end = get_index_close(benchmark, yg["last_date"])
            bench_ret = (b_end / b_start - 1) * 100 if (b_start and b_end and b_start > 0) else 0
            print(f"  {y:<8}{strat_ret:>+9.2f}%{bench_ret:>+9.2f}%{strat_ret-bench_ret:>+9.2f}%")

    # ── 终报 ──
    final_value = daily_vals[-1]["value"]
    total_return = (final_value / TOTAL - 1) * 100
    first_d, last_d = daily_vals[0]["date"], daily_vals[-1]["date"]
    b_total_start = get_index_close(benchmark, first_d)
    b_total_end = get_index_close(benchmark, last_d)
    b_total_ret = (b_total_end / b_total_start - 1) * 100 if (b_total_start and b_total_end) else 0

    days = len(trade_dates)
    years_span = days / 252
    annual_return = ((final_value / TOTAL) ** (1 / years_span) - 1) * 100 if years_span > 0 else 0

    vals = np.array([d["value"] for d in daily_vals])
    cummax = np.maximum.accumulate(vals)
    drawdowns = (vals - cummax) / cummax
    max_dd = float(np.min(drawdowns)) * 100
    rets = np.diff(vals) / vals[:-1]
    sharpe = (np.mean(rets) * 252 - 0.025) / (np.std(rets) * np.sqrt(252)) if len(rets) > 0 else 0
    win_rate = (trade_wins / trade_total * 100) if trade_total > 0 else 0

    if not quiet:
        print(f"\n{'='*70}")
        print(f"  📈 最终汇总")
        print(f"{'='*70}")
        print(f"  初始资金: {TOTAL:>12,.2f}")
        print(f"  最终资产: {final_value:>12,.2f}")
        print(f"  总盈亏:   {final_value-TOTAL:>+12,.2f} 元")
        print(f"  总收益率: {total_return:>+11.2f}%")
        print(f"  年化收益: {annual_return:>+11.2f}%")
        print(f"  基准收益: {b_total_ret:>+11.2f}% ( {benchmark} )")
        print(f"  超额收益: {total_return-b_total_ret:>+11.2f}%")
        print(f"  最大回撤: {max_dd:>+11.2f}%")
        print(f"  夏普比率: {sharpe:>11.4f}")
        print(f"  轮动胜率: {win_rate:>11.2f}%  ({trade_wins}/{trade_total} 笔往返)")
        print(f"  换仓次数: {rebalance_count:>12} 次")
        print(f"  涨停买不进: {n_limit_up_skip:>10} 次  |  跌停卖不出(续持): {n_limit_down_hold:>8} 次  |  退市归零: {n_delist:>6} 次")
        print(f"{'='*70}\n")

    # ── 生成 HTML 净值曲线报告（非 quiet 时）──
    if not quiet:
        try:
            _cfg = {
                "pool_mode": pool_mode,
                "pool_desc": POOL_DESC.get(pool_mode, pool_mode),
                "hold_count": hold_count,
                "start": start_date,
                "end": end_date,
                "capital": TOTAL,
                "empty": empty_jan_apr,
                "stop": enable_stop_loss,
                "fund": fundamental_filter,
                "benchmark": benchmark,
            }
            _metrics = {
                "total_return": total_return,
                "annual_return": annual_return,
                "max_dd": max_dd,
                "sharpe": sharpe,
                "win_rate": win_rate,
                "b_total_ret": b_total_ret,
            }
            _html_path = emit_html_report(daily_vals, year_groups, _cfg, _metrics)
            print(f"  [OK] HTML 净值曲线报告 → {_html_path}\n")
        except Exception as _e:
            print(f"  [WARN] HTML 报告生成失败: {_e}\n")

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "rebalance_count": rebalance_count,
        "daily_values": daily_vals,
    }


def _decay(d):
    """递减禁止买回计数。"""
    for k in list(d.keys()):
        d[k] -= 1
        if d[k] <= 0:
            del d[k]


def emit_html_report(daily_vals, year_groups, cfg, metrics, out_path=None):
    """生成自包含 HTML 回测报告（含分年度净值曲线图 + 年度收益柱状图）。

    纯内联 SVG，无外部 CDN 依赖，可离线打开。
    配色遵循 A 股习惯：涨=红(#e0392b)，跌=绿(#1a9e57)。
    归一化：起点=100（策略/基准各自首值）。
    """
    import os

    if out_path is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(
            out_dir, "sc_backtest_%s_%d_%s_%s.html" % (cfg["pool_mode"], cfg["hold_count"], cfg["start"], cfg["end"])
        )

    benchmark = cfg["benchmark"]

    # ── 归一化序列（起点=100）──
    s0 = daily_vals[0]["value"]
    bvals, lb = [], None
    for d in daily_vals:
        b = d.get("bench")
        if b is not None:
            lb = b
        bvals.append(lb)
    b0 = next((b for b in bvals if b is not None), None)
    dates = [d["date"] for d in daily_vals]
    s_norm = [d["value"] / s0 * 100 for d in daily_vals]
    if b0:
        b_norm = [(b / b0 * 100) if b is not None else None for b in bvals]
        fb = None
        for i, v in enumerate(b_norm):
            if v is not None:
                fb = v
            b_norm[i] = fb
    else:
        b_norm = [None] * len(s_norm)

    # ── 折线图（策略 vs 基准 净值曲线）──
    W, H, PAD = 940, 380, 56
    n = len(s_norm)
    series = list(s_norm) + [v for v in b_norm if v is not None]
    vmin, vmax = min(series), max(series)
    if vmax == vmin:
        vmax = vmin + 1
    vmin = max(0.0, vmin - (vmax - vmin) * 0.05)

    def X(i):
        return PAD + (W - 2 * PAD) * (i / (n - 1) if n > 1 else 0)

    def Y(v):
        return H - PAD - (H - 2 * PAD) * ((v - vmin) / (vmax - vmin))

    s_path = "M " + " L ".join("%.1f,%.1f" % (X(i), Y(v)) for i, v in enumerate(s_norm))
    if b0:
        b_path = "M " + " L ".join("%.1f,%.1f" % (X(i), Y(v)) for i, v in enumerate(b_norm))
    else:
        b_path = ""

    # 年份分隔线 + 标签
    ymarks, last_y = [], None
    for i, dt in enumerate(dates):
        y = dt[:4]
        if y != last_y:
            ymarks.append((X(i), y))
            last_y = y
    ygrid = "".join(
        "<line x1='%d' y1='%.1f' x2='%d' y2='%.1f' stroke='#eee' stroke-width='1'/>" % (PAD, Y(v), W - PAD, Y(v))
        + "<text x='%d' y='%.1f' text-anchor='end' font-size='11' fill='#888'>%.0f</text>" % (PAD - 8, Y(v) + 4, v)
        for k in range(4) for v in [vmin + (vmax - vmin) * k / 3]
    )
    ylines = "".join("<line x1='%.1f' y1='%d' x2='%.1f' y2='%d' stroke='#f0f0f0' stroke-width='1'/>" % (x, PAD, x, H - PAD) for x, _ in ymarks[1:])
    ylabs = "".join("<text x='%.1f' y='%d' text-anchor='middle' font-size='11' fill='#666'>%s</text>" % (x, H - PAD + 18, y) for x, y in ymarks)
    line_svg = (
        "<svg viewBox='0 0 %d %d' width='100%%' preserveAspectRatio='xMidYMid meet' font-family='Menlo,Consolas,monospace'>" % (W, H)
        + ygrid + ylines + ylabs
        + ("<path d='%s' fill='none' stroke='#5b6b7c' stroke-width='1.6' stroke-dasharray='5,4'/>" % b_path if b_path else "")
        + "<path d='%s' fill='none' stroke='#e0392b' stroke-width='2.4'/>" % s_path
        + "</svg>"
    )

    # ── 年度收益柱状图（策略 vs 基准，按涨红跌绿）──
    bars = []
    for y in sorted(year_groups.keys()):
        yg = year_groups[y]
        sr = (yg["last"] / yg["first"] - 1) * 100 if yg["first"] > 0 else 0
        bs = get_index_close(benchmark, yg["first_date"])
        be = get_index_close(benchmark, yg["last_date"])
        br = (be / bs - 1) * 100 if (bs and be and bs > 0) else 0
        bars.append((y, sr, br))
    BW, BH, BP = 940, 260, 56
    nb = len(bars)
    maxabs = max([max(abs(s), abs(b)) for _, s, b in bars] + [1.0])
    zeroy = BH / 2

    def BX(i):
        return BP + (BW - 2 * BP) * (i / (nb - 1) if nb > 1 else 0.5)

    def BY(v):
        return zeroy - (zeroy - 30) * (v / maxabs)

    barw = max(14.0, (BW - 2 * BP) / nb * 0.5)
    rects = ""
    for i, (y, sr, br) in enumerate(bars):
        x = BX(i) - barw / 2
        top = BY(max(sr, 0.0))
        bot = BY(min(sr, 0.0))
        hgt = max(1.0, abs(bot - top))
        color = "#e0392b" if sr >= 0 else "#1a9e57"
        rects += "<rect x='%.1f' y='%.1f' width='%.1f' height='%.1f' fill='%s'/>" % (x, top, barw, hgt, color)
        ty = top - 5 if sr >= 0 else bot + 13
        rects += "<text x='%.1f' y='%.1f' text-anchor='middle' font-size='10' fill='%s'>%+.0f%%</text>" % (BX(i), ty, color, sr)
        rects += "<text x='%.1f' y='%d' text-anchor='middle' font-size='11' fill='#555'>%s</text>" % (BX(i), BH - 12, y)
    bar_svg = (
        "<svg viewBox='0 0 %d %d' width='100%%' font-family='Menlo,Consolas,monospace'>" % (BW, BH)
        + "<line x1='%d' y1='%.1f' x2='%d' y2='%.1f' stroke='#999' stroke-width='1'/>" % (BP, zeroy, BW - BP, zeroy)
        + rects + "</svg>"
    )

    # ── 指标卡 + 年度表 ──
    def card(name, val, cls=""):
        return "<div class='card %s'><div class='cname'>%s</div><div class='cval'>%s</div></div>" % (cls, name, val)

    tr = metrics["total_return"]
    ar = metrics["annual_return"]
    mdd = metrics["max_dd"]
    sh = metrics["sharpe"]
    btr = metrics["b_total_ret"]
    cards = (
        card("总收益率", "%+.2f%%" % tr, "up" if tr >= 0 else "down")
        + card("年化收益", "%+.2f%%" % ar, "up" if ar >= 0 else "down")
        + card("最大回撤", "%+.2f%%" % mdd, "down" if mdd < 0 else "")
        + card("夏普比率", "%.2f" % sh)
        + card("超额基准", "%+.2f%%" % (tr - btr), "up" if (tr - btr) >= 0 else "down")
    )
    rows = ""
    for y, sr, br in bars:
        rows += (
            "<tr><td>%s</td><td class='%s'>%+.2f%%</td><td class='%s'>%+.2f%%</td><td class='%s'>%+.2f%%</td></tr>"
            % (y, "up" if sr >= 0 else "down", sr, "up" if br >= 0 else "down", br, "up" if (sr - br) >= 0 else "down", sr - br)
        )

    cap_str = format(cfg["capital"], ",.0f")
    cfg_line = (
        "选股宇宙=%s(%s) &nbsp; 持仓%d只 &nbsp; 区间%s~%s &nbsp; 总资金%s元<br>"
        "空仓1/4月=%s &nbsp; 三层止损=%s &nbsp; 基本面过滤=%s &nbsp; 基准=%s"
        % (
            cfg["pool_mode"], cfg["pool_desc"], cfg["hold_count"], cfg["start"], cfg["end"], cap_str,
            "开" if cfg["empty"] else "关", "开" if cfg["stop"] else "关", "开" if cfg["fund"] else "关", benchmark,
        )
    )

    html = """<!DOCTYPE html>
<html lang='zh-CN'>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>小市值轮动回测 · 净值曲线</title>
<style>
  body { margin:0; background:#f5f6f8; color:#222; font-family:'Segoe UI','Microsoft YaHei',sans-serif; }
  .wrap { max-width:1000px; margin:0 auto; padding:24px 18px 60px; }
  h1 { font-size:20px; margin:0 0 4px; }
  .sub { color:#888; font-size:13px; line-height:1.6; margin-bottom:18px; }
  .cards { display:flex; flex-wrap:wrap; gap:12px; margin-bottom:22px; }
  .card { flex:1 1 150px; background:#fff; border-radius:10px; padding:14px 16px; box-shadow:0 1px 3px rgba(0,0,0,.06); }
  .cname { font-size:12px; color:#999; margin-bottom:6px; }
  .cval { font-size:22px; font-weight:700; font-family:Menlo,Consolas,monospace; }
  .up { color:#e0392b; } .down { color:#1a9e57; }
  .panel { background:#fff; border-radius:10px; padding:18px 18px 8px; box-shadow:0 1px 3px rgba(0,0,0,.06); margin-bottom:20px; }
  .ptitle { font-size:15px; font-weight:600; margin:0 0 2px; }
  .pnote { font-size:12px; color:#999; margin:0 0 10px; }
  .legend { font-size:12px; color:#666; margin:2px 0 6px; }
  .legend i { display:inline-block; width:18px; height:3px; vertical-align:middle; margin:0 4px 0 12px; }
  table { width:100%; border-collapse:collapse; font-size:13px; margin-top:6px; }
  th,td { padding:7px 8px; text-align:right; border-bottom:1px solid #eee; font-family:Menlo,Consolas,monospace; }
  th:first-child,td:first-child { text-align:left; font-family:inherit; }
  th { color:#999; font-weight:600; background:#fafafa; }
</style>
</head>
<body>
<div class='wrap'>
  <h1>小市值轮动策略 · 回测净值曲线</h1>
  <div class='sub'>[[CFG]]</div>
  <div class='cards'>[[CARDS]]</div>

  <div class='panel'>
    <div class='ptitle'>净值曲线（起点=100）</div>
    <div class='pnote'>按年分段展示；红色为策略、灰色虚线为基准指数。</div>
    <div class='legend'><i style='background:#e0392b'></i>策略<i style='background:#5b6b7c'></i>基准([[BENCH]])</div>
    [[LINE_SVG]]
  </div>

  <div class='panel'>
    <div class='ptitle'>分年度收益（策略 vs 基准）</div>
    <div class='pnote'>柱为策略当年收益，按涨红跌绿着色；数字为策略收益率。</div>
    [[BAR_SVG]]
    <table>
      <thead><tr><th>年份</th><th>策略收益</th><th>基准收益</th><th>超额收益</th></tr></thead>
      <tbody>[[ROWS]]</tbody>
    </table>
  </div>
  <div class='sub'>本报告由 backtest_small_cap_rotation.py 自动生成（emit_html_report），所有数据基于无前视回测；曲线/柱状图纯内联 SVG，可离线打开。</div>
</div>
</body>
</html>
"""
    html = (
        html.replace("[[CFG]]", cfg_line)
        .replace("[[CARDS]]", cards)
        .replace("[[LINE_SVG]]", line_svg)
        .replace("[[BAR_SVG]]", bar_svg)
        .replace("[[ROWS]]", rows)
        .replace("[[BENCH]]", benchmark)
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path


def run_survivor_bias_comparison(
    start_date="20200102", end_date="20251231", hold_count=7,
    capital=None, benchmark="932000.SH", empty_jan_apr=False,
    enable_stop_loss=False, fundamental_filter=False, pool_mode="zz2000",
):
    """对照：含退市股(LEFT JOIN) vs 剔除退市股(INNER JOIN)，量化幸存者偏差幅度。"""
    print(f"\n{'='*70}")
    print(f"  🔬 幸存者偏差对照（含退市 vs 剔除退市）")
    print(f"  区间: {start_date} ~ {end_date}  |  持仓: {hold_count} 只  |  选股宇宙: {POOL_DESC.get(pool_mode, pool_mode)}")
    print(f"  空仓1/4月: {'开' if empty_jan_apr else '关'}  |  三层止损: {'开' if enable_stop_loss else '关'}")
    print(f"{'='*70}")

    r_with = run_backtest(start_date, end_date, hold_count=hold_count, capital=capital,
                          benchmark=benchmark, empty_jan_apr=empty_jan_apr,
                          enable_stop_loss=enable_stop_loss, fundamental_filter=fundamental_filter,
                          exclude_delisted=False, pool_mode=pool_mode, quiet=False)
    r_without = run_backtest(start_date, end_date, hold_count=hold_count, capital=capital,
                             benchmark=benchmark, empty_jan_apr=empty_jan_apr,
                             enable_stop_loss=enable_stop_loss, fundamental_filter=fundamental_filter,
                             exclude_delisted=True, pool_mode=pool_mode, quiet=True)

    if r_with is None or r_without is None:
        print("  [ERROR] 对照运行失败")
        return

    def fmt(v, pct=True):
        return f"{v:+.2f}%" if pct else f"{v:.2f}"

    print(f"\n  {'指标':<12}{'含退市(LEFT)':>16}{'剔除退市(INNER)':>18}{'偏差(虚增)':>14}")
    print(f"  {'─'*58}")
    rows = [
        ("总收益率", r_with["total_return"], r_without["total_return"]),
        ("年化收益", r_with["annual_return"], r_without["annual_return"]),
        ("最大回撤", r_with["max_drawdown"], r_without["max_drawdown"]),
        ("轮动胜率", r_with["win_rate"], r_without["win_rate"]),
        ("夏普比率", r_with["sharpe"], r_without["sharpe"]),
    ]
    for name, a, b in rows:
        print(f"  {name:<12}{fmt(a):>16}{fmt(b):>18}{fmt(a-b):>14}")

    # 方向澄清：含退市(LEFT)=真实诚实版；剔除退市(INNER)=幸存者偏差美化版
    bias_pp = r_without["total_return"] - r_with["total_return"]      # 剔除版比含退市版虚高多少个百分点
    annual_bias_pp = r_without["annual_return"] - r_with["annual_return"]
    inflate_pct = ((1 + r_without["total_return"] / 100) / (1 + r_with["total_return"] / 100) - 1) * 100

    print(f"\n  📌 结论：剔除退市股（幸存者偏差版 / INNER JOIN）相比含退市股（真实版 / LEFT JOIN）"
          f"\n      虚增总收益率 {bias_pp:+.2f} 个百分点，年化 {annual_bias_pp:+.2f} 个百分点；"
          f"\n      若错误地剔除退市股，回测收益会被高估约 {inflate_pct:+.1f}%（真实含退市版仅 {r_with['total_return']:+.2f}%）。"
          f"\n  ✅ 本策略默认采用 LEFT JOIN 含退市股，如实反映退市归零拖累，不存在幸存者偏差美化。")


def run_sensitivity(
    start_date="20200102", end_date="20251231", capital=None, benchmark="932000.SH",
    empty_jan_apr=False, enable_stop_loss=False, fundamental_filter=False,
    hold_grid=(5, 7, 10, 15), liq_grid=(30000, 50000, 80000, 100000), pool_mode="zz2000",
):
    """参数敏感性：持仓数 × 流动性门槛网格回测。"""
    import csv, os
    print(f"\n{'='*70}")
    print(f"  🔬 参数敏感性分析（持仓数 × 流动性门槛）")
    print(f"  区间: {start_date} ~ {end_date}  |  基准: {benchmark}  |  选股宇宙: {POOL_DESC.get(pool_mode, pool_mode)}")
    print(f"  空仓1/4月: {'开' if empty_jan_apr else '关'}  |  三层止损: {'开' if enable_stop_loss else '关'}")
    print(f"{'='*70}")

    header = ["持仓数", "流动性门槛(万)", "总收益率", "年化收益", "最大回撤", "夏普", "胜率", "换仓次数"]
    table = []
    for hc in hold_grid:
        for liq_k in liq_grid:
            r = run_backtest(start_date, end_date, hold_count=hc, capital=capital,
                             benchmark=benchmark, empty_jan_apr=empty_jan_apr,
                             enable_stop_loss=enable_stop_loss, fundamental_filter=fundamental_filter,
                             min_avg_amount_k=liq_k, pool_mode=pool_mode, quiet=True)
            if r is None:
                continue
            table.append({
                "持仓数": hc, "流动性门槛(万)": liq_k / 10.0,
                "总收益率": round(r["total_return"], 2), "年化收益": round(r["annual_return"], 2),
                "最大回撤": round(r["max_drawdown"], 2), "夏普": round(r["sharpe"], 3),
                "胜率": round(r["win_rate"], 2), "换仓次数": r["rebalance_count"],
            })
            print(f"  持仓{hc:<3} 门槛{liq_k/10:>6.0f}万 → "
                  f"总收益{r['total_return']:>+8.2f}%  年化{r['annual_return']:>+8.2f}%  "
                  f"回撤{r['max_drawdown']:>7.2f}%  夏普{r['sharpe']:>6.3f}  胜率{r['win_rate']:>6.2f}%")

    # 保存 CSV
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"sc_sensitivity_{start_date}_{end_date}.csv")
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for row in table:
            w.writerow(row)
    print(f"\n  [OK] 敏感性结果已保存 → {out_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="小市值轮动策略 · 回测引擎 (v2)")
    p.add_argument("--start-date", default="20200102", help="回测开始 YYYYMMDD")
    p.add_argument("--end-date", default="20251231", help="回测结束 YYYYMMDD")
    p.add_argument("--hold-count", type=int, default=7, help="持仓只数(流通市值最小N只)")
    p.add_argument("--capital", type=int, default=None, help="总资金(元), 默认读 config.total_capital")
    p.add_argument("--benchmark", default="932000.SH", help="基准指数代码(层2风控用)")
    p.add_argument("--empty-jan-apr", action="store_true", help="1/4月空仓(年报/一季报窗口)")
    p.add_argument("--stop-loss", action="store_true", help="开启三层止损")
    p.add_argument("--fundamental", action="store_true", help="开启基本面过滤(最近年报 eps>0 盈利, 基于 fina_indicator; 数据缺失则保留)")
    p.add_argument("--exclude-delisted", action="store_true", help="剔除已退市股(INNER JOIN, 用于幸存者偏差对照)")
    p.add_argument("--min-avg-amount-k", type=float, default=None, help="流动性门槛(日均成交额,千元), 默认30000(=3000万)")
    p.add_argument("--pool-mode", choices=list(POOL_DESC.keys()), default="zz2000",
                   help="选股宇宙: cyb(纯创业板) / zz2000(中证2000风格·含微盘尾) / zz1000(中证1000风格·剔除微盘尾)")
    p.add_argument("--mode", choices=["single", "compare", "sensitivity"], default="single",
                   help="single=单次回测; compare=含退市vs剔除退市对照; sensitivity=持仓数/流动性网格")
    p.add_argument("--hold-grid", default="5,7,10,15", help="sensitivity模式: 持仓数网格(逗号分隔)")
    p.add_argument("--liq-grid", default="30000,50000,80000,100000", help="sensitivity模式: 流动性门槛网格(千元)")
    args = p.parse_args()

    if args.mode == "single":
        run_backtest(
            args.start_date, args.end_date, hold_count=args.hold_count,
            capital=args.capital, benchmark=args.benchmark,
            empty_jan_apr=args.empty_jan_apr, enable_stop_loss=args.stop_loss,
            fundamental_filter=args.fundamental, exclude_delisted=args.exclude_delisted,
            min_avg_amount_k=args.min_avg_amount_k, pool_mode=args.pool_mode,
        )
    elif args.mode == "compare":
        run_survivor_bias_comparison(
            args.start_date, args.end_date, hold_count=args.hold_count,
            capital=args.capital, benchmark=args.benchmark,
            empty_jan_apr=args.empty_jan_apr, enable_stop_loss=args.stop_loss,
            fundamental_filter=args.fundamental, pool_mode=args.pool_mode,
        )
    elif args.mode == "sensitivity":
        run_sensitivity(
            args.start_date, args.end_date, capital=args.capital,
            benchmark=args.benchmark, empty_jan_apr=args.empty_jan_apr,
            enable_stop_loss=args.stop_loss, fundamental_filter=args.fundamental,
            hold_grid=[int(x) for x in args.hold_grid.split(",")],
            liq_grid=[float(x) for x in args.liq_grid.split(",")],
            pool_mode=args.pool_mode,
        )
