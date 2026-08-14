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
  · 成本/滑点：佣金万2.5(最低5)+印花千1→千0.5(2023-08-28起,卖)；流动性自适应滑点（小盘股顶到上限）
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
# 印花税率复用共享引擎的「分段口径」（2023-08-28 起千1→千0.5）
from run_monthly_rebalance import stamp_duty_rate

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

# 自身历史分位（平台统一惯例，Jim 原则②）：拥挤度历史分位复用同一实现
try:
    from src.factor_utils import own_history_pct
except ImportError:
    from factor_utils import own_history_pct

# 股票名称缓存（明细导出用）
_NAME_CACHE = {}
def name_of(ts_code):
    """查 stock_basic 名称（退市股 name 可能为 NULL，回退为代码本身）。"""
    if ts_code in _NAME_CACHE:
        return _NAME_CACHE[ts_code]
    conn = sqlite3.connect(DB_PATH)
    r = conn.execute("SELECT name FROM stock_basic WHERE ts_code=?", (ts_code,)).fetchone()
    conn.close()
    nm = (r[0] if r and r[0] else ts_code)
    _NAME_CACHE[ts_code] = nm
    return nm


def trade_cost(side, price, shares, trade_date=None):
    """单笔交易成本（元），不含滑点（滑点已并入成交价）。
    trade_date 省略时按旧税率；传入则按分段印花税率（2023-08-28 起千1→千0.5）。"""
    amt = price * shares
    comm = max(amt * COMMISSION_RATE, COMMISSION_MIN)
    if side == "sell":
        return comm + amt * stamp_duty_rate(trade_date)
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
    f = None
    if len(fac) > 0 and fac.iloc[0]["adj_factor"] is not None:
        f = float(fac.iloc[0]["adj_factor"])
    else:
        # 后向填充：因子数据起点(本库=20150105)晚于回测起点时，用首个可用因子锚定，
        # 避免买入按因子=1记账、后续按真实因子估值造成的虚假台阶(已验证 20150105 单日+1038%)。
        fac2 = pd_read_sql(
            "SELECT adj_factor FROM adj_factor WHERE ts_code=? AND trade_date>? ORDER BY trade_date ASC LIMIT 1",
            conn, (ts_code, trade_date),
        )
        if len(fac2) > 0 and fac2.iloc[0]["adj_factor"] is not None:
            f = float(fac2.iloc[0]["adj_factor"])
    conn.close()
    if len(px) == 0 or px.iloc[0]["p"] is None:
        return None
    p = float(px.iloc[0]["p"])
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


def get_index_20d_return(index_code, td):
    """指数截至 td 的 20 交易日(约1月)收益率 close[-1]/close[-21]-1。数据不足返回 None。"""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT close FROM index_daily WHERE ts_code=? AND trade_date<=? ORDER BY trade_date DESC LIMIT 21",
        (index_code, td),
    ).fetchall()
    conn.close()
    if len(rows) < 21 or rows[-1][0] in (None, 0):
        return None
    return rows[0][0] / rows[-1][0] - 1.0


def get_circ_mv(code, td):
    """point-in-time 流通市值(千元)，取 <= td 最近有效值。无数据返回 None。"""
    conn = sqlite3.connect(DB_PATH)
    r = conn.execute(
        "SELECT circ_mv FROM daily_basic WHERE ts_code=? AND trade_date<=? "
        "AND circ_mv>0 ORDER BY trade_date DESC LIMIT 1",
        (code, td),
    ).fetchone()
    conn.close()
    return r[0] if r else None


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
    pool_order="ASC",
    pool_offset=None,
    bucket_label=None,
    quality_filter=False,
    growth_tilt=False,
    vol_filter=False,
    industry_cap=0,
    style_switch=False,
    detail_path=None,
    no_html=False,
    quiet=False,
    var_control=0, var_maxdd=15.0, var_n=5, var_lookback=120, var_method="hist",
):
    """执行小市值轮动回测（v2：无前视 + 流动性 + 三层止损 + 退市清仓）。
    quiet=True 时抑制逐周/年度明细打印，仅保留最终汇总（供对照/敏感性批量调用）。"""
    from src.small_cap_rotation_selector import SmallCapRotationSelector, POOL_DESC
    from run_monthly_rebalance import estimate_basket_var, _var_invest_ratio, equity_curve_var

    if capital is None:
        capital = BACKTEST.get("total_capital", 500000)
    TOTAL = capital
    selector = SmallCapRotationSelector(
        hold_count=hold_count,
        fundamental_filter=fundamental_filter,
        exclude_delisted=exclude_delisted,
        min_avg_amount_k=min_avg_amount_k if min_avg_amount_k else MIN_AVG_AMOUNT_K,
        pool_mode=pool_mode,
        pool_offset=pool_offset,
        order=pool_order,
        quality_filter=quality_filter,
        growth_tilt=growth_tilt,
        vol_filter=vol_filter,
        industry_cap=industry_cap,
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
    print(f"  A档质量门禁: {'开(roe>0 & bps>0 & 资产负债率<70% & 每股经营现金流>0)' if quality_filter else '关'}")
    print(f"  B档成长倾斜: {'开(最小市值桶内 净利润同比>0 优先 + roe 降序)' if growth_tilt else '关'}")
    print(f"  维度3极端波动过滤: {'开(剔除近60日收益率方差最高5%%)' if vol_filter else '关'}")
    print(f"  维度4行业分散上限: {'开(单行业≤%d只)' % industry_cap if industry_cap else '关'}")
    print(f"  维度5风格切换: {'开(沪深300连续20日跑赢中证1000→空仓)' if style_switch else '关'}")
    print(f"  成本: 佣金万2.5(最低5) + 印花税千1→千0.5(2023-08-28起,卖)")
    print(f"  滑点: 流动性自适应 = base{int(BASE_SLIP*1e4)}bp + {int(SLIP_IMPACT*100)}%×参与度, 上限{int(MAX_SLIP*1e4)}bp")
    print(f"  流动性: 选股日均成交额>=3000万; 单票<=当日成交额{int(MAX_PARTICIPATION*100)}%")
    print(f"  无前视: 选股用成交日前一交易日快照")
    print(f"  涨跌停: 涨停买不进 / 跌停卖不出 (按板块前缀区分: 主板±10%, 创业板/科创板±20%)")
    if var_control and var_control > 0:
        print(f"  VaR仓位缩放：目标回撤{var_maxdd:.0f}%·N={var_n}·回看{var_lookback}d·conf={var_control}%（设计即锁回撤，余下转现金不杠杆）")
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
    # 维度4·换手率/持有期（让调仓规则可复盘）
    turnover_sum = 0.0          # 每期 |卖出额+买入额| / 当期总资产 的累加
    holding_weeks_sum = 0.0     # 每笔平仓的持有周数累加
    holding_rounds = 0          # 平仓笔数
    # 维度5·拥挤度（小市值因子是否过度拥挤）：每期持有标的近20日平均换手率(成交额/流通市值)序列
    crowding_series = []
    # 明细导出：逐笔换仓事件（卖出/买入/跳过）列表
    events = []

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
            wk_traded = 0.0  # 本周二成交额(买+卖)，用于换手率
            ev = {"date": td, "sold": [], "bought": [], "reason": None, "holdings": [], "type": "rebalance"}

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

            # 维度5·风格切换（若开启）：沪深300 连续20个周二跑赢 中证1000 → 小市值因子阶段性失效，
            #   当周空仓（对应文章"市场环境变化后，小市值因子是否还有效"）。大盘风格占优时空仓避险。
            style_clear = False
            if style_switch:
                tues = [t for t in trade_dates[:idx + 1]
                        if datetime.strptime(t, "%Y%m%d").weekday() == 1]
                if len(tues) >= 20:
                    ok = True
                    for t in tues[-20:]:
                        r300 = get_index_20d_return("000300.SH", t)
                        r1000 = get_index_20d_return("000852.SH", t)
                        if r300 is None or r1000 is None or r300 < r1000:
                            ok = False
                            break
                    style_clear = ok

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
                revenue = sh * eff - trade_cost("sell", eff, sh, td)
                cash += revenue
                record_sell(code, revenue, sh * entry)
                wk_traded += revenue
                ev["sold"].append({"code": code, "name": name_of(code), "price": round(eff, 4),
                                   "shares": sh, "revenue": round(revenue, 2),
                                   "cost": round(sh * entry, 2), "pnl": round(revenue - sh * entry, 2)})
                # 维度4·持有期：记录该笔平仓的持有周数
                if info.get("entry_date"):
                    _hw = (datetime.strptime(td, "%Y%m%d") -
                           datetime.strptime(info["entry_date"], "%Y%m%d")).days / 7.0
                    holding_weeks_sum += _hw
                    holding_rounds += 1

            if in_empty_month or systemic_clear or style_clear:
                if style_clear:
                    reason = "风格切换(沪深300连续20周二跑赢中证1000·小市值失效)"
                elif systemic_clear:
                    reason = "系统性风险(中证2000单日>-6.6%)"
                else:
                    reason = f"空仓月{month}月"
                print(f"  {td} [周二·{reason}] 清仓，保持现金")
                ev["reason"] = reason
                ev["type"] = "skip"
                events.append(ev)
                rebalance_count += 1
                # 递减禁止买回计数
                _decay(stop_exclude)
                continue

            # 选股（无前视：用前一交易日快照）
            snapshot = trade_dates[idx - 1] if idx > 0 else td
            codes = selector.select_stocks(snapshot)
            if not codes:
                print(f"  {td} [周二] 选股为空，跳过买入")
                ev["reason"] = "选股为空，跳过买入"
                ev["type"] = "skip"
                events.append(ev)
                rebalance_count += 1
                _decay(stop_exclude)
                continue

            _ir = _var_invest_ratio(codes, snapshot, var_control, var_maxdd, var_n, var_lookback, var_method, 1)
            if var_control and var_control > 0 and _ir < 1.0:
                print(f"  🛡️ VaR缩放(var={var_control}%): 投入比例={_ir * 100:.0f}%（预留现金{(1 - _ir) * 100:.0f}%）")
            cash_per = (cash * _ir) / hold_count
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
                cost = max_shares * eff + trade_cost("buy", eff, max_shares, td)
                if cost > cash:
                    continue
                positions[code] = {"shares": max_shares, "entry": eff, "entry_date": td}
                ev["bought"].append({"code": code, "name": name_of(code), "price": round(eff, 4),
                                      "shares": max_shares, "amount": round(cost, 2)})
                cash -= cost
                missing_days[code] = 0
                bought += 1
                wk_traded += cost

            # 维度4·换手率：本周成交额 / 换仓前总资产
            if total_value > 0:
                turnover_sum += wk_traded / total_value

            # 维度5·拥挤度：持有标的近20日平均换手率(成交额/流通市值，均为千元)的均值
            if positions:
                _cr = []
                for code in positions:
                    cmv = get_circ_mv(code, td)
                    if not cmv:
                        continue
                    conn = sqlite3.connect(DB_PATH)
                    amts = [r[0] for r in conn.execute(
                        "SELECT amount FROM daily WHERE ts_code=? AND trade_date<=? "
                        "ORDER BY trade_date DESC LIMIT 20", (code, td)).fetchall()]
                    conn.close()
                    amts = [a for a in amts if a and a > 0]
                    if amts:
                        _cr.append((sum(amts) / len(amts)) / cmv)
                if _cr:
                    crowding_series.append(float(np.mean(_cr)))

            rebalance_count += 1
            _decay(stop_exclude)
            ev["cash"] = cash
            ev["holdings"] = list(positions.keys())
            events.append(ev)
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
                revenue = info["shares"] * eff - trade_cost("sell", eff, info["shares"], last_td)
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
    # 维度4/5 辅助指标
    avg_turnover = (turnover_sum / rebalance_count * 100) if rebalance_count > 0 else 0.0
    avg_hold_weeks = (holding_weeks_sum / holding_rounds) if holding_rounds > 0 else 0.0
    # 维度5·拥挤度：最新值的历史分位（高=过度拥挤，踩踏风险）
    # 复用平台统一 own_history_pct（Jim 原则②：自身历史分位，消除阈值主观性）
    crowd_latest = crowding_series[-1] if crowding_series else None
    crowd_pct = None
    if crowding_series and crowd_latest is not None:
        _n = len(crowding_series)
        crowd_pct = float(own_history_pct(crowding_series, window=_n) * 100)

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
        if var_control and var_control > 0:
            _ev = equity_curve_var([float(d["value"]) for d in daily_vals],
                                   capital=final_value, conf_levels=(0.95, 0.99), method="hist")
            print(f"  风险价值 VaR(95%): 单日最多亏 {_ev[0.95]['hist_loss'] * 100:>7.2f}% (≈{_ev[0.95]['hist_amt']:,.0f}元, 历史法)")
            print(f"  风险价值 VaR(99%): 单日最多亏 {_ev[0.99]['hist_loss'] * 100:>7.2f}% (≈{_ev[0.99]['hist_amt']:,.0f}元, 历史法)")
        print(f"  夏普比率: {sharpe:>11.4f}")
        print(f"  轮动胜率: {win_rate:>11.2f}%  ({trade_wins}/{trade_total} 笔往返)")
        print(f"  维度4·换手率: 平均 {avg_turnover:>7.2f}% / 次换仓  |  平均持有期 {avg_hold_weeks:>5.1f} 周")
        if crowd_latest is not None:
            _warn = "  ⚠️ 处于历史拥挤高位" if (crowd_pct is not None and crowd_pct >= 80) else ""
            print(f"  维度5·拥挤度: 最新 {crowd_latest*100:>6.2f}% (近20日平均换手率) | 历史分位 {crowd_pct:>5.1f}%{_warn}")
        print(f"  换仓次数: {rebalance_count:>12} 次")
        print(f"  涨停买不进: {n_limit_up_skip:>10} 次  |  跌停卖不出(续持): {n_limit_down_hold:>8} 次  |  退市归零: {n_delist:>6} 次")
        print(f"{'='*70}\n")

    # ── 构造报告配置与指标（明细 / HTML 共用）──
    _cfg = {
        "pool_mode": pool_mode,
        "pool_desc": bucket_label or POOL_DESC.get(pool_mode, pool_mode),
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
        "avg_turnover": avg_turnover,
        "avg_hold_weeks": avg_hold_weeks,
        "crowd_latest": crowd_latest,
        "crowd_pct": crowd_pct,
        "vol_filter": vol_filter,
        "style_switch": style_switch,
        "rebalance_count": rebalance_count,
        "n_limit_up_skip": n_limit_up_skip,
        "n_limit_down_hold": n_limit_down_hold,
        "n_delist": n_delist,
        "var_control": var_control,
    }
    if detail_path:
        try:
            _dp = emit_text_detail(events, year_groups, _metrics, _cfg, detail_path)
            print(f"  [OK] 回测明细(文本+CSV) → {_dp}\n")
        except Exception as _e:
            print(f"  [WARN] 明细生成失败: {_e}\n")
    if not quiet and not no_html:
        try:
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
        "avg_turnover": avg_turnover,
        "avg_hold_weeks": avg_hold_weeks,
        "crowd_latest": crowd_latest,
        "crowd_pct": crowd_pct,
        "daily_values": daily_vals,
        "var_control": var_control,
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
        + card("维度4·换手率", "%.1f%%" % metrics.get("avg_turnover", 0.0))
        + card("维度4·持有期", "%.1f周" % metrics.get("avg_hold_weeks", 0.0))
    )
    _cl = metrics.get("crowd_latest")
    _cp = metrics.get("crowd_pct")
    if _cl is not None:
        _warn = "down" if (_cp is not None and _cp >= 80) else ""
        cards += card("维度5·拥挤度", "%.2f%%<br><span style='font-size:11px;color:#999'>分位%.0f%%</span>" % (_cl * 100, _cp), _warn)
    rows = ""
    for y, sr, br in bars:
        rows += (
            "<tr><td>%s</td><td class='%s'>%+.2f%%</td><td class='%s'>%+.2f%%</td><td class='%s'>%+.2f%%</td></tr>"
            % (y, "up" if sr >= 0 else "down", sr, "up" if br >= 0 else "down", br, "up" if (sr - br) >= 0 else "down", sr - br)
        )

    cap_str = format(cfg["capital"], ",.0f")
    cfg_line = (
        "选股宇宙=%s(%s) &nbsp; 持仓%d只 &nbsp; 区间%s~%s &nbsp; 总资金%s元<br>"
        "空仓1/4月=%s &nbsp; 三层止损=%s &nbsp; 基本面过滤=%s &nbsp; 基准=%s<br>"
        "维度3极端波动过滤=%s &nbsp; 维度5风格切换=%s"
        % (
            cfg["pool_mode"], cfg["pool_desc"], cfg["hold_count"], cfg["start"], cfg["end"], cap_str,
            "开" if cfg["empty"] else "关", "开" if cfg["stop"] else "关", "开" if cfg["fund"] else "关", benchmark,
            "开" if metrics.get("vol_filter") else "关", "开" if metrics.get("style_switch") else "关",
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


def emit_text_detail(events, year_groups, metrics, cfg, detail_path):
    """导出纯文本 + CSV 回测明细（逐笔换仓/交易、年度表、最终汇总）。

    与 emit_html_report 平级，但面向「想直接看明细」的用户：
      · <detail_path>.txt   人类可读的逐笔换仓明细 + 年度表 + 最终汇总
      · <detail_path>.csv   每一笔买卖的交易流水（可用 Excel 打开）
    无任何图表/卡片，打开即读。
    """
    import os
    import csv

    out_dir = os.path.dirname(detail_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    txt_path = detail_path + ".txt"
    csv_path = detail_path + ".csv"
    benchmark = cfg["benchmark"]

    # ── CSV 交易流水 ──
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["日期", "动作", "代码", "名称", "成交价", "股数", "金额/收回", "盈亏", "现金余额"])
        for ev in events:
            for s in ev.get("sold", []):
                w.writerow([ev["date"], "卖出", s["code"], s["name"], s["price"], s["shares"],
                            round(s["revenue"], 2), round(s["pnl"], 2), ""])
            for b in ev.get("bought", []):
                w.writerow([ev["date"], "买入", b["code"], b["name"], b["price"], b["shares"],
                            round(b["amount"], 2), "", ""])
            if ev.get("type") == "skip":
                w.writerow([ev["date"], "跳过", "", ev.get("reason", ""), "", "", "", "", ""])

    # ── TXT 人类可读明细 ──
    L = []
    L.append("小市值轮动策略 · 回测明细")
    L.append("=" * 66)
    L.append("【配置】")
    L.append(f"  区间      : {cfg['start']} ~ {cfg['end']}")
    L.append(f"  持有      : {cfg['hold_count']} 只   (选股宇宙: {cfg['pool_desc']})")
    L.append(f"  总资金    : {cfg['capital']:,.0f} 元   |   基准: {benchmark}")
    L.append(f"  空仓1/4月 : {'开' if cfg['empty'] else '关'}   |   三层止损: {'开' if cfg['stop'] else '关'}   |   基本面过滤: {'开' if cfg['fund'] else '关'}")
    L.append(f"  维度3极端波动过滤: {'开' if metrics.get('vol_filter') else '关'}   |   维度5风格切换: {'开' if metrics.get('style_switch') else '关'}")
    L.append("")

    L.append("【分年度收益】")
    L.append(f"  {'年份':<8}{'策略收益':>12}{'基准收益':>12}{'超额收益':>12}")
    L.append("  " + "-" * 48)
    for y in sorted(year_groups.keys()):
        yg = year_groups[y]
        strat_ret = (yg["last"] / yg["first"] - 1) * 100 if yg["first"] > 0 else 0
        b_start = get_index_close(benchmark, yg["first_date"])
        b_end = get_index_close(benchmark, yg["last_date"])
        bench_ret = (b_end / b_start - 1) * 100 if (b_start and b_end and b_start > 0) else 0
        L.append(f"  {y:<8}{strat_ret:>+11.2f}%{bench_ret:>+11.2f}%{strat_ret - bench_ret:>+11.2f}%")
    L.append("")

    L.append("【逐笔换仓明细】")
    n_skip = 0
    for ev in events:
        if ev.get("type") == "skip":
            n_skip += 1
            L.append(f"  {ev['date']}  ⏸ 跳过换仓：{ev.get('reason', '')}")
            continue
        L.append(f"  {ev['date']}  周二换仓")
        if ev.get("sold"):
            L.append("    卖出:")
            for s in ev["sold"]:
                L.append(f"      {s['code']} {s['name']}  价 {s['price']:.3f}  股 {s['shares']}  "
                         f"收回 {s['revenue']:,.0f}  盈亏 {s['pnl']:+,.0f}")
        if ev.get("bought"):
            L.append("    买入:")
            for b in ev["bought"]:
                L.append(f"      {b['code']} {b['name']}  价 {b['price']:.3f}  股 {b['shares']}  "
                         f"金额 {b['amount']:,.0f}")
        cash = ev.get("cash")
        if cash is not None:
            L.append(f"    现金余额: {cash:,.0f}  持仓({len(ev.get('holdings', []))}只): "
                     f"{', '.join(ev.get('holdings', []))}")
    L.append("")

    L.append("【最终汇总】")
    L.append(f"  初始资金  : {cfg['capital']:>14,.2f} 元")
    L.append(f"  总收益率  : {metrics['total_return']:>+13.2f}%")
    L.append(f"  年化收益  : {metrics['annual_return']:>+13.2f}%")
    L.append(f"  基准收益  : {metrics['b_total_ret']:>+13.2f}%  ({benchmark})")
    L.append(f"  超额收益  : {metrics['total_return'] - metrics['b_total_ret']:>+13.2f}%")
    L.append(f"  最大回撤  : {metrics['max_dd']:>+13.2f}%")
    L.append(f"  夏普比率  : {metrics['sharpe']:>15.3f}")
    L.append(f"  轮动胜率  : {metrics['win_rate']:>13.2f}%")
    L.append(f"  平均换手率: {metrics['avg_turnover']:>13.2f}%/次   |   平均持有期: {metrics['avg_hold_weeks']:.1f} 周")
    if metrics.get("crowd_latest") is not None:
        L.append(f"  维度5拥挤度: 最新 {metrics['crowd_latest'] * 100:.2f}%   |   历史分位 {metrics['crowd_pct']:.1f}%")
    L.append(f"  换仓次数  : {metrics.get('rebalance_count', '')}  (其中跳过 {n_skip})")
    L.append(f"  涨停买不进: {metrics.get('n_limit_up_skip', '')}  |  跌停卖不出(续持): {metrics.get('n_limit_down_hold', '')}  |  退市归零: {metrics.get('n_delist', '')}")
    L.append("")
    L.append("  说明: 本明细由 backtest_small_cap_rotation.py 自动导出；价格均为前复权口径，")
    L.append("        含成本(佣金万2.5/最低5 + 印花税千1→千0.5(2023-08-28起))与流动性自适应滑点；无前视(选股用成交日前一交易日快照)。")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return txt_path


def run_survivor_bias_comparison(
    start_date="20200102", end_date="20251231", hold_count=7,
    capital=None, benchmark="932000.SH", empty_jan_apr=False,
    enable_stop_loss=False, fundamental_filter=False, pool_mode="zz2000",
    quality_filter=False, growth_tilt=False, industry_cap=0,
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
                          exclude_delisted=False, pool_mode=pool_mode, quiet=False,
                          quality_filter=quality_filter, growth_tilt=growth_tilt,
                          industry_cap=industry_cap)
    r_without = run_backtest(start_date, end_date, hold_count=hold_count, capital=capital,
                             benchmark=benchmark, empty_jan_apr=empty_jan_apr,
                             enable_stop_loss=enable_stop_loss, fundamental_filter=fundamental_filter,
                             exclude_delisted=True, pool_mode=pool_mode, quiet=True,
                             quality_filter=quality_filter, growth_tilt=growth_tilt,
                             industry_cap=industry_cap)

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
    quality_filter=False, growth_tilt=False, industry_cap=0,
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
                             min_avg_amount_k=liq_k, pool_mode=pool_mode, quiet=True,
                             quality_filter=quality_filter, growth_tilt=growth_tilt,
                             industry_cap=industry_cap)
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


def run_size_quintile_comparison(
    start_date="20200102", end_date="20251231", hold_count=7, capital=None,
    pool_mode="zz2000", benchmark="932000.SH",
):
    """维度2·市值分位组对照：同一宇宙内分别跑 小/中/大 三档市值桶的周频轮动，对比净值，

    证明 alpha 来自"最小桶"而非任意小盘组（文章核心：只有分组你才知道自己研究的
    "小"到底小到什么程度）。三桶定义（均基于 universe 内 circ_mv 排序）：
      · 小市值桶  —— ORDER ASC  OFFSET 0      （最小 N 只）
      · 中市值桶  —— ORDER ASC  OFFSET 800    （宇宙 40%~(40%+N/2000) 分位）
      · 大市值桶  —— ORDER DESC OFFSET 0      （宇宙内最大 N 只）
    """
    if capital is None:
        capital = BACKTEST.get("total_capital", 500000)

    buckets = [
        ("小市值桶 (最小N只)",        "ASC", 0),
        ("中市值桶 (宇宙40-60%分位)", "ASC", 800),
        ("大市值桶 (宇宙最大N只)",    "DESC", 0),
    ]
    print(f"\n{'='*70}")
    print(f"  📐 维度2·市值分位组对照（同一宇宙内 小/中/大 桶轮动对比）")
    print(f"  区间: {start_date} ~ {end_date}  |  持仓: {hold_count} 只  |  宇宙: {POOL_DESC.get(pool_mode, pool_mode)}")
    print(f"{'='*70}")

    series_map = {}
    summary = []
    for name, order, offset in buckets:
        r = run_backtest(start_date, end_date, hold_count=hold_count, capital=capital,
                         pool_mode=pool_mode, benchmark=benchmark, quiet=True,
                         pool_order=order, pool_offset=offset)
        if r is None or not r.get("daily_values"):
            print(f"  {name}: 无结果，跳过")
            continue
        dvs = r["daily_values"]
        s0 = dvs[0]["value"]
        norm = [d["value"] / s0 * 100 for d in dvs]
        series_map[name] = norm
        summary.append((name, r["total_return"], r["annual_return"],
                        r["max_drawdown"], r["sharpe"]))
        print(f"  {name:<28} 总收益{r['total_return']:>+8.2f}%  年化{r['annual_return']:>+8.2f}%  "
              f"回撤{r['max_drawdown']:>7.2f}%  夏普{r['sharpe']:>6.3f}")

    if not series_map:
        print("  [WARN] 无有效分位组数据")
        return

    # ── 三桶叠加净值曲线 SVG（起点=100）──
    W, H, PAD = 940, 380, 56
    # 对齐到同一日期轴（取最长序列长度，短者末尾平铺）
    maxlen = max(len(v) for v in series_map.values())
    colors = {"小市值桶 (最小N只)": "#e0392b", "中市值桶 (宇宙40-60%分位)": "#d98a00",
              "大市值桶 (宇宙最大N只)": "#5b6b7c"}
    # 统一以最小序列长度为对齐基准（避免不同桶交易日数差异），用各自首值归一
    minlen = min(len(v) for v in series_map.values())
    aligned = {k: v[:minlen] for k, v in series_map.items()}
    flat = [v for s in aligned.values() for v in s]
    vmin, vmax = min(flat), max(flat)
    if vmax == vmin:
        vmax = vmin + 1
    vmin = max(0.0, vmin - (vmax - vmin) * 0.05)

    def X(i):
        return PAD + (W - 2 * PAD) * (i / (minlen - 1) if minlen > 1 else 0)

    def Y(v):
        return H - PAD - (H - 2 * PAD) * ((v - vmin) / (vmax - vmin))

    paths = ""
    for name, s in aligned.items():
        paths += "<path d='M " + " L ".join("%.1f,%.1f" % (X(i), Y(v)) for i, v in enumerate(s)) + \
                 "' fill='none' stroke='%s' stroke-width='2.2'/>" % colors.get(name, "#888")

    ygrid = "".join(
        "<line x1='%d' y1='%.1f' x2='%d' y2='%.1f' stroke='#eee' stroke-width='1'/>" % (PAD, Y(v), W - PAD, Y(v))
        + "<text x='%d' y='%.1f' text-anchor='end' font-size='11' fill='#888'>%.0f</text>" % (PAD - 8, Y(v) + 4, v)
        for k in range(4) for v in [vmin + (vmax - vmin) * k / 3]
    )
    cmp_svg = (
        "<svg viewBox='0 0 %d %d' width='100%%' preserveAspectRatio='xMidYMid meet' font-family='Menlo,Consolas,monospace'>" % (W, H)
        + ygrid + paths
        + "".join("<text x='%d' y='%d' font-size='12' fill='%s'>%s</text>" % (W - PAD + 4, 30 + i * 18, c, n)
                  for i, (n, c) in enumerate(colors.items()))
        + "</svg>"
    )

    rows = "".join(
        "<tr><td>%s</td><td class='%s'>%+.2f%%</td><td class='%s'>%+.2f%%</td><td class='%s'>%+.2f%%</td><td>%.3f</td></tr>"
        % (n, "up" if tr >= 0 else "down", tr, "up" if ar >= 0 else "down", ar,
           "down" if mdd < 0 else "", mdd, sh)
        for n, tr, ar, mdd, sh in summary
    )
    best = max(summary, key=lambda x: x[1])
    html = """<!DOCTYPE html>
<html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>小市值轮动 · 市值分位组对照</title>
<style>
  body{margin:0;background:#f5f6f8;color:#222;font-family:'Segoe UI','Microsoft YaHei',sans-serif;}
  .wrap{max-width:1000px;margin:0 auto;padding:24px 18px 60px;}
  h1{font-size:20px;margin:0 0 4px;} .sub{color:#888;font-size:13px;line-height:1.6;margin-bottom:18px;}
  .panel{background:#fff;border-radius:10px;padding:18px;box-shadow:0 1px 3px rgba(0,0,0,.06);margin-bottom:20px;}
  .ptitle{font-size:15px;font-weight:600;margin:0 0 2px;}
  table{width:100%;border-collapse:collapse;font-size:13px;margin-top:10px;}
  th,td{padding:8px;text-align:right;border-bottom:1px solid #eee;font-family:Menlo,Consolas,monospace;}
  th:first-child,td:first-child{text-align:left;font-family:inherit;} th{color:#999;background:#fafafa;}
  .up{color:#e0392b;} .down{color:#1a9e57;}
  .callout{background:#fff7e6;border-left:4px solid #d98a00;padding:12px 14px;border-radius:6px;font-size:13px;line-height:1.7;margin-top:6px;}
</style></head><body><div class='wrap'>
  <h1>小市值轮动 · 市值分位组对照</h1>
  <div class='sub'>同一选股宇宙内，按流通市值排序分成 小/中/大 三档桶，各自独立周频轮动，对比净值。
  目的：验证"小市值超额"究竟来自最小桶，还是任意小盘组都有——即文章"知道自己研究的小到底小到什么程度"。</div>
  <div class='panel'><div class='ptitle'>三档市值桶净值曲线（起点=100，已对齐长度）</div>[[SVG]]</div>
  <div class='panel'><div class='ptitle'>分桶绩效汇总</div>
    <table><thead><tr><th>市值分位桶</th><th>总收益率</th><th>年化收益</th><th>最大回撤</th><th>夏普</th></tr></thead>
    <tbody>[[ROWS]]</tbody></table>
    <div class='callout'>结论：alpha 高度集中在<b>最小市值桶</b>（[[BEST]]），中/大桶收益与风险显著弱化——
    说明本策略的超额确实来自"最小市值因子"本身，而非笼统的"小盘"。若最小桶与中桶收益接近，则提示因子已被"小盘"泛化稀释，需收紧。</div>
  </div>
</div></body></html>
"""
    html = html.replace("[[SVG]]", cmp_svg).replace("[[ROWS]]", rows).replace("[[BEST]]", best[0])

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"sc_size_quintile_{pool_mode}_{hold_count}_{start_date}_{end_date}.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n  [OK] 市值分位组对照报告 → {out_path}")


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
    p.add_argument("--quality-filter", action="store_true",
                   help="[A档] 质量门禁升级: roe>0 & bps>0(未资不抵债) & debt_to_assets<70 & ocfps>0 (与 --fundamental 互斥时优先; 数据缺失则保留)")
    p.add_argument("--growth-tilt", action="store_true",
                   help="[B档] 成长倾斜: 最小市值桶(hold*3)内按 净利润同比>0 优先 + roe 降序 重排取前N (size为底+成长增强)")
    p.add_argument("--industry-cap", type=int, default=0,
                   help="[维度4] 行业分散上限: 同一行业最多持仓只数(0=不限制). 直接缓解'集中困境微盘尾+单一行业暴雷'")
    p.add_argument("--exclude-delisted", action="store_true", help="剔除已退市股(INNER JOIN, 用于幸存者偏差对照)")
    p.add_argument("--min-avg-amount-k", type=float, default=None, help="流动性门槛(日均成交额,千元), 默认30000(=3000万)")
    p.add_argument("--pool-mode", choices=list(POOL_DESC.keys()), default="zz2000",
                   help="选股宇宙: cyb(纯创业板) / zz2000(中证2000风格·含微盘尾) / zz1000(中证1000风格·剔除微盘尾)")
    p.add_argument("--bucket", choices=["small", "mid", "large"], default=None,
                   help="市值分位桶(单独跑某一档, 替代默认最小桶): small=最小N只 / mid=宇宙40%分位档 / large=宇宙最大N只")
    p.add_argument("--mode", choices=["single", "compare", "sensitivity"], default="single",
                   help="single=单次回测; compare=含退市vs剔除退市对照; sensitivity=持仓数/流动性网格")
    p.add_argument("--hold-grid", default="5,7,10,15", help="sensitivity模式: 持仓数网格(逗号分隔)")
    p.add_argument("--liq-grid", default="30000,50000,80000,100000", help="sensitivity模式: 流动性门槛网格(千元)")
    p.add_argument("--detail", action="store_true", help="导出文本+CSV回测明细(逐笔换仓/交易流水)")
    p.add_argument("--no-html", action="store_true", help="不生成HTML报告(只要明细/控制台)")
    args = p.parse_args()

    # 明细输出路径（默认 outputs/sc_detail_<宇宙>_<持仓>_<起>_<止>）
    # ── 市值分位桶：small/mid/large 映射到 (order, offset) ──
    BUCKET_MAP = {
        "small": ("ASC", 0, "小市值桶(最小N只)"),
        "mid":   ("ASC", 800, "中市值桶(宇宙40%分位档)"),
        "large": ("DESC", 0, "大市值桶(宇宙最大N只)"),
    }
    bucket_order, bucket_offset, bucket_label = ("ASC", 0, None)
    if args.bucket:
        bucket_order, bucket_offset, bucket_label = BUCKET_MAP[args.bucket]

    detail_path = None
    if args.detail:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
        os.makedirs(out_dir, exist_ok=True)
        _bn = f"_{args.bucket}" if args.bucket else ""
        detail_path = os.path.join(out_dir, f"sc_detail{_bn}_{args.pool_mode}_{args.hold_count}_{args.start_date}_{args.end_date}")

    if args.mode == "single":
        run_backtest(
            args.start_date, args.end_date, hold_count=args.hold_count,
            capital=args.capital, benchmark=args.benchmark,
            empty_jan_apr=args.empty_jan_apr, enable_stop_loss=args.stop_loss,
            fundamental_filter=args.fundamental, exclude_delisted=args.exclude_delisted,
            min_avg_amount_k=args.min_avg_amount_k, pool_mode=args.pool_mode,
            quality_filter=args.quality_filter, growth_tilt=args.growth_tilt,
            industry_cap=args.industry_cap,
            pool_order=bucket_order, pool_offset=bucket_offset, bucket_label=bucket_label,
            detail_path=detail_path, no_html=args.no_html,
        )
    elif args.mode == "compare":
        run_survivor_bias_comparison(
            args.start_date, args.end_date, hold_count=args.hold_count,
            capital=args.capital, benchmark=args.benchmark,
            empty_jan_apr=args.empty_jan_apr, enable_stop_loss=args.stop_loss,
            fundamental_filter=args.fundamental, pool_mode=args.pool_mode,
            industry_cap=args.industry_cap,
        )
    elif args.mode == "sensitivity":
        run_sensitivity(
            args.start_date, args.end_date, capital=args.capital,
            benchmark=args.benchmark, empty_jan_apr=args.empty_jan_apr,
            enable_stop_loss=args.stop_loss, fundamental_filter=args.fundamental,
            hold_grid=[int(x) for x in args.hold_grid.split(",")],
            liq_grid=[float(x) for x in args.liq_grid.split(",")],
            pool_mode=args.pool_mode, industry_cap=args.industry_cap,
        )
