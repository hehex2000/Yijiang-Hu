# -*- coding: utf-8 -*-
"""
MACD 趋势跟随择时策略（Jim 合规 · 无 KDJ）
==========================================
顶替已退休的 macd_kdj 逐股择时插件——纯 MACD 趋势跟随，无 KDJ 按钮。

逐股 timing 语义（每只股票独立决定多/空）：
  - 多头区 = DIF > DEA（MACD 柱为正，即 DIF 在信号线上方 = 动量向上）。
    入场 = 状态由 False→True（即金叉），出场 = 状态由 True→False（死叉）。
    这正是旧 macd_kdj 的 MACD 部分，但**彻底去掉 KDJ 按钮**。
  - 指数门控（regime gate）：股票池对应指数站上 MA200 才允许做多；跌破 → 强制空仓。
    （对应 Jim「趋势和震荡分开看」+ 道氏「分季节」；在熊市/震荡里不接飞刀。）

记账口径与 run_macd_regime 完全一致（hfq 后复权、买入日因子归一化、
卖 0.99955 / 买 1.0002），可直接与基准(指数买入持有)对照。

汇总方式：逐股独立子账户等权。每只股票分得 1/N 资金，独立跑 MACD timing，
组合净值 = 各子账户之和（天然等权）。这是最忠实的「逐股择时」表达方式，
也规避了 per-stock 插件「76.5% 零交易」那种逆向抄底的坑——趋势跟随在牛市会正常开火。

复用 run_macd_regime 的底层 machinery（_conn / _load_code / _factor / index_above_ma /
ta.MACD / hfq 记账），不重复造轮子。
"""
import sys, os, sqlite3, bisect, argparse
import numpy as np
import pandas as pd
import talib as ta

import run_macd_regime as mr          # 复用底层 machinery
import run_magic_formula as mf        # 复用 _get_pool_constituents
import config                          # 复用平台全局股票池设置 SELECTION["stock_pool"]


def _timing_one(code, start, end, idx_code, P):
    """对单只股票跑 MACD 趋势跟随 timing，返回 (dates_in_window, sub_values)。

    无足够历史或窗口内无交易日 → 返回 None（调用方按等权基准 sub_cap 处理）。
    全程加载完整历史做 warmup（DIF/DEA 与指数 MA200 需要前置数据），
    但只输出 [start, end] 窗口内的逐日子账户净值。
    """
    mr._load_code(code)
    alldates, opens, highs, lows, closes = mr._PX[code]
    # 完整历史索引
    si = bisect.bisect_left(alldates, start)   # 窗口起点
    ei = bisect.bisect_right(alldates, end)    # 窗口终点
    if ei <= si:
        return None
    # 前置 warmup 检查：窗口起点前需有足够 bar 让 DIF/DEA 出值。
    # talib.MACD 的首个非 NaN 出现在第 (slow + signal - 1) 根 bar（12/26/9 → 第34根，索引33）。
    # 注意：EMA 是嵌套计算（DEA 是 DIF 的 EMA），不是 fast+slow+signal 串联相加；
    # 指数 MA 门控用的是**指数**序列，与个股 bar 数无关，且 index_above_ma 内部
    # 自带 `i < win-1 → return True` 越界保护，因此绝不能把 ma 计入个股 warmup。
    warm = P["slow"] + P["signal"] + 10        # 26+9+10 = 45
    if si < warm:
        return None

    alldates = np.array(alldates)
    opens_a = np.asarray(opens, dtype=float)
    closes_a = np.asarray(closes, dtype=float)
    n_all = len(closes_a)

    dif, dea, _ = ta.MACD(closes_a, fastperiod=P["fast"], slowperiod=P["slow"],
                          signalperiod=P["signal"])
    # 多头状态（用 T-1 收盘，T 开盘执行，杜绝未来函数）
    long_state = np.zeros(n_all, dtype=bool)
    for t in range(1, n_all):
        d, e = dif[t - 1], dea[t - 1]
        if np.isnan(d) or np.isnan(e):
            long_state[t] = long_state[t - 1]
            continue
        s = bool(d > e)                       # DIF 在信号线上方 = 多头区
        if P.get("zero_line"):
            s = s and (d > 0)                  # 额外要求站上零轴（更确认的上行，减少假死叉whipshaw）
        if P["regime"]:
            g = mr.index_above_ma(idx_code, alldates[t - 1], P["ma"])
            s = s and g
        long_state[t] = s

    sub_cap = float(P["sub_cap"])
    cash = sub_cap
    shares = 0
    buy_factor = 1.0
    prev_long = False
    out_dates = []
    out_vals = []
    for t in range(si, ei):
        td = alldates[t]
        f = mr._factor(code, td) or 1.0
        raw_open = opens_a[t]
        raw_close = closes_a[t]
        if np.isnan(raw_open) or np.isnan(raw_close):
            # 缺数据日：维持现状估值
            if shares > 0:
                v = cash + shares * raw_close * (f / buy_factor)
            else:
                v = cash
            out_dates.append(td); out_vals.append(v)
            continue

        long_now = long_state[t]
        if long_now and not prev_long:
            # 金叉 → 开盘买入（用 1/N 子资金，留 2% 现金缓冲）
            if shares == 0:
                px = raw_open                       # 原始开盘价
                budget = cash * 0.98
                # 子账户等权用分数股表达(notional)：1/N 资金≈52元远买不起整手(100股)，
                # 若按整手取整 sh 恒为 0 → 全市场无任何持仓 → 净值恒等于初始资金(满仓现金)。
                # 故此处用分数股满仓，忠实表达「逐股等权 timing 信号」的平均收益。
                sh = budget / (px * mr.BUY_MULT)
                if sh > 0:
                    cost = sh * px * mr.BUY_MULT
                    if cost <= cash:
                        shares = sh
                        cash -= cost
                        buy_factor = f
        elif (not long_now) and prev_long:
            # 死叉 → 开盘卖出
            if shares > 0:
                hfq_px = raw_open * (f / buy_factor)
                cash += shares * hfq_px * mr.SELL_MULT
                shares = 0
        prev_long = long_now

        if shares > 0:
            hfq_close = raw_close * (f / buy_factor)
            v = cash + shares * hfq_close
        else:
            v = cash
        out_dates.append(td); out_vals.append(v)

    # 末日平仓
    if shares > 0:
        f = mr._factor(code, alldates[ei - 1]) or 1.0
        hfq_close = closes_a[ei - 1] * (f / buy_factor)
        cash += shares * hfq_close * mr.SELL_MULT
        out_vals[-1] = cash

    return out_dates, out_vals


def _universe(pool, asof):
    """返回 pool 在 asof 时点的成分股（已排序）。pool='all' → 当日全部在市 A 股。"""
    if pool == "all":
        c = mr._conn()
        rows = c.execute(
            "SELECT DISTINCT ts_code FROM daily WHERE trade_date=?", (asof,)).fetchall()
        c.close()
        return sorted(str(r[0]) for r in rows)
    return sorted(mf._get_pool_constituents(pool, asof) or [])


def _bench_of(pool):
    """池 → 对照基准指数。all 用中证全指 000985.SH。"""
    if pool == "all":
        return "000985.SH"
    return mr._POOL_INDEX.get(pool, "000300.SH")


def _report(daily_vals, trade_dates, capital, bench, start, end, N, regime, ma):
    dates = [d["date"] for d in daily_vals]
    vals = np.array([d["value"] for d in daily_vals], dtype=float)
    total, ann, mdd, sharpe, pk, tr = mr._metrics(vals)

    b0 = mr.index_close(bench, dates[0]); b1 = mr.index_close(bench, dates[-1])
    b_total = (b1 / b0 - 1) if (b0 and b1) else 0

    print(f"\n{'='*72}\n  MACD 趋势跟随择时 vs 买入持有(指数) 对照（hfq）\n{'='*72}")
    print(f"  {'年份':<8}{'策略':>10}{'基准':>10}{'超额':>10}")
    print(f"  {'─'*40}")
    yg = mr._yearly(dates, vals, capital)
    byg = mr._yearly(dates, [mr.index_close(bench, d) for d in dates], mr.index_close(bench, dates[0]))
    for y in sorted(yg):
        s0, s1, _ = yg[y]; sret = s1 / s0 - 1
        if y in byg:
            b0y, b1y, _ = byg[y]; bret = b1y / b0y - 1
            print(f"  {y:<8}{sret:>+9.2%}{bret:>+9.2%}{sret - bret:>+9.2%}")
    print(f"  {'─'*40}")
    print(f"  {'全程':<7}{total:>+9.2%}{b_total:>+9.2%}{total - b_total:>+9.2%}")

    print(f"\n{'='*72}\n  策略最终汇总\n{'='*72}")
    print(f"  初始资金: {capital:,.0f}  最终资产: {vals[-1]:,.0f}")
    print(f"  总收益: {total:+.2%}  年化: {ann:+.2%}")
    print(f"  最大回撤: {mdd:+.2%}  (峰 {dates[pk]} → 谷 {dates[tr]})")
    print(f"  夏普: {sharpe:.4f}")
    print(f"  股票池有效标的数(子账户): {N}  | 指数门控: {'开(MA%d)' % ma if regime else '关'}")

    out_dir = "data/results/macd_strategy"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = f"{out_dir}/macd_timing_{start}_{end}.csv"
    pd.DataFrame(daily_vals).to_csv(csv_path, index=False)
    print(f"\n  日净值 → {csv_path}\n")
    return {"total": total, "annual": ann, "mdd": mdd, "sharpe": sharpe}


def run_timing(start_date, end_date, pool="hs300", capital=1000000,
               fast=12, slow=26, signal=9, ma=200, regime=False,
               zero_line=False, max_stocks=None):
    idx_code = _bench_of(pool)
    c = mr._conn()
    trade_dates = [r[0] for r in c.execute(
        "SELECT DISTINCT CAST(trade_date AS TEXT) FROM daily "
        "WHERE trade_date BETWEEN ? AND ? ORDER BY trade_date",
        (start_date, end_date)).fetchall()]
    c.close()
    if not trade_dates:
        print("[ERROR] 无交易日")
        return None

    # 宇宙：取窗口起点时的成分股快照（与旧 macd_kdj 逐股框架一致）；all = 全A在市
    const = _universe(pool, trade_dates[0])
    if not const:
        print("[ERROR] 成分股为空")
        return None
    if max_stocks:
        const = const[:max_stocks]
    N = len(const)
    sub_cap = float(capital) / N

    P = dict(fast=fast, slow=slow, signal=signal, ma=ma, regime=regime,
             zero_line=zero_line, sub_cap=sub_cap)

    print("=" * 72)
    print("  MACD 趋势跟随择时策略（逐股 DIF>DEA 多头区 + 指数MA%d门控 · 无KDJ）" % ma)
    print("=" * 72)
    print(f"  区间: {start_date}~{end_date} | 池: {pool} | 有效标的: {N} | 资金: {capital:,}")
    print(f"  MACD: ({fast},{slow},{signal})  多头区=DIF>DEA(金叉)  指数门控: {'开' if regime else '关'}")
    print("=" * 72)

    # 组合净值 = 各子账户之和；每个子账户起点 sub_cap，缺数据股按恒定 sub_cap(现金)计入
    port = np.zeros(len(trade_dates), dtype=float)
    active = 0
    for code in const:
        res = _timing_one(code, start_date, end_date, idx_code, P)
        if res is None:
            port += sub_cap
            continue
        dts, vals = res
        vmap = {d: v for d, v in zip(dts, vals)}
        arr = np.array([vmap.get(td, sub_cap) for td in trade_dates], dtype=float)
        port += arr
        active += 1
    print(f"  实际产生交易的标的: {active}/{N}")

    daily_vals = [{"date": td, "value": float(v)} for td, v in zip(trade_dates, port)]
    rep = _report(daily_vals, trade_dates, capital, idx_code, start_date, end_date, N, regime, ma)
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="MACD 趋势跟随择时策略（无KDJ·顶替旧macd_kdj）")
    ap.add_argument("start_date", nargs="?", default="20140101")
    ap.add_argument("end_date", nargs="?", default="20260731")
    ap.add_argument("--pool", default=config.SELECTION.get("stock_pool", "zz800"),
                    help="hs300/zz500/zz800/zz1000/all(全A，基准000985.SH)；"
                         "默认跟随 config.py 全局 stock_pool=%s" % config.SELECTION.get("stock_pool", "zz800"))
    ap.add_argument("--capital", type=int, default=1000000)
    ap.add_argument("--fast", type=int, default=12)
    ap.add_argument("--slow", type=int, default=26)
    ap.add_argument("--signal", type=int, default=9)
    ap.add_argument("--regime", dest="regime", action="store_true", default=False,
                    help="开启指数MA门控（默认关；MA200在牛市易长期空仓，纯MACD timing更贴合旧macd_kdj）")
    ap.add_argument("--ma", type=int, default=200, help="指数门控均线窗口（需配合 --regime）")
    ap.add_argument("--zero-line", dest="zero_line", action="store_true", default=False,
                    help="多头区额外要求 DIF>0（站上零轴，减少假死叉whipshaw，更确认的上行）")
    ap.add_argument("--max-stocks", type=int, default=None, help="限制宇宙大小（冒烟测试用）")
    a = ap.parse_args()
    run_timing(a.start_date, a.end_date, pool=a.pool, capital=a.capital,
               fast=a.fast, slow=a.slow, signal=a.signal, ma=a.ma,
               regime=a.regime, zero_line=a.zero_line, max_stocks=a.max_stocks)
