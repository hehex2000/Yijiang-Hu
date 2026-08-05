# -*- coding: utf-8 -*-
"""
5日均线「五句话」短线纪律策略（组合级资金池 · 完整成本口径）
=========================================================
来源：抖音 掌柜肖肖《一根线五句话》（评级 C 偏 B）。本脚本只实现其「止损纪律」骨架，
验收看回撤/亏损尾部，而非总收益（作者自述："亏的时候不至于亏到爬不起来"）。

设计要点（见 plan_ma5_swing.md）：
  · MA5 用「绝对后复权」价（raw × adj_factor），消除除权跳空造成的假跌破。
  · T+1 硬约束：T 日收盘判定信号 → T+1 开盘成交，杜绝未来函数。
  · 组合级资金池 + 持仓上限 max_pos（默认 10），信号间资金竞争流转。
  · 规则4+5 叠加采用 **A 方案**：c=1 减 50% / c=2 不动 / c=3 清仓（站回即重置）。
  · 黑名单（用户决策3，默认开）：清仓退出之日起 blacklist_days(20) 个交易日不再入选。
  · 成本用 run_monthly_rebalance.calc_fee 完整口径（佣金0.025%min5 + 滑点0.1% + 分段印花税）。

复用：run_macd_regime(_conn/_load_code/_PX/_factor/index_close/_metrics/_yearly/_POOL_INDEX)、
      run_magic_formula(_get_pool_constituents)、run_monthly_rebalance(calc_fee)。
"""
import sys, os, sqlite3, bisect, argparse, datetime
import numpy as np

import run_macd_regime as reg
import run_magic_formula as mf
import run_monthly_rebalance as rmb

DB_PATH = reg.DB_PATH


# ───────────────────────────── 参数 ─────────────────────────────
def default_params():
    return dict(
        ma=5,
        body_min=0.05,          # 大阳线实体涨幅下限
        vol_mult=2.0,           # 放量倍数
        lookback=20,            # 规则1 观察窗（近 N 日至少1根放量大阳线）
        touch_tol=0.01,         # 回踩容差 low≤MA5×(1+tol)
        entry_dev_max=0.05,     # 不能已经飞了：close/MA5-1 ≤ 此值
        signal_valid=10,        # 大阳线后 N 日内出现回踩才算数
        dev_tp=0.10,            # 止盈偏离
        tp_ratio=0.5,           # 止盈减仓比例
        tp_max=1,               # 止盈触发上限次数
        cut_ratio=0.5,          # 规则4 减仓比例
        exit_days=3,            # 规则5 连续跌破天数
        max_pos=10,             # 同时持仓上限
        amount_min=50000.0,     # 20日均成交额下限（千元）→ 5000万
        blacklist_days=20,      # 决策3：清仓起 N 交易日不再入选
        blacklist_on=True,      # 决策3：黑名单默认开
        limit_up_as_body=True,  # 一字涨停 pct_chg≥9.8% 算作放量阳线兜底
        # ── 退出模式（纪律隔离消融用，默认 full=五句话纪律）──
        exit_mode="full",       # full=五句话 / fixedN=固定持有N日 / reversal=收破MA5止
        hold_days=10,           # fixedN：持有交易日数
        max_hold_days=60,       # reversal：最长持有（防无限）
        reserve=0.02,           # 单笔买入留 2% 现金缓冲
        verbose=False,
    )


# ───────────────────────────── 数据层 ─────────────────────────────
_ST_CACHE = {}

def _build(code):
    """为单只股票构建后复权序列与指标数组，返回 dict（按本地日期索引）。"""
    if code in _ST_CACHE:
        return _ST_CACHE[code]
    reg._load_code(code)
    dates, o, h, l, c = reg._PX[code]
    fd, ff = reg._FAC[code]
    n = len(dates)
    if n < 30:
        _ST_CACHE[code] = None
        return None
    dates = np.array(dates)
    o = np.asarray(o, float); h = np.asarray(h, float)
    l = np.asarray(l, float); c = np.asarray(c, float)
    # 重新取 vol/amount（_PX 不含，单独查，顺序与 _PX 一致）
    conn = reg._conn()
    rows = conn.execute(
        "SELECT CAST(trade_date AS TEXT), vol, amount, pct_chg, pre_close "
        "FROM daily WHERE ts_code=? ORDER BY trade_date", (code,)).fetchall()
    conn.close()
    v = np.array([(r[1] or 0.0) for r in rows], float)
    amt = np.array([(r[2] or 0.0) for r in rows], float)
    pct = np.array([(r[3] or 0.0) for r in rows], float)
    prec = np.array([(r[4] or 0.0) for r in rows], float)

    # 复权因子对齐（前向填充）
    fac = np.ones(n)
    j = 0
    for i in range(n):
        d = dates[i]
        while j + 1 < len(fd) and fd[j + 1] <= d:
            j += 1
        f = ff[j]
        fac[i] = float(f) if (f is not None and f > 0) else (fac[i - 1] if i > 0 else 1.0)

    # 后复权价
    ho = o * fac; hh = h * fac; hl = l * fac; hc = c * fac

    # MA5（后复权收盘）— 用 convolve 保证窗口为最近 5 日（避免 cumsum 错位）
    ma5 = np.full(n, np.nan)
    if n >= 5:
        ma5[4:] = np.convolve(hc, np.ones(5) / 5.0, mode="valid")
    ma5_up = np.zeros(n, bool)
    ma5_up[1:] = ma5[1:] > ma5[:-1]

    # MA20 量（排除当日 & 停牌日 vol=0）用于放量判定
    ma20vol_excl = np.zeros(n)
    for i in range(20, n):
        win = v[i - 20:i]
        m = win > 0
        if m.any():
            ma20vol_excl[i] = win[m].mean()
        elif i > 0:
            ma20vol_excl[i] = v[i - 1]

    # 规则1：放量大阳线（近 lookback 窗口至少 1 根）
    has_yang = np.zeros(n, bool)
    for i in range(1, n):
        if v[i] <= 0:
            continue
        yang_body = (c[i] > o[i]) and (o[i] > 0) and ((c[i] - o[i]) / o[i] >= P_BODY)
        yang_vol = ma20vol_excl[i] > 0 and (v[i] >= P_VOL * ma20vol_excl[i])
        limit_up = (pct[i] >= 9.8) if P_LIMITUP else False
        has_yang[i] = (yang_body and yang_vol) or limit_up

    # 20日均成交额（千元）— convolve 保证 20 日窗口
    amount20 = np.zeros(n)
    if n >= 20:
        amount20[19:] = np.convolve(amt, np.ones(20) / 20.0, mode="valid")

    # 涨跌停（开盘即板，无法成交）
    is_limit_up_open = (o == h) & (o >= prec * 1.098) & (prec > 0)
    is_limit_down_open = (o == l) & (o <= prec * 0.902) & (prec > 0)

    idx_of = {d: i for i, d in enumerate(dates.tolist())}
    d = dict(dates=dates, ho=ho, hh=hh, hl=hl, hc=hc, vol=v, pct=pct,
             ma5=ma5, ma5_up=ma5_up, ma20vol_excl=ma20vol_excl,
             has_yang=has_yang, amount20=amount20, prec=prec,
             is_limit_up_open=is_limit_up_open, is_limit_down_open=is_limit_down_open,
             idx_of=idx_of, n=n)
    _ST_CACHE[code] = d
    return d


# 模块级参数占位（_build 内引用，run 时填充）
P_BODY = 0.05; P_VOL = 2.0; P_LIMITUP = True


def _universe(pool, asof):
    if pool == "all":
        conn = reg._conn()
        rows = conn.execute(
            "SELECT DISTINCT ts_code FROM daily WHERE trade_date=?", (asof,)).fetchall()
        conn.close()
        return sorted(str(r[0]) for r in rows)
    return sorted(mf._get_pool_constituents(pool, asof) or [])


def _bench_of(pool):
    if pool == "all":
        return "000985.SH"
    return reg._POOL_INDEX.get(pool, "000300.SH")


def _is_st(code, name_cache):
    if code in name_cache:
        return name_cache[code]
    conn = reg._conn()
    r = conn.execute("SELECT name FROM stock_basic WHERE ts_code=?", (code,)).fetchone()
    conn.close()
    nm = (r[0] if r else "") or ""
    st = ("ST" in nm.upper())
    name_cache[code] = st
    return st


# ───────────────────────────── 主回测 ─────────────────────────────
def run_backtest(start_date, end_date, pool="hs300", capital=1000000,
                 zero_cost=False, **kw):
    global P_BODY, P_VOL, P_LIMITUP
    P = default_params()
    P.update(kw)
    P_BODY = P["body_min"]; P_VOL = P["vol_mult"]; P_LIMITUP = P["limit_up_as_body"]

    idx_code = _bench_of(pool)
    conn = reg._conn()
    trade_dates = [r[0] for r in conn.execute(
        "SELECT DISTINCT CAST(trade_date AS TEXT) FROM daily "
        "WHERE trade_date BETWEEN ? AND ? ORDER BY trade_date",
        (start_date, end_date)).fetchall()]
    conn.close()
    if not trade_dates:
        print("[ERROR] 无交易日")
        return None

    # 上市满 lookback+60 日过滤
    sd = datetime.date(int(start_date[:4]), int(start_date[4:6]), int(start_date[6:8]))
    list_cut = (sd - datetime.timedelta(days=P["lookback"] + 60)).strftime("%Y%m%d")
    universe = [c for c in _universe(pool, trade_dates[0])
                if _listed_before(c, list_cut)]
    if not universe:
        print("[ERROR] 成分股为空")
        return None

    # 预构建数据（ST/成交额过滤在信号时再做）
    data = {}
    for code in universe:
        d = _build(code)
        if d is not None:
            data[code] = d
    codes = list(data.keys())
    if not codes:
        print("[ERROR] 无可用数据")
        return None
    N = len(codes)

    name_cache = {}
    # 黑名单：code -> 解禁全局索引（该索引前不可入选）
    blacklist_until = {}

    cash = float(capital)
    positions = {}   # code -> dict(shares, buy_open_hfq, buy_date, tp_done, counter)
    # 待执行订单（昨日收盘决策 → 今日开盘执行）
    pend_buys = []       # list of code（按放量倍数排序后填充）
    pend_sells = {}      # code -> 'full' or ratio(float)

    # 成本审计
    aud = dict(n_buy=0, n_sell=0, n_reduce=0, comm=0.0, stamp=0.0, slip=0.0,
               gross_profit=0.0, trades=0, hold_days=[], win=0)

    daily_vals = []
    gi_of = {d: i for i, d in enumerate(trade_dates)}

    warm_min = max(20, P["signal_valid"] + 1)

    for gi, td in enumerate(trade_dates):
        # ── 1) 执行昨日决策（今日开盘）──
        # 卖出
        for code, kind in list(pend_sells.items()):
            d = data.get(code)
            if d is None or code not in positions:
                continue
            i = d["idx_of"].get(td)
            if i is None or i >= d["n"]:
                continue
            if d["is_limit_down_open"][i]:
                continue  # 跌停封死 → 顺延（次日会重新决策）
            hfq_open = d["ho"][i]
            pos = positions[code]
            if kind == "full":
                sh = pos["shares"]
            else:
                sh = int(pos["shares"] * kind / 100) * 100
                sh = min(sh, pos["shares"])
            if sh <= 0:
                continue
            if zero_cost:
                fee = 0.0
            else:
                fee = rmb.calc_fee("sell", hfq_open, sh, trade_date=int(td))
            proceeds = sh * hfq_open - fee
            cash += proceeds
            # 成本拆分
            if not zero_cost:
                bd = rmb.calc_fee_breakdown("sell", hfq_open, sh, trade_date=int(td))
                aud["comm"] += bd["commission"]; aud["stamp"] += bd["stamp_duty"]
                aud["slip"] += bd["slippage"]; aud["n_sell"] += 1
            if kind == "full":
                # 记录完整交易盈亏
                pnl = (hfq_open - pos["buy_open_hfq"]) * pos["shares"] - fee
                aud["gross_profit"] += pnl
                aud["trades"] += 1
                if pnl > 0:
                    aud["win"] += 1
                aud["hold_days"].append(_hold_len(pos["buy_date"], td, trade_dates))
                del positions[code]
                if P["blacklist_on"]:
                    blacklist_until[code] = gi + P["blacklist_days"]
            else:
                pos["shares"] -= sh
                aud["n_reduce"] += 1
        pend_sells = {}

        # 买入（按放量倍数降序 + code 字典序，容量受限）
        def _buy_key(c):
            ii = data[c]["idx_of"].get(td)
            if ii is None or ii >= data[c]["n"]:
                return (0.0, c)          # 当日无数据 → 排末尾，循环内会被跳过
            mv = max(data[c]["ma20vol_excl"][ii], 1e-9)
            return (-data[c]["vol"][ii] / mv, c)
        pend_buys.sort(key=_buy_key)
        for code in pend_buys:
            if len(positions) >= P["max_pos"]:
                break
            if code in positions:
                continue
            if P["blacklist_on"] and gi < blacklist_until.get(code, -1):
                continue
            d = data.get(code)
            if d is None:
                continue
            i = d["idx_of"].get(td)
            if i is None or i >= d["n"]:
                continue
            if d["is_limit_up_open"][i]:
                continue  # 一字涨停买不进 → 放弃该笔
            hfq_open = d["ho"][i]
            if hfq_open <= 0:
                continue
            budget = min(capital / P["max_pos"], cash * (1 - P["reserve"]))
            sh = int(budget / hfq_open / 100) * 100
            if sh <= 0:
                continue
            if zero_cost:
                fee = 0.0
            else:
                fee = rmb.calc_fee("buy", hfq_open, sh, trade_date=int(td))
            cost = sh * hfq_open + fee
            if cost > cash + 1e-6:
                continue
            cash -= cost
            positions[code] = dict(shares=sh, buy_open_hfq=hfq_open,
                                   buy_date=td, tp_done=False, counter=0, buy_gi=gi)
            if not zero_cost:
                bd = rmb.calc_fee_breakdown("buy", hfq_open, sh, trade_date=int(td))
                aud["comm"] += bd["commission"]; aud["slip"] += bd["slippage"]
                aud["n_buy"] += 1
        pend_buys = []

        # ── 2) 观察今日收盘 → 生成明日决策 ──
        for code, d in data.items():
            i = d["idx_of"].get(td)
            if i is None or i < warm_min or i >= d["n"]:
                continue
            # 持仓：按 exit_mode 执行退出决策（入场/成本/黑名单均不变，仅退出规则不同）
            if code in positions:
                pos = positions[code]
                if P["exit_mode"] == "fixedN":
                    # 固定持有 N 交易日后清仓（无纪律、无止盈、无回撤响应）
                    if gi - pos["buy_gi"] >= P["hold_days"]:
                        pend_sells[code] = "full"
                    continue
                if P["exit_mode"] == "reversal":
                    # 持有至收破 MA5（thesis 失效）或触顶 max_hold_days
                    broke = (d["ma5"][i] > 0) and (d["hc"][i] < d["ma5"][i])
                    if broke or (gi - pos["buy_gi"] >= P["max_hold_days"]):
                        pend_sells[code] = "full"
                    continue
                # full：规则3 止盈 + 规则4+5 A 方案（五句话纪律）
                below = d["hc"][i] < d["ma5"][i]
                pos["counter"] = pos["counter"] + 1 if below else 0
                c = pos["counter"]
                # 规则3 止盈
                dev = d["hc"][i] / d["ma5"][i] - 1 if d["ma5"][i] > 0 else 0
                if (not pos["tp_done"]) and dev >= P["dev_tp"]:
                    pend_sells.setdefault(code, P["tp_ratio"])
                    pos["tp_done"] = True
                    continue  # 当日只触发止盈，不叠加减仓
                # 规则4+5 A 方案
                if c >= P["exit_days"]:
                    pend_sells[code] = "full"
                elif c == 1:
                    pend_sells.setdefault(code, P["cut_ratio"])
                # c==2 不动
                continue
            # 空仓：买入候选（过滤 ST/成交额/已黑名单）
            if P["blacklist_on"] and gi < blacklist_until.get(code, -1):
                continue
            if _is_st(code, name_cache):
                continue
            if d["amount20"][i] < P["amount_min"]:
                continue
            # 规则1：signal_valid 日内有大阳线
            lo = max(1, i - P["signal_valid"])
            if not d["has_yang"][lo:i].any():
                continue
            # 规则2：回踩买点
            ma5 = d["ma5"][i]
            if not (ma5 > 0 and d["ma5_up"][i]):
                continue
            if not (d["hl"][i] <= ma5 * (1 + P["touch_tol"])):
                continue
            if not (d["hc"][i] >= ma5):
                continue
            dev = d["hc"][i] / ma5 - 1
            if dev > P["entry_dev_max"]:
                continue
            pend_buys.append(code)

        # ── 3) 收盘市值 ──
        val = cash
        for code, pos in positions.items():
            d = data.get(code)
            if d is None:
                continue
            i = d["idx_of"].get(td)
            if i is None or i >= d["n"]:
                continue
            val += pos["shares"] * d["hc"][i]
        daily_vals.append({"date": td, "value": val})

    # 末日清仓（计入成本/黑名单）
    last_td = trade_dates[-1]
    for code in list(positions.keys()):
        d = data.get(code)
        if d is None:
            continue
        i = d["idx_of"].get(last_td)
        if i is None or i >= d["n"]:
            continue
        hfq_close = d["hc"][i]
        sh = positions[code]["shares"]
        if zero_cost:
            fee = 0.0
        else:
            fee = rmb.calc_fee("sell", hfq_close, sh, trade_date=int(last_td))
        cash += sh * hfq_close - fee
        if not zero_cost:
            bd = rmb.calc_fee_breakdown("sell", hfq_close, sh, trade_date=int(last_td))
            aud["comm"] += bd["commission"]; aud["stamp"] += bd["stamp_duty"]
            aud["slip"] += bd["slippage"]; aud["n_sell"] += 1
        # 末日平仓不算"放弃"，不进黑名单
    if daily_vals:
        daily_vals[-1]["value"] = cash

    return _report(daily_vals, trade_dates, capital, idx_code, pool, P, aud, zero_cost, N)


def _listed_before(code, cut):
    conn = reg._conn()
    r = conn.execute("SELECT list_date FROM stock_basic WHERE ts_code=?", (code,)).fetchone()
    conn.close()
    return r and r[0] and r[0] <= cut


def _hold_len(buy_date, sell_date, trade_dates):
    a = gi_of_hold(buy_date, trade_dates)
    b = gi_of_hold(sell_date, trade_dates)
    return max(0, b - a)


_gi_cache = {}

def gi_of_hold(date, trade_dates):
    if date in _gi_cache:
        return _gi_cache[date]
    # trade_dates 已排序；bisect
    i = bisect.bisect_left(trade_dates, date)
    _gi_cache[date] = i
    return i


# ───────────────────────────── 报告 ─────────────────────────────
def _report(daily_vals, trade_dates, capital, bench, pool, P, aud, zero_cost, N):
    import pandas as pd
    dates = [d["date"] for d in daily_vals]
    vals = np.array([d["value"] for d in daily_vals], float)
    total, ann, mdd, sharpe, pk, tr = reg._metrics(vals)

    b0 = reg.index_close(bench, dates[0]); b1 = reg.index_close(bench, dates[-1])
    b_total = (b1 / b0 - 1) if (b0 and b1) else 0

    _MODE_LABEL = {"full": "五句话纪律", "fixedN": f"固定持有{P['hold_days']}日",
                   "reversal": "收破MA5止"}
    mode_label = _MODE_LABEL.get(P["exit_mode"], P["exit_mode"])
    print(f"\n{'='*72}")
    print(f"  5日均线短线策略 [退出模式={mode_label}]  {'[零成本对照]' if zero_cost else '[完整成本]'}")
    print(f"  池={pool}  区间={trade_dates[0]}~{trade_dates[-1]}  初始={capital:,.0f}  "
          f"max_pos={P['max_pos']}  黑名单={'开(%d日)'%P['blacklist_days'] if P['blacklist_on'] else '关'}")
    print(f"{'='*72}")
    print(f"  {'年份':<8}{'策略':>10}{'基准':>10}{'超额':>10}")
    print(f"  {'─'*40}")
    yg = reg._yearly(dates, vals, capital)
    byg = reg._yearly(dates, [reg.index_close(bench, d) for d in dates],
                      reg.index_close(bench, dates[0]))
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
    print(f"  有效标的数: {N}")

    # 成本审计
    cost_total = aud["comm"] + aud["stamp"] + aud["slip"]
    print(f"\n{'='*72}\n  ━━━ 摩擦成本审计 ━━━\n{'='*72}")
    print(f"  买入笔数      : {aud['n_buy']}")
    print(f"  卖出笔数(含减仓): {aud['n_sell']}  (其中减仓 {aud['n_reduce']} 次)")
    print(f"  完整交易笔数  : {aud['trades']}  (盈利 {aud['win']} / 亏损 {aud['trades']-aud['win']})")
    if aud["hold_days"]:
        hd = np.array(aud["hold_days"])
        print(f"  持仓天数      : 中位 {np.median(hd):.0f}日 / 平均 {hd.mean():.1f}日")
    if not zero_cost:
        print(f"  佣金合计      : ¥{aud['comm']:,.0f}  ({aud['comm']/capital:+.2%})")
        print(f"  印花税合计    : ¥{aud['stamp']:,.0f}  ({aud['stamp']/capital:+.2%})")
        print(f"  滑点合计      : ¥{aud['slip']:,.0f}  ({aud['slip']/capital:+.2%})")
        print(f"  成本总计      : ¥{cost_total:,.0f}  ({cost_total/capital:+.2%})")
        if aud["gross_profit"] != 0:
            print(f"  毛收益(零成本): {aud['gross_profit']/capital:+.2%}")
            print(f"  成本吞噬比例  : {cost_total/max(abs(aud['gross_profit']),1e-9):.1%}  ← 关键")
    else:
        print("  [零成本模式] 所有费率置 0，用于与完整成本对照量化摩擦代价。")

    out_dir = "data/results/ma5_swing"
    os.makedirs(out_dir, exist_ok=True)
    csv = f"{out_dir}/ma5_{pool}_{P['exit_mode']}_{trade_dates[0]}_{trade_dates[-1]}{'_zero' if zero_cost else ''}.csv"
    pd.DataFrame(daily_vals).to_csv(csv, index=False)
    print(f"\n  日净值 → {csv}\n")
    return {"total": total, "annual": ann, "mdd": mdd, "sharpe": sharpe,
            "bench": b_total, "cost": cost_total}


# ───────────────────────────── CLI ─────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="5日均线五句话短线纪律策略")
    ap.add_argument("start", help="起始日 YYYYMMDD")
    ap.add_argument("end", help="结束日 YYYYMMDD")
    ap.add_argument("--pool", default="hs300", choices=["hs300", "zz500", "zz800", "zz1000", "all"])
    ap.add_argument("--capital", type=float, default=1000000)
    ap.add_argument("--max-pos", type=int, default=10)
    ap.add_argument("--body-min", type=float, default=0.05)
    ap.add_argument("--vol-mult", type=float, default=2.0)
    ap.add_argument("--lookback", type=int, default=20)
    ap.add_argument("--touch-tol", type=float, default=0.01)
    ap.add_argument("--entry-dev-max", type=float, default=0.05)
    ap.add_argument("--signal-valid", type=int, default=10)
    ap.add_argument("--dev-tp", type=float, default=0.10)
    ap.add_argument("--tp-ratio", type=float, default=0.5)
    ap.add_argument("--cut-ratio", type=float, default=0.5)
    ap.add_argument("--exit-days", type=int, default=3)
    ap.add_argument("--exit-mode", default="full",
                    choices=["full", "fixedN", "reversal"],
                    help="消融：full=五句话纪律 / fixedN=固定持有N日 / reversal=收破MA5止")
    ap.add_argument("--hold-days", type=int, default=10,
                    help="fixedN 模式持有交易日数")
    ap.add_argument("--max-hold-days", type=int, default=60,
                    help="reversal 模式最长持有交易日数")
    ap.add_argument("--amount-min", type=float, default=50000.0)
    ap.add_argument("--blacklist-days", type=int, default=20)
    ap.add_argument("--no-blacklist", action="store_true")
    ap.add_argument("--zero-cost", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    run_backtest(
        args.start, args.end, pool=args.pool, capital=args.capital,
        zero_cost=args.zero_cost,
        max_pos=args.max_pos, body_min=args.body_min, vol_mult=args.vol_mult,
        lookback=args.lookback, touch_tol=args.touch_tol, entry_dev_max=args.entry_dev_max,
        signal_valid=args.signal_valid, dev_tp=args.dev_tp, tp_ratio=args.tp_ratio,
        cut_ratio=args.cut_ratio, exit_days=args.exit_days, amount_min=args.amount_min,
        exit_mode=args.exit_mode, hold_days=args.hold_days, max_hold_days=args.max_hold_days,
        blacklist_days=args.blacklist_days, blacklist_on=not args.no_blacklist,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
