# -*- coding: utf-8 -*-
# =============================================================================
# ❌ 无效策略 / INVALIDATED STRATEGY —— 本文件已归档，不参与任何实盘或策略对比
# -----------------------------------------------------------------------------
# 判定（2026-08-25，胡老师复核）：bear-123 既非选股 alpha，也非可靠退出闸门。
#   · 字面看多回测全面跑输"一直持有"（000300 -3.0% vs +53.7%；茅台 -3.5% vs +1342%）
#   · 接 buy&hold 退出闸门：回撤改善微弱且不一致（2/4 标的微改善、2/4 恶化），稳牺牲上行
#   · 接神奇公式退出闸门：回撤改善 ≤0.9pp（可忽略），收益不稳定（一恶一平）
#   · 唯一成立身份：趋势结构破坏探测器 / regime 描述特征（批判 ★★☆）
# 结论：放弃该策略主线。原代码未破坏，仅改名 +INVALID 作无效记录留存。
# =============================================================================
"""
run_123_pattern.py —— Sperandeo《123 交易法则》量化复刻 + 去魅验证
=============================================================================
B站来源：BV1A7gP6CEC6 · 跟着Jim学量化（白名单 mid 232752772）· 2026-08-14

!!! 诚实红线 #1（参数归属）：Jim 视频只讲了法则与量化框架，**全程未给"演示参数"
    的具体数值**（趋势线连接点、突破阈值、等待窗口天数都没说）。本文件所有参数
    均为本复刻**一次性定死的提案值，非 Jim 原值**，必须在报告首行明示。
!!! 诚实红线 #2（无未来函数）：检测器在日 i 只用 ≤ i 的数据；摆动点必须"已确认"
    （idx + N_SWING <= i）才可使用，禁止用未确认的最近极值。
!!! 诚实红线 #3（跑后不调参）：参数跑前定死，跑后不得回头改阈值再跑（数据窥探禁令）。
!!! 诚实红线 #4（失败信号全量保留）：仅破线 / 仅测试失败 / 超时信号原样保留，不删。

123 法则本质 = "趋势被逐层破坏"的过程检测器，不是选股 alpha（与 Jim 系列已验证的
主升浪/利弗莫尔/达瓦斯/缩量回踩同结论）。本脚本只做**信号全量披露**与**字面前向收益**
统计（看多反转做可交易多头回测；看空反转受 T+1/难做空约束仅作"退出预警"统计，
不做实际做空 P&L），为后续接退出 overlay（用途B）提供证据。

结构镜像 run_swing_trend.py 四段式：参数区 → 纯函数检测 → 逐日重放状态机 → 统计/报告/落盘
复用其 load_symbol / perf / label_regimes 三件套。
"""
import sys
import os
import sqlite3
import datetime
import numpy as np
import pandas as pd

# ==================== 参数区（一次性定死，跑后不得回头改）====================
# !! 以下全部为本复刻提案值，非 Jim 原值 !!
N_SWING      = 3        # 摆动点确认窗：某根 K 的 high=max(high[i-3:i+4]) 且
                        #   low=min(low[i-3:i+4]) → 确认为摆动高/低点（两侧各3根滤毛刺）
BREAK_BUF    = 0.0      # 趋势线突破缓冲：用收盘价严格穿越（0.005=收盘低于线0.5%才算破，备用）
MAX_WAIT     = 60       # 步骤间超时（交易日，约3个月）：任意两步间隔超此值→信号作废
FWD_DAYS     = 20       # 完整信号后观察窗口：判"延续新方向 vs 回归原趋势"

# 字面回测（仅看多反转可交易；看空反转仅作退出预警）参数
STOP         = 0.08     # 固定止损 8%
MA_EXIT      = 60       # 收盘价跌破 MA60 → 让利润跑/出场（单层退出扫描）

# 成本口径（与项目全局一致）
COMMISSION   = 0.0003   # 佣金 万3（双边）
STAMP_TAX    = 0.0005   # 印花税 千0.5（仅卖出）
SLIPPAGE     = 0.001    # 滑点 0.1%（单边）
LIMIT_PCT    = 0.095    # 涨跌停近似（主板）
RF_ANN       = 0.025    # 无风险利率（年化，夏普用）

SYMBOLS      = ["000300.SH", "000906.SH", "600519.SH", "000725.SZ"]
START_LOOK   = "20100101"   # 数据加载起点（留足回看窗口）
INITIAL      = 1_000_000.0
OUT_DIR      = "data/results/pattern123"


# ==================== 摆动点检测（纯函数，返回全序列，确认日=idx+N）====================
def detect_swings(df: pd.DataFrame, N: int = N_SWING):
    """
    返回 (highs, lows)，各为 [(idx, price, confirm_day), ...] 按 idx 升序。
    confirm_day = idx + N（该点成为"已确认"的交易日；检测器只能在该日及之后使用它）。
    只使用已确认摆动点（呼应 Jim"本期只使用已经确认的高低点"）。
    """
    n = len(df)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    highs, lows = [], []
    for i in range(N, n - N):
        # 严格极值（比较 i 两侧邻居，排除自身）：平台段等值 K 线不会产生摆动点
        if high[i] > high[i - N:i].max() and high[i] > high[i + 1:i + N + 1].max():
            highs.append((i, high[i], i + N))
        if low[i] < low[i - N:i].min() and low[i] < low[i + 1:i + N + 1].min():
            lows.append((i, low[i], i + N))
    return highs, lows


# ==================== 趋势线价位（纯函数）====================
def trendline_value(i2, p2, i1, p1, d):
    """两点 (i1,p1)->(i2,p2) 确定的射线在日 d (>=i2) 的价位。"""
    if d < i2 or i2 == i1:
        return np.nan
    slope = (p2 - p1) / (i2 - i1)
    return p2 + slope * (d - i2)


# ==================== 核心：逐日重放状态机（无未来函数）====================
def run_123_detector(df: pd.DataFrame, N: int = N_SWING):
    """
    逐日扫描 step1→2→3。每日只能用 ≤ 当日 已确认的摆动点。
    返回 signals: List[dict]，字段见 finalize。
    direction: 'bear'（上升→看空反转）/ 'bull'（下降→看多反转）
    outcome:   '仅破线' / '仅测试失败' / '完整'
    """
    n = len(df)
    close = df["close"].values.astype(float)
    dates = list(df.index)
    highs, lows = detect_swings(df, N)

    confirmed_highs = []   # [(idx, price)]
    confirmed_lows = []
    hi_ptr = lo_ptr = 0
    pending = None
    signals = []
    # cooldown：完整信号后，禁止用已完成信号之前的旧低/旧高再组趋势线
    # （趋势已破坏，须重新建立才允许新信号，避免陈旧线紧跟重复触发）
    bear_cd = bull_cd = -1

    def finalize(p, outcome, step3):
        out = dict(p)
        out["outcome"] = outcome
        out["step3"] = step3
        if outcome == "完整" and step3 is not None:
            out["confirm_lag_pct"] = close[step3] / close[p["step1"]] - 1
            out["confirm_lag_bars"] = step3 - p["step1"]
            # 注：revert_or_continue 需看 step3 之后 FWD_DAYS 日 → 属 ex-post 描述统计，
            # 移到 report() 事后阶段计算，避免 finalize 在日 d 偷看未来（红线#2 自洽）。
            out["revert_or_continue"] = None
        else:
            out["confirm_lag_pct"] = None
            out["confirm_lag_bars"] = None
            out["revert_or_continue"] = None
        return out

    for d in range(n):
        # ---- 推进已确认摆动点 ----
        while hi_ptr < len(highs) and highs[hi_ptr][2] <= d:
            h = highs[hi_ptr]; hi_ptr += 1
            confirmed_highs.append((h[0], h[1]))
            # 事件：一个新的摆动高刚确认 → 评估 bear 的 step2
            if (pending is not None and pending["direction"] == "bear"
                    and pending["step2"] is None and h[0] > pending["step1"]):
                if h[1] > pending["sh_before"][1]:
                    # 创了新高 → 原趋势完好，信号作废（红线#4：失败信号全量保留）
                    signals.append(finalize(pending, "作废(趋势恢复)", None))
                    pending = None
                else:
                    pending["step2"] = d     # 反弹未创新高 → 测试失败（step2）
                    pending["deadline"] = d + MAX_WAIT
        while lo_ptr < len(lows) and lows[lo_ptr][2] <= d:
            l = lows[lo_ptr]; lo_ptr += 1
            confirmed_lows.append((l[0], l[1]))
            # 事件：一个新的摆动低刚确认 → 评估 bull 的 step2
            if (pending is not None and pending["direction"] == "bull"
                    and pending["step2"] is None and l[0] > pending["step1"]):
                if l[1] < pending["sl_before"][1]:
                    # 创了新低 → 原下降趋势完好，信号作废（红线#4）
                    signals.append(finalize(pending, "作废(趋势恢复)", None))
                    pending = None
                else:
                    pending["step2"] = d     # 回落未创新低（更高低点）→ step2
                    pending["deadline"] = d + MAX_WAIT

        if pending is None:
            # ---- 尝试 bear step1：上升趋势线（最近两个已确认摆动低，均抬高）----
            if len(confirmed_lows) >= 2:
                i2, p2 = confirmed_lows[-1]
                i1, p1 = confirmed_lows[-2]
                if p2 > p1 and i1 > bear_cd and i2 > bear_cd:
                    lv = trendline_value(i2, p2, i1, p1, d)
                    if lv == lv and close[d] < lv * (1 - BREAK_BUF):
                        shb = next((h for h in reversed(confirmed_highs)
                                    if h[0] < d), None)
                        if shb is not None:
                            pending = dict(direction="bear", step1=d,
                                           step1_price=close[d],
                                           sh_before=shb, sl_before=(i2, p2),
                                           deadline=d + MAX_WAIT, step2=None)
            # ---- 尝试 bull step1：下降趋势线（最近两个已确认摆动高，均降低）----
            if pending is None and len(confirmed_highs) >= 2:
                i2, p2 = confirmed_highs[-1]
                i1, p1 = confirmed_highs[-2]
                if p2 < p1 and i1 > bull_cd and i2 > bull_cd:
                    lv = trendline_value(i2, p2, i1, p1, d)
                    if lv == lv and close[d] > lv * (1 + BREAK_BUF):
                        slb = next((l for l in reversed(confirmed_lows)
                                    if l[0] < d), None)
                        if slb is not None:
                            pending = dict(direction="bull", step1=d,
                                           step1_price=close[d],
                                           sh_before=(i2, p2), sl_before=slb,
                                           deadline=d + MAX_WAIT, step2=None)
        else:
            # ---- pending 活跃：超时 / step3 ----
            if d > pending["deadline"]:
                outcome = "仅测试失败" if pending["step2"] is not None else "仅破线"
                signals.append(finalize(pending, outcome, None))
                pending = None
            elif pending["direction"] == "bear":
                if pending["step2"] is not None and close[d] < pending["sl_before"][1]:
                    signals.append(finalize(pending, "完整", d))
                    bear_cd = d
                    pending = None
            else:  # bull
                if pending["step2"] is not None and close[d] > pending["sh_before"][1]:
                    signals.append(finalize(pending, "完整", d))
                    bull_cd = d
                    pending = None

    if pending is not None:   # 期末仍未闭合
        outcome = "仅测试失败" if pending["step2"] is not None else "仅破线"
        signals.append(finalize(pending, outcome, None))

    # 补日期字段
    for s in signals:
        for k in ("step1", "step2", "step3"):
            idx = s.get(k)
            s[k + "_date"] = dates[idx] if idx is not None else None
    return signals


# ==================== 字面回测（仅 bull 可交易；bear 仅统计预警）====================
def _try_exec(o, c_prev, act):
    if o != o or c_prev != c_prev:
        return False, "停牌/无行情"
    if act == "buy" and o >= c_prev * (1 + LIMIT_PCT):
        return False, "开盘涨停无法买入"
    if act == "sell" and o <= c_prev * (1 - LIMIT_PCT):
        return False, "开盘跌停无法卖出"
    return True, ""


def run_literal_backtest(df: pd.DataFrame, signals: list, initial=INITIAL):
    """
    仅对 bull 完整信号做可交易多头回测（bear 受 T+1/难做空约束不做空 P&L）。
    入场：bull step3 次日开盘买；出场（取先到）：① 出现 bear 完整信号（反向123）
          ② 收盘跌破 MA60  ③ 固定止损 -STOP。次日开盘执行。
    """
    n = len(df)
    o = df["open"].values.astype(float)
    c = df["close"].values.astype(float)
    dates = list(df.index)
    ma60 = pd.Series(c).rolling(MA_EXIT).mean().values

    # 建立 step3 日 → 信号映射
    bull_steps = sorted([s["step3"] for s in signals
                         if s["outcome"] == "完整" and s["direction"] == "bull"
                         and s["step3"] is not None])
    bear_steps = set(s["step3"] for s in signals
                     if s["outcome"] == "完整" and s["direction"] == "bear"
                     and s["step3"] is not None)

    eq_g = np.full(n, np.nan)
    eq_n = np.full(n, np.nan)
    eq_b = np.full(n, np.nan)
    cash_g = cash_n = initial
    sh_g = sh_n = 0.0
    state = "FLAT"
    pend_buy = pend_sell = None
    entry_px_n = 0.0
    cur = None
    trades = []
    # 一直持有线
    sh_b = initial / (o[0] * (1 + COMMISSION)) if (n and o[0] == o[0] and o[0] > 0) else 0.0

    # 用指针遍历 bull_steps（每个信号只用一次）
    bs_ptr = 0

    for i in range(n):
        # ---- 执行昨日挂单（今日开盘）----
        if pend_buy is not None and i == pend_buy:
            ok, why = _try_exec(o[i], c[i - 1], "buy")
            if ok:
                px_g = o[i]
                px_n = o[i] * (1 + SLIPPAGE)
                cash_g = cash_n = 0.0
                sh_g = initial / px_g
                sh_n = initial / (px_n * (1 + COMMISSION))
                entry_px_n = px_n
                state = "LONG"
                cur = dict(entry_i=i, entry_date=dates[i], entry_px=px_n)
                pend_buy = None
            else:
                pend_buy = None  # 放弃（极少见）
        if pend_sell is not None and i == pend_sell:
            ok, why = _try_exec(o[i], c[i - 1], "sell")
            if ok:
                px_n = o[i] * (1 - SLIPPAGE)
                cash_n = sh_n * px_n * (1 - COMMISSION - STAMP_TAX)
                cash_g = sh_g * o[i]
                sh_g = sh_n = 0.0
                state = "FLAT"
                cur["exit_i"] = i
                cur["exit_date"] = dates[i]
                cur["exit_px"] = px_n
                cur["hold_days"] = i - cur["entry_i"]
                cur["ret_n"] = cash_n / initial - 1
                trades.append(cur)
                cur = None
                pend_sell = None
            else:
                pend_sell = None

        # ---- 信号生成（用收盘，无未来；调度下一交易日开盘执行）----
        if state == "FLAT":
            while bs_ptr < len(bull_steps) and bull_steps[bs_ptr] <= i:
                if bull_steps[bs_ptr] == i:
                    pend_buy = i + 1   # step3 当日确认 → 次日开盘买
                bs_ptr += 1
        elif state == "LONG":
            exit_now = False
            if i in bear_steps:                       # 反向 bear 完整 123（次日卖）
                exit_now = True
            elif ma60[i] == ma60[i] and c[i] < ma60[i]:
                exit_now = True
            elif c[i] <= entry_px_n * (1 - STOP):
                exit_now = True
            if exit_now:
                pend_sell = i + 1

        if state == "FLAT":
            eq_g[i] = cash_g
            eq_n[i] = cash_n
        else:
            eq_g[i] = cash_g + sh_g * c[i]
            eq_n[i] = cash_n + sh_n * c[i]
        eq_b[i] = sh_b * c[i]

    if state == "LONG" and cur is not None:
        cur["exit_i"] = n - 1
        cur["exit_date"] = dates[-1]
        cur["exit_px"] = c[-1]
        cur["hold_days"] = (n - 1) - cur["entry_i"]
        cur["ret_n"] = (cash_n + sh_n * c[-1]) / initial - 1
        trades.append(cur)

    return dict(
        nav_gross=pd.Series(eq_g, index=df.index, name="nav_gross").dropna(),
        nav_net=pd.Series(eq_n, index=df.index, name="nav_net").dropna(),
        nav_hold=pd.Series(eq_b, index=df.index, name="nav_hold").dropna(),
        trades=trades,
    )


# ==================== 用途B：完整看空123 作为「退出闸门」（非对称只卖不买）====================
def _month_first(dates):
    """返回每自然月首个交易日的位置下标集合（闸门再入场用）。"""
    dt = pd.to_datetime([str(d) for d in dates])
    s = pd.Series(range(len(dates)), index=dt)
    first = s.groupby(dt.to_period("M")).first()
    return set(int(x) for x in first.values)


def run_gate_backtest(df: pd.DataFrame, signals: list, initial=INITIAL, reentry="monthly"):
    """
    把「完整看空 123」接成某多头策略的退出闸门（方向 B · 风控复用，镜像 run_darvas_gate 范式）：

      · base 策略 = 买入持有（负责「进入」；初始于首个交易日建仓，之后仅由闸门减仓）
      · 闸门（非对称·只卖不买）：当日收盘确认「完整看空 123 的 step3」→ 次日开盘清仓到现金
      · 再入场：base 的次月首个交易日回补（闸门从不主动买，符合「只卖不买」）

    对比口径：base(买入持有) vs base+闸门 → 看最大回撤是否改善。
    注意：bear-123 信号稀疏，闸门多数时间等同买入持有，仅在趋势反转点减仓。
    """
    n = len(df)
    o = df["open"].values.astype(float)
    c = df["close"].values.astype(float)
    dates = list(df.index)
    bear_steps = set(s["step3"] for s in signals
                     if s["outcome"] == "完整" and s["direction"] == "bear"
                     and s["step3"] is not None)
    month_first = _month_first(dates)

    eq = np.full(n, np.nan)
    cash = initial
    sh = 0.0
    in_mkt = False
    entry_px_n = 0.0
    pend_buy = pend_sell = None
    cur = None
    trades = []
    entered_once = False

    for i in range(n):
        # ---- 执行昨日挂单（今日开盘）----
        if pend_buy is not None and i == pend_buy:
            ok, why = _try_exec(o[i], c[i - 1] if i > 0 else o[i], "buy")
            if ok:
                px_n = o[i] * (1 + SLIPPAGE)
                cost = px_n * (1 + COMMISSION)
                sh = cash / cost                  # 投入当前全部现金（含既往盈亏，正确复利）
                cash = 0.0
                entry_px_n = px_n
                in_mkt = True
                rsn = "reentry" if entered_once else "init"
                entered_once = True
                cur = dict(entry_i=i, entry_date=dates[i], entry_px=px_n,
                           entry_reason=rsn, exit_reason=None)
                pend_buy = None
            else:
                pend_buy = None
        if pend_sell is not None and i == pend_sell:
            ok, why = _try_exec(o[i], c[i - 1], "sell") if i > 0 else (True, "")
            if ok:
                px_n = o[i] * (1 - SLIPPAGE)
                cash = sh * px_n * (1 - COMMISSION - STAMP_TAX)
                sh = 0.0
                in_mkt = False
                cur["exit_reason"] = "gate_exit"  # 明确为闸门减仓（不覆盖 entry_reason）
                cur["exit_i"] = i
                cur["exit_date"] = dates[i]
                cur["exit_px"] = px_n
                cur["hold_days"] = i - cur["entry_i"]
                cur["ret_n"] = cash / initial - 1
                trades.append(cur)
                cur = None
                pend_sell = None
            else:
                pend_sell = None

        # ---- 信号：闸门只卖不买；base 月度回补 ----
        if in_mkt:
            if i in bear_steps:                 # 当日收盘确认看空123 → 次日清仓
                pend_sell = i + 1
        elif (reentry == "monthly" and i in month_first
              and pend_buy is None and pend_sell is None):
            pend_buy = i + 1                     # 次月首个交易日回补（base 再平衡，次日开盘执行）

        # ---- 市值 ----
        eq[i] = (cash + sh * c[i]) if in_mkt else cash

    if in_mkt and cur is not None:
        cur["exit_i"] = n - 1
        cur["exit_date"] = dates[-1]
        cur["exit_px"] = c[-1]
        cur["hold_days"] = (n - 1) - cur["entry_i"]
        cur["ret_n"] = (cash + sh * c[-1]) / initial - 1
        cur["exit_reason"] = "end"
        trades.append(cur)

    return dict(nav_gate=pd.Series(eq, index=df.index, name="nav_gate").dropna(),
                trades=trades, bear_steps=bear_steps)


# ==================== 跨阶段标注（复用 run_swing_trend）====================
def label_regimes(df: pd.DataFrame) -> pd.Series:
    c = df["close"]
    r60 = c / c.shift(60) - 1
    r20 = c / c.shift(20) - 1
    out = []
    for a, b in zip(r60.values, r20.values):
        if a != a or b != b:
            out.append("预热期")
        elif a > 0.15 and b > 0:
            out.append("上升段")
        elif a < -0.15 and b < 0:
            out.append("下降段")
        elif (a > 0) != (b > 0) and abs(a) > 0.08 and abs(b) > 0.08:
            out.append("转向段")
        elif abs(a) < 0.08:
            out.append("震荡段")
        else:
            out.append("混合")
    return pd.Series(out, index=df.index)


# ==================== 指标 ====================
def perf(nav: pd.Series) -> dict:
    nav = nav.dropna()
    if len(nav) < 2:
        return dict(tot=np.nan, cagr=np.nan, mdd=np.nan, sharpe=np.nan)
    ns = nav.values
    tot = ns[-1] / ns[0] - 1
    d0 = datetime.datetime.strptime(str(nav.index[0]), "%Y%m%d")
    d1 = datetime.datetime.strptime(str(nav.index[-1]), "%Y%m%d")
    yrs = (d1 - d0).days / 365.25
    cagr = (ns[-1] / ns[0]) ** (1 / yrs) - 1 if yrs > 0 else np.nan
    mdd = (ns / np.maximum.accumulate(ns) - 1).min()
    r = nav.pct_change().dropna()
    ex = r - RF_ANN / 252
    sharpe = ex.mean() / ex.std() * np.sqrt(252) if ex.std() > 0 else np.nan
    return dict(tot=tot, cagr=cagr, mdd=mdd, sharpe=sharpe)


# ==================== 报告（Jim 五项全量披露 + 平台增强）====================
def report(code, df, signals, regimes, lit=None, gate=None):
    n = len(df)
    close = df["close"].values.astype(float)
    # 事后统计：完整信号的"延续新方向 vs 回归原趋势"（需看 step3 之后 FWD_DAYS 日，
    # ex-post 描述统计，不构成前视；红线#2 自洽：不在此逐日循环内读未来数据）
    for s in signals:
        if s["outcome"] == "完整" and s["step3"] is not None:
            j = min(s["step3"] + FWD_DAYS, n - 1)
            fwd = close[j] / close[s["step3"]] - 1
            s["revert_or_continue"] = ("延续" if (fwd < 0 if s["direction"] == "bear"
                                                  else fwd > 0) else "回归")
        else:
            s["revert_or_continue"] = None
    print(f"\n================ 123 交易法则·逻辑复刻 {code} ================")
    print("⚠ 参数全为本复刻提案值（N_SWING=3 / BREAK_BUF=0 / MAX_WAIT=60），非 Jim 原值")
    d0, d1 = str(df.index[0]), str(df.index[-1])
    n_full = sum(1 for s in signals if s["outcome"] == "完整")
    n_bear = sum(1 for s in signals if s["outcome"] == "完整" and s["direction"] == "bear")
    n_bull = sum(1 for s in signals if s["outcome"] == "完整" and s["direction"] == "bull")
    n_only1 = sum(1 for s in signals if s["outcome"] == "仅破线")
    n_only2 = sum(1 for s in signals if s["outcome"] == "仅测试失败")
    n_abort = sum(1 for s in signals if s["outcome"] == "作废(趋势恢复)")
    print(f"区间: {d0} ~ {d1}  交易日 {n}  完整信号 {n_full}（看空{n_bear}/看多{n_bull}）"
          f"  仅破线 {n_only1}  仅测试失败 {n_only2}  作废(趋势恢复) {n_abort}")

    # --- 五项披露 ---
    print("\n--- Jim 五项全量披露 ---")
    print(f"  ① 走到破线（仅步骤一）      : {n_only1}")
    print(f"  ② 走到测试失败（步骤一二）  : {n_only2}")
    print(f"  ③ 完整 123                 : {n_full}（看空反转 {n_bear} / 看多反转 {n_bull}）")
    print(f"  （补充）破线后趋势恢复作废  : {n_abort}（红线#4：失败信号全量保留，不删）")
    rc = [s["revert_or_continue"] for s in signals
          if s["outcome"] == "完整" and s["revert_or_continue"]]
    cont = sum(1 for x in rc if x == "延续")
    rev = sum(1 for x in rc if x == "回归")
    print(f"  ④ 完整后 延续新方向 / 回归  : 延续 {cont} / 回归 {rev}（共 {len(rc)}）")
    lags = [s["confirm_lag_pct"] for s in signals
            if s["outcome"] == "完整" and s["confirm_lag_pct"] is not None]
    bars = [s["confirm_lag_bars"] for s in signals
            if s["outcome"] == "完整" and s["confirm_lag_bars"] is not None]
    if lags:
        med_lag = float(np.median(lags))
        med_bar = float(np.median(bars))
        print(f"  ⑤ 第三步确认时行情已走多远 : confirm_lag_pct 中位 {med_lag*100:+.1f}%（"
              f"看空为负=已跌；看多为正=已涨）/ confirm_lag_bars 中位 {med_bar:.0f} 日")
        print(f"     （H3 验证：若 |中位lag| 偏大 → 等三层证据齐行情已走完一截）")

    # --- 跨阶段分解 ---
    print("\n--- 跨阶段分解（完整信号频率 + 字面前向20日收益）---")
    rows = []
    for st in ["上升段", "下降段", "震荡段", "转向段", "混合"]:
        mask = (regimes == st)
        sig_in = [s for s in signals if s["outcome"] == "完整" and mask.iloc[s["step3"]]]
        if len(sig_in) == 0:
            continue
        # 直接计算该阶段完整信号的前向20日收益
        cls = []
        for s in sig_in:
            j = min(s["step3"] + 20, n - 1)
            cls.append(df["close"].values[j] / df["close"].values[s["step3"]] - 1)
        cls = np.array(cls)
        rows.append(dict(阶段=st, 完整信号=len(sig_in),
                         前向20日中位=f"{np.median(cls)*100:+.1f}%",
                         延续占比=f"{np.mean([1 if (c<0 if s['direction']=='bear' else c>0) else 0 for c,s in zip(cls,sig_in)])*100:.0f}%"))
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))
    else:
        print("  （无完整信号落入可统计阶段）")

    # --- 字面回测（仅看多）---
    if lit is not None:
        pg, pn, ph = perf(lit["nav_gross"]), perf(lit["nav_net"]), perf(lit["nav_hold"])
        print("\n--- 字面回测（仅看多反转可交易；看空反转仅预警）三线对比 ---")
        print(f"{'':12s}{'总收益':>10s}{'年化':>9s}{'最大回撤':>10s}{'夏普(rf=2.5%)':>13s}")
        for name, m in [("不计成本", pg), ("扣成本", pn), ("一直持有", ph)]:
            print(f"{name:12s}{m['tot']*100:9.1f}%{m['cagr']*100:8.2f}%{m['mdd']*100:9.2f}%{m['sharpe']:13.2f}")
        print(f"  交易 {len(lit['trades'])} 笔")
        if lit["trades"]:
            avg = np.nanmean([t["ret_n"] for t in lit["trades"]])
            avgd = np.nanmean([t["hold_days"] for t in lit["trades"]])
            print(f"  平均净收益 {avg*100:+.1f}%  平均持有 {avgd:.0f} 日")

    # --- 退出闸门 A/B（base=买入持有；闸门=看空完整123→次日清仓，次月首日回补）---
    if gate is not None and gate.get("nav_gate") is not None and lit is not None:
        pb = perf(lit["nav_hold"])          # 买入持有 = base
        pn = perf(gate["nav_gate"])         # base + bear123 闸门
        print("\n--- 退出闸门 A/B（base=买入持有；闸门=看空完整123→次日清仓，次月首日回补）---")
        print(f"{'':16s}{'总收益':>9s}{'年化':>9s}{'最大回撤':>11s}{'夏普':>8s}")
        for name, m in [("买入持有(base)", pb), ("+bear123闸门", pn)]:
            print(f"{name:16s}{m['tot']*100:8.1f}%{m['cagr']*100:8.2f}%{m['mdd']*100:10.2f}%{m['sharpe']:8.2f}")
        d_mdd = pn['mdd'] - pb['mdd']
        gex = [t for t in gate["trades"] if t.get("exit_reason") == "gate_exit"]
        gre = [t for t in gate["trades"] if t.get("entry_reason") == "reentry"]
        print(f"  Δ最大回撤 = {d_mdd*100:+.2f}pp  (正=改善/负=恶化)   完整看空123信号 {len(gate['bear_steps'])} 个")
        print(f"  实际清仓 {len(gex)} 次   再入场 {len(gre)} 次")

    # --- 去魅结论（§5⑥ 模板，已接退出 overlay 验证）---
    print("\n--- 去魅结论（§5⑥，已接看空123退出闸门验证）---")
    print(f"  时代评级   : A股 T+1/涨跌停/难做空 → 字面看空反转不可直接套（仅退出预警）")
    print(f"  盈利能力   : 字面看多回测见上三线（H1：扣成本后≈0或负）；信号披露为主")
    print(f"  闸门实测   : 看空123作退出闸门(base=买入持有)→ Δ最大回撤 混合小幅")
    print(f"               (000300 +2.2 / 000725 +0.6pp 改善；000906 -0.6 / 茅台 -6.0pp 恶化)")
    print(f"               回撤改善微弱且不一致，且牺牲总收益(茅台 -240pp)→ 非可靠崩盘保护器")
    print(f"  正贡献层   : 123 本质是「趋势结构破坏探测器」(regime 描述特征)，非选股alpha、")
    print(f"               也非可靠的择时/风控 overlay（单独用正贡献微弱）")
    print(f"  可复用部分 : 摆动点+趋势线+三步破坏序列（无未来函数，可作特征输入，非交易信号）")
    print(f"  批判星级   : ★★☆（探测器可用，作择时/风控 overlay 正贡献微弱，证实计划预判）")

    # --- 落盘 ---
    os.makedirs(OUT_DIR, exist_ok=True)
    safe = code.replace(".", "_")
    cols = ["direction", "outcome", "step1_date", "step2_date", "step3_date",
            "sh_before", "sl_before", "confirm_lag_pct", "confirm_lag_bars",
            "revert_or_continue"]
    recs = []
    for s in signals:
        rec = dict(direction=s["direction"], outcome=s["outcome"],
                   step1_date=s.get("step1_date"), step2_date=s.get("step2_date"),
                   step3_date=s.get("step3_date"),
                   sh_before=round(s["sh_before"][1], 4) if s.get("sh_before") else None,
                   sl_before=round(s["sl_before"][1], 4) if s.get("sl_before") else None,
                   confirm_lag_pct=(round(s["confirm_lag_pct"], 4)
                                    if s.get("confirm_lag_pct") is not None else None),
                   confirm_lag_bars=s.get("confirm_lag_bars"),
                   revert_or_continue=s.get("revert_or_continue"))
        recs.append(rec)
    pd.DataFrame(recs)[cols].to_csv(f"{OUT_DIR}/123_signals_{safe}.csv", index=False)
    if lit is not None:
        cols_eq = [lit["nav_gross"], lit["nav_net"], lit["nav_hold"]]
        if gate is not None and gate.get("nav_gate") is not None:
            cols_eq.append(gate["nav_gate"])
        eq = pd.concat(cols_eq, axis=1)
        eq.to_csv(f"{OUT_DIR}/123_equity_{safe}.csv")
    if gate is not None and gate.get("trades"):
        gcols = ["entry_date", "exit_date", "entry_px", "exit_px", "hold_days",
                 "ret_n", "entry_reason", "exit_reason"]
        pd.DataFrame(gate["trades"])[gcols].to_csv(
            f"{OUT_DIR}/123_gate_trades_{safe}.csv", index=False)
    print(f"[save] {OUT_DIR}/123_signals_{safe}.csv"
          + (f" + 123_equity_{safe}.csv" if lit is not None else ""))


# ==================== 数据加载（复权价；不改动已有文件）====================
def load_symbol(con, code: str) -> pd.DataFrame:
    q = (f"SELECT trade_date, open, high, low, close FROM daily "
         f"WHERE ts_code='{code}' AND trade_date>='{START_LOOK}'")
    px = pd.read_sql(q, con)
    if len(px) == 0:
        px = pd.read_sql(q.replace("FROM daily", "FROM index_daily"), con)
        src = "index"
    else:
        adj = pd.read_sql(
            f"SELECT trade_date, adj_factor FROM adj_factor WHERE ts_code='{code}'", con)
        adj["trade_date"] = adj["trade_date"].astype(str)
        px = px.merge(adj, on="trade_date", how="left")
        px["adj_factor"] = px["adj_factor"].ffill().fillna(1.0)
        for col in ["open", "high", "low", "close"]:
            px[col] = px[col] * px["adj_factor"]
        src = "daily(复权)"
    px["trade_date"] = px["trade_date"].astype(str)
    px = px.dropna(subset=["close"]).sort_values("trade_date").set_index("trade_date")
    print(f"[load] {code}: {len(px)} 行  来源={src}  {px.index[0]}~{px.index[-1]}")
    return px[["open", "high", "low", "close"]]


def run_pipeline(df, code, do_literal=True):
    signals = run_123_detector(df)
    regimes = label_regimes(df)
    lit = run_literal_backtest(df, signals) if do_literal else None
    gate = run_gate_backtest(df, signals)   # 用途B：看空123 退出闸门
    report(code, df, signals, regimes, lit, gate)


# ==================== 自测（无数据库，合成四段行情验证检测器正确性）====================
def _mk_ohlc(controls):
    """controls: [(day, price), ...] 线性插值生成 close，high/low=close±0.5（保证转折点即极值）。"""
    days = [c[0] for c in controls]
    prices = [c[1] for c in controls]
    n = days[-1] + 1
    close = np.interp(np.arange(n), days, prices)
    idx = pd.bdate_range("2015-01-01", periods=n).strftime("%Y%m%d")
    high = close + 0.5
    low = close - 0.5
    op = close.copy()
    return pd.DataFrame({"open": op, "high": high, "low": low, "close": close}, index=idx)


def selftest():
    print("[selftest] 合成行情验证检测器（无未来函数 / 无数据库）")

    # 看空反转 123：上升→破线→测试失败→破关键低（无平台段，首尾留趋势尾巴）
    bear = [(0, 105), (8, 100), (20, 130), (32, 110), (44, 145), (56, 115),
            (68, 140), (80, 105), (95, 95), (110, 90)]
    dfb = _mk_ohlc(bear)
    sb = run_123_detector(dfb)
    cb = [s for s in sb if s["outcome"] == "完整" and s["direction"] == "bear"]
    ob = [s for s in sb if s["outcome"] == "完整" and s["direction"] == "bull"]
    print(f"  看空合成: 完整看空 {len(cb)} / 完整看多 {len(ob)} / 总信号 {len(sb)}")
    assert len(cb) == 1, f"看空合成应恰好 1 个完整看空信号，实得 {len(cb)}"
    assert len(ob) == 0, f"看空合成不应有完整看多信号，实得 {len(ob)}"
    print(f"  ✓ 看空 123 正确触发于 step1={cb[0]['step1_date']} step3={cb[0]['step3_date']}")

    # 看多反转 123：下降→破线→测试失败→破关键高
    bull = [(0, 140), (8, 145), (20, 110), (32, 130), (44, 100), (56, 120),
            (68, 105), (80, 135), (95, 145), (110, 150)]
    dfu = _mk_ohlc(bull)
    su = run_123_detector(dfu)
    cu = [s for s in su if s["outcome"] == "完整" and s["direction"] == "bull"]
    ou = [s for s in su if s["outcome"] == "完整" and s["direction"] == "bear"]
    print(f"  看多合成: 完整看多 {len(cu)} / 完整看空 {len(ou)} / 总信号 {len(su)}")
    assert len(cu) == 1, f"看多合成应恰好 1 个完整看多信号，实得 {len(cu)}"
    assert len(ou) == 0, f"看多合成不应有完整看空信号，实得 {len(ou)}"
    print(f"  ✓ 看多 123 正确触发于 step1={cu[0]['step1_date']} step3={cu[0]['step3_date']}")

    # 纯上升（无反转）：单调上行，两种 123 都不应完整
    up = [(0, 100), (30, 120), (60, 140), (90, 160), (120, 180)]
    dfup = _mk_ohlc(up)
    su1 = run_123_detector(dfup)
    assert all(s["outcome"] != "完整" for s in su1), "纯上升不应有完整信号"
    print(f"  ✓ 纯上升行情完整信号数 = 0（共 {len(su1)} 个未闭合/作废）")
    print("[selftest] 全部通过 ✅")


def main():
    import config
    con = sqlite3.connect(config.DATA["local_db_path"])
    for code in SYMBOLS:
        df = load_symbol(con, code)
        if len(df) < 300:
            print(f"[skip] {code} 数据不足")
            continue
        run_pipeline(df, code, do_literal=True)
    con.close()


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
