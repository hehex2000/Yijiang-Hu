# -*- coding: utf-8 -*-
"""
run_swing_trend.py —— 《波段操作法》五步流程量化实现（BV16agJ6EE8a 逻辑复刻）
=============================================================================
视频把"波段=低买高卖"拆成五步：定周期→识方向→跟变化→记结束→算结果。
本脚本把五步全部映射为可计算规则，并严格执行视频的三条铁律：
  1) 三线对比：不计成本 / 扣成本 / 同期一直持有（不跟"完美波段"比，跟"什么都不做"比）
  2) 错误分类不藏：正确结束 / 提前结束(暂时反走) / 发现太晚 / 未执行 全部原样保留
  3) 跨阶段验证：同一套规则在 方向清楚 / 来回震荡 / 突然转向 三种阶段分别统计

!!! 参数声明：视频只给流程未给数值。以下 N_MOM / MIN_SWING / D_CONFIRM / K_EXIT 等
    为本复刻一次性定死的假设值（文件头集中声明；按视频原话"标准一旦定好，不能看到
    结果以后再改"——跑完回测后不得回头调参再跑，否则就是数据窥探）。

五步 → 量化映射：
  步骤1 定周期  -> N_MOM(动量观察窗) + MIN_SWING("值得关注的波动"幅度门槛)
  步骤2 识方向  -> 三件事：①20日动量方向 ②20日滚动高低点同向移动 ③连续 D_CONFIRM 日
  步骤3 跟变化  -> 持仓期同三件事；反向条件须持续 K_EXIT 日才出场（"不要一两天
                   反着走就立刻换方向"）；方向不明日照实计数
  步骤4 记结束  -> 反向确认即结束，并当场归档错误分类（用出场后数据做"事后归因"，
                   仅用于统计，绝不回流到信号）
  步骤5 算结果  -> 三线净值 + 佣金/印花税/滑点 + 滞后/追高度量 + 分阶段分解

回测正确性（沿用本项目已修正的全部原则）：
  - 信号 t 日收盘计算，t+1 开盘执行（含滑点）—— 无未来函数
  - 复权价（adj_factor），三线同口径，对比公平
  - 涨跌停/停牌：顺延最多 EXEC_WAIT_MAX 日，失败原样记入"未执行"，不悄悄丢弃
  - 净值从 rolling 窗口预热完成后起算，三线同起点，无空仓期稀释
  - 夏普减无风险利率；错误归因段明确标注"事后分析允许用未来数据"
"""
import sys
import sqlite3
import datetime
import numpy as np
import pandas as pd

# ==================== 参数区（一次性定死，回测后不得回头改）====================
N_MOM         = 20      # 步骤1/2①：观察周期——20日动量衡量"价格整体抬高还是降低"
N_HHLL        = 20      # 步骤2②：20日滚动高低点，衡量"新高/新低往哪边移动"
HHLL_STEP     = 5       # 步骤2②：高低点与5日前对比（上移/下移）
MIN_SWING     = 0.05    # 步骤1："价格变化多大才算值得关注的波动"→|20日动量|>=5%
D_CONFIRM     = 3       # 步骤2③："不只一两天"→连续3日成立才确认入场
K_EXIT        = 3       # 步骤4："变化持续出现达到提前定好的程度"→反向连续3日才出场
EXEC_WAIT_MAX = 3       # 涨跌停/停牌顺延最多3日，失败记"未执行"
LIMIT_PCT     = 0.095   # 涨跌停判定近似（主板；创业板/科创板标的需改0.195）
RF_ANN        = 0.025   # 无风险利率（年化，夏普用）

# 成本口径（与项目全局一致；若 config 有全局设置请对齐这三行）
COMMISSION    = 0.0003  # 佣金 万3（双边）
STAMP_TAX     = 0.0005  # 印花税 千0.5（仅卖出）
SLIPPAGE      = 0.001   # 滑点 0.1%（单边）

# 错误归因参数（仅事后统计，不进入信号）
FWD_DAYS      = 10      # 出场后观察10日判断是否"提前结束"
RECONF_PCT    = 0.03    # 出场后10日内再走3%原方向 → 判"提前结束(暂时反走)"
CHASE_PCT     = 0.08    # 买价高于段内最低价8% → 判"发现太晚"

SYMBOLS       = ["600519.SH", "000001.SZ"]   # 默认标的（跨波动特征对比）
START_LOOK    = "20120101"                   # 数据加载起点（留足回看窗口）
INITIAL       = 1_000_000.0


# ==================== 步骤1+2：方向条件（纯函数，无未来数据）====================
def compute_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入: df(index=trade_date, columns=[open,high,low,close]，均复权)
    输出: 追加 up_cond / dn_cond 两列（当日收盘即可得，无未来函数）
      up_cond = 20日动量>=5% 且 20日滚动高点上移 且 20日滚动低点上移   （三件事同向·涨）
      dn_cond = 20日动量<=-5% 且 20日滚动高点下移 且 20日滚动低点下移  （三件事同向·跌）
    两列同时为 False 即视频说的"互相矛盾/看不清，继续等"。
    """
    out = df.copy()
    close, high, low = out["close"], out["high"], out["low"]
    mom = close / close.shift(N_MOM) - 1
    hh = high.rolling(N_HHLL).max()
    ll = low.rolling(N_HHLL).min()
    hh_up = hh > hh.shift(HHLL_STEP)
    ll_up = ll > ll.shift(HHLL_STEP)
    hh_dn = hh < hh.shift(HHLL_STEP)
    ll_dn = ll < ll.shift(HHLL_STEP)
    out["mom"] = mom
    out["up_cond"] = ((mom >= MIN_SWING) & hh_up & ll_up).fillna(False)
    out["dn_cond"] = ((mom <= -MIN_SWING) & hh_dn & ll_dn).fillna(False)
    return out


# ==================== 步骤3+4：状态机回测（三线同账本推进）====================
def _try_exec(i, o, c, act):
    """执行日 i 开盘可否成交：停牌/开盘涨跌停不可成交。返回 (ok, 原因)。"""
    if o[i] != o[i] or c[i - 1] != c[i - 1]:
        return False, "停牌/无行情"
    if act == "buy" and o[i] >= c[i - 1] * (1 + LIMIT_PCT):
        return False, "开盘涨停无法买入"
    if act == "sell" and o[i] <= c[i - 1] * (1 - LIMIT_PCT):
        return False, "开盘跌停无法卖出"
    return True, ""


def run_swing_backtest(df: pd.DataFrame, initial=INITIAL) -> dict:
    """
    状态机：FLAT --up连续D_CONFIRM日--> 挂买单 --> 次日开盘成交 --> LONG
            LONG --dn连续K_EXIT日--> 挂卖单 --> 次日开盘成交 --> FLAT
    执行：t日收盘出信号，t+1开盘成交；不可成交顺延<=EXEC_WAIT_MAX日，失败记 skipped。
    三套账本：gross(无成本) / net(佣金+印花税+滑点) / hold(期初买入一直持有,净口径)。
    """
    n = len(df)
    o = df["open"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    up = df["up_cond"].values.astype(bool)
    dn = df["dn_cond"].values.astype(bool)
    dates = df.index

    cash_g = cash_n = initial
    sh_g = sh_n = 0.0
    state = "FLAT"
    pending = None            # {"act","sig","first","age"}
    up_st = dn_st = 0
    false_starts = 0          # FLAT期方向苗头(1~D-1日)消失次数 = "中途改过几次"的代理
    trades, skipped = [], []
    eq_g = np.full(n, np.nan)
    eq_n = np.full(n, np.nan)
    eq_b = np.full(n, np.nan)

    # 一直持有线：首日开盘买入（净口径），持有到期末按收盘市值计
    if n and o[0] == o[0] and o[0] > 0:
        sh_b = initial / (o[0] * (1 + COMMISSION))
    else:
        sh_b = 0.0

    cur = None
    for i in range(n):
        # ---- A. 执行昨日挂单（今日开盘）----
        if pending is not None:
            ok, why = _try_exec(i, o, c, pending["act"])
            if ok:
                if pending["act"] == "buy":
                    px_g = o[i]
                    px_n = o[i] * (1 + SLIPPAGE)
                    cur = dict(entry_i=i, entry_date=dates[i],
                               entry_px_g=px_g, entry_px=px_n,
                               first_cond_i=pending["first"],
                               cash_before_g=cash_g, cash_before_n=cash_n,
                               unclear=0)
                    sh_g = cash_g / px_g
                    sh_n = cash_n / (px_n * (1 + COMMISSION))
                    cash_g = cash_n = 0.0
                    state = "LONG"
                else:
                    px_g = o[i]
                    px_n = o[i] * (1 - SLIPPAGE)
                    cash_g = sh_g * px_g
                    cash_n = sh_n * px_n * (1 - COMMISSION - STAMP_TAX)
                    sh_g = sh_n = 0.0
                    state = "FLAT"
                    cur["exit_i"] = i
                    cur["exit_date"] = dates[i]
                    cur["exit_px_g"] = px_g
                    cur["exit_px"] = px_n
                    cur["hold_days"] = i - cur["entry_i"]
                    # 净收益=现金口径（含买卖全部费用）；毛收益=无成本口径
                    cur["ret_n"] = cash_n / cur["cash_before_n"] - 1
                    cur["ret_g"] = cash_g / cur["cash_before_g"] - 1
                    # 确认代价：入场价相对"方向首次成立日"收盘的滞后
                    fc = cur["first_cond_i"]
                    cur["lag"] = cur["entry_px_g"] / c[fc] - 1 if c[fc] > 0 else np.nan
                    # 追高程度：入场价相对段内最低价（含确认期）
                    lo_i = max(0, fc - N_MOM)
                    seg_low = np.nanmin(l[lo_i:i + 1])
                    cur["chase"] = cur["entry_px_g"] / seg_low - 1 if seg_low > 0 else np.nan
                    trades.append(cur)
                    cur = None
                pending = None
            else:
                pending["age"] += 1
                if pending["age"] > EXEC_WAIT_MAX:
                    skipped.append(dict(date=dates[i], act=pending["act"], reason=why))
                    pending = None      # 放弃；条件仍满足则明日自然重挂

        # ---- B. 收盘后更新条件（只看当日及以前，无未来函数）----
        prev_up_st = up_st
        up_st = up_st + 1 if up[i] else 0
        dn_st = dn_st + 1 if dn[i] else 0
        if state == "FLAT" and prev_up_st > 0 and not up[i]:
            false_starts += 1           # 方向苗头消失（"中途改向"记录）
        if state == "LONG" and not up[i] and not dn[i] and cur is not None:
            cur["unclear"] += 1         # 持仓期"看不清"天数
        if pending is None:
            if state == "FLAT" and up_st >= D_CONFIRM:
                pending = dict(act="buy", sig=i, first=i - up_st + 1, age=0)
            elif state == "LONG" and dn_st >= K_EXIT:
                pending = dict(act="sell", sig=i, age=0)

        # ---- C. 收盘估值（三线）----
        if state == "FLAT":
            eq_g[i] = cash_g
            eq_n[i] = cash_n
        else:
            eq_g[i] = cash_g + sh_g * c[i]
            eq_n[i] = cash_n + sh_n * c[i]
        eq_b[i] = sh_b * c[i]

    # 期末若仍持仓：按末日收盘虚拟平仓（只补记录，不产生费用）——原样呈现
    if state == "LONG" and cur is not None:
        cur["exit_i"] = n - 1
        cur["exit_date"] = dates[n - 1]
        cur["exit_px_g"] = cur["exit_px"] = c[n - 1]
        cur["hold_days"] = (n - 1) - cur["entry_i"]
        cur["ret_n"] = (cash_n + sh_n * c[n - 1]) / cur["cash_before_n"] - 1
        cur["ret_g"] = (cash_g + sh_g * c[n - 1]) / cur["cash_before_g"] - 1
        fc = cur["first_cond_i"]
        cur["lag"] = cur["entry_px_g"] / c[fc] - 1 if c[fc] > 0 else np.nan
        lo_i = max(0, fc - N_MOM)
        seg_low = np.nanmin(l[lo_i:cur["entry_i"] + 1])
        cur["chase"] = cur["entry_px_g"] / seg_low - 1 if seg_low > 0 else np.nan
        cur["open_at_end"] = True
        trades.append(cur)

    return dict(
        nav_gross=pd.Series(eq_g, index=dates, name="nav_gross").dropna(),
        nav_net=pd.Series(eq_n, index=dates, name="nav_net").dropna(),
        nav_hold=pd.Series(eq_b, index=dates, name="nav_hold").dropna(),
        trades=trades, skipped=skipped, false_starts=false_starts,
    )


# ==================== 步骤4：错误归因（事后统计，允许用未来数据，不回流信号）====================
def classify_exits(df: pd.DataFrame, trades: list) -> list:
    h = df["high"].values.astype(float)
    c = df["close"].values.astype(float)
    n = len(df)
    for tr in trades:
        j = tr["exit_i"]
        px = tr["exit_px_g"]
        if j + 1 >= n:
            tr["exit_cls"] = "期末持仓"
        else:
            hi = h[j + 1: j + 1 + FWD_DAYS]
            ce = c[j + FWD_DAYS] if j + FWD_DAYS < n else c[n - 1]
            if len(hi) and np.nanmax(hi) >= px * (1 + RECONF_PCT):
                tr["exit_cls"] = "提前结束(暂时反走)"
            elif ce <= px * (1 - RECONF_PCT):
                tr["exit_cls"] = "正确结束"
            else:
                tr["exit_cls"] = "中性"
        tr["late_cls"] = "发现太晚" if (tr["chase"] == tr["chase"] and tr["chase"] > CHASE_PCT) else "及时"
    return trades


# ==================== 跨阶段验证：行情阶段自动标记（仅用历史滚动窗，无未来）====================
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


# ==================== 汇总输出 ====================
def report(code: str, df: pd.DataFrame, res: dict, regimes: pd.Series):
    trades = res["trades"]
    pg, pn, ph = perf(res["nav_gross"]), perf(res["nav_net"]), perf(res["nav_hold"])
    print(f"\n================ 波段操作法·逻辑复刻 {code} ================")
    d0, d1 = str(df.index[0]), str(df.index[-1])
    print(f"区间: {d0} ~ {d1}  交易日 {len(df)}  交易 {len(trades)} 笔  未执行 {len(res['skipped'])} 次")

    print("\n--- 三线对比（步骤5：不跟完美波段比，跟什么都不做比）---")
    print(f"{'':12s}{'总收益':>10s}{'年化':>9s}{'最大回撤':>10s}{'夏普(rf=2.5%)':>13s}")
    for name, m in [("不计成本", pg), ("扣成本", pn), ("一直持有", ph)]:
        print(f"{name:12s}{m['tot']*100:9.1f}%{m['cagr']*100:8.2f}%{m['mdd']*100:9.2f}%{m['sharpe']:13.2f}")

    if trades:
        print("\n--- 波段明细（每笔原样保留，含滞后/追高/归因）---")
        print(f"{'入场日':>9s}{'出场日':>9s}{'天数':>5s}{'毛收益':>8s}{'净收益':>8s}"
              f"{'确认滞后':>8s}{'追高':>7s}{'看不清日':>6s}  出场归因/及时性")
        for tr in trades:
            print(f"{tr['entry_date']:>9s}{tr['exit_date']:>9s}{tr['hold_days']:5d}"
                  f"{tr['ret_g']*100:7.1f}%{tr['ret_n']*100:7.1f}%"
                  f"{tr['lag']*100:7.1f}%{tr['chase']*100:6.1f}%{tr['unclear']:6d}"
                  f"  {tr['exit_cls']}/{tr['late_cls']}")

        cls = pd.Series([t["exit_cls"] for t in trades]).value_counts()
        late = pd.Series([t["late_cls"] for t in trades]).value_counts()
        print("\n--- 错误分类汇总（步骤4：当场归档，不删除）---")
        for k, v in cls.items():
            print(f"  {k}: {v} 笔")
        for k, v in late.items():
            print(f"  {k}: {v} 笔")
        if res["false_starts"]:
            print(f"  假启动(方向苗头消失): {res['false_starts']} 次")
        if res["skipped"]:
            for s in res["skipped"]:
                print(f"  未执行 {s['date']} {s['act']}: {s['reason']}")
        avg_lag = np.nanmean([t["lag"] for t in trades])
        avg_chase = np.nanmean([t["chase"] for t in trades])
        print(f"  平均确认滞后 {avg_lag*100:.1f}%（视频：'识别通常比实际高低点晚一步'的量化）"
              f"  平均追高 {avg_chase*100:.1f}%")

    # 跨阶段验证
    print("\n--- 跨阶段验证（同一套规则 × 三种行情阶段）---")
    rn = res["nav_net"].pct_change()
    rb = res["nav_hold"].pct_change()
    rows = []
    for st in ["上升段", "下降段", "震荡段", "转向段", "混合"]:
        mask = (regimes == st).reindex(rn.index).fillna(False)
        if mask.sum() < 20:
            continue
        a = rn[mask].mean() * 252
        b = rb[mask].mean() * 252
        rows.append(dict(阶段=st, 天数=int(mask.sum()),
                         策略年化=f"{a*100:6.2f}%", 持有年化=f"{b*100:6.2f}%",
                         年化超额=f"{(a-b)*100:+6.2f}pp"))
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))

    # 落盘
    safe = code.replace(".", "_")
    if trades:
        cols = ["entry_date", "exit_date", "hold_days", "ret_g", "ret_n",
                "lag", "chase", "unclear", "exit_cls", "late_cls"]
        pd.DataFrame([{k: t.get(k) for k in cols} for t in trades]).to_csv(
            f"swing_trades_{safe}.csv", index=False)
    eq = pd.concat([res["nav_gross"], res["nav_net"], res["nav_hold"]], axis=1)
    eq.to_csv(f"swing_equity_{safe}.csv")
    print(f"[save] swing_trades_{safe}.csv / swing_equity_{safe}.csv")


# ==================== 数据加载（复权价；不改动任何已有文件）====================
def load_symbol(con, code: str) -> pd.DataFrame:
    q = (f"SELECT trade_date, open, high, low, close FROM daily "
         f"WHERE ts_code='{code}' AND trade_date>='{START_LOOK}'")
    px = pd.read_sql(q, con)
    if len(px) == 0:   # 指数在 index_daily（无复权需求）
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


def run_pipeline(df: pd.DataFrame, code: str):
    """完整管线：条件→回测→归因→阶段→报告（供真实数据与自测共用）。"""
    cond = compute_conditions(df)
    # 预热期截断：三线同起点，无空仓稀释
    warm = max(N_MOM, N_HHLL + HHLL_STEP)
    cond = cond.iloc[warm:]
    res = run_swing_backtest(cond)
    res["trades"] = classify_exits(cond, res["trades"])
    regimes = label_regimes(cond)
    report(code, cond, res, regimes)


# ==================== 自测（无数据库，合成四段行情验证管线）====================
def selftest():
    rng = np.random.default_rng(7)
    n = 1500
    drifts = np.concatenate([
        np.full(300, 0.0015),    # 上升段（方向清楚）
        np.full(300, 0.0),       # 震荡段（来回震荡）
        np.full(300, -0.0018),   # 下降段（方向清楚·跌）
        np.full(300, 0.0),       # 震荡
        np.full(300, 0.002),     # 转向段（V型反转）
    ])
    ret = drifts + rng.normal(0, 0.02, n)
    close = 100 * np.cumprod(1 + ret)
    idx = pd.bdate_range("2015-01-01", periods=n).strftime("%Y%m%d")
    op = close * (1 + rng.normal(0, 0.004, n))
    hi = np.maximum(op, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    lo = np.minimum(op, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    df = pd.DataFrame({"open": op, "high": hi, "low": lo, "close": close}, index=idx)
    print("[selftest] 合成四段行情（上升/震荡/下降/震荡/转向）1500 日")
    run_pipeline(df, "SELFTEST")


def main():
    import config   # 项目全局配置（本地库路径）
    con = sqlite3.connect(config.DATA["local_db_path"])
    for code in SYMBOLS:
        df = load_symbol(con, code)
        if len(df) < 300:
            print(f"[skip] {code} 数据不足")
            continue
        run_pipeline(df, code)
    con.close()


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
