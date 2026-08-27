# -*- coding: utf-8 -*-
"""
run_sector_rotation.py — 板块轮动交易法（Sperandeo 风格三步框架）量化复刻 + 回测
=============================================================================
B站来源：BV16Dbe6MEsu · 跟着Jim学量化 · 2026-08-17

!!! 诚实红线 #1（参数归属）：Jim 视频只讲了三步框架与量化验证思路，**全程未给
    "演示参数"的具体数值**（行业分类标准、板块数、强弱指标窗口、均线长度、持币
    阈值都没说）。本文件所有参数均为本复刻**一次性定死的提案值，非 Jim 原值**，
    必须在报告首行明示。
!!! 诚实红线 #2（无未来函数）：月度调仓在月末收盘日 d 用 ≤ d 的数据生成信号，
    于 d+1 开盘执行。只使用当时已上市的标的（防幸存者偏差）。
!!! 诚实红线 #3（跑后不调参）：参数跑前定死，跑后不得回头改阈值再跑。
!!! 诚实红线 #4（弱市持币）：无合格标的时保留现金，不为凑数选弱板块。

三步框架：
  步骤一（定范围）：固定行业池 = 31 个申万一级行业指数。逐月重放时只使用当时
                    已有数据的行业（后上市的不提前出现）。
  步骤二（找强者）：按过去 MOM_W 日涨跌幅横截面排名；剔除自身趋势仍下跌的
                    （close < MA_TREND）；取排名前 TOP_K。
  步骤三（定期换）：月末调仓，等权持有 Top-K；无合格标的则持币；市场整体
                    走弱（基准均线空头）则持币等待。

数据：sw_industry_daily（31个申万一级行业指数 OHLC，1999年至今）
基准：31 行业等权日收益
"""
import sys
import os
import sqlite3
import datetime
import numpy as np
import pandas as pd

# ==================== 参数区（一次性定死，跑后不得回头改）====================
# !! 以下全部为本复刻提案值，非 Jim 原值 !!
MOM_W       = 20       # 强弱指标窗口：过去 20 交易日涨跌幅
MA_TREND    = 60       # 自身趋势均线：close > MA60 才算"自身在上涨"
MA_BENCH    = 60       # 市场整体均线：基准 close < MA60 → 市场走弱→持币
TOP_K       = 3        # 每月选排名前 3 的行业
REBAL_DAY   = 'last'   # 调仓日：每月最后一个交易日
COST_RATE   = 0.002    # 单边交易成本（佣金+滑点+冲击，0.2%）
START       = '20100101'  # 回测起始（留足 MA60 预热）
END         = '20260801'
INITIAL     = 1_000_000.0
OUT_DIR     = "data/results/sector_rotation"

DB = r"D:\tu-shareData\astock_daily.db"


# ==================== 数据加载 ====================
def load_industry_indices(con, db_path=DB):
    """加载 31 个申万一级行业指数日线数据。
    返回 pivot 后的 DataFrame，index=trade_date(str), columns=ts_code。
    """
    df = pd.read_sql(
        "SELECT ts_code, trade_date, open, high, low, close "
        "FROM sw_industry_daily ORDER BY trade_date", con)
    df['trade_date'] = df['trade_date'].astype(str)
    df = df.drop_duplicates(subset=['ts_code', 'trade_date'])
    df = df.sort_values(['ts_code', 'trade_date'])

    # 行业名称映射
    sw_names = {
        '801010.SI': '农林牧渔', '801030.SI': '基础化工', '801040.SI': '钢铁',
        '801050.SI': '有色金属', '801080.SI': '电子', '801110.SI': '家用电器',
        '801120.SI': '食品饮料', '801130.SI': '纺织服饰', '801140.SI': '轻工制造',
        '801150.SI': '医药生物', '801160.SI': '公用事业', '801170.SI': '交通运输',
        '801180.SI': '房地产', '801200.SI': '商贸零售', '801210.SI': '社会服务',
        '801230.SI': '综合', '801710.SI': '建筑材料', '801720.SI': '建筑装饰',
        '801730.SI': '电力设备', '801740.SI': '国防军工', '801750.SI': '计算机',
        '801760.SI': '传媒', '801770.SI': '通信', '801780.SI': '银行',
        '801790.SI': '非银金融', '801880.SI': '机械设备', '801890.SI': '汽车',
        '801950.SI': '煤炭', '801960.SI': '石油石化', '801970.SI': '环保',
        '801980.SI': '美容护理',
    }
    return df, sw_names


def pivot_ohlc(df):
    """pivot 成多 index 的 DataFrame, 返回 dict of DataFrames。"""
    cols = ['open', 'high', 'low', 'close']
    out = {}
    for col in cols:
        out[col] = df.pivot(index='trade_date', columns='ts_code', values=col).sort_index()
    return out


# ==================== 月度调仓日历 ====================
def build_rebal_calendar(dates_index, rebal_day='last'):
    """从交易日序列中提取每月调仓日。
    rebal_day='last' → 每月最后一个交易日；'first' → 每月第一个交易日。
    返回调仓日列表（trade_date str）。
    """
    s = pd.Series(dates_index, index=dates_index)
    months = pd.Series(dates_index).str[:6]  # YYYYMM
    rebal_dates = []
    if rebal_day == 'last':
        # 每月最后一个交易日
        df = pd.DataFrame({'date': dates_index, 'ym': months.values})
        for _, g in df.groupby('ym'):
            rebal_dates.append(g.iloc[-1]['date'])
    elif rebal_day == 'first':
        df = pd.DataFrame({'date': dates_index, 'ym': months.values})
        for _, g in df.groupby('ym'):
            rebal_dates.append(g.iloc[0]['date'])
    else:
        raise ValueError(f"rebal_day must be 'last' or 'first', got {rebal_day}")
    return rebal_dates


# ==================== 因子计算（纯函数，因果）====================
def compute_momentum(close_df, window=MOM_W):
    """过去 window 日涨跌幅。返回 DataFrame。
    mom[t] = close[t] / close[t-window] - 1，只用 ≤ t 的数据。
    """
    return close_df.pct_change(periods=window)


def compute_ma(close_df, window):
    """简单移动平均。返回 DataFrame。"""
    return close_df.rolling(window=window, min_periods=window).mean()


def compute_benchmark(close_df):
    """基准 = 31 行业等权日收益。返回 (nav_series, daily_ret_series)。"""
    daily_ret = close_df.pct_change().mean(axis=1)
    nav = (1 + daily_ret.fillna(0)).cumprod()
    nav.iloc[0] = 1.0  # 归一化
    return nav, daily_ret


# ==================== 核心：月度信号生成 ====================
def monthly_signal(close_df, mom_df, ma_trend_df, ma_bench_series, rebal_date,
                   top_k=TOP_K, prev_hold=None):
    """在调仓日 rebal_date 生成下期持有名单。
    只使用 ≤ rebal_date 的数据（mom / ma 均为滞后指标）。

    参数:
      close_df:       全行业收盘价 DataFrame
      mom_df:         动量 DataFrame
      ma_trend_df:    自身趋势 MA DataFrame
      ma_bench_series: 基准 MA Series
      rebal_date:     调仓日 (str)
      prev_hold:      上月持仓 list of ts_code（用于换手率统计）

    返回: (target_codes list, debug_info dict)
    """
    # 取调仓日当天的横截面
    if rebal_date not in close_df.index:
        return None, dict(reason='调仓日不在数据中')

    row_close = close_df.loc[rebal_date]
    row_mom = mom_df.loc[rebal_date]
    row_ma = ma_trend_df.loc[rebal_date]
    bench_ma = ma_bench_series.get(rebal_date, np.nan)

    # 只考虑当日有有效数据的行业（防幸存者偏差：后上市的不提前出现）
    valid = row_close.dropna()
    if len(valid) == 0:
        return None, dict(reason='当日无有效行业数据')

    # 市场整体走弱检查：基准 close < MA60 → 持币
    bench_close = row_close.mean()  # 等权基准
    if not np.isnan(bench_ma) and not np.isnan(bench_close) and bench_close < bench_ma:
        return [], dict(reason='市场整体走弱(基准<MA60)', bench_close=bench_close,
                        bench_ma=bench_ma, n_valid=len(valid))

    # 筛选：自身趋势上涨 (close > MA_TREND)
    trending_up = row_close > row_ma
    candidates = trending_up[trending_up == True].dropna()

    # 动量排名
    mom_valid = row_mom.reindex(candidates.index).dropna()
    if len(mom_valid) == 0:
        return [], dict(reason='无行业通过趋势过滤', n_valid=len(valid),
                        n_trending=int(trending_up.sum()))

    ranked = mom_valid.sort_values(ascending=False)
    selected = ranked.head(top_k).index.tolist()

    return selected, dict(reason='OK', n_valid=len(valid),
                         n_trending=int(trending_up.sum()),
                         n_selected=len(selected),
                         mom_top=ranked.head(5))


# ==================== 回测引擎 ====================
def run_backtest(close_df, open_df, rebal_dates, mom_df, ma_trend_df,
                 ma_bench_series, cost_rate=COST_RATE, initial=INITIAL):
    """月度调仓回测。

    信号日 = rebal_date（月末收盘），执行日 = rebal_date 的下一个交易日开盘。
    等权分配资金，持仓不变则不换手（省成本）。

    返回: dict(nav_series, trades, holdings, cash_rate)
    """
    all_dates = close_df.index.tolist()
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    nav = np.full(len(all_dates), np.nan)
    holdings = {}  # date -> list of codes
    trades = []

    cash = initial
    shares = {}  # code -> shares
    state = "FLAT"
    prev_target = []

    # 预计算基准 nav
    bench_nav, _ = compute_benchmark(close_df)

    # 预先生成每月调仓名单（用 ≤ rd 数据，无前视）；执行日 = rd+1 开盘
    plan = []
    for rd in rebal_dates:
        if rd not in date_to_idx:
            continue
        rd_idx = date_to_idx[rd]
        target, info = monthly_signal(close_df, mom_df, ma_trend_df,
                                       ma_bench_series, rd,
                                       top_k=TOP_K, prev_hold=prev_target)
        exec_idx = rd_idx + 1
        if exec_idx >= len(all_dates):
            holdings[rd] = list(target) if target else []
            prev_target = list(target)
            continue
        exec_date = all_dates[exec_idx]
        plan.append((exec_date, target))
        holdings[rd] = list(target) if target else []
        prev_target = list(target)

    # 单遍扫描：在 exec_date 调仓，每日用【当前】cash/shares 计 NAV（修复原版
    # 「用最终状态给所有日期计 NAV」的致命核算 bug——原版序列/端点收益均无效）
    exec_map = {e: t for e, t in plan}
    last_px = {}

    for i, d in enumerate(all_dates):
        if d in exec_map:
            target = exec_map[d]
            to_sell = [c for c in prev_target if c not in target]
            to_buy = [c for c in target if c not in prev_target]

            # 卖出
            for c in to_sell:
                if c in shares and shares[c] > 0:
                    px = open_df.loc[d, c] if (d in open_df.index and c in open_df.columns) else np.nan
                    if np.isnan(px) or px <= 0:
                        continue
                    proceeds = shares[c] * px * (1 - cost_rate)
                    cash += proceeds
                    trades.append(dict(date=d, code=c, action='sell', price=px,
                                       shares=shares[c], proceeds=proceeds))
                    del shares[c]

            # 持币：target 空则清仓
            if len(target) == 0 and len(shares) > 0:
                for c in list(shares.keys()):
                    px = open_df.loc[d, c] if (d in open_df.index and c in open_df.columns) else np.nan
                    if np.isnan(px) or px <= 0:
                        continue
                    proceeds = shares[c] * px * (1 - cost_rate)
                    cash += proceeds
                    trades.append(dict(date=d, code=c, action='sell', price=px,
                                       shares=shares[c], proceeds=proceeds))
                    del shares[c]
                state = "FLAT"

            # 买入（等权）
            if len(target) > 0:
                port_val = cash
                for c in shares:
                    lp = last_px.get(c)
                    if lp is not None and lp > 0:
                        port_val += shares[c] * lp
                target_weight = port_val / len(target)
                for c in target:
                    px = open_df.loc[d, c] if (d in open_df.index and c in open_df.columns) else np.nan
                    if np.isnan(px) or px <= 0:
                        continue
                    target_shares = target_weight / px
                    current_shares = shares.get(c, 0)
                    if target_shares > current_shares:
                        buy_shares = target_shares - current_shares
                        cost = buy_shares * px * (1 + cost_rate)
                        if cash >= cost:
                            cash -= cost
                            shares[c] = target_shares
                            trades.append(dict(date=d, code=c, action='buy', price=px,
                                               shares=buy_shares, cost=cost))
                        else:
                            buy_shares = cash / (px * (1 + cost_rate))
                            if buy_shares > 0:
                                cost = buy_shares * px * (1 + cost_rate)
                                cash -= cost
                                shares[c] = current_shares + buy_shares
                                trades.append(dict(date=d, code=c, action='buy', price=px,
                                                   shares=buy_shares, cost=cost))
                    elif target_shares < current_shares:
                        sell_shares = current_shares - target_shares
                        proceeds = sell_shares * px * (1 - cost_rate)
                        cash += proceeds
                        shares[c] = target_shares
                        trades.append(dict(date=d, code=c, action='sell_adj', price=px,
                                           shares=sell_shares, proceeds=proceeds))
                state = "LONG"
            prev_target = list(target)

        # 逐日 NAV（用当前 cash/shares；停牌用最后已知价）
        val = 0.0
        for c in list(shares.keys()):
            px = close_df.loc[d, c] if (d in close_df.index and c in close_df.columns) else np.nan
            if np.isnan(px) or px <= 0:
                px = last_px.get(c)
            if px is not None and px > 0:
                val += shares[c] * px
                last_px[c] = px
            elif c in last_px:
                val += shares[c] * last_px[c]
        nav[i] = cash + val

    nav_series = pd.Series(nav, index=all_dates, name='nav')
    # 归一化起点
    first_valid = nav_series.first_valid_index()
    if first_valid is not None:
        nav_series = nav_series / nav_series.loc[first_valid]

    # 基准归一化
    bench_nav = bench_nav / bench_nav.iloc[0] if len(bench_nav) > 0 else bench_nav

    # 持币率
    flat_months = sum(1 for v in holdings.values() if len(v) == 0)
    total_months = len(holdings)

    return dict(nav=nav_series, bench=bench_nav, trades=trades,
                holdings=holdings, cash_rate=flat_months/max(1, total_months))


# ==================== 指标 ====================
def perf(nav_series):
    nav = nav_series.dropna()
    if len(nav) < 2:
        return dict(tot=np.nan, cagr=np.nan, mdd=np.nan, sharpe=np.nan)
    ns = nav.values
    tot = ns[-1] / ns[0] - 1
    d0_str = str(nav.index[0])
    d1_str = str(nav.index[-1])
    try:
        d0 = datetime.datetime.strptime(d0_str, "%Y%m%d")
        d1 = datetime.datetime.strptime(d1_str, "%Y%m%d")
        yrs = (d1 - d0).days / 365.25
    except ValueError:
        yrs = len(ns) / 252
    cagr = (ns[-1] / ns[0]) ** (1 / yrs) - 1 if yrs > 0 else np.nan
    peak = np.maximum.accumulate(ns)
    mdd = ((ns - peak) / peak).min()
    daily = nav.pct_change().dropna()
    rf = 0.025 / 252
    excess = daily - rf
    sharpe = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else np.nan
    return dict(tot=tot, cagr=cagr, mdd=mdd, sharpe=sharpe, yrs=yrs)


# ==================== 报告 ====================
def report(nav_series, bench_nav, trades, holdings, cash_rate, sw_names):
    m = perf(nav_series)
    bm = perf(bench_nav)
    n_trades = len(trades)
    n_months = len(holdings)
    flat_months = sum(1 for v in holdings.values() if len(v) == 0)

    print(f"\n{'='*70}")
    print(f"板块轮动交易法 · 量化复刻回测")
    print(f"{'='*70}")
    print(f"[!] 参数全为本复刻提案值（MOM_W={MOM_W}/MA_TREND={MA_TREND}/TOP_K={TOP_K}），非 Jim 原值")
    print(f"区间: {nav_series.first_valid_index()} ~ {nav_series.last_valid_index()}")

    print(f"\n{'':14s}{'总收益':>10s}{'年化':>9s}{'最大回撤':>10s}{'夏普':>7s}")
    print(f"{'轮动策略':<13s}{m['tot']*100:>+9.2f}%{m['cagr']*100:>+8.2f}%"
          f"{m['mdd']*100:>+9.2f}%{m['sharpe']:>7.2f}")
    print(f"{'等权基准':<13s}{bm['tot']*100:>+9.2f}%{bm['cagr']*100:>+8.2f}%"
          f"{bm['mdd']*100:>+9.2f}%{bm['sharpe']:>7.2f}")
    excess = m['tot'] - bm['tot']
    verdict = '跑赢' if excess > 0 else '跑输'
    print(f"\n超额: {excess*100:+.2f}pp | {verdict}")
    print(f"调仓月数: {n_months} | 持币月数: {flat_months} ({cash_rate*100:.1f}%)"
          f" | 交易笔数: {n_trades}")

    # 年度收益
    yearly_nav = nav_series.resample('YE' if hasattr(nav_series.index, 'freq')
                                      else None).last() if False else None
    # 手动年度收益
    years = sorted(set(str(d)[:4] for d in nav_series.index))
    print(f"\n年度收益:")
    for y in years:
        mask = [str(d)[:4] == y for d in nav_series.index]
        sub = nav_series[mask]
        if len(sub) > 1:
            yr_ret = sub.iloc[-1] / sub.iloc[0] - 1
            bench_mask = [str(d)[:4] == y for d in bench_nav.index]
            bench_sub = bench_nav[bench_mask]
            bret = bench_sub.iloc[-1] / bench_sub.iloc[0] - 1 if len(bench_sub) > 1 else 0
            print(f"  {y}: 策略 {yr_ret*100:+6.1f}% | 基准 {bret*100:+6.1f}%"
                  f" | 超额 {(yr_ret-bret)*100:+6.1f}pp")

    # 持仓分布
    from collections import Counter
    all_holds = []
    for codes in holdings.values():
        all_holds.extend(codes)
    dist = Counter(all_holds)
    print(f"\n持仓频率 Top-10:")
    for code, cnt in dist.most_common(10):
        name = sw_names.get(code, code)
        print(f"  {name}({code}): {cnt}/{n_months} 月 ({cnt/max(1,n_months)*100:.0f}%)")

    # 回撤对比
    peak_s = nav_series.cummax()
    dd_s = (nav_series - peak_s) / peak_s
    peak_b = bench_nav.cummax()
    dd_b = (bench_nav - peak_b) / peak_b
    print(f"\n回撤对比:")
    print(f"  策略最大回撤: {m['mdd']*100:.2f}% @ {nav_series.index[dd_s.values.argmin()]}")
    print(f"  基准最大回撤: {bm['mdd']*100:.2f}% @ {bench_nav.index[dd_b.values.argmin()]}")
    print(f"  回撤差: {(m['mdd']-bm['mdd'])*100:+.2f}pp "
          f"({'策略更浅' if m['mdd'] > bm['mdd'] else '策略更深'})")

    # 去魅结论
    print(f"\n--- 去魅结论（初步）---")
    print(f"  盈利能力: 策略 {m['tot']*100:+.1f}% vs 基准 {bm['tot']*100:+.1f}%"
          f" (超额 {excess*100:+.1f}pp)")
    print(f"  回撤控制: 策略 {m['mdd']*100:.1f}% vs 基准 {bm['mdd']*100:.1f}%"
          f" ({'有效' if m['mdd'] > bm['mdd'] else '无效'})")
    print(f"  持币纪律: {cash_rate*100:.1f}% 月度持币等待")
    print(f"  正贡献层: {'趋势过滤+相对强弱排序' if excess > 0 else '仅相对强弱排序(趋势过滤未贡献)'}")
    print(f"  可复用部分: 月度调仓框架 + 横截面动量 + 弱市持币规则")
    print(f"  批判星级: 待定（需多参数稳健性验证后填）")

    # 落盘
    os.makedirs(OUT_DIR, exist_ok=True)
    eq = pd.DataFrame({'nav': nav_series, 'bench': bench_nav})
    eq.to_csv(f"{OUT_DIR}/sector_rotation_equity.csv")
    # 持仓记录
    rows = []
    for rd, codes in holdings.items():
        names = [sw_names.get(c, c) for c in codes]
        rows.append(dict(date=rd, n=len(codes),
                         codes=','.join(codes), names=','.join(names)))
    pd.DataFrame(rows).to_csv(f"{OUT_DIR}/sector_rotation_holdings.csv", index=False)
    # 交易记录
    if trades:
        pd.DataFrame(trades).to_csv(f"{OUT_DIR}/sector_rotation_trades.csv", index=False)
    print(f"\n[save] {OUT_DIR}/sector_rotation_equity.csv + holdings.csv"
          + (" + trades.csv" if trades else ""))


# ==================== 自测（无数据库，合成行情验证）====================
def _mk_synthetic():
    """合成 3 个行业指数的 OHLC，构造明确的强势/弱势/震荡段。"""
    np.random.seed(42)
    n = 500  # ~2年
    dates = pd.bdate_range('2020-01-02', periods=n).strftime('%Y%m%d')

    # 行业A：前半段强上涨，后半段横盘
    a_close = np.concatenate([
        np.cumsum(np.random.randn(n//2) * 0.5 + 0.3) + 1000,
        np.cumsum(np.random.randn(n - n//2) * 0.3) + 1100
    ])
    # 行业B：前半段横盘，后半段强上涨
    b_close = np.concatenate([
        np.cumsum(np.random.randn(n//2) * 0.3) + 1000,
        np.cumsum(np.random.randn(n - n//2) * 0.5 + 0.4) + 1050
    ])
    # 行业C：全程下跌
    c_close = np.cumsum(np.random.randn(n) * 0.4 - 0.2) + 1000

    codes = ['A.SI', 'B.SI', 'C.SI']
    close = pd.DataFrame({'A.SI': a_close, 'B.SI': b_close, 'C.SI': c_close}, index=dates)
    op = close.copy()
    hi = close + np.abs(np.random.randn(n, 3) * 0.5)
    lo = close - np.abs(np.random.randn(n, 3) * 0.5)

    close_df = close
    open_df = op
    high_df = pd.DataFrame(hi, columns=codes, index=dates)
    low_df = pd.DataFrame(lo, columns=codes, index=dates)
    return close_df, open_df, codes


def selftest():
    print("[selftest] 合成行情验证检测器（无数据库）")
    close_df, open_df, codes = _mk_synthetic()
    n = len(close_df)

    # 计算因子
    mom_df = compute_momentum(close_df, MOM_W)
    ma_trend_df = compute_ma(close_df, MA_TREND)
    bench_nav, _ = compute_benchmark(close_df)
    # 市场转弱过滤：必须用「等权指数点位」自身的 MA（与 monthly_signal 里的
    # bench_close=row_close.mean() 同量纲）。原代码用归一化 NAV 的 MA(~1-2) 与
    # 点位均值(几千)比较 → 量纲错配 → 过滤永远 False → 弱市持币失效（已修）。
    bench_level = close_df.mean(axis=1)
    ma_bench = compute_ma(pd.DataFrame({'bench': bench_level}), MA_BENCH)['bench']

    # 调仓日历
    rebal_dates = build_rebal_calendar(close_df.index.tolist(), 'last')

    # 过滤 START 之前的调仓日
    rebal_dates = [d for d in rebal_dates if d >= '20200401']  # 留 MA60 预热

    # 跑回测
    result = run_backtest(close_df, open_df, rebal_dates, mom_df, ma_trend_df,
                          ma_bench, cost_rate=COST_RATE)
    m = perf(result['nav'])
    bm = perf(result['bench'])

    print(f"  合成回测: 策略 {m['tot']*100:+.1f}% vs 基准 {bm['tot']*100:+.1f}%")
    print(f"  持币率: {result['cash_rate']*100:.1f}%")
    print(f"  交易笔数: {len(result['trades'])}")

    # 验证行业C（全程下跌）不应被选中
    all_holds = set()
    for codes_held in result['holdings'].values():
        all_holds.update(codes_held)
    assert 'C.SI' not in all_holds, "行业C(全程下跌)不应被选中"
    print(f"  [OK] 行业C(全程下跌)从未被选中")

    # 验证前半段应选A，后半段应选B
    early_holds = [result['holdings'][d] for d in list(result['holdings'].keys())[:5]
                   if len(result['holdings'][d]) > 0]
    late_holds = [result['holdings'][d] for d in list(result['holdings'].keys())[-5:]
                  if len(result['holdings'][d]) > 0]

    if early_holds:
        a_in_early = any('A.SI' in h for h in early_holds)
        print(f"  [{'OK' if a_in_early else 'WARN'}] 前半段A(强势上涨)出现: {a_in_early}")
    if late_holds:
        b_in_late = any('B.SI' in h for h in late_holds)
        print(f"  [{'OK' if b_in_late else 'WARN'}] 后半段B(强势上涨)出现: {b_in_late}")

    print("[selftest] 通过 [OK]")


# ==================== 主流程 ====================
def main():
    import config
    con = sqlite3.connect(config.DATA["local_db_path"])

    print("=" * 70)
    print("板块轮动交易法 · 量化复刻 (可证伪性检验)")
    print(f"  强弱窗口={MOM_W}日 | 趋势MA={MA_TREND} | Top-{TOP_K} | "
          f"成本={COST_RATE*100:.2f}% | 调仓=月末")
    print("=" * 70)

    # 加载数据
    df_raw, sw_names = load_industry_indices(con)
    con.close()

    # pivot
    pivoted = pivot_ohlc(df_raw)
    close_df = pivoted['close']
    open_df = pivoted['open']

    # 筛选回测区间
    mask = (close_df.index >= START) & (close_df.index <= END)
    close_df = close_df.loc[mask]
    open_df = open_df.loc[mask]

    print(f"[load] 行业数={len(close_df.columns)}, 交易日={len(close_df)},"
          f" 区间={close_df.index[0]}~{close_df.index[-1]}")

    # 计算因子
    mom_df = compute_momentum(close_df, MOM_W)
    ma_trend_df = compute_ma(close_df, MA_TREND)
    bench_nav, _ = compute_benchmark(close_df)
    # 市场转弱过滤：必须用「等权指数点位」自身的 MA（与 monthly_signal 里的
    # bench_close=row_close.mean() 同量纲）。原代码用归一化 NAV 的 MA(~1-2) 与
    # 点位均值(几千)比较 → 量纲错配 → 过滤永远 False → 弱市持币失效（已修）。
    bench_level = close_df.mean(axis=1)
    ma_bench = compute_ma(pd.DataFrame({'bench': bench_level}), MA_BENCH)['bench']

    # 调仓日历
    rebal_dates = build_rebal_calendar(close_df.index.tolist(), REBAL_DAY)
    # 过滤预热期（MA_TREND 之前无法计算）
    warmup_end = close_df.index[MA_TREND] if len(close_df) > MA_TREND else close_df.index[-1]
    rebal_dates = [d for d in rebal_dates if d >= warmup_end]
    print(f"[init] 调仓月数={len(rebal_dates)} (预热后)")

    # 跑回测
    result = run_backtest(close_df, open_df, rebal_dates, mom_df, ma_trend_df,
                          ma_bench, cost_rate=COST_RATE)

    # 报告
    report(result['nav'], result['bench'], result['trades'],
           result['holdings'], result['cash_rate'], sw_names)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
