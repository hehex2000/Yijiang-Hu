#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单因子回测平台（动态加载版）
================================================================================
一键测试任意量化因子：
    factors/ 目录里每放一个因子模块，平台就自动发现并回测它，
    最后产出"所有因子横向对比"报告，直接看出哪个因子真的有效。

因子模块约定（factors/<name>.py）：
    NAME       : str  唯一 ID（也是输出前缀）
    DIRECTION  : str  'asc' = 因子最小的一档做多头（如反转/低波/APB）
                      'desc'= 因子最大的一档做多头（如动量）
    compute(df, **kw) -> DataFrame   在原 df 上新增一列 NAME（逐笔因子值）
        df 已含列：ts_code, trade_date, open, high, low, close,
                    vol, amount, pct_chg, adj_factor, *_adj(前复权价)

回测口径（2026-07-08 统一对齐）：
    - 前复权（adj_factor，最新因子归一）
    - 成本：佣金万2.5（最低5元）+ 印花税千1→千0.5(2023-08-28起,仅卖)+ 滑点千1（买卖）
    - 幸存者偏差：股票池取自行情库实际交易代码（含已退市），退市股无法
      卖出按归零（全额亏损）计入，不再静默剔除美化收益
    - 强制 1 手超分配 bug 已修（资金不足 1 手则跳过该股）
    - 涨跌停 / ST / 北交所过滤
    - 流动性过滤（可配置，默认 20 日日均成交额 ≥ 5000 万）

APB5D 已证实在 A 股日线数据上无效（RankIC≈0），作为示例因子保留，
用于证明"框架没问题、是因子本身无效"。新增因子只需在 factors/ 放一个文件。

输入数据库：D:\tu-shareData\astock_daily.db（平台唯一主库）
输出目录：D:\tu-shareData\factor_backtests\
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # 无界面环境也能出图
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import argparse
import importlib.util
import traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 印花税率复用共享引擎的「分段口径」（2023-08-28 起千1→千0.5）
from run_monthly_rebalance import stamp_duty_rate

# 解决中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# =========================================================================== #
# 参数配置（可被命令行覆盖）
# =========================================================================== #

DB_PATH = r"D:\tu-shareData\astock_daily.db"

# 回测参数
TOP_N = 10
START_DATE = "2017-12-29"
END_DATE = "2022-01-07"
BENCHMARK_INDEX = "000300.SH"
INITIAL_CASH = 100000           # 初始资金（元，10万）

# 交易成本参数（与 run_backtest 统一口径：佣金+印花+滑点）
COMMISSION_RATE = 0.00025       # 佣金：万分之2.5
MIN_COMMISSION = 5.0            # 单笔最低佣金：5元
STAMP_DUTY_RATE = 0.001         # 历史常量（已改用 stamp_duty_rate 分段：2023-08-28 起千1→千0.5，仅卖出）
SLIPPAGE_RATE = 0.001           # 滑点：千分之一（买卖双向，模拟冲击成本）

# 流动性过滤（单位：元；<=0 关闭）
LIQUIDITY_MIN_AVG_AMOUNT = 50_000_000   # 20日日均成交额 ≥ 5000万
LIQUIDITY_LOOKBACK = 20

# 因子计算所需历史回看天数（动量需 ~12月，留足余量）
HISTORY_LOOKBACK_DAYS = 600

# IC 判定阈值（方向修正后的 |IC| 超过此值视为"有信号"）
IC_GOOD_THRESHOLD = 0.02

# 输出目录
OUT_DIR = Path(r"D:\tu-shareData\factor_backtests")
FACTORS_DIR = Path(__file__).parent / "factors"

# 日志频率
PRINT_EVERY = 10000

# WINDOW 缺省（部分因子用）
ROLLING_WINDOW = 5


# =========================================================================== #
# 模块化函数：数据加载
# =========================================================================== #

def load_raw_data(start_dt, end_dt):
    """加载日线 OHLCV + amount + 复权因子，并做前复权"""
    print(f"[1/N] 加载日线数据 {start_dt.date()} ~ {end_dt.date()}...")

    start_actual = (start_dt - pd.Timedelta(days=HISTORY_LOOKBACK_DAYS)).strftime("%Y%m%d")
    end_actual = (end_dt + pd.Timedelta(days=10)).strftime("%Y%m%d")

    query = """
    SELECT d.ts_code, d.trade_date, d.open, d.high, d.low, d.close,
           d.vol, d.amount, d.pct_chg, a.adj_factor
    FROM daily d
    LEFT JOIN adj_factor a ON d.ts_code = a.ts_code AND d.trade_date = a.trade_date
    WHERE d.trade_date >= ? AND d.trade_date <= ?
    ORDER BY d.trade_date, d.ts_code
    """

    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(query, conn, params=(start_actual, end_actual))

    df['trade_date'] = pd.to_datetime(df['trade_date'], format="%Y%m%d")
    df = df[(df['vol'] > 0) & (df['amount'] > 0) & (df['adj_factor'].notnull())].copy()

    # 前复权：以最后一天的复权因子为基准
    print("  应用前复权...")
    latest_adj = df.groupby('ts_code')['adj_factor'].transform('last')
    ratio = df['adj_factor'] / latest_adj
    df['open_adj'] = df['open'] * ratio
    df['high_adj'] = df['high'] * ratio
    df['low_adj'] = df['low'] * ratio
    df['close_adj'] = df['close'] * ratio

    print(f"  日线数据：{len(df):,} 行 ({df['ts_code'].nunique()} 只股票)")
    return df


def load_benchmark_data(start_dt, end_dt):
    """加载基准指数（沪深300）"""
    query = """
    SELECT trade_date, close
    FROM index_daily
    WHERE ts_code = ? AND trade_date >= ? AND trade_date <= ?
    ORDER BY trade_date
    """
    start_f = start_dt.strftime("%Y%m%d")
    end_f = end_dt.strftime("%Y%m%d")
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(query, conn, params=(BENCHMARK_INDEX, start_f, end_f))
    df['trade_date'] = pd.to_datetime(df['trade_date'], format="%Y%m%d")
    return df


def build_universe(raw_data, as_of_dt):
    """
    构建股票池（修复幸存者偏差）：

    关键修复 —— 不再用 stock_basic（仅含当前存活股），而是直接取自行情库
    实际有交易的代码集合，因此天然包含区间内已退市的股票。退市股若被选中
    后无法卖出，会在回测中按归零（全额亏损）计入，避免"只选活着的"虚高收益。

    过滤：ST / *ST（精确匹配，不再误删含字母 S 的正常股）、北交所、流动性。
    """
    print("[2/N] 构建股票池（取自行情库实际交易代码，含退市股）...")
    codes = sorted(raw_data['ts_code'].unique())
    df = pd.DataFrame({'ts_code': codes})

    # 名称（用于 ST 过滤；退市股若不在 stock_basic 则无名称，跳过 ST 过滤）
    with sqlite3.connect(DB_PATH) as conn:
        sb = pd.read_sql_query("SELECT ts_code, name FROM stock_basic", conn)
    nm = dict(zip(sb['ts_code'], sb['name']))
    df['name'] = df['ts_code'].map(nm).fillna('')
    df = df[~df['name'].str.contains('ST', case=False, na=False)]   # 修正 bug3：精确 ST
    df = df[~df['ts_code'].str.endswith('.BJ')]
    df = df[~df['ts_code'].str.startswith('688')]   # 屏蔽科创板(688)：投资门槛对散户不友好

    # 流动性过滤（基于区间起始日的滚动成交额，单位千元需 ×1000）
    if LIQUIDITY_MIN_AVG_AMOUNT > 0:
        end_str = as_of_dt.strftime("%Y%m%d")
        start_str = (as_of_dt - pd.Timedelta(days=int(LIQUIDITY_LOOKBACK * 1.5))).strftime("%Y%m%d")
        window = raw_data[(raw_data['trade_date'] >= start_str) & (raw_data['trade_date'] <= end_str)]
        avg_amt = window.groupby('ts_code')['amount'].mean() * 1000  # 千元 -> 元
        keep = set(avg_amt[avg_amt >= LIQUIDITY_MIN_AVG_AMOUNT].index)
        before = len(df)
        df = df[df['ts_code'].isin(keep)]
        print(f"  [流动性] 日均成交额≥{LIQUIDITY_MIN_AVG_AMOUNT/1e8:.2f}亿 "
              f"过滤：保留 {len(df)} 只 / 剔除 {before - len(df)} 只")

    print(f"  股票池：{len(df):,} 只（含区间内退市股，退市按归零计入）")
    return df[['ts_code']].reset_index(drop=True)


# =========================================================================== #
# 因子动态发现
# =========================================================================== #

def discover_factors(factors_dir: Path):
    """
    扫描 factors/*.py，收集所有暴露 NAME / DIRECTION / compute 的因子模块。
    返回 [(mod, meta), ...]，meta = dict(name, direction, path)
    """
    factors_dir = Path(factors_dir)
    found = []
    if not factors_dir.exists():
        print(f"  [WARN] 因子目录不存在: {factors_dir}")
        return found

    for py_file in sorted(factors_dir.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        try:
            spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if not (hasattr(mod, "NAME") and hasattr(mod, "compute") and hasattr(mod, "DIRECTION")):
                continue
            name = str(mod.NAME)
            direction = str(mod.DIRECTION).lower()
            if direction not in ("asc", "desc"):
                print(f"  [WARN] 因子 {name} DIRECTION 非法({direction})，跳过")
                continue
            found.append((mod, {"name": name, "direction": direction, "path": str(py_file)}))
            print(f"  ✓ 发现因子: {name} (方向={direction}, 文件={py_file.name})")
        except Exception as e:
            print(f"  [WARN] 加载因子 {py_file.name} 失败: {e}")

    if not found:
        print("  [ERR] 未发现任何因子！请在 factors/ 放置因子模块。")
    return found


# =========================================================================== #
# 模块化函数：因子聚合为月度
# =========================================================================== #

def group_to_monthly(df_factor, factor_col, start_dt, end_dt):
    """按月聚合：取每月最后交易日的因子值"""
    df = df_factor.copy()
    df = df[(df['trade_date'] >= start_dt) & (df['trade_date'] <= end_dt)]
    df['year'] = df['trade_date'].dt.year
    df['month'] = df['trade_date'].dt.month
    idx = df.groupby(['ts_code', 'year', 'month'])['trade_date'].transform(max) == df['trade_date']
    monthly = df[idx][['ts_code', 'year', 'month', factor_col]].copy()
    monthly.reset_index(drop=True, inplace=True)
    return monthly


# =========================================================================== #
# 模块化函数：回测逻辑（因子无关，含全部修复）
# =========================================================================== #

def monthly_loop(monthly_factor, factor_col, direction, raw_data, stocks_universe, start_dt, end_dt):
    """月循环：选股 -> 收益 -> RankIC。direction 决定多头取因子哪一端。"""
    print(f"  月度回测循环（因子={factor_col}, 多头方向={direction}）...")

    raw = raw_data.copy()
    raw['year'] = raw['trade_date'].dt.year
    raw['month'] = raw['trade_date'].dt.month

    monthly_first = raw.groupby(['ts_code', 'year', 'month'])['close_adj'].first().reset_index()
    monthly_first.rename(columns={'close_adj': 'close_first'}, inplace=True)
    monthly_last = raw.groupby(['ts_code', 'year', 'month'])['close_adj'].last().reset_index()
    monthly_last.rename(columns={'close_adj': 'close_last'}, inplace=True)
    monthly_pct = raw.groupby(['ts_code', 'year', 'month'])['pct_chg'].last().reset_index()

    monthly_ret = monthly_first.merge(monthly_last, on=['ts_code', 'year', 'month'])
    monthly_ret = monthly_ret.merge(monthly_pct, on=['ts_code', 'year', 'month'])
    monthly_ret['ret'] = (monthly_ret['close_last'] - monthly_ret['close_first']) / monthly_ret['close_first']
    monthly_ret.loc[monthly_ret['ret'].abs() >= 0.5, 'ret'] = np.nan   # 剔除停牌复牌异常

    ret_lookup = monthly_ret.set_index(['ts_code', 'year', 'month'])['ret']
    close_first_lookup = monthly_ret.set_index(['ts_code', 'year', 'month'])['close_first']
    close_last_lookup = monthly_ret.set_index(['ts_code', 'year', 'month'])['close_last']

    # 月份序列（从 start_dt 所在月末开始）
    months = []
    cur = start_dt + pd.offsets.MonthEnd(0)
    while cur <= end_dt:
        months.append({'year': cur.year, 'month': cur.month, 'last_day': cur})
        cur = cur + pd.offsets.MonthEnd(1)
    print(f"  回测月份: {len(months)} 个")

    benchmark = load_benchmark_data(start_dt, end_dt).set_index('trade_date')['close']

    capital_per_stock = INITIAL_CASH / TOP_N
    print(f"  成本: 每股资金={capital_per_stock:,.0f}元, 佣金万2.5(最低5元), "
          f"印花千1→千0.5(2023-08-28起,仅卖), 滑点千1(买卖)")

    records, long_stocks, short_stocks = [], [], []
    valid_codes = set(stocks_universe['ts_code'])

    for i, m in enumerate(months):
        y, mon = m['year'], m['month']
        next_month = mon + 1 if mon < 12 else 1
        next_year = y if mon < 12 else y + 1

        factors = monthly_factor[(monthly_factor['year'] == y) & (monthly_factor['month'] == mon)].copy()
        factors = factors[factors['ts_code'].isin(valid_codes)]
        factors = factors[factors[factor_col].notnull()]

        if len(factors) == 0:
            records.append(_empty_record(y, mon, "无因子数据"))
            continue

        # 涨跌停过滤（用当月实际最后交易日 pct_chg）
        month_mask = (raw_data['trade_date'].dt.year == y) & (raw_data['trade_date'].dt.month == mon)
        actual_last_day = raw_data.loc[month_mask, 'trade_date'].max()
        if pd.isna(actual_last_day):
            records.append(_empty_record(y, mon, "无行情"))
            continue
        last_data = raw_data[raw_data['trade_date'] == actual_last_day][['ts_code', 'pct_chg']].copy()
        factors = factors.merge(last_data, on='ts_code', how='left')
        factors = factors[(factors['pct_chg'].abs() < 9.8) & (factors['pct_chg'].notnull())]

        if len(factors) < TOP_N * 2:
            records.append(_empty_record(y, mon, f"因子值不足{TOP_N*2}", n=len(factors)))
            continue

        # 排序选股（direction 决定多头端）
        factors = factors.sort_values(factor_col, ascending=(direction == 'asc'))
        long_list = factors.head(TOP_N)['ts_code'].tolist()
        short_list = factors.tail(TOP_N)['ts_code'].tolist()

        long_ret, n_long, long_cost = _simulate_portfolio(
            long_list, INITIAL_CASH, next_year, next_month,
            close_first_lookup, close_last_lookup)
        short_ret, n_short, short_cost = _simulate_portfolio(
            short_list, INITIAL_CASH, next_year, next_month,
            close_first_lookup, close_last_lookup)

        ls_ret = long_ret - short_ret if pd.notnull(long_ret) and pd.notnull(short_ret) else np.nan

        # 基准（下月收益）
        bench_ret = np.nan
        b_start = pd.to_datetime(f"{next_year}-{next_month:02d}-01")
        b_end = min(b_start + pd.offsets.MonthEnd(0), end_dt)
        bench_range = benchmark[(benchmark.index >= b_start) & (benchmark.index <= b_end)]
        if len(bench_range) >= 2:
            bench_ret = (bench_range.iloc[-1] - bench_range.iloc[0]) / bench_range.iloc[0]

        # RankIC（全部候选股当月因子 vs 下月收益）
        ic = np.nan
        next_rets = monthly_ret[(monthly_ret['year'] == next_year) & (monthly_ret['month'] == next_month)]
        ic_data = factors[[factor_col, 'ts_code']].merge(
            next_rets[['ts_code', 'ret']], on='ts_code', how='inner').dropna()
        if len(ic_data) > 10:
            corr = ic_data[factor_col].corr(ic_data['ret'], method='spearman')
            if pd.notnull(corr):
                ic = corr
        # 方向修正后的 IC（多头端正确时应为正）
        ic_signed = ic if direction == 'desc' else (-ic if pd.notnull(ic) else np.nan)

        status = f" | 多头 {long_ret:.4f}" if pd.notna(long_ret) else " | 多头 N/A"
        status += f" | 多空 {ls_ret:.4f}" if pd.notna(ls_ret) else " | 多空 N/A"
        status += f" | IC {ic:.4f}(方向修正 {ic_signed:+.4f})" if pd.notna(ic) else " | IC N/A"
        status += f" | 成本 {long_cost:.4%}/{short_cost:.4%}" if pd.notna(long_ret) else ""
        status += f" | 持仓 {n_long}/{n_short}"
        print(f"  [{i+1:2d}/{len(months)}] {y}-{mon:02d} | 选 {TOP_N} 只{status}")

        records.append({
            'year': y, 'month': mon,
            'long_ret': long_ret, 'short_ret': short_ret,
            'longshort_ret': ls_ret, 'bench_ret': bench_ret,
            'ic': ic, 'ic_signed': ic_signed, 'N_factor': len(factors),
            'N_long': n_long, 'N_short': n_short,
            'long_cost': long_cost, 'short_cost': short_cost,
        })
        long_stocks.append({'year': y, 'month': mon, 'ts_code': ','.join(long_list[:20])})
        short_stocks.append({'year': y, 'month': mon, 'ts_code': ','.join(short_list[:20])})

    records_df = pd.DataFrame(records)
    print(f"  回测完成: {len(records_df)} 个月")
    return records_df, pd.DataFrame(long_stocks), pd.DataFrame(short_stocks)


def _empty_record(y, mon, reason, n=0):
    return {'year': y, 'month': mon, 'long_ret': np.nan, 'short_ret': np.nan,
            'longshort_ret': np.nan, 'bench_ret': np.nan, 'ic': np.nan,
            'ic_signed': np.nan, 'N_factor': n, 'N_long': 0, 'N_short': 0,
            'long_cost': 0.0, 'short_cost': 0.0}


def _simulate_portfolio(stock_list, capital, next_year, next_month,
                        close_first_lookup, close_last_lookup):
    """
    真实买卖模拟：等权分配、整手、现金管理、扣费，并正确处理退市（归零）。
    返回 (组合净收益率, 实际持仓数, 总成本率)
    """
    capital_per_stock = capital / len(stock_list)
    total_invested = total_sell_proceeds = total_cost = 0.0
    n_held = 0

    for ts in stock_list:
        try:
            buy_price = close_first_lookup.loc[(ts, next_year, next_month)]
        except KeyError:
            continue
        if pd.isna(buy_price) or buy_price <= 0:
            continue
        shares = int(capital_per_stock / buy_price / 100) * 100
        if shares < 100:
            continue   # 修正 bug2：资金不足 1 手则跳过，不再强行买入导致超分配
        buy_amount = shares * buy_price
        buy_comm = max(buy_amount * COMMISSION_RATE, MIN_COMMISSION)
        slip_buy = buy_amount * SLIPPAGE_RATE

        try:
            sell_price = close_last_lookup.loc[(ts, next_year, next_month)]
        except KeyError:
            sell_price = np.nan
        if pd.isna(sell_price) or sell_price <= 0:
            # 退市 / 长期停牌无法卖出 -> 归零，全额亏损（正确计入幸存者偏差）
            sell_amount = 0.0
        else:
            sell_amount = shares * sell_price
        sell_comm = max(sell_amount * COMMISSION_RATE, MIN_COMMISSION)
        stamp = sell_amount * stamp_duty_rate(int(f"{next_year}{next_month:02d}28"))  # 月末卖出→分段印花税率
        slip_sell = sell_amount * SLIPPAGE_RATE

        total_cost += buy_comm + sell_comm + stamp + slip_buy + slip_sell
        total_invested += buy_amount
        total_sell_proceeds += sell_amount
        n_held += 1

    if total_invested == 0 or n_held == 0:
        return np.nan, 0, 0.0
    net_ret = (total_sell_proceeds - total_invested - total_cost) / total_invested
    cost_rate = total_cost / total_invested
    return net_ret, n_held, cost_rate


# =========================================================================== #
# 模块化函数：指标
# =========================================================================== #

def calc_metrics(records, direction):
    ret_long = records['long_ret'].dropna()
    ret_ls = records['longshort_ret'].dropna()
    ret_bench = records['bench_ret'].dropna()
    ic_ser = records['ic'].dropna()
    ic_dir_ser = records['ic_signed'].dropna()

    metrics = {
        'N': len(ret_long),
        'long_ann_ret': ret_long.mean() * 12,
        'ls_ann_ret': ret_ls.mean() * 12,
        'bench_ann_ret': ret_bench.mean() * 12,
        'long_ann_vol': ret_long.std() * np.sqrt(12),
        'ls_ann_vol': ret_ls.std() * np.sqrt(12),
        'bench_ann_vol': ret_bench.std() * np.sqrt(12),
        'long_sharp': ret_long.mean() / ret_long.std() if ret_long.std() > 0 else np.nan,
        'ls_sharp': ret_ls.mean() / ret_ls.std() if ret_ls.std() > 0 else np.nan,
        'bench_sharp': ret_bench.mean() / ret_bench.std() if ret_bench.std() > 0 else np.nan,
        'ic_mean': ic_ser.mean(),
        'ic_dir_mean': ic_dir_ser.mean(),       # 方向修正后的 IC
        'ic_std': ic_ser.std(),
        'ir': ic_dir_ser.mean() / ic_dir_ser.std() if ic_dir_ser.std() > 0 else np.nan,
        'ic_pos': (ic_dir_ser > 0).sum() / len(ic_dir_ser) if len(ic_dir_ser) > 0 else 0,
        'ic_good': abs(ic_dir_ser.mean()) > IC_GOOD_THRESHOLD if len(ic_dir_ser) > 0 else False,
        'long_mdd': np.nan, 'ls_mdd': np.nan, 'bench_mdd': np.nan,
    }

    if len(ret_long) > 0:
        cum = (ret_long + 1).cumprod()
        metrics['long_mdd'] = ((cum - cum.cummax()) / cum.cummax()).min()
    if len(ret_ls) > 0:
        cum = (ret_ls + 1).cumprod()
        metrics['ls_mdd'] = ((cum - cum.cummax()) / cum.cummax()).min()
    if len(ret_bench) > 0:
        cum = (ret_bench + 1).cumprod()
        metrics['bench_mdd'] = ((cum - cum.cummax()) / cum.cummax()).min()

    return metrics, ret_long, ret_ls, ret_bench, ic_ser


# =========================================================================== #
# 报告 / 图表 / 导出（每个因子一份）
# =========================================================================== #

def generate_report(metrics, ret_long, ret_ls, ret_bench, ic_ser, meta, out_path):
    direction = meta['direction']
    report = f"""
{'='*78}
单因子回测报告 — {meta['name']}（方向={direction}）
{'='*78}

回测周期：    {START_DATE} ~ {END_DATE} (共 {metrics['N']} 个月)
调仓：        月度，多头取因子{direction}端 TOP {TOP_N}
基准：        {BENCHMARK_INDEX} (沪深300)

一、收益指标
{'-'*78}
{'':<12} {'年化收益':>11} {'年化波动':>11} {'夏普':>9} {'最大回撤':>10}
多头         {metrics['long_ann_ret']:11.3%} {metrics['long_ann_vol']:11.3%} {metrics['long_sharp']:9.3f} {metrics['long_mdd']:10.3%}
多空(多-空)  {metrics['ls_ann_ret']:11.3%} {metrics['ls_ann_vol']:11.3%} {metrics['ls_sharp']:9.3f} {metrics['ls_mdd']:10.3%}
基准         {metrics['bench_ann_ret']:11.3%} {metrics['bench_ann_vol']:11.3%} {metrics['bench_sharp']:9.3f} {metrics['bench_mdd']:10.3%}

二、因子评价（RankIC，已按方向修正）
{'-'*78}
RankIC 均值    : {metrics['ic_mean']:8.3%}  (原始)
方向修正 IC    : {metrics['ic_dir_mean']:8.3%}
IR 值          : {metrics['ir']:8.3f}
正 IC 占比     : {metrics['ic_pos']:8.3%}
信号判定       : {'✅ 有信号' if metrics['ic_good'] else '❌ 无效'}

三、结论
{'-'*78}
1. 多头年化：{metrics['long_ann_ret']:.2%}，相对基准 {metrics['long_ann_ret']-metrics['bench_ann_ret']:+.2%}
2. 多空年化：{metrics['ls_ann_ret']:.2%}，夏普 {metrics['ls_sharp']:.2f}
3. 方向修正 IC：{metrics['ic_dir_mean']:.2%}（阈值 ±{IC_GOOD_THRESHOLD:.0%}）

四、口径备注
{'-'*78}
  - 股票池：行情库实际交易代码（含退市），退市按归零计入
  - 成本：佣金万2.5(最低5元)+印花千1→千0.5(2023-08-28起,仅卖)+滑点千1(买卖)
  - 过滤：ST/*ST、北交所、±9.8%涨跌停、流动性(日均成交额≥{LIQUIDITY_MIN_AVG_AMOUNT/1e8:.1f}亿)
  - 复权：前复权(adj_factor)
"""
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(report)
    return report


def plot_results(records, metrics, meta, out_path):
    ret_long = pd.Series(records['long_ret'].values,
                         index=records['year'].astype(str) + "-" + records['month'].astype(str).str.zfill(2))
    ret_ls = pd.Series(records['longshort_ret'].values, index=ret_long.index)
    ret_bench = pd.Series(records['bench_ret'].values, index=ret_long.index)
    ic_ser = pd.Series(records['ic_signed'].values, index=ret_long.index)
    ret_long, ret_ls, ret_bench, ic_ser = ret_long.dropna(), ret_ls.dropna(), ret_bench.dropna(), ic_ser.dropna()
    if len(ret_long) == 0:
        print("  [plot] 无有效数据，跳过绘图")
        return
    cum_long, cum_ls, cum_bench = (ret_long+1).cumprod(), (ret_ls+1).cumprod(), (ret_bench+1).cumprod()

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    axes[0].plot(cum_long.index, cum_long.values, label=f"{meta['name']} 多头", color='tab:red', lw=2)
    axes[0].plot(cum_ls.index, cum_ls.values, label=f"{meta['name']} 多空", color='tab:orange', lw=2.2)
    axes[0].plot(cum_bench.index, cum_bench.values, label="沪深300", color='tab:blue', ls='--', lw=1.5)
    axes[0].set_title(f"累计收益 — {meta['name']}（方向={meta['direction']}）", fontsize=14)
    axes[0].set_ylabel("累计净值"); axes[0].legend(fontsize=10); axes[0].grid(True, alpha=.3)

    x = ret_long.index; pos = np.arange(len(x)); w = 0.35
    axes[1].bar(pos-w/2, ret_long, w, label="多头", color='tab:red', alpha=.7)
    axes[1].bar(pos+w/2, ret_ls, w, label="多空", color='tab:orange', alpha=.7)
    axes[1].set_title("月度收益", fontsize=14); axes[1].set_ylabel("月度收益")
    axes[1].legend(fontsize=10); axes[1].grid(True, alpha=.3)
    axes[1].set_xticks(pos); axes[1].set_xticklabels(x, rotation=45, fontsize=8)

    xi = ic_ser.index; pi = np.arange(len(xi))
    axes[2].bar(pi, ic_ser, color='tab:green', alpha=.8, width=.6, label="方向修正 RankIC")
    axes[2].axhline(metrics['ic_dir_mean'], color='black', ls='--',
                    label=f"IC均值={metrics['ic_dir_mean']:.3%}", alpha=.7)
    axes[2].axhline(0, color='gray', lw=.8)
    axes[2].set_title(f"月度 RankIC(方向修正, 均值={metrics['ic_dir_mean']:.3%}, IR={metrics['ir']:.3f})", fontsize=14)
    axes[2].set_ylabel("RankIC"); axes[2].legend(fontsize=10); axes[2].grid(True, alpha=.3)
    step = max(1, len(xi)//12)
    axes[2].set_xticks(pi[::step]); axes[2].set_xticklabels(xi[::step], rotation=45, fontsize=8)

    plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  图表已保存: {out_path}")


def export_results(records, long_stocks, short_stocks, out_prefix):
    records = records.copy()
    records['month'] = records['year'].astype(str) + "-" + records['month'].astype(str).str.zfill(2)
    cols = ['month', 'year', 'long_ret', 'short_ret', 'longshort_ret', 'bench_ret',
            'ic', 'ic_signed', 'long_cost', 'short_cost', 'N_long', 'N_short']
    records[cols].to_csv(f"{out_prefix}_detail.csv", index=False, encoding='utf-8')
    cum_long = (records['long_ret'] + 1).cumprod()
    cum_ls = (records['longshort_ret'] + 1).cumprod()
    cum_bench = (records['bench_ret'] + 1).cumprod()
    pd.DataFrame({'month': records['month'], 'long': cum_long, 'longshort': cum_ls, 'benchmark': cum_bench}
                 ).to_csv(f"{out_prefix}_cumulative.csv", index=False, encoding='utf-8')
    long_stocks.to_csv(f"{out_prefix}_long.csv", index=False, encoding='utf-8')
    short_stocks.to_csv(f"{out_prefix}_short.csv", index=False, encoding='utf-8')
    print(f"  导出: {out_prefix}_detail.csv / _cumulative.csv / _long.csv / _short.csv")


# =========================================================================== #
# 跨因子对比
# =========================================================================== #

def compare_factors(results):
    """
    results: list of dict(factor_meta, metrics, records, ret_long, ...)
    产出对比表 CSV + 文本报告 + 叠加净值图。
    """
    print("\n" + "=" * 78)
    print("跨因子对比")
    print("=" * 78)

    rows = []
    for r in results:
        m = r['metrics']; meta = r['meta']
        rows.append({
            'factor': meta['name'],
            'direction': meta['direction'],
            'ic_dir_mean': m['ic_dir_mean'],
            'ic_ir': m['ir'],
            'ic_pos': m['ic_pos'],
            'long_ann_ret': m['long_ann_ret'],
            'ls_ann_ret': m['ls_ann_ret'],
            'bench_ann_ret': m['bench_ann_ret'],
            'long_mdd': m['long_mdd'],
            'ls_mdd': m['ls_mdd'],
            'long_sharpe': m['long_sharp'],
            'ls_sharpe': m['ls_sharp'],
            'signal': '✅' if m['ic_good'] else '❌',
        })
    cmp_df = pd.DataFrame(rows).sort_values('ic_dir_mean', ascending=False).reset_index(drop=True)
    cmp_path = OUT_DIR / "comparison.csv"
    cmp_df.to_csv(cmp_path, index=False, encoding='utf-8')
    print(f"\n对比表已保存: {cmp_path}\n")
    print(cmp_df.to_string(index=False))

    # 文本报告（含多重检验提醒）
    n = len(cmp_df)
    best = cmp_df.iloc[0]
    report = f"""
{'='*78}
跨因子对比报告（共 {n} 个因子）
{'='*78}

{compare_factors.__doc__}
因子排名（按方向修正 RankIC 降序）：
{cmp_df.to_string(index=False)}

多重检验提醒（数据挖掘偏差）：
  本次同时测试 {n} 个因子，最优者的 IC/收益会被"挑最好的"放大约 √{n} ≈ {np.sqrt(n):.1f} 倍。
  只有样本外或更长区间依然稳健的因子，才值得信任。
  建议：每个因子先登记假设（方向、预期 IC），最终用 √N 折扣或样本外验证。

结论速览：
  方向修正 IC 最强：{best['factor']} ({best['ic_dir_mean']:+.2%})
  多头年化最高     ：{cmp_df.loc[cmp_df['long_ann_ret'].idxmax(),'factor']} "
                    f"({cmp_df['long_ann_ret'].max():+.2%})
  有信号的因子     ：{', '.join(cmp_df.loc[cmp_df['signal']=='✅','factor'].tolist()) or '无'}
"""
    rep_path = OUT_DIR / "comparison_report.txt"
    with open(rep_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(report)

    # 叠加净值图（各因子多头累计净值 vs 基准）
    plt.figure(figsize=(14, 7))
    for r in results:
        rec = r['records']
        rl = pd.Series(rec['long_ret'].values,
                       index=rec['year'].astype(str)+"-"+rec['month'].astype(str).str.zfill(2)).dropna()
        if len(rl) == 0:
            continue
        cum = (rl + 1).cumprod()
        plt.plot(cum.index, cum.values, label=f"{r['meta']['name']}(多头)", lw=1.6)
    # 基准
    if results:
        rec0 = results[0]['records']
        rb = pd.Series(rec0['bench_ret'].values,
                       index=rec0['year'].astype(str)+"-"+rec0['month'].astype(str).str.zfill(2)).dropna()
        if len(rb) > 0:
            plt.plot((rb+1).cumprod().index, (rb+1).cumprod().values,
                     label="沪深300(基准)", color='black', ls='--', lw=1.5)
    plt.title("各因子多头累计净值对比", fontsize=15)
    plt.ylabel("累计净值"); plt.legend(fontsize=9, ncol=2); plt.grid(True, alpha=.3)
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout(); plt.savefig(OUT_DIR / "comparison_plot.png", dpi=150, bbox_inches='tight'); plt.close()
    print(f"  对比图已保存: {OUT_DIR / 'comparison_plot.png'}")


# =========================================================================== #
# 主流程
# =========================================================================== #

def run_one_factor(mod, meta, raw_data, universe, start_dt, end_dt):
    name = meta['name']; direction = meta['direction']
    print(f"\n{'#'*78}\n# 因子：{name}（方向={direction}）\n{'#'*78}")
    try:
        df_with_factor = mod.compute(raw_data.copy())
    except Exception as e:
        print(f"  [ERR] 因子 {name} 计算失败: {e}")
        traceback.print_exc()
        return None

    if name not in df_with_factor.columns:
        print(f"  [ERR] 因子 {name} 未产出列 '{name}'，跳过")
        return None

    monthly_factor = group_to_monthly(df_with_factor, name, start_dt, end_dt)
    if len(monthly_factor) == 0:
        print(f"  [ERR] 因子 {name} 无有效月度值，跳过")
        return None

    records, long_stocks, short_stocks = monthly_loop(
        monthly_factor, name, direction, raw_data, universe, start_dt, end_dt)
    metrics, ret_long, ret_ls, ret_bench, ic_ser = calc_metrics(records, direction)

    prefix = str(OUT_DIR / name)
    generate_report(metrics, ret_long, ret_ls, ret_bench, ic_ser, meta, f"{prefix}_report.txt")
    plot_results(records, metrics, meta, f"{prefix}_plot.png")
    export_results(records, long_stocks, short_stocks, prefix)

    return {'meta': meta, 'metrics': metrics, 'records': records,
            'ret_long': ret_long, 'ret_ls': ret_ls, 'ret_bench': ret_bench, 'ic_ser': ic_ser}


def main():
    global TOP_N, INITIAL_CASH
    parser = argparse.ArgumentParser(description="单因子动态回测平台")
    parser.add_argument("--factor", help="只跑指定因子（NAME）；不填则跑 factors/ 下全部")
    parser.add_argument("--list", action="store_true", help="列出已发现的因子后退出")
    parser.add_argument("--top-n", type=int, default=TOP_N)
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end", default=END_DATE)
    parser.add_argument("--cash", type=float, default=INITIAL_CASH)
    parser.add_argument("--factors-dir", default=str(FACTORS_DIR))
    args = parser.parse_args()

    TOP_N = args.top_n
    INITIAL_CASH = args.cash

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start_dt = pd.to_datetime(args.start)
    end_dt = pd.to_datetime(args.end)

    print("►►► 单因子动态回测平台 ►►►")
    print(f"  区间: {start_dt.date()} ~ {end_dt.date()} | TOP_N={TOP_N} | 资金={INITIAL_CASH:,.0f}元")

    raw_data = load_raw_data(start_dt, end_dt)
    universe = build_universe(raw_data, start_dt)

    factors = discover_factors(Path(args.factors_dir))
    if args.list:
        for _, meta in factors:
            print(f"  - {meta['name']} (方向={meta['direction']})")
        return

    if args.factor:
        factors = [(m, meta) for (m, meta) in factors if meta['name'] == args.factor]
        if not factors:
            print(f"  [ERR] 未找到因子: {args.factor}")
            return

    if not factors:
        return

    results = []
    for mod, meta in factors:
        res = run_one_factor(mod, meta, raw_data, universe, start_dt, end_dt)
        if res is not None:
            results.append(res)

    if len(results) > 1:
        compare_factors(results)
    elif len(results) == 1:
        print("\n►►► 仅一个因子，跳过跨因子对比 ►►►")

    print("\n►►► 全部完成！输出目录:", OUT_DIR, "►►►")


if __name__ == "__main__":
    main()
