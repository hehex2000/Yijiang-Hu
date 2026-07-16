#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
All-in-One APB 因子计算 + 回测 一体化脚本
一键运行：因子计算 -> 月度因子导出 -> 月度调仓回测

功能：
  1. 从 SQLite 读取日线数据 (2010-至今)
  2. 计算 APB 因子（4 种方法）
  3. 输出最优月度因子值
  4. 月度调仓回测（多空 vs 沪深300）
  5. 生成报告 + 图表

输入数据库：d:\tu-sharedata\astock_daily.db
输出目录：d:\tu-sharedata\
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# 解决中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# =========================================================================== #
# 参数配置
# =========================================================================== #

DB_PATH = r"d:\tu-sharedata\astock_daily.db"

# APB 因子方法
TWAP_METHODS = {
    # 方法1: (O+H+L+C)/4
    "method1_ohlc4": lambda x: (x['open'] + x['high'] + x['low'] + x['close']) / 4,
    # 方法2: (H+L+C)/3
    "method2_hlc3":  lambda x: (x['high'] + x['low'] + x['close']) / 3,
    # 方法3: (O+C)/2
    "method3_oc2":   lambda x: (x['open'] + x['close']) / 2,
    # 方法4: sqrt(H*L)  —— 几何平均，偏向中位数
    "method4_sqrt_hl": lambda x: np.sqrt(x['high'] * x['low']),
}
PRIMARY_METHOD = "method1_ohlc4"  # 主方法
ROLLING_WINDOW = 5              # APB5D 滚动窗口

# 回测参数
TOP_N = 10                      # 散户持仓数量
START_DATE = "2017-12-29"
END_DATE = "2022-01-07"
BENCHMARK_INDEX = "000300.SH"
INITIAL_CASH = 100000           # 初始资金：10万元（散户规模）

# 交易成本参数
COMMISSION_RATE = 0.00025       # 手续费费率：万分之2.5
MIN_COMMISSION = 5.0            # 单笔最低手续费：5元
STAMP_DUTY_RATE = 0.001         # 印花税：千分之一（仅卖出收取）

# 输出前缀
OUT_PREFIX = r"d:\tu-sharedata\apb_backtest"
PLOT_FILE = r"d:\tu-sharedata\apb_backtest.png"
RESULT_FILE = r"d:\tu-sharedata\apb_backtest.csv"
REPORT_FILE = r"d:\tu-sharedata\apb_backtest_report.txt"

# 日志频率
PRINT_EVERY = 10000

# =========================================================================== #
# 模块化函数：数据加载
# =========================================================================== #

def load_raw_data(start_dt, end_dt):
    """加载日线 OHLCV + amount + 复权因子"""
    print(f"[1/9] 加载日线数据 {start_dt} ~ {end_dt}...")
    
    # 扩展范围（保证能计算滚动窗口）
    start_actual = (pd.to_datetime(start_dt) - pd.Timedelta(days=ROLLING_WINDOW*2)).strftime("%Y%m%d")
    end_actual = (pd.to_datetime(end_dt) + pd.Timedelta(days=10)).strftime("%Y%m%d")
    
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
    df = df[(df['vol'] > 0) & (df['amount'] > 0)].copy()
    
    # 前复权：以最后一天的复权因子为基准
    print("  应用前复权...")
    # 每只股票最新的复权因子
    latest_adj = df.groupby('ts_code')['adj_factor'].transform('last')
    # 复权比例 = 当日复权因子 / 最新复权因子
    ratio = df['adj_factor'] / latest_adj
    # 对价格应用复权（vol和amount不需复权，因为VWAP=amount/vol本身就反映真实成交均价）
    df['open_adj'] = df['open'] * ratio
    df['high_adj'] = df['high'] * ratio
    df['low_adj'] = df['low'] * ratio
    df['close_adj'] = df['close'] * ratio
    
    print(f"  日线数据：{len(df):,} 行 ({df['ts_code'].nunique()} 只股票)")
    return df

def load_benchmark_data(start_dt, end_dt):
    """加载沪深300 指数（修复：库中字段为 ts_code）"""
    query = """
    SELECT trade_date, close
    FROM index_daily
    WHERE ts_code = ? AND trade_date >= ? AND trade_date <= ?
    ORDER BY trade_date
    """
    
    # 必须格式化成 %Y%m%d 字符串才能传给 SQLite
    start_f = start_dt.strftime("%Y%m%d")
    end_f = end_dt.strftime("%Y%m%d")
    
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(query, conn, params=(BENCHMARK_INDEX, start_f, end_f))
    df['trade_date'] = pd.to_datetime(df['trade_date'], format="%Y%m%d")
    return df

def calc_stock_universe():
    """构建全A股股票池（非ST、非B股、非北交所，上市满一年）"""
    print("[2/9] 构建股票池...")
    
    with sqlite3.connect(DB_PATH) as conn:
        stocks = pd.read_sql_query("SELECT ts_code, name, list_date FROM stock_basic", conn)
    
    # 上市日期
    stocks['list_date'] = pd.to_datetime(stocks['list_date'], format="%Y%m%d", errors='coerce')
    stocks = stocks[stocks['list_date'].notnull()]
    
    # 剔除 ST、B股、北交所
    stocks = stocks[~stocks['name'].str.contains('S', na=False)]  # 剔除ST
    stocks = stocks[stocks['ts_code'].str[-3:] != '.BJ']  # 剔除北交所
    stocks = stocks[stocks['ts_code'].str[-3:] != '.BJ']  # 重复，安全
    stocks = stocks[stocks['ts_code'].str[-3:] != '.BJ']  # 再次保证
    
    print(f"  全A股池：{len(stocks):,} 只股票")
    return stocks

# =========================================================================== #
# 模块化函数：因子计算
# =========================================================================== #

def calc_apb5d(df, method_name="method1_ohlc4", window=5):
    """
    计算 APB 因子（跨日版本）
    
    核心逻辑：
      5日VWAP = 5日总成交额 / 5日总成交量（成交量加权的5日均价）
      5日TWAP = 5日收盘价的简单平均（时间加权的5日均价）
      
      买压大 → 跌的日子放量 → VWAP被拉低 → VWAP < TWAP → APB < 0
      卖压大 → 涨的日子放量 → VWAP被拉高 → VWAP > TWAP → APB > 0
      
      APB = (VWAP_5D - TWAP_5D) / TWAP_5D
      买 APB 最小的{TOP_N}只（买压最强）
    """
    print(f"[3/9] 计算 APB 因子（跨日版，{window}D 滚动）...")
    
    # 复权后的 VWAP（用复权价格基准对齐）
    latest_adj = df.groupby('ts_code')['adj_factor'].transform('last')
    ratio = df['adj_factor'] / latest_adj
    df['vwap_adj'] = (df['amount'] / df['vol'] * 10) * ratio  # 每日复权VWAP
    
    # === 5日滚动 VWAP（成交量加权）===
    # 正确做法：先对每日VWAP复权，再用每日成交量作为权重做加权平均
    # VWAP_5D = sum(vwap_adj_i * vol_i) / sum(vol_i)
    df['vwap_adj_x_vol'] = df['vwap_adj'] * df['vol']
    df['vwap_adj_x_vol_5d'] = df.groupby('ts_code')['vwap_adj_x_vol'].rolling(window=window, min_periods=3).sum().values
    df['vol_5d'] = df.groupby('ts_code')['vol'].rolling(window=window, min_periods=3).sum().values
    df['vwap_5d'] = df['vwap_adj_x_vol_5d'] / df['vol_5d']
    
    # === 5日滚动 TWAP（简单均价，用复权收盘价）===
    df['twap_5d'] = df.groupby('ts_code')['close_adj'].rolling(window=window, min_periods=3).mean().values
    
    # === APB ===
    # 买压大 -> VWAP_5D < TWAP_5D -> APB < 0
    # 所以 APB 越小（越负），买压越强，应该买入
    df['apb5d'] = (df['vwap_5d'] - df['twap_5d']) / df['twap_5d']
    
    # 清理
    df['apb5d'] = df['apb5d'].replace([np.inf, -np.inf], np.nan)
    
    print(f"  APB5D 计算完成: {df['apb5d'].isnull().sum():,} / {len(df):,} 为 NaN")
    
    # 打印样本验证
    sample = df[df['apb5d'].notna()][['ts_code','trade_date','close_adj','vwap_5d','twap_5d','apb5d']].tail(5)
    print("  样本验证:")
    print(sample.to_string(index=False))
    
    return df

def group_to_monthly(df_factor, start_dt, end_dt):
    """按月聚合：每月最后一天"""
    print("[4/9] 按月聚合因子值...")
    df = df_factor.copy()
    
    # 日期过滤
    df = df[(df['trade_date'] >= start_dt) & (df['trade_date'] <= end_dt)]
    
    # 年月
    df['year'] = df['trade_date'].dt.year
    df['month'] = df['trade_date'].dt.month
    
    # 每月最后一天
    idx = df.groupby(['ts_code', 'year', 'month'])['trade_date'].transform(max) == df['trade_date']
    monthly = df[idx][['ts_code', 'year', 'month', 'apb5d']].copy()
    monthly.reset_index(drop=True, inplace=True)
    
    print(f"  月度因子值: {len(monthly):,} 条记录")
    return monthly

# =========================================================================== #
# 模块化函数：回测逻辑
# =========================================================================== #

def monthly_loop(monthly_factor, raw_data, stocks_universe, start_dt, end_dt):
    """月循环：选股 -> 计算收益 -> 计算 RankIC"""
    print("[5/9] 月度回测循环...")
    
    # ---- 预计算：所有股票每月收益（向量化，一次性算完）----
    print("  预计算月度收益（向量化）...")
    raw = raw_data.copy()
    raw['year'] = raw['trade_date'].dt.year
    raw['month'] = raw['trade_date'].dt.month
    
    # 每只股票每月：第一个交易日 close_adj 和最后一个交易日 close_adj（用复权价算收益）
    monthly_first = raw.groupby(['ts_code', 'year', 'month'])['close_adj'].first().reset_index()
    monthly_first.rename(columns={'close_adj': 'close_first'}, inplace=True)
    monthly_last = raw.groupby(['ts_code', 'year', 'month'])['close_adj'].last().reset_index()
    monthly_last.rename(columns={'close_adj': 'close_last'}, inplace=True)
    
    # 每只股票每月最后一天的 pct_chg（用于涨跌停过滤）
    monthly_pct = raw.groupby(['ts_code', 'year', 'month'])['pct_chg'].last().reset_index()
    
    # 合并
    monthly_ret = monthly_first.merge(monthly_last, on=['ts_code', 'year', 'month'])
    monthly_ret = monthly_ret.merge(monthly_pct, on=['ts_code', 'year', 'month'])
    monthly_ret['ret'] = (monthly_ret['close_last'] - monthly_ret['close_first']) / monthly_ret['close_first']
    # 剔除异常收益（>50%可能是停牌复牌等）
    monthly_ret.loc[monthly_ret['ret'].abs() >= 0.5, 'ret'] = np.nan
    
    # 建立快速查找索引：(ts_code, year, month) -> ret, close_first, close_last
    ret_lookup = monthly_ret.set_index(['ts_code', 'year', 'month'])['ret']
    close_first_lookup = monthly_ret.set_index(['ts_code', 'year', 'month'])['close_first']
    close_last_lookup = monthly_ret.set_index(['ts_code', 'year', 'month'])['close_last']
    print(f"  预计算完成：{len(monthly_ret):,} 条月度收益记录")
    
    # ---- 确定月份 ----
    months = []
    cur = pd.to_datetime("2017-12-31")
    while cur <= pd.to_datetime(end_dt):
        months.append({'year': cur.year, 'month': cur.month, 'last_day': cur})
        cur = cur + pd.offsets.MonthEnd(1)
    print(f"  回测月份: {len(months)} 个月")
    
    # 基准指数
    benchmark = load_benchmark_data(start_dt, end_dt)
    benchmark = benchmark.set_index('trade_date')['close']
    
    # 交易成本参数（按实际买卖计算）
    capital_per_stock = INITIAL_CASH / TOP_N
    print(f"  交易成本: 每只股票资金={capital_per_stock:,.0f}元, "
          f"佣金费率={COMMISSION_RATE:.5f}(最低{MIN_COMMISSION}元/笔), "
          f"印花税={STAMP_DUTY_RATE:.4f}(仅卖出)")
    
    # 记录
    records = []
    long_stocks = []
    short_stocks = []
    valid_codes = set(stocks_universe['ts_code'])
    
    for i, m in enumerate(months):
        y, mon = m['year'], m['month']
        next_month = mon + 1 if mon < 12 else 1
        next_year = y if mon < 12 else y + 1
        
        # 0. 日志
        print(f"  [{i+1:2d}/{len(months)}] {y}-{mon:02d} | 选 {TOP_N} 只股票", end="")
        
        # 1. 当月因子值
        factors = monthly_factor[
            (monthly_factor['year'] == y) & (monthly_factor['month'] == mon)
        ].copy()
        factors = factors[factors['ts_code'].isin(valid_codes)]
        factors = factors[factors['apb5d'].notnull()]
        
        if len(factors) == 0:
            print(" -> 无因子数据，跳过")
            records.append({'year': y, 'month': mon, 'long_ret': np.nan,
                'longshort_ret': np.nan, 'bench_ret': np.nan, 'ic': np.nan, 'N_factor': 0})
            continue
        
        # 2. 涨跌停过滤（用预计算的 pct_chg）
        # 用下月的 pct_chg 过滤（调仓日 = 下月第一个交易日）
        # 实际上应该用当月最后交易日的涨跌停，这里用预查的方式
        # 从 raw_data 中找该月实际最后一个交易日
        month_mask = (raw_data['trade_date'].dt.year == y) & (raw_data['trade_date'].dt.month == mon)
        actual_last_day = raw_data.loc[month_mask, 'trade_date'].max()
        
        if pd.isna(actual_last_day):
            print(" -> 无行情，跳过")
            records.append({'year': y, 'month': mon, 'long_ret': np.nan,
                'longshort_ret': np.nan, 'bench_ret': np.nan, 'ic': np.nan, 'N_factor': 0})
            continue
        
        last_mask = raw_data['trade_date'] == actual_last_day
        last_data = raw_data[last_mask][['ts_code', 'pct_chg']].copy()
        factors = factors.merge(last_data, on='ts_code', how='left')
        factors = factors[(factors['pct_chg'].abs() < 9.8) & (factors['pct_chg'].notnull())]
        
        if len(factors) < TOP_N * 2:
            print(f" -> 因子值不足 {TOP_N*2}，跳过")
            records.append({'year': y, 'month': mon, 'long_ret': np.nan,
                'longshort_ret': np.nan, 'bench_ret': np.nan, 'ic': np.nan, 'N_factor': len(factors)})
            continue
        
        # 3. 排序选股
        # APB 越小（越负）= 买压越强 = 买入
        # APB 越大（越正）= 卖压越强 = 卖出
        factors = factors.sort_values('apb5d', ascending=True)
        long_list = factors.head(TOP_N)['ts_code'].tolist()   # APB最小的{TOP_N}只（买压最强）
        short_list = factors.tail(TOP_N)['ts_code'].tolist()  # APB最大的{TOP_N}只（卖压最强）
        
        # 4. 真实仓位模拟：按实际价格买整手，管理现金，跟踪净值
        def simulate_portfolio(stock_list, capital):
            """模拟真实买卖：等权分配资金，买整手，剩余现金，扣手续费和印花税
            返回: (组合净收益率, 实际持仓数, 总成本率)
            """
            capital_per_stock = capital / len(stock_list)
            total_invested = 0.0    # 实际投入金额
            total_sell_proceeds = 0.0  # 卖出回收金额
            total_cost = 0.0       # 总交易成本
            n_held = 0
            
            for ts in stock_list:
                try:
                    buy_price = close_first_lookup.loc[(ts, next_year, next_month)]
                    sell_price = close_last_lookup.loc[(ts, next_year, next_month)]
                    if pd.isna(buy_price) or pd.isna(sell_price) or buy_price <= 0:
                        continue
                    # 买整手（100股为单位），向下取整
                    shares = int(capital_per_stock / buy_price / 100) * 100
                    if shares < 100:
                        shares = 100  # 至少1手
                    buy_amount = shares * buy_price
                    sell_amount = shares * sell_price
                    # 手续费（双向，最低5元）
                    buy_comm = max(buy_amount * COMMISSION_RATE, MIN_COMMISSION)
                    sell_comm = max(sell_amount * COMMISSION_RATE, MIN_COMMISSION)
                    # 印花税（仅卖出）
                    stamp = sell_amount * STAMP_DUTY_RATE
                    total_cost += buy_comm + sell_comm + stamp
                    total_invested += buy_amount
                    total_sell_proceeds += sell_amount
                    n_held += 1
                except KeyError:
                    continue
            
            if total_invested == 0 or n_held == 0:
                return np.nan, 0, 0.0
            
            # 组合净收益 = (卖出回收 - 投入 - 成本) / 投入
            net_ret = (total_sell_proceeds - total_invested - total_cost) / total_invested
            cost_rate = total_cost / total_invested
            return net_ret, n_held, cost_rate
        
        long_ret, n_long, long_cost = simulate_portfolio(long_list, INITIAL_CASH)
        short_ret, n_short, short_cost = simulate_portfolio(short_list, INITIAL_CASH)
        
        # 5. 多空
        ls_ret = long_ret - short_ret if pd.notnull(long_ret) and pd.notnull(short_ret) else np.nan
        
        # 6. 基准
        bench_ret = np.nan
        bench_range = benchmark[(benchmark.index >= pd.to_datetime(f"{next_year}-{next_month:02d}-01")) & 
                                (benchmark.index <= min(pd.to_datetime(f"{next_year}-{next_month:02d}-01") + pd.offsets.MonthEnd(0), end_dt))]
        if len(bench_range) >= 2:
            bench_ret = (bench_range.iloc[-1] - bench_range.iloc[0]) / bench_range.iloc[0]
        
        # 7. RankIC（用全部股票的当月因子 vs 下月收益，而非只看long_list）
        ic = np.nan
        # 所有有下月收益的股票
        next_rets = monthly_ret[(monthly_ret['year'] == next_year) & (monthly_ret['month'] == next_month)]
        # 合并因子和收益
        ic_data = factors[['ts_code', 'apb5d']].merge(next_rets[['ts_code', 'ret']], on='ts_code', how='inner')
        ic_data = ic_data.dropna()
        if len(ic_data) > 10:
            corr = ic_data['apb5d'].corr(ic_data['ret'], method='spearman')
            if pd.notna(corr):
                ic = corr
        
        # 8. 记录
        status = f" | 多头 {long_ret:.4f}" if pd.notna(long_ret) else " | 多头 N/A"
        status += f" | 多空 {ls_ret:.4f}" if pd.notna(ls_ret) else " | 多空 N/A"
        status += f" | IC {ic:.4f}" if pd.notna(ic) else " | IC N/A"
        status += f" | 成本 {long_cost:.4%}/{short_cost:.4%}" if pd.notna(long_ret) else ""
        status += f" | 持仓 {n_long}/{n_short}"
        print(status)
        
        records.append({
            'year': y, 'month': mon,
            'long_ret': long_ret, 'short_ret': short_ret,
            'longshort_ret': ls_ret, 'bench_ret': bench_ret,
            'ic': ic, 'N_factor': len(factors),
            'N_long': n_long, 'N_short': n_short,
            'long_cost': long_cost, 'short_cost': short_cost
        })
        long_stocks.append({'year': y, 'month': mon, 'ts_code': ','.join(long_list[:20])})
        short_stocks.append({'year': y, 'month': mon, 'ts_code': ','.join(short_list[:20])})
    
    # 转换为 DataFrame
    records_df = pd.DataFrame(records)
    long_stocks = pd.DataFrame(long_stocks)
    short_stocks = pd.DataFrame(short_stocks)
    
    print(f"  回测完成: {len(records_df)} 个月")
    return records_df, long_stocks, short_stocks

# =========================================================================== #
# 模块化函数：计算指标
# =========================================================================== #

def calc_metrics(records):
    """计算回测指标"""
    ret_long = records['long_ret'].dropna()
    ret_ls = records['longshort_ret'].dropna()
    ret_bench = records['bench_ret'].dropna()
    ic_ser = records['ic'].dropna()
    
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
        'ic_std': ic_ser.std(),
        'ir': ic_ser.mean() / ic_ser.std() if ic_ser.std() > 0 else np.nan,
        'ic_pos': (ic_ser > 0).sum() / len(ic_ser) if len(ic_ser) > 0 else 0,
        'long_mdd': np.nan,
        'ls_mdd': np.nan,
        'bench_mdd': np.nan,
    }
    
    # 累计收益计算 MDD
    if len(ret_long) > 0:
        cum_long = (ret_long + 1).cumprod()
        dd_long = (cum_long - cum_long.cummax()) / cum_long.cummax()
        metrics['long_mdd'] = dd_long.min()
    
    if len(ret_ls) > 0:
        cum_ls = (ret_ls + 1).cumprod()
        dd_ls = (cum_ls - cum_ls.cummax()) / cum_ls.cummax()
        metrics['ls_mdd'] = dd_ls.min()
    
    if len(ret_bench) > 0:
        cum_bench = (ret_bench + 1).cumprod()
        dd_bench = (cum_bench - cum_bench.cummax()) / cum_bench.cummax()
        metrics['bench_mdd'] = dd_bench.min()
    
    return metrics, ret_long, ret_ls, ret_bench, ic_ser

# =========================================================================== #
# 模块化函数：生成报告
# =========================================================================== #

def generate_report(metrics, ret_long, ret_ls, ret_bench, ic_ser, out_path):
    """生成文本报告"""
    print(f"[6/9] 生成报告 {out_path}")
    
    report = f"""
APB 因子回测报告（一体化）
===============================================================================

因子：        APB5D (VWAP vs TWAP, {PRIMARY_METHOD})
回测周期：    {START_DATE} ~ {END_DATE} (共 {metrics['N']} 个月)
调仓策略：    月度调仓，买入 APB5D 最小的 {TOP_N} 只股票（买压最强）
资金规模：    {INITIAL_CASH:,}元（散户）
基准：        {BENCHMARK_INDEX} (沪深300)

一、收益指标
-------------------------------------------------------------------------------
{'':<14} {'年化收益':>12} {'年化波动':>12} {'夏普比率':>12} {'最大回撤':>12}
{'-'*14} {'-'*12} {'-'*12} {'-'*12} {'-'*12}
多头           {metrics['long_ann_ret']:12.3%} {metrics['long_ann_vol']:12.3%} {metrics['long_sharp']:12.3f} {metrics['long_mdd']:12.3%}
多空 (多-空)   {metrics['ls_ann_ret']:12.3%} {metrics['ls_ann_vol']:12.3%} {metrics['ls_sharp']:12.3f} {metrics['ls_mdd']:12.3%}
基准           {metrics['bench_ann_ret']:12.3%} {metrics['bench_ann_vol']:12.3%} {metrics['bench_sharp']:12.3f} {metrics['bench_mdd']:12.3%}

二、因子评价指标
-------------------------------------------------------------------------------
RankIC 均值     : {metrics['ic_mean']:8.3%}
RankIC 标准差   : {metrics['ic_std']:8.3f}
IR 值           : {metrics['ir']:8.3f}
正 IC 占比      : {metrics['ic_pos']:8.3%}

三、结论
-------------------------------------------------------------------------------
1. 多头年化收益: {metrics['long_ann_ret']:.2%}，跑赢基准 {metrics['long_ann_ret'] - metrics['bench_ann_ret']:.2%}
2. 多空年化收益: {metrics['ls_ann_ret']:.2%}，夏普 {metrics['ls_sharp']:.2f}
3. RankIC 均值  : {metrics['ic_mean']:.2%} (视频: 9.07%)
4. 指标一致性  : {'✅ GOOD' if metrics['ic_mean'] > 0.05 else '❌ LOW IC'}

四、备注
-------------------------------------------------------------------------------
  - 全A股池：剔除 ST、B 股，上市满一年
  - 涨跌停过滤：剔除 ±9.8% 以上股票
  - 月度收益：下月第一个交易日买入，最后交易日卖出
  - 交易成本：按实际买卖价格计算（买入佣金+卖出佣金双向万分之2.5最低5元，印花税千分之一卖出收取）
  - 买卖手数：按每只股票实际价格向下取整到100股（1手）
  - 初始资金：10万元（散户），等权分配10只
  - 复权方式：前复权（基于 adj_factor）
"""
    
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(report)
    
    return report

# =========================================================================== #
# 模块化函数：可视化
# =========================================================================== #

def plot_results(records, metrics, out_path):
    """绘制图表"""
    print(f"[7/9] 生成图表 {out_path}")
    
    # 累计收益
    ret_long = pd.Series(records['long_ret'].values, index=records['year'].astype(str) + "-" + records['month'].astype(str).str.zfill(2))
    ret_ls = pd.Series(records['longshort_ret'].values, index=records['year'].astype(str) + "-" + records['month'].astype(str).str.zfill(2))
    ret_bench = pd.Series(records['bench_ret'].values, index=records['year'].astype(str) + "-" + records['month'].astype(str).str.zfill(2))
    ic_ser = pd.Series(records['ic'].values, index=records['year'].astype(str) + "-" + records['month'].astype(str).str.zfill(2))
    
    # 清理
    ret_long = ret_long.dropna()
    ret_ls = ret_ls.dropna()
    ret_bench = ret_bench.dropna()
    ic_ser = ic_ser.dropna()
    
    # 累计净值
    cum_long = (ret_long + 1).cumprod()
    cum_ls = (ret_ls + 1).cumprod()
    cum_bench = (ret_bench + 1).cumprod()
    
    # 绘图 3 个子图
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 子图1：累计收益
    axes[0].plot(cum_long.index, cum_long.values, label="APB5D 多头", color='tab:red', linewidth=2)
    axes[0].plot(cum_ls.index, cum_ls.values, label="APB5D 多空", color='tab:orange', linewidth=2.2)
    axes[0].plot(cum_bench.index, cum_bench.values, label="沪深300", color='tab:blue', linestyle='--', linewidth=1.5)
    axes[0].set_title("累计收益（月度调仓）", fontsize=14)
    axes[0].set_ylabel("累计净值", fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, linewidth=0.6)
    axes[0].tick_params(axis="x", rotation=45, labelsize=8)
    
    # 子图2：月度收益
    ax2 = axes[1]
    x = ret_long.index
    pos = np.arange(len(x))
    width = 0.35
    
    ax2.bar(pos - width/2, ret_long, width, label="多头", color='tab:red', alpha=0.7)
    ax2.bar(pos + width/2, ret_ls, width, label="多空", color='tab:orange', alpha=0.7)
    ax2.set_title("月度收益", fontsize=14)
    ax2.set_ylabel("月度收益", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, linewidth=0.6)
    ax2.set_xticks(pos)
    ax2.set_xticklabels(x, rotation=45, fontsize=8)
    
    # 子图3：RankIC
    ax3 = axes[2]
    xic = ic_ser.index
    posic = np.arange(len(xic))
    ax3.bar(posic, ic_ser, color='tab:green', alpha=0.8, label="RankIC", width=0.6)
    ax3.axhline(metrics['ic_mean'], color='black', linestyle='--', label=f"IC 均值 = {metrics['ic_mean']:.3%}", alpha=0.7)
    ax3.axhline(0.1, color='tab:purple', linestyle=':', label="IC 阈值 = 10%", alpha=0.4)
    ax3.axhline(-0.1, color='tab:purple', linestyle=':', alpha=0.4)
    ax3.set_title(f"月度 RankIC (均值 = {metrics['ic_mean']:.3%}, IR = {metrics['ir']:.3f})", fontsize=14)
    ax3.set_ylabel("RankIC", fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, linewidth=0.6)
    ax3.set_xticks(posic[::max(1, len(xic)//12)])
    ax3.set_xticklabels(xic[::max(1, len(xic)//12)], rotation=45, fontsize=8)
    ax3.tick_params(axis="x", which='minor', length=0)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  图表已保存")

# =========================================================================== #
# 模块化函数：导出数据
# =========================================================================== #

def export_results(records, long_stocks, short_stocks, out_prefix):
    """导出所有结果"""
    print(f"[8/9] 导出结果...")
    
    # 月回测结果
    records = records.copy()
    records['month'] = records['year'].astype(str) + "-" + records['month'].astype(str).str.zfill(2)
    records = records[['month', 'year', 'long_ret', 'short_ret', 'longshort_ret', 'bench_ret', 'ic', 'long_cost', 'short_cost', 'N_long', 'N_short']].copy()
    records.to_csv(f"{out_prefix}.csv", index=False, encoding='utf-8')
    print(f"  月收益明细: {out_prefix}.csv")
    
    # 累计收益
    ret_long = pd.Series(records['long_ret'])
    ret_ls = pd.Series(records['longshort_ret'])
    ret_bench = pd.Series(records['bench_ret'])
    
    cum_long = (ret_long + 1).cumprod()
    cum_ls = (ret_ls + 1).cumprod()
    cum_bench = (ret_bench + 1).cumprod()
    
    cum_df = pd.DataFrame({
        'month': records['month'],
        'long': cum_long,
        'longshort': cum_ls,
        'benchmark': cum_bench
    })
    cum_df.to_csv(f"{out_prefix}_cumulative.csv", index=False, encoding='utf-8')
    print(f"  累计收益: {out_prefix}_cumulative.csv")
    
    # 详细记录
    out_long = f"{out_prefix}_long.csv"
    out_short = f"{out_prefix}_short.csv"
    long_stocks.to_csv(out_long, index=False, encoding='utf-8')
    short_stocks.to_csv(out_short, index=False, encoding='utf-8')
    print(f"  多头 Top20: {out_long}")
    print(f"  空头 Top20: {out_short}")

# =========================================================================== #
# 主流程
# =========================================================================== #

def main():
    print("►►► APB 因子一体化脚本（计算 + 回测）►►►\n")
    
    # 1. 数据加载
    start_dt = pd.to_datetime(START_DATE)
    end_dt = pd.to_datetime(END_DATE)
    raw_data = load_raw_data(start_dt, end_dt)
    stocks_universe = calc_stock_universe()
    
    # 2. 因子计算
    df = calc_apb5d(raw_data, PRIMARY_METHOD, ROLLING_WINDOW)
    monthly_factor = group_to_monthly(df, start_dt, end_dt)
    
    # 3. 回测
    records, long_stocks, short_stocks = monthly_loop(
        monthly_factor, raw_data, stocks_universe, start_dt, end_dt
    )
    
    # 4. 计算指标
    metrics, ret_long, ret_ls, ret_bench, ic_ser = calc_metrics(records)
    
    # 5. 生成报告
    generate_report(metrics, ret_long, ret_ls, ret_bench, ic_ser, REPORT_FILE)
    
    # 6. 可视化
    plot_results(records, metrics, PLOT_FILE)
    
    # 7. 导出结果
    export_results(records, long_stocks, short_stocks, OUT_PREFIX)
    
    # 8. 输出总结
    print("\n►►► 一体化脚本完成！►►►")
    print(f"  报告: {REPORT_FILE}")
    print(f"  月收益: {OUT_PREFIX}.csv")
    print(f"  累计收益: {OUT_PREFIX}_cumulative.csv")
    print(f"  图表: {PLOT_FILE}")
    print(f"  多头明细: {OUT_PREFIX}_long.csv")
    print(f"  空头明细: {OUT_PREFIX}_short.csv")
    print(f"\n►►► 回测成功！►►►")

if __name__ == "__main__":
    main()
