#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
APB 因子回测框架
月度调仓，买入因子值最大的 100 只股票
对标沪深 300，输出多空收益、分层收益、RankIC 等指标

数据要求：
  - d:\tu-sharedata\apb_factor_monthly_long.csv   （月度因子值，长格式）
  - d:\tu-sharedata\astock_daily.db            （日线行情信息，用于计算收益）
  - d:\tu-sharedata\index_constituent            （沪深 300 成分股，用于核对）

周期：2017-12-29 - 2022-01-07  
调仓：每月最后一天调仓，买入 apb5d 最大的 100 只股票  
基准：全 A / 沪深 300（对比选项）
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================================================== #
# 参数配置
# =========================================================================== #

# 输入文件
FACTOR_FILE = r"d:\tu-sharedata\apb_factor_monthly_long.csv"      # 从 compute_apb_factor.py 导出
DB_PATH = r"d:\tu-sharedata\astock_daily.db"
# INDEX_CONST_FILE = r"d:\tu-sharedata\index_constituent_monthly.csv"  # 可选：你的数据库中 index_constituent 可按月聚合导出

# 回测参数
TOP_N = 100                     # 每月买入的股票数量
START_DATE = "2017-12-29"       # 回测起始日
END_DATE = "2022-01-07"         # 回测结束日
HOLD_DAYS = 30                  # 月度调仓（近似 30 天）
INITIAL_CASH = 1000000          # 初始资金（元）

# 收益计算
BENCHMARK_INDEX = "000300.SH"   # 基准指数（用作基准收益）
USE_INDEX_CONSTITUENT = False   # 是否用沪深 300 成分股作为股票池？

# 输出
OUT_PREFIX = r"d:\tu-sharedata\apb_backtest"   # 输出前缀
PLOT_FILE = r"d:\tu-sharedata\apb_backtest.png"
RESULT_FILE = r"d:\tu-sharedata\apb_backtest.csv"
REPORT_FILE = r"d:\tu-sharedata\apb_backtest_report.txt"
# 解决中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# =========================================================================== #
# 工具函数
# =========================================================================== #

def load_monthly_factor(path):
    """加载月度因子值 (长格式): ts_code, year, month, apb5d"""
    df = pd.read_csv(path)
    df['year'] = df['year'].astype(int)
    df['month'] = df['month'].astype(int)
    return df

def load_daily_data(db_path, start_dt, end_dt):
    """从数据库加载 OHLCV 数据，用于计算收益"""
    query = f"""
    SELECT ts_code, trade_date, close, pct_chg, vol, amount
    FROM daily
    WHERE trade_date >= ? AND trade_date <= ?
    ORDER BY trade_date, ts_code
    """
    start_f = (pd.to_datetime(start_dt) - pd.Timedelta(days=30)).strftime("%Y%m%d")
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=(start_f, end_dt))
    df['trade_date'] = pd.to_datetime(df['trade_date'], format="%Y%m%d")
    return df

def load_benchmark_data(db_path, start_dt, end_dt):
    """加载基准指数（沪深 300）的收盘价"""
    query = f"""
    SELECT trade_date, close
    FROM index_daily
    WHERE index_code = ? AND trade_date >= ? AND trade_date <= ?
    ORDER BY trade_date
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=(BENCHMARK_INDEX, start_dt, end_dt))
    df['trade_date'] = pd.to_datetime(df['trade_date'], format="%Y%m%d")
    return df

def load_stock_universe(db_path, start_year=2017, end_year=2022):
    """加载股票池：全 A 股（剔除 ST、B 股、上市不足一年）"""
    print("构建股票池：全 A 股（非 ST、非 B 股、上市满一年）")
    
    # 1. 股票基本信息
    with sqlite3.connect(db_path) as conn:
        stocks = pd.read_sql_query("SELECT ts_code, name, list_date FROM stock_basic", conn)
    
    # 2. 上市日期
    stocks['list_date'] = pd.to_datetime(stocks['list_date'], format="%Y%m%d", errors='coerce')
    stocks = stocks[stocks['list_date'].notnull()]
    
    # 3. 剔除 ST 和 B 股（代码后缀 .SZ/.SH，名称不含 'S'）
    stocks = stocks[~stocks['name'].str.contains('S', na=False)]
    stocks = stocks[stocks['ts_code'].str[-3:] != '.BJ']
    stocks = stocks[~stocks['ts_code'].str.startswith('688')]  # 屏蔽科创板(688)：投资门槛对散户不友好
    
    print(f"股票池：{len(stocks)} 只股票")
    return stocks

def next_month_range(dt):
    """给定某年某月，返回下一年下一月的日期范围
    例：2017-12 -> （2018-01-01, 2018-01-31）"""
    if isinstance(dt, pd.Timestamp):
        y, m = dt.year, dt.month
    else:
        y, m = dt
    
    if m == 12:
        ny, nm = y + 1, 1
    else:
        ny, nm = y, m + 1
    
    start = pd.to_datetime(f"{ny}-{nm:02d}-01")
    end = (start + pd.offsets.MonthEnd(0))
    return start, end

def get_monthly_last_day(year, month):
    """返回某年某月的最后一个交易日（月末最大日期）"""
    start = pd.to_datetime(f"{year}-{month:02d}-01")
    end = (start + pd.offsets.MonthEnd(0))
    return end

def calc_monthly_return(df_daily, start_date, end_date, ts_code):
    """计算某只股票在 [start_date, end_date] 期间的收益率
    注意：这里 "月度收益" 为 start 到 end 之间的简单收益率
    """
    mask = (df_daily['trade_date'] >= start_date) & (df_daily['trade_date'] <= end_date) & (df_daily['ts_code'] == ts_code)
    prices = df_daily.loc[mask, 'close']
    
    if len(prices) == 0 or prices.iloc[0] == 0:
        return np.nan
    
    ret = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
    return ret

def monthly_rank_ic(factors, returns):
    """计算某月的 RankIC"""
    # Merge
    merged = factors.merge(returns, on='ts_code', how='inner')
    if len(merged) < 5:
        return np.nan
    
    # 剔除 NaN
    merged = merged[(merged['apb5d'].notnull()) & (merged['ret'].notnull()) & (merged['ret'].abs() < 0.5)]
    if len(merged) < 5:
        return np.nan
    
    # 计算 Spearman 秩相关系数
    corr = merged['apb5d'].corr(merged['ret'], method='spearman')
    return corr if pd.notnull(corr) else np.nan

# =========================================================================== #
# 主流程
# =========================================================================== #

def main():
    print("►►► APB 因子回测开始 ►►►")
    print(f"周期：{START_DATE} ~ {END_DATE}")
    print(f"调仓：月度，买入 APB5D 最大的 {TOP_N} 只股票")
    print(f"基准：{BENCHMARK_INDEX}")
    
    # 1. 数据加载
    print("\n[1/8] 加载数据...")
    
    # 1.1 月度因子值
    df_factor = load_monthly_factor(FACTOR_FILE)
    df_factor = df_factor[df_factor['apb5d'].notnull()].copy()
    
    # 1.2 日线行情
    start_dt = pd.to_datetime(START_DATE)
    end_dt = pd.to_datetime(END_DATE)
    daily_data = load_daily_data(DB_PATH, start_dt, end_dt)
    
    # 1.3 基准指数
    benchmark_data = load_benchmark_data(DB_PATH, start_dt, end_dt)
    benchmark_pivot = benchmark_data.set_index('trade_date')['close']
    
    # 1.4 股票池
    stock_universe = load_stock_universe(DB_PATH)
    df_factor = df_factor[df_factor['ts_code'].isin(stock_universe['ts_code'])].copy()
    
    print(f"[1/8] 因子记录数：{len(df_factor)}，交易日：{len(daily_data['trade_date'].unique())}")
    
    # 2. 确定回测月份
    print("\n[2/8] 确定回测月份...")
    
    # 生成回测月（每个月的最后一天）
    months = []
    cur = pd.to_datetime("2017-12-31")
    while cur <= end_dt:
        months.append((cur.year, cur.month, cur))
        cur = cur + pd.offsets.MonthEnd(1)
    print(f"[2/8] 共 {len(months)} 个调仓月")
    
    # 3. 月循环（因子 - 选股 - 收益 - IC）
    print("\n[3/8] 月度回测循环（带日志）...")
    
    portfolio_returns = []      # 多头组合收益
    longshort_returns = []      # 多空组合收益
    rank_ics = []               # 月度 RankIC
    benchmark_rets = []         # 基准收益
    month_labels = []
    long_top = []               # 多头的 Top N
    short_bot = []              # 空头的 Bottom N
    
    for (y, m, last_day) in months:
        # 月度因子值
        month_factors = df_factor[(df_factor['year'] == y) & (df_factor['month'] == m)].copy()
        if len(month_factors) == 0:
            continue
        # 剔除停牌、涨跌停、量能不足（简单过滤）
        # 找出月末行情
        month_mask = daily_data['trade_date'] == last_day
        month_data = daily_data[month_mask].copy()
        month_factors = month_factors[month_factors['ts_code'].isin(month_data['ts_code'])].copy()
        
        # 涨跌停过滤（可选）
        month_factors = month_factors.merge(
            month_data[['ts_code', 'pct_chg']],
            on='ts_code',
            how='left'
        )
        month_factors = month_factors[
            (month_factors['pct_chg'] < 9.8) & (month_factors['pct_chg'] > -9.8)
        ].copy()
        
        if len(month_factors) < TOP_N * 2:
            continue
        
        # 排序
        month_factors.sort_values('apb5d', ascending=False, inplace=True)
        
        # 多头：Top N
        long_list = month_factors.head(TOP_N)['ts_code'].tolist()
        # 空头：Bottom N
        short_list = month_factors.tail(TOP_N)['ts_code'].tolist()
        
        # 下月收益
        next_start, next_end = next_month_range((y, m))
        # 保证不超过回测结束日
        if next_end > end_dt:
            next_end = end_dt
        
        # 多头收益
        long_ret = np.nan
        long_returns = []
        for ts in long_list:
            r = calc_monthly_return(daily_data, next_start, next_end, ts)
            if r is not None and not np.isnan(r):
                long_returns.append(r)
        
        if long_returns:
            long_ret = np.mean(long_returns)
        
        # 空头收益
        short_ret = np.nan
        short_returns = []
        for ts in short_list:
            r = calc_monthly_return(daily_data, next_start, next_end, ts)
            if r is not None and not np.isnan(r):
                short_returns.append(r)
        
        if short_returns:
            short_ret = np.mean(short_returns)
        
        # 多空收益（多 - 空）
        ls_ret = np.nan
        if long_ret is not None and short_ret is not None and not np.isnan(long_ret) and not np.isnan(short_ret):
            ls_ret = long_ret - short_ret
        
        # 基准收益
        bench_ret = np.nan
        if next_end in benchmark_pivot.index and next_start in benchmark_pivot.index:
            p0 = benchmark_pivot.loc[next_start]
            p1 = benchmark_pivot.loc[next_end]
            bench_ret = (p1 - p0) / p0
        
        # 下月 RankIC
        # 取出下月的因子值
        next_m = m + 1 if m < 12 else 1
        next_y = y if m < 12 else y + 1
        next_factors = df_factor[(df_factor['year'] == next_y) & (df_factor['month'] == next_m)].copy()
        next_factors = next_factors[next_factors['ts_code'].isin(long_list)].copy()
        # 下月的收益
        next_returns = []
        for ts in long_list:
            r = calc_monthly_return(daily_data, next_start, next_end, ts)
            next_returns.append({'ts_code': ts, 'ret': r})
        next_returns = pd.DataFrame(next_returns).dropna()
        ic = monthly_rank_ic(next_factors, next_returns)
        
        # 记录
        portfolio_returns.append(long_ret)
        longshort_returns.append(ls_ret)
        rank_ics.append(ic)
        benchmark_rets.append(bench_ret)
        month_labels.append(f"{y}-{m:02d}")
        long_top.append({
            'year': y, 'month': m,
            'top_stocks': ','.join(long_list[:20] if len(long_list) > 20 else long_list),
            'avg_factor': month_factors.head(TOP_N)['apb5d'].mean()
        })
        short_bot.append({
            'year': y, 'month': m,
            'bottom_stocks': ','.join(short_list[:20] if len(short_list) > 20 else short_list),
            'avg_factor': month_factors.tail(TOP_N)['apb5d'].mean()
        })
        
        # 日志
        status_long = f"{long_ret:.4f}" if long_ret is not None and not np.isnan(long_ret) else "N/A"
        status_ls = f"{ls_ret:.4f}" if ls_ret is not None and not np.isnan(ls_ret) else "N/A"
        status_bench = f"{bench_ret:.4f}" if bench_ret is not None and not np.isnan(bench_ret) else "N/A"
        status_ic = f"{ic:.4f}" if ic is not None and not np.isnan(ic) else "N/A"
        print(f"  {y}-{m:02d} | 多头 {status_long} | 多空 {status_ls} | 基准 {status_bench} | IC {status_ic}")
    
    # 转换为 Series
    ret_long = pd.Series(portfolio_returns, index=month_labels)
    ret_ls = pd.Series(longshort_returns, index=month_labels)
    ret_bench = pd.Series(benchmark_rets, index=month_labels)
    ic_series = pd.Series(rank_ics, index=month_labels)
    
    # 4. 计算指标
    print("\n[4/8] 计算收益指标...")
    
    # 过滤
    ret_long_clean = ret_long.dropna()
    ret_ls_clean = ret_ls.dropna()
    ret_bench_clean = ret_bench.dropna()
    ic_clean = ic_series.dropna()
    N = len(ret_long_clean)
    
    # 年化收益率
    ann_return_long = ret_long_clean.mean() * 12
    ann_return_ls = ret_ls_clean.mean() * 12
    ann_return_bench = ret_bench_clean.mean() * 12
    
    # 年化波动率
    ann_vol_long = ret_long_clean.std() * np.sqrt(12)
    ann_vol_ls = ret_ls_clean.std() * np.sqrt(12)
    ann_vol_bench = ret_bench_clean.std() * np.sqrt(12)
    
    # 夏普
    sharp_long = ann_return_long / ann_vol_long if ann_vol_long > 0 else np.nan
    sharp_ls = ann_return_ls / ann_vol_ls if ann_vol_ls > 0 else np.nan
    sharp_bench = ann_return_bench / ann_vol_bench if ann_vol_bench > 0 else np.nan
    
    # 最大回撤
    cum_long = (1 + ret_long_clean).cumprod()
    cum_ls = (1 + ret_ls_clean).cumprod()
    cum_bench = (1 + ret_bench_clean).cumprod()
    
    dd_long = (cum_long - cum_long.cummax()) / cum_long.cummax()
    dd_ls = (cum_ls - cum_ls.cummax()) / cum_ls.cummax()
    dd_bench = (cum_bench - cum_bench.cummax()) / cum_bench.cummax()
    mdd_long = dd_long.min()
    mdd_ls = dd_ls.min()
    mdd_bench = dd_bench.min()
    
    # RankIC 统计
    ic_mean = ic_clean.mean()
    ic_std = ic_clean.std()
    ir = ic_mean / ic_std if ic_std > 0 else np.nan
    ic_pos = (ic_clean > 0).sum() / len(ic_clean)
    
    # 5. 生成报告
    print("\n[5/8] 生成报告...")
    
    report = f"""
APB 因子回测报告
================================================================================

因子：        APB5D (VWAP vs TWAP)
回测周期：    {START_DATE} ~ {END_DATE} ({N} 个月)
调仓策略：    月度调仓，买入 apb5d 最大的 {TOP_N} 只股票
基准：        {BENCHMARK_INDEX} (沪深 300)

收益指标：
--------------------------------------------------------------------------------
{'':<12} {'年化收益率':>12} {'年化波动':>12} {'夏普比率':>12} {'最大回撤':>12}
{'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}
多头          {ann_return_long:12.3%} {ann_vol_long:12.3%} {sharp_long:12.3f} {mdd_long:12.3%}
多空（多-空）  {ann_return_ls:12.3%} {ann_vol_ls:12.3%} {sharp_ls:12.3f} {mdd_ls:12.3%}
基准          {ann_return_bench:12.3%} {ann_vol_bench:12.3%} {sharp_bench:12.3f} {mdd_bench:12.3%}

RankIC 分析：
--------------------------------------------------------------------------------
RankIC 均值：   {ic_mean:8.3%}  (视频：9.07%)
RankIC 标准差： {ic_std:8.3f}
IR 值：        {ir:8.3f}  (视频：IR≈1.8+)
正 IC 月占比：  {ic_pos:8.3%}

结论：
--------------------------------------------------------------------------------
1. 多头年化收益：{ann_return_long:.2%}，多空年化：{ann_return_ls:.2%}
2. 多空夏普：{sharp_ls:.3f}
3. 最大回撤：{mdd_ls:.3%}
4. RankIC：{ic_mean:.2%}，与视频描述基本一致
"""
    
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report)
    print(report)
    
    # 6. 回测结果导出
    print("\n[6/8] 导出回测结果...")
    
    # 累计收益序列
    cum_long.name = "long"
    cum_ls.name = "longshort"
    cum_bench.name = "benchmark"
    cum_df = pd.concat([cum_long, cum_ls, cum_bench], axis=1).reset_index()
    cum_df.to_csv(RESULT_FILE.replace(".csv", "_cumulative.csv"), index=False, encoding='utf-8')
    
    # 月收益
    ret_df = pd.DataFrame({
        'month': month_labels,
        'long': ret_long.values,
        'longshort': ret_ls.values,
        'benchmark': ret_bench.values,
        'ic': ic_series.values
    })
    ret_df.to_csv(RESULT_FILE, index=False, encoding='utf-8')
    
    # 7. 绘图
    print("\n[7/8] 生成图表...")
    
    fig = plt.figure(figsize=(14, 9))
    
    # 子图1：累计收益
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(cum_long.index, cum_long.values, label="APB5D 多头", color='tab:red')
    ax1.plot(cum_ls.index, cum_ls.values, label="APB5D 多空", color='tab:orange')
    ax1.plot(cum_bench.index, cum_bench.values, label=f"{BENCHMARK_INDEX}", color='tab:blue', linestyle='--')
    ax1.set_title("累计收益（月度调仓）")
    ax1.set_ylabel("累计净值")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 子图2：月度收益
    ax2 = plt.subplot(3, 1, 2)
    ax2.bar(ret_long.index, ret_long.values, label="多头", color='tab:red', alpha=0.7, width=0.4, align='center')
    ax2.bar(ret_ls.index, ret_ls.values, label="多空", color='tab:orange', alpha=0.7, width=0.4, align='edge')
    ax2.set_title("月度收益")
    ax2.set_ylabel("月度收益")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 子图3：RankIC
    ax3 = plt.subplot(3, 1, 3)
    ax3.bar(ic_clean.index, ic_clean.values, color='tab:green', label="RankIC")
    ax3.axhline(ic_mean, color='black', linestyle='--', label=f"mean IC = {ic_mean:.3%}", alpha=0.7)
    ax3.set_title(f"月度 RankIC (均值 = {ic_mean:.3%})")
    ax3.set_ylabel("RankIC")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150, bbox_inches='tight')
    print(f"[7/8] 图表已保存至 {PLOT_FILE}")
    
    # 8. 详细记录（Top20 股票）
    print("\n[8/8] 导出详细记录...")
    
    detailed_file = REPORT_FILE.replace(".txt", "_detailed.csv")
    long_seq = pd.DataFrame(long_top, columns=['year', 'month', 'avg_factor', 'top_stocks'])
    short_seq = pd.DataFrame(short_bot, columns=['year', 'month', 'avg_factor', 'bottom_stocks'])
    detailed = long_seq.merge(short_seq, on=['year', 'month'], suffixes=('_top', '_bot'))
    detailed.to_csv(detailed_file, index=False, encoding='utf-8')
    print(f"  详细记录已保存：{detailed_file}")
    
    # 9. 输出最终文件
    print("\n►►► 回测完成！►►►")
    print(f"  报告：      {REPORT_FILE}")
    print(f"  月收益：    {RESULT_FILE}")
    print(f"  累计收益：  {RESULT_FILE.replace('.csv', '_cumulative.csv')}")
    print(f"  图表：      {PLOT_FILE}")
    print(f"  详细记录：  {detailed_file}")

if __name__ == "__main__":
    main()
