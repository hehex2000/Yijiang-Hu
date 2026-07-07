#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
APB (Ask-Pressure-Balance) 因子计算脚本
从日线数据计算 APB 因子，并按月输出因子值

核心公式：APB = (VWAP - TWAP) / TWAP
  其中：VWAP = amount / (vol * 10)
        TWAP = "等全均价" 的近似值（多种定义可选）

输入数据库：d:\tu-sharedata\astock_daily.db
输出文件：apb_factor_monthly.csv (月度因子值，每行一只股票的月度 APB 均值)
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

# =========================================================================== #
# 参数配置
# =========================================================================== #

DB_PATH = r"d:\tu-sharedata\astock_daily.db"
OUTPUT_PATH = r"d:\tu-sharedata\apb_factor_monthly.csv"

# "等全均价" 的定义方式（可自定义）
TWAP_METHODS = {
    # 方法1: (O+H+L+C)/4
    "method1_ohlc4": lambda x: (x['open'] + x['high'] + x['low'] + x['close']) / 4,
    # 方法2: (H+L+C)/3
    "method2_hlc3": lambda x: (x['high'] + x['low'] + x['close']) / 3,
    # 方法3: (O+C)/2
    "method3_oc2": lambda x: (x['open'] + x['close']) / 2,
    # 方法4: sqrt(H*L)  —— 几何平均，偏向中位数
    "method4_sqrt_hl": lambda x: np.sqrt(x['high'] * x['low']),
}

# 选择主计算方法（推荐 method1_ohlc4 作为基线）
PRIMARY_METHOD = "method1_ohlc4"

# 计算 APB5D 的滚动窗口（视频中为 5 日）
ROLLING_WINDOW = 5

# 回测周期（视频中的样本期）
START_YEAR = 2017
END_YEAR = 2022

# 月度因子导出起始日（确保有足够数据生成滚动窗口值）
OUTPUT_START_DATE = f"{START_YEAR}-12-29"
OUTPUT_END_DATE = f"{END_YEAR}-01-07"

# 进度打印间隔
PRINT_EVERY = 10000

# =========================================================================== #
# 主函数
# =========================================================================== #

def connect_db():
    """连接数据库"""
    return sqlite3.connect(DB_PATH)

def load_data_sqlite():
    """
    从数据库加载日线数据
    返回 DataFrame: [ts_code, trade_date, open, high, low, close, vol, amount]
    """
    print("[1/6] 正在从数据库加载日线数据...")
    
    query = """
    SELECT ts_code, trade_date, open, high, low, close, vol, amount
    FROM daily
    WHERE trade_date >= ? AND trade_date <= ?
    ORDER BY trade_date, ts_code
    """
    
    # 扩展时间范围，确保能计算滚动窗口（提前 N 日）
    start_dt = (pd.to_datetime(OUTPUT_START_DATE) - pd.Timedelta(days=ROLLING_WINDOW*2)).strftime("%Y%m%d")
    end_dt = (pd.to_datetime(OUTPUT_END_DATE) + pd.Timedelta(days=10)).strftime("%Y%m%d")
    
    with connect_db() as conn:
        df = pd.read_sql_query(query, conn, params=(start_dt, end_dt))
    
    # 转换日期类型
    df['trade_date'] = pd.to_datetime(df['trade_date'], format="%Y%m%d")
    
    # 检查量价缺失
    missing = df[['open', 'high', 'low', 'close', 'vol', 'amount']].isnull().sum()
    print(f"  数据量: {len(df):,} 行，覆盖 {df['trade_date'].nunique()} 个交易日")
    if missing.sum() > 0:
        print(
            f"  警告: 量价存在缺失 (vol 缺失 {missing['vol']}, amount 缺失 {missing['amount']})",
            "\n  自动填充：0 量填充为 1",
            "\n  自动填充：0 额填充为 close * vol"
        )
        df['vol'] = df['vol'].replace(0, 1) 
        df['amount'] = df['amount'].replace(0, df['close'] * df['vol'])
    
    print("[1/6] 日线数据加载完成.")
    return df

def calc_apb_factor(df, method_name="method1_ohlc4", window=5):
    """
    计算 APB 因子（逐日）
    返回: 与原 df 同形的 DataFrame，新增列 ['vwap', 'twap', 'apb_daily', 'apb5d']
    """
    if method_name not in TWAP_METHODS:
        raise ValueError(f"未知 TWAP 方法: {method_name}")
    
    print(f"[2/6] 正在计算 APB 因子（{method_name}，{window}D 滚动）...")
    
    # VWAP = amount / (vol * 10) ，因 vol 单位是万股，amount 单位是万元
    df['vwap'] = df['amount'] / (df['vol'] * 10)  
    
    # TWAP（"等全均价"）
    twap_func = TWAP_METHODS[method_name]
    df['twap'] = twap_func(df)
    
    # APB 偏差
    df['apb_daily'] = (df['vwap'] - df['twap']) / df['twap']
    
    # APB5D：滚动平均
    # 按股票分组后滚动
    df['apb5d'] = df.groupby('ts_code', group_keys=False)['apb_daily'].rolling(
        window=window,
        min_periods=max(1, window-1)  # 至少 window-1 个数据点
    ).mean().values
    
    print(f"[2/6] APB 因子（{method_name}）计算完成.")
    return df

def group_by_month(df, start_date=None, end_date=None):
    """
    按月度聚合因子值（取月末最后一天的 apb5d 值）
    返回: [ts_code, year, month, apb5d_month]
    """
    print("[3/6] 正在按月聚合因子值（取每月最后一天）...")
    
    # 设置日期边界
    if start_date:
        df = df[df['trade_date'] >= start_date]
    if end_date:
        df = df = df[df['trade_date'] <= end_date]
    
    # 年月列
    df['year'] = df['trade_date'].dt.year
    df['month'] = df['trade_date'].dt.month
    
    # 按月聚合：每个股票每月取最后一天的数据（按 trade_date 取最大）
    idx = df.groupby(['ts_code', 'year', 'month'])['trade_date'].transform(max) == df['trade_date']
    monthly = df[idx][['ts_code', 'year', 'month', 'apb5d']].copy()
    monthly.reset_index(drop=True, inplace=True)
    
    print(f"[3/6] 共生成 {len(monthly)} 条月度因子记录")
    return monthly

def export_monthly_factor(df_monthly, output_path=None):
    """
    导出月度因子值（WIDE 格式，每行一只股票，每列一个月的因子值）
    """
    if not output_path:
        output_path = OUTPUT_PATH
    
    print(f"[4/6] 正在导出月度因子值到 {output_path}...")
    
    # 月度编号（YYYYMM）
    df_monthly['ym'] = df_monthly['year'] * 100 + df_monthly['month']
    
    # Pivot 成宽表（每行一只股票，每列一月）
    pivot = df_monthly.pivot_table(
        index='ts_code',
        columns='ym',
        values='apb5d',
        aggfunc='first'
    ).reset_index()
    
    # 重命名列
    pivot.columns = ['ts_code'] + [f"APB5D_{col}" for col in pivot.columns[1:]]
    pivot.sort_values('ts_code', inplace=True)
    
    # 写入文件
    pivot.to_csv(output_path, index=False, encoding='utf-8')
    print(f"[4/6] 月度因子值已导出，共 {len(pivot)} 只股票，{len(pivot.columns)-1} 个月.")
    
    # 同时保存 LONG 格式（便于后续进一步选股回测）
    output_long = output_path.replace(".csv", "_long.csv")
    df_monthly.to_csv(output_long, index=False, encoding='utf-8')
    print(f"[4/6] 月度因子值（LONG 格式）已保存到 {output_long}")
    
    return pivot

def export_verification_stats(df_monthly, df_raw):
    """
    输出统计报告（用于验证）
    """
    print("[5/6] 正在产出统计报告...")
    
    report = f"""
APB 因子计算结果
================================================================================
数据周期：   {df_raw['trade_date'].min().date()} ~ {df_raw['trade_date'].max().date()}
回测期：     {OUTPUT_START_DATE} ~ {OUTPUT_END_DATE}
因子方法：   {PRIMARY_METHOD}
滚动窗口：   {ROLLING_WINDOW} 日

月度因子数据（{df_monthly.shape[0]} 只股票，{df_monthly['year'].nunique()}年 × {df_monthly['month'].nunique()}月）：
--------------------------------------------------------------------------------
"""
    
    for ym in sorted(df_monthly['year']*100 + df_monthly['month']):
        y, m = divmod(ym, 100)
        vals = df_monthly[(df_monthly['year']==y) & (df_monthly['month']==m)]['apb5d']
        nonnan = vals.dropna()
        report += f"  {y}-{m:02d} : N={len(vals)}, NaN={len(vals)-len(nonnan)}, mean={nonnan.mean():.5f}, std={nonnan.std():.5f}\n"
    
    report += f"\n月度因子取值范围（{len(df_monthly)} 条记录）：\n"
    report += f"  Min  = {df_monthly['apb5d'].min():.5f}\n"
    report += f"  Max  = {df_monthly['apb5d'].max():.5f}\n"
    report += f"  NaN  = {df_monthly['apb5d'].isnull().sum()} / {len(df_monthly)}\n"
    
    report_path = OUTPUT_PATH.replace(".csv", "_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"[5/6] 统计报告已保存到 {report_path}")
    print(report)

def main():
    """主流程"""
    print("►►► APB 因子计算开始 (方法: {}, {}-{}, 滚动{}D) ►►►\n".format(
        PRIMARY_METHOD,
        OUTPUT_START_DATE,
        OUTPUT_END_DATE,
        ROLLING_WINDOW
    ))
    
    # 1. 加载数据
    df_raw = load_data_sqlite()
    
    # 2. 计算因子
    df_factor = calc_apb_factor(df_raw, PRIMARY_METHOD, ROLLING_WINDOW)
    
    # 3. 按月聚合
    df_monthly = group_by_month(
        df_factor,
        start_date=pd.to_datetime(OUTPUT_START_DATE),
        end_date=pd.to_datetime(OUTPUT_END_DATE)
    )
    
    # 4. 导出因子值
    export_monthly_factor(df_monthly)
    
    # 5. 输出统计
    export_verification_stats(df_monthly, df_raw)
    
    print("\n►►► APB 因子计算完成！►►►")
    print(f"最终输出文件：{OUTPUT_PATH}")

if __name__ == "__main__":
    main()
