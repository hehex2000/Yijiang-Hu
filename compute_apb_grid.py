#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
APB 因子网格对比脚本
计算多种"等全均价"定义下的 APB 因子，并比较其统计特征

作用：尝试 4 种 TWAP 定义，找出哪个最符合视频描述的因子表现
"""

import sqlite3
import pandas as pd
import numpy as np

# =========================================================================== #
# 参数配置
# =========================================================================== #

DB_PATH = r"d:\tu-sharedata\astock_daily.db"

# "等全均价" 的所有方法
TWAP_METHODS = {
    "method1_ohlc4": lambda x: (x['open'] + x['high'] + x['low'] + x['close']) / 4,
    "method2_hlc3":  lambda x: (x['high'] + x['low'] + x['close']) / 3,
    "method3_oc2":   lambda x: (x['open'] + x['close']) / 2,
    "method4_sqrt_hl": lambda x: np.sqrt(x['high'] * x['low']),
}

# 时间范围（取回测期的样本来比较）
SAMPLE_START = "2017-12-01"
ROLLING_WINDOW = 5

# 输出前缀
OUTPUT_PREFIX = r"d:\tu-sharedata\apb_comparison"

# 延迟打印
import sys
sys.setrecursionlimit(100000)

# =========================================================================== #
# 函数
# =========================================================================== #

def load_sample_data():
    """加载样本数据（前 500 个交易日）"""
    query = f"""
    SELECT ts_code, trade_date, open, high, low, close, vol, amount
    FROM daily
    WHERE trade_date >= ? AND trade_date <= ?
    ORDER BY ts_code, trade_date
    """
    start = (pd.to_datetime(SAMPLE_START) - pd.Timedelta(days=ROLLING_WINDOW*2)).strftime("%Y%m%d")
    end = (pd.to_datetime(SAMPLE_START) + pd.Timedelta(days=500)).strftime("%Y%m%d")
    
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(query, conn, params=(start, end))
    
    df['trade_date'] = pd.to_datetime(df['trade_date'], format="%Y%m%d")
    # 剔除量价为零（部分停牌）
    df = df[(df['vol'] > 0) & (df['amount'] > 0)]
    
    return df

def calc_apb_all_methods(df, window=5):
    """用所有方法计算 APB5D"""
    print(f"计算 4 种方法的 APB 因子（{len(df['ts_code'].unique())} 只股票）...")
    
    # VWAP
    df['vwap'] = df['amount'] / (df['vol'] * 10)
    
    results = []
    
    for method_name, twap_func in TWAP_METHODS.items():
        print(f"  正在计算 {method_name}...")
        
        # TWAP
        twap_col = f'twap_{method_name}'
        df[twap_col] = twap_func(df)
        
        # APB
        apb_col = f'apb_{method_name}'
        df[apb_col] = (df['vwap'] - df[twap_col]) / df[twap_col]
        
        # APB5D
        apb5d_col = f'apb5d_{method_name}'
        df[apb5d_col] = df.groupby('ts_code', group_keys=False)[apb_col].rolling(
            window=window,
            min_periods=3
        ).mean().values
        
        # 取出月末值（每月最后一天）
        df['year'] = df['trade_date'].dt.year
        df['month'] = df['trade_date'].dt.month
        idx = df.groupby(['ts_code', 'year', 'month'])['trade_date'].transform(max) == df['trade_date']
        monthly_vals = df[idx][['ts_code', 'year', 'month', apb5d_col]]
        monthly_vals.rename({apb5d_col: 'apb5d'}, axis=1, inplace=True)
        
        # 统计
        nonnan = monthly_vals['apb5d'].dropna()
        stat = {
            'method': method_name,
            'N': len(monthly_vals),
            'NaN': len(monthly_vals) - len(nonnan),
            'mean': nonnan.mean(),
            'std': nonnan.std(),
            'min': nonnan.min(),
            'max': nonnan.max(),
            'skew': nonnan.skew(),
            'kurt': nonnan.kurt(),
            'data': monthly_vals
        }
        results.append(stat)
    
    return results

def compare_methods(results):
    """输出方法对比"""
    print("\n► APB5D 方法对比（月度因子样本统计）\n" + "="*70)
    print(f"{'Method':<15} {'N':>8} {'NaN%':>6} {'mean':>8} {'std':>8} {'skew':>8} {'kurt':>8}")
    print("-"*70)
    
    for s in results:
        print(f"{s['method']:<15} {s['N']:8} {s['NaN']/s['N']*100:5.1f}% {s['mean']:8.5f} {s['std']:8.5f} {s['skew']:8.3f} {s['kurt']:8.2f}")
    
    # 选标准差最大、偏度接近 0、峰度合理（< 10）的方法作为候选
    # 标准差大通常代表区分度高
    std_val = [s['std'] for s in results]
    best_idx = np.argmax(std_val)
    print(f"\n► 推荐方法：{results[best_idx]['method']}")
    print(f"   理由：标准差最大（{results[best_idx]['std']:.5f}），因子区分度高")
    
    return results[best_idx]

def export_results(results, prefix):
    """导出结果"""
    for s in results:
        out = prefix + f"_{s['method']}_monthly.csv"
        s['data'].to_csv(out, index=False, encoding='utf-8')
        print(f"导出 {s['method']} 月度因子值：{out}")

def main():
    """主流程"""
    print("►►► APB 多方法对比 ►►►")
    
    # 1. 加载样本
    df = load_sample_data()
    print(f"样本：{len(df['ts_code'].unique())} 只股票，{df['trade_date'].nunique()} 天")
    
    # 2. 计算所有方法
    results = calc_apb_all_methods(df)
    
    # 3. 对比
    best = compare_methods(results)
    
    # 4. 导出
    export_results(results, OUTPUT_PREFIX)
    
    print("\n►►► 多方法对比完成！►►►")
    print(f"推荐使用主脚本，设置 PRIMARY_METHOD = '{best['method']}'")

if __name__ == "__main__":
    main()
