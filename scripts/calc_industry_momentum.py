"""
计算行业动量因子并保存到数据库
数据源：daily 表（日线数据） + stock_basic 表（行业分类）
目标表：industry_momentum（新建）
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

DB_PATH = r"D:\tu-shareData\astock_daily.db"


def init_industry_momentum_table(db_path):
    """初始化行业动量因子表"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS industry_momentum (
            ts_code TEXT,
            trade_date TEXT,
            industry_momentum REAL,
            industry_momentum_z REAL,
            PRIMARY KEY (ts_code, trade_date)
        )
    """)
    
    conn.commit()
    conn.close()
    print("✅ industry_momentum 表初始化完成")


def calc_industry_momentum_for_date(db_path, trade_date, lookback_months=6):
    """
    计算指定交易日期的行业动量因子
    
    Args:
        db_path: 数据库路径
        trade_date: 交易日期（YYYYMMDD格式）
        lookback_months: 回看月数（默认6个月）
    
    Returns:
        DataFrame: 包含 ts_code, industry_momentum, industry_momentum_z
    """
    conn = sqlite3.connect(db_path)
    
    # 计算起始日期
    trade_date_dt = datetime.strptime(trade_date, "%Y%m%d")
    start_date_dt = trade_date_dt - pd.DateOffset(months=lookback_months)
    start_date = start_date_dt.strftime("%Y%m%d")
    
    # 1. 获取行业分类（从 stock_basic 表）
    industry_df = pd.read_sql("""
        SELECT ts_code, industry
        FROM stock_basic
        WHERE industry IS NOT NULL AND industry != ''
    """, conn)
    
    if industry_df.empty:
        print(f"  ⚠️ {trade_date} 无行业分类数据")
        conn.close()
        return None
    
    # 2. 获取日线数据（回看期内）
    daily_df = pd.read_sql("""
        SELECT ts_code, trade_date, close
        FROM daily
        WHERE trade_date >= ? AND trade_date <= ?
        ORDER BY ts_code, trade_date
    """, conn, params=(start_date, trade_date))
    
    conn.close()
    
    if daily_df.empty:
        print(f"  ⚠️ {trade_date} 无日线数据")
        return None
    
    # 3. 计算每个股票的收益率
    daily_df['trade_date'] = pd.to_datetime(daily_df['trade_date'], format='%Y%m%d')
    daily_df = daily_df.sort_values(['ts_code', 'trade_date'])
    
    def calc_return(group):
        prices = group['close'].values
        if len(prices) < 2:
            return np.nan
        return (prices[-1] / prices[0]) - 1
    
    stock_returns = daily_df.groupby('ts_code').apply(calc_return).reset_index()
    stock_returns.columns = ['ts_code', 'return']
    
    # 4. 合并行业分类
    merged = stock_returns.merge(industry_df, on='ts_code', how='left')
    merged['industry'] = merged['industry'].fillna('未知')
    
    # 5. 按行业计算平均收益率（行业动量）
    industry_momentum = merged.groupby('industry')['return'].mean().reset_index()
    industry_momentum.columns = ['industry', 'industry_momentum']
    
    # 6. 合并回股票层面
    result = merged[['ts_code', 'industry']].merge(industry_momentum, on='industry', how='left')
    result = result[['ts_code', 'industry_momentum']]
    
    # 7. 标准化（z-score）
    result['industry_momentum_z'] = (result['industry_momentum'] - result['industry_momentum'].mean()) / result['industry_momentum'].std()
    
    # 8. 添加交易日期
    result['trade_date'] = trade_date
    
    return result[['ts_code', 'trade_date', 'industry_momentum', 'industry_momentum_z']]


def save_industry_momentum_to_db(db_path, df):
    """保存行业动量因子到数据库"""
    if df is None or df.empty:
        return 0
    
    # 确保表存在
    init_industry_momentum_table(db_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    count = 0
    for _, row in df.iterrows():
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO industry_momentum
                (ts_code, trade_date, industry_momentum, industry_momentum_z)
                VALUES (?, ?, ?, ?)
            """, (
                row['ts_code'],
                row['trade_date'],
                row['industry_momentum'],
                row['industry_momentum_z']
            ))
            count += 1
        except Exception as e:
            print(f"❌ 插入数据失败: {e}")
            continue
    
    conn.commit()
    conn.close()
    return count


def batch_calc_industry_momentum(db_path, start_date="20200101", end_date="20231231"):
    """
    批量计算行业动量因子（每个交易日）
    
    Args:
        db_path: 数据库路径
        start_date: 开始日期
        end_date: 结束日期
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 获取所有交易日
    cursor.execute("""
        SELECT DISTINCT trade_date
        FROM daily
        WHERE trade_date >= ? AND trade_date <= ?
        ORDER BY trade_date
    """, (start_date, end_date))
    
    trade_dates = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    print(f"📊 共有 {len(trade_dates)} 个交易日需要计算")
    
    # 初始化数据库表
    init_industry_momentum_table(db_path)
    
    # 批量计算
    total_count = 0
    for i, trade_date in enumerate(trade_dates):
        print(f"\n📅 计算 {trade_date} ({i+1}/{len(trade_dates)})...")
        
        result = calc_industry_momentum_for_date(db_path, trade_date, lookback_months=6)
        if result is not None:
            count = save_industry_momentum_to_db(db_path, result)
            total_count += count
            print(f"  ✅ 保存 {count} 条")
        else:
            print(f"  ⚠️ 计算失败")
        
        # 每100个交易日暂停一下（避免数据库锁）
        if (i + 1) % 100 == 0:
            time.sleep(1)
    
    print(f"\n🎉 批量计算完成！共保存 {total_count} 条")


def main():
    import sys
    
    print("=" * 60)
    print("计算行业动量因子并保存到数据库")
    print("=" * 60)
    
    # 解析命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == "--batch":
            # 批量计算模式
            batch_calc_industry_momentum(DB_PATH, start_date="20200101", end_date="20231231")
        else:
            # 计算指定日期
            trade_date = sys.argv[1]
            print(f"\n🧪 计算 {trade_date} 的行业动量因子...")
            result = calc_industry_momentum_for_date(DB_PATH, trade_date, lookback_months=6)
            
            if result is not None:
                count = save_industry_momentum_to_db(DB_PATH, result)
                print(f"  ✅ 保存 {count} 条")
            else:
                print(f"  ❌ 计算失败")
    else:
        # 默认：只计算测试日期
        test_date = "20230101"
        print(f"\n🧪 测试计算 {test_date} 的行业动量因子...")
        result = calc_industry_momentum_for_date(DB_PATH, test_date, lookback_months=6)
        
        if result is not None:
            print(f"  ✅ 计算完成，共 {len(result)} 只股票")
            print("\n📊 行业动量排名（TOP 10）：")
            top10 = result.sort_values('industry_momentum', ascending=False).head(10)
            for _, row in top10.iterrows():
                print(f"  {row['ts_code']:10s}  动量: {row['industry_momentum']:+.2%}  Z-score: {row['industry_momentum_z']:+.2f}")
        else:
            print(f"  ❌ 计算失败")


if __name__ == "__main__":
    main()
