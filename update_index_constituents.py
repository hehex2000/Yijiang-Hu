#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从Tushare下载指数成分股和板块信息，更新到本地数据库
支持参数化运行

数据源: Tushare API
本地数据库: D:\tu-shareData\astock_daily.db

支持的指数代码：
- 沪深300: 000300.SH
- 中证500: 000905.SH
- 中证1000: 000852.SH

使用方法：
  python update_index_constituents.py --index 000300.SH
  python update_index_constituents.py --index all
  python update_index_constituents.py --sector
"""

import sqlite3
import pandas as pd
import argparse
import os
import sys
from datetime import datetime

# ========== 命令行参数 ==========
parser = argparse.ArgumentParser(description='下载指数成分股和板块信息')
parser.add_argument('--index', type=str, default=None, 
                    help='指数代码（000300.SH=沪深300, 000905.SH=中证500, 000852.SH=中证1000），或 "all" 下载全部')
parser.add_argument('--sector', action='store_true', 
                    help='下载行业板块分类信息')
parser.add_argument('--trade-date', type=str, default=None,
                    help='指定交易日期（格式:YYYYMMDD），默认使用最新')
args = parser.parse_args()

# ========== 配置 ==========
# 数据库路径
DB_PATH = r"D:\tu-shareData\astock_daily.db"

# Tushare token（从Tushare-Downloader配置文件中读取）
TUSHARE_TOKEN = None

# 指数列表
INDEX_LIST = {
    '000300.SH': '沪深300',
    '000905.SH': '中证500',
    '000852.SH': '中证1000',
    '000016.SH': '上证50',
    '399006.SZ': '创业板指',
    '000688.SH': '科创50',
    '000906.SH': '中证800',
    '000985.SH': '中证全指',
}


def get_tushare_token():
    """从Tushare-Downloader配置文件读取token"""
    global TUSHARE_TOKEN
    
    config_path = r"C:\Users\99395\WorkBuddy\Tushare-Downloader\config.py"
    
    if not os.path.exists(config_path):
        raise Exception(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('TUSHARE_TOKEN'):
                TUSHARE_TOKEN = line.split('=')[1].strip().strip('"').strip("'")
                break
    
    if TUSHARE_TOKEN is None:
        raise Exception("配置文件中未找到 TUSHARE_TOKEN")
    
    print(f"✓ 成功读取Tushare token")
    return TUSHARE_TOKEN


def get_db_connection():
    """获取数据库连接"""
    if not os.path.exists(DB_PATH):
        raise Exception(f"数据库文件不存在: {DB_PATH}")
    return sqlite3.connect(DB_PATH)


def create_tables(conn):
    """创建指数成分股和板块信息表"""
    cursor = conn.cursor()
    
    # 1. 指数成分股表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS index_constituent (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_code TEXT NOT NULL,
            index_code TEXT NOT NULL,
            index_name TEXT,
            trade_date TEXT NOT NULL,
            weight REAL,
            create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ts_code, index_code, trade_date)
        )
    """)
    
    # 2. 行业板块分类表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sector_classify (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_code TEXT NOT NULL,
            sector_code TEXT NOT NULL,
            sector_name TEXT,
            trade_date TEXT NOT NULL,
            create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ts_code, sector_code, trade_date)
        )
    """)
    
    # 3. 指数基本信息表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS index_info (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            index_code TEXT NOT NULL UNIQUE,
            index_name TEXT,
            publish_date TEXT,
            create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    print("✓ 数据库表已创建/确认")
    cursor.close()


def download_index_constituent(pro, index_code: str, trade_date: str = None):
    """
    下载单个指数的成分股
    优先使用Tushare API，失败则使用AkShare（免费）
    """
    index_name = INDEX_LIST.get(index_code, index_code)
    
    print(f"\n正在下载 {index_name} ({index_code}) 成分股...")
    
    df = None
    
    # 方法1：使用Tushare API
    try:
        print(f"  尝试 Tushare API...")
        
        # 先尝试 index_member
        df = pro.index_member(index_code=index_code)
        
        if df is None or len(df) == 0:
            # 再尝试 index_weight
            if trade_date is None:
                df_date = pro.trade_cal(exchange='SSE', start_date='20240101', end_date=datetime.now().strftime('%Y%m%d'))
                df_date = df_date[df_date['is_open'] == 1]
                if len(df_date) > 0:
                    trade_date = df_date.iloc[-1]['cal_date']
                else:
                    trade_date = datetime.now().strftime('%Y%m%d')
            
            df = pro.index_weight(index_code=index_code, trade_date=trade_date)
        
        if df is not None and len(df) > 0:
            print(f"  ✓ 通过 Tushare API 获取到 {len(df)} 只成分股")
            df['index_name'] = index_name
            if 'trade_date' not in df.columns and trade_date:
                df['trade_date'] = trade_date
            return df
    except Exception as e:
        print(f"  Tushare API 失败: {e}")
    
    # 方法2：使用AkShare（免费，无需积分）
    try:
        print(f"  尝试 AkShare API...")
        import akshare as ak
        
        # 根据指数代码选择AkShare接口
        # 注意：AkShare主要支持中证指数公司的指数
        ak_symbol = None
        if index_code == '000300.SH':
            ak_symbol = "000300"  # 沪深300
        elif index_code == '000905.SH':
            ak_symbol = "000905"  # 中证500
        elif index_code == '000852.SH':
            ak_symbol = "000852"  # 中证1000
        elif index_code == '000016.SH':
            ak_symbol = "000016"  # 上证50
        elif index_code == '000906.SH':
            ak_symbol = "000906"  # 中证800
        elif index_code == '000985.SH':
            ak_symbol = "000985"  # 中证全指
        
        if ak_symbol:
            df = ak.index_stock_cons_csindex(symbol=ak_symbol)
        
        if df is not None and len(df) > 0:
            print(f"  ✓ AkShare 返回 {len(df)} 行，原始列名: {list(df.columns)}")
            
            # 列名映射（AkShare中文列名 → 标准列名）
            new_columns = []
            for col in df.columns:
                if '代码' in col and '指数' not in col:  # 成分券代码
                    new_columns.append('con_code')
                elif '名称' in col and '指数' not in col:  # 成分券名称
                    new_columns.append('stock_name')
                else:
                    # 保留其他列（如日期、交易所等）
                    new_columns.append(col)
            
            df.columns = new_columns
            
            # 只保留需要的列
            df = df[['con_code', 'stock_name']].copy()
            df['index_code'] = index_code
            df['index_name'] = index_name
            df['trade_date'] = datetime.now().strftime('%Y%m%d')
            df['weight'] = None
            
            print(f"  ✓ 通过 AkShare API 获取到 {len(df)} 只成分股")
            print(f"  处理后列名: {list(df.columns)}")
            return df
        else:
            print(f"  AkShare 未返回数据（可能不支持该指数: {index_code}）")
    except Exception as e:
        print(f"  AkShare API 失败: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"  ❌ 未获取到数据（Tushare 和 AkShare 都失败）")
    return None


def save_index_constituent(conn, df: pd.DataFrame):
    """保存指数成分股到数据库"""
    cursor = conn.cursor()
    
    # 检查DataFrame的列名
    print(f"  DataFrame列名: {list(df.columns)}")
    
    # 确定列名（兼容不同API返回格式）
    code_col = 'con_code' if 'con_code' in df.columns else 'ts_code'
    index_code_col = 'index_code' if 'index_code' in df.columns else 'ts_code'
    
    # 插入数据（忽略重复）
    insert_sql = """
        INSERT OR IGNORE INTO index_constituent 
        (ts_code, index_code, index_name, trade_date, weight)
        VALUES (?, ?, ?, ?, ?)
    """
    
    count = 0
    for _, row in df.iterrows():
        try:
            cursor.execute(insert_sql, (
                row[code_col],
                row[index_code_col],
                row['index_name'] if 'index_name' in row else None,
                row['trade_date'],
                row['weight'] if 'weight' in row else None
            ))
            count += 1
        except Exception as e:
            print(f"  插入失败 {row[code_col]}: {e}")
    
    conn.commit()
    cursor.close()
    
    print(f"  ✓ 成功保存 {count} 条成分股数据")
    return count


def download_sector_classify(pro, trade_date: str = None):
    """
    下载行业板块分类
    """
    print(f"\n正在下载行业板块分类...")
    
    try:
        # 使用 index_classify 获取行业分类
        # market: 市场代码 (SSE=上交所, SZSE=深交所)
        df = pro.index_classify(market='SSE', src='SW2014')  # 申万2014版
        
        if df is None or len(df) == 0:
            print(f"  ❌ 未获取到行业分类数据")
            return None
        
        print(f"  ✓ 获取到 {len(df)} 条行业分类数据")
        
        return df
        
    except Exception as e:
        print(f"  ❌ 下载失败: {e}")
        return None


def save_sector_classify(conn, df: pd.DataFrame, trade_date: str):
    """保存行业板块分类到数据库"""
    cursor = conn.cursor()
    
    # 检查DataFrame列名
    print(f"  板块DataFrame列名: {list(df.columns)}")
    
    # 确定列名
    ts_code_col = 'ts_code' if 'ts_code' in df.columns else df.columns[0]
    sector_code_col = 'index_code' if 'index_code' in df.columns else df.columns[1]
    sector_name_col = 'industry_name' if 'industry_name' in df.columns else df.columns[2]
    
    # 插入数据（忽略重复）
    insert_sql = """
        INSERT OR IGNORE INTO sector_classify 
        (ts_code, sector_code, sector_name, trade_date)
        VALUES (?, ?, ?, ?)
    """
    
    count = 0
    for _, row in df.iterrows():
        try:
            cursor.execute(insert_sql, (
                row[ts_code_col],
                row[sector_code_col],
                row[sector_name_col],
                trade_date
            ))
            count += 1
        except Exception as e:
            print(f"  插入失败 {row[ts_code_col]}: {e}")
    
    conn.commit()
    cursor.close()
    
    print(f"  ✓ 成功保存 {count} 条板块分类数据")
    return count


def main():
    """主函数"""
    print("=" * 70)
    print("指数成分股和板块信息下载工具")
    print("=" * 70)
    
    # 1. 获取Tushare token
    try:
        token = get_tushare_token()
    except Exception as e:
        print(f"❌ {e}")
        return
    
    # 2. 初始化Tushare API
    import tushare as ts
    ts.set_token(token)
    pro = ts.pro_api()
    
    # 3. 连接数据库
    try:
        conn = get_db_connection()
        print(f"✓ 数据库连接成功: {DB_PATH}")
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")
        return
    
    # 4. 创建表（如果不存在）
    create_tables(conn)
    
    # 5. 处理指数成分股
    if args.index:
        if args.index == 'all':
            # 下载所有指数
            for index_code in INDEX_LIST.keys():
                df = download_index_constituent(pro, index_code, args.trade_date)
                if df is not None:
                    save_index_constituent(conn, df)
        else:
            # 下载指定指数
            if args.index not in INDEX_LIST:
                print(f"❌ 不支持的指数代码: {args.index}")
                print(f"   支持的指数: {list(INDEX_LIST.keys())}")
                conn.close()
                return
            
            df = download_index_constituent(pro, args.index, args.trade_date)
            if df is not None:
                save_index_constituent(conn, df)
    
    # 6. 处理板块分类
    if args.sector:
        trade_date = args.trade_date if args.trade_date else datetime.now().strftime('%Y%m%d')
        df = download_sector_classify(pro, trade_date)
        if df is not None:
            save_sector_classify(conn, df, trade_date)
    
    # 7. 如果没有指定任何操作，显示帮助
    if not args.index and not args.sector:
        print("\n提示: 请使用 --index 或 --sector 参数")
        print("示例:")
        print("  python update_index_constituents.py --index 000300.SH")
        print("  python update_index_constituents.py --index all")
        print("  python update_index_constituents.py --sector")
        parser.print_help()
    
    conn.close()
    print("\n✓ 全部完成！")


if __name__ == '__main__':
    main()
