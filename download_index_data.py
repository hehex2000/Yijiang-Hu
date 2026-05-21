#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
指数数据下载脚本
从Tushare下载指数日线数据并保存到本地SQLite数据库
支持：沪深300(000300.SH)、中证500(000905.SH)、中证1000(000852.SH)等
"""

import sys
import os
from datetime import datetime, timedelta
from loguru import logger

# 添加Tushare-Downloader配置路径
sys.path.insert(0, 'C:/Users/99395/WorkBuddy/Tushare-Downloader')
import tushare as ts
import pandas as pd
import sqlite3

# 数据库路径
DB_PATH = 'D:/tu-shareData/astock_daily.db'

# Tushare token
try:
    from config import TUSHARE_TOKEN
    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()
    logger.info("✓ 成功读取Tushare token")
except Exception as e:
    logger.error(f"✗ 读取Tushare token失败: {e}")
    sys.exit(1)


def init_db():
    """初始化数据库，创建指数表"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 创建指数日线表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS index_daily (
            ts_code      TEXT,
            trade_date   TEXT,
            close        REAL,
            open         REAL,
            high         REAL,
            low          REAL,
            pre_close    REAL,
            change       REAL,
            pct_chg      REAL,
            vol          REAL,
            amount       REAL,
            PRIMARY KEY (ts_code, trade_date)
        )
    ''')

    conn.commit()
    conn.close()
    logger.info("✓ 数据库表已创建/确认: index_daily")


def get_latest_date(ts_code):
    """获取数据库中该指数最新的交易日期"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT MAX(trade_date) FROM index_daily WHERE ts_code = ?",
        (ts_code,)
    )
    result = cursor.fetchone()[0]

    conn.close()

    if result:
        logger.info(f"  数据库中最新日期: {result}")
        return result
    else:
        logger.info(f"  数据库中没有该指数数据")
        return None


def download_index_data(ts_code, start_date='20050101', end_date=None):
    """
    从Tushare下载指数日线数据

    Parameters:
    - ts_code: 指数代码，如 000300.SH (沪深300)
    - start_date: 开始日期，格式 YYYYMMDD
    - end_date: 结束日期，格式 YYYYMMDD，默认今天
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')

    logger.info(f"正在下载指数 {ts_code} 数据...")
    logger.info(f"  时间范围: {start_date} ~ {end_date}")

    try:
        # 使用Tushare pro API下载指数日线
        df = pro.index_daily(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            fields='ts_code,trade_date,close,open,high,low,pre_close,change,pct_chg,vol,amount'
        )

        if df.empty:
            logger.warning(f"  未获取到数据，请检查指数代码和时间范围")
            return None

        logger.info(f"  ✓ 获取到 {len(df)} 条数据")
        return df

    except Exception as e:
        logger.error(f"  下载失败: {e}")
        return None


def save_to_db(df):
    """保存数据到数据库（去重）"""
    conn = sqlite3.connect(DB_PATH)

    # 使用 INSERT OR IGNORE 避免重复插入
    count_before = len(df)

    try:
        df.to_sql('index_daily', conn, if_exists='append', index=False)

        # 检查实际插入了多少行
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM index_daily")
        total_count = cursor.fetchone()[0]

        conn.close()

        logger.info(f"  ✓ 成功保存数据到数据库 (总计 {total_count} 条)")

    except Exception as e:
        logger.error(f"  保存失败: {e}")
        conn.close()


def download_index(ts_code, index_name=None):
    """下载单个指数的完整流程"""
    if index_name is None:
        index_name = ts_code

    logger.info("=" * 70)
    logger.info(f"开始处理: {index_name} ({ts_code})")
    logger.info("=" * 70)

    # 检查数据库中已有数据
    latest_date = get_latest_date(ts_code)

    if latest_date:
        # 从最新日期的下一天开始下载
        start_date = (datetime.strptime(latest_date, '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')
        logger.info(f"  增量下载，从 {start_date} 开始")
    else:
        # 首次下载，从2005年开始
        start_date = '20050101'
        logger.info(f"  首次下载，从 {start_date} 开始")

    # 下载数据
    df = download_index_data(ts_code, start_date=start_date)

    if df is not None and not df.empty:
        # 保存到数据库
        save_to_db(df)
    else:
        logger.info(f"  没有新数据需要下载")

    logger.info("")


def main():
    """主函数"""
    logger.info("=" * 70)
    logger.info("指数数据下载工具")
    logger.info("=" * 70)

    # 初始化数据库
    init_db()

    # 定义要下载的指数
    indices = [
        ('000300.SH', '沪深300'),
        ('000905.SH', '中证500'),
        ('000852.SH', '中证1000'),
        ('000001.SH', '上证指数'),
        ('399001.SZ', '深证成指'),
        ('399006.SZ', '创业板指'),
    ]

    # 下载所有指数
    for ts_code, index_name in indices:
        download_index(ts_code, index_name)

    logger.info("=" * 70)
    logger.info("✓ 全部完成！")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
