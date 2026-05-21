#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
财务指标数据下载脚本
从Tushare下载fina_indicator数据并保存到本地SQLite数据库

数据说明：
- 提供A股上市公司财务指标数据
- 包括ROE、ROA、毛利率、净利率、资产负债率等核心指标
- 按报告期更新（季报、年报）
"""

import sys
import time
from datetime import datetime
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
    """初始化数据库，创建财务指标表"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 创建财务指标表（基于Tushare fina_indicator接口）
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fina_indicator (
            -- 基本信息
            ts_code         TEXT,
            ann_date        TEXT,
            end_date        TEXT,

            -- 每股收益
            eps             REAL,
            dt_eps          REAL,
            diluted2_eps    REAL,

            -- 每股指标
            total_revenue_ps REAL,
            revenue_ps      REAL,
            capital_rese_ps REAL,
            surplus_rese_ps REAL,
            undist_profit_ps REAL,
            bps             REAL,
            ocfps           REAL,
            cfps            REAL,
            fcff_ps         REAL,
            fcfe_ps         REAL,
            retainedps      REAL,
            ebit_ps         REAL,

            -- 盈利能力
            gross_margin    REAL,
            op_income       REAL,
            ebit            REAL,
            ebitda          REAL,
            netprofit_margin REAL,
            grossprofit_margin REAL,

            -- 回报率
            roe             REAL,
            roe_waa         REAL,
            roe_dt          REAL,
            roa             REAL,
            roic            REAL,
            roe_yearly      REAL,
            roa_yearly      REAL,
            npta            REAL,

            -- 偿债能力
            current_ratio   REAL,
            quick_ratio     REAL,
            cash_ratio      REAL,
            debt_to_assets  REAL,
            assets_to_eqt   REAL,

            -- 营运能力
            ar_turn         REAL,
            ca_turn         REAL,
            fa_turn         REAL,
            assets_turn     REAL,

            -- 现金流
            fcff            REAL,
            fcfe            REAL,
            ocf_to_debt     REAL,

            -- 增长率
            op_yoy          REAL,
            ebt_yoy         REAL,
            netprofit_yoy   REAL,
            dt_netprofit_yoy REAL,
            ocf_yoy         REAL,
            roe_yoy         REAL,
            bps_yoy         REAL,
            assets_yoy      REAL,
            eqt_yoy         REAL,

            PRIMARY KEY (ts_code, end_date)
        )
    ''')

    conn.commit()
    conn.close()
    logger.info("✓ 数据库表已创建/确认: fina_indicator")


def get_stock_list():
    """获取数据库中所有股票列表"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT DISTINCT ts_code FROM daily WHERE ts_code IS NOT NULL AND ts_code != '' ORDER BY ts_code")
    stocks = [row[0] for row in cursor.fetchall()]

    conn.close()

    logger.info(f"从数据库获取到 {len(stocks)} 只股票")
    return stocks


def get_downloaded_stocks():
    """获取已经下载过财务数据的股票列表"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT DISTINCT ts_code FROM fina_indicator")
        downloaded = set(row[0] for row in cursor.fetchall())
    except:
        downloaded = set()

    conn.close()

    logger.info(f"已下载财务指标数据的股票: {len(downloaded)} 只")
    return downloaded


def download_and_save_batch(ts_codes, start_date, end_date, batch_size=100):
    """
    分批下载并保存财务指标数据
    每处理完一批就立即保存到数据库，避免超时丢失数据
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')

    logger.info(f"开始下载财务指标数据...")
    logger.info(f"  时间范围: {start_date} ~ {end_date}")
    logger.info(f"  待下载股票: {len(ts_codes)} 只")
    logger.info(f"  每 {batch_size} 只保存一次")

    all_success = 0
    all_failed = 0

    # 分批处理
    for batch_start in range(0, len(ts_codes), batch_size):
        batch_end = min(batch_start + batch_size, len(ts_codes))
        batch = ts_codes[batch_start:batch_end]

        logger.info(f"  处理批次 {batch_start//batch_size + 1}: 股票 {batch_start+1}-{batch_end}/{len(ts_codes)}")

        batch_data = []
        batch_success = 0
        batch_failed = 0

        for i, ts_code in enumerate(batch):
            try:
                # 请求间隔，避免触发限流
                if i > 0:
                    time.sleep(0.35)

                # 不指定fields，获取所有字段
                df = pro.fina_indicator(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date
                )

                if not df.empty:
                    batch_data.append(df)
                    batch_success += 1

            except Exception as e:
                batch_failed += 1
                if batch_failed <= 3:
                    logger.warning(f"    下载 {ts_code} 失败: {e}")

        # 保存本批次数据到数据库
        if batch_data:
            batch_df = pd.concat(batch_data, ignore_index=True)
            save_to_db(batch_df)
            logger.info(f"    ✓ 批次保存成功: {len(batch_df)} 条数据")

        all_success += batch_success
        all_failed += batch_failed

        # 批次间暂停，避免触发限流
        if batch_end < len(ts_codes):
            logger.info(f"    暂停1秒...")
            time.sleep(1.0)

    logger.info(f"✓ 全部批次处理完成")
    logger.info(f"  成功: {all_success} 只股票")
    logger.info(f"  失败: {all_failed} 只股票")


def save_to_db(df):
    """保存数据到数据库（去重）"""
    if df is None or df.empty:
        logger.warning("  没有数据需要保存")
        return

    conn = sqlite3.connect(DB_PATH)

    try:
        # 使用 INSERT OR IGNORE 避免重复插入
        df.to_sql('fina_indicator', conn, if_exists='append', index=False)
        logger.info(f"  ✓ 成功保存 {len(df)} 条数据到数据库")

    except Exception as e:
        logger.error(f"  保存失败: {e}")
        # 尝试逐行插入，跳过重复数据
        cursor = conn.cursor()
        success_count = 0
        for _, row in df.iterrows():
            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO fina_indicator VALUES (
                        ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,
                        ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
                    )
                ''', tuple(row))
                success_count += 1
            except:
                pass

        conn.commit()
        logger.info(f"  ✓ 逐行插入成功 {success_count} 条数据")

    finally:
        conn.close()


def main():
    """主函数"""
    logger.info("=" * 70)
    logger.info("财务指标数据下载工具")
    logger.info("=" * 70)

    # 初始化数据库
    init_db()

    # 获取股票列表
    stock_list = get_stock_list()

    if not stock_list:
        logger.error("未找到股票数据，请先下载股票日线数据")
        return

    # 获取已下载的股票，实现断点续传
    downloaded = get_downloaded_stocks()
    pending_stocks = [s for s in stock_list if s not in downloaded]

    logger.info(f"待下载股票: {len(pending_stocks)} 只")
    logger.info(f"跳过已下载: {len(downloaded)} 只")

    if not pending_stocks:
        logger.info("✓ 所有股票数据已下载完成！")
        return

    # 下载财务指标数据（分批保存）
    start_date = '20150101'  # 从2015年开始
    download_and_save_batch(pending_stocks, start_date=start_date, end_date=None, batch_size=100)

    # 统计数据库中的数据量
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM fina_indicator")
    total_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT ts_code) FROM fina_indicator")
    stock_count = cursor.fetchone()[0]

    conn.close()

    logger.info("=" * 70)
    logger.info(f"✓ 本次下载完成！")
    logger.info(f"  本次获取: {len(df) if df is not None else 0} 条财务指标数据")
    logger.info(f"  总计: {total_count} 条财务指标数据")
    logger.info(f"  覆盖: {stock_count} 只股票")
    logger.info("=" * 70)
    logger.info("提示: 如需继续下载剩余股票，请再次运行本脚本")


if __name__ == '__main__':
    main()
