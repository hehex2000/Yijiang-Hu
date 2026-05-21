"""
更新沪深300指数数据（补充2023年数据）
从 AkShare 获取沪深300指数日线数据，追加到现有CSV文件
"""

import sys
import os
import pandas as pd
from loguru import logger

HS300_FILE = os.path.join(os.path.dirname(__file__), 'data', 'hs300_index_daily.csv')


def update_hs300_index():
    """更新沪深300指数数据"""
    try:
        import akshare as ak
    except ImportError:
        logger.error("AkShare not installed. Please install: pip install akshare")
        return False
    
    # 读取现有数据
    if os.path.exists(HS300_FILE):
        df_existing = pd.read_csv(HS300_FILE)
        last_date = df_existing['trade_date'].max()
        logger.info(f"现有数据最后日期: {last_date}")
    else:
        df_existing = pd.DataFrame()
        last_date = 20221230
    
    # 获取2023年数据
    logger.info("正在获取沪深300指数数据（2023年）...")
    
    try:
        # AkShare API: stock_zh_index_daily_em
        # symbol: sh000300 (沪深300指数)
        df = ak.stock_zh_index_daily_em(symbol="sh000300")
        
        if df.empty:
            logger.warning("未获取到数据")
            return False
        
        # 过滤日期范围（只取2023年）
        df = df[df['日期'] >= '2023-01-01']
        df = df[df['日期'] <= '2023-12-31']
        
        if df.empty:
            logger.warning("2023年无数据")
            return False
        
        # 转换日期格式为 int (YYYYMMDD)
        df['trade_date'] = pd.to_datetime(df['日期']).dt.strftime('%Y%m%d').astype(int)
        
        # 重命名列
        df = df.rename(columns={
            '日期': 'date',
            '开盘': 'open',
            '最高': 'high',
            '最低': 'low',
            '收盘': 'close',
            '成交量': 'vol',
            '成交额': 'amount'
        })
        
        # 计算涨跌额和涨跌幅
        df['pre_close'] = df['close'].shift(-1)
        df['change'] = df['close'] - df['pre_close']
        df['pct_chg'] = df['change'] / df['pre_close'] * 100
        
        # 添加 ts_code 列
        df['ts_code'] = '000300.SH'
        
        # 选择需要的列
        df_out = df[['ts_code', 'trade_date', 'close', 'open', 'high', 'low', 'pre_close', 'change', 'pct_chg', 'vol', 'amount']].copy()
        
        # 合并数据
        if df_existing.empty:
            df_combined = df_out
        else:
            df_combined = pd.concat([df_existing, df_out], ignore_index=True)
        
        # 去重
        df_combined = df_combined.drop_duplicates(subset=['trade_date'], keep='last')
        df_combined = df_combined.sort_values('trade_date').reset_index(drop=True)
        
        # 保存
        df_combined.to_csv(HS300_FILE, index=False, encoding='utf-8-sig')
        
        logger.info(f"✓ 数据已更新: {len(df_combined)} 条记录")
        logger.info(f"  日期范围: {df_combined['trade_date'].min()} 至 {df_combined['trade_date'].max()}")
        
        return True
        
    except Exception as e:
        logger.error(f"获取沪深300指数数据失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    update_hs300_index()
