"""
测试并下载申万一级行业指数数据
支持 Tushare 和 AkShare 两种数据源
"""
import sqlite3
import akshare as ak
import pandas as pd
from loguru import logger
import time

DB_PATH = "D:/tu-shareData/astock_daily.db"

def test_tushare_industry_index():
    """测试 Tushare 是否支持申万行业指数"""
    try:
        import tushare as ts
        
        # 需要先设置 token（从用户配置中读取）
        # ts.set_token('你的token')
        
        # 测试接口
        pro = ts.pro_api()
        
        # 尝试获取申万一级行业指数数据
        # 申万一级行业指数代码格式: 801020.SI
        df = pro.index_daily(
            ts_code='801020.SI',  # 采掘行业
            start_date='20230101',
            end_date='20230131'
        )
        
        if df is not None and len(df) > 0:
            logger.success("✓ Tushare 支持申万行业指数")
            return True, df
        else:
            logger.warning("Tushare 不支持申万行业指数或没有数据")
            return False, None
            
    except Exception as e:
        logger.error(f"Tushare 测试失败: {e}")
        return False, None

def test_akshare_industry_index():
    """测试 AkShare 是否支持申万行业指数"""
    try:
        # AkShare 获取申万行业指数数据
        # 接口: stock_zh_a_hist_indicator
        df = ak.stock_zh_a_hist_indicator(
            symbol="801020",  # 申万一级行业代码（不带.SI后缀）
            period="daily",
            start_date="20230101",
            end_date="20230131"
        )
        
        if df is not None and len(df) > 0:
            logger.success("✓ AkShare 支持申万行业指数")
            return True, df
        else:
            logger.warning("AkShare 不支持申万行业指数或没有数据")
            return False, None
            
    except Exception as e:
        logger.error(f"AkShare 测试失败: {e}")
        return False, None

def get_sw_level1_codes(db_path: str):
    """从数据库获取申万一级行业代码"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT DISTINCT sector_code, sector_name 
        FROM sector_classify 
        WHERE sector_code LIKE '801%.SI'
        ORDER BY sector_code
    """)
    
    results = cursor.fetchall()
    conn.close()
    
    return results

def download_with_akshare(db_path: str, sector_code: str, sector_name: str, 
                          start_date: str = "20200101", end_date: str = "20241231"):
    """使用 AkShare 下载行业指数数据"""
    try:
        # 移除 .SI 后缀
        code = sector_code.replace('.SI', '')
        
        df = ak.stock_zh_a_hist_indicator(
            symbol=code,
            period="daily",
            start_date=start_date,
            end_date=end_date
        )
        
        if df is not None and len(df) > 0:
            # 保存到数据库
            count = save_to_db(db_path, df, sector_code)
            logger.info(f"✓ {sector_code} ({sector_name}): 保存了 {count} 条数据")
            return count
        else:
            logger.warning(f"✗ {sector_code} ({sector_name}): 没有数据")
            return 0
            
    except Exception as e:
        logger.error(f"✗ {sector_code} ({sector_name}) 下载失败: {e}")
        return 0

def save_to_db(db_path: str, df: pd.DataFrame, ts_code: str):
    """保存行业指数数据到数据库"""
    conn = sqlite3.connect(db_path)
    
    # 确保表存在
    conn.execute("""
        CREATE TABLE IF NOT EXISTS industry_index_daily (
            ts_code TEXT,
            trade_date TEXT,
            close REAL,
            open REAL,
            high REAL,
            low REAL,
            pre_close REAL,
            change REAL,
            pct_chg REAL,
            vol REAL,
            amount REAL,
            PRIMARY KEY (ts_code, trade_date)
        )
    """)
    
    # 插入数据
    count = 0
    for _, row in df.iterrows():
        try:
            conn.execute("""
                INSERT OR IGNORE INTO industry_index_daily
                (ts_code, trade_date, close, open, high, low, pre_close, change, pct_chg, vol, amount)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ts_code,
                str(row.get('日期', '')),
                float(row.get('收盘', 0)),
                float(row.get('开盘', 0)),
                float(row.get('最高', 0)),
                float(row.get('最低', 0)),
                float(row.get('前收', 0)),
                float(row.get('涨跌额', 0)),
                float(row.get('涨跌幅', 0)),
                float(row.get('成交量', 0)),
                float(row.get('成交额', 0))
            ))
            count += 1
        except Exception as e:
            logger.debug(f"插入失败: {e}")
    
    conn.commit()
    conn.close()
    
    return count

def main():
    logger.info("开始下载申万一级行业指数数据...")
    
    # 1. 获取行业代码列表
    industries = get_sw_level1_codes(DB_PATH)
    logger.info(f"找到 {len(industries)} 个申万一级行业")
    
    # 2. 测试数据源
    logger.info("测试数据源...")
    tushare_ok, _ = test_tushare_industry_index()
    
    if not tushare_ok:
        logger.info("Tushare 不可用，使用 AkShare...")
    
    # 3. 下载数据
    total_count = 0
    
    for sector_code, sector_name in industries:
        logger.info(f"正在下载: {sector_code} ({sector_name})")
        
        if not tushare_ok:
            # 使用 AkShare
            count = download_with_akshare(DB_PATH, sector_code, sector_name)
        else:
            # 使用 Tushare（需要 token）
            logger.warning("Tushare 需要 token，请先配置")
            break
        
        total_count += count
        
        # 避免触发限流
        time.sleep(0.5)
    
    logger.success(f"下载完成！共保存 {total_count} 条行业指数数据")

if __name__ == "__main__":
    logger.add("download_industry_index.log", rotation="500 MB")
    main()
