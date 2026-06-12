"""
使用 Tushare Pro 下载 2012-2024 年财务数据并写入数据库
Tushare 付费版，稳定可靠
"""

import sqlite3
import tushare as ts
import pandas as pd
import time
from datetime import datetime

# ========== 配置 ==========
DB_PATH = "D:/tu-shareData/astock_daily.db"
TUSHARE_TOKEN = "YOUR_TUSHARE_TOKEN_HERE"  # ← 替换为你的Tushare token
# ==============================


def get_ts_pro():
    """初始化 Tushare Pro API"""
    ts.set_token(TUSHARE_TOKEN)
    return ts.pro_api()


def get_stock_list(db_path=DB_PATH):
    """
    获取所有A股代码列表
    
    Returns:
        list: ts_code 列表 (如 ['000001.SZ', '600000.SH'])
    """
    conn = sqlite3.connect(db_path)
    
    # 尝试从 stock_basic 表获取
    try:
        df = pd.read_sql_query("SELECT ts_code FROM stock_basic ORDER BY ts_code", conn)
        if len(df) > 0:
            conn.close()
            return df['ts_code'].tolist()
    except:
        pass
    
    # 从 daily 表获取
    df = pd.read_sql_query("SELECT DISTINCT ts_code FROM daily WHERE trade_date >= '20120101' LIMIT 5000", conn)
    conn.close()
    
    return df['ts_code'].tolist()


def download_financial_data(ts_pro, ts_code, start_date="20120101", end_date="20241231"):
    """
    使用 Tushare API 下载单只股票的财务数据
    
    Args:
        ts_pro: Tushare Pro API 对象
        ts_code: 股票代码 (000001.SZ)
        start_date: 开始日期 (YYYYMMDD)
        end_date: 结束日期 (YYYYMMDD)
        
    Returns:
        DataFrame: 财务数据，失败返回 None
    """
    try:
        # 调用 Tushare API: fina_indicator
        # 字段说明：https://tusharepro.com/documentation/fia.html
        df = ts_pro.fina_indicator(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            fields="ts_code,end_date,current_ratio,roe,fcff,op_yoy,eps"
        )
        
        if df is not None and len(df) > 0:
            return df
        
    except Exception as e:
        # 限速错误处理
        if "超限" in str(e) or "rate" in str(e).lower():
            print(f"    ⚠️ 限速，等待60秒...")
            time.sleep(60)
            try:
                df = ts_pro.fina_indicator(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date,
                    fields="ts_code,end_date,current_ratio,roe,fcff,op_yoy,eps"
                )
                return df
            except:
                pass
        
    return None


def create_table_if_not_exists(db_path=DB_PATH):
    """创建 fina_indicator 表（如果不存在）"""
    conn = sqlite3.connect(db_path)
    
    create_sql = """
    CREATE TABLE IF NOT EXISTS fina_indicator (
        ts_code TEXT,
        end_date TEXT,
        current_ratio REAL,
        roe REAL,
        fcff REAL,
        op_yoy REAL,
        eps REAL,
        PRIMARY KEY (ts_code, end_date)
    )
    """
    
    conn.execute(create_sql)
    conn.commit()
    conn.close()
    
    print("✓ 数据表已就绪: fina_indicator")


def insert_financial_data(df, db_path=DB_PATH):
    """
    将财务数据写入数据库（INSERT OR REPLACE）
    
    Args:
        df: 财务数据 DataFrame
        db_path: 数据库路径
    """
    if df is None or len(df) == 0:
        return 0
    
    conn = sqlite3.connect(db_path)
    
    # 准备数据
    records = []
    for _, row in df.iterrows():
        record = (
            row['ts_code'],
            row['end_date'],
            row.get('current_ratio', None),
            row.get('roe', None),
            row.get('fcff', None),
            row.get('op_yoy', None),
            row.get('eps', None)
        )
        records.append(record)
    
    # 批量插入（替换重复）
    conn.executemany("""
        INSERT OR REPLACE INTO fina_indicator 
        (ts_code, end_date, current_ratio, roe, fcff, op_yoy, eps)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, records)
    
    conn.commit()
    affected = conn.total_changes
    conn.close()
    
    return len(records)


def main():
    """主函数：批量下载 2012-2024 年财务数据"""
    
    print("=" * 80)
    print("使用 Tushare Pro 下载财务数据 (2012-2024)")
    print("=" * 80)
    
    # 1. 初始化 Tushare Pro
    print("\n[1/5] 初始化 Tushare Pro API...")
    try:
        ts_pro = get_ts_pro()
        print("  ✓ Tushare Pro API 初始化成功")
    except Exception as e:
        print(f"  ✗ 初始化失败: {e}")
        print("  请检查 TUSHARE_TOKEN 是否正确")
        return
    
    # 2. 创建数据表
    print("\n[2/5] 检查数据表...")
    create_table_if_not_exists()
    
    # 3. 获取股票列表
    print("\n[3/5] 获取股票列表...")
    stock_list = get_stock_list()
    print(f"  ✓ 获取到 {len(stock_list)} 只股票")
    
    # 4. 批量下载财务数据
    print("\n[4/5] 开始下载财务数据...")
    print(f"  时间范围: 2012-01-01 至 2024-12-31")
    print(f"  股票数量: {len(stock_list)}")
    print(f"  预计时间: {len(stock_list) * 0.1 / 60:.1f} 分钟")
    print("=" * 80)
    
    success_count = 0
    fail_count = 0
    total_records = 0
    
    for i, ts_code in enumerate(stock_list):
        # 进度显示
        if i % 50 == 0:
            print(f"\n进度: {i}/{len(stock_list)} ({i/len(stock_list)*100:.1f}%)")
            print(f"  成功: {success_count}, 失败: {fail_count}, 总记录: {total_records}")
        
        # 下载数据
        df = download_financial_data(ts_pro, ts_code, "20120101", "20241231")
        
        if df is not None and len(df) > 0:
            # 写入数据库
            affected = insert_financial_data(df)
            success_count += 1
            total_records += len(df)
            
            if i % 50 == 0:
                print(f"  ✓ {ts_code}: {len(df)} 条记录")
        else:
            fail_count += 1
            
            if i % 50 == 0:
                print(f"  ✗ {ts_code}: 无数据")
        
        # 限速：Tushare 有偿用户一般 200次/分钟
        time.sleep(0.1)
    
    print("\n" + "=" * 80)
    print("[5/5] 下载完成！")
    print("=" * 80)
    print(f"  成功: {success_count} 只股票")
    print(f"  失败: {fail_count} 只股票")
    print(f"  总记录: {total_records} 条")
    
    # 5. 验证数据
    print("\n[验证] 检查数据库...")
    conn = sqlite3.connect(DB_PATH)
    
    # 检查数据量
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM fina_indicator")
    total = cursor.fetchone()[0]
    print(f"  ✓ fina_indicator 表现在有 {total} 条记录")
    
    # 检查年份分布
    cursor.execute("""
        SELECT SUBSTR(end_date, 1, 4) as year, COUNT(*) 
        FROM fina_indicator 
        GROUP BY year 
        ORDER BY year
    """)
    years = cursor.fetchall()
    print(f"  年份分布:")
    for year, count in years:
        print(f"    {year}: {count} 条")
    
    conn.close()
    
    print("\n✓ 全部完成！")


if __name__ == "__main__":
    main()
