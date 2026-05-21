"""
下载申万一级行业指数数据到本地数据库
数据来源：Tushare index_daily 接口
目标表：industry_index_daily（新建）
"""
import sqlite3
import time
import tushare as ts
import pandas as pd

# Tushare token
TS_TOKEN = "761165a821532fe625262d6b33e144b9859a887c004acbcb981c319b"

# 数据库路径
DB_PATH = r"D:\tu-shareData\astock_daily.db"

# 申万一级行业指数代码（Tushare格式：801XXX.SI）
INDUSTRY_INDEX_CODES = [
    "801010.SI",  # 农林牧渔
    "801020.SI",  # 采掘
    "801030.SI",  # 化工
    "801040.SI",  # 钢铁
    "801050.SI",  # 有色金属
    "801080.SI",  # 建筑材料
    "801110.SI",  # 家用电器
    "801120.SI",  # 食品饮料
    "801130.SI",  # 纺织服装
    "801140.SI",  # 轻工制造
    "801150.SI",  # 医药生物
    "801160.SI",  # 公用事业
    "801170.SI",  # 交通运输
    "801180.SI",  # 房地产
    "801200.SI",  # 商业贸易
    "801210.SI",  # 休闲服务
    "801220.SI",  # 综合
    "801710.SI",  # 建筑材料
    "801720.SI",  # 建筑装饰
    "801730.SI",  # 电气设备
    "801740.SI",  # 国防军工
    "801750.SI",  # 计算机
    "801760.SI",  # 传媒
    "801770.SI",  # 通信
    "801780.SI",  # 银行
    "801790.SI",  # 非银金融
    "801880.SI",  # 汽车
    "801890.SI",  # 机械设备
]

# 行业名称映射
INDUSTRY_NAMES = {
    "801010.SI": "农林牧渔",
    "801020.SI": "采掘",
    "801030.SI": "化工",
    "801040.SI": "钢铁",
    "801050.SI": "有色金属",
    "801080.SI": "建筑材料",
    "801110.SI": "家用电器",
    "801120.SI": "食品饮料",
    "801130.SI": "纺织服装",
    "801140.SI": "轻工制造",
    "801150.SI": "医药生物",
    "801160.SI": "公用事业",
    "801170.SI": "交通运输",
    "801180.SI": "房地产",
    "801200.SI": "商业贸易",
    "801210.SI": "休闲服务",
    "801220.SI": "综合",
    "801710.SI": "建筑材料",
    "801720.SI": "建筑装饰",
    "801730.SI": "电气设备",
    "801740.SI": "国防军工",
    "801750.SI": "计算机",
    "801760.SI": "传媒",
    "801770.SI": "通信",
    "801780.SI": "银行",
    "801790.SI": "非银金融",
    "801880.SI": "汽车",
    "801890.SI": "机械设备",
}


def init_database(db_path):
    """初始化数据库，创建行业指数表"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 创建行业指数日线表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS industry_index_daily (
            ts_code TEXT,
            trade_date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            pre_close REAL,
            change REAL,
            pct_chg REAL,
            vol REAL,
            amount REAL,
            PRIMARY KEY (ts_code, trade_date)
        )
    """)
    
    # 创建行业信息表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS industry_info (
            ts_code TEXT PRIMARY KEY,
            industry_name TEXT,
            industry_code TEXT
        )
    """)
    
    conn.commit()
    conn.close()
    print("✅ 数据库表初始化完成")


def download_industry_index(ts_pro, ts_code, start_date="20200101", end_date="20231231"):
    """下载单个行业指数数据"""
    try:
        df = ts_pro.index_daily(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date
        )
        return df
    except Exception as e:
        print(f"❌ 下载 {ts_code} 失败: {e}")
        return None


def save_to_database(db_path, df):
    """保存行业指数数据到数据库"""
    if df is None or df.empty:
        return 0
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 插入数据（忽略重复）
    count = 0
    for _, row in df.iterrows():
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO industry_index_daily
                (ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row['ts_code'],
                row['trade_date'],
                row['open'],
                row['high'],
                row['low'],
                row['close'],
                row['pre_close'],
                row['change'],
                row['pct_chg'],
                row['vol'],
                row['amount']
            ))
            count += 1
        except Exception as e:
            print(f"❌ 插入数据失败: {e}")
            continue
    
    conn.commit()
    conn.close()
    return count


def main():
    # 初始化Tushare
    ts.set_token(TS_TOKEN)
    ts_pro = ts.pro_api()
    
    print("🚀 开始下载申万一级行业指数数据...")
    
    # 初始化数据库
    init_database(DB_PATH)
    
    # 保存行业信息
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    for ts_code, industry_name in INDUSTRY_NAMES.items():
        cursor.execute("""
            INSERT OR REPLACE INTO industry_info (ts_code, industry_name, industry_code)
            VALUES (?, ?, ?)
        """, (ts_code, industry_name, ts_code[:6]))
    conn.commit()
    conn.close()
    print(f"✅ 行业信息已保存（{len(INDUSTRY_NAMES)}个行业）")
    
    # 下载每个行业指数数据
    total_count = 0
    for ts_code in INDUSTRY_INDEX_CODES:
        industry_name = INDUSTRY_NAMES.get(ts_code, ts_code)
        print(f"\n📥 下载 {ts_code} ({industry_name})...")
        
        df = download_industry_index(ts_pro, ts_code)
        if df is not None and not df.empty:
            count = save_to_database(DB_PATH, df)
            total_count += count
            print(f"  ✅ 下载 {len(df)} 条，保存 {count} 条")
        else:
            print(f"  ⚠️ 无数据")
        
        # 限速：Tushare 限制 120次/分钟
        time.sleep(0.5)
    
    print(f"\n🎉 下载完成！共保存 {total_count} 条数据")


if __name__ == "__main__":
    main()
