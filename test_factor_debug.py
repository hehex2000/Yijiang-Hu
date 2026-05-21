"""
测试脚本 - 调试因子计算问题
"""
import sqlite3
import pandas as pd
from loguru import logger

DB_PATH = "D:/tu-shareData/astock_daily.db"

def test_get_value_factors(code: str, date: str = '20200103'):
    """测试获取价值因子"""
    conn = sqlite3.connect(DB_PATH)
    
    query = f"""
        SELECT d.pe, d.pb, d.ps, d.dv_ratio, s.name
        FROM daily_basic d
        JOIN stock_basic s ON d.ts_code = s.ts_code
        WHERE d.ts_code = (
            SELECT CASE 
                WHEN SUBSTR('{code}', 1, 1) IN ('6') THEN '{code}.SH'
                ELSE '{code}.SZ'
            END
        )
        AND d.trade_date = '{date}'
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"\n股票 {code}:")
    print(f"  查询结果行数: {len(df)}")
    
    if len(df) > 0:
        print(f"  PE: {df['pe'].iloc[0]}")
        print(f"  PB: {df['pb'].iloc[0]}")
        print(f"  PS: {df['ps'].iloc[0]}")
        print(f"  DV_ratio: {df['dv_ratio'].iloc[0]}")
        print(f"  Name: {df['name'].iloc[0]}")
        
        # 测试因子计算逻辑
        factors = {}
        if not pd.isna(df['pe'].iloc[0]) and df['pe'].iloc[0] > 0:
            factors['pe'] = -df['pe'].iloc[0]
            print(f"  ✓ PE因子有效: {factors['pe']}")
        else:
            print(f"  ✗ PE因子无效: {df['pe'].iloc[0]}")
            
        if not pd.isna(df['pb'].iloc[0]) and df['pb'].iloc[0] > 0:
            factors['pb'] = -df['pb'].iloc[0]
            print(f"  ✓ PB因子有效: {factors['pb']}")
        else:
            print(f"  ✗ PB因子无效: {df['pb'].iloc[0]}")
        
        print(f"  最终因子数: {len(factors)}")
        return factors
    else:
        print(f"  ✗ 没有查询到数据")
        return None

if __name__ == '__main__':
    # 测试几只股票
    test_codes = ['000063', '600000', '000001', '600016', '601398']
    
    for code in test_codes:
        factors = test_get_value_factors(code)
        if factors:
            print(f"  => 成功计算因子: {factors}")
        else:
            print(f"  => 因子计算失败")
