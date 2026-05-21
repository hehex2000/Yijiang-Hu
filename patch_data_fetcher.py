# -*- coding: utf-8 -*-
"""
补丁 - 让 data_fetcher.py 优先从本地数据库读取财务和估值数据
"""
import sqlite3
import pandas as pd
import os

# 本地数据库路径
DB_PATH = "D:/tu-shareData/astock_daily.db"

def get_financial_data_from_local_db(code: str) -> pd.DataFrame:
    """从本地数据库读取财务指标数据"""
    try:
        conn = sqlite3.connect(DB_PATH)
        ts_code = code + ".SZ" if code.startswith(("0", "3")) else code + ".SH"
        
        query = """
            SELECT *
            FROM fina_indicator
            WHERE ts_code = ?
            ORDER BY end_date DESC
            LIMIT 1
        """
        
        df = pd.read_sql_query(query, conn, params=(ts_code,))
        conn.close()
        
        if df is not None and len(df) > 0:
            # 转换为AkShare格式
            column_mapping = {
                'end_date': '截止日期',
                'ts_code': '代码',
                'roe': '净资产收益率',
                'roa': '总资产报酬率',
                'netprofit_yoy': '净利润同比增长率',
                'revenue_yoy': '营业收入同比增长率',
                'asset_liability_ratio': '资产负债率',
                'current_ratio': '流动比率',
                'asset_turnover': '资产周转率'
            }
            df = df.rename(columns=column_mapping)
            return df
        return None
    except Exception as e:
        print(f"读取财务数据失败: {e}")
        return None

def get_valuation_data_from_local_db(code: str) -> pd.DataFrame:
    """从本地数据库读取估值指标数据"""
    try:
        conn = sqlite3.connect(DB_PATH)
        ts_code = code + ".SZ" if code.startswith(("0", "3")) else code + ".SH"
        
        query = """
            SELECT *
            FROM daily_basic
            WHERE ts_code = ?
            ORDER BY trade_date DESC
            LIMIT 1
        """
        
        df = pd.read_sql_query(query, conn, params=(ts_code,))
        conn.close()
        
        if df is not None and len(df) > 0:
            # 转换为AkShare格式
            column_mapping = {
                'trade_date': '日期',
                'ts_code': '代码',
                'pe': '市盈率',
                'pb': '市净率',
                'ps': '市销率',
                'ps_ttm': '市销率(TTM)',
                'total_mv': '总市值',
                'circ_mv': '流通市值'
            }
            df = df.rename(columns=column_mapping)
            
            # 市值单位转换（万元 → 亿元）
            if '总市值' in df.columns:
                df['总市值'] = df['总市值'] / 10000
            if '流通市值' in df.columns:
                df['流通市值'] = df['流通市值'] / 10000
            
            return df
        return None
    except Exception as e:
        print(f"读取估值数据失败: {e}")
        return None

if __name__ == "__main__":
    # 测试
    test_code = "000001"
    
    print(f"测试股票: {test_code}")
    print()
    
    # 测试财务数据
    print("测试1: 读取财务数据...")
    financial_df = get_financial_data_from_local_db(test_code)
    if financial_df is not None:
        print(f"[OK] 成功读取财务数据")
        print(f"  列名: {list(financial_df.columns)}")
        print(f"  数据条数: {len(financial_df)}")
    else:
        print(f"[FAIL] 未找到财务数据")
    print()
    
    # 测试估值数据
    print("测试2: 读取估值数据...")
    valuation_df = get_valuation_data_from_local_db(test_code)
    if valuation_df is not None:
        print(f"[OK] 成功读取估值数据")
        print(f"  列名: {list(valuation_df.columns)}")
        print(f"  数据条数: {len(valuation_df)}")
    else:
        print(f"[FAIL] 未找到估值数据")
    print()
    
    print("测试完成!")
