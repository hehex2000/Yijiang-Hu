"""
多因子选股脚本 - 选择TOP 10股票（价值因子）
基于2020-01-01数据，使用价值因子：PE、PB、PS、股息率
"""

import sqlite3
import pandas as pd
from loguru import logger
from typing import List, Dict
import os
import sys

# 数据库路径
DB_PATH = "D:/tu-shareData/astock_daily.db"


def get_all_a_stocks(base_date: str = '20200101') -> List[str]:
    """获取所有A股代码列表（在指定日期前上市）"""
    conn = sqlite3.connect(DB_PATH)
    
    query = f"""
        SELECT ts_code FROM stock_basic
        WHERE list_date <= '{base_date}'
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # 转换为简单格式（000063，不是000063.SZ）
    codes = []
    for ts_code in df['ts_code'].tolist():
        if ts_code.endswith('.SZ'):
            codes.append(ts_code[:6])
        elif ts_code.endswith('.SH'):
            codes.append(ts_code[:6])
    
    logger.info(f"获取到 {len(codes)} 只A股")
    return codes


def get_value_factors(code: str, db_path: str, date: str = '20200101') -> Dict:
    """
    获取价值因子
    因子说明：
    - PE（市盈率）：越低越好，负值表示低估
    - PB（市净率）：越低越好
    - PS（市销率）：越低越好
    - DV_ratio（股息率）：越高越好
    """
    conn = sqlite3.connect(db_path)
    
    # 获取daily_basic数据
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
    
    if len(df) == 0:
        return None
    
    factors = {}
    
    # 1. 市盈率（PE）-- 越低越好（允许PE无效，只要有PB就行）
    if not pd.isna(df['pe'].iloc[0]) and df['pe'].iloc[0] > 0:
        factors['pe'] = -df['pe'].iloc[0]  # 负值表示越低越好
    
    # 2. 市净率（PB）-- 越低越好
    if not pd.isna(df['pb'].iloc[0]) and df['pb'].iloc[0] > 0:
        factors['pb'] = -df['pb'].iloc[0]
    
    # 3. 市销率（PS）-- 越低越好（允许PS=0）
    if not pd.isna(df['ps'].iloc[0]) and df['ps'].iloc[0] >= 0:
        factors['ps'] = -df['ps'].iloc[0]
    
    # 4. 股息率 -- 越高越好
    if not pd.isna(df['dv_ratio'].iloc[0]) and df['dv_ratio'].iloc[0] > 0:
        factors['dv_ratio'] = df['dv_ratio'].iloc[0]
    
    # 保存股票名称
    if not pd.isna(df['name'].iloc[0]):
        factors['name'] = df['name'].iloc[0]
    
    # PE和PB至少有一个有效才返回
    return factors if ('pe' in factors or 'pb' in factors) else None


def normalize_factors(factors_df: pd.DataFrame) -> pd.DataFrame:
    """标准化因子（z-score）"""
    normalized = factors_df.copy()
    
    # 对每列进行z-score标准化
    for col in factors_df.columns:
        if col == 'code' or col == 'name':
            continue
        mean_val = factors_df[col].mean()
        std_val = factors_df[col].std()
        if std_val > 0:
            normalized[col] = (factors_df[col] - mean_val) / std_val
        else:
            normalized[col] = 0
    
    return normalized


def select_top10_stocks(base_date: str = '20200103', 
                         output_file: str = 'data/results/top10_value_stocks.csv'):
    """
    执行选股，选出TOP 10股票
    
    参数:
        base_date: 基准日期，格式 'YYYYMMDD'（必须是交易日）
        output_file: 输出文件路径
    """
    logger.info("="*50)
    logger.info(f"开始多因子选股（价值因子）...")
    logger.info(f"基准日期: {base_date}")
    logger.info("="*50)
    
    # 1. 获取所有A股列表
    print("\n[1/4] 获取所有A股列表...")
    stocks = get_all_a_stocks(base_date)
    print(f"✓ 获取到 {len(stocks)} 只股票\n")
    
    # 2. 计算因子
    print("[2/4] 计算价值因子（PE、PB、PS、股息率）...")
    results = []
    
    for i, code in enumerate(stocks):
        if i % 50 == 0:
            print(f"  进度: {i}/{len(stocks)}...")
        
        # 获取价值因子
        value_factors = get_value_factors(code, DB_PATH, base_date)
        
        if value_factors is None:
            continue
        
        # 合并因子
        result = {'code': code}
        result.update(value_factors)
        results.append(result)
    
    if len(results) == 0:
        print("错误：没有计算出任何有效因子！")
        return []
    
    factors_df = pd.DataFrame(results)
    print(f"✓ 因子计算完成: {len(factors_df)} 只股票\n")
    
    # 3. 标准化因子
    print("[3/4] 标准化因子...")
    normalized_df = normalize_factors(factors_df)
    
    # 4. 计算综合得分（等权重）
    factor_cols = [col for col in normalized_df.columns if col != 'code' and col != 'name' and col != 'total_score']
    normalized_df['total_score'] = normalized_df[factor_cols].sum(axis=1)
    
    # 5. 选出TOP 10
    print("[4/4] 选出TOP 10股票...")
    top10 = normalized_df.nlargest(10, 'total_score')[['code', 'total_score'] + factor_cols]
    
    print(f"✓ 选股完成: TOP 10 股票\n")
    print(top10.to_string(index=False))
    print()
    
    # 6. 保存结果
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    top10.to_csv(output_file, index=False, encoding='utf-8-sig')
    logger.info(f"结果已保存到: {output_file}")
    
    return top10['code'].tolist()


if __name__ == '__main__':
    # 初始化日志
    logger.remove()
    logger.add(sys.stdout, level="INFO", 
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")
    
    # 选择TOP 10股票（基于2020-01-03数据，第一个交易日）
    top10_codes = select_top10_stocks(base_date='20200103')
    
    print("="*70)
    print(f"TOP 10 股票代码: {top10_codes}")
    print("="*70)
