# -*- coding: utf-8 -*-
"""
完整选股脚本 - 2023-01-01 选股
从本地数据库读取所有数据，计算因子，选股
"""
import sys
import os
import pandas as pd
import numpy as np
import sqlite3
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 本地数据库路径
DB_PATH = "D:/tu-shareData/astock_daily.db"

print("\n" + "="*70)
print("完整选股 - 2023-01-01")
print("因子：价值 + 成长 + 质量 + 低波动")
print("="*70 + "\n")

# ========== 1. 从本地数据库读取数据 ==========
print("[1/5] 从本地数据库读取数据...")

def get_hs300_components():
    """获取沪深300成分股"""
    conn = sqlite3.connect(DB_PATH)
    # 注意：本地数据库可能没有HS300成分股表，这里读取所有股票
    # 实际使用时应该从 index_constituent 表读取，或者从Tushare获取
    df = pd.read_sql_query("SELECT ts_code, symbol, name FROM stock_basic LIMIT 50", conn)
    conn.close()
    
    # 转换为简单格式（无后缀）
    df['code'] = df['symbol']
    print(f"  获取到 {len(df)} 只股票（测试用前50只）")
    return df[['code', 'name']]

def get_stock_data_from_db(code, start_date="20230101", end_date="20231231"):
    """从本地数据库读取单只股票的数据"""
    conn = sqlite3.connect(DB_PATH)
    ts_code = code + ".SZ" if code.startswith(("0", "3")) else code + ".SH"
    
    # 1. 读取历史行情
    query_hist = f"""
        SELECT trade_date, open, high, low, close, vol, amount
        FROM daily
        WHERE ts_code = ? AND trade_date BETWEEN ? AND ?
        ORDER BY trade_date
    """
    hist_df = pd.read_sql_query(query_hist, conn, params=(ts_code, start_date, end_date))
    
    # 重命名列为AkShare格式
    if len(hist_df) > 0:
        hist_df.columns = ['日期', '开盘', '最高', '最低', '收盘', '成交量', '成交额']
    
    # 2. 读取估值数据
    query_valuation = f"""
        SELECT trade_date, pe, pb, ps, ps_ttm, total_mv, circ_mv
        FROM daily_basic
        WHERE ts_code = ?
        ORDER BY trade_date DESC
        LIMIT 1
    """
    valuation_df = pd.read_sql_query(query_valuation, conn, params=(ts_code,))
    
    # 重命名列为AkShare格式
    if len(valuation_df) > 0:
        valuation_df.columns = ['日期', '市盈率', '市净率', '市销率', '市销率(TTM)', '总市值', '流通市值']
        # 市值单位转换（万元 → 亿元）
        valuation_df['总市值'] = valuation_df['总市值'] / 10000
        valuation_df['流通市值'] = valuation_df['流通市值'] / 10000
    
    # 3. 读取财务数据
    query_financial = f"""
        SELECT *
        FROM fina_indicator
        WHERE ts_code = ?
        ORDER BY end_date DESC
        LIMIT 1
    """
    financial_df = pd.read_sql_query(query_financial, conn, params=(ts_code,))
    
    # 重命名列为AkShare格式
    if len(financial_df) > 0:
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
        financial_df = financial_df.rename(columns=column_mapping)
    
    conn.close()
    
    return {
        'hist': hist_df if len(hist_df) > 0 else None,
        'valuation': valuation_df if len(valuation_df) > 0 else None,
        'financial': financial_df if len(financial_df) > 0 else None
    }

# 获取股票池
stock_pool = get_hs300_components()
print("[OK] 数据读取完成\n")

# ========== 2. 计算因子 ==========
print("[2/5] 计算因子...")

def calculate_value_factors(valuation_data):
    """计算价值因子"""
    factors = {}
    if valuation_data is not None and len(valuation_data) > 0:
        factors['VF1_PE'] = valuation_data['市盈率'].values[0] if '市盈率' in valuation_data.columns else np.nan
        factors['VF2_PB'] = valuation_data['市净率'].values[0] if '市净率' in valuation_data.columns else np.nan
        factors['VF3_PS'] = valuation_data['市销率'].values[0] if '市销率' in valuation_data.columns else np.nan
    else:
        factors['VF1_PE'] = np.nan
        factors['VF2_PB'] = np.nan
        factors['VF3_PS'] = np.nan
    return factors

def calculate_growth_factors(financial_data):
    """计算成长因子"""
    factors = {}
    if financial_data is not None and len(financial_data) > 0:
        factors['GF1_revenue_growth'] = financial_data['营业收入同比增长率'].values[0] if '营业收入同比增长率' in financial_data.columns else np.nan
        factors['GF2_net_profit_growth'] = financial_data['净利润同比增长率'].values[0] if '净利润同比增长率' in financial_data.columns else np.nan
        factors['GF3_ROE'] = financial_data['净资产收益率'].values[0] if '净资产收益率' in financial_data.columns else np.nan
    else:
        factors['GF1_revenue_growth'] = np.nan
        factors['GF2_net_profit_growth'] = np.nan
        factors['GF3_ROE'] = np.nan
    return factors

def calculate_quality_factors(financial_data):
    """计算质量因子"""
    factors = {}
    if financial_data is not None and len(financial_data) > 0:
        factors['QF1_asset_liability_ratio'] = financial_data['资产负债率'].values[0] if '资产负债率' in financial_data.columns else np.nan
        factors['QF2_current_ratio'] = financial_data['流动比率'].values[0] if '流动比率' in financial_data.columns else np.nan
        factors['QF3_asset_turnover'] = financial_data['资产周转率'].values[0] if '资产周转率' in financial_data.columns else np.nan
    else:
        factors['QF1_asset_liability_ratio'] = np.nan
        factors['QF2_current_ratio'] = np.nan
        factors['QF3_asset_turnover'] = np.nan
    return factors

def calculate_volatility_factors(hist_data):
    """计算低波动因子"""
    factors = {}
    if hist_data is not None and len(hist_data) > 20:
        close_prices = hist_data['收盘'].values
        returns = np.diff(close_prices) / close_prices[:-1]
        
        # 历史波动率（年化）
        daily_vol = np.std(returns[-20:])
        annualized_vol = daily_vol * np.sqrt(252)
        factors['LVF1_hist_vol'] = annualized_vol
    else:
        factors['LVF1_hist_vol'] = np.nan
    return factors

# 计算所有股票的因子
all_factors = []

for idx, row in stock_pool.iterrows():
    code = row['code']
    name = row['name']
    
    if idx % 10 == 0:
        print(f"  进度: {idx+1}/{len(stock_pool)}")
    
    try:
        # 读取数据
        data = get_stock_data_from_db(code)
        
        # 计算因子
        factors = {'code': code, 'name': name}
        factors.update(calculate_value_factors(data['valuation']))
        factors.update(calculate_growth_factors(data['financial']))
        factors.update(calculate_quality_factors(data['financial']))
        factors.update(calculate_volatility_factors(data['hist']))
        
        all_factors.append(factors)
    except Exception as e:
        logger.warning(f"计算 {code} 因子失败: {e}")
        continue

factors_df = pd.DataFrame(all_factors)
print(f"\n[OK] 因子计算完成: {len(factors_df)} 只股票\n")

# ========== 3. 因子标准化和打分 ==========
print("[3/5] 因子标准化和打分...")

def normalize_and_score(df):
    """标准化因子并打分"""
    # 复制DataFrame
    result = df.copy()
    
    # 需要反向的因子（值越小越好）
    reverse_factors = ['VF1_PE', 'VF2_PB', 'VF3_PS', 'QF1_asset_liability_ratio', 'LVF1_hist_vol']
    
    # 标准化每个因子
    factor_cols = [c for c in df.columns if c.startswith(('VF', 'GF', 'QF', 'LVF'))]
    
    for col in factor_cols:
        if col in df.columns:
            # 去除NaN
            valid_data = df[col].dropna()
            
            if len(valid_data) == 0:
                continue
            
            # 标准化（Z-Score）
            mean_val = valid_data.mean()
            std_val = valid_data.std()
            
            if std_val == 0:
                result[f"{col}_score"] = 0
            else:
                result[f"{col}_score"] = (df[col] - mean_val) / std_val
            
            # 如果是反向因子，取负值
            if col in reverse_factors:
                result[f"{col}_score"] = -result[f"{col}_score"]
    
    # 计算总分
    score_cols = [c for c in result.columns if c.endswith('_score')]
    result['total_score'] = result[score_cols].mean(axis=1)
    
    return result

processed_df = normalize_and_score(factors_df)
print("[OK] 因子标准化和打分完成\n")

# ========== 4. 选股 ==========
print("[4/5] 执行选股（TOP 20）...")

# 按总分排序
selected_df = processed_df.sort_values('total_score', ascending=False).head(20)
selected_df['rank'] = range(1, len(selected_df) + 1)

print(f"[OK] 选股完成: {len(selected_df)} 只股票\n")

# ========== 5. 打印结果 ==========
print("[5/5] 打印结果...\n")

print("="*70)
print("TOP 20 股票:")
print("="*70)
print(selected_df[['rank', 'code', 'name', 'total_score']].to_string(index=False))
print()

# 打印因子统计
print("="*70)
print("因子统计:")
print("="*70)

factor_cols = [c for c in processed_df.columns if c.startswith(('VF', 'GF', 'QF', 'LVF'))]
for col in factor_cols:
    if col in processed_df.columns:
        non_nan = processed_df[col].notna().sum()
        mean_val = processed_df[col].mean()
        print(f"  {col}: 非空={non_nan}/{len(processed_df)}, 均值={mean_val:.4f}" 
              if not pd.isna(mean_val) else f"  {col}: 非空={non_nan}/{len(processed_df)}, 均值=NaN")

print("\n" + "="*70)
print("选股完成!")
print("="*70 + "\n")

# 导出到CSV
output_dir = "data/results"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "selection_20230101.csv")

processed_df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"[OK] 结果已保存到: {output_path}\n")
