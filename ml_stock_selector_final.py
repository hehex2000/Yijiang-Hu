#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ML增强选股 - 完整可用版
使用随机森林优化因子权重，生成TOP 10股票
"""

import sqlite3
import pandas as pd
import numpy as np
from loguru import logger
from typing import List, Dict, Tuple, Optional
import os
import sys

# 添加项目根目录到路径
project_root = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from select_top5_stocks import get_all_a_stocks, calculate_technical_factors, get_fundamental_factors, normalize_factors, DB_PATH


def prepare_ml_training_data(train_date: str = '20181231', n_stocks: int = 100) -> Tuple[pd.DataFrame, pd.Series]:
    """
    准备ML训练数据（简化版 - 使用部分股票）
    
    Args:
        train_date: 训练基准日期
        n_stocks: 使用多少只股票进行训练（加速）
        
    Returns:
        (X, y): 特征DataFrame和标签Series
    """
    logger.info("="*50)
    logger.info("准备ML训练数据...")
    logger.info("="*50)
    
    # 1. 获取A股列表（使用前N只，确保有数据）
    print("\n[1/4] 获取A股列表...")
    stocks = get_all_a_stocks()[:n_stocks]
    print(f"✓ 使用 {len(stocks)} 只股票进行训练\n")
    
    # 2. 计算因子和未来收益率
    print("[2/4] 计算因子和未来收益率（用于训练标签）...")
    results = []
    
    for i, code in enumerate(stocks):
        if i % 10 == 0:
            print(f"  进度: {i}/{len(stocks)}...")
        
        # 计算因子
        tech_factors = calculate_technical_factors(code, DB_PATH, train_date)
        fund_factors = get_fundamental_factors(code, DB_PATH, train_date)
        
        if tech_factors is None or fund_factors is None:
            continue
        
        # 计算未来20日收益率（标签）
        future_ret = _calc_future_return(code, train_date, days=20)
        
        if future_ret is None:
            continue
        
        # 合并数据
        result = {'code': code, 'future_return': future_ret}
        result.update(tech_factors)
        result.update(fund_factors)
        results.append(result)
    
    if len(results) < 10:
        raise ValueError(f"训练数据不足！仅获得 {len(results)} 条有效数据")
    
    df = pd.DataFrame(results)
    print(f"✓ 数据准备完成: {len(df)} 条记录\n")
    
    # 3. 标准化因子
    print("[3/4] 标准化因子...")
    factor_cols = [col for col in df.columns if col not in ['code', 'future_return']]
    normalized_df = normalize_factors(df[factor_cols])
    normalized_df['code'] = df['code'].values
    normalized_df['future_return'] = df['future_return'].values
    
    # 4. 分离特征和标签
    print("[4/4] 分离特征和标签...")
    X = normalized_df[factor_cols]
    y = normalized_df['future_return']
    
    print(f"✓ 特征维度: {X.shape}")
    print(f"✓ 标签维度: {y.shape}\n")
    
    return X, y


def _calc_future_return(code: str, base_date: str, days: int = 20) -> Optional[float]:
    """计算未来N日收益率（用于训练标签）"""
    conn = sqlite3.connect(DB_PATH)
    
    # 计算未来日期
    from datetime import datetime, timedelta
    base_dt = datetime.strptime(base_date, '%Y%m%d')
    future_dt = base_dt + timedelta(days=days)
    future_date = future_dt.strftime('%Y%m%d')
    
    # 获取基准日期和未来日期的价格
    query = f"""
        SELECT d.trade_date, d.close, af.adj_factor
        FROM daily d
        JOIN adj_factor af ON d.ts_code = af.ts_code 
                           AND d.trade_date = af.trade_date
        WHERE d.ts_code = (
            SELECT CASE 
                WHEN SUBSTR('{code}', 1, 1) IN ('6') THEN '{code}.SH'
                ELSE '{code}.SZ'
            END
        )
        AND d.trade_date BETWEEN '{base_date}' AND '{future_date}'
        ORDER BY d.trade_date
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if len(df) < 2:
        return None
    
    # 计算前复权价格
    latest_adj = df['adj_factor'].iloc[-1]
    df['adj_close'] = df['close'] * (df['adj_factor'] / latest_adj)
    
    # 计算收益率
    base_price = df['adj_close'].iloc[0]
    future_price = df['adj_close'].iloc[-1]
    
    return (future_price - base_price) / base_price


def train_ml_model(X: pd.DataFrame, y: pd.Series, model_type: str = 'random_forest'):
    """
    训练ML模型
    
    Returns:
        训练好的模型
    """
    logger.info("="*50)
    logger.info(f"训练ML模型 ({model_type})...")
    logger.info("="*50)
    
    if model_type == 'random_forest':
        print(f"\n训练随机森林模型...")
        
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, r2_score
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            rf = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            rf.fit(X_train, y_train)
            
            y_pred = rf.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"  随机森林 - MSE: {mse:.6f}, R2: {r2:.4f}")
            print(f"  特征重要性:")
            importances = rf.feature_importances_
            for col, imp in zip(X.columns, importances):
                print(f"    {col}: {imp:.4f}")
            
            print(f"✓ 随机森林训练完成\n")
            
            return rf
            
        except ImportError:
            logger.error("请安装scikit-learn: pip install scikit-learn")
            raise
    
    elif model_type == 'xgboost':
        print(f"\n训练XGBoost模型...")
        
        try:
            import xgboost as xgb
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, r2_score
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)
            
            params = {
                'max_depth': 6,
                'eta': 0.3,
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'seed': 42
            }
            
            bst = xgb.train(params, dtrain, num_boost_round=100)
            
            y_pred = bst.predict(dtest)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"  XGBoost - MSE: {mse:.6f}, R2: {r2:.4f}")
            print(f"  特征重要性:")
            importances = bst.get_score(importance_type='weight')
            for col, imp in importances.items():
                print(f"    {col}: {imp:.4f}")
            
            print(f"✓ XGBoost训练完成\n")
            
            return bst
            
        except ImportError:
            logger.error("请安装xgboost: pip install xgboost")
            raise
    
    return None


def select_stocks_ml(pred_date: str = '20200101', top_n: int = 10, 
                       model=None, model_type: str = 'random_forest') -> pd.DataFrame:
    """
    使用ML模型选股
    
    Returns:
        选股结果DataFrame（包含ML得分）
    """
    logger.info("="*50)
    logger.info("使用ML模型选股...")
    logger.info("="*50)
    
    if model is None:
        raise ValueError("模型未训练！请先调用 train_ml_model()")
    
    # 1. 获取所有A股列表
    print("\n[1/3] 获取所有A股列表...")
    stocks = get_all_a_stocks()
    print(f"✓ 获取到 {len(stocks)} 只股票\n")
    
    # 2. 计算因子
    print("[2/3] 计算因子...")
    results = []
    
    for i, code in enumerate(stocks):
        if i % 50 == 0:
            print(f"  进度: {i}/{len(stocks)}...")
        
        tech_factors = calculate_technical_factors(code, DB_PATH, pred_date)
        fund_factors = get_fundamental_factors(code, DB_PATH, pred_date)
        
        if tech_factors is None or fund_factors is None:
            continue
        
        result = {'code': code}
        result.update(tech_factors)
        result.update(fund_factors)
        results.append(result)
    
    factors_df = pd.DataFrame(results)
    print(f"✓ 因子计算完成: {len(factors_df)} 只股票\n")
    
    # 3. 标准化因子
    factor_cols = [col for col in factors_df.columns if col != 'code']
    normalized_df = normalize_factors(factors_df[factor_cols])
    normalized_df['code'] = factors_df['code'].values
    
    # 4. 使用ML模型预测
    print("[3/3] 使用ML模型预测...")
    X_pred = normalized_df[factor_cols]
    y_pred = model.predict(X_pred)
    
    # 合并结果
    result_df = normalized_df[['code']].copy()
    result_df['ml_score'] = y_pred
    
    # 选出TOP N
    top_stocks = result_df.nlargest(top_n, 'ml_score')
    
    print(f"✓ ML选股完成: TOP {top_n} 股票\n")
    print(top_stocks[['code', 'ml_score']].to_string(index=False))
    print()
    
    return top_stocks


def save_ml_model(model, model_path: str = 'data/models/ml_model.pkl'):
    """保存训练好的模型"""
    import pickle
    import os
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"✓ 模型已保存: {model_path}")


def load_ml_model(model_path: str = 'data/models/ml_model.pkl'):
    """加载已训练的模型"""
    import pickle
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型文件: {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    logger.info(f"✓ 模型已加载: {model_path}")
    
    return model


def run_ml_stock_selection(train_date: str = '20181231', 
                          pred_date: str = '20200101', 
                          top_n: int = 10,
                          model_type: str = 'random_forest',
                          output_file: str = 'data/results/top10_stocks_ml.csv'):
    """
    运行完整的ML选股流程
    
    Args:
        train_date: 训练数据基准日期
        pred_date: 预测基准日期
        top_n: 选择前N只股票
        model_type: 模型类型 ('random_forest', 'xgboost')
        output_file: 输出文件路径
    """
    logger.remove()
    logger.add(sys.stdout, level="INFO", 
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")
    
    logger.info("="*70)
    logger.info("机器学习选股系统（完整版）")
    logger.info("="*70 + "\n")
    
    try:
        # 1. 准备训练数据
        X, y = prepare_ml_training_data(train_date=train_date, n_stocks=100)
        
        # 2. 训练模型
        model = train_ml_model(X, y, model_type=model_type)
        
        # 3. 保存模型
        model_path = f'data/models/{model_type}_model.pkl'
        save_ml_model(model, model_path)
        
        # 4. 使用ML模型选股
        top_stocks = select_stocks_ml(pred_date=pred_date, top_n=top_n, 
                                    model=model, model_type=model_type)
        
        # 5. 保存结果
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        top_stocks.to_csv(output_file, index=False, encoding='utf-8-sig')
        logger.info(f"✓ ML选股结果已保存: {output_file}")
        
        return top_stocks['code'].tolist()
        
    except Exception as e:
        logger.error(f"ML选股失败: {e}")
        import traceback
        traceback.print_exc()
        return []


if __name__ == '__main__':
    # 运行ML选股
    top_stocks = run_ml_stock_selection(
        train_date='20181231',   # 使用2018年及之前的数据训练
        pred_date='20200101',    # 预测2020年的股票
        top_n=10,               # 选择TOP 10
        model_type='random_forest'  # 使用随机森林
    )
    
    print("="*70)
    print(f"ML选股结果 - TOP 10 股票代码: {top_stocks}")
    print("="*70)
