#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ML选股演示脚本 - 简化版
使用随机森林和XGBoost选股，基于现有因子
"""

import sqlite3
import pandas as pd
import numpy as np
from loguru import logger
from typing import List, Dict, Tuple, Optional
import os
import sys
from datetime import datetime, timedelta
import pickle

# 添加项目根目录到路径
project_root = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from select_top5_stocks import get_all_a_stocks, calculate_technical_factors, get_fundamental_factors, normalize_factors, DB_PATH


def prepare_ml_data_simple(train_end_date: str = '20181231', sample_size: int = 100) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    准备ML训练数据（简化版 - 仅使用部分股票）
    
    Args:
        train_end_date: 训练数据截止日期
        sample_size: 采样股票数量（加速测试）
        
    Returns:
        (X, y): 特征DataFrame和标签Series
    """
    logger.info("="*50)
    logger.info("准备ML训练数据（简化版）...")
    logger.info("="*50)
    
    # 1. 获取所有A股列表
    print("\n[1/4] 获取所有A股列表...")
    stocks = get_all_a_stocks()
    
    # 仅使用前 sample_size 只股票（加速测试）
    stocks = stocks[:sample_size]
    print(f"✓ 使用 {len(stocks)} 只股票进行训练\n")
    
    # 2. 计算因子和未来收益率
    print("[2/4] 计算因子和未来收益率（用于训练标签）...")
    results = []
    
    for i, code in enumerate(stocks):
        if i % 10 == 0:
            print(f"  进度: {i}/{len(stocks)}...")
        
        # 计算因子（以train_end_date为基准）
        tech_factors = calculate_technical_factors(code, DB_PATH, train_end_date)
        
        # 获取基本面因子
        fund_factors = get_fundamental_factors(code, DB_PATH, train_end_date)
        
        if tech_factors is None or fund_factors is None:
            continue
        
        # 计算未来20日收益率（简化：使用2019年1月的数据）
        future_return = _calculate_simple_return(code, train_end_date, days=20)
        
        if future_return is None:
            continue
        
        # 合并数据
        result = {'code': code, 'future_return': future_return}
        result.update(tech_factors)
        result.update(fund_factors)
        results.append(result)
    
    if len(results) < 10:
        raise ValueError(f"训练数据不足！仅获得 {len(results)} 条有效数据")
    
    df = pd.DataFrame(results)
    print(f"✓ 数据准备完成: {len(df)} 只股票\n")
    
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


def _calculate_simple_return(code: str, base_date: str, days: int = 20) -> Optional[float]:
    """
    计算未来N日收益率（简化版）
    """
    conn = sqlite3.connect(DB_PATH)
    
    # 计算未来日期
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


def train_and_save_models(X: pd.DataFrame, y: pd.Series, output_dir: str = 'data/models'):
    """
    训练并保存ML模型
    """
    logger.info("="*50)
    logger.info("训练ML模型...")
    logger.info("="*50)
    
    os.makedirs(output_dir, exist_ok=True)
    models = {}
    
    # 1. 训练随机森林
    print("\n[1/2] 训练随机森林模型...")
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 创建并训练模型
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        rf.fit(X_train, y_train)
        
        # 评估模型
        y_pred = rf.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"  随机森林 - MSE: {mse:.6f}, R2: {r2:.4f}")
        
        # 特征重要性
        importances = rf.feature_importances_
        print(f"  特征重要性:")
        for col, imp in zip(X.columns, importances):
            print(f"    {col}: {imp:.4f}")
        
        # 保存模型
        model_path = os.path.join(output_dir, 'random_forest_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(rf, f)
        
        models['random_forest'] = rf
        
        print(f"✓ 随机森林训练完成，模型已保存: {model_path}\n")
        
    except ImportError:
        logger.error("请安装scikit-learn: pip install scikit-learn")
        raise
    
    # 2. 训练XGBoost
    print("[2/2] 训练XGBoost模型...")
    try:
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 创建DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # 参数
        params = {
            'max_depth': 6,
            'eta': 0.3,
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'seed': 42
        }
        
        # 训练模型
        num_rounds = 100
        bst = xgb.train(params, dtrain, num_rounds)
        
        # 评估模型
        y_pred = bst.predict(dtest)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"  XGBoost - MSE: {mse:.6f}, R2: {r2:.4f}")
        
        # 特征重要性
        importances = bst.get_score(importance_type='weight')
        print(f"  特征重要性:")
        for col, imp in importances.items():
            print(f"    {col}: {imp:.4f}")
        
        # 保存模型
        model_path = os.path.join(output_dir, 'xgboost_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(bst, f)
        
        models['xgboost'] = bst
        
        print(f"✓ XGBoost训练完成，模型已保存: {model_path}\n")
        
    except ImportError:
        logger.warning("未安装xgboost，跳过XGBoost训练: pip install xgboost")
    
    logger.info(f"✓ 所有模型训练完成: {list(models.keys())}")
    
    return models


def select_stocks_with_ml(pred_date: str = '20200101', top_n: int = 10, 
                           model_dir: str = 'data/models') -> pd.DataFrame:
    """
    使用ML模型选股
    """
    logger.info("="*50)
    logger.info("使用ML模型选股...")
    logger.info("="*50)
    
    # 1. 加载模型
    print("\n[1/4] 加载ML模型...")
    models = {}
    
    rf_path = os.path.join(model_dir, 'random_forest_model.pkl')
    if os.path.exists(rf_path):
        with open(rf_path, 'rb') as f:
            models['random_forest'] = pickle.load(f)
        print(f"✓ 随机森林模型已加载: {rf_path}")
    
    xgb_path = os.path.join(model_dir, 'xgboost_model.pkl')
    if os.path.exists(xgb_path):
        with open(xgb_path, 'rb') as f:
            models['xgboost'] = pickle.load(f)
        print(f"✓ XGBoost模型已加载: {xgb_path}")
    
    if len(models) == 0:
        raise FileNotFoundError(f"未找到模型文件！请先运行训练脚本")
    
    print(f"✓ 已加载 {len(models)} 个模型\n")
    
    # 2. 获取所有A股列表
    print("[2/4] 获取所有A股列表...")
    stocks = get_all_a_stocks()
    print(f"✓ 获取到 {len(stocks)} 只股票\n")
    
    # 3. 计算因子
    print("[3/4] 计算因子...")
    results = []
    
    for i, code in enumerate(stocks):
        if i % 50 == 0:
            print(f"  进度: {i}/{len(stocks)}...")
        
        # 计算因子
        tech_factors = calculate_technical_factors(code, DB_PATH, pred_date)
        fund_factors = get_fundamental_factors(code, DB_PATH, pred_date)
        
        if tech_factors is None or fund_factors is None:
            continue
        
        result = {'code': code}
        result.update(tech_factors)
        result.update(fund_factors)
        results.append(result)
    
    if len(results) == 0:
        raise ValueError("未计算到任何股票的因子！请检查数据")
    
    factors_df = pd.DataFrame(results)
    print(f"✓ 因子计算完成: {len(factors_df)} 只股票\n")
    
    # 4. 标准化因子
    factor_cols = [col for col in factors_df.columns if col != 'code']
    normalized_df = normalize_factors(factors_df[factor_cols])
    normalized_df['code'] = factors_df['code'].values
    
    # 5. 使用ML模型预测
    print("[4/4] 使用ML模型预测...")
    X_pred = normalized_df[factor_cols]
    
    all_predictions = []
    
    for model_name, model in models.items():
        # 预测
        y_pred = model.predict(X_pred)
        
        # 合并结果
        temp_df = normalized_df[['code']].copy()
        temp_df[f'{model_name}_score'] = y_pred
        
        all_predictions.append(temp_df)
    
    # 合并所有模型的预测结果
    result_df = all_predictions[0]
    for df in all_predictions[1:]:
        result_df = result_df.merge(df, on='code', how='outer')
    
    # 计算平均得分（如果多个模型）
    if len(all_predictions) > 1:
        score_cols = [col for col in result_df.columns if col.endswith('_score')]
        result_df['ml_score'] = result_df[score_cols].mean(axis=1)
    else:
        result_df['ml_score'] = result_df['random_forest_score'] if 'random_forest_score' in result_df.columns else result_df['xgboost_score']
    
    # 选出TOP N
    top_stocks = result_df.nlargest(top_n, 'ml_score')
    
    print(f"✓ ML选股完成: TOP {top_n} 股票\n")
    print(top_stocks[['code', 'ml_score']].to_string(index=False))
    print()
    
    return top_stocks


def run_ml_selection_workflow():
    """
    运行完整的ML选股流程（训练 + 预测）
    """
    logger.remove()
    logger.add(sys.stdout, level="INFO", 
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")
    
    logger.info("="*70)
    logger.info("机器学习选股系统（简化版）")
    logger.info("="*70 + "\n")
    
    try:
        # 1. 准备训练数据（使用100只股票加速测试）
        X, y = prepare_ml_data_simple(train_end_date='20181231', sample_size=100)
        
        # 2. 训练并保存模型
        models = train_and_save_models(X, y)
        
        # 3. 使用ML模型选股
        top_stocks = select_stocks_with_ml(pred_date='20200101', top_n=10)
        
        # 4. 保存结果
        output_file = 'data/results/top10_stocks_ml.csv'
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
    top_stocks = run_ml_selection_workflow()
    
    print("="*70)
    print(f"ML选股结果 - TOP 10 股票代码: {top_stocks}")
    print("="*70)
