#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ML增强选股 - 使用随机森林和XGBoost优化因子权重
基于现有多因子选股，用ML学习最优因子权重
"""

import sqlite3
import pandas as pd
import numpy as np
from loguru import logger
from typing import List, Dict, Tuple, Optional
import os
import sys
import pickle

# 添加项目根目录到路径
project_root = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from select_top5_stocks import get_all_a_stocks, calculate_technical_factors, get_fundamental_factors, normalize_factors, DB_PATH


def prepare_training_data_multi_period(end_date: str = '20191231', periods: int = 12) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    准备训练数据（多个时间点，增加样本量）
    
    Args:
        end_date: 结束日期
        periods: 使用过去N个月的数据点
        
    Returns:
        (X, y): 特征DataFrame和标签Series
    """
    logger.info("="*50)
    logger.info("准备ML训练数据（多时间点）...")
    logger.info("="*50)
    
    all_results = []
    
    # 生成多个时间点
    end_dt = datetime.strptime(end_date, '%Y%m%d')
    dates = []
    for i in range(periods):
        dt = end_dt - pd.DateOffset(months=i*6)  # 每6个月一个时间点
        dates.append(dt.strftime('%Y%m%d'))
    
    print(f"\n[1/3] 使用 {len(dates)} 个时间点生成训练数据...")
    
    for i, date in enumerate(dates):
        print(f"  处理时间点 {i+1}/{len(dates)}: {date}...")
        
        # 获取所有A股
        stocks = get_all_a_stocks()
        
        for j, code in enumerate(stocks[:200]):  # 每个时间点只用200只股票，加速
            if j % 50 == 0:
                print(f"    进度: {j}/{min(200, len(stocks))}...")
            
            # 计算因子
            tech_factors = calculate_technical_factors(code, DB_PATH, date)
            fund_factors = get_fundamental_factors(code, DB_PATH, date)
            
            if tech_factors is None or fund_factors is None:
                continue
            
            # 计算未来20日收益率（标签）
            future_return = _calculate_future_return_simple(code, date, days=20)
            
            if future_return is None:
                continue
            
            # 合并数据
            result = {'code': code, 'date': date, 'future_return': future_return}
            result.update(tech_factors)
            result.update(fund_factors)
            all_results.append(result)
    
    if len(all_results) < 50:
        raise ValueError(f"训练数据不足！仅获得 {len(all_results)} 条有效数据")
    
    df = pd.DataFrame(all_results)
    print(f"\n✓ 数据准备完成: {len(df)} 条记录\n")
    
    # 标准化因子
    print("[2/3] 标准化因子...")
    factor_cols = [col for col in df.columns if col not in ['code', 'date', 'future_return']]
    normalized_df = normalize_factors(df[factor_cols])
    normalized_df['code'] = df['code'].values
    normalized_df['date'] = df['date'].values
    normalized_df['future_return'] = df['future_return'].values
    
    # 分离特征和标签
    print("[3/3] 分离特征和标签...")
    X = normalized_df[factor_cols]
    y = normalized_df['future_return']
    
    print(f"✓ 特征维度: {X.shape}")
    print(f"✓ 标签维度: {y.shape}\n")
    
    return X, y


def _calculate_future_return_simple(code: str, base_date: str, days: int = 20) -> Optional[float]:
    """计算未来N日收益率（简化版）"""
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


def train_ml_models(X: pd.DataFrame, y: pd.Series, model_type: str = 'both') -> Dict:
    """
    训练ML模型
    
    Returns:
        训练好的模型字典
    """
    logger.info("="*50)
    logger.info("训练ML模型...")
    logger.info("="*50)
    
    models = {}
    
    # 1. 训练随机森林
    if model_type in ['random_forest', 'both']:
        print("\n[1/2] 训练随机森林模型...")
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
            
            models['random_forest'] = rf
            print(f"✓ 随机森林训练完成\n")
            
        except ImportError:
            logger.error("请安装scikit-learn: pip install scikit-learn")
            raise
    
    # 2. 训练XGBoost
    if model_type in ['xgboost', 'both']:
        print("[2/2] 训练XGBoost模型...")
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
            
            num_rounds = 100
            bst = xgb.train(params, dtrain, num_rounds)
            
            y_pred = bst.predict(dtest)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"  XGBoost - MSE: {mse:.6f}, R2: {r2:.4f}")
            print(f"  特征重要性:")
            importances = bst.get_score(importance_type='weight')
            for col, imp in importances.items():
                print(f"    {col}: {imp:.4f}")
            
            models['xgboost'] = bst
            print(f"✓ XGBoost训练完成\n")
            
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
    print("\n[1/3] 加载ML模型...")
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
    
    # 2. 获取所有A股列表并计算因子
    print("[2/3] 计算因子...")
    stocks = get_all_a_stocks()
    
    results = []
    factor_cols = None
    
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
    
    # 3. 标准化因子
    factor_cols = [col for col in factors_df.columns if col != 'code']
    normalized_df = normalize_factors(factors_df[factor_cols])
    normalized_df['code'] = factors_df['code'].values
    
    # 4. 使用ML模型预测
    print("[3/3] 使用ML模型预测...")
    X_pred = normalized_df[factor_cols]
    
    all_predictions = []
    
    for model_name, model in models.items():
        y_pred = model.predict(X_pred)
        
        temp_df = normalized_df[['code']].copy()
        temp_df[f'{model_name}_score'] = y_pred
        
        all_predictions.append(temp_df)
    
    # 合并所有模型的预测结果
    result_df = all_predictions[0]
    for df in all_predictions[1:]:
        result_df = result_df.merge(df, on='code', how='outer')
    
    # 计算平均得分
    if len(all_predictions) > 1:
        score_cols = [col for col in result_df.columns if col.endswith('_score')]
        result_df['ml_score'] = result_df[score_cols].mean(axis=1)
    else:
        result_df['ml_score'] = result_df[f'{list(models.keys())[0]}_score']
    
    # 选出TOP N
    top_stocks = result_df.nlargest(top_n, 'ml_score')
    
    print(f"✓ ML选股完成: TOP {top_n} 股票\n")
    print(top_stocks[['code', 'ml_score']].to_string(index=False))
    print()
    
    return top_stocks


def save_models(models: Dict, output_dir: str = 'data/models'):
    """保存训练好的模型"""
    os.makedirs(output_dir, exist_ok=True)
    
    for model_name, model in models.items():
        model_path = os.path.join(output_dir, f'{model_name}_model.pkl')
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"✓ 模型已保存: {model_path}")


def run_ml_stock_selection(train_end_date: str = '20191231', 
                          pred_date: str = '20200101', 
                          top_n: int = 10,
                          model_type: str = 'both',
                          output_file: str = 'data/results/top10_stocks_ml.csv'):
    """
    运行完整的ML选股流程
    """
    logger.remove()
    logger.add(sys.stdout, level="INFO", 
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")
    
    logger.info("="*70)
    logger.info("机器学习选股系统（多时间点训练）")
    logger.info("="*70 + "\n")
    
    try:
        # 1. 准备训练数据（多时间点）
        X, y = prepare_training_data_multi_period(end_date=train_end_date, periods=6)
        
        # 2. 训练模型
        models = train_ml_models(X, y, model_type=model_type)
        
        # 3. 保存模型
        save_models(models)
        
        # 4. 使用ML模型选股
        top_stocks = select_stocks_with_ml(pred_date=pred_date, top_n=top_n)
        
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
    from datetime import datetime, timedelta
    
    # 运行ML选股
    top_stocks = run_ml_stock_selection(
        train_end_date='20191231',  # 使用2019年及之前的数据训练
        pred_date='20200101',        # 预测2020年的股票
        top_n=10,                     # 选择TOP 10
        model_type='both'             # 使用随机森林和XGBoost
    )
    
    print("="*70)
    print(f"ML选股结果 - TOP 10 股票代码: {top_stocks}")
    print("="*70)
