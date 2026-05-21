#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ML选股测试脚本
验证ML模块是否正常工作
"""

import sys
import os

# 添加项目根目录到路径
project_root = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from ml_stock_selector import MLStockSelector, run_ml_stock_selection
from loguru import logger


def test_ml_selector():
    """测试ML选股功能"""
    logger.remove()
    logger.add(sys.stdout, level="INFO", 
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")
    
    logger.info("="*70)
    logger.info("测试ML选股功能")
    logger.info("="*70 + "\n")
    
    try:
        # 1. 测试导入
        logger.info("[1/4] 测试导入...")
        selector = MLStockSelector(model_type='random_forest')
        logger.info("✓ 导入成功\n")
        
        # 2. 测试训练数据准备（使用少量股票测试）
        logger.info("[2/4] 测试训练数据准备...")
        # 注意：这里会使用所有A股，可能需要较长时间
        # 建议使用小规模测试数据
        logger.warning("注意：准备训练数据可能需要较长时间（遍历所有A股）")
        logger.warning("建议：使用小规模测试数据或已训练好的模型\n")
        
        # 3. 测试模型训练（如果有训练数据）
        logger.info("[3/4] 测试模型训练...")
        logger.warning("跳过完整训练（需要大量数据）")
        logger.warning("建议：先运行完整训练脚本，或加载已保存的模型\n")
        
        # 4. 测试模型加载和预测
        logger.info("[4/4] 测试模型加载和预测...")
        try:
            selector.load_models()
            logger.info("✓ 模型加载成功")
            
            # 使用ML模型选股
            top_stocks = selector.select_stocks_with_ml(pred_date='20200101', top_n=10)
            
            logger.info("✓ ML选股成功")
            logger.info(f"TOP 10 股票: {top_stocks['code'].tolist()}\n")
            
        except FileNotFoundError as e:
            logger.warning(f"未找到已训练的模型: {e}")
            logger.warning("请先运行训练脚本: python ml_stock_selector.py\n")
        
        logger.info("="*70)
        logger.info("测试完成")
        logger.info("="*70)
        
        return True
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_quick_test():
    """快速测试（使用少量数据）"""
    logger.remove()
    logger.add(sys.stdout, level="INFO", 
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")
    
    logger.info("="*70)
    logger.info("快速测试ML选股（使用少量数据）")
    logger.info("="*70 + "\n")
    
    try:
        from select_top5_stocks import get_all_a_stocks
        
        # 1. 获取少量股票用于测试
        logger.info("[1/3] 获取测试股票列表...")
        all_stocks = get_all_a_stocks()
        test_stocks = all_stocks[:50]  # 仅使用前50只股票进行测试
        logger.info(f"✓ 测试股票数量: {len(test_stocks)}\n")
        
        # 2. 准备训练数据（仅使用测试股票）
        logger.info("[2/3] 准备训练数据（测试模式）...")
        selector = MLStockSelector(model_type='random_forest')
        
        # 手动准备小规模训练数据
        from select_top5_stocks import calculate_technical_factors, get_fundamental_factors, normalize_factors, DB_PATH
        import pandas as pd
        import numpy as np
        
        results = []
        for code in test_stocks:
            tech_factors = selector._calculate_factors_with_date(code, '20191231')
            if tech_factors is None:
                continue
            
            future_return = selector._calculate_future_return(code, '20191231', days=20)
            if future_return is None:
                continue
            
            result = {'code': code, 'future_return': future_return}
            result.update(tech_factors)
            results.append(result)
        
        if len(results) < 10:
            logger.error("测试数据不足！")
            return False
        
        df = pd.DataFrame(results)
        factor_cols = [col for col in df.columns if col not in ['code', 'future_return']]
        normalized_df = normalize_factors(df[factor_cols])
        normalized_df['code'] = df['code'].values
        normalized_df['future_return'] = df['future_return'].values
        
        X = normalized_df[factor_cols]
        y = normalized_df['future_return']
        
        selector.feature_columns = factor_cols
        
        logger.info(f"✓ 训练数据准备完成: X={X.shape}, y={y.shape}\n")
        
        # 3. 训练模型
        logger.info("[3/3] 训练模型...")
        models = selector.train_models(X, y)
        
        logger.info("✓ 模型训练完成\n")
        
        # 4. 测试预测
        logger.info("[4/4] 测试预测...")
        top_stocks = selector.select_stocks_with_ml(pred_date='20200101', top_n=10)
        
        logger.info("✓ 预测完成")
        logger.info(f"TOP 10 股票: {top_stocks['code'].tolist()}\n")
        
        logger.info("="*70)
        logger.info("快速测试完成！")
        logger.info("="*70)
        
        return True
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    # 运行快速测试
    success = run_quick_test()
    
    if success:
        print("\n" + "="*70)
        print("✓ 测试通过！ML模块可以正常工作")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("✗ 测试失败！请检查错误信息")
        print("="*70)
