"""
机器学习选股模块 - 使用随机森林和XGBoost选股
基于多因子 + 未来收益率训练模型，预测股票排名
"""

import sqlite3
import pandas as pd
import numpy as np
from loguru import logger
from typing import List, Dict, Tuple, Optional
import os
import sys
from datetime import datetime, timedelta

# 添加项目根目录到路径
project_root = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from select_top5_stocks import get_all_a_stocks, calculate_technical_factors, get_fundamental_factors, normalize_factors, DB_PATH


class MLStockSelector:
    """机器学习选股器"""
    
    def __init__(self, model_type: str = 'random_forest', test_size: float = 0.2):
        """
        初始化ML选股器
        
        Args:
            model_type: 模型类型 ('random_forest', 'xgboost', 'both')
            test_size: 测试集比例
        """
        self.model_type = model_type
        self.test_size = test_size
        self.models = {}
        self.feature_columns = []
        
        logger.info(f"MLStockSelector initialized: model_type={model_type}")
    
    
    def prepare_training_data(self, train_end_date: str = '20191231') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        准备训练数据（因子 + 未来收益率标签）
        
        Args:
            train_end_date: 训练数据截止日期（如 '20191231'）
            
        Returns:
            (X, y): 特征DataFrame和标签Series
        """
        logger.info("="*50)
        logger.info("准备ML训练数据...")
        logger.info("="*50)
        
        # 1. 获取所有A股列表
        print("\n[1/5] 获取所有A股列表...")
        stocks = get_all_a_stocks()
        print(f"✓ 获取到 {len(stocks)} 只股票\n")
        
        # 2. 计算因子和未来收益率
        print("[2/5] 计算因子和未来收益率（用于训练标签）...")
        results = []
        
        for i, code in enumerate(stocks):
            if i % 50 == 0:
                print(f"  进度: {i}/{len(stocks)}...")
            
            # 计算因子（以train_end_date为基准）
            tech_factors = self._calculate_factors_with_date(code, train_end_date)
            
            if tech_factors is None:
                continue
            
            # 计算未来20日收益率（标签）
            future_return = self._calculate_future_return(code, train_end_date, days=20)
            
            if future_return is None:
                continue
            
            # 合并数据
            result = {'code': code, 'future_return': future_return}
            result.update(tech_factors)
            results.append(result)
        
        df = pd.DataFrame(results)
        print(f"✓ 数据准备完成: {len(df)} 只股票\n")
        
        # 3. 标准化因子
        print("[3/5] 标准化因子...")
        factor_cols = [col for col in df.columns if col not in ['code', 'future_return']]
        normalized_df = normalize_factors(df[factor_cols])
        normalized_df['code'] = df['code'].values
        normalized_df['future_return'] = df['future_return'].values
        
        # 4. 分离特征和标签
        print("[4/5] 分离特征和标签...")
        X = normalized_df[factor_cols]
        y = normalized_df['future_return']
        
        self.feature_columns = factor_cols
        
        print(f"✓ 特征维度: {X.shape}")
        print(f"✓ 标签维度: {y.shape}\n")
        
        return X, y
    
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        训练ML模型
        
        Args:
            X: 特征DataFrame
            y: 标签Series（未来收益率）
            
        Returns:
            训练好的模型字典 {'random_forest': model1, 'xgboost': model2}
        """
        logger.info("="*50)
        logger.info("训练ML模型...")
        logger.info("="*50)
        
        models = {}
        
        # 1. 训练随机森林
        if self.model_type in ['random_forest', 'both']:
            print("\n[5/5] 训练随机森林模型...")
            rf_model = self._train_random_forest(X, y)
            models['random_forest'] = rf_model
            print(f"✓ 随机森林训练完成\n")
        
        # 2. 训练XGBoost
        if self.model_type in ['xgboost', 'both']:
            print("[5/5] 训练XGBoost模型...")
            xgb_model = self._train_xgboost(X, y)
            models['xgboost'] = xgb_model
            print(f"✓ XGBoost训练完成\n")
        
        self.models = models
        
        logger.info(f"✓ 所有模型训练完成: {list(models.keys())}")
        
        return models
    
    
    def select_stocks_with_ml(self, pred_date: str = '20200101', top_n: int = 10) -> pd.DataFrame:
        """
        使用ML模型选股
        
        Args:
            pred_date: 预测基准日期（如 '20200101'）
            top_n: 选择前N只股票
            
        Returns:
            选股结果DataFrame（包含ML得分）
        """
        logger.info("="*50)
        logger.info("使用ML模型选股...")
        logger.info("="*50)
        
        if len(self.models) == 0:
            raise ValueError("模型未训练！请先调用 train_models()")
        
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
            
            # 计算因子
            tech_factors = self._calculate_factors_with_date(code, pred_date)
            
            if tech_factors is None:
                continue
            
            result = {'code': code}
            result.update(tech_factors)
            results.append(result)
        
        factors_df = pd.DataFrame(results)
        print(f"✓ 因子计算完成: {len(factors_df)} 只股票\n")
        
        # 3. 标准化因子
        factor_cols = [col for col in factors_df.columns if col != 'code']
        normalized_df = normalize_factors(factors_df[factor_cols])
        normalized_df['code'] = factors_df['code'].values
        
        # 4. 使用ML模型预测
        print("[3/3] 使用ML模型预测...")
        X_pred = normalized_df[self.feature_columns]
        
        all_predictions = []
        
        for model_name, model in self.models.items():
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
            result_df['ml_score'] = result_df[f'{self.model_type}_score']
        
        # 选出TOP N
        top_stocks = result_df.nlargest(top_n, 'ml_score')
        
        print(f"✓ ML选股完成: TOP {top_n} 股票\n")
        print(top_stocks[['code', 'ml_score']].to_string(index=False))
        print()
        
        return top_stocks
    
    
    def _calculate_factors_with_date(self, code: str, base_date: str) -> Optional[Dict]:
        """
        计算指定日期的因子（技术因子 + 基本面因子）
        
        Args:
            code: 股票代码
            base_date: 基准日期（YYYYMMDD格式）
            
        Returns:
            因子字典
        """
        conn = sqlite3.connect(DB_PATH)
        
        # 获取价格数据（从基准日期前60天开始）
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
            AND d.trade_date BETWEEN '20180101' AND '{base_date}'
            ORDER BY d.trade_date
        """
        
        df = pd.read_sql_query(query, conn)
        
        if len(df) < 60:
            conn.close()
            return None
        
        # 计算前复权价格
        latest_adj = df['adj_factor'].iloc[-1]
        df['adj_close'] = df['close'] * (df['adj_factor'] / latest_adj)
        
        # 技术因子
        factors = {}
        
        # 1. 动量因子（过去20日收益率）
        if len(df) >= 20:
            momentum_20 = (df['adj_close'].iloc[-1] / df['adj_close'].iloc[-20] - 1)
            factors['momentum_20'] = momentum_20
        
        # 2. 波动率因子（过去20日收益率标准差）
        if len(df) >= 20:
            returns = df['adj_close'].pct_change().iloc[-20:]
            volatility = returns.std()
            factors['volatility'] = -volatility  # 波动率越小越好
        
        # 3. 均线因子（当前价格与MA60的比率）
        df['ma60'] = df['adj_close'].rolling(window=60).mean()
        if not pd.isna(df['ma60'].iloc[-1]):
            ma_ratio = df['adj_close'].iloc[-1] / df['ma60'].iloc[-1]
            factors['ma_ratio'] = ma_ratio  # 价格在MA60上方越好
        
        # 获取基本面因子
        # 转换base_date为前一年年报日期
        year = int(base_date[:4]) - 1
        report_date = f"{year}1231"
        
        query_fund = f"""
            SELECT pe, pb, ps, dv_ratio
            FROM daily_basic
            WHERE ts_code = (
                SELECT CASE 
                    WHEN SUBSTR('{code}', 1, 1) IN ('6') THEN '{code}.SH'
                    ELSE '{code}.SZ'
                END
            )
            AND trade_date = '{report_date}'
        """
        
        df_fund = pd.read_sql_query(query_fund, conn)
        conn.close()
        
        if len(df_fund) == 0:
            return None
        
        # 基本面因子
        if not pd.isna(df_fund['pe'].iloc[0]) and df_fund['pe'].iloc[0] > 0:
            factors['pe'] = -df_fund['pe'].iloc[0]  # 市盈率越低越好
        
        if not pd.isna(df_fund['pb'].iloc[0]) and df_fund['pb'].iloc[0] > 0:
            factors['pb'] = -df_fund['pb'].iloc[0]  # 市净率越低越好
        
        if not pd.isna(df_fund['ps'].iloc[0]) and df_fund['ps'].iloc[0] > 0:
            factors['ps'] = -df_fund['ps'].iloc[0]  # 市销率越低越好
        
        if not pd.isna(df_fund['dv_ratio'].iloc[0]):
            factors['dv_ratio'] = df_fund['dv_ratio'].iloc[0]  # 股息率越高越好
        
        return factors if factors else None
    
    
    def _calculate_future_return(self, code: str, base_date: str, days: int = 20) -> Optional[float]:
        """
        计算未来N日收益率（用于训练标签）
        
        Args:
            code: 股票代码
            base_date: 基准日期
            days: 未来天数（如20日、60日）
            
        Returns:
            未来N日收益率（小数形式，如0.15表示15%）
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
    
    
    def _train_random_forest(self, X: pd.DataFrame, y: pd.Series):
        """训练随机森林模型"""
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, r2_score
            
            # 划分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=42
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
            for col, imp in zip(X.columns, importances):
                print(f"    {col}: {imp:.4f}")
            
            return rf
            
        except ImportError:
            logger.error("请安装scikit-learn: pip install scikit-learn")
            raise
    
    
    def _train_xgboost(self, X: pd.DataFrame, y: pd.Series):
        """训练XGBoost模型"""
        try:
            import xgboost as xgb
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, r2_score
            
            # 划分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=42
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
            for col, imp in importances.items():
                print(f"    {col}: {imp:.4f}")
            
            return bst
            
        except ImportError:
            logger.error("请安装xgboost: pip install xgboost")
            raise
    
    
    def save_models(self, output_dir: str = 'data/models'):
        """保存训练好的模型"""
        import pickle
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = os.path.join(output_dir, f'{model_name}_model.pkl')
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            logger.info(f"✓ 模型已保存: {model_path}")
    
    
    def load_models(self, model_dir: str = 'data/models'):
        """加载已训练的模型"""
        import pickle
        import os
        
        self.models = {}
        
        for model_name in ['random_forest', 'xgboost']:
            model_path = os.path.join(model_dir, f'{model_name}_model.pkl')
            
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
                
                logger.info(f"✓ 模型已加载: {model_path}")
        
        if len(self.models) == 0:
            raise FileNotFoundError(f"未找到模型文件: {model_dir}")
        
        return self.models
    


def run_ml_stock_selection(train_end_date: str = '20191231', 
                          pred_date: str = '20200101', 
                          top_n: int = 10,
                          model_type: str = 'both',
                          output_file: str = 'data/results/top10_stocks_ml.csv'):
    """
    运行完整的ML选股流程
    
    Args:
        train_end_date: 训练数据截止日期
        pred_date: 预测基准日期
        top_n: 选择前N只股票
        model_type: 模型类型 ('random_forest', 'xgboost', 'both')
        output_file: 输出文件路径
    """
    logger.remove()
    logger.add(sys.stdout, level="INFO", 
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")
    
    logger.info("="*70)
    logger.info("机器学习选股系统")
    logger.info("="*70 + "\n")
    
    # 1. 初始化选股器
    selector = MLStockSelector(model_type=model_type)
    
    # 2. 准备训练数据
    X, y = selector.prepare_training_data(train_end_date=train_end_date)
    
    # 3. 训练模型
    selector.train_models(X, y)
    
    # 4. 保存模型
    selector.save_models()
    
    # 5. 使用ML模型选股
    top_stocks = selector.select_stocks_with_ml(pred_date=pred_date, top_n=top_n)
    
    # 6. 保存结果
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    top_stocks.to_csv(output_file, index=False, encoding='utf-8-sig')
    logger.info(f"✓ ML选股结果已保存: {output_file}")
    
    return top_stocks['code'].tolist()


if __name__ == '__main__':
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
