"""
多因子选股脚本 - 选择TOP 5/10股票
支持规则选股和ML选股（随机森林、XGBoost）
技术因子 + 基本面因子，以2020-01-01为基准点
"""

import sqlite3
import pandas as pd
import numpy as np
from loguru import logger
from typing import List, Dict, Optional
import os
import sys
from datetime import datetime, timedelta

# 添加项目根目录到路径
project_root = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from select_top5_stocks import get_all_a_stocks, calculate_technical_factors, get_fundamental_factors, normalize_factors, DB_PATH


class MLStockSelector:
    """机器学习选股器（集成版）"""
    
    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.model = None
        self.feature_columns = []
        
        logger.info(f"MLStockSelector initialized: model_type={model_type}")
    
    def prepare_training_data(self, train_end_date: str = '20181231', sample_stocks: int = 50) -> tuple:
        """
        准备训练数据（简化版 - 使用部分股票）
        """
        logger.info("="*50)
        logger.info("准备ML训练数据（简化版）...")
        logger.info("="*50)
        
        # 获取所有A股列表
        print("\n[1/4] 获取A股列表...")
        stocks = get_all_a_stocks()
        
        # 仅使用部分股票（加速测试）
        stocks = stocks[:sample_stocks]
        print(f"✓ 使用 {len(stocks)} 只股票进行训练\n")
        
        # 计算因子和未来收益率
        print("[2/4] 计算因子和未来收益率...")
        results = []
        
        for i, code in enumerate(stocks):
            if i % 10 == 0:
                print(f"  进度: {i}/{len(stocks)}...")
            
            # 计算因子
            tech_factors = calculate_technical_factors(code, DB_PATH, train_end_date)
            fund_factors = get_fundamental_factors(code, DB_PATH, train_end_date)
            
            if tech_factors is None or fund_factors is None:
                continue
            
            # 计算未来20日收益率
            future_return = self._calculate_future_return(code, train_end_date, days=20)
            
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
        print(f"✓ 数据准备完成: {len(df)} 条记录\n")
        
        # 标准化因子
        print("[3/4] 标准化因子...")
        factor_cols = [col for col in df.columns if col not in ['code', 'future_return']]
        normalized_df = normalize_factors(df[factor_cols])
        normalized_df['code'] = df['code'].values
        normalized_df['future_return'] = df['future_return'].values
        
        # 分离特征和标签
        print("[4/4] 分离特征和标签...")
        X = normalized_df[factor_cols]
        y = normalized_df['future_return']
        
        self.feature_columns = factor_cols
        
        print(f"✓ 特征维度: {X.shape}")
        print(f"✓ 标签维度: {y.shape}\n")
        
        return X, y
    
    def train_model(self, X: pd.DataFrame, y: pd.Series):
        """训练ML模型"""
        logger.info("="*50)
        logger.info("训练ML模型...")
        logger.info("="*50)
        
        if self.model_type == 'random_forest':
            print("\n训练随机森林模型...")
            self.model = self._train_random_forest(X, y)
            print(f"✓ 随机森林训练完成\n")
            
        elif self.model_type == 'xgboost':
            print("\n训练XGBoost模型...")
            self.model = self._train_xgboost(X, y)
            print(f"✓ XGBoost训练完成\n")
            
        logger.info(f"✓ 模型训练完成")
        
        return self.model
    
    def select_stocks_with_ml(self, pred_date: str = '20200101', top_n: int = 10) -> pd.DataFrame:
        """使用ML模型选股"""
        logger.info("="*50)
        logger.info("使用ML模型选股...")
        logger.info("="*50)
        
        if self.model is None:
            raise ValueError("模型未训练！请先调用 train_model()")
        
        # 获取所有A股列表
        print("\n[1/3] 获取A股列表...")
        stocks = get_all_a_stocks()
        print(f"✓ 获取到 {len(stocks)} 只股票\n")
        
        # 计算因子
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
        
        # 标准化因子
        factor_cols = [col for col in factors_df.columns if col != 'code']
        normalized_df = normalize_factors(factors_df[factor_cols])
        normalized_df['code'] = factors_df['code'].values
        
        # 使用ML模型预测
        print("[3/3] 使用ML模型预测...")
        X_pred = normalized_df[self.feature_columns]
        y_pred = self.model.predict(X_pred)
        
        # 合并结果
        result_df = normalized_df[['code']].copy()
        result_df['ml_score'] = y_pred
        
        # 选出TOP N
        top_stocks = result_df.nlargest(top_n, 'ml_score')
        
        print(f"✓ ML选股完成: TOP {top_n} 股票\n")
        print(top_stocks[['code', 'ml_score']].to_string(index=False))
        print()
        
        return top_stocks
    
    def _calculate_future_return(self, code: str, base_date: str, days: int = 20) -> Optional[float]:
        """计算未来N日收益率（用于训练标签）"""
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
            
            # 特征重要性
            importances = rf.feature_importances_
            print(f"  特征重要性:")
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
            
            # 特征重要性
            importances = bst.get_score(importance_type='weight')
            print(f"  特征重要性:")
            for col, imp in importances.items():
                print(f"    {col}: {imp:.4f}")
            
            return bst
            
        except ImportError:
            logger.error("请安装xgboost: pip install xgboost")
            raise
    

# 为了向后兼容，保留原来的函数
def select_top5_stocks(output_file: str = 'data/results/top5_stocks.csv'):
    """执行选股，选出TOP 5股票（原来的规则选股）"""
    # ... (保留原来的逻辑)
    pass


def run_ml_stock_selection(train_end_date: str = '20181231', 
                        pred_date: str = '20200101', 
                        top_n: int = 10,
                        model_type: str = 'random_forest',
                        output_file: str = 'data/results/top10_stocks_ml.csv'):
    """
    运行完整的ML选股流程
    """
    logger.remove()
    logger.add(sys.stdout, level="INFO", 
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")
    
    logger.info("="*70)
    logger.info("机器学习选股系统（集成版）")
    logger.info("="*70 + "\n")
    
    try:
        # 1. 初始化选股器
        selector = MLStockSelector(model_type=model_type)
        
        # 2. 准备训练数据（使用50只股票加速）
        X, y = selector.prepare_training_data(train_end_date=train_end_date, sample_stocks=50)
        
        # 3. 训练模型
        selector.train_model(X, y)
        
        # 4. 使用ML模型选股
        top_stocks = selector.select_stocks_with_ml(pred_date=pred_date, top_n=top_n)
        
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
        train_end_date='20181231',  # 使用2018年及之前的数据训练
        pred_date='20200101',        # 预测2020年的股票
        top_n=10,                     # 选择TOP 10
        model_type='random_forest'    # 使用随机森林
    )
    
    print("="*70)
    print(f"ML选股结果 - TOP 10 股票代码: {top_stocks}")
    print("="*70)

# 数据库路径
DB_PATH = "D:/tu-shareData/astock_daily.db"


def get_all_a_stocks() -> List[str]:
    """获取所有A股代码列表"""
    conn = sqlite3.connect(DB_PATH)
    
    query = """
        SELECT ts_code FROM stock_basic
        WHERE list_date <= '20200101'
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


def calculate_technical_factors(code: str, db_path: str, base_date: str = '20200101') -> Dict:
    """计算技术因子"""
    conn = sqlite3.connect(db_path)
    
    # 获取价格数据（从基准日期前60天开始，用于计算均线）
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
        AND d.trade_date BETWEEN '20190101' AND '20201231'
        ORDER BY d.trade_date
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if len(df) < 60:
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
    
    return factors if factors else None


def get_fundamental_factors(code: str, db_path: str, date: str = '20201231') -> Dict:
    """获取基本面因子"""
    conn = sqlite3.connect(db_path)
    
    # 获取基本面数据（2020年年报）
    query = f"""
        SELECT pe, pb, ps, dv_ratio
        FROM daily_basic
        WHERE ts_code = (
            SELECT CASE 
                WHEN SUBSTR('{code}', 1, 1) IN ('6') THEN '{code}.SH'
                ELSE '{code}.SZ'
            END
        )
        AND trade_date = '{date}'
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if len(df) == 0:
        return None
    
    factors = {}
    
    # 1. 市盈率（PE）-- 越低越好
    if not pd.isna(df['pe'].iloc[0]) and df['pe'].iloc[0] > 0:
        factors['pe'] = -df['pe'].iloc[0]  # 负值表示越低越好
    
    # 2. 市净率（PB）-- 越低越好
    if not pd.isna(df['pb'].iloc[0]) and df['pb'].iloc[0] > 0:
        factors['pb'] = -df['pb'].iloc[0]
    
    # 3. 市销率（PS）-- 越低越好
    if not pd.isna(df['ps'].iloc[0]) and df['ps'].iloc[0] > 0:
        factors['ps'] = -df['ps'].iloc[0]
    
    # 4. 股息率 -- 越高越好
    if not pd.isna(df['dv_ratio'].iloc[0]):
        factors['dv_ratio'] = df['dv_ratio'].iloc[0]
    
    return factors if factors else None


def normalize_factors(factors_df: pd.DataFrame) -> pd.DataFrame:
    """标准化因子（z-score）"""
    normalized = factors_df.copy()
    
    # 对每列进行z-score标准化
    for col in factors_df.columns:
        if col == 'code':
            continue
        mean_val = factors_df[col].mean()
        std_val = factors_df[col].std()
        if std_val > 0:
            normalized[col] = (factors_df[col] - mean_val) / std_val
        else:
            normalized[col] = 0
    
    return normalized


def select_top5_stocks(output_file: str = 'data/results/top5_stocks.csv'):
    """执行选股，选出TOP 5股票"""
    logger.info("="*50)
    logger.info("开始多因子选股...")
    logger.info("="*50)
    
    # 1. 获取所有A股列表
    print("\n[1/4] 获取所有A股列表...")
    stocks = get_all_a_stocks()
    print(f"✓ 获取到 {len(stocks)} 只股票\n")
    
    # 2. 计算因子
    print("[2/4] 计算技术因子和基本面因子...")
    results = []
    
    for i, code in enumerate(stocks):
        if i % 20 == 0:
            print(f"  进度: {i}/{len(stocks)}...")
        
        # 计算技术因子
        tech_factors = calculate_technical_factors(code, DB_PATH)
        
        # 获取基本面因子
        fund_factors = get_fundamental_factors(code, DB_PATH)
        
        if tech_factors is None or fund_factors is None:
            continue
        
        # 合并因子
        result = {'code': code}
        result.update(tech_factors)
        result.update(fund_factors)
        results.append(result)
    
    factors_df = pd.DataFrame(results)
    print(f"✓ 因子计算完成: {len(factors_df)} 只股票\n")
    
    # 3. 标准化因子
    print("[3/4] 标准化因子...")
    normalized_df = normalize_factors(factors_df)
    
    # 4. 计算综合得分（等权重）
    factor_cols = [col for col in normalized_df.columns if col != 'code']
    normalized_df['total_score'] = normalized_df[factor_cols].sum(axis=1)
    
    # 5. 选出TOP 5
    print("[4/4] 选出TOP 5股票...")
    top5 = normalized_df.nlargest(5, 'total_score')[['code', 'total_score'] + factor_cols]
    
    print(f"✓ 选股完成: TOP 5 股票\n")
    print(top5.to_string(index=False))
    print()
    
    # 6. 保存结果
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    top5.to_csv(output_file, index=False, encoding='utf-8-sig')
    logger.info(f"结果已保存到: {output_file}")
    
    return top5['code'].tolist()


if __name__ == '__main__':
    # 初始化日志
    logger.remove()
    logger.add(sys.stdout, level="INFO", 
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")
    
    top5_codes = select_top5_stocks()
    
    print("="*70)
    print(f"TOP 5 股票代码: {top5_codes}")
    print("="*70)
