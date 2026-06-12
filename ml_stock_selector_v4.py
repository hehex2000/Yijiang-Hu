"""
ML选股模块 v4 - 改进版
======================
改进点：
1. 标签定义：从回归（预测收益率）改为分类（预测是否未来收益率TOP 30%）
2. 训练数据时间范围：2010-2019年（避免2022年数据泄露）
3. 增加训练样本量：使用更多时间点（每半年一个时间点，共20个时间点）
4. 提升模型性能：使用分类模型、调参、交叉验证
5. 确保从本地数据库读取数据（速度快）

使用方法：
  python ml_stock_selector_v4.py
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# 修复Windows下的编码问题
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except:
        pass

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from loguru import logger
import sqlite3
from datetime import datetime, timedelta
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 检查ML库是否可用
try:
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    import xgboost as xgb
    SKLEARN_AVAILABLE = True
    XGBOOST_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    XGBOOST_AVAILABLE = False
    logger.warning("scikit-learn or xgboost not installed")

# 导入现有系统的模块
try:
    from src.data_fetcher import DataFetcher
    from src.factor_calculator import FactorCalculator
    from src.factor_processor import FactorProcessor
    FROM_EXISTING = True
    logger.info("✓ 成功导入现有系统模块")
except ImportError as e:
    FROM_EXISTING = False
    logger.warning(f"无法导入现有系统模块: {e}")


class MLStockSelector:
    """ML选股器（改进版 - 使用分类模型）"""
    
    def __init__(self, config=None, data_fetcher=None):
        """
        初始化ML选股器
        
        Args:
            config: 配置字典（可选）
            data_fetcher: 外部传入的 DataFetcher 实例（可选）
        """
        self.config = config or {}
        self.models = {}
        self.feature_columns = []
        self.task_type = "classification"  # 新增：任务类型
        
        # 如果外部传入了 data_fetcher，直接使用
        if data_fetcher is not None:
            self.data_fetcher = data_fetcher
            logger.info("✓ 使用外部传入的 data_fetcher（不创建新实例）")
            
            # 初始化 factor_calculator 和 factor_processor
            if FROM_EXISTING:
                self.factor_calculator = FactorCalculator()
                self.factor_calculator.enable_money_flow = False
                self.factor_processor = FactorProcessor()
                logger.info("✓ FactorCalculator 和 FactorProcessor 初始化成功")
            
            # 尝试自动加载已有模型
            self._auto_load_models()
            logger.info("MLStockSelector initialized (external data_fetcher)")
            return
        
        # 否则，按原逻辑初始化
        logger.info("未传入 data_fetcher，使用默认配置初始化...")
        if FROM_EXISTING:
            try:
                import config as cfg
                data_config = {
                    "primary_source": cfg.DATA["primary_source"],
                    "local_db_path": cfg.DATA["local_db_path"],
                    "tushare_token": cfg.DATA["tushare_token"],
                    "use_akshare_backup": cfg.DATA["use_akshare_backup"],
                    "use_tushare_backup": cfg.DATA["use_tushare_backup"],
                }
            except:
                # 如果导入config失败，使用默认配置（禁用备份）
                data_config = {
                    "primary_source": "local_db",
                    "local_db_path": r"D:\tu-sharedata\astock_daily.db",
                    "tushare_token": None,
                    "use_akshare_backup": False,  # ← 禁用AkShare备份
                    "use_tushare_backup": False,  # ← 禁用Tushare备份
                }
            
            self.data_fetcher = DataFetcher(**data_config)
            self.factor_calculator = FactorCalculator()
            self.factor_calculator.enable_money_flow = False
            self.factor_processor = FactorProcessor()
            logger.info("✓ 现有系统模块初始化成功（默认配置）")
        
        logger.info("MLStockSelector initialized")
    
    def prepare_training_data_multi_period(self, start_date="20160701", end_date="20211231", 
                                         periods=20, period_months=6, lookback_days=20,
                                         stock_pool="hs300", stock_codes=None):
        """
        准备多时间点训练数据（改进版）
        
        Args:
            start_date: 训练数据起始日期（YYYYMMDD）
            end_date: 训练数据截止日期（YYYYMMDD）
            periods: 使用过去N个时间点
            period_months: 每个时间点间隔月数
            lookback_days: 计算因子用的历史数据天数
            stock_pool: 股票池（"hs300", "zz500", "zz800", "all"）
            stock_codes: 可选，指定股票代码列表（用于测试）
            
        Returns:
            X: 特征矩阵（DataFrame）
            y: 标签（分类：是否未来收益率TOP 30%，Series）
        """
        if stock_codes:
            logger.info(f"准备多时间点训练数据: {start_date}~{end_date}, periods={periods}, 自定义{len(stock_codes)}只股票")
            return self._prepare_multi_period_with_codes(
                start_date, end_date, periods, period_months, lookback_days, stock_codes
            )
        
        logger.info(f"准备多时间点训练数据: {start_date}~{end_date}, periods={periods}")
        
        # 解析日期范围
        start_dt = datetime.strptime(start_date, "%Y%m%d")
        end_dt = datetime.strptime(end_date, "%Y%m%d")
        
        all_X = []
        all_y = []
        
        # 生成多个时间点（从end_date向前推）
        for i in range(periods):
            # 计算当前时间点的日期
            months_back = i * period_months
            point_date = end_dt - pd.DateOffset(months=months_back)
            
            # 确保不早于start_date
            if point_date < start_dt:
                break
            
            point_date_str = point_date.strftime("%Y%m%d")
            
            logger.info(f"处理时间点 {i+1}/{periods}: {point_date_str}")
            
            # 获取该时间点的训练数据
            X_point, y_point = self._prepare_single_period(
                point_date_str, lookback_days, stock_pool
            )
            
            if X_point is not None and y_point is not None and len(X_point) > 0:
                all_X.append(X_point)
                all_y.append(y_point)
        
        # 合并所有时间点的数据
        if len(all_X) == 0:
            logger.error("没有生成任何训练数据！")
            return None, None
        
        X_all = pd.concat(all_X, ignore_index=True)
        y_all = pd.concat(all_y, ignore_index=True)
        
        logger.info(f"✓ 训练数据准备完成: {len(X_all)} 个样本, {len(X_all.columns)} 个特征")
        
        return X_all, y_all
    
    def _prepare_multi_period_with_codes(self, start_date, end_date, periods, period_months, lookback_days, stock_codes):
        """准备多时间点训练数据（使用指定的股票列表）"""
        start_dt = datetime.strptime(start_date, "%Y%m%d")
        end_dt = datetime.strptime(end_date, "%Y%m%d")
        
        all_X = []
        all_y = []
        
        # 生成多个时间点
        for i in range(periods):
            months_back = i * period_months
            point_date = end_dt - pd.DateOffset(months=months_back)
            
            if point_date < start_dt:
                break
            
            point_date_str = point_date.strftime("%Y%m%d")
            
            logger.info(f"  处理时间点 {i+1}/{periods}: {point_date_str}")
            
            X_point, y_point = self._prepare_single_period_with_codes(
                point_date_str, lookback_days, stock_codes
            )
            
            if X_point is not None and y_point is not None and len(X_point) > 0:
                all_X.append(X_point)
                all_y.append(y_point)
        
        if len(all_X) == 0:
            logger.error("没有生成任何训练数据！")
            return None, None
        
        X_all = pd.concat(all_X, ignore_index=True)
        y_all = pd.concat(all_y, ignore_index=True)
        
        logger.info(f"✓ 训练数据准备完成: {len(X_all)} 个样本, {len(X_all.columns)} 个特征")
        
        return X_all, y_all
    
    def _prepare_single_period_with_codes(self, date, lookback_days, stock_codes):
        """准备单个时间点的训练数据（使用指定的股票列表）"""
        logger.info(f"  计算因子（指定股票，基准日期: {date}）...")
        
        logger.info(f"  股票池: {len(stock_codes)} 只股票（自定义）")
        
        # 计算因子
        factors_df = self.factor_calculator.calculate_all_factors(
            stock_codes,
            self.data_fetcher,
            end_date=date
        )
        
        if factors_df is None or len(factors_df) == 0:
            logger.warning(f"时间点 {date}: 因子计算失败")
            return None, None
        
        logger.info(f"  ✓ 因子计算完成: {len(factors_df)} 只股票")
        
        # 处理因子
        processed_df = self.factor_processor.process(factors_df)
        
        if processed_df is None or len(processed_df) == 0:
            logger.warning(f"时间点 {date}: 因子处理失败")
            return None, None
        
        logger.info(f"  ✓ 因子处理完成: {len(processed_df)} 只股票")
        
        # 计算未来收益率（标签）
        logger.info(f"  计算未来收益率（标签）...")
        returns = self._calculate_future_returns(stock_codes, date)
        
        if returns is None or len(returns) == 0:
            logger.warning(f"时间点 {date}: 未来收益率计算失败")
            return None, None
        
        # 合并特征和标签
        merged = processed_df.merge(returns, on='code', how='inner')
        
        if len(merged) == 0:
            logger.warning(f"时间点 {date}: 合并后无数据")
            return None, None
        
        # 提取特征和标签
        feature_cols = [col for col in processed_df.columns if col not in ['code', 'name']]
        X = merged[feature_cols]
        y = merged['label_cls']  # 新增：分类标签
        
        logger.info(f"  ✓ 时间点 {date} 完成: {len(X)} 个样本")
        
        return X, y
    
    def _prepare_single_period(self, date, lookback_days=20, stock_pool="hs300"):
        """准备单个时间点的训练数据"""
        # 获取股票池（支持指定日期）
        try:
            if stock_pool == "hs300":
                stocks_df = self.data_fetcher.get_hs300_components(date=date)
            elif stock_pool == "zz500":
                stocks_df = self.data_fetcher.get_zz500_components(date=date)
            elif stock_pool == "zz800":
                stocks_df = self.data_fetcher.get_zz800_components(date=date)
            else:
                logger.error(f"不支持的股票池: {stock_pool}")
                return None, None
        except Exception as e:
            logger.warning(f"时间点 {date}: 获取成分股失败 ({e})，跳过该时间点")
            return None, None
        
        if stocks_df is None or len(stocks_df) == 0:
            logger.warning(f"时间点 {date}: 股票池为空")
            return None, None
        
        stock_codes = stocks_df["code"].tolist()
        
        # 计算因子（指定基准日期）
        factors_df = self.factor_calculator.calculate_all_factors(
            stock_codes, 
            self.data_fetcher,
            end_date=date
        )
        
        if factors_df is None or len(factors_df) == 0:
            logger.warning(f"时间点 {date}: 因子计算失败")
            return None, None
        
        # 处理因子（标准化等）
        processed_df = self.factor_processor.process(factors_df)
        
        if processed_df is None or len(processed_df) == 0:
            logger.warning(f"时间点 {date}: 因子处理失败")
            return None, None
        
        # 计算标签：未来20日收益率（分类标签）
        future_returns = self._calculate_future_returns(stock_codes, date, days=20)
        
        if future_returns is None:
            logger.warning(f"时间点 {date}: 无法计算未来收益率")
            return None, None
        
        # 合并特征和标签
        merged = processed_df.merge(future_returns, on="code", how="inner")
        
        # 选择特征列
        factor_columns = [col for col in processed_df.columns 
                        if col.startswith(('VF', 'GF', 'QF', 'MF', 'TF', 'LVF', 'MWF')) or col == 'total_score']
        
        X = merged[factor_columns]
        y = merged["label_cls"]  # 分类标签
        
        logger.info(f"  时间点 {date}: {len(X)} 个样本")
        
        return X, y
    
    def _get_price_on_date(self, code, date, lookback_days=5):
        """获取指定日期或之前最近交易日的收盘价"""
        try:
            date_dt = datetime.strptime(date, "%Y%m%d")
            start_dt = date_dt - timedelta(days=lookback_days)
            start_date = start_dt.strftime("%Y%m%d")
            
            df = self.data_fetcher.get_stock_history(
                code, 
                start_date=start_date,
                end_date=date
            )
            
            if df is not None and len(df) > 0:
                if '日期' in df.columns:
                    df_sorted = df.sort_values('日期', ascending=False)
                elif 'trade_date' in df.columns:
                    df_sorted = df.sort_values('trade_date', ascending=False)
                else:
                    df_sorted = df.iloc[::-1]
                
                for col in ['收盘', 'close', 'Close', '收盘价']:
                    if col in df_sorted.columns:
                        return df_sorted.iloc[0][col]
            
            logger.debug(f"无法获取 {code} 在 {date} 附近的收盘价")
            return None
            
        except Exception as e:
            logger.debug(f"获取 {code} 在 {date} 的价格失败: {e}")
            return None
    
    def _calculate_future_returns(self, stock_codes, date, days=20):
        """
        计算未来收益率（作为ML标签）
        改进：同时计算回归标签和分类标签
        
        Args:
            stock_codes: 股票代码列表
            date: 当前日期（YYYYMMDD）
            days: 未来天数
            
        Returns:
            DataFrame with columns: code, future_return, label_cls
        """
        # 计算未来日期
        date_dt = datetime.strptime(date, "%Y%m%d")
        future_date = date_dt + timedelta(days=days)
        future_date_str = future_date.strftime("%Y%m%d")
        
        # 获取当前价格
        current_prices = {}
        future_prices = {}
        
        for code in stock_codes:
            current_price = self._get_price_on_date(code, date)
            if current_price is not None:
                current_prices[code] = current_price
            
            future_price = self._get_price_on_date(code, future_date_str)
            if future_price is not None:
                future_prices[code] = future_price
        
        # 计算收益率
        returns = []
        for code in stock_codes:
            if code in current_prices and code in future_prices:
                ret = (future_prices[code] - current_prices[code]) / current_prices[code]
                returns.append({"code": code, "future_return": ret})
        
        if len(returns) == 0:
            return None
        
        returns_df = pd.DataFrame(returns)
        
        # 新增：分类标签（未来收益率是否在前30%）
        if len(returns_df) > 10:  # 至少有10个样本才计算分位数
            threshold = returns_df["future_return"].quantile(0.7)
            returns_df["label_cls"] = (returns_df["future_return"] > threshold).astype(int)
        else:
            returns_df["label_cls"] = 0
        
        return returns_df
    
    def train_models(self, X, y, model_type="both", test_size=0.2):
        """
        训练ML模型（改进版 - 使用分类模型）
        
        Args:
            X: 特征矩阵
            y: 标签（分类：0/1）
            model_type: "random_forest", "xgboost", "both"
            test_size: 测试集比例
            
        Returns:
            trained_models: 训练好的模型字典
        """
        logger.info(f"开始训练分类模型: model_type={model_type}")
        
        # 保存特征列名和任务类型
        self.feature_columns = X.columns.tolist()
        self.task_type = "classification"
        
        # 检查标签分布
        n_positive = y.sum()
        n_total = len(y)
        logger.info(f"  标签分布: 正样本={n_positive} ({n_positive/n_total:.1%}), 负样本={n_total-n_positive} ({(n_total-n_positive)/n_total:.1%})")
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        trained_models = {}
        
        # 训练随机森林分类器
        if model_type in ["random_forest", "both"]:
            logger.info("训练随机森林分类器...")
            
            rf = RandomForestClassifier(
                n_estimators=200,  # 增加树的数量
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced"  # 处理类别不平衡
            )
            rf.fit(X_train, y_train)
            
            # 评估
            y_pred = rf.predict(X_test)
            y_pred_proba = rf.predict_proba(X_test)[:, 1]
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            logger.info(f"  随机森林 - Acc: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
            
            # 交叉验证
            cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='roc_auc')
            logger.info(f"  随机森林 - CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            
            trained_models["random_forest"] = rf
        
        # 训练XGBoost分类器
        if model_type in ["xgboost", "both"] and XGBOOST_AVAILABLE:
            logger.info("训练XGBoost分类器...")
            
            xgb_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                scale_pos_weight=(n_total - n_positive) / n_positive if n_positive > 0 else 1  # 处理类别不平衡
            )
            xgb_model.fit(X_train, y_train)
            
            # 评估
            y_pred = xgb_model.predict(X_test)
            y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            logger.info(f"  XGBoost - Acc: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
            
            # 交叉验证
            cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='roc_auc')
            logger.info(f"  XGBoost - CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            
            trained_models["xgboost"] = xgb_model
        
        self.models = trained_models
        
        logger.info(f"✓ 模型训练完成: {list(trained_models.keys())}")
        
        return trained_models
    
    def _auto_load_models(self, model_dir="data/models"):
        """自动加载已有模型（初始化时调用）"""
        model_path = Path(model_dir)
        if not model_path.exists():
            logger.info(f"模型目录不存在: {model_dir}，跳过自动加载")
            return
        
        loaded = 0
        # 尝试加载随机森林模型
        rf_path = model_path / "random_forest_model.pkl"
        if rf_path.exists():
            try:
                with open(rf_path, "rb") as f:
                    data = pickle.load(f)
                self.models["random_forest"] = data["model"]
                self.feature_columns = data.get("feature_columns", [])
                self.task_type = data.get("task_type", "classification")
                logger.info(f"✓ 已加载随机森林模型: {rf_path}")
                loaded += 1
            except Exception as e:
                logger.warning(f"加载随机森林模型失败: {e}")
        
        # 尝试加载XGBoost模型
        xgb_path = model_path / "xgboost_model.pkl"
        if xgb_path.exists():
            try:
                with open(xgb_path, "rb") as f:
                    data = pickle.load(f)
                self.models["xgboost"] = data["model"]
                if not self.feature_columns:
                    self.feature_columns = data.get("feature_columns", [])
                self.task_type = data.get("task_type", "classification")
                logger.info(f"✓ 已加载XGBoost模型: {xgb_path}")
                loaded += 1
            except Exception as e:
                logger.warning(f"加载XGBoost模型失败: {e}")
        
        if loaded > 0:
            logger.info(f"✓ 自动加载完成: {loaded} 个模型")
        else:
            logger.info(f"未找到已有模型文件（{model_dir}）")
    
    def select_stocks_with_ml(self, date, stock_pool="hs300", top_n=10, 
                               model_type="both", stock_codes=None):
        """
        使用ML模型选股（改进版 - 使用分类模型的概率输出）
        
        Args:
            date: 选股日期（YYYYMMDD）
            stock_pool: 股票池
            top_n: 选择TOP N股票
            model_type: 使用的模型类型
            stock_codes: 可选，指定股票代码列表
            
        Returns:
            DataFrame with columns: code, ml_score
        """
        logger.info(f"使用ML模型选股: date={date}, top_n={top_n}")
        
        if len(self.models) == 0:
            logger.error("模型未训练！请先运行 train_models()")
            return None
        
        # 获取股票池
        if stock_codes is not None:
            logger.info(f"  使用指定的 {len(stock_codes)} 只股票")
            stock_codes_list = stock_codes
        else:
            logger.info(f"  获取股票池: {stock_pool}")
            if stock_pool == "hs300":
                stocks_df = self.data_fetcher.get_hs300_components(date=date)
            elif stock_pool == "zz500":
                # 修正：使用中证500成分股（不是中证800）
                stocks_df = self.data_fetcher.get_zz500_components(date=date)
            elif stock_pool == "zz800":
                stocks_df = self.data_fetcher.get_zz800_components(date=date)
            else:
                logger.error(f"不支持的股票池: {stock_pool}")
                return None
            
            if stocks_df is None or len(stocks_df) == 0:
                logger.error("获取股票池失败！")
                return None
            
            # 保存股票池信息（含名称）供后续使用
            self._current_stocks_df = stocks_df
            
            stock_codes_list = stocks_df["code"].tolist()
        
        # 计算因子
        factors_df = self.factor_calculator.calculate_all_factors(
            stock_codes_list,
            self.data_fetcher,
            end_date=date
        )
        
        if factors_df is None or len(factors_df) == 0:
            logger.error("因子计算失败！")
            return None
        
        # 处理因子
        processed_df = self.factor_processor.process(factors_df)
        
        if processed_df is None or len(processed_df) == 0:
            logger.error("因子处理失败！")
            return None
        
        # 准备特征矩阵
        X = processed_df[self.feature_columns]
        
        # 使用模型预测（分类模型使用predict_proba）
        scores = {}
        
        for model_name, model in self.models.items():
            if model_type != "both" and model_name != model_type:
                continue
            
            # 分类模型：使用正样本的概率作为分数
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(X)
                if probabilities.shape[1] == 2:
                    predictions = probabilities[:, 1]  # 正样本概率
                else:
                    predictions = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]
            else:
                # 回归模型：直接使用预测值
                predictions = model.predict(X)
            
            scores[model_name] = predictions
            logger.debug(f"  [{model_name}] 预测分数范围: {predictions.min():.4f} ~ {predictions.max():.4f}")
        
        # 综合评分（平均）
        if len(scores) > 0:
            ml_scores = np.mean(list(scores.values()), axis=0)
            logger.info(f"  [调试] ML分数 top 10: {ml_scores[::-1][:10]}")
            logger.info(f"  [调试] 使用ML分数排序选股...")
        else:
            logger.error("没有可用的模型！")
            return None
        
        # 生成结果
        results = pd.DataFrame({
            "code": processed_df["code"],
            "ml_score": ml_scores
        })
        
        # 添加股票名称
        if hasattr(self, "_current_stocks_df") and self._current_stocks_df is not None:
            # 创建 code -> name 映射
            name_map = dict(zip(self._current_stocks_df["code"], self._current_stocks_df["name"]))
            results["name"] = results["code"].map(name_map)
            logger.info(f"✓ 已添加股票名称（{results['name'].notna().sum()} 只）")
        else:
            results["name"] = ""
        
        # 排序并选择TOP N
        results = results.sort_values("ml_score", ascending=False).head(top_n)
        
        logger.info(f"✓ ML选股完成: {len(results)} 只股票")
        
        return results
    
    def save_models(self, output_dir="data/models"):
        """保存训练好的模型"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = Path(output_dir) / f"{model_name}_model.pkl"
            
            with open(model_path, "wb") as f:
                pickle.dump({
                    "model": model,
                    "feature_columns": self.feature_columns,
                    "task_type": self.task_type,
                    "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }, f)
            
            logger.info(f"✓ 模型已保存: {model_path}")
    
    def load_models(self, input_dir="data/models"):
        """加载训练好的模型"""
        input_path = Path(input_dir)
        
        if not input_path.exists():
            logger.error(f"模型目录不存在: {input_dir}")
            return False
        
        self.models = {}
        
        # 加载随机森林
        rf_path = input_path / "random_forest_model.pkl"
        if rf_path.exists():
            with open(rf_path, "rb") as f:
                data = pickle.load(f)
                self.models["random_forest"] = data["model"]
                self.feature_columns = data["feature_columns"]
                self.task_type = data.get("task_type", "classification")
            logger.info(f"✓ 已加载随机森林模型: {rf_path}")
        
        # 加载XGBoost
        xgb_path = input_path / "xgboost_model.pkl"
        if xgb_path.exists():
            with open(xgb_path, "rb") as f:
                data = pickle.load(f)
                self.models["xgboost"] = data["model"]
                if len(self.feature_columns) == 0:
                    self.feature_columns = data["feature_columns"]
                self.task_type = data.get("task_type", "classification")
            logger.info(f"✓ 已加载XGBoost模型: {xgb_path}")
        
        if len(self.models) == 0:
            logger.error("没有找到任何模型文件！")
            return False
        
        logger.info(f"✓ 模型加载完成: {list(self.models.keys())}")
        return True
    

def run_ml_stock_selection(train_start_date="20100101", train_end_date="20191231", 
                            pred_date="20200101", periods=20, top_n=10, 
                            model_type="both", stock_pool="hs300", force_retrain=False):
    """
    运行完整的ML选股流程（改进版）
    
    Args:
        train_start_date: 训练数据起始日期
        train_end_date: 训练数据截止日期
        pred_date: 预测日期
        periods: 训练时间点数量
        top_n: 选择TOP N股票
        model_type: 模型类型
        stock_pool: 股票池
        force_retrain: 是否强制重新训练（覆盖已有模型）
        
    Returns:
        results: 选股结果DataFrame
    """
    logger.info("=" * 70)
    logger.info("ML选股流程开始（改进版）")
    logger.info("=" * 70)
    
    # 1. 初始化
    logger.info("[1/5] 初始化ML选股器...")
    ml_selector = MLStockSelector()
    logger.info("✓ 初始化完成\n")
    
    # 2. 检查是否有训练好的模型
    logger.info("[2/5] 检查模型文件...")
    if not force_retrain and ml_selector.load_models():
        logger.info("✓ 找到训练好的模型，跳过训练\n")
    else:
        if force_retrain:
            logger.info("强制重新训练模式，忽略已有模型...\n")
        
        logger.info("未找到模型文件，开始训练...")
        
        # 3. 准备训练数据（使用改进的参数）
        logger.info("[3/5] 准备训练数据...")
        logger.info(f"  训练数据时间范围: {train_start_date} ~ {train_end_date}")
        logger.info(f"  时间点数量: {periods} (每半年一个时间点)")
        
        X, y = ml_selector.prepare_training_data_multi_period(
            start_date=train_start_date,
            end_date=train_end_date,
            periods=periods,
            stock_pool=stock_pool
        )
        
        if X is None or y is None:
            logger.error("训练数据准备失败！")
            return None
        
        logger.info(f"✓ 训练数据准备完成: {len(X)} 个样本\n")
        
        # 4. 训练模型
        logger.info("[4/5] 训练模型...")
        ml_selector.train_models(X, y, model_type=model_type)
        
        # 5. 保存模型
        ml_selector.save_models()
        logger.info("✓ 模型训练完成并保存\n")
    
    # 6. 使用模型预测
    logger.info("[5/5] 使用ML模型选股...")
    results = ml_selector.select_stocks_with_ml(
        date=pred_date,
        stock_pool=stock_pool,
        top_n=top_n,
        model_type=model_type
    )
    
    if results is None:
        logger.error("ML选股失败！")
        return None
    
    logger.info("✓ ML选股完成\n")
    
    # 7. 保存结果
    output_dir = Path("data/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    # 命名规则: ml-年份日期-topN.csv（如 ml-20260603-top20.csv）
    date_str = datetime.now().strftime("%Y%m%d")
    output_path = output_dir / f"ml-{date_str}-top{top_n}.csv"
    
    results.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"✓ 结果已保存: {output_path}")
    
    logger.info("=" * 70)
    logger.info(f"TOP {top_n} 股票（ML选股 - 改进版）:")
    logger.info("=" * 70)
    
    return results
    

if __name__ == "__main__":
    import argparse
    
    # 配置日志（只输出到控制台，避免文件权限问题）
    logger.remove()
    logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="ML选股器（改进版）")
    parser.add_argument("--train-only", action="store_true", help="只训练模型，不选股")
    parser.add_argument("--retrain", action="store_true", help="强制重新训练模型（覆盖已有模型）")
    parser.add_argument("--pred-date", type=str, default="20220103", help="预测日期（选股日期）")
    parser.add_argument("--top-n", type=int, default=20, help="选择TOP N股票")
    parser.add_argument("--model-type", type=str, default="both", choices=["random_forest", "xgboost", "both"], help="模型类型")
    parser.add_argument("--stock-pool", type=str, default="zz500", help="股票池（hs300/zz500/zz800）")
    parser.add_argument("--train-start", type=str, default="20100101", help="训练数据起始日期")
    parser.add_argument("--train-end", type=str, default="20191231", help="训练数据截止日期")
    parser.add_argument("--periods", type=int, default=20, help="训练时间点数量")
    
    args = parser.parse_args()
    
    # 运行ML选股（改进版）
    results = run_ml_stock_selection(
        train_start_date=args.train_start,
        train_end_date=args.train_end,
        pred_date=args.pred_date,
        periods=args.periods,
        top_n=args.top_n,
        model_type=args.model_type,
        stock_pool=args.stock_pool,
        force_retrain=args.retrain  # 新增：强制重新训练
    )
    
    if results is not None:
        print("\n" + "=" * 70)
        print("选股结果:")
        print("=" * 70)
        print(results.to_string(index=False))
        print("=" * 70)
    else:
        print("\nML选股失败！请检查日志。")
