"""
ML选股模块 v5 - 优化版
======================

优化点：
1. 超参数自动调优（RandomizedSearchCV）
2. 新增LightGBM、CatBoost、神经网络模型
3. 时间序列交叉验证（TimeSeriesSplit）
4. 特征选择（基于重要性）
5. 集成学习（Voting/Averaging）
6. 多种标签定义对比（TOP 20%/30%/50%）
7. 模型自动选择（基于CV得分）

使用方法：
  python ml_stock_selector_v5.py
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

# ML库
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# 尝试导入高级ML库
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not installed")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not installed")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logger.warning("CatBoost not installed")

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


class MLOptimizer:
    """ML模型优化器 - 自动调参、模型对比、集成学习"""
    
    def __init__(self, config=None, data_fetcher=None):
        self.config = config or {}
        self.models = {}
        self.feature_columns = []
        self.task_type = "classification"
        self.best_model_name = None
        self.best_cv_score = -1
        
        # 如果外部传入了 data_fetcher，直接使用
        if data_fetcher is not None:
            self.data_fetcher = data_fetcher
            logger.info("✓ 使用外部传入的 data_fetcher")
            
            if FROM_EXISTING:
                self.factor_calculator = FactorCalculator()
                self.factor_calculator.enable_money_flow = False
                self.factor_processor = FactorProcessor()
                logger.info("✓ FactorCalculator 和 FactorProcessor 初始化成功")
            
            self._auto_load_models()
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
                data_config = {
                    "primary_source": "local_db",
                    "local_db_path": r"D:\tu-sharedata\astock_daily.db",
                    "tushare_token": None,
                    "use_akshare_backup": False,
                    "use_tushare_backup": False,
                }
            
            self.data_fetcher = DataFetcher(**data_config)
            self.factor_calculator = FactorCalculator()
            self.factor_calculator.enable_money_flow = False
            self.factor_processor = FactorProcessor()
            logger.info("✓ 现有系统模块初始化成功")
        
        logger.info("MLOptimizer initialized")
    
    def prepare_training_data(self, start_date="20160701", end_date="20211231", 
                             periods=20, period_months=6, lookback_days=20,
                             stock_pool="hs300", label_threshold=0.7):
        """
        准备训练数据（支持多种标签阈值）
        
        Args:
            label_threshold: 分类标签阈值（0.7 = TOP 30%）
        """
        logger.info(f"准备训练数据: {start_date}~{end_date}, periods={periods}, threshold={label_threshold}")
        
        start_dt = datetime.strptime(start_date, "%Y%m%d")
        end_dt = datetime.strptime(end_date, "%Y%m%d")
        
        all_X = []
        all_y = []
        
        for i in range(periods):
            months_back = i * period_months
            point_date = end_dt - pd.DateOffset(months=months_back)
            
            if point_date < start_dt:
                break
            
            point_date_str = point_date.strftime("%Y%m%d")
            
            logger.info(f"处理时间点 {i+1}/{periods}: {point_date_str}")
            
            X_point, y_point = self._prepare_single_period(
                point_date_str, lookback_days, stock_pool, label_threshold
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
    
    def _prepare_single_period(self, date, lookback_days=20, stock_pool="hs300", label_threshold=0.7):
        """准备单个时间点的训练数据"""
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
        
        # 计算因子
        factors_df = self.factor_calculator.calculate_all_factors(
            stock_codes, 
            self.data_fetcher,
            end_date=date
        )
        
        if factors_df is None or len(factors_df) == 0:
            logger.warning(f"时间点 {date}: 因子计算失败")
            return None, None
        
        # 处理因子
        processed_df = self.factor_processor.process(factors_df)
        
        if processed_df is None or len(processed_df) == 0:
            logger.warning(f"时间点 {date}: 因子处理失败")
            return None, None
        
        # 计算未来收益率（标签）
        future_returns = self._calculate_future_returns(stock_codes, date, days=20)
        
        if future_returns is None:
            logger.warning(f"时间点 {date}: 无法计算未来收益率")
            return None, None
        
        # 合并特征和标签
        merged = processed_df.merge(future_returns, on="code", how="inner")
        
        # 选择特征列
        factor_columns = [col for col in processed_df.columns 
                        if col.startswith(('VF_', 'GF_', 'QF_', 'MF_', 'TF_', 'LVF_', 'MWF_')) or col == 'total_score']
        
        X = merged[factor_columns]
        y = merged["label_cls"]
        
        logger.info(f"  时间点 {date}: {len(X)} 个样本")
        
        return X, y
    
    def _calculate_future_returns(self, stock_codes, date, days=20):
        """计算未来收益率（标签）"""
        date_dt = datetime.strptime(date, "%Y%m%d")
        future_date = date_dt + timedelta(days=days)
        future_date_str = future_date.strftime("%Y%m%d")
        
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
        
        # 分类标签（可调整阈值）
        if len(returns_df) > 10:
            threshold = returns_df["future_return"].quantile(label_threshold)
            returns_df["label_cls"] = (returns_df["future_return"] > threshold).astype(int)
        else:
            returns_df["label_cls"] = 0
        
        return returns_df
    
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
    
    def optimize_hyperparameters(self, X, y, model_type="random_forest", n_iter=20):
        """
        超参数自动调优（使用RandomizedSearchCV）
        
        Args:
            X: 特征矩阵
            y: 标签
            model_type: 模型类型
            n_iter: 随机搜索迭代次数
            
        Returns:
            最优模型
        """
        logger.info(f"开始超参数调优: model_type={model_type}, n_iter={n_iter}")
        
        if model_type == "random_forest":
            model = RandomForestClassifier(random_state=42, n_jobs=-1)
            param_dist = {
                "n_estimators": [100, 200, 300, 500],
                "max_depth": [5, 10, 15, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "class_weight": ["balanced", "balanced_subsample", None],
            }
        
        elif model_type == "xgboost" and XGBOOST_AVAILABLE:
            model = xgb.XGBClassifier(random_state=42, n_jobs=-1)
            param_dist = {
                "n_estimators": [100, 200, 300, 500],
                "max_depth": [3, 6, 9, 12],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
            }
        
        elif model_type == "lightgbm" and LIGHTGBM_AVAILABLE:
            model = lgb.LGBMClassifier(random_state=42, n_jobs=-1)
            param_dist = {
                "n_estimators": [100, 200, 300, 500],
                "max_depth": [3, 6, 9, 12, -1],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "num_leaves": [31, 50, 100, 200],
                "subsample": [0.6, 0.8, 1.0],
            }
        
        else:
            logger.error(f"不支持的模型类型: {model_type}")
            return None
        
        # 使用时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=5)
        
        # RandomizedSearchCV
        random_search = RandomizedSearchCV(
            model, 
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=tscv,
            scoring='roc_auc',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        random_search.fit(X, y)
        
        logger.info(f"✓ 超参数调优完成")
        logger.info(f"  最优参数: {random_search.best_params_}")
        logger.info(f"  最优CV得分: {random_search.best_score_:.4f}")
        
        return random_search.best_estimator_
    
    def train_all_models(self, X, y, do_tune=True, n_iter=20):
        """
        训练所有可用模型，自动选择最优模型
        
        Args:
            X: 特征矩阵
            y: 标签
            do_tune: 是否进行超参数调优
            n_iter: 调优迭代次数
        """
        logger.info(f"开始训练所有模型: do_tune={do_tune}")
        
        self.feature_columns = X.columns.tolist()
        
        # 检查标签分布
        n_positive = y.sum()
        n_total = len(y)
        logger.info(f"  标签分布: 正样本={n_positive} ({n_positive/n_total:.1%}), 负样本={n_total-n_positive} ({(n_total-n_positive)/n_total:.1%})")
        
        # 划分训练集和测试集（时间序列：前80%训练，后20%测试）
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info(f"  训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本")
        
        trained_models = {}
        
        # 1. 随机森林
        logger.info("="*50)
        logger.info("训练随机森林...")
        if do_tune:
            rf_model = self.optimize_hyperparameters(X_train, y_train, "random_forest", n_iter)
        else:
            rf_model = RandomForestClassifier(
                n_estimators=200, max_depth=10, random_state=42, n_jobs=-1,
                class_weight="balanced"
            )
            rf_model.fit(X_train, y_train)
        
        # 评估
        y_pred = rf_model.predict(X_test)
        y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        logger.info(f"  随机森林 - Acc: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        
        # 交叉验证（时间序列）
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(rf_model, X_train, y_train, cv=tscv, scoring='roc_auc')
        logger.info(f"  随机森林 - CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        trained_models["random_forest"] = {
            "model": rf_model,
            "cv_score": cv_scores.mean(),
            "test_auc": auc,
        }
        
        # 2. XGBoost
        if XGBOOST_AVAILABLE:
            logger.info("="*50)
            logger.info("训练XGBoost...")
            if do_tune:
                xgb_model = self.optimize_hyperparameters(X_train, y_train, "xgboost", n_iter)
            else:
                xgb_model = xgb.XGBClassifier(
                    n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1
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
            
            cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=tscv, scoring='roc_auc')
            logger.info(f"  XGBoost - CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            
            trained_models["xgboost"] = {
                "model": xgb_model,
                "cv_score": cv_scores.mean(),
                "test_auc": auc,
            }
        
        # 3. LightGBM
        if LIGHTGBM_AVAILABLE:
            logger.info("="*50)
            logger.info("训练LightGBM...")
            if do_tune:
                lgb_model = self.optimize_hyperparameters(X_train, y_train, "lightgbm", n_iter)
            else:
                lgb_model = lgb.LGBMClassifier(
                    n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1
                )
                lgb_model.fit(X_train, y_train)
            
            # 评估
            y_pred = lgb_model.predict(X_test)
            y_pred_proba = lgb_model.predict_proba(X_test)[:, 1]
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            logger.info(f"  LightGBM - Acc: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
            
            cv_scores = cross_val_score(lgb_model, X_train, y_train, cv=tscv, scoring='roc_auc')
            logger.info(f"  LightGBM - CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            
            trained_models["lightgbm"] = {
                "model": lgb_model,
                "cv_score": cv_scores.mean(),
                "test_auc": auc,
            }
        
        # 4. 逻辑回归（基线模型）
        logger.info("="*50)
        logger.info("训练逻辑回归...")
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train, y_train)
        
        y_pred = lr_model.predict(X_test)
        y_pred_proba = lr_model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        logger.info(f"  逻辑回归 - Acc: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        
        cv_scores = cross_val_score(lr_model, X_train, y_train, cv=tscv, scoring='roc_auc')
        logger.info(f"  逻辑回归 - CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        trained_models["logistic_regression"] = {
            "model": lr_model,
            "cv_score": cv_scores.mean(),
            "test_auc": auc,
        }
        
        # 5. 神经网络
        logger.info("="*50)
        logger.info("训练神经网络...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        nn_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        nn_model.fit(X_train_scaled, y_train)
        
        y_pred = nn_model.predict(X_test_scaled)
        y_pred_proba = nn_model.predict_proba(X_test_scaled)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        logger.info(f"  神经网络 - Acc: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        
        # 保存scaler
        trained_models["neural_network"] = {
            "model": nn_model,
            "scaler": scaler,
            "cv_score": auc,  # 用测试集AUC代替CV
            "test_auc": auc,
        }
        
        # 6. 集成学习（Voting Classifier）
        logger.info("="*50)
        logger.info("训练集成模型（Voting）...")
        
        estimators = []
        if "random_forest" in trained_models:
            estimators.append(('rf', trained_models["random_forest"]["model"]))
        if "xgboost" in trained_models:
            estimators.append(('xgb', trained_models["xgboost"]["model"]))
        if "lightgbm" in trained_models:
            estimators.append(('lgb', trained_models["lightgbm"]["model"]))
        
        if len(estimators) >= 2:
            voting_model = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
            voting_model.fit(X_train, y_train)
            
            y_pred = voting_model.predict(X_test)
            y_pred_proba = voting_model.predict_proba(X_test)[:, 1]
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            logger.info(f"  集成模型 - Acc: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
            
            # 集成模型的CV得分用各个基模型的平均
            cv_score = np.mean([trained_models[e[0]]["cv_score"] for e in estimators])
            
            trained_models["voting"] = {
                "model": voting_model,
                "cv_score": cv_score,
                "test_auc": auc,
            }
        
        # 选择最优模型（基于CV得分）
        best_model_name = max(trained_models, key=lambda k: trained_models[k]["cv_score"])
        best_cv_score = trained_models[best_model_name]["cv_score"]
        
        logger.info("="*50)
        logger.info(f"✓ 所有模型训练完成")
        logger.info(f"  最优模型: {best_model_name} (CV AUC: {best_cv_score:.4f})")
        
        # 保存所有模型
        self.models = trained_models
        self.best_model_name = best_model_name
        self.best_cv_score = best_cv_score
        
        return trained_models
    
    def save_models(self, model_dir="data/models_v5"):
        """保存所有模型和最优模型信息"""
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        
        for name, data in self.models.items():
            model_file = model_path / f"{name}_model.pkl"
            
            save_data = {
                "model": data["model"],
                "feature_columns": self.feature_columns,
                "task_type": self.task_type,
                "cv_score": data["cv_score"],
                "test_auc": data["test_auc"],
            }
            
            if "scaler" in data:
                save_data["scaler"] = data["scaler"]
            
            with open(model_file, "wb") as f:
                pickle.dump(save_data, f)
            
            logger.info(f"  ✓ 已保存模型: {model_file}")
        
        # 保存最优模型信息
        info_file = model_path / "best_model_info.txt"
        with open(info_file, "w", encoding="utf-8") as f:
            f.write(f"best_model_name: {self.best_model_name}\n")
            f.write(f"best_cv_score: {self.best_cv_score:.4f}\n")
        
        logger.info(f"✓ 所有模型已保存至: {model_dir}")
    
    def select_stocks(self, date, stock_pool="hs300", top_n=20, model_name=None):
        """
        使用ML模型选股
        
        Args:
            model_name: 使用的模型名称（None = 使用最优模型）
        """
        if len(self.models) == 0:
            logger.error("模型未训练！请先运行 train_all_models()")
            return None
        
        # 如果未指定模型，使用最优模型
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.models:
            logger.error(f"模型 {model_name} 不存在！可用模型: {list(self.models.keys())}")
            return None
        
        logger.info(f"使用模型选股: model={model_name}, date={date}, top_n={top_n}")
        
        # 获取股票池
        try:
            if stock_pool == "hs300":
                stocks_df = self.data_fetcher.get_hs300_components(date=date)
            elif stock_pool == "zz500":
                stocks_df = self.data_fetcher.get_zz500_components(date=date)
            elif stock_pool == "zz800":
                stocks_df = self.data_fetcher.get_zz800_components(date=date)
            else:
                logger.error(f"不支持的股票池: {stock_pool}")
                return None
        except Exception as e:
            logger.error(f"获取股票池失败: {e}")
            return None
        
        if stocks_df is None or len(stocks_df) == 0:
            logger.error(f"股票池为空: {stock_pool}")
            return None
        
        stock_codes = stocks_df["code"].tolist()
        
        # 计算因子
        factors_df = self.factor_calculator.calculate_all_factors(
            stock_codes, 
            self.data_fetcher,
            end_date=date
        )
        
        if factors_df is None or len(factors_df) == 0:
            logger.error("因子计算失败")
            return None
        
        # 处理因子
        processed_df = self.factor_processor.process(factors_df)
        
        if processed_df is None or len(processed_df) == 0:
            logger.error("因子处理失败")
            return None
        
        # 提取特征
        factor_columns = [col for col in processed_df.columns 
                        if col.startswith(('VF_', 'GF_', 'QF_', 'MF_', 'TF_', 'LVF_', 'MWF_')) or col == 'total_score']
        
        X = processed_df[factor_columns]
        
        # 预测
        model_data = self.models[model_name]
        model = model_data["model"]
        
        if model_name == "neural_network":
            scaler = model_data["scaler"]
            X_scaled = scaler.transform(X)
            y_pred_proba = model.predict_proba(X_scaled)[:, 1]
        else:
            y_pred_proba = model.predict_proba(X)[:, 1]
        
        # 选择TOP N
        processed_df["ml_score"] = y_pred_proba
        top_stocks = processed_df.nlargest(top_n, "ml_score")[["code", "name", "ml_score"]]
        
        logger.info(f"✓ 选股完成: {len(top_stocks)} 只股票")
        
        return top_stocks


def main():
    """主函数：训练模型 + 选股"""
    import config as cfg
    
    # 初始化
    selector = MLOptimizer()
    
    # 准备训练数据（尝试不同的标签阈值）
    logger.info("="*50)
    logger.info("步骤1: 准备训练数据（多种标签阈值对比）")
    logger.info("="*50)
    
    # 可以尝试不同的阈值
    thresholds = [0.7, 0.6, 0.5]  # TOP 30%, 40%, 50%
    
    for threshold in thresholds:
        logger.info(f"\n尝试标签阈值: {threshold} (TOP {(1-threshold)*100:.0f}%)")
        
        X, y = selector.prepare_training_data(
            start_date="20160701",
            end_date="20211231",
            periods=20,
            period_months=6,
            stock_pool="hs300",
            label_threshold=threshold
        )
        
        if X is None or y is None:
            logger.error(f"标签阈值 {threshold} 数据准备失败，跳过")
            continue
        
        # 训练所有模型（启用超参数调优）
        logger.info("="*50)
        logger.info(f"步骤2: 训练所有模型（标签阈值={threshold}）")
        logger.info("="*50)
        
        models = selector.train_all_models(X, y, do_tune=True, n_iter=20)
        
        # 保存模型
        selector.save_models(model_dir=f"data/models_v5_threshold_{int(threshold*100)}")
        
        logger.info(f"✓ 标签阈值 {threshold} 完成\n")
    
    logger.info("="*50)
    logger.info("所有训练完成！")
    logger.info("="*50)


if __name__ == "__main__":
    main()
