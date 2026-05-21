"""
因子处理模块 - FactorProcessor Class
实现数据清洗、标准化、中性化和打分
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict
from loguru import logger


class FactorProcessor:
    """因子处理器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化因子处理器
        
        Args:
            config: 配置字典（可选）
        """
        self.config = config or {}
        
        # 定义因子方向："positive" 表示正向因子，"negative" 表示反向因子
        self.factor_directions = {
            # 价值因子（反向）
            "VF1_PE": "negative",
            "VF2_PB": "negative",
            "VF3_PS": "negative",
            "VF4_PEG": "negative",
            "VF5_EV_EBITDA": "negative",
            "VF6_dividend_yield": "positive",
            
            # 成长因子（正向）
            "GF1_revenue_growth": "positive",
            "GF2_net_profit_growth": "positive",
            "GF3_ROE": "positive",
            "GF4_ROA": "positive",
            "GF5_gross_margin_growth": "positive",
            
            # 质量因子（混合）
            "QF1_asset_liab_ratio": "negative",
            "QF2_current_ratio": "positive",
            "QF3_asset_turnover": "positive",
            "QF4_cash_flow_quality": "positive",
            "QF5_cash_flow_to_revenue": "positive",
            
            # 动量因子（正向）
            "MF1_return_1m": "positive",
            "MF2_return_3m": "positive",
            "MF3_return_6m": "positive",
            "MF4_return_12m": "positive",
            "MF5_relative_strength": "positive",
            
            # 技术因子（正向）
            "TF1_ma_bullish": "positive",
            "TF2_MACD": "positive",
            "TF3_RSI": "positive",
            "TF4_volume_ratio": "positive",
            "TF5_bollinger_position": "positive",
        }
        
        logger.info("FactorProcessor initialized")
    
    def process(self, factors_df: pd.DataFrame, 
                industry_data: Optional[pd.DataFrame] = None,
                market_cap_data: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        处理因子的主方法
        
        Args:
            factors_df: 因子DataFrame（每行一只股票，每列一个因子）
            industry_data: 行业数据（可选，用于行业中性化）
            market_cap_data: 市值数据（可选，用于市值中性化）
            
        Returns:
            处理后的因子DataFrame（包含综合得分）
        """
        logger.info(f"Processing {len(factors_df)} stocks with {len(factors_df.columns)-1} factors...")
        
        df = factors_df.copy()
        
        # 1. 统一因子方向（将所有因子转换为正向因子）
        df = self._unify_factor_direction(df)
        logger.info("✓ Factor direction unified")
        
        # 2. 数据清洗（去极值、缺失值处理）
        df = self._clean_factors(df)
        logger.info("✓ Factor cleaning completed")
        
        # 3. 因子标准化
        standardization_method = self.config.get("standardization_method", "zscore")
        df = self._standardize_factors(df, method=standardization_method)
        logger.info(f"✓ Factor standardization completed (method: {standardization_method})")
        
        # 4. 因子中性化（行业、市值）
        if industry_data is not None:
            df = self._neutralize_industry(df, industry_data)
            logger.info("✓ Industry neutralization completed")
        
        if market_cap_data is not None:
            df = self._neutralize_market_cap(df, market_cap_data)
            logger.info("✓ Market cap neutralization completed")
        
        # 5. 计算综合得分
        df = self._calculate_scores(df)
        logger.info("✓ Scoring completed")
        
        logger.info(f"✓ Factor processing completed: {len(df)} stocks")
        
        return df
    
    def _unify_factor_direction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        统一因子方向（将所有因子转换为正向因子）
        
        反向因子（如PE、PB）转换为正向：使用 1 / (1 + abs(value)) 或 -value
        """
        for column in df.columns:
            if column == "code":
                continue
            
            direction = self.factor_directions.get(column, "positive")
            
            if direction == "negative":
                # 反向因子转换为正向
                # 方法1：取负值（适用于PE、PB等，越小越好 → 越大越好）
                df[column] = -df[column]
                
                # 方法2：使用倒数（适用于某些比率指标）
                # df[column] = 1 / (1 + df[column].abs())
                
                logger.debug(f"Converted negative factor: {column}")
        
        return df
    
    def _get_factor_columns(self, df: pd.DataFrame) -> list:
        """
        获取因子列名（排除非因子列）
        
        Returns:
            因子列名列表
        """
        # 因子列以 VF, GF, QF, MF, TF 开头
        factor_prefixes = ('VF', 'GF', 'QF', 'MF', 'TF')
        factor_cols = [col for col in df.columns if col.startswith(factor_prefixes)]
        return factor_cols
    
    def _clean_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据清洗：去极值、缺失值处理
        """
        # 只处理因子列（排除code, name, market_cap等非因子列）
        factor_cols = self._get_factor_columns(df)
        
        # 1. 去极值（Winsorization）
        winsorize_method = self.config.get("winsorize_method", "winsorize")
        
        if winsorize_method == "winsorize":
            lower = self.config.get("winsorize_lower", 0.01)
            upper = self.config.get("winsorize_upper", 0.99)
            
            for column in factor_cols:
                df[column] = self._winsorize(df[column], lower, upper)
            
            logger.debug(f"Winsorization completed: lower={lower}, upper={upper}")
        
        # 2. 缺失值处理
        missing_value_method = self.config.get("missing_value_method", "fill_median")
        
        for column in factor_cols:
            if missing_value_method == "fill_median":
                median_value = df[column].median()
                df[column] = df[column].fillna(median_value)
            elif missing_value_method == "fill_mean":
                mean_value = df[column].mean()
                df[column] = df[column].fillna(mean_value)
            elif missing_value_method == "drop":
                # 删除包含缺失值的行
                df = df.dropna(subset=[column])
        
        return df
    
    def _winsorize(self, series: pd.Series, lower: float, upper: float) -> pd.Series:
        """
        去极值处理（分位数法）
        
        Args:
            series: 数据序列
            lower: 下限分位数（如 0.01）
            upper: 上限分位数（如 0.99）
            
        Returns:
            去极值后的序列
        """
        lower_bound = series.quantile(lower)
        upper_bound = series.quantile(upper)
        
        return series.clip(lower_bound, upper_bound)
    
    def _standardize_factors(self, df: pd.DataFrame, method: str = "zscore") -> pd.DataFrame:
        """
        因子标准化
        
        Args:
            df: 因子DataFrame
            method: 标准化方法 ("zscore" 或 "rank")
            
        Returns:
            标准化后的DataFrame
        """
        # 只处理因子列
        factor_cols = self._get_factor_columns(df)
        
        for column in factor_cols:
            if method == "zscore":
                # Z-Score 标准化
                mean_val = df[column].mean()
                std_val = df[column].std()
                
                if std_val != 0:
                    df[column] = (df[column] - mean_val) / std_val
                else:
                    df[column] = 0
            
            elif method == "rank":
                # 分位数排名法
                df[column] = df[column].rank(pct=True)
        
        return df
    
    def _neutralize_industry(self, df: pd.DataFrame, 
                            industry_data: pd.DataFrame) -> pd.DataFrame:
        """
        行业中性化（残差法）
        
        Args:
            df: 因子DataFrame
            industry_data: 行业数据（包含 "code" 和 "industry" 列）
            
        Returns:
            行业中性化后的DataFrame
        """
        # 合并行业数据
        df = df.merge(industry_data[["code", "industry"]], on="code", how="left")
        
        # 对每个因子进行行业中性化
        factor_columns = [col for col in df.columns if col.startswith(("VF", "GF", "QF", "MF", "TF"))]
        
        for column in factor_columns:
            # 计算每个行业的平均因子值
            industry_mean = df.groupby("industry")[column].transform("mean")
            
            # 用原始因子值减去行业均值
            df[column] = df[column] - industry_mean
            
            logger.debug(f"Industry neutralization: {column}")
        
        # 删除行业列
        df = df.drop("industry", axis=1)
        
        return df
    
    def _neutralize_market_cap(self, df: pd.DataFrame, 
                               market_cap_data: pd.Series) -> pd.DataFrame:
        """
        市值中性化（回归法）
        
        Args:
            df: 因子DataFrame
            market_cap_data: 市值数据（索引为股票代码，值为市值）
            
        Returns:
            市值中性化后的DataFrame
        """
        try:
            import statsmodels.api as sm
            
            # 合并市值数据
            df["log_market_cap"] = np.log(market_cap_data[df["code"]].values)
            
            # 对每个因子进行市值中性化
            factor_columns = [col for col in df.columns if col.startswith(("VF", "GF", "QF", "MF", "TF"))]
            
            for column in factor_columns:
                X = sm.add_constant(df["log_market_cap"])
                model = sm.OLS(df[column], X).fit()
                df[column] = model.resid
                
                logger.debug(f"Market cap neutralization: {column}")
            
            # 删除市值列
            df = df.drop("log_market_cap", axis=1)
        
        except ImportError:
            logger.warning("statsmodels not installed, skipping market cap neutralization")
        
        return df
    
    def _calculate_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算综合得分（等权重）
        
        打分逻辑：
        1. 计算每个大类因子的平均分
        2. 计算五大类因子的平均分（等权重）
        """
        # 定义大类因子分组
        factor_groups = {
            "value_score": ["VF1_PE", "VF2_PB", "VF3_PS", "VF4_PEG", "VF5_EV_EBITDA", "VF6_dividend_yield"],
            "growth_score": ["GF1_revenue_growth", "GF2_net_profit_growth", "GF3_ROE", "GF4_ROA", "GF5_gross_margin_growth"],
            "quality_score": ["QF1_asset_liab_ratio", "QF2_current_ratio", "QF3_asset_turnover", "QF4_cash_flow_quality", "QF5_cash_flow_to_revenue"],
            "momentum_score": ["MF1_return_1m", "MF2_return_3m", "MF3_return_6m", "MF4_return_12m", "MF5_relative_strength"],
            "technical_score": ["TF1_ma_bullish", "TF2_MACD", "TF3_RSI", "TF4_volume_ratio", "TF5_bollinger_position"],
        }
        
        # 计算每个大类因子的平均分
        for group_name, factor_list in factor_groups.items():
            # 只使用存在的因子
            available_factors = [f for f in factor_list if f in df.columns]
            
            if len(available_factors) > 0:
                df[group_name] = df[available_factors].mean(axis=1)
            else:
                df[group_name] = np.nan
        
        # 计算综合得分（五大类因子等权重）
        score_columns = list(factor_groups.keys())
        available_scores = [s for s in score_columns if s in df.columns]
        
        if len(available_scores) > 0:
            df["total_score"] = df[available_scores].mean(axis=1)
        else:
            df["total_score"] = np.nan
        
        # 按综合得分排序
        df = df.sort_values("total_score", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        
        logger.info(f"Scoring completed: {len(available_scores)} score components")
        
        return df


if __name__ == "__main__":
    # 测试代码
    from loguru import logger
    from src.data_fetcher import DataFetcher
    from src.factor_calculator import FactorCalculator
    
    # 初始化日志
    logger.add("factor_processor_test.log", rotation="500 MB")
    
    # 创建数据获取器
    data_fetcher = DataFetcher(use_tushare=False)
    
    # 创建因子计算器
    calculator = FactorCalculator()
    
    # 创建因子处理器
    processor = FactorProcessor()
    
    # 测试：处理因子数据
    print("\n" + "="*50)
    print("Test: Process factors")
    print("="*50)
    
    # 计算因子
    test_codes = ["000001", "000002", "600000"]
    factors_df = calculator.calculate_all_factors(test_codes, data_fetcher)
    
    # 处理因子
    processed_df = processor.process(factors_df)
    
    print(f"✓ Processed {len(processed_df)} stocks")
    print("\nTop 3 stocks:")
    print(processed_df[["rank", "code", "total_score", "value_score", "growth_score"]].head(3))
    
    print("\n" + "="*50)
    print("All tests completed!")
    print("="*50)
