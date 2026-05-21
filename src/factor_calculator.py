"""
因子计算模块 - FactorCalculator Class
计算五大类因子：价值、成长、质量、动量、低波动
默认只启用价值和成长因子，质量、动量、技术、低波动可选

使用 TA-Lib 重构技术指标计算，提高性能
"""

import akshare as ak
import pandas as pd
import numpy as np
import talib  # 新增：导入 TA-Lib
from typing import Dict, Optional, List
from loguru import logger


class FactorCalculator:
    """因子计算器（支持可选因子类型）"""
    
    def __init__(self, enable_quality: bool = True, 
                 enable_momentum: bool = False,
                 enable_technical: bool = False,
                 enable_volatility: bool = False,
                 enable_money_flow: bool = True):
        """
        初始化因子计算器
        
        Args:
            enable_quality: 是否启用质量因子（默认开启）
            enable_momentum: 是否启用动量因子（默认关闭）
            enable_technical: 是否启用技术因子（默认关闭）
            enable_volatility: 是否启用低波动因子（默认关闭）
            enable_money_flow: 是否启用资金流因子（默认开启）
        """
        self.enable_quality = enable_quality
        self.enable_momentum = enable_momentum
        self.enable_technical = enable_technical
        self.enable_volatility = enable_volatility
        self.enable_money_flow = enable_money_flow
        
        enabled = ["价值", "成长"]
        if enable_quality: enabled.append("质量")
        if enable_momentum: enabled.append("动量")
        if enable_technical: enabled.append("技术")
        if enable_volatility: enabled.append("低波动")
        if enable_money_flow: enabled.append("资金流")
        
        logger.info(f"FactorCalculator initialized (启用因子: {', '.join(enabled)})")
    
    def calculate_all_factors(self, stock_codes: List[str], 
                              data_fetcher,
                              max_workers: int = 5) -> pd.DataFrame:
        """
        计算所有股票的因子（支持多线程并行）
        
        Args:
            stock_codes: 股票代码列表
            data_fetcher: DataFetcher实例（用于获取数据）
            max_workers: 最大线程数（默认5）
            
        Returns:
            因子DataFrame，每行代表一只股票，每列代表一个因子
        """
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        logger.info(f"Calculating factors for {len(stock_codes)} stocks (max_workers={max_workers})...")
        
        all_factors = []
        failed_codes = []
        
        def process_single(code_with_idx):
            """处理单只股票的因子计算"""
            idx, code = code_with_idx
            try:
                logger.debug(f"Processing {code} ({idx+1}/{len(stock_codes)})")
                factors = self.calculate_single_stock_factors(code, data_fetcher)
                return factors, None
            except Exception as e:
                logger.error(f"Failed to calculate factors for {code}: {e}")
                return None, code
        
        # 使用多线程并行计算
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务（带索引用于进度显示）
            futures = {executor.submit(process_single, (i, code)): code 
                      for i, code in enumerate(stock_codes)}
            
            completed = 0
            for future in as_completed(futures):
                completed += 1
                if completed % 10 == 0:
                    logger.info(f"Progress: {completed}/{len(stock_codes)}")
                
                factors, failed_code = future.result()
                if factors is not None:
                    all_factors.append(factors)
                if failed_code is not None:
                    failed_codes.append(failed_code)
        
        if len(failed_codes) > 0:
            logger.warning(f"Failed to calculate {len(failed_codes)} stocks: {failed_codes[:10]}")
        
        # 转换为DataFrame
        df_factors = pd.DataFrame(all_factors)
        logger.info(f"✓ Factor calculation completed: {len(df_factors)} stocks")
        
        return df_factors
    
    def calculate_single_stock_factors(self, code: str, 
                                       data_fetcher,
                                       start_date: str = None,
                                       end_date: str = None) -> Dict[str, float]:
        """
        计算单只股票的所有因子
        
        Args:
            code: 股票代码
            data_fetcher: DataFetcher实例
            start_date: 开始日期（格式: "20230101"），None表示使用默认值
            end_date: 结束日期（格式: "20241231"），None表示使用默认值
            
        Returns:
            因子字典 {"factor_name": value}
        """
        factors = {"code": code}
        
        # 设置默认日期范围
        if start_date is None:
            start_date = "20230101"
        if end_date is None:
            from datetime import datetime
            end_date = datetime.now().strftime("%Y%m%d")
        
        # 获取股票基本信息（名称和市值）
        stock_info = data_fetcher.get_stock_info(code)
        if stock_info:
            factors["name"] = stock_info.get("name", "")
            factors["market_cap"] = stock_info.get("market_cap", np.nan)
        else:
            factors["name"] = ""
            factors["market_cap"] = np.nan
        
        # 获取历史行情数据（用于获取当前股价和技术因子）
        # 使用传入的日期范围，如果未传入则使用默认值
        hist_data = data_fetcher.get_stock_history(
            code, 
            start_date=start_date, 
            end_date=end_date
        )
        
        # 获取当前股价（前收）
        factors["current_price"] = self._extract_current_price(hist_data)
        
        # 获取财务数据
        financial_data = data_fetcher.get_financial_data(code)
        
        # 获取估值数据
        valuation_data = data_fetcher.get_valuation_data(code)
        
        # 如果market_cap为空，尝试从估值数据中提取（Tushare daily_basic 含总市值）
        if pd.isna(factors.get("market_cap")):
            if valuation_data is not None and len(valuation_data) > 0:
                if '总市值' in valuation_data.columns:
                    total_mv = valuation_data['总市值'].values[0]
                    if total_mv is not None:
                        try:
                            # Tushare返回的是百万元，转换为元
                            factors["market_cap"] = float(total_mv) * 1e6
                        except Exception:
                            pass
        
        # 计算价值因子（默认启用）
        factors.update(self._calc_value_factors(valuation_data, financial_data))
        
        # 计算成长因子（默认启用）
        factors.update(self._calc_growth_factors(financial_data))
        
        # 计算质量因子（可选）
        if self.enable_quality:
            factors.update(self._calc_quality_factors(financial_data))
        
        # 计算动量因子（可选）
        if self.enable_momentum:
            factors.update(self._calc_momentum_factors(hist_data))
        
        # 计算技术因子（可选）
        if self.enable_technical:
            factors.update(self._calc_technical_factors(hist_data))
        
        # 计算低波动因子（可选）
        if self.enable_volatility:
            factors.update(self._calc_volatility_factors(hist_data))
        
        # 计算资金流因子（可选）
        if self.enable_money_flow:
            money_flow_data = data_fetcher.get_money_flow_data(code)
            factors.update(self._calc_money_flow_factors(money_flow_data))
        
        return factors
    
    def _extract_current_price(self, hist_data: Optional[pd.DataFrame]) -> float:
        """
        从历史数据中提取当前股价（前收/最新收盘价）
        
        Args:
            hist_data: 历史行情数据
            
        Returns:
            当前股价（如果无法获取则返回NaN）
        """
        try:
            if hist_data is not None and len(hist_data) > 0:
                # 按日期排序，取最新的一天
                if '日期' in hist_data.columns:
                    hist_sorted = hist_data.sort_values('日期', ascending=False)
                else:
                    hist_sorted = hist_data.iloc[::-1]  # 反转顺序（假设是按日期升序）
                
                # 获取最新收盘价
                if '收盘' in hist_sorted.columns:
                    return float(hist_sorted.iloc[0]['收盘'])
                
                # 如果是Tushare格式（已转换为AkShare格式），应该有'收盘'列
                logger.debug(f"Cannot find '收盘' column in hist_data")
        except Exception as e:
            logger.debug(f"Error extracting current price: {e}")
        
        return np.nan
    
    def _calc_value_factors(self, valuation_data: Optional[pd.DataFrame], 
                             financial_data: Optional[pd.DataFrame]) -> Dict[str, float]:
        """
        计算价值因子
        
        因子列表:
        - VF1_PE: 市盈率（反向）
        - VF2_PB: 市净率（反向）
        - VF3_PS: 市销率（反向）
        - VF4_PEG: PEG（反向）
        - VF5_EV_EBITDA: 企业价值/息税折旧摊销前利润（反向）
        - VF6_dividend_yield: 股息率（正向）
        """
        factors = {}
        
        # 默认值
        factors["VF1_PE"] = np.nan
        factors["VF2_PB"] = np.nan
        factors["VF3_PS"] = np.nan
        factors["VF4_PEG"] = np.nan
        factors["VF5_EV_EBITDA"] = np.nan
        factors["VF6_dividend_yield"] = np.nan
        
        try:
            # 从估值数据中获取
            if valuation_data is not None and len(valuation_data) > 0:
                # AkShare 返回的估值数据格式
                # 注意：实际列名需要根据 AkShare 返回结果调整
                if "市盈率-动态" in valuation_data.columns:
                    pe = valuation_data["市盈率-动态"].values[0]
                    factors["VF1_PE"] = float(pe) if not pd.isna(pe) else np.nan
                
                if "市净率" in valuation_data.columns:
                    pb = valuation_data["市净率"].values[0]
                    factors["VF2_PB"] = float(pb) if not pd.isna(pb) else np.nan
                
                if "市销率" in valuation_data.columns:
                    ps = valuation_data["市销率"].values[0]
                    factors["VF3_PS"] = float(ps) if not pd.isna(ps) else np.nan
                
                if "股息率" in valuation_data.columns:
                    div_yield = valuation_data["股息率"].values[0]
                    factors["VF6_dividend_yield"] = float(div_yield) if not pd.isna(div_yield) else np.nan
            
            # 从财务数据中获取 PEG (需要自己计算: PE / 净利润增长率)
            if financial_data is not None and len(financial_data) > 0:
                # 获取 PE
                pe = factors["VF1_PE"]
                
                # 获取净利润增长率
                if "净利润增长率" in financial_data.columns:
                    growth = financial_data["净利润增长率"].values[0]
                    growth = float(growth) if not pd.isna(growth) else np.nan
                    
                    # 计算 PEG
                    if not np.isnan(pe) and not np.isnan(growth) and growth != 0:
                        factors["VF4_PEG"] = pe / (growth * 100)  # growth 是百分比
            
            logger.debug(f"Value factors calculated: PE={factors['VF1_PE']}, PB={factors['VF2_PB']}")
            
        except Exception as e:
            logger.error(f"Error calculating value factors: {e}")
        
        return factors
    
    def _calc_growth_factors(self, financial_data: Optional[pd.DataFrame]) -> Dict[str, float]:
        """
        计算成长因子
        
        因子列表:
        - GF1_revenue_growth: 营收增长率（正向）
        - GF2_net_profit_growth: 净利润增长率（正向）
        - GF3_ROE: 净资产收益率（正向）
        - GF4_ROA: 总资产收益率（正向）
        - GF5_gross_margin_growth: 毛利率增长率（正向）
        """
        factors = {}
        
        # 默认值
        factors["GF1_revenue_growth"] = np.nan
        factors["GF2_net_profit_growth"] = np.nan
        factors["GF3_ROE"] = np.nan
        factors["GF4_ROA"] = np.nan
        factors["GF5_gross_margin_growth"] = np.nan
        
        try:
            if financial_data is not None and len(financial_data) > 0:
                # AkShare 返回的财务指标数据格式
                # 注意：实际列名需要根据 AkShare 返回结果调整
                
                if "营业总收入同比增长率" in financial_data.columns:
                    rev_growth = financial_data["营业总收入同比增长率"].values[0]
                    factors["GF1_revenue_growth"] = float(rev_growth) if not pd.isna(rev_growth) else np.nan
                
                if "净利润同比增长率" in financial_data.columns:
                    profit_growth = financial_data["净利润同比增长率"].values[0]
                    factors["GF2_net_profit_growth"] = float(profit_growth) if not pd.isna(profit_growth) else np.nan
                
                if "净资产收益率" in financial_data.columns:
                    roe = financial_data["净资产收益率"].values[0]
                    factors["GF3_ROE"] = float(roe) if not pd.isna(roe) else np.nan
                
                if "总资产净利率" in financial_data.columns:
                    roa = financial_data["总资产净利率"].values[0]
                    factors["GF4_ROA"] = float(roa) if not pd.isna(roa) else np.nan
                
                # 毛利率增长率（需要计算）
                if "销售毛利率" in financial_data.columns:
                    gross_margin = financial_data["销售毛利率"].values
                    if len(gross_margin) >= 2:
                        factors["GF5_gross_margin_growth"] = float(gross_margin[0] - gross_margin[1])
            
            logger.debug(f"Growth factors calculated: ROE={factors['GF3_ROE']}")
            
        except Exception as e:
            logger.error(f"Error calculating growth factors: {e}")
        
        return factors
    
    def _calc_quality_factors(self, financial_data: Optional[pd.DataFrame]) -> Dict[str, float]:
        """
        计算质量因子
        
        因子列表:
        - QF1_asset_liability_ratio: 资产负债率（反向）
        - QF2_current_ratio: 流动比率（正向）
        - QF3_asset_turnover: 资产周转率（正向）
        - QF4_cash_flow_quality: 现金流质量（正向）
        - QF5_cash_flow_to_revenue: 净利润现金含量（正向）
        """
        factors = {}
        
        # 默认值
        factors["QF1_asset_liability_ratio"] = np.nan
        factors["QF2_current_ratio"] = np.nan
        factors["QF3_asset_turnover"] = np.nan
        factors["QF4_cash_flow_quality"] = np.nan
        factors["QF5_cash_flow_to_revenue"] = np.nan
        
        try:
            if financial_data is not None and len(financial_data) > 0:
                # AkShare 返回的财务指标数据格式
                # 注意：实际列名需要根据 AkShare 返回结果调整
                
                if "资产负债率" in financial_data.columns:
                    asset_liab_ratio = financial_data["资产负债率"].values[0]
                    factors["QF1_asset_liability_ratio"] = float(asset_liab_ratio) if not pd.isna(asset_liab_ratio) else np.nan
                
                if "流动比率" in financial_data.columns:
                    current_ratio = financial_data["流动比率"].values[0]
                    factors["QF2_current_ratio"] = float(current_ratio) if not pd.isna(current_ratio) else np.nan
                
                if "总资产周转率" in financial_data.columns:
                    asset_turnover = financial_data["总资产周转率"].values[0]
                    factors["QF3_asset_turnover"] = float(asset_turnover) if not pd.isna(asset_turnover) else np.nan
                
                if "经营活动现金流量净额/净利润" in financial_data.columns:
                    cf_quality = financial_data["经营活动现金流量净额/净利润"].values[0]
                    factors["QF4_cash_flow_quality"] = float(cf_quality) if not pd.isna(cf_quality) else np.nan
                
                if "销售商品提供劳务收到的现金/营业收入" in financial_data.columns:
                    cf_to_rev = financial_data["销售商品提供劳务收到的现金/营业收入"].values[0]
                    factors["QF5_cash_flow_to_revenue"] = float(cf_to_rev) if not pd.isna(cf_to_rev) else np.nan
            
            logger.debug(f"Quality factors calculated: AssetLiabRatio={factors['QF1_asset_liability_ratio']}")
            
        except Exception as e:
            logger.error(f"Error calculating quality factors: {e}")
        
        return factors
    
    def _calc_momentum_factors(self, hist_data: Optional[pd.DataFrame]) -> Dict[str, float]:
        """
        计算动量因子（使用 TA-Lib ROC 提高性能）
        
        因子列表:
        - MF1_return_1m: 1个月收益率（正向）
        - MF2_return_3m: 3个月收益率（正向）
        - MF3_return_6m: 6个月收益率（正向）
        - MF4_return_12m: 12个月收益率（正向）
        - MF5_relative_strength: 相对强度（正向）
        """
        factors = {}
        
        # 默认值
        factors["MF1_return_1m"] = np.nan
        factors["MF2_return_3m"] = np.nan
        factors["MF3_return_6m"] = np.nan
        factors["MF4_return_12m"] = np.nan
        factors["MF5_relative_strength"] = np.nan
        
        try:
            if hist_data is not None and len(hist_data) > 0:
                # 获取收盘价
                close_prices = hist_data["收盘"].values
                
                # 检查数据长度是否足够
                if len(close_prices) < 250:
                    logger.debug(f"Not enough data for momentum factors: {len(close_prices)} < 250")
                    return factors
                
                # 1. 使用 TA-Lib ROC (Rate of Change) 计算收益率
                # ROC 返回百分比变化，需要除以 100 得到小数
                
                # 1个月收益率（约20个交易日）
                if len(close_prices) >= 20:
                    roc_1m = talib.ROC(close_prices, timeperiod=20)
                    if not np.isnan(roc_1m[-1]):
                        factors["MF1_return_1m"] = roc_1m[-1] / 100.0
                
                # 3个月收益率（约60个交易日）
                if len(close_prices) >= 60:
                    roc_3m = talib.ROC(close_prices, timeperiod=60)
                    if not np.isnan(roc_3m[-1]):
                        factors["MF2_return_3m"] = roc_3m[-1] / 100.0
                
                # 6个月收益率（约120个交易日）
                if len(close_prices) >= 120:
                    roc_6m = talib.ROC(close_prices, timeperiod=120)
                    if not np.isnan(roc_6m[-1]):
                        factors["MF3_return_6m"] = roc_6m[-1] / 100.0
                
                # 12个月收益率（约250个交易日）
                if len(close_prices) >= 250:
                    roc_12m = talib.ROC(close_prices, timeperiod=250)
                    if not np.isnan(roc_12m[-1]):
                        factors["MF4_return_12m"] = roc_12m[-1] / 100.0
                
                # 2. 相对强度（个股收益率 / 市场收益率）
                # 这里假设市场收益率为0.05（5%），实际应该从指数计算
                market_return = 0.05
                if not np.isnan(factors["MF4_return_12m"]):
                    factors["MF5_relative_strength"] = factors["MF4_return_12m"] / market_return
            
            logger.debug(f"Momentum factors calculated: 1m return={factors['MF1_return_1m']}")
            
        except Exception as e:
            logger.error(f"Error calculating momentum factors: {e}")
        
        return factors
    
    def _calc_technical_factors(self, hist_data: Optional[pd.DataFrame]) -> Dict[str, float]:
        """
        计算技术因子（使用 TA-Lib 提高性能）
        
        因子列表:
        - TF1_ma_bullish: 均线多头（正向）
        - TF2_MACD: MACD指标（正向）
        - TF3_RSI: RSI指标（正向，50-70为佳）
        - TF4_volume_ratio: 成交量比（正向）
        - TF5_bollinger_position: 布林带位置（正向，接近下轨为好）
        """
        factors = {}
        
        # 默认值
        factors["TF1_ma_bullish"] = np.nan
        factors["TF2_MACD"] = np.nan
        factors["TF3_RSI"] = np.nan
        factors["TF4_volume_ratio"] = np.nan
        factors["TF5_bollinger_position"] = np.nan
        
        try:
            if hist_data is not None and len(hist_data) > 0:
                # 获取收盘价和成交量
                close_prices = hist_data["收盘"].values
                
                # 检查数据长度是否足够
                if len(close_prices) < 26:
                    logger.debug(f"Not enough data for technical factors: {len(close_prices)} < 26")
                    return factors
                
                # 1. 均线多头 (MA5 > MA10 > MA20) - 使用 TA-Lib
                if len(close_prices) >= 20:
                    ma5 = talib.SMA(close_prices, timeperiod=5)
                    ma10 = talib.SMA(close_prices, timeperiod=10)
                    ma20 = talib.SMA(close_prices, timeperiod=20)
                    
                    # 取最新值（忽略 NaN）
                    ma5_latest = ma5[-1] if not np.isnan(ma5[-1]) else np.nan
                    ma10_latest = ma10[-1] if not np.isnan(ma10[-1]) else np.nan
                    ma20_latest = ma20[-1] if not np.isnan(ma20[-1]) else np.nan
                    
                    if not np.isnan(ma5_latest) and not np.isnan(ma10_latest) and not np.isnan(ma20_latest):
                        # 1 if MA5 > MA10 > MA20, else 0
                        factors["TF1_ma_bullish"] = 1.0 if (ma5_latest > ma10_latest and ma10_latest > ma20_latest) else 0.0
                
                # 2. MACD - 使用 TA-Lib
                if len(close_prices) >= 26:
                    macd, macd_signal, macd_hist = talib.MACD(
                        close_prices, 
                        fastperiod=12, 
                        slowperiod=26, 
                        signalperiod=9
                    )
                    
                    # 取最新的 MACD 值
                    if not np.isnan(macd[-1]):
                        factors["TF2_MACD"] = float(macd[-1])
                
                # 3. RSI - 使用 TA-Lib
                if len(close_prices) >= 14:
                    rsi = talib.RSI(close_prices, timeperiod=14)
                    
                    # 取最新的 RSI 值
                    if not np.isnan(rsi[-1]):
                        factors["TF3_RSI"] = float(rsi[-1])
                
                # 4. 成交量比 (当日成交量 / 20日均量)
                if "成交量" in hist_data.columns and len(hist_data) >= 20:
                    volumes = hist_data["成交量"].values
                    vol_ratio = volumes[-1] / np.mean(volumes[-20:])
                    factors["TF4_volume_ratio"] = vol_ratio
                
                # 5. 布林带位置 - 使用 TA-Lib
                if len(close_prices) >= 20:
                    upper, middle, lower = talib.BBANDS(
                        close_prices, 
                        timeperiod=20, 
                        nbdevup=2, 
                        nbdevdn=2, 
                        matype=0
                    )
                    
                    # 计算位置 = (收盘价 - 下轨) / (上轨 - 下轨)
                    if not np.isnan(upper[-1]) and not np.isnan(lower[-1]) and upper[-1] != lower[-1]:
                        current_price = close_prices[-1]
                        bb_pos = (current_price - lower[-1]) / (upper[-1] - lower[-1])
                        factors["TF5_bollinger_position"] = float(bb_pos)
            
            logger.debug(f"Technical factors calculated: MA_bullish={factors['TF1_ma_bullish']}")
            
        except Exception as e:
            logger.error(f"Error calculating technical factors: {e}")
        
        return factors
    
    def _calc_volatility_factors(self, hist_data: Optional[pd.DataFrame]) -> Dict[str, float]:
        """
        计算低波动因子（使用 TA-Lib STDDEV 提高性能）
        
        因子列表:
        - LVF1_hist_vol: 历史波动率（反向，越低越好）
        - LVF2_beta: 贝塔系数（反向）
        - LVF3_downside_vol: 下行波动率（反向）
        - LVF4_idiosyncratic_vol: 特质波动率（反向）
        - LVF5_VAR: 风险价值VaR（反向，越小越好）
        """
        factors = {}
        
        # 默认值
        factors["LVF1_hist_vol"] = np.nan
        factors["LVF2_beta"] = np.nan
        factors["LVF3_downside_vol"] = np.nan
        factors["LVF4_idiosyncratic_vol"] = np.nan
        factors["LVF5_VAR"] = np.nan
        
        try:
            if hist_data is not None and len(hist_data) > 0:
                # 获取收盘价
                close_prices = hist_data["收盘"].values
                
                # 计算日收益率
                returns = np.diff(close_prices) / close_prices[:-1]
                
                # 检查数据长度是否足够
                if len(returns) < 20:
                    logger.debug(f"Not enough data for volatility factors: {len(returns)} < 20")
                    return factors
                
                # 1. 历史波动率（20日收益率标准差，年化）- 使用 TA-Lib STDDEV
                if len(returns) >= 20:
                    # 将 returns 转换为 numpy array (TA-Lib 需要)
                    returns_array = np.array(returns[-20:], dtype=float)
                    
                    # 使用 TA-Lib STDDEV 计算标准差
                    stddev = talib.STDDEV(returns_array, timeperiod=len(returns_array))
                    
                    if not np.isnan(stddev[-1]):
                        daily_vol = stddev[-1]
                        annualized_vol = daily_vol * np.sqrt(252)  # 年化波动率
                        factors["LVF1_hist_vol"] = annualized_vol
                
                # 2. 下行波动率（只计算负收益的标准差）
                if len(returns) >= 20:
                    downside_returns = returns[returns < 0]
                    if len(downside_returns) > 0:
                        # 使用 TA-Lib STDDEV
                        downside_array = np.array(downside_returns, dtype=float)
                        downside_stddev = talib.STDDEV(downside_array, timeperiod=len(downside_array))
                        
                        if not np.isnan(downside_stddev[-1]):
                            downside_vol = downside_stddev[-1] * np.sqrt(252)
                            factors["LVF3_downside_vol"] = downside_vol
                
                # 3. VaR（风险价值）- 5%分位数
                if len(returns) >= 20:
                    var_95 = np.percentile(returns, 5)  # 95% VaR
                    factors["LVF5_VAR"] = abs(var_95)  # 取绝对值，方便比较
                
                # 4. 贝塔系数（需要市场收益率，这里用0.8-1.2随机模拟）
                # 实际使用时应该获取市场指数（如沪深300）的收益率
                # 这里简化为：LVF2_beta = cov(个股收益, 市场收益) / var(市场收益)
                if not np.isnan(factors["LVF1_hist_vol"]):
                    # 假设市场波动率为0.2（20%），用相对波动率估算beta
                    market_vol = 0.20
                    estimated_beta = factors["LVF1_hist_vol"] / market_vol
                    factors["LVF2_beta"] = estimated_beta
                
                # 5. 特质波动率（总波动率 - 系统性波动率）
                if not np.isnan(factors["LVF1_hist_vol"]) and not np.isnan(factors["LVF2_beta"]):
                    market_vol = 0.20
                    systematic_vol = abs(factors["LVF2_beta"]) * market_vol
                    idiosyncratic_vol = np.sqrt(factors["LVF1_hist_vol"]**2 - systematic_vol**2)
                    factors["LVF4_idiosyncratic_vol"] = max(idiosyncratic_vol, 0)  # 不能为负
            
            logger.debug(f"Volatility factors calculated: HistVol={factors['LVF1_hist_vol']:.4f}")
            
        except Exception as e:
            logger.error(f"Error calculating volatility factors: {e}")
        
        return factors
    

    def _calc_money_flow_factors(self, money_flow_data: Optional[pd.DataFrame]) -> Dict[str, float]:
        """
        计算资金流因子
        
        因子列表:
        - MF1_ultra_large_inflow: 超大单净流入（正向）
        - MF2_large_inflow: 大单净流入（正向）
        - MF3_medium_inflow: 中单净流入（正向）
        - MF4_small_inflow: 小单净流入（反向，散户通常是反向指标）
        - MF5_main_inflow: 主力资金净流入（超大单+大单，正向）
        - MF6_inflow_intensity: 资金流强度（净流入/成交额，正向）
        """
        factors = {}
        
        # 默认值
        factors["MF1_ultra_large_inflow"] = np.nan
        factors["MF2_large_inflow"] = np.nan
        factors["MF3_medium_inflow"] = np.nan
        factors["MF4_small_inflow"] = np.nan
        factors["MF5_main_inflow"] = np.nan
        factors["MF6_inflow_intensity"] = np.nan
        
        try:
            if money_flow_data is not None and len(money_flow_data) > 0:
                # AkShare 返回的资金流数据格式
                # 可能包含的列：超大单净流入, 大单净流入, 中单净流入, 小单净流入, 净流入, 成交额
                
                # 超大单净流入
                if "超大单净流入" in money_flow_data.columns:
                    ultra_large = money_flow_data["超大单净流入"].values[0]
                    factors["MF1_ultra_large_inflow"] = float(ultra_large) if not pd.isna(ultra_large) else np.nan
                
                # 大单净流入
                if "大单净流入" in money_flow_data.columns:
                    large = money_flow_data["大单净流入"].values[0]
                    factors["MF2_large_inflow"] = float(large) if not pd.isna(large) else np.nan
                
                # 中单净流入
                if "中单净流入" in money_flow_data.columns:
                    medium = money_flow_data["中单净流入"].values[0]
                    factors["MF3_medium_inflow"] = float(medium) if not pd.isna(medium) else np.nan
                
                # 小单净流入（反向指标，取负值）
                if "小单净流入" in money_flow_data.columns:
                    small = money_flow_data["小单净流入"].values[0]
                    small_val = float(small) if not pd.isna(small) else np.nan
                    # 小单净流入越大，说明散户买入越多，通常是反向指标，所以取负值
                    factors["MF4_small_inflow"] = -small_val if not pd.isnan(small_val) else np.nan
                
                # 主力资金净流入（超大单 + 大单）
                if not pd.isna(factors["MF1_ultra_large_inflow"]) and not pd.isna(factors["MF2_large_inflow"]):
                    factors["MF5_main_inflow"] = factors["MF1_ultra_large_inflow"] + factors["MF2_large_inflow"]
                
                # 资金流强度（净流入 / 成交额）
                if "净流入" in money_flow_data.columns and "成交额" in money_flow_data.columns:
                    net_inflow = money_flow_data["净流入"].values[0]
                    turnover = money_flow_data["成交额"].values[0]
                    
                    if not pd.isna(net_inflow) and not pd.isna(turnover) and turnover != 0:
                        factors["MF6_inflow_intensity"] = float(net_inflow) / float(turnover)
                
                logger.debug(f"Money flow factors calculated: MainInflow={factors['MF5_main_inflow']}")
            else:
                logger.debug(f"No money flow data available")
            
        except Exception as e:
            logger.error(f"Error calculating money flow factors: {e}")
        
        return factors


if __name__ == "__main__":
    # 测试代码
    from loguru import logger
    from src.data_fetcher import DataFetcher
    
    # 初始化日志
    logger.add("factor_calculator_test.log", rotation="500 MB")
    
    # 创建数据获取器
    data_fetcher = DataFetcher(use_tushare=False)
    
    # 创建因子计算器
    calculator = FactorCalculator()
    
    # 测试1: 计算单只股票的因子
    print("\n" + "="*50)
    print("Test 1: Calculate single stock factors (000001)")
    print("="*50)
    
    factors = calculator.calculate_single_stock_factors("000001", data_fetcher)
    print(f"✓ Calculated {len(factors)} factors for 000001")
    for key, value in factors.items():
        print(f"  {key}: {value}")
    
    # 测试2: 计算多只股票的因子
    print("\n" + "="*50)
    print("Test 2: Calculate multiple stocks factors")
    print("="*50)
    
    test_codes = ["000001", "000002", "600000"]
    df_factors = calculator.calculate_all_factors(test_codes, data_fetcher)
    print(f"✓ Calculated factors for {len(df_factors)} stocks")
    print(df_factors.head())
    
    print("\n" + "="*50)
    print("All tests completed!")
    print("="*50)
