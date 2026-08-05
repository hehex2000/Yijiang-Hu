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
import quantstats as qs  # 新增：导入 quantstats（风险指标计算）
from typing import Dict, Optional, List
from loguru import logger

# 共享工具：自身历史分位 / 量价 regime（汲取自 Jim 换手率科普，平台统一惯例）
try:
    from .factor_utils import own_history_pct, volume_price_regime, latest_price_change
except ImportError:  # 允许以脚本方式直接运行
    from factor_utils import own_history_pct, volume_price_regime, latest_price_change


class FactorCalculator:
    """因子计算器（支持可选因子类型）"""
    
    def __init__(self, enable_quality: bool = True, 
                 enable_momentum: bool = False,
                 enable_technical: bool = False,
                 enable_volatility: bool = False,
                 enable_money_flow: bool = True,
                 enable_industry_momentum: bool = False,
                 enable_risk: bool = False,
                 peg_stability_years: int = 3):
        """
        初始化因子计算器
        
        Args:
            enable_quality: 是否启用质量因子（默认开启）
            enable_momentum: 是否启用动量因子（默认关闭）
            enable_technical: 是否启用技术因子（默认关闭）
            enable_volatility: 是否启用低波动因子（默认关闭）
            enable_money_flow: 是否启用资金流因子（默认开启）
            enable_industry_momentum: 是否启用行业动量因子（默认关闭）
            enable_risk: 是否启用风险因子（夏普比率、贝塔等，默认关闭）
        """
        self.enable_quality = enable_quality
        self.enable_momentum = enable_momentum
        self.enable_technical = enable_technical
        self.enable_volatility = enable_volatility
        self.enable_money_flow = enable_money_flow
        self.enable_industry_momentum = enable_industry_momentum
        self.enable_risk = enable_risk
        self.peg_stability_years = peg_stability_years
        
        enabled = ["价值", "成长"]
        if enable_quality: enabled.append("质量")
        if enable_momentum: enabled.append("动量")
        if enable_technical: enabled.append("技术")
        if enable_volatility: enabled.append("低波动")
        if enable_money_flow: enabled.append("资金流")
        if enable_industry_momentum: enabled.append("行业动量")
        if enable_risk: enabled.append("风险")
        
        logger.info(f"FactorCalculator initialized (启用因子: {', '.join(enabled)})")
    
    def calculate_all_factors(self, stock_codes: List[str], 
                             data_fetcher,
                             start_date: str = None,
                             end_date: str = None,
                             max_workers: int = 5) -> pd.DataFrame:
        """
        计算所有股票的因子（支持多线程并行）
        
        Args:
            stock_codes: 股票代码列表
            data_fetcher: DataFetcher实例（用于获取数据）
            start_date: 开始日期（格式: "20230101"），None表示自动计算
            end_date: 结束日期（格式: "20241231"），None表示使用当前日期
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
                factors = self.calculate_single_stock_factors(code, data_fetcher, start_date, end_date)
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
        # 动量因子需要 250 个交易日 ≈ 1.2 年，低波动也需要 ≈ 1 年
        # 默认取 end_date 前 2 年作为 start_date，确保数据足够
        if end_date is None:
            from datetime import datetime
            end_date = datetime.now().strftime("%Y%m%d")
        if start_date is None:
            from datetime import datetime, timedelta
            try:
                ed = datetime.strptime(end_date, "%Y%m%d")
                sd = ed - timedelta(days=730)   # 前 2 年
                start_date = sd.strftime("%Y%m%d")
            except Exception:
                start_date = "20200101"
        
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
        # 使用 end_date 作为报告期（如果 end_date 是 YYYYMMDD 格式）
        # 否则使用默认值
        report_date = end_date if end_date else "20211231"
        financial_data = data_fetcher.get_financial_data_single(code)
        # P1: 多期财务历史，用于 PEG 增长稳定性护栏
        financial_history = None
        try:
            if hasattr(data_fetcher, "get_financial_history"):
                financial_history = data_fetcher.get_financial_history(code, years=self.peg_stability_years)
        except Exception as e:
            logger.debug(f"Failed to fetch financial history for {code}: {e}")
            financial_history = None
        
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
        factors.update(self._calc_value_factors(valuation_data, financial_data, financial_history))
        
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
            # 获取市场收益率（用于计算 Beta）
            market_returns = np.array([])
            try:
                if hasattr(data_fetcher, 'get_index_returns'):
                    # 默认使用沪深300指数作为市场基准
                    market_returns = data_fetcher.get_index_returns(
                        index_code="000300.SH",
                        start_date=start_date,
                        end_date=end_date
                    )
                    if len(market_returns) > 0:
                        logger.debug(f"Got market returns: {len(market_returns)} days")
            except Exception as e:
                logger.warning(f"Failed to fetch market returns: {e}")
            
            factors.update(self._calc_volatility_factors(hist_data, market_returns))
        
        # 计算资金流因子（可选）
        if self.enable_money_flow:
            money_flow_data = data_fetcher.get_money_flow_data(code)
            factors.update(self._calc_money_flow_factors(money_flow_data))
        
        # 计算行业动量因子（可选）
        if self.enable_industry_momentum:
            try:
                # end_date 是方法的参数，表示选股日期
                industry_momentum_data = data_fetcher.get_industry_momentum_factor(code, end_date)
                if industry_momentum_data:
                    factors["industry_momentum"] = industry_momentum_data.get("industry_momentum", np.nan)
                    factors["industry_momentum_z"] = industry_momentum_data.get("industry_momentum_z", np.nan)
                    logger.debug(f"✓ Industry momentum factor added for {code}")
            except Exception as e:
                logger.error(f"Error fetching industry momentum for {code}: {e}")
                factors["industry_momentum"] = np.nan
                factors["industry_momentum_z"] = np.nan
        
        
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
                             financial_data: Optional[pd.DataFrame],
                             financial_history: Optional[pd.DataFrame] = None) -> Dict[str, float]:
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
            # 从估值数据中获取（优先 Tushare 原生列名，兼容 AkShare 中文别名）
            if valuation_data is not None and len(valuation_data) > 0:
                # 市盈率（反向）
                pe_val = None
                for col in ["pe_ttm", "pe", "PE_TTM", "市盈率-动态", "市盈率"]:
                    if col in valuation_data.columns:
                        pe_val = valuation_data[col].values[0]
                        break
                if pe_val is not None and not pd.isna(pe_val):
                    factors["VF1_PE"] = float(pe_val)

                # 市净率（反向）
                pb = None
                for col in ["pb", "市净率"]:
                    if col in valuation_data.columns:
                        pb = valuation_data[col].values[0]
                        break
                if pb is not None and not pd.isna(pb):
                    factors["VF2_PB"] = float(pb)

                # 市销率（反向）
                ps = None
                for col in ["ps_ttm", "ps", "市销率(TTM)", "市销率"]:
                    if col in valuation_data.columns:
                        ps = valuation_data[col].values[0]
                        break
                if ps is not None and not pd.isna(ps):
                    factors["VF3_PS"] = float(ps)

                # 股息率（正向）
                div_yield = None
                for col in ["dv_ttm", "dv_ratio", "股息率"]:
                    if col in valuation_data.columns:
                        div_yield = valuation_data[col].values[0]
                        break
                if div_yield is not None and not pd.isna(div_yield):
                    factors["VF6_dividend_yield"] = float(div_yield)

                # EV/EBITDA（反向，本地库无此列则保持 NaN）
                ev_ebitda_val = None
                for col in ["ev_ebitda", "EV_EBITDA", "企业价值倍率"]:
                    if col in valuation_data.columns:
                        ev_ebitda_val = valuation_data[col].values[0]
                        break
                if ev_ebitda_val is not None and not pd.isna(ev_ebitda_val):
                    factors["VF5_EV_EBITDA"] = float(ev_ebitda_val)

            # 从财务数据中获取 PEG = PE / 净利润同比增长率
            # 注意：Tushare 的 *_yoy 已是百分比（如 25.5 即 25.5%），无需再 x100
            # P1(价值陷阱护栏) 视频《低PE是打折还是陷阱》三道防护：
            if financial_data is not None and len(financial_data) > 0:
                pe = factors["VF1_PE"]
                growth = None
                for col in ["netprofit_yoy", "净利润同比增长率", "净利润增长率", "dt_netprofit_yoy"]:
                    if col in financial_data.columns and not pd.isna(financial_data[col].values[0]):
                        growth = float(financial_data[col].values[0])
                        break
                if (not np.isnan(pe)) and pe > 0 and growth is not None and (not pd.isna(growth)) and growth > 0:
                    peg_candidate = pe / growth
                    # 护栏② 基数效应：净利高增但营收没同步增长 / ROE<=0，
                    # 可能是低基数或一次性收益虚增，PEG 不可信 → 置 NaN
                    op_yoy = None
                    for col in ["or_yoy", "tr_yoy", "营业总收入同比增长率", "营业收入同比增长率"]:
                        if col in financial_data.columns and not pd.isna(financial_data[col].values[0]):
                            op_yoy = float(financial_data[col].values[0]); break
                    roe = None
                    for col in ["roe", "净资产收益率"]:
                        if col in financial_data.columns and not pd.isna(financial_data[col].values[0]):
                            roe = float(financial_data[col].values[0]); break
                    base_effect_ok = True
                    if op_yoy is not None and (op_yoy <= 0 or (roe is not None and roe <= 0)):
                        base_effect_ok = False
                    # 护栏③ 增长稳定性：多期历史连续正增长(年报口径)才保留 PEG
                    stable_ok = self._is_growth_stable(financial_history, years=self.peg_stability_years)
                    if base_effect_ok and stable_ok is not False:
                        factors["VF4_PEG"] = peg_candidate
                    else:
                        factors["VF4_PEG"] = np.nan
                else:
                    # 护栏① 无稳定利润 / 负增长：PEG 仅在 PE>0 且 增长率>0 有意义；
                    # 利润下滑(growth<0)算出的负PEG绝非"便宜"，置 NaN，
                    # 否则 factor_processor 里 VF4_PEG 是反向因子会把负PEG当超便宜加分
                    factors["VF4_PEG"] = np.nan

            logger.debug(f"Value factors calculated: PE={factors['VF1_PE']}, PB={factors['VF2_PB']}")

        except Exception as e:
            logger.error(f"Error calculating value factors: {e}")

        return factors

    def _is_growth_stable(self, history, years: int = 3):
        """
        增长稳定性护栏（视频坑③）。
        取最近 years 个年报(end_date 末4位=='1231')记录，要求 netprofit_yoy 与 or_yoy 全部为正。
        返回 True(稳定) / False(不稳定) / None(无年报数据，放行)。
        """
        if history is None or len(history) == 0:
            return None
        try:
            df = history.copy()
            df["end_date"] = df["end_date"].astype(str)
            annual = df[df["end_date"].str.endswith("1231")]
            if len(annual) == 0:
                return None
            annual = annual.sort_values("end_date").tail(years)
            if "netprofit_yoy" not in annual.columns or "or_yoy" not in annual.columns:
                return None
            np_yoy = annual["netprofit_yoy"]
            op_yoy = annual["or_yoy"]
            if np_yoy.notna().all() and op_yoy.notna().all() and (np_yoy > 0).all() and (op_yoy > 0).all():
                return True
            return False
        except Exception:
            return None

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
                # 优先 Tushare 原生列名，兼容 AkShare 中文别名
                # GF1 营收同比增长率（正向）
                rev_growth_val = None
                for col in ["or_yoy", "tr_yoy", "营业总收入同比增长率", "营业收入同比增长率", "营业总收入增长率", "营收增长率"]:
                    if col in financial_data.columns and not pd.isna(financial_data[col].values[0]):
                        rev_growth_val = float(financial_data[col].values[0])
                        break
                if rev_growth_val is not None:
                    factors["GF1_revenue_growth"] = rev_growth_val

                # GF2 净利润同比增长率（正向）
                profit_growth = None
                for col in ["netprofit_yoy", "净利润同比增长率", "净利润增长率", "dt_netprofit_yoy"]:
                    if col in financial_data.columns and not pd.isna(financial_data[col].values[0]):
                        profit_growth = float(financial_data[col].values[0])
                        break
                if profit_growth is not None:
                    factors["GF2_net_profit_growth"] = profit_growth

                # GF3 ROE（正向）
                roe = None
                for col in ["roe", "净资产收益率"]:
                    if col in financial_data.columns and not pd.isna(financial_data[col].values[0]):
                        roe = float(financial_data[col].values[0])
                        break
                if roe is not None:
                    factors["GF3_ROE"] = roe

                # GF4 ROA（正向）
                roa = None
                for col in ["roa", "总资产报酬率", "总资产净利率"]:
                    if col in financial_data.columns and not pd.isna(financial_data[col].values[0]):
                        roa = float(financial_data[col].values[0])
                        break
                if roa is not None:
                    factors["GF4_ROA"] = roa

                # GF5 毛利率水平/同比增长（单期用毛利率水平近似）
                gross_margin_val = None
                for col in ["grossprofit_margin", "销售毛利率", "毛利率"]:
                    if col in financial_data.columns:
                        gross_margin_val = financial_data[col].values
                        break
                if gross_margin_val is not None and len(gross_margin_val) >= 2 and not pd.isna(gross_margin_val[1]) and gross_margin_val[1] != 0:
                    factors["GF5_gross_margin_growth"] = (float(gross_margin_val[0]) - float(gross_margin_val[1])) / abs(float(gross_margin_val[1])) * 100
                elif gross_margin_val is not None and len(gross_margin_val) >= 1 and not pd.isna(gross_margin_val[0]):
                    # 单期数据：用毛利率水平作代理（已在百分比）
                    factors["GF5_gross_margin_growth"] = float(gross_margin_val[0])

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
                # 优先 Tushare 原生列名，兼容 AkShare 中文别名
                # QF1 资产负债率（反向）
                if "debt_to_assets" in financial_data.columns and not pd.isna(financial_data["debt_to_assets"].values[0]):
                    factors["QF1_asset_liability_ratio"] = float(financial_data["debt_to_assets"].values[0])
                elif "资产负债率" in financial_data.columns and not pd.isna(financial_data["资产负债率"].values[0]):
                    factors["QF1_asset_liability_ratio"] = float(financial_data["资产负债率"].values[0])

                # QF2 流动比率（正向）
                if "current_ratio" in financial_data.columns and not pd.isna(financial_data["current_ratio"].values[0]):
                    factors["QF2_current_ratio"] = float(financial_data["current_ratio"].values[0])
                elif "流动比率" in financial_data.columns and not pd.isna(financial_data["流动比率"].values[0]):
                    factors["QF2_current_ratio"] = float(financial_data["流动比率"].values[0])

                # QF3 总资产周转率（正向）
                if "assets_turn" in financial_data.columns and not pd.isna(financial_data["assets_turn"].values[0]):
                    factors["QF3_asset_turnover"] = float(financial_data["assets_turn"].values[0])
                elif "总资产周转率" in financial_data.columns and not pd.isna(financial_data["总资产周转率"].values[0]):
                    factors["QF3_asset_turnover"] = float(financial_data["总资产周转率"].values[0])

                # QF4 现金流质量（正向）：ocf_to_debt = 经营现金流/债务
                if "ocf_to_debt" in financial_data.columns and not pd.isna(financial_data["ocf_to_debt"].values[0]):
                    factors["QF4_cash_flow_quality"] = float(financial_data["ocf_to_debt"].values[0])
                elif "经营活动现金流量净额/净利润" in financial_data.columns and not pd.isna(financial_data["经营活动现金流量净额/净利润"].values[0]):
                    factors["QF4_cash_flow_quality"] = float(financial_data["经营活动现金流量净额/净利润"].values[0])

                # QF5 净利润现金含量（正向）：ocfps / revenue_ps
                try:
                    ocfps = financial_data["ocfps"].values[0] if "ocfps" in financial_data.columns else np.nan
                    revps = financial_data["revenue_ps"].values[0] if "revenue_ps" in financial_data.columns else np.nan
                    if not pd.isna(ocfps) and not pd.isna(revps) and revps != 0:
                        factors["QF5_cash_flow_to_revenue"] = float(ocfps) / float(revps)
                    elif "销售商品提供劳务收到的现金/营业收入" in financial_data.columns and not pd.isna(financial_data["销售商品提供劳务收到的现金/营业收入"].values[0]):
                        factors["QF5_cash_flow_to_revenue"] = float(financial_data["销售商品提供劳务收到的现金/营业收入"].values[0])
                except Exception:
                    pass

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
                # 获取收盘价（健壮的列名检测）
                close_col = None
                for col in ['收盘', 'close', 'Close', '收盘价']:
                    if col in hist_data.columns:
                        close_col = col
                        break
                
                if close_col is None:
                    logger.warning(f"No close price column found in hist_data. Columns: {list(hist_data.columns)}")
                    return factors
                
                close_prices = hist_data[close_col].values
                
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
                
                # 2. 相对强度（个股12个月收益率 vs 市场12个月收益率）
                # 理想情况从沪深300指数计算市场收益率，简化版本用合理的市场均值估算
                # A股长期年化收益约 8%-12%，此处用 8% 作为保守估计
                market_return = 0.08
                if not pd.isna(factors["MF4_return_12m"]):
                    factors["MF5_relative_strength"] = factors["MF4_return_12m"] - market_return
                    # 超额收益比简单比值更合理：(个股收益 - 市场收益)
            
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
        - TF4_turnover_pct: 换手率自身历史分位（自由流通口径 turnover_rate_f，
               Jim 原则②：和它自己平时比，消除规模偏差；仅描述活跃度，不预测方向）
        - TF5_bollinger_position: 布林带位置（正向，接近下轨为好）
        - TF6_volume_price_regime: 量价 regime（Jim 原则③：放量须配对价格动作，
               0=常态 1=放量+价变大 2=放量+价滞 3=放量+价跌；中性描述器）
        """
        factors = {}
        
        # 默认值
        factors["TF1_ma_bullish"] = np.nan
        factors["TF2_MACD"] = np.nan
        factors["TF3_RSI"] = np.nan
        factors["TF4_turnover_pct"] = np.nan
        factors["TF5_bollinger_position"] = np.nan
        factors["TF6_volume_price_regime"] = np.nan
        factors["TF7_kdj_j"] = np.nan
        
        try:
            if hist_data is not None and len(hist_data) > 0:
                # 获取收盘价和成交量（健壮的列名检测）
                close_col = None
                volume_col = None
                
                for col in ['收盘', 'close', 'Close', '收盘价']:
                    if col in hist_data.columns:
                        close_col = col
                        break
                
                for col in ['成交量', 'volume', 'Volume', 'vol']:
                    if col in hist_data.columns:
                        volume_col = col
                        break
                
                if close_col is None:
                    logger.warning(f"No close price column found in hist_data. Columns: {list(hist_data.columns)}")
                    return factors
                
                close_prices = hist_data[close_col].values
                
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
                
                # 4. 换手率自身历史分位（自由流通口径 turnover_rate_f）
                #    Jim 原则①分母显式(用 turnover_rate_f=流通股本口径) + ②和它自己平时比(自身分位)
                #    + 护栏(只描述活跃度，不预测方向)。消除原始成交量跨股票规模偏差。
                _turn_col = None
                for _c in ("turnover_rate_f", "turnover_rate"):
                    if _c in hist_data.columns:
                        _turn_col = _c
                        break
                if _turn_col is not None and len(hist_data) >= 40:
                    _tr = hist_data[_turn_col].astype(float).dropna()
                    if len(_tr) >= 30:
                        factors["TF4_turnover_pct"] = own_history_pct(_tr, window=60)
                elif "成交量" in hist_data.columns and len(hist_data) >= 20:
                    # 兜底：无换手率列时保留旧量比口径（仅作降级，不进方向 composite）
                    volumes = hist_data["成交量"].values
                    factors["TF4_turnover_pct"] = volumes[-1] / np.mean(volumes[-20:])
                
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

                # 6. 量价 regime（Jim 原则③：放量须配对价格动作，单维不可解读）
                #    TF4_turnover_pct 已达自身异常阈值时，再叠加当天价格方向分三类记录。
                _tp = factors.get("TF4_turnover_pct")
                if _tp is not None and not pd.isna(_tp):
                    _pc = latest_price_change(hist_data)
                    if _pc is not None:
                        factors["TF6_volume_price_regime"] = volume_price_regime(_tp, _pc)

                        # 7. KDJ 的 J 线（仅取 J，作为"价格位置变化速度"描述器）
                        #    Jim 第4期去魅：KDJ 测量近期价格位置及其变化速度，金差=计算结果非未来结果；
                        #    平台只取 J（放大 K-D 差，捕捉"在区间内移动的加速度"），注册 neutral，
                        #    不进 technical_score、不预测方向，仅作 overlay / MACD 策略语境确认门。
                        _hi_col = _lo_col = None
                        for _cand in (("最高", "最低"), ("high", "low")):
                            if _cand[0] in hist_data.columns and _cand[1] in hist_data.columns:
                                _hi_col, _lo_col = _cand
                                break
                        if _hi_col is not None and len(close_prices) >= 14:
                            try:
                                _hi = hist_data[_hi_col].astype(float).to_numpy()
                                _lo = hist_data[_lo_col].astype(float).to_numpy()
                                _k, _d = talib.STOCH(
                                    _hi, _lo, close_prices,
                                    fastk_period=9, slowk_period=3, slowk_matype=0,
                                    slowd_period=3, slowd_matype=0,
                                )
                                if not np.isnan(_k[-1]) and not np.isnan(_d[-1]):
                                    factors["TF7_kdj_j"] = float(3.0 * _k[-1] - 2.0 * _d[-1])
                            except Exception as _e:
                                logger.debug(f"KDJ-J calc skipped: {_e}")

            logger.debug(f"Technical factors calculated: MA_bullish={factors['TF1_ma_bullish']}")
            
        except Exception as e:
            logger.error(f"Error calculating technical factors: {e}")
        
        return factors
    
    def _calc_volatility_factors(self, hist_data: Optional[pd.DataFrame], 
                                   market_returns: np.ndarray = np.array([])) -> Dict[str, float]:
        """
        计算低波动因子（使用 TA-Lib STDDEV/BETA 提高性能）
        
        因子列表:
        - LVF1_hist_vol: 历史波动率（反向，越低越好）
        - LVF2_beta: 贝塔系数（反向，使用 TA-Lib BETA 函数）
        - LVF3_downside_vol: 下行波动率（反向）
        - LVF4_idiosyncratic_vol: 特质波动率（反向）
        - LVF5_VAR: 风险价值VaR（反向，越小越好）
        - LVF6_sharpe: 夏普比率（正向，越高越好）
        - LVF7_sortino: 索提诺比率（正向，越高越好）
        """
        factors = {}
        
        # 默认值
        factors["LVF1_hist_vol"] = np.nan
        factors["LVF2_beta"] = np.nan
        factors["LVF3_downside_vol"] = np.nan
        factors["LVF4_idiosyncratic_vol"] = np.nan
        factors["LVF5_VAR"] = np.nan
        
        # 初始化 returns 变量（防止在未定义时访问）
        returns = np.array([])
        
        try:
            if hist_data is not None and len(hist_data) > 0:
                # 获取收盘价（健壮的列名检测）
                close_col = None
                for col in ['收盘', 'close', 'Close', '收盘价']:
                    if col in hist_data.columns:
                        close_col = col
                        break
                
                if close_col is None:
                    logger.warning(f"No close price column found in hist_data. Columns: {list(hist_data.columns)}")
                    return factors
                
                close_prices = hist_data[close_col].values
                
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
                
                # 4. 贝塔系数（使用 TA-Lib BETA 函数）
                # Beta = Cov(资产收益, 市场收益) / Var(市场收益)
                if len(market_returns) > 0 and len(returns) >= 20:
                    try:
                        # 确保两个收益率序列长度一致
                        min_len = min(len(returns), len(market_returns))
                        asset_ret = returns[-min_len:]  # 取最近的 min_len 个收益率
                        mkt_ret = market_returns[-min_len:]
                        
                        # 使用 TA-Lib BETA 函数计算 Beta
                        # BETA(real0, real1, timeperiod=5)
                        # real0: 资产收益率序列
                        # real1: 市场收益率序列
                        beta_array = talib.BETA(
                            np.array(asset_ret, dtype=float),
                            np.array(mkt_ret, dtype=float),
                            timeperiod=min(20, min_len)
                        )
                        
                        # 取最后一个值（最新的 Beta）
                        if not np.isnan(beta_array[-1]):
                            factors["LVF2_beta"] = float(beta_array[-1])
                            logger.debug(f"Beta calculated using TA-Lib: {factors['LVF2_beta']:.4f}")
                    except Exception as e:
                        logger.warning(f"Failed to calculate Beta using TA-Lib: {e}")
                        # 备用：使用简化版本
                        if not pd.isna(factors["LVF1_hist_vol"]):
                            market_vol = 0.20
                            raw_beta = factors["LVF1_hist_vol"] / market_vol
                            factors["LVF2_beta"] = max(0.0, min(raw_beta, 3.0))
                
                # 5. 特质波动率（总波动率² - 系统性波动率²，取非负）
                if not np.isnan(factors["LVF1_hist_vol"]) and not np.isnan(factors["LVF2_beta"]):
                    market_vol = 0.20
                    systematic_vol = abs(factors["LVF2_beta"]) * market_vol
                    # 防止负数开平方
                    vol_squared_diff = np.maximum(0, factors["LVF1_hist_vol"]**2 - systematic_vol**2)
                    idiosyncratic_vol = np.sqrt(vol_squared_diff)
                    factors["LVF4_idiosyncratic_vol"] = idiosyncratic_vol
                
                # 6. 夏普比率（Sharpe Ratio）- 使用 quantstats 计算
                if len(returns) >= 20:
                    try:
                        # 将 returns 转换为 pandas Series（quantstats 需要）
                        returns_series = pd.Series(returns)
                        
                        # 使用 quantstats 计算夏普比率（年化）
                        # rf: 无风险利率（简化为 0）
                        # periods: 252（交易日数）
                        # annualize: 是否年化（True）
                        sharpe = qs.stats.sharpe(returns_series, rf=0.0, periods=252, annualize=True)
                        
                        if not np.isnan(sharpe):
                            factors["LVF6_sharpe"] = float(sharpe)
                            logger.debug(f"Sharpe Ratio calculated using quantstats: {sharpe:.4f}")
                    except Exception as e:
                        logger.warning(f"Failed to calculate Sharpe using quantstats: {e}")
                        # 备用：使用手动计算
                        avg_return = np.mean(returns) * 252
                        annualized_vol = factors.get("LVF1_hist_vol", np.nan)
                        if not np.isnan(annualized_vol) and annualized_vol > 0:
                            sharpe = avg_return / annualized_vol
                            factors["LVF6_sharpe"] = float(sharpe)
                
                # 7. 索提诺比率（Sortino Ratio）- 使用 quantstats 计算
                if len(returns) >= 20:
                    try:
                        # 将 returns 转换为 pandas Series（quantstats 需要）
                        returns_series = pd.Series(returns)
                        
                        # 使用 quantstats 计算索提诺比率（年化）
                        # rf: 无风险利率（简化为 0）
                        # periods: 252（交易日数）
                        # annualize: 是否年化（True）
                        sortino = qs.stats.sortino(returns_series, rf=0.0, periods=252, annualize=True)
                        
                        if not np.isnan(sortino):
                            factors["LVF7_sortino"] = float(sortino)
                            logger.debug(f"Sortino Ratio calculated using quantstats: {sortino:.4f}")
                    except Exception as e:
                        logger.warning(f"Failed to calculate Sortino using quantstats: {e}")
                        # 备用：使用手动计算
                        downside_vol = factors.get("LVF3_downside_vol", np.nan)
                        if not np.isnan(downside_vol) and downside_vol > 0:
                            avg_return = np.mean(returns) * 252
                            sortino = avg_return / downside_vol
                            factors["LVF7_sortino"] = float(sortino)
            
            logger.debug(f"Volatility factors calculated: HistVol={factors['LVF1_hist_vol']:.4f}, Sharpe={factors.get('LVF6_sharpe', np.nan):.4f}")
            
            # 8. 溃疡指数（Ulcer Index）- 使用 quantstats 计算
            # 溃疡指数是更好的风险控制指标，衡量回撤深度和持续时间
            ulcer_index = self._calc_ulcer_index(returns)
            if not np.isnan(ulcer_index):
                factors["LVF8_ulcer"] = ulcer_index
                logger.debug(f"Ulcer Index calculated: {ulcer_index:.4f}")
            
            # 9. 最大回撤（Max Drawdown）- 使用 quantstats 计算
            max_dd = self._calc_max_drawdown(returns)
            if not np.isnan(max_dd):
                factors["LVF9_max_drawdown"] = max_dd
                logger.debug(f"Max Drawdown calculated: {max_dd:.4f}")
            
        except Exception as e:
            logger.error(f"Error calculating volatility factors: {e}")
        
        return factors
    
    def _calc_ulcer_index(self, returns: np.ndarray) -> float:
        """
        计算溃疡指数（Ulcer Index）- 使用 quantstats
        
        溃疡指数衡量回撤的深度和持续时间，是更好的风险控制指标
        公式：UI = sqrt(avg((Drawdown_i)^2))
        
        Args:
            returns: 收益率序列
            
        Returns:
            溃疡指数（越小越好）
        """
        if len(returns) < 20:
            return np.nan
        
        try:
            # 将 returns 转换为 pandas Series（quantstats 需要）
            # 注意：quantstats 需要日期索引，否则会报错
            # 创建一个假的日期索引（用 pd.date_range()）
            dates = pd.date_range(start='2020-01-01', periods=len(returns), freq='D')
            returns_series = pd.Series(returns, index=dates)
            
            # 使用 quantstats 计算溃疡指数
            ulcer = qs.stats.ulcer_index(returns_series)
            
            if not np.isnan(ulcer):
                logger.debug(f"Ulcer Index calculated using quantstats: {ulcer:.4f}")
                return float(ulcer)
            else:
                return np.nan
                
        except Exception as e:
            logger.warning(f"Failed to calculate Ulcer Index using quantstats: {e}")
            # 备用：手动计算溃疡指数
            try:
                # 计算累计净值
                cumulative = np.cumprod(1 + returns)
                
                # 计算历史最高净值
                running_max = np.maximum.accumulate(cumulative)
                
                # 计算回撤百分比
                drawdown = (cumulative - running_max) / running_max
                
                # 计算溃疡指数：回撤平方和的平均值，然后开根号
                ulcer_sq = np.mean(drawdown ** 2)
                ulcer = np.sqrt(ulcer_sq)
                
                return float(ulcer)
            except Exception as e2:
                logger.warning(f"Failed to calculate Ulcer Index manually: {e2}")
                return np.nan
    
    def _calc_max_drawdown(self, returns: np.ndarray) -> float:
        """
        计算最大回撤（Max Drawdown）- 使用 quantstats
        
        最大回撤衡量投资组合在特定时期内从峰值到谷值的最大损失
        
        Args:
            returns: 收益率序列
            
        Returns:
            最大回撤（负数，如 -0.15 表示 -15%）
        """
        if len(returns) < 20:
            return np.nan
        
        try:
            # 将 returns 转换为 pandas Series（quantstats 需要）
            # 注意：quantstats 需要日期索引，否则会报错
            dates = pd.date_range(start='2020-01-01', periods=len(returns), freq='D')
            returns_series = pd.Series(returns, index=dates)
            
            # 使用 quantstats 计算最大回撤
            # 注意：qs.stats.max_drawdown() 返回的是负数（如 -0.15 表示 -15%）
            max_dd = qs.stats.max_drawdown(returns_series)
            
            if not np.isnan(max_dd):
                logger.debug(f"Max Drawdown calculated using quantstats: {max_dd:.4f}")
                return float(max_dd)
            else:
                return np.nan
                
        except Exception as e:
            logger.warning(f"Failed to calculate Max Drawdown using quantstats: {e}")
            # 备用：手动计算最大回撤
            try:
                # 计算累计净值
                cumulative = np.cumprod(1 + returns)
                
                # 计算历史最高净值
                running_max = np.maximum.accumulate(cumulative)
                
                # 计算回撤百分比
                drawdown = (cumulative - running_max) / running_max
                
                # 最大回撤（负数）
                max_dd = np.min(drawdown)
                
                return float(max_dd)
            except Exception as e2:
                logger.warning(f"Failed to calculate Max Drawdown manually: {e2}")
                return np.nan
    

    def _calc_money_flow_factors(self, money_flow_data: Optional[pd.DataFrame]) -> Dict[str, float]:
        """
        计算资金流因子（前缀 MWF = Money_Flow，与动量 MF 区分）
        
        因子列表:
        - MWF1_ultra_large_inflow: 超大单净流入（正向）
        - MWF2_large_inflow: 大单净流入（正向）
        - MWF3_medium_inflow: 中单净流入（正向）
        - MWF4_small_inflow: 小单净流入（反向，散户通常是反向指标）
        - MWF5_main_inflow: 主力资金净流入（超大单+大单，正向）
        - MWF6_inflow_intensity: 资金流强度（净流入/成交额，正向）
        """
        factors = {}
        
        # 默认值
        factors["MWF1_ultra_large_inflow"] = np.nan
        factors["MWF2_large_inflow"] = np.nan
        factors["MWF3_medium_inflow"] = np.nan
        factors["MWF4_small_inflow"] = np.nan
        factors["MWF5_main_inflow"] = np.nan
        factors["MWF6_inflow_intensity"] = np.nan
        
        try:
            if money_flow_data is not None and len(money_flow_data) > 0:
                # AkShare 返回的资金流数据格式
                # 可能包含的列：超大单净流入, 大单净流入, 中单净流入, 小单净流入, 净流入, 成交额
                
                # 超大单净流入
                if "超大单净流入" in money_flow_data.columns:
                    ultra_large = money_flow_data["超大单净流入"].values[0]
                    factors["MWF1_ultra_large_inflow"] = float(ultra_large) if not pd.isna(ultra_large) else np.nan
                
                # 大单净流入
                if "大单净流入" in money_flow_data.columns:
                    large = money_flow_data["大单净流入"].values[0]
                    factors["MWF2_large_inflow"] = float(large) if not pd.isna(large) else np.nan
                
                # 中单净流入
                if "中单净流入" in money_flow_data.columns:
                    medium = money_flow_data["中单净流入"].values[0]
                    factors["MWF3_medium_inflow"] = float(medium) if not pd.isna(medium) else np.nan
                
                # 小单净流入（反向指标，取负值）
                if "小单净流入" in money_flow_data.columns:
                    small = money_flow_data["小单净流入"].values[0]
                    small_val = float(small) if not pd.isna(small) else np.nan
                    # 小单净流入越大，说明散户买入越多，通常是反向指标，所以取负值
                    factors["MWF4_small_inflow"] = -small_val if not pd.isnan(small_val) else np.nan
                
                # 主力资金净流入（超大单 + 大单）
                if not pd.isna(factors["MWF1_ultra_large_inflow"]) and not pd.isna(factors["MWF2_large_inflow"]):
                    factors["MWF5_main_inflow"] = factors["MWF1_ultra_large_inflow"] + factors["MWF2_large_inflow"]
                
                # 资金流强度（净流入 / 成交额）
                if "净流入" in money_flow_data.columns and "成交额" in money_flow_data.columns:
                    net_inflow = money_flow_data["净流入"].values[0]
                    turnover = money_flow_data["成交额"].values[0]
                    
                    if not pd.isna(net_inflow) and not pd.isna(turnover) and turnover != 0:
                        factors["MWF6_inflow_intensity"] = float(net_inflow) / float(turnover)
                
                logger.debug(f"Money flow factors calculated: MainInflow={factors['MWF5_main_inflow']}")
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
