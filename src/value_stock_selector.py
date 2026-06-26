"""
价值投资策略 - 选股核心模块
基于价值投资量化策略.md文档
"""

import pandas as pd
import numpy as np
import sqlite3
import os
from typing import Optional, Dict, List
from loguru import logger


class ValueStockSelector:
    """价值投资选股器"""
    
    def __init__(self, config: dict, data_fetcher):
        """
        初始化选股器
        
        Args:
            config: 配置字典（VALUE_STRATEGY）
            data_fetcher: 数据获取器（DataFetcher实例）
        """
        self.config = config
        self.data_fetcher = data_fetcher
        
        # 从配置中读取参数
        self.date = config.get("date", "20240102")
        self.report_date = config.get("report_date", "20231231")  # 新增：财务数据报告期
        self.stock_pool = config.get("stock_pool", "hs300")
        self.market_cap_quantile = config.get("market_cap_quantile", 0.5)
        self.current_ratio_quantile = config.get("current_ratio_quantile", 0.5)
        self.roe_quantile = config.get("roe_quantile", 0.5)
        self.free_cash_flow_years = config.get("free_cash_flow_years", 5)
        self.revenue_growth_min = config.get("revenue_growth_min", 0.06)
        self.revenue_growth_max = config.get("revenue_growth_max", 0.30)
        self.eps_min = config.get("eps_min", 0.08)
        self.eps_max = config.get("eps_max", 0.50)
        self.top_n = config.get("top_n", 0)
        self.output_dir = config.get("output_dir", "data/results/value_strategy")
        self.output_file = config.get("output_file", "value_selection_{date}.csv")
        
        logger.info(f"ValueStockSelector initialized (date={self.date}, pool={self.stock_pool})")
        logger.info(f"  Criteria: market_cap > P{int(self.market_cap_quantile*100)}, "
                   f"current_ratio > P{int(self.current_ratio_quantile*100)}, "
                   f"ROE > P{int(self.roe_quantile*100)}, "
                   f"revenue_growth=[{self.revenue_growth_min:.0%}, {self.revenue_growth_max:.0%}], "
                   f"EPS=[{self.eps_min}, {self.eps_max}]")
        if self.free_cash_flow_years > 0:
            logger.info(f"  Free cash flow: require {self.free_cash_flow_years} consecutive years > 0")
        else:
            logger.info(f"  Free cash flow check: DISABLED")

        # 成分股缓存
        self._hs300_cache = None
        self._zz500_cache = None
        self._zz800_cache = None
        self._zz1000_cache = None
    
    # ------------------------------------------------------------------ #
    #  成分股查询（直接从本地数据库，不依赖 data_fetcher）
    # ------------------------------------------------------------------ #
    def _get_conn(self):
        return sqlite3.connect(self.data_fetcher.local_db_path)

    def _get_hs300_constituents(self) -> set:
        """获取沪深300成分股（缓存），返回6位代码集合"""
        if self._hs300_cache is not None:
            return self._hs300_cache
        conn = self._get_conn()
        df = pd.read_sql_query(
            "SELECT ts_code FROM index_constituent WHERE index_code = '000300.SH'",
            conn,
        )
        conn.close()
        # 去掉交易所后缀，只保留6位代码
        self._hs300_cache = set(df["ts_code"].str[:6].tolist()) if len(df) > 0 else set()
        return self._hs300_cache

    def _get_zz500_constituents(self) -> set:
        """获取中证500成分股（缓存），返回6位代码集合"""
        if self._zz500_cache is not None:
            return self._zz500_cache
        conn = self._get_conn()
        df = pd.read_sql_query(
            "SELECT ts_code FROM index_constituent WHERE index_code = '000905.SH'",
            conn,
        )
        conn.close()
        self._zz500_cache = set(df["ts_code"].str[:6].tolist()) if len(df) > 0 else set()
        return self._zz500_cache

    def _get_zz800_constituents(self) -> set:
        """获取中证800成分股（缓存），返回6位代码集合"""
        if self._zz800_cache is not None:
            return self._zz800_cache
        conn = self._get_conn()
        df = pd.read_sql_query(
            "SELECT ts_code FROM index_constituent WHERE index_code = '000906.SH'",
            conn,
        )
        conn.close()
        self._zz800_cache = set(df["ts_code"].str[:6].tolist()) if len(df) > 0 else set()
        return self._zz800_cache

    def _get_zz1000_constituents(self) -> set:
        """获取中证1000成分股（缓存），返回6位代码集合"""
        if self._zz1000_cache is not None:
            return self._zz1000_cache
        conn = self._get_conn()
        df = pd.read_sql_query(
            "SELECT ts_code FROM index_constituent WHERE index_code = '000852.SH'",
            conn,
        )
        conn.close()
        self._zz1000_cache = set(df["ts_code"].str[:6].tolist()) if len(df) > 0 else set()
        logger.info(f"  [中证1000] 获取到 {len(self._zz1000_cache)} 只成分股")
        return self._zz1000_cache

    def get_market_benchmarks(self, date: str) -> Dict[str, float]:
        """
        获取全A股市场基准值（中位数/平均值）
        
        Args:
            date: 交易日期（格式: "YYYYMMDD"）
            
        Returns:
            {
                'market_cap_median': float,
                'current_ratio_mean': float,
                'roe_mean': float,
            }
        """
        logger.info("=" * 60)
        logger.info("[1/4] 获取全A股市场基准值...")
        logger.info("=" * 60)
        
        benchmarks = {}
        
        # 1. 获取全A股市值数据，计算中位数
        logger.info(f"[1.1] 获取全A股市值数据 (date={date})...")
        df_mv = self.data_fetcher.get_market_cap_all_a(date)
        
        if df_mv is not None and len(df_mv) > 0:
            # 计算中位数
            market_cap_median = df_mv['total_mv'].quantile(self.market_cap_quantile)
            benchmarks['market_cap_median'] = market_cap_median
            logger.info(f"  ✓ 全A股市值数据: {len(df_mv)} 只股票")
            logger.info(f"  市值 P{int(self.market_cap_quantile*100)} 分位数: {market_cap_median:.0f} 万元")
        else:
            logger.error("  ✗ 无法获取全A股市值数据！")
            # 使用一个默认值（比如 500000 万元 = 50亿）
            benchmarks['market_cap_median'] = 500000
            logger.warning(f"  使用默认值: 500000 万元 (50亿)")
        
        # 2. 获取全A股流动比率数据，计算25%分位数
        logger.info(f"[1.2] 获取全A股流动比率数据 (end_date={self.report_date})...")
        df_cr = self.data_fetcher.get_current_ratio_all_a(self.report_date)
        
        if df_cr is not None and len(df_cr) > 0:
            # 计算25%分位数（更宽松的阈值）
            current_ratio_quantile = df_cr['current_ratio'].quantile(0.25)
            benchmarks['current_ratio_quantile'] = current_ratio_quantile
            logger.info(f"  ✓ 全A股流动比率数据: {len(df_cr)} 只股票")
            logger.info(f"  流动比率25%分位数: {current_ratio_quantile:.2f}")
        else:
            logger.error("  ✗ 无法获取全A股流动比率数据！")
            # 使用默认值（通常流动比率 > 2 算健康）
            benchmarks['current_ratio_quantile'] = 1.5
            logger.warning(f"  使用默认值: 1.5")
        
        # 3. 获取全A股ROE数据，计算25%分位数
        logger.info(f"[1.3] 获取全A股ROE数据 (end_date={self.report_date})...")
        df_roe = self.data_fetcher.get_roe_all_a(self.report_date)
        
        if df_roe is not None and len(df_roe) > 0:
            # 计算25%分位数（更宽松的阈值）
            roe_quantile = df_roe['roe'].quantile(0.25)
            benchmarks['roe_quantile'] = roe_quantile
            logger.info(f"  ✓ 全A股ROE数据: {len(df_roe)} 只股票")
            logger.info(f"  ROE 25%分位数: {roe_quantile:.2%}")
        else:
            logger.error("  ✗ 无法获取全A股ROE数据！")
            # 使用默认值（A股ROE 25%分位数约 2%-4%）
            benchmarks['roe_quantile'] = 0.02
            logger.warning(f"  使用默认值: 2%")
        
        logger.info(f"✓ 市场基准值获取完成")
        return benchmarks
    
    def fetch_stock_data(self, stock_codes: List[str], date: str) -> pd.DataFrame:
        """
        获取股票财务数据
        
        Args:
            stock_codes: 股票代码列表
            date: 报告期（格式: "YYYYMMDD"）
            
        Returns:
            DataFrame with all required financial indicators
        """
        logger.info("=" * 60)
        logger.info(f"[2/4] 获取 {len(stock_codes)} 只股票的财务数据...")
        logger.info("=" * 60)
        
        # 调用 DataFetcher 的 get_financial_data 方法
        # 注意：使用 report_date（报告期）来查询财务数据
        df = self.data_fetcher.get_financial_data(stock_codes, self.report_date)
        
        if df is None or len(df) == 0:
            logger.error("  ✗ 无法获取股票财务数据！")
            return pd.DataFrame()
        
        logger.info(f"  ✓ 成功获取 {len(df)} 只股票的财务数据")
        
        # 打印数据概览
        logger.info(f"  数据列: {list(df.columns)}")
        logger.info(f"  数据预览:")
        logger.info(f"\n{df.head(3).to_string(index=False)}")
        
        return df
    
    def apply_selection_criteria(self, df: pd.DataFrame, 
                                benchmarks: Dict[str, float],
                                date: str = None) -> pd.DataFrame:
        """
        应用六大选股条件筛选股票
        
        Args:
            df: 股票财务数据
            benchmarks: 市场基准值
            date: 选股日期（用于自由现金流检查）
            
        Returns:
            筛选后的DataFrame（包含所有通过条件的股票）
        """
        logger.info("=" * 60)
        logger.info("[3/4] 应用选股条件筛选股票...")
        logger.info("=" * 60)
        
        if df is None or len(df) == 0:
            logger.warning("输入DataFrame为空，无法筛选")
            return pd.DataFrame()
        
        result_df = df.copy()
        original_count = len(result_df)
        
        logger.info(f"初始股票数量: {original_count}")
        logger.info(f"市场基准值: {benchmarks}")
        
        # ===== 条件1: 市值 > 全市场中位数 =====
        logger.info("\n[条件1] 市值 > 全市场中位数...")
        if 'total_mv' in result_df.columns and 'market_cap_median' in benchmarks:
            mask1 = result_df['total_mv'] > benchmarks['market_cap_median']
            result_df = result_df[mask1]
            logger.info(f"  条件: total_mv > {benchmarks['market_cap_median']:.0f} 万元")
            logger.info(f"  通过: {len(result_df)} / {original_count} ({(len(result_df)/original_count*100):.1f}%)")
        else:
            logger.warning("  无法应用条件1: 缺少 total_mv 或 market_cap_median")
        
        # ===== 条件2: 流动比率 > 全市场25%分位数 =====
        logger.info("\n[条件2] 流动比率 > 全市场25%分位数...")
        if 'current_ratio' in result_df.columns and 'current_ratio_quantile' in benchmarks:
            mask2 = result_df['current_ratio'] > benchmarks['current_ratio_quantile']
            result_df = result_df[mask2]
            logger.info(f"  条件: current_ratio > {benchmarks['current_ratio_quantile']:.2f}")
            logger.info(f"  通过: {len(result_df)} / {original_count} ({(len(result_df)/original_count*100):.1f}%)")
        else:
            logger.warning("  无法应用条件2: 缺少 current_ratio 或 current_ratio_quantile")
        
        # ===== 条件3: ROE > 全市场25%分位数 =====
        logger.info("\n[条件3] ROE > 全市场25%分位数...")
        if 'roe' in result_df.columns and 'roe_quantile' in benchmarks:
            mask3 = result_df['roe'] > benchmarks['roe_quantile']
            result_df = result_df[mask3]
            logger.info(f"  条件: roe > {benchmarks['roe_quantile']:.2%}")
            logger.info(f"  通过: {len(result_df)} / {original_count} ({(len(result_df)/original_count*100):.1f}%)")
        else:
            logger.warning("  无法应用条件3: 缺少 roe 或 roe_quantile")
        
        # ===== 条件4: 近N年自由现金流为正 =====
        if self.free_cash_flow_years > 0:
            logger.info(f"\n[条件4] 近 {self.free_cash_flow_years} 年自由现金流为正...")
            
            if date is not None:
                # 计算对应的年报报告期（取前一年的12-31）
                # 例如：date=20240102 → end_date=20231231
                date_year = int(date[:4])
                end_date = f"{date_year - 1}1231"
                
                logger.info(f"  检查历史FCFF数据 (end_date={end_date}, years={self.free_cash_flow_years})...")
                
                # 调用DataFetcher获取历史FCFF数据
                fcff_df = self.data_fetcher.get_historical_fcff(
                    stock_codes=result_df['ts_code'].tolist(),
                    end_date=end_date,
                    years=self.free_cash_flow_years
                )
                
                if fcff_df is not None and len(fcff_df) > 0:
                    # 合并到result_df
                    result_df = result_df.merge(
                        fcff_df[['ts_code', 'fcff_years_all_positive']], 
                        on='ts_code', 
                        how='left'
                    )
                    
                    # 应用条件4：fcff_years_all_positive == True
                    mask4 = result_df['fcff_years_all_positive'] == True
                    result_df = result_df[mask4]
                    
                    logger.info(f"  条件: 近{self.free_cash_flow_years}年FCFF都>0")
                    logger.info(f"  通过: {len(result_df)} / {original_count} ({len(result_df)/original_count*100:.1f}%)")
                else:
                    logger.warning("  无法获取历史FCFF数据，条件4跳过")
            else:
                logger.warning("  无法应用条件4: 缺少 date 参数")
        else:
            logger.info(f"\n[条件4] 自由现金流检查: 已跳过 (free_cash_flow_years=0)")
        
        # ===== 条件5: 营收增长率区间 =====
        logger.info(f"\n[条件5] 营收增长率 >= {self.revenue_growth_min:.0%}...")
        if 'op_yoy' in result_df.columns:
            # 下限检查
            mask5 = result_df['op_yoy'] >= self.revenue_growth_min * 100
            
            # 上限检查（如果revenue_growth_max > 0）
            if self.revenue_growth_max > 0:
                logger.info(f"  条件: op_yoy >= {self.revenue_growth_min:.0%} AND op_yoy <= {self.revenue_growth_max:.0%}")
                mask5 = mask5 & (result_df['op_yoy'] <= self.revenue_growth_max * 100)
            else:
                logger.info(f"  条件: op_yoy >= {self.revenue_growth_min:.0%} (无上限)")
            
            result_df = result_df[mask5]
            logger.info(f"  通过: {len(result_df)} / {original_count} ({(len(result_df)/original_count*100):.1f}%)")
        else:
            logger.warning("  无法应用条件5: 缺少 op_yoy")
        
        # ===== 条件6: EPS区间 =====
        logger.info(f"\n[条件6] EPS >= {self.eps_min}...")
        if 'eps' in result_df.columns:
            # 下限检查
            mask6 = result_df['eps'] >= self.eps_min
            
            # 上限检查（如果eps_max > 0）
            if self.eps_max > 0:
                logger.info(f"  条件: eps >= {self.eps_min} AND eps <= {self.eps_max}")
                mask6 = mask6 & (result_df['eps'] <= self.eps_max)
            else:
                logger.info(f"  条件: eps >= {self.eps_min} (无上限)")
            
            result_df = result_df[mask6]
            logger.info(f"  通过: {len(result_df)} / {original_count} ({(len(result_df)/original_count*100):.1f}%)")
        else:
            logger.warning("  无法应用条件6: 缺少 eps")
        
        # ===== 条件7: 股价 > 60日均线（趋势确认）=====
        if date is not None:
            result_df = self._filter_by_ma60(result_df, date)
        
        # ===== 条件8: 买入前1个月涨幅 > -5%（不在自由落体）=====
        if date is not None:
            result_df = self._filter_by_momentum(result_df, date)
        
        # ===== 条件9: 最近成交量 > 60日平均（有资金关注）=====
        if date is not None:
            result_df = self._filter_by_volume(result_df, date)
        
        # ===== 最终统计 =====
        logger.info("\n" + "=" * 60)
        logger.info(f"[筛选完成] 最终股票数量: {len(result_df)} / {original_count}")
        logger.info("=" * 60)
        
        if len(result_df) > 0:
            logger.info(f"\n筛选结果预览 (前10只):")
            logger.info(f"\n{result_df.head(10).to_string(index=False)}")
        
        return result_df
    
    def _filter_by_ma60(self, df: pd.DataFrame, trade_date: str) -> pd.DataFrame:
        """
        过滤：股价 > 60日均线（趋势确认）
        
        Args:
            df: 候选股票DataFrame
            trade_date: 交易日期
            
        Returns:
            过滤后的DataFrame
        """
        if len(df) == 0:
            return df
        
        logger.info(f"\n[条件7] 股价 > 60日均线...")

        passed_codes = []

        with sqlite3.connect(self.data_fetcher.local_db_path) as conn:
            for _, row in df.iterrows():
                ts_code = row['ts_code']

                # 获取过去60个交易日的收盘价
                df_prices = pd.read_sql_query("""
                    SELECT trade_date, close
                    FROM daily
                    WHERE ts_code = ? AND trade_date <= ?
                    ORDER BY trade_date DESC
                    LIMIT 60
                """, conn, params=(ts_code, trade_date))

                if len(df_prices) < 60:
                    continue  # 数据不足60天，跳过

                # 计算60日均线（简单平均）
                ma60 = df_prices['close'].mean()

                # 获取最新收盘价（trade_date）
                latest_price = pd.read_sql_query("""
                    SELECT close
                    FROM daily
                    WHERE ts_code = ? AND trade_date = ?
                """, conn, params=(ts_code, trade_date))

                if len(latest_price) == 0:
                    continue  # 无数据

                close = latest_price.iloc[0]['close']

                # 判断：收盘价 > 60日均线
                if close > ma60:
                    passed_codes.append(ts_code)

        result = df[df['ts_code'].isin(passed_codes)].copy()
        logger.info(f"  条件: close > MA60")
        logger.info(f"  通过: {len(result)} / {len(df)} ({len(result)/len(df)*100:.1f}%)")

        return result

    def _filter_by_momentum(self, df: pd.DataFrame, trade_date: str) -> pd.DataFrame:
        """
        过滤：买入前1个月涨幅 > -5%（不在自由落体）

        Args:
            df: 候选股票DataFrame
            trade_date: 交易日期

        Returns:
            过滤后的DataFrame
        """
        if len(df) == 0:
            return df

        logger.info(f"\n[条件8] 买入前1个月涨幅 > -5%...")

        with sqlite3.connect(self.data_fetcher.local_db_path) as conn:
            # 获取 trade_date 前20个交易日的日期
            df_dates = pd.read_sql_query("""
                SELECT DISTINCT trade_date
                FROM daily
                WHERE trade_date <= ?
                ORDER BY trade_date DESC
                LIMIT 21
            """, conn, params=(trade_date,))

            if len(df_dates) < 21:
                logger.warning("  数据不足，跳过此条件")
                return df

            # 当前日期
            current_date = df_dates.iloc[0]['trade_date']
            # 前20个交易日（约1个月）
            prev_date = df_dates.iloc[20]['trade_date']

            passed_codes = []
            for _, row in df.iterrows():
                ts_code = row['ts_code']

                # 获取当前价格和1个月前价格
                df_prices = pd.read_sql_query("""
                    SELECT trade_date, close
                    FROM daily
                    WHERE ts_code = ? AND trade_date IN (?, ?)
                    ORDER BY trade_date
                """, conn, params=(ts_code, prev_date, current_date))

                if len(df_prices) < 2:
                    continue  # 数据不足

                price_prev = df_prices.iloc[0]['close']
                price_curr = df_prices.iloc[1]['close']

                # 计算涨幅
                gain_1m = (price_curr - price_prev) / price_prev

                # 判断：涨幅 > -5%
                if gain_1m > -0.05:
                    passed_codes.append(ts_code)

        result = df[df['ts_code'].isin(passed_codes)].copy()
        logger.info(f"  条件: 1个月涨幅 > -5%")
        logger.info(f"  通过: {len(result)} / {len(df)} ({len(result)/len(df)*100:.1f}%)")

        return result

    def _filter_by_volume(self, df: pd.DataFrame, trade_date: str) -> pd.DataFrame:
        """
        过滤：最近成交量 > 60日平均成交量（有资金关注）

        Args:
            df: 候选股票DataFrame
            trade_date: 交易日期

        Returns:
            过滤后的DataFrame
        """
        if len(df) == 0:
            return df

        logger.info(f"\n[条件9] 最近成交量 > 60日平均...")

        passed_codes = []

        with sqlite3.connect(self.data_fetcher.local_db_path) as conn:
            for _, row in df.iterrows():
                ts_code = row['ts_code']

                # 获取过去60个交易日的成交量
                df_vol = pd.read_sql_query("""
                    SELECT trade_date, vol
                    FROM daily
                    WHERE ts_code = ? AND trade_date <= ?
                    ORDER BY trade_date DESC
                    LIMIT 60
                """, conn, params=(ts_code, trade_date))

                if len(df_vol) < 60:
                    continue  # 数据不足60天，跳过

                # 计算60日平均成交量
                vol_avg_60 = df_vol['vol'].mean()

                # 获取最新成交量（trade_date）
                latest_vol = pd.read_sql_query("""
                    SELECT vol
                    FROM daily
                    WHERE ts_code = ? AND trade_date = ?
                """, conn, params=(ts_code, trade_date))

                if len(latest_vol) == 0:
                    continue  # 无数据

                vol_latest = latest_vol.iloc[0]['vol']

                # 判断：最新成交量 > 60日平均
                if vol_latest > vol_avg_60:
                    passed_codes.append(ts_code)

        result = df[df['ts_code'].isin(passed_codes)].copy()
        logger.info(f"  条件: 最新成交量 > 60日平均")
        logger.info(f"  通过: {len(result)} / {len(df)} ({len(result)/len(df)*100:.1f}%)")

        return result
    
    def select_stocks(self, date: str = None, 
                     top_n: int = None) -> pd.DataFrame:
        """
        执行选股（主流程）
        
        Args:
            date: 选股日期（默认使用配置值）
            top_n: 选股数量（默认使用配置值）
            
        Returns:
            筛选后的股票DataFrame
        """
        # 使用参数或配置值
        date = date or self.date
        top_n = top_n or self.top_n
        
        logger.info("\n" + "=" * 60)
        logger.info("价值投资策略 - 选股系统")
        logger.info("=" * 60)
        logger.info(f"选股日期: {date}")
        logger.info(f"股票池: {self.stock_pool}")
        logger.info(f"选股数量: {top_n if top_n > 0 else '不限制 (按条件筛选)'}")
        logger.info("=" * 60 + "\n")
        
        # ===== Step 1: 获取股票池 =====
        logger.info("[Step 1] 获取股票池...")
        # ===== Step 1: 获取股票池（直接从本地数据库，不依赖 data_fetcher）=====
        logger.info("[Step 1] 获取股票池...")
        if self.stock_pool == "hs300":
            constituents = self._get_hs300_constituents()
            stock_pool_df = pd.DataFrame({"code": [c[:6] for c in constituents]})
            logger.info(f"  [沪深300] 获取到 {len(stock_pool_df)} 只成分股")
        elif self.stock_pool == "zz500":
            constituents = self._get_zz500_constituents()
            stock_pool_df = pd.DataFrame({"code": [c[:6] for c in constituents]})
            logger.info(f"  [中证500] 获取到 {len(stock_pool_df)} 只成分股")
        elif self.stock_pool == "zz800":
            constituents = self._get_zz800_constituents()
            stock_pool_df = pd.DataFrame({"code": [c[:6] for c in constituents]})
            logger.info(f"  [中证800] 获取到 {len(stock_pool_df)} 只成分股")
        elif self.stock_pool == "zz1000":
            constituents = self._get_zz1000_constituents()
            stock_pool_df = pd.DataFrame({"code": [c[:6] for c in constituents]})
            logger.info(f"  [中证1000] 获取到 {len(stock_pool_df)} 只成分股")
        elif self.stock_pool == "all":
            # 全A股模式：从数据库 stock_basic 表获取所有A股
            import sqlite3
            conn = sqlite3.connect(self.data_fetcher.local_db_path)
            stock_pool_df = pd.read_sql_query(
                "SELECT ts_code, name FROM stock_basic WHERE ts_code NOT LIKE '%.BJ' ORDER BY ts_code",
                conn,
            )
            conn.close()
            # 提取6位代码（去掉交易所后缀 .SZ/.SH/.BJ）
            stock_pool_df['code'] = stock_pool_df['ts_code'].str.extract(r'(\d{6})', expand=False)
            stock_pool_df = stock_pool_df[['code', 'name']].dropna(subset=['code'])
            logger.info(f"  [全A股] 获取到 {len(stock_pool_df)} 只股票")
        else:
            logger.error(f"未知的股票池: {self.stock_pool}")
            return pd.DataFrame()
        
        if stock_pool_df is None or len(stock_pool_df) == 0:
            logger.error(f"无法获取 {self.stock_pool} 成分股！")
            return pd.DataFrame()
        
        stock_codes = stock_pool_df['code'].tolist()
        logger.info(f"✓ 获取到 {len(stock_codes)} 只 {self.stock_pool.upper()} 成分股\n")
        
        # ===== Step 2: 获取市场基准值（全A股）=====
        benchmarks = self.get_market_benchmarks(date)
        
        # ===== Step 3: 获取股票财务数据 =====
        financial_df = self.fetch_stock_data(stock_codes, date)
        
        if financial_df is None or len(financial_df) == 0:
            logger.error("无法获取股票财务数据，选股失败！")
            return pd.DataFrame()
        
        # ===== Step 4: 应用选股条件 =====
        selected_df = self.apply_selection_criteria(financial_df, benchmarks, date)
        
        # ===== Step 5: 限制选股数量（如果需要）=====
        if top_n > 0 and len(selected_df) > top_n:
            logger.info(f"\n限制选股数量: {len(selected_df)} -> {top_n}")
            selected_df = selected_df.head(top_n)
        
        # ===== Step 6: 添加选股日期列 =====
        if len(selected_df) > 0:
            selected_df.insert(0, 'selection_date', date)
        
        logger.info("\n" + "=" * 60)
        logger.info(f"[选股完成] 共找到 {len(selected_df)} 只符合条件的股票")
        logger.info("=" * 60 + "\n")

        # 统一列名：确保 code 列为6位数字格式（与 dividend_low_vol_selector.py 一致）
        if "ts_code" in selected_df.columns and "code" not in selected_df.columns:
            selected_df["code"] = selected_df["ts_code"].str.extract(r"(\d{6})", expand=False)

        # 补全 name 列（与 dividend_low_vol_selector 行为一致）
        if "name" not in selected_df.columns and "ts_code" in selected_df.columns:
            import sqlite3
            name_map = {}
            with sqlite3.connect(self.data_fetcher.local_db_path) as conn:
                for ts_code in selected_df["ts_code"].unique():
                    r = pd.read_sql_query(
                        "SELECT name FROM stock_basic WHERE ts_code = ? LIMIT 1",
                        conn, params=(ts_code,),
                    )
                    name_map[ts_code] = r.iloc[0, 0] if len(r) > 0 else ts_code
            selected_df["name"] = selected_df["ts_code"].map(name_map)

        return selected_df
    
    def export_to_csv(self, df: pd.DataFrame, 
                      filename: Optional[str] = None,
                      output_dir: Optional[str] = None) -> str:
        """
        导出选股结果到CSV
        
        Args:
            df: 要导出的DataFrame
            filename: 文件名（可选，自动生成）
            output_dir: 输出目录（可选，使用配置值）
            
        Returns:
            保存的文件路径
        """
        if df is None or len(df) == 0:
            logger.warning("DataFrame为空，不导出")
            return ""
        
        # 确定输出目录
        output_dir = output_dir or self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 确定文件名
        if filename is None:
            # 使用配置中的文件名模板
            date_str = self.date
            filename = self.output_file.format(date=date_str)
        
        # 完整文件路径
        filepath = os.path.join(output_dir, filename)
        
        # 导出到CSV
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        
        logger.info(f"✓ 选股结果已保存到: {filepath}")
        logger.info(f"  共 {len(df)} 只股票")
        
        return filepath
