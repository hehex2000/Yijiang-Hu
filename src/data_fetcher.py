"""
数据获取模块 - DataFetcher Class
支持 AkShare、Tushare 和本地 SQLite 数据库
"""

import akshare as ak
import pandas as pd
import time
import threading
import sqlite3
import os
from typing import Optional, Callable, Any
from loguru import logger


class RateLimiter:
    """线程安全的速率限制器（令牌桶算法）"""
    
    def __init__(self, calls_per_minute: int = 200):
        """
        Args:
            calls_per_minute: 每分钟最大调用次数
        """
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute
        self.last_request_time = 0.0
        self.lock = threading.Lock()
    
    def wait(self):
        """等待直到可以发送下一个请求"""
        with self.lock:
            now = time.time()
            time_since_last = now - self.last_request_time
            
            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                time.sleep(sleep_time)
            
            self.last_request_time = time.time()


class DataFetcher:
    """数据获取器 - 支持混合数据源，可配置主数据源"""
    
    def __init__(self, primary_source: str = "akshare", 
                 tushare_token: str = None,
                 use_akshare_backup: bool = True,
                 use_tushare_backup: bool = True,
                 local_db_path: str = None):
        """
        初始化数据获取器
        
        Args:
            primary_source: 主数据源 ("akshare", "tushare", "local_db")
            tushare_token: Tushare API token
            use_akshare_backup: 是否启用AkShare作为备用数据源
            use_tushare_backup: 是否启用Tushare作为备用数据源
            local_db_path: 本地SQLite数据库路径（如 "D:\\tu-shareData\\astock_daily.db"）
        """
        self.primary_source = primary_source.lower()
        self.use_akshare = True  # AkShare 不需要token，总是可用
        self.use_tushare = False
        self.ts_pro = None
        self.local_db_path = local_db_path
        self.local_db_available = False
        
        # 初始化本地数据库连接
        if local_db_path and os.path.exists(local_db_path):
            try:
                # 测试连接
                conn = sqlite3.connect(local_db_path)
                conn.close()
                self.local_db_available = True
                logger.info(f"Local DB initialized: {local_db_path}")
                
                # 如果指定了 local_db_path 且有效，自动设置为主数据源
                if primary_source == "akshare" or primary_source == "tushare":
                    logger.info(f"Auto-setting primary_source to 'local_db' (local DB available)")
                    self.primary_source = "local_db"
            except Exception as e:
                logger.warning(f"Failed to connect to local DB: {e}")
        elif local_db_path:
            logger.warning(f"Local DB path not found: {local_db_path}")
        
        # 初始化Tushare（如果需要）
        if tushare_token:
            try:
                import tushare as ts
                ts.set_token(tushare_token)
                self.ts_pro = ts.pro_api()
                self.use_tushare = True
                logger.info("Tushare initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Tushare: {e}")
        
        # 确定备用数据源是否可用
        self.akshare_backup_available = use_akshare_backup and self.use_akshare
        self.tushare_backup_available = use_tushare_backup and self.use_tushare
        
        # 初始化速率限制器（Tushare限制200次/分钟）
        self.rate_limiter = RateLimiter(calls_per_minute=200)
        
        # 预取股票名称缓存（从本地数据库或Tushare批量获取）
        self._stock_name_cache = {}
        if self.local_db_available:
            try:
                conn = sqlite3.connect(self.local_db_path)
                df_basic = pd.read_sql_query("SELECT ts_code, symbol, name FROM stock_basic", conn)
                conn.close()
                if df_basic is not None and len(df_basic) > 0:
                    for _, row in df_basic.iterrows():
                        simple_code = str(row['symbol'])  # 000001 无后缀
                        self._stock_name_cache[simple_code] = str(row['name'])
                    logger.info(f"✓ 已缓存 {len(self._stock_name_cache)} 只股票名称(本地DB)")
            except Exception as e:
                logger.warning(f"本地DB批量获取股票名称失败: {e}")
        
        elif self.use_tushare and self.ts_pro is not None:
            try:
                df_basic = self.ts_pro.stock_basic(
                    exchange='', list_status='L', fields='ts_code,symbol,name'
                )
                if df_basic is not None and len(df_basic) > 0:
                    for _, row in df_basic.iterrows():
                        simple_code = str(row['symbol'])  # 000001 无后缀
                        self._stock_name_cache[simple_code] = str(row['name'])
                logger.info(f"✓ 已缓存 {len(self._stock_name_cache)} 只股票名称(Tushare)")
            except Exception as e:
                logger.warning(f"Tushare批量获取股票名称失败: {e}")
        
        logger.info(f"DataFetcher initialized (Primary: {self.primary_source}, "
                   f"LocalDB: {self.local_db_available}, "
                   f"AkShare backup: {self.akshare_backup_available}, "
                   f"Tushare backup: {self.tushare_backup_available})")
    
    def fetch_with_fallback(self, func_ak: Callable, func_ts: Callable, 
                          *args, **kwargs) -> Any:
        """
        根据配置的主数据源，优先使用主源，失败时使用备用源
        包含重试机制和延迟
        """
        import time
        
        # 根据主数据源配置决定调用顺序
        if self.primary_source == "tushare":
            # 主: Tushare, 备: AkShare
            return self._fetch_with_order(
                func_primary=func_ts,
                func_backup=func_ak if self.akshare_backup_available else None,
                primary_name="Tushare",
                backup_name="AkShare",
                *args, **kwargs
            )
        else:
            # 主: AkShare, 备: Tushare (默认行为)
            return self._fetch_with_order(
                func_primary=func_ak,
                func_backup=func_ts if self.tushare_backup_available else None,
                primary_name="AkShare",
                backup_name="Tushare",
                *args, **kwargs
            )
    
    def _fetch_with_order(self, func_primary: Callable, func_backup: Callable,
                         primary_name: str, backup_name: str,
                         *args, **kwargs) -> Any:
        """
        执行抓取，优先使用主源，失败后使用备用源
        
        Args:
            func_primary: 主数据源函数
            func_backup: 备用数据源函数（可以是None）
            primary_name: 主数据源名称（用于日志）
            backup_name: 备用数据源名称（用于日志）
        """
        # 尝试使用主数据源（最多3次）
        for attempt in range(3):
            try:
                # 速率限制（仅对Tushare API）
                if primary_name == "Tushare":
                    self.rate_limiter.wait()
                
                logger.debug(f"Trying {primary_name}: {func_primary.__name__} (attempt {attempt+1}/3)")
                result = func_primary(*args, **kwargs)
                time.sleep(0.1)  # 避免请求过快
                return result
            except Exception as e:
                logger.warning(f"{primary_name} failed (attempt {attempt+1}/3): {e}")
                if attempt < 2:
                    time.sleep(2)  # 重试前等待2秒
                else:
                    # 3次都失败，尝试备用数据源
                    if func_backup is not None:
                        try:
                            logger.info(f"Trying {backup_name} fallback: {func_backup.__name__}")
                            result = func_backup(*args, **kwargs)
                            return result
                        except Exception as e2:
                            logger.error(f"{backup_name} also failed: {e2}")
                            raise Exception(f"Both {primary_name} and {backup_name} failed.")
                    else:
                        raise e
    
    def get_hs300_components(self) -> pd.DataFrame:
        """
        获取沪深300成分股
        
        Returns:
            包含代码、名称、行业等信息的DataFrame
        """
        logger.info("Fetching HS300 components...")
        
        def _get_hs300_ak():
            """使用AkShare获取沪深300成分股"""
            df = ak.index_stock_cons_csindex(symbol="000300")
            # AkShare 返回格式: 日期, 指数代码, 指数名称, 指数英文名称, 成分券代码, 成分券名称, ...
            # 提取成分券代码和成分券名称
            result = df[['成分券代码', '成分券名称']].copy()
            result.columns = ['code', 'name']
            return result
        
        def _get_hs300_ts():
            """使用Tushare获取沪深300成分股"""
            if self.ts_pro is None:
                raise Exception("Tushare not initialized")
            
            # 使用 index_member API 获取当前成分股
            df = self.ts_pro.index_member(index_code='000300.SH')
            
            if df is not None and len(df) > 0:
                # 过滤出当前仍在成分股中的股票 (out_date 为 NaN)
                import numpy as np
                current = df[df['out_date'].isna()] if 'out_date' in df.columns else df
                
                if len(current) > 0:
                    # 转换为标准格式（简单格式，不含.SH/.SZ后缀）
                    result = current[['con_code']].copy()
                    result.columns = ['code']
                    # 移除 .SH 或 .SZ 后缀，统一为简单格式
                    result['code'] = result['code'].str.replace(r'\.(SH|SZ)$', '', regex=True)
                    result['name'] = ''  # Tushare 不提供股票名称
                    return result
            
            # 如果 index_member 没有数据，尝试使用 index_weight
            try:
                df_weight = self.ts_pro.index_weight(
                    index_code='000300.SH',
                    start_date='20240101',
                    end_date='20241231'
                )
                if df_weight is not None and len(df_weight) > 0:
                    latest_date = df_weight['trade_date'].max()
                    latest = df_weight[df_weight['trade_date'] == latest_date]
                    result = latest[['con_code']].copy()
                    result.columns = ['code']
                    # 移除后缀
                    result['code'] = result['code'].str.replace(r'\.(SH|SZ)$', '', regex=True)
                    result['name'] = ''
                    return result
            except:
                pass
            
            raise Exception("No HS300 components data available from Tushare")
        
        try:
            result = self.fetch_with_fallback(_get_hs300_ak, _get_hs300_ts)
            logger.info(f"✓ Got {len(result)} HS300 components")
            return result
        except Exception as e:
            logger.error(f"Failed to fetch HS300 components: {e}")
            raise
    
    def get_stock_info(self, code: str) -> dict:
        """
        获取股票基本信息（名称、市值）
        优先使用Tushare批量缓存，失败后用AkShare单只查询
        
        Args:
            code: 股票代码
            
        Returns:
            包含股票信息的字典（始终返回dict，避免None中断流程）
        """
        logger.debug(f"Fetching stock info for {code}")
        
        # 1. 优先使用Tushare缓存（启动时已批量获取）
        if hasattr(self, '_stock_name_cache') and code in self._stock_name_cache:
            name = self._stock_name_cache[code]
            logger.debug(f"✓ 从Tushare缓存获取股票名称: {code} = {name}")
            return {
                'code': code,
                'name': name,
                'market_cap': None  # 市值从valuation数据获取
            }
        
        # 2. 使用AkShare单只股票查询（高效，不需要拉全市场数据）
        try:
            df = ak.stock_individual_info_em(symbol=code)
            if df is not None and len(df) > 0:
                info = dict(zip(df['item'], df['value']))
                name = info.get('股票简称', '')
                market_cap = info.get('总市值', None)
                # 解析市值字符串（如 "123.45亿" 或 "123456.78万"）
                if market_cap is not None and isinstance(market_cap, str):
                    market_cap = market_cap.replace(',', '')
                    try:
                        if '亿' in market_cap:
                            market_cap = float(market_cap.replace('亿', '')) * 1e8
                        elif '万' in market_cap:
                            market_cap = float(market_cap.replace('万', '')) * 1e4
                        else:
                            market_cap = float(market_cap)
                    except Exception:
                        market_cap = None
                return {
                    'code': code,
                    'name': name,
                    'market_cap': market_cap
                }
        except Exception as e:
            logger.debug(f"AkShare get_stock_info 失败 {code}: {e}")
        
        # 3. 兜底：返回空名称（不中断流程）
        logger.warning(f"未能获取股票信息: {code}")
        return {
            'code': code,
            'name': '',
            'market_cap': None
        }
    
    def get_stock_history(self, code: str, start_date: str, end_date: str, 
                         adjust: str = "qfq") -> pd.DataFrame:
        """
        获取股票历史行情数据
        
        Args:
            code: 股票代码（如 "000001"）
            start_date: 开始日期（格式: "20230101"）
            end_date: 结束日期（格式: "20241231"）
            adjust: 复权类型 ("qfq": 前复权, "hfq": 后复权, "": 不复权)
            
        Returns:
            历史行情DataFrame
        """
        logger.debug(f"Fetching history for {code} from {start_date} to {end_date}")
        
        # 如果主数据源是本地数据库，从SQLite读取
        if self.primary_source == "local_db" and self.local_db_available:
            try:
                result = self._get_history_from_local_db(code, start_date, end_date)
                if result is not None and len(result) > 0:
                    logger.debug(f"✓ Got {len(result)} days of history for {code} (local DB)")
                    return result
                else:
                    logger.warning(f"No history data in local DB for {code}")
            except Exception as e:
                logger.error(f"Failed to fetch history from local DB for {code}: {e}")
        
        # 否则使用AkShare/Tushare
        def _get_history_ak():
            """使用AkShare获取历史行情"""
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=adjust
            )
            return df
        
        def _get_history_ts():
            """使用Tushare获取历史行情"""
            if self.ts_pro is None:
                raise Exception("Tushare not initialized")
            
            # 转换股票代码格式 (000001 -> 000001.SZ)
            ts_code = self._convert_code_to_ts_format(code)
            
            df = self.ts_pro.daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )
            
            # 转换为AkShare格式（中文字段名）
            df = self._convert_ts_to_ak_format(df, "history")
            
            return df
        
        try:
            result = self.fetch_with_fallback(_get_history_ak, _get_history_ts)
            
            if result is None or len(result) == 0:
                logger.warning(f"No history data for {code}")
                return None
            
            logger.debug(f"✓ Got {len(result)} days of history for {code}")
            return result
        except Exception as e:
            logger.error(f"Failed to fetch history for {code}: {e}")
            return None
    
    def _get_history_from_local_db(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """从本地数据库获取历史行情"""
        try:
            # 转换代码格式 (000001 -> 000001.SZ 或 600000.SH)
            ts_code = self._convert_code_to_ts_format(code)
            
            conn = sqlite3.connect(self.local_db_path)
            
            query = """
            SELECT trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount
            FROM daily
            WHERE ts_code = ? AND trade_date >= ? AND trade_date <= ?
            ORDER BY trade_date
            """
            
            df = pd.read_sql_query(query, conn, params=(ts_code, start_date, end_date))
            conn.close()
            
            if df is not None and len(df) > 0:
                # 重命名列为AkShare格式
                df.columns = ['日期', '开盘', '最高', '最低', '收盘', '前收盘', '涨跌额', '涨跌幅', '成交量', '成交额']
                return df
            
            return None
        except Exception as e:
            logger.error(f"Error reading history from local DB: {e}")
            raise
    
    def _get_financial_from_local_db(self, code: str) -> Optional[pd.DataFrame]:
        """
        从本地数据库 fina_indicator 表读取财务指标
        
        Args:
            code: 股票代码（如 "000001"）
            
        Returns:
            财务指标DataFrame（最新一条数据）
        """
        try:
            # 转换代码格式
            ts_code = self._convert_code_to_ts_format(code)
            
            conn = sqlite3.connect(self.local_db_path)
            
            # 从 fina_indicator 表读取最新的财务指标
            # 注意：fina_indicator 表中没有 revenue_yoy (营业收入同比增长率) 字段
            query = """
            SELECT 
                end_date as '截止日期',
                ts_code as '代码',
                roe as '净资产收益率',
                roa as '总资产报酬率',
                roic as '投入资本回报率',
                netprofit_yoy as '净利润同比增长率',
                op_yoy as '营业利润同比增长率',
                debt_to_assets as '资产负债率'
            FROM fina_indicator
            WHERE ts_code = ?
            ORDER BY end_date DESC
            LIMIT 1
            """
            
            df = pd.read_sql_query(query, conn, params=(ts_code,))
            conn.close()
            
            if df is not None and len(df) > 0:
                logger.debug(f"✓ Got financial data for {code} from local DB")
                return df
            else:
                logger.warning(f"No financial data in local DB for {code}")
                return None
                
        except Exception as e:
            logger.error(f"Error reading financial data from local DB for {code}: {e}")
            return None
    
    def _get_valuation_from_local_db(self, code: str) -> Optional[pd.DataFrame]:
        """
        从本地数据库 daily_basic 表读取估值指标
        
        Args:
            code: 股票代码（如 "000001"）
            
        Returns:
            估值指标DataFrame（最新一条数据）
        """
        try:
            # 转换代码格式
            ts_code = self._convert_code_to_ts_format(code)
            
            conn = sqlite3.connect(self.local_db_path)
            
            # 从 daily_basic 表读取最新的估值指标
            query = """
            SELECT 
                trade_date as '日期',
                ts_code as '代码',
                pe as '市盈率',
                pb as '市净率',
                ps as '市销率',
                ps_ttm as '市销率(TTM)',
                total_mv as '总市值',
                circ_mv as '流通市值'
            FROM daily_basic
            WHERE ts_code = ?
            ORDER BY trade_date DESC
            LIMIT 1
            """
            
            df = pd.read_sql_query(query, conn, params=(ts_code,))
            conn.close()
            
            if df is not None and len(df) > 0:
                # 注意：Tushare daily_basic 表中的 total_mv 和 circ_mv 单位是万元
                # 转换为亿元（除以 10000）
                if '总市值' in df.columns:
                    df['总市值'] = df['总市值'] / 10000
                if '流通市值' in df.columns:
                    df['流通市值'] = df['流通市值'] / 10000
                
                logger.debug(f"✓ Got valuation data for {code} from local DB")
                return df
            else:
                logger.warning(f"No valuation data in local DB for {code}")
                return None
                
        except Exception as e:
            logger.error(f"Error reading valuation data from local DB for {code}: {e}")
            return None
    
    def get_valuation_data(self, code: str) -> Optional[pd.DataFrame]:
        """
        获取股票估值指标（PE、PB、PS等）
        优先从本地数据库读取，失败后用AkShare/Tushare
        
        Args:
            code: 股票代码
            
        Returns:
            估值指标DataFrame
        """
        logger.debug(f"Fetching valuation data for {code}")
        
        # 优先从本地数据库读取
        if self.primary_source == "local_db" and self.local_db_available:
            try:
                result = self._get_valuation_from_local_db(code)
                if result is not None and len(result) > 0:
                    logger.debug(f"✓ Got valuation data for {code} (local DB)")
                    return result
                else:
                    logger.warning(f"No valuation data in local DB for {code}, trying fallback...")
            except Exception as e:
                logger.error(f"Failed to fetch valuation from local DB for {code}: {e}")
        
        # 否则使用AkShare/Tushare
        def _get_valuation_ak():
            """使用AkShare获取估值指标"""
            df = ak.stock_zh_valuation_comparison_em(symbol=code)
            return df
        
        def _get_valuation_ts():
            """使用Tushare获取估值指标"""
            if self.ts_pro is None:
                raise Exception("Tushare not initialized")
            
            # 转换股票代码格式
            ts_code = self._convert_code_to_ts_format(code)
            
            # 获取最近一年的估值数据
            from datetime import datetime, timedelta
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
            
            # Tushare daily_basic 接口包含估值指标
            df = self.ts_pro.daily_basic(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )
            
            if df is not None and len(df) > 0:
                # 转换为AkShare格式（保持兼容性）
                df = self._convert_ts_to_ak_format(df, "valuation")
                # 选取最新的一天数据
                df = df.head(1)
            
            return df
        
        try:
            result = self.fetch_with_fallback(_get_valuation_ak, _get_valuation_ts)
            
            if result is None or len(result) == 0:
                logger.warning(f"No valuation data for {code}")
                return None
            
            logger.debug(f"✓ Got valuation data for {code}")
            return result
        except Exception as e:
            logger.error(f"Failed to fetch valuation for {code}: {e}")
            return None
    
    def get_financial_data(self, code: str) -> Optional[pd.DataFrame]:
        """
        获取股票财务指标数据
        优先从本地数据库读取，失败后用AkShare/Tushare
        
        Args:
            code: 股票代码
            
        Returns:
            财务指标DataFrame
        """
        logger.debug(f"Fetching financial data for {code}")
        
        # 优先从本地数据库读取
        if self.primary_source == "local_db" and self.local_db_available:
            try:
                result = self._get_financial_from_local_db(code)
                if result is not None and len(result) > 0:
                    logger.debug(f"✓ Got financial data for {code} (local DB)")
                    return result
                else:
                    logger.warning(f"No financial data in local DB for {code}, trying fallback...")
            except Exception as e:
                logger.error(f"Failed to fetch financial from local DB for {code}: {e}")
        
        # 否则使用AkShare/Tushare
        def _get_financial_ak():
            """使用AkShare获取财务指标"""
            df = ak.stock_financial_analysis_indicator(symbol=code)
            return df
        
        def _get_financial_ts():
            """使用Tushare获取财务指标"""
            if self.ts_pro is None:
                raise Exception("Tushare not initialized")
            
            # 转换股票代码格式
            ts_code = self._convert_code_to_ts_format(code)
            
            # 获取最近一年的财务指标（不指定period，默认返回季度数据）
            from datetime import datetime, timedelta
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
            
            try:
                # Tushare fina_indicator 接口（不指定period参数）
                df = self.ts_pro.fina_indicator(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date
                )
                
                # 转换为AkShare格式
                if df is not None and len(df) > 0:
                    # 选取最新的一条数据
                    df = df.sort_values('end_date', ascending=False).head(1)
                    df = self._convert_ts_to_ak_format(df, "financial")
                
                return df
            except Exception as e:
                logger.warning(f"Tushare financial API failed: {e}")
                raise Exception(f"Failed to fetch financial data from Tushare: {e}")
        
        try:
            result = self.fetch_with_fallback(_get_financial_ak, _get_financial_ts)
            
            if result is None or len(result) == 0:
                logger.warning(f"No financial data for {code}")
                return None
            
            logger.debug(f"✓ Got financial data for {code}")
            return result
        except Exception as e:
            logger.error(f"Failed to fetch financial data for {code}: {e}")
            return None
    
    def get_money_flow_data(self, code: str) -> Optional[pd.DataFrame]:
        """
        获取股票资金流数据
        
        Args:
            code: 股票代码
            
        Returns:
            资金流指标DataFrame
        """
        logger.debug(f"Fetching money flow data for {code}")
        
        def _get_money_flow_ak():
            """使用AkShare获取资金流数据"""
            # 使用个股资金流排名接口（可以获取最新一天的资金流数据）
            try:
                # 获取沪深A股资金流排名（最新一天）
                df = ak.stock_individual_fund_flow_rank(indicator="今日")
                
                if df is not None and len(df) > 0:
                    # 筛选出目标股票
                    # AkShare 返回格式: 代码, 名称, 最新价, 涨跌幅, 涨跌额, 成交量, 成交额, 超大单净流入, 大单净流入, 中单净流入, 小单净流入, 净流入, ...
                    # 注意：列名可能不同，需要根据实际返回结果调整
                    
                    # 尝试匹配股票代码（可能有不同格式）
                    code_match = None
                    for col in ['代码', 'code', '股票代码']:
                        if col in df.columns:
                            # 尝试不同格式匹配
                            match_rows = df[df[col].astype(str).str.contains(code)]
                            if len(match_rows) > 0:
                                code_match = col
                                df = match_rows
                                break
                    
                    if code_match is not None and len(df) > 0:
                        logger.debug(f"✓ Got money flow data for {code}")
                        return df.iloc[:1]  # 返回第一行
                    else:
                        logger.warning(f"Stock {code} not found in money flow ranking")
                        return None
                else:
                    logger.warning(f"No money flow data available")
                    return None
            except Exception as e:
                logger.error(f"Error fetching money flow from AkShare: {e}")
                raise
        
        def _get_money_flow_ts():
            """使用Tushare获取资金流数据"""
            if self.ts_pro is None:
                raise Exception("Tushare not initialized")
            
            # 转换股票代码格式
            ts_code = self._convert_code_to_ts_format(code)
            
            # 获取最近一天的资金流数据
            from datetime import datetime, timedelta
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
            
            try:
                # Tushare moneyflow 接口
                df = self.ts_pro.moneyflow(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if df is not None and len(df) > 0:
                    # 选取最新的一天数据
                    df = df.sort_values('trade_date', ascending=False).head(1)
                    
                    # 转换为AkShare格式（中文字段名）
                    column_mapping = {
                        'trade_date': '日期',
                        'ts_code': '代码',
                        'buy_lg_amount': '超大单净流入',
                        'buy_elg_amount': '大单净流入',
                        'buy_md_amount': '中单净流入',
                        'buy_sm_amount': '小单净流入',
                        'net_mf_amount': '净流入'
                    }
                    df = df.rename(columns=column_mapping)
                    
                    logger.debug(f"✓ Got money flow data for {code} from Tushare")
                    return df
                else:
                    logger.warning(f"No money flow data for {code} from Tushare")
                    return None
            except Exception as e:
                logger.warning(f"Tushare moneyflow API failed: {e}")
                raise Exception(f"Failed to fetch money flow data from Tushare: {e}")
        
        try:
            result = self.fetch_with_fallback(_get_money_flow_ak, _get_money_flow_ts)
            
            if result is None or len(result) == 0:
                logger.warning(f"No money flow data for {code}")
                return None
            
            logger.debug(f"✓ Got money flow data for {code}")
            return result
        except Exception as e:
            logger.error(f"Failed to fetch money flow data for {code}: {e}")
            return None
    
    def _convert_code_to_ts_format(self, code: str) -> str:
        """
        将股票代码转换为Tushare格式
        
        Args:
            code: 股票代码（如 "000001" 或 "600000" 或 "000001.SZ"）
            
        Returns:
            Tushare格式代码（如 "000001.SZ" 或 "600000.SH"）
        """
        # 如果已经包含后缀，直接返回
        if '.' in code:
            return code
        
        # 简单判断：6开头为上海，0或3开头为深圳
        if code.startswith('6'):
            return f"{code}.SH"
        else:
            return f"{code}.SZ"
    
    def _convert_ts_to_ak_format(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        将Tushare数据转换为AkShare格式（中文字段名）
        
        Args:
            df: Tushare格式的DataFrame
            data_type: 数据类型 ("history", "valuation", "financial")
            
        Returns:
            AkShare格式的DataFrame
        """
        if df is None or len(df) == 0:
            return df
        
        df_ak = df.copy()
        
        if data_type == "history":
            # 历史行情：英文字段 → 中文字段
            column_mapping = {
                'trade_date': '日期',
                'ts_code': '代码',
                'open': '开盘',
                'high': '最高',
                'low': '最低',
                'close': '收盘',
                'pre_close': '前收盘',
                'change': '涨跌额',
                'pct_chg': '涨跌幅',
                'vol': '成交量',
                'amount': '成交额'
            }
            df_ak = df_ak.rename(columns=column_mapping)
            
        elif data_type == "valuation":
            # 估值数据
            column_mapping = {
                'trade_date': '日期',
                'ts_code': '代码',
                'pe': '市盈率',
                'pb': '市净率',
                'ps': '市销率',
                'total_mv': '总市值',
                'circ_mv': '流通市值'
            }
            df_ak = df_ak.rename(columns=column_mapping)
            
        elif data_type == "financial":
            # 财务数据 - 根据实际返回的列进行映射
            column_mapping = {
                'end_date': '截止日期',
                'ts_code': '代码',
                'roe': '净资产收益率',
                'roa': '总资产报酬率',
                'netprofit_yoy': '净利润同比增长率',
                'revenue_yoy': '营业收入同比增长率'
            }
            df_ak = df_ak.rename(columns=column_mapping)
        
        return df_ak
    
    def batch_fetch_histories(self, codes: list, start_date: str, end_date: str,
                            max_workers: int = 5, sleep_interval: float = 0.1) -> dict:
        """
        批量获取历史行情数据
        
        Args:
            codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            max_workers: 最大线程数
            sleep_interval: 请求间隔（秒）
            
        Returns:
            {code: DataFrame} 字典
        """
        logger.info(f"Batch fetching histories for {len(codes)} stocks...")
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def fetch_single(code):
            time.sleep(sleep_interval)  # 避免请求过快
            return code, self.get_stock_history(code, start_date, end_date)
        
        results = {}
        failed = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_single, code): code for code in codes}
            
            for future in as_completed(futures):
                code = futures[future]
                try:
                    _, data = future.result()
                    if data is not None:
                        results[code] = data
                    else:
                        failed.append(code)
                except Exception as e:
                    logger.error(f"Failed to fetch {code}: {e}")
                    failed.append(code)
        
        logger.info(f"✓ Batch fetch completed: {len(results)} success, {len(failed)} failed")
        
        if len(failed) > 0:
            logger.warning(f"Failed codes: {failed[:10]}")  # 只显示前10个
        
        return results


if __name__ == "__main__":
    # 测试代码
    from loguru import logger
    
    # 初始化日志
    logger.add("data_fetcher_test.log", rotation="500 MB")
    
    # 创建数据获取器
    fetcher = DataFetcher(use_tushare=False)
    
    # 测试1: 获取沪深300成分股
    print("\n" + "="*50)
    print("Test 1: Get HS300 components")
    print("="*50)
    hs300 = fetcher.get_hs300_components()
    print(f"✓ Got {len(hs300)} components")
    print(hs300.head())
    
    # 测试2: 获取单只股票历史行情
    print("\n" + "="*50)
    print("Test 2: Get stock history (000001)")
    print("="*50)
    history = fetcher.get_stock_history("000001", "20240101", "20240301")
    if history is not None:
        print(f"✓ Got {len(history)} days of history")
        print(history.head())
    
    # 测试3: 获取估值指标
    print("\n" + "="*50)
    print("Test 3: Get valuation data (000001)")
    print("="*50)
    valuation = fetcher.get_valuation_data("000001")
    if valuation is not None:
        print(f"✓ Got valuation data")
        print(valuation.head())
    
    # 测试4: 获取财务指标
    print("\n" + "="*50)
    print("Test 4: Get financial data (000001)")
    print("="*50)
    financial = fetcher.get_financial_data("000001")
    if financial is not None:
        print(f"✓ Got financial data")
        print(financial.head())
    
    print("\n" + "="*50)
    print("All tests completed!")
    print("="*50)


class SQLiteDataFetcher:
    """从本地 SQLite 数据库读取股票数据"""
    
    def __init__(self, db_path: str = "D:/tu-shareData/astock_daily.db"):
        """
        初始化 SQLite 数据获取器
        
        Args:
            db_path: SQLite 数据库文件路径
        """
        self.db_path = db_path
        logger.info(f"SQLiteDataFetcher initialized with db_path: {db_path}")
        
        # 测试数据库连接（创建临时连接测试）
        try:
            test_conn = sqlite3.connect(self.db_path)
            test_conn.close()
            logger.info("✓ SQLite database connection test successful")
        except Exception as e:
            logger.error(f"Failed to connect to SQLite database: {e}")
            raise
    
    def _get_connection(self):
        """创建并返回一个新的数据库连接"""
        return sqlite3.connect(self.db_path)
    
    def get_hs300_components(self) -> pd.DataFrame:
        """
        获取沪深300成分股
        注意：如果数据库中不包含指数成分股信息，则返回所有股票
        """
        logger.info("Fetching stock list from SQLite database...")
        
        try:
            conn = self._get_connection()
            
            # 从 stock_basic 表获取所有股票
            query = """
                SELECT ts_code, symbol as code, name
                FROM stock_basic
                WHERE ts_code LIKE '%.SZ' OR ts_code LIKE '%.SH'
                ORDER BY symbol
            """
            
            df = pd.read_sql_query(query, conn)
            
            # 移除 .SZ 和 .SH 后缀
            df['code'] = df['code'].str.replace(r'\.(SH|SZ)$', '', regex=True)
            
            # 如果数据库中没有行业信息，添加空列
            if 'industry' not in df.columns:
                df['industry'] = ''
            
            logger.info(f"✓ Got {len(df)} stocks from database")
            return df[['code', 'name', 'industry']]
            
        except Exception as e:
            logger.error(f"Failed to fetch stock list from SQLite: {e}")
            raise
    
    def get_stock_info(self, code: str) -> dict:
        """
        获取股票基本信息（名称、市值等）
        
        Args:
            code: 股票代码（如 "000001"）
            
        Returns:
            包含股票信息的字典
        """
        logger.debug(f"Fetching stock info for {code} from SQLite")
        
        try:
            conn = self._get_connection()
            
            # 转换代码格式（000001 -> 000001.SZ 或 600000.SH）
            ts_code = self._convert_code_to_ts_format(code)
            
            # 从 stock_basic 获取股票名称
            query = f"""
                SELECT name
                FROM stock_basic
                WHERE ts_code = '{ts_code}'
            """
            
            cursor = conn.cursor()
            cursor.execute(query)
            result = cursor.fetchone()
            
            name = result[0] if result else ''
            
            # 获取最新市值为最近交易日的 total_mv
            query = f"""
                SELECT total_mv
                FROM daily_basic
                WHERE ts_code = '{ts_code}'
                ORDER BY trade_date DESC
                LIMIT 1
            """
            
            cursor.execute(query)
            result = cursor.fetchone()
            
            market_cap = result[0] if result else None
            
            # 注意：Tushare daily_basic 表中的 total_mv 单位是万元
            # 转换为亿元（除以 10000）
            if market_cap is not None:
                market_cap = float(market_cap) / 10000
            
            return {
                'code': code,
                'name': name,
                'market_cap': market_cap
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch stock info for {code} from SQLite: {e}")
            return {
                'code': code,
                'name': '',
                'market_cap': None
            }
    
    def get_stock_history(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取股票历史行情数据
        
        Args:
            code: 股票代码（如 "000001"）
            start_date: 开始日期（格式: "20230101"）
            end_date: 结束日期（格式: "20241231"）
            
        Returns:
            历史行情DataFrame
        """
        logger.debug(f"Fetching history for {code} from {start_date} to {end_date}")
        
        try:
            conn = self._get_connection()
            
            # 转换代码格式
            ts_code = self._convert_code_to_ts_format(code)
            
            # 从 daily 表获取历史行情
            query = f"""
                SELECT 
                    trade_date as '日期',
                    open as '开盘',
                    high as '最高',
                    low as '最低',
                    close as '收盘',
                    pre_close as '前收盘',
                    change as '涨跌额',
                    pct_chg as '涨跌幅',
                    vol as '成交量',
                    amount as '成交额'
                FROM daily
                WHERE ts_code = '{ts_code}'
                  AND trade_date BETWEEN '{start_date}' AND '{end_date}'
                ORDER BY trade_date
            """
            
            df = pd.read_sql_query(query, conn)
            
            if len(df) == 0:
                logger.warning(f"No history data for {code}")
                return None
            
            logger.debug(f"✓ Got {len(df)} days of history for {code}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch history for {code} from SQLite: {e}")
            return None
    
    def get_valuation_data(self, code: str) -> pd.DataFrame:
        """
        获取股票估值指标（PE、PB、PS等）
        
        Args:
            code: 股票代码
            
        Returns:
            估值指标DataFrame
        """
        logger.debug(f"Fetching valuation data for {code} from SQLite")
        
        try:
            conn = self._get_connection()
            
            # 转换代码格式
            ts_code = self._convert_code_to_ts_format(code)
            
            # 从 daily_basic 表获取最新估值指标
            query = f"""
                SELECT 
                    trade_date as '日期',
                    ts_code as '代码',
                    pe as '市盈率',
                    pb as '市净率',
                    ps as '市销率',
                    ps_ttm as '市销率(TTM)',
                    total_mv as '总市值',
                    circ_mv as '流通市值'
                FROM daily_basic
                WHERE ts_code = '{ts_code}'
                ORDER BY trade_date DESC
                LIMIT 1
            """
            
            df = pd.read_sql_query(query, conn)
            
            if len(df) == 0:
                logger.warning(f"No valuation data for {code}")
                return None
            
            # 注意：Tushare daily_basic 表中的 total_mv 和 circ_mv 单位是万元
            # 转换为亿元（除以 10000）
            if '总市值' in df.columns:
                df['总市值'] = df['总市值'] / 10000
            if '流通市值' in df.columns:
                df['流通市值'] = df['流通市值'] / 10000
            
            # 添加股票代码列（简单格式）
            df['code'] = code
            
            logger.debug(f"✓ Got valuation data for {code}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch valuation for {code} from SQLite: {e}")
            return None
    
    def get_financial_data(self, code: str) -> pd.DataFrame:
        """
        获取股票财务指标数据
        
        Args:
            code: 股票代码
            
        Returns:
            财务指标DataFrame
        """
        logger.debug(f"Fetching financial data for {code} from SQLite")
        
        # 注意：当前数据库中可能没有详细的财务指标表
        # 如果有 fina_indicator 表，可以在这里查询
        # 如果没有，返回 None 或使用 daily_basic 中的数据
        
        logger.warning(f"Financial data not available in SQLite database for {code}")
        return None
    
    def get_latest_price(self, code: str) -> float:
        """
        获取股票最新收盘价
        
        Args:
            code: 股票代码
            
        Returns:
            最新收盘价
        """
        logger.debug(f"Fetching latest price for {code} from SQLite")
        
        try:
            conn = self._get_connection()
            
            # 转换代码格式
            ts_code = self._convert_code_to_ts_format(code)
            
            # 从 daily 表获取最新收盘价
            query = f"""
                SELECT close
                FROM daily
                WHERE ts_code = '{ts_code}'
                ORDER BY trade_date DESC
                LIMIT 1
            """
            
            cursor = conn.cursor()
            cursor.execute(query)
            result = cursor.fetchone()
            
            if result:
                return float(result[0])
            else:
                logger.warning(f"No price data for {code}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to fetch latest price for {code} from SQLite: {e}")
            return None
    
    def _convert_code_to_ts_format(self, code: str) -> str:
        """
        将股票代码转换为Tushare格式
        
        Args:
            code: 股票代码（如 "000001" 或 "600000" 或 "000001.SZ"）
            
        Returns:
            Tushare格式代码（如 "000001.SZ" 或 "600000.SH"）
        """
        # 如果已经包含后缀，直接返回
        if '.' in code:
            return code
        
        # 简单判断：6开头为上海，0或3开头为深圳
        if code.startswith('6'):
            return f"{code}.SH"
        else:
            return f"{code}.SZ"
    
    def close(self):
        """关闭数据库连接"""
        if self.conn is not None:
            self.conn.close()
            self.conn = None
            logger.info("SQLite connection closed")

