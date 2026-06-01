"""
数据获取模块 - DataFetcher Class
支持 AkShare、Tushare 和本地 SQLite 数据库
"""

import akshare as ak
import pandas as pd
import numpy as np
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
        
        # 初始化速率限制器（Tushare限制：付费账户约50-100次/分钟，这里设保守值）
        self.rate_limiter = RateLimiter(calls_per_minute=50)
        
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
    
    def _convert_code_to_ts_format(self, code: str) -> str:
        """
        将简单代码格式转换为Tushare格式
        000001 -> 000001.SZ
        600000 -> 600000.SH
        """
        code = str(code).strip()
        
        # 判断交易所
        if code.startswith('6'):
            # 上海交易所
            return f"{code}.SH"
        elif code.startswith(('0', '3')):
            # 深圳交易所
            return f"{code}.SZ"
        elif code.startswith('8') or code.startswith('4'):
            # 新三板
            return f"{code}.BJ"
        else:
            # 默认深圳
            return f"{code}.SZ"
    
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
        # 尝试使用主数据源（最多5次）
        for attempt in range(5):
            try:
                # 速率限制（仅对Tushare API）
                if primary_name == "Tushare":
                    self.rate_limiter.wait()
                
                logger.debug(f"Trying {primary_name}: {func_primary.__name__} (attempt {attempt+1}/5)")
                result = func_primary(*args, **kwargs)
                time.sleep(0.3)  # 避免请求过快（从0.1增加到0.3）
                return result
            except Exception as e:
                logger.warning(f"{primary_name} failed (attempt {attempt+1}/5): {e}")
                if attempt < 4:
                    # 指数退避：2秒, 4秒, 8秒, 16秒
                    sleep_time = 2 ** (attempt + 1)
                    logger.info(f"Waiting {sleep_time}s before retry...")
                    time.sleep(sleep_time)
                else:
                    # 5次都失败，尝试备用数据源
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
        
        # 根据配置标志决定使用哪些数据源
        try:
            if self.akshare_backup_available and self.tushare_backup_available:
                # 两个数据源都可用，按优先级尝试
                if self.primary_source == "akshare":
                    result = self.fetch_with_fallback(_get_hs300_ak, _get_hs300_ts)
                else:
                    result = self.fetch_with_fallback(_get_hs300_ts, _get_hs300_ak)
            elif self.tushare_backup_available:
                # 只用 Tushare
                logger.info("AkShare backup disabled, using Tushare only for HS300")
                result = _get_hs300_ts()
                if result is None or len(result) == 0:
                    raise Exception("No HS300 components data from Tushare")
            elif self.akshare_backup_available:
                # 只用 AkShare
                logger.info("Tushare backup disabled, using AkShare only for HS300")
                result = _get_hs300_ak()
                if result is None or len(result) == 0:
                    raise Exception("No HS300 components data from AkShare")
            else:
                # 两个数据源都禁用
                logger.error("Both AkShare and Tushare backup disabled, cannot fetch HS300 components")
                raise Exception("No data source available for HS300 components")
            
            logger.info(f"✓ Got {len(result)} HS300 components")
            return result
        except Exception as e:
            logger.error(f"Failed to fetch HS300 components: {e}")
            raise
    
    def get_zz800_components(self) -> pd.DataFrame:
        """
        获取中证800成分股（沪深300 + 中证500）
        指数代码：000906.SH
        """
        logger.info("Fetching CSI 800 (ZZ800) components...")

        def _get_zz800_ak():
            """使用AkShare获取中证800成分股"""
            # 正确代码：000906=中证800，000985=中证全指（全市场≈5000只）
            df = ak.index_stock_cons_csindex(symbol="000906")
            result = df[['成分券代码', '成分券名称']].copy()
            result.columns = ['code', 'name']
            return result

        def _get_zz800_ts():
            """使用Tushare获取中证800成分股"""
            if self.ts_pro is None:
                raise Exception("Tushare not initialized")

            # 中证800 = 沪深300 + 中证500
            df300 = self.ts_pro.index_member(index_code='000300.SH')
            df500 = self.ts_pro.index_member(index_code='000906.SH')

            if (df300 is not None and len(df300) > 0 and
                df500 is not None and len(df500) > 0):
                # 过滤出当前仍在成分股中的股票 (out_date 为 NaN)
                if 'out_date' in df300.columns:
                    current300 = df300[df300['out_date'].isna()]
                else:
                    # 如果没有 out_date 列，假设所有都是当前成分股
                    logger.warning("df300 没有 out_date 列，使用所有成分股")
                    current300 = df300
                
                if 'out_date' in df500.columns:
                    current500 = df500[df500['out_date'].isna()]
                else:
                    logger.warning("df500 没有 out_date 列，使用所有成分股")
                    current500 = df500
                
                combined = pd.concat([current300, current500]).drop_duplicates(subset=['con_code'])
                
                # 验证数量（ZZ800 应该 ≈800 只）
                if len(combined) > 3000:
                    logger.warning(f"ZZ800 成分股数量异常: {len(combined)} 只，可能包含历史数据")
                
                result = combined[['con_code']].copy()
                result['code'] = result['con_code'].str.replace(r'\.(SH|SZ)$', '', regex=True)
                result['name'] = ''
                logger.info(f"Tushare ZZ800: 获取 {len(combined)} 只成分股（过滤后）")
                return result[['code', 'name']]
            raise Exception("No ZZ800 components data available from Tushare")
        
        # 根据配置标志决定使用哪些数据源
        try:
            if self.akshare_backup_available and self.tushare_backup_available:
                # 两个数据源都可用，按优先级尝试
                if self.primary_source == "akshare":
                    result = self.fetch_with_fallback(_get_zz800_ak, _get_zz800_ts)
                else:
                    result = self.fetch_with_fallback(_get_zz800_ts, _get_zz800_ak)
            elif self.tushare_backup_available:
                # 只用 Tushare
                logger.info("AkShare backup disabled, using Tushare only for ZZ800")
                result = _get_zz800_ts()
                if result is None or len(result) == 0:
                    raise Exception("No ZZ800 components data from Tushare")
            elif self.akshare_backup_available:
                # 只用 AkShare
                logger.info("Tushare backup disabled, using AkShare only for ZZ800")
                result = _get_zz800_ak()
                if result is None or len(result) == 0:
                    raise Exception("No ZZ800 components data from AkShare")
            else:
                # 两个数据源都禁用
                logger.error("Both AkShare and Tushare backup disabled, cannot fetch ZZ800 components")
                raise Exception("No data source available for ZZ800 components")
            
            logger.info(f"✓ Got {len(result)} ZZ800 components")
            return result
        except Exception as e:
            logger.error(f"Failed to fetch ZZ800 components: {e}")
            raise

    def get_stock_info(self, code: str) -> dict:
        """
        获取股票基本信息（名称、市值）
        优先从本地数据库读取，失败后用Tushare/AkShare
        
        Args:
            code: 股票代码
            
        Returns:
            包含股票信息的字典（始终返回dict，避免None中断流程）
        """
        logger.debug(f"Fetching stock info for {code}")
        
        # 1. 优先从本地数据库读取（stock_basic表）
        if self.local_db_available:
            try:
                import sqlite3
                ts_code = self._convert_code_to_ts_format(code)
                conn = sqlite3.connect(self.local_db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT name, market_cap 
                    FROM stock_basic 
                    WHERE ts_code = ? 
                    ORDER BY trade_date DESC 
                    LIMIT 1
                """, (ts_code,))
                
                row = cursor.fetchone()
                conn.close()
                
                if row is not None:
                    name = row[0] if row[0] else ''
                    market_cap = float(row[1]) if row[1] else None
                    logger.debug(f"✓ 从本地DB获取股票信息: {code} = {name}")
                    return {
                        'code': code,
                        'name': name,
                        'market_cap': market_cap
                    }
                else:
                    logger.debug(f"本地DB中没有股票信息: {code}")
            except Exception as e:
                logger.debug(f"从本地DB读取股票信息失败: {e}")
        
        # 2. 使用Tushare批量缓存（启动时已批量获取）
        if hasattr(self, '_stock_name_cache') and code in self._stock_name_cache:
            name = self._stock_name_cache[code]
            logger.debug(f"✓ 从Tushare缓存获取股票名称: {code} = {name}")
            return {
                'code': code,
                'name': name,
                'market_cap': None  # 市值从valuation数据获取
            }
        
        # 3. 使用AkShare单只股票查询（高效，不需要拉全市场数据）
        if self.akshare_backup_available:
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
        else:
            logger.debug(f"AkShare backup disabled, skipping get_stock_info for {code}")
        
        # 4. 兜底：返回空名称（不中断流程）
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
        
        # 定义 AkShare / Tushare 内嵌获取函数
        def _get_history_ak():
            """使用 AkShare 获取历史行情"""
            ak_adjust = ""
            if adjust == "qfq":
                ak_adjust = "qfq"
            elif adjust == "hfq":
                ak_adjust = "hfq"
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=ak_adjust
            )
            return df

        def _get_history_ts():
            """使用 Tushare 获取历史行情"""
            if self.ts_pro is None:
                raise Exception("Tushare not initialized")
            ts_code = self._convert_code_to_ts_format(code)
            df = self.ts_pro.daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )
            if df is not None and len(df) > 0:
                df = df.rename(columns={
                    'trade_date': '日期',
                    'open': '开盘',
                    'high': '最高',
                    'low': '最低',
                    'close': '收盘',
                    'vol': '成交量',
                    'amount': '成交额',
                    'pct_chg': '涨跌幅',
                    'change': '涨跌额'
                })
            return df

        # 否则根据配置决定是否尝试AkShare/Tushare
        # 根据配置标志决定使用哪些数据源
        if self.akshare_backup_available and self.tushare_backup_available:
            # 两个数据源都可用，按优先级尝试
            if self.primary_source == "akshare":
                result = self.fetch_with_fallback(_get_history_ak, _get_history_ts)
            else:
                result = self.fetch_with_fallback(_get_history_ts, _get_history_ak)
        elif self.tushare_backup_available:
            # 只用 Tushare
            logger.info(f"AkShare backup disabled, using Tushare only for {code} history")
            result = _get_history_ts()
        elif self.akshare_backup_available:
            # 只用 AkShare
            logger.info(f"Tushare backup disabled, using AkShare only for {code} history")
            result = _get_history_ak()
        else:
            # 两个数据源都禁用
            logger.debug(f"Both AkShare and Tushare backup disabled, skipping for {code}")
            return None
        
        if result is None or len(result) == 0:
            logger.warning(f"No history data for {code}")
            return None
            
        logger.debug(f"✓ Got {len(result)} days of history for {code}")
        return result
    
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
                    # 根据配置决定fallback行为
                    if not self.tushare_backup_available and not self.akshare_backup_available:
                        logger.warning(f"No valuation data in local DB for {code}, and all backups disabled")
                    elif not self.tushare_backup_available:
                        logger.warning(f"No valuation data in local DB for {code}, trying AkShare backup...")
                    elif not self.akshare_backup_available:
                        logger.warning(f"No valuation data in local DB for {code}, trying Tushare backup...")
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
        
        # 根据配置标志决定使用哪些数据源
        try:
            if self.tushare_backup_available:
                # Tushare 可用，优先使用
                if self.akshare_backup_available:
                    # Tushare 失败后用 AkShare 备份
                    result = self._fetch_with_order(_get_valuation_ts, _get_valuation_ak, "Tushare", "AkShare")
                else:
                    # 不允许 AkShare 备份，只用 Tushare
                    try:
                        result = _get_valuation_ts()
                        if result is not None and len(result) > 0:
                            logger.debug(f"✓ Got valuation data for {code} (Tushare only)")
                            return result
                        else:
                            logger.warning(f"No valuation data from Tushare for {code}")
                            return None
                    except Exception as e:
                        logger.error(f"Failed to fetch valuation for {code}: {e}")
                        return None
            elif self.akshare_backup_available:
                # 只用 AkShare
                logger.info("Tushare backup disabled, using AkShare only for valuation")
                result = _get_valuation_ak()
            else:
                # 两个数据源都禁用
                logger.warning(f"Both Tushare and AkShare backup disabled, skipping valuation for {code}")
                return None
            
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
                    # 根据配置决定fallback行为
                    if not self.tushare_backup_available and not self.akshare_backup_available:
                        logger.warning(f"No financial data in local DB for {code}, and all backups disabled")
                    elif not self.tushare_backup_available:
                        logger.warning(f"No financial data in local DB for {code}, trying AkShare backup...")
                    elif not self.akshare_backup_available:
                        logger.warning(f"No financial data in local DB for {code}, trying Tushare backup...")
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
        
        # 根据配置标志决定使用哪些数据源
        try:
            if self.tushare_backup_available:
                # Tushare 可用，优先使用
                if self.akshare_backup_available:
                    # Tushare 失败后用 AkShare 备份
                    result = self._fetch_with_order(_get_financial_ts, _get_financial_ak, "Tushare", "AkShare")
                else:
                    # 不允许 AkShare 备份，只用 Tushare
                    try:
                        result = _get_financial_ts()
                        if result is not None and len(result) > 0:
                            logger.debug(f"✓ Got financial data for {code} (Tushare only)")
                            return result
                        else:
                            logger.warning(f"No financial data from Tushare for {code}")
                            return None
                    except Exception as e:
                        logger.error(f"Failed to fetch financial for {code}: {e}")
                        return None
            elif self.akshare_backup_available:
                # 只用 AkShare
                logger.info("Tushare backup disabled, using AkShare only for financial")
                result = _get_financial_ak()
            else:
                # 两个数据源都禁用
                logger.warning(f"Both Tushare and AkShare backup disabled, skipping financial for {code}")
                return None
            
            if result is None or len(result) == 0:
                logger.warning(f"No financial data for {code}")
                return None
            
            logger.debug(f"✓ Got financial data for {code}")
            return result
        except Exception as e:
            logger.error(f"Failed to fetch financial data for {code}: {e}")
            return None
    
    def get_industry_momentum_factor(self, code: str, trade_date: str) -> dict:
        """
        获取股票在指定交易日的行业动量因子
        
        Args:
            code: 股票代码（如 "000001"）
            trade_date: 交易日期（格式: "20230101"）
            
        Returns:
            字典 {"industry_momentum": value, "industry_momentum_z": value}
            如果未找到，返回 {"industry_momentum": np.nan, "industry_momentum_z": np.nan}
        """
        logger.debug(f"Fetching industry momentum factor for {code} on {trade_date}")
        
        if not self.local_db_available:
            logger.warning(f"Local DB not available, cannot fetch industry momentum for {code}")
            return {"industry_momentum": np.nan, "industry_momentum_z": np.nan}
        
        try:
            import sqlite3
            conn = sqlite3.connect(self.local_db_path)
            cursor = conn.cursor()
            
            # 查询 industry_momentum 表
            cursor.execute("""
                SELECT industry_momentum, industry_momentum_z
                FROM industry_momentum
                WHERE ts_code = ? AND trade_date = ?
            """, (code, trade_date))
            
            row = cursor.fetchone()
            conn.close()
            
            if row is not None:
                result = {
                    "industry_momentum": float(row[0]) if row[0] is not None else np.nan,
                    "industry_momentum_z": float(row[1]) if row[1] is not None else np.nan
                }
                logger.debug(f"✓ Got industry momentum for {code}: {result}")
                return result
            else:
                logger.debug(f"No industry momentum data for {code} on {trade_date}")
            return {"industry_momentum": np.nan, "industry_momentum_z": np.nan}
                
        except Exception as e:
            logger.error(f"Error fetching industry momentum for {code}: {e}")
            return {"industry_momentum": np.nan, "industry_momentum_z": np.nan}
    
    def _convert_ts_to_ak_format(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        将Tushare格式的数据转换为AkShare格式
        
        Args:
            df: Tushare格式的DataFrame
            data_type: 数据类型 ("valuation" 或 "financial")
            
        Returns:
            AkShare格式的DataFrame
        """
        if df is None or len(df) == 0:
            return df
        
        logger.debug(f"Converting Tushare format to AkShare format (type: {data_type})")
        
        if data_type == "valuation":
            # Tushare估值数据 -> AkShare格式
            # Tushare列名: ts_code, trade_date, pe, pb, ps, dv_ratio, etc.
            # AkShare列名: 代码, 名称, 市盈率, 市净率, 市销率, 股息率, etc.
            
            # 创建新的DataFrame
            ak_df = pd.DataFrame()
            ak_df["代码"] = df["ts_code"].apply(lambda x: x.split('.')[0] if '.' in str(x) else x)
            
            # 获取股票名称
            try:
                from src.data_fetcher import DataFetcher
                ak_df["名称"] = ak_df["代码"].apply(lambda x: self._get_stock_name(x))
            except:
                ak_df["名称"] = ""
            
            # 映射估值指标
            if "pe" in df.columns:
                ak_df["市盈率"] = df["pe"]
            if "pb" in df.columns:
                ak_df["市净率"] = df["pb"]
            if "ps" in df.columns:
                ak_df["市销率"] = df["ps"]
            if "dv_ratio" in df.columns:
                ak_df["股息率"] = df["dv_ratio"]
            if "total_mv" in df.columns:
                ak_df["总市值"] = df["total_mv"]
            if "circ_mv" in df.columns:
                ak_df["流通市值"] = df["circ_mv"]
            
            return ak_df
            
        elif data_type == "financial":
            # Tushare财务数据 -> AkShare格式
            # Tushare列名: ts_code, end_date, eps, roe, debt_to_assets, etc.
            # AkShare列名: 代码, 名称, 每股收益, 净资产收益率, 资产负债率, etc.
            
            # 创建新的DataFrame
            ak_df = pd.DataFrame()
            ak_df["代码"] = df["ts_code"].apply(lambda x: x.split('.')[0] if '.' in str(x) else x)
            
            # 获取股票名称
            try:
                ak_df["名称"] = ak_df["代码"].apply(lambda x: self._get_stock_name(x))
            except:
                ak_df["名称"] = ""
            
            # 映射财务指标
            if "eps" in df.columns:
                ak_df["每股收益"] = df["eps"]
            if "roe" in df.columns:
                ak_df["净资产收益率"] = df["roe"]
            if "debt_to_assets" in df.columns:
                ak_df["资产负债率"] = df["debt_to_assets"]
            if "assets_to_eqt" in df.columns:
                ak_df["资产周转率"] = df["assets_to_eqt"]
            if "ocf_to_or" in df.columns:
                ak_df["经营现金流收益率"] = df["ocf_to_or"]
            if "grossprofit_margin" in df.columns:
                ak_df["毛利率"] = df["grossprofit_margin"]
            if "netprofit_margin" in df.columns:
                ak_df["净利润率"] = df["netprofit_margin"]
            
            return ak_df
        
        else:
            logger.warning(f"Unknown data_type: {data_type}, returning original DataFrame")
            return df
    
    def _get_stock_name(self, code: str) -> str:
        """
        获取股票名称（从缓存或数据库）
        
        Args:
            code: 股票代码
            
        Returns:
            股票名称
        """
        # 先从缓存查找
        if hasattr(self, '_stock_name_cache') and code in self._stock_name_cache:
            return self._stock_name_cache[code]
        
        # 从本地数据库查找
        if self.local_db_available:
            try:
                import sqlite3
                conn = sqlite3.connect(self.local_db_path)
                cursor = conn.cursor()
                
                # 查询股票名称（从valuation表或stock_basic表）
                cursor.execute("""
                    SELECT name FROM stock_basic WHERE ts_code = ? LIMIT 1
                """, (code,))
                
                row = cursor.fetchone()
                conn.close()
                
                if row is not None:
                    name = row[0]
                    # 缓存结果
                    if not hasattr(self, '_stock_name_cache'):
                        self._stock_name_cache = {}
                    self._stock_name_cache[code] = name
                    return name
            except Exception as e:
                logger.debug(f"Error fetching stock name for {code}: {e}")
        
        # 默认返回空字符串
        return ""
