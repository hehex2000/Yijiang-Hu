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
                # 不调用 ts.set_token()（会写入 ~/tk.csv，沙箱环境会权限拒绝）
                # 直接传 token 给 pro_api()，避免文件访问
                self.ts_pro = ts.pro_api(token=tushare_token)
                self.use_tushare = True
                logger.info("Tushare initialized successfully (token passed to pro_api)")
            except Exception as e:
                logger.warning(f"Failed to initialize Tushare: {e}")
        
        # 确定备用数据源是否可用
        self.akshare_backup_available = use_akshare_backup and self.use_akshare
        self.tushare_backup_available = use_tushare_backup and self.use_tushare
        
        # 初始化速率限制器（Tushare限制：付费账户150次/分钟，这里设150）
        self.rate_limiter = RateLimiter(calls_per_minute=150)
        
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
        
        elif not self._stock_name_cache and self.use_tushare and self.ts_pro is not None:
            try:
                # 速率限制（Tushare API）
                self.rate_limiter.wait()
                
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
        已带后缀(如 000001.SZ)则原样返回，避免重复拼接成 000001.SZ.SZ
        """
        code = str(code).strip()
        # 已带交易所后缀，直接返回
        if code[-3:] in (".SZ", ".SH", ".BJ", ".HK"):
            return code
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
    
    def _get_hs300_from_local_db(self, date: str) -> pd.DataFrame:
        """
        从本地数据库获取指定日期的HS300成分股
        
        Args:
            date: 交易日期（格式: "20230101"）
            
        Returns:
            包含代码、名称的DataFrame
        """
        if not self.local_db_available:
            return None
        
        try:
            import sqlite3
            conn = sqlite3.connect(self.local_db_path)
            cursor = conn.cursor()
            
            # 1. 找到小于等于目标日期的最大trade_date
            cursor.execute("""
                SELECT MAX(trade_date) 
                FROM index_constituent 
                WHERE index_code = '000300.SH' AND trade_date <= ?
            """, (date,))
            
            closest_date = cursor.fetchone()[0]
            
            if closest_date is None:
                logger.warning(f"No HS300 data found for date <= {date}")
                conn.close()
                return None
            
            logger.info(f"Found HS300 constituents for date: {closest_date} (requested: {date})")
            
            # 2. 获取该日期的所有成分股
            cursor.execute("""
                SELECT ts_code, weight 
                FROM index_constituent 
                WHERE index_code = '000300.SH' AND trade_date = ?
            """, (closest_date,))
            
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                return None
            
            # 3. 转换为DataFrame格式
            import pandas as pd
            result = pd.DataFrame(rows, columns=['ts_code', 'weight'])
            
            # 移除 .SH 或 .SZ 后缀，统一为简单格式
            result['code'] = result['ts_code'].str.replace(r'\.(SH|SZ)$', '', regex=True)
            
            # 获取股票名称（从缓存或数据库）
            result['name'] = result['code'].apply(lambda x: self._stock_name_cache.get(x, ''))
            
            logger.info(f"✓ Got {len(result)} HS300 constituents from local DB (date: {closest_date})")
            return result[['code', 'name']]
            
        except Exception as e:
            logger.error(f"Error fetching HS300 from local DB: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_hs300_components(self, date: str = None) -> pd.DataFrame:
        """
        获取沪深300成分股
        
        Args:
            date: 交易日期（格式: "20230101"），如果提供则尝试从本地数据库获取该日期的成分股
            
        Returns:
            包含代码、名称、行业等信息的DataFrame
        """
        logger.info(f"Fetching HS300 components... (date={date})")
        
        # 1. 优先从本地数据库获取
        if self.local_db_available:
            # 如果date=None，自动取本地DB中最新日期
            query_date = date
            if query_date is None:
                try:
                    conn = sqlite3.connect(self.local_db_path)
                    row = conn.execute(
                        "SELECT MAX(trade_date) FROM index_constituent WHERE index_code='000300.SH'"
                    ).fetchone()
                    conn.close()
                    if row and row[0]:
                        query_date = str(row[0]).replace("-", "")
                        logger.info(f"date=None, 自动使用最新日期: {query_date}")
                except Exception as e:
                    logger.warning(f"查询最新HS300日期失败: {e}")
            
            if query_date is not None:
                try:
                    result = self._get_hs300_from_local_db(query_date)
                    if result is not None and len(result) > 0:
                        logger.info(f"✓ Got {len(result)} HS300 components from local DB (date={query_date})")
                        return result
                    else:
                        logger.warning(f"No HS300 data in local DB for date={query_date}")
                except Exception as e:
                    logger.warning(f"Local DB query failed for HS300 date={query_date}: {e}")
        
        # 本地DB没有数据或查询失败，检查是否启用备份数据源
        if not self.tushare_backup_available and not self.akshare_backup_available:
            # 没有启用任何备份数据源，直接抛出异常
            logger.error(f"Local DB failed for HS300 date={date}, and no backup sources enabled")
            raise Exception(f"Failed to fetch HS300 components for date={date} (local DB only)")
        
        logger.info(f"Local DB failed for HS300 date={date}, trying backups...")
        
        # 嵌套函数定义：AkShare获取
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
            
            # 速率限制（Tushare API）
            self.rate_limiter.wait()
            
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
                # 速率限制（Tushare API）
                self.rate_limiter.wait()
                
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
        
        # ── 0. 优先从本地数据库读取（index_constituent 表）───
        if self.local_db_available:
            try:
                logger.debug("Trying local DB for HS300 components...")
                import sqlite3
                conn = sqlite3.connect(self.local_db_path)
                cursor = conn.cursor()
                
                # 读取沪深300成分股（index_code = '000300.SH'）
                cursor.execute("""
                    SELECT ts_code FROM index_constituent
                    WHERE index_code = '000300.SH'
                """)
                rows = cursor.fetchall()
                conn.close()
                
                if rows and len(rows) > 200:
                    import pandas as pd
                    codes = [r[0].replace('.SH', '').replace('.SZ', '') for r in rows]
                    result = pd.DataFrame({'code': codes, 'name': [''] * len(codes)})
                    logger.info(f"✅ Local DB HS300: 获取 {len(result)} 只成分股")
                    return result[['code', 'name']]
                else:
                    logger.warning(f"Local DB HS300 成分股数量不足: {len(rows) if rows else 0}")
            except Exception as e:
                logger.warning(f"Local DB failed for HS300 components: {e}")
        
        # ─ 1. 尝试 Tushare ─
        if self.tushare_backup_available:
            try:
                logger.debug("Trying Tushare for HS300 components...")
                result = _get_hs300_ts()
                if result is not None and len(result) > 0:
                    logger.debug(f"✓ Got {len(result)} HS300 components (Tushare)")
                    return result
                else:
                    logger.warning("Tushare returned empty data for HS300 components")
            except Exception as e:
                logger.warning(f"Tushare failed for HS300 components: {e}")
        
        # 2. Tushare 失败，尝试 AkShare
        if self.akshare_backup_available:
            try:
                logger.debug("Trying AkShare for HS300 components...")
                result = _get_hs300_ak()
                if result is not None and len(result) > 0:
                    logger.debug(f"✓ Got {len(result)} HS300 components (AkShare)")
                    return result
                else:
                    logger.warning("AkShare returned empty data for HS300 components")
            except Exception as e:
                logger.warning(f"AkShare failed for HS300 components: {e}")
        
        # 3. 所有数据源都失败
        logger.error("All data sources failed for HS300 components")
        raise Exception("Failed to fetch HS300 components from all sources")
    
    def get_zz800_components(self, date: str = None) -> pd.DataFrame:
        """
        获取中证800成分股（沪深300 + 中证500）
        指数代码：000906.SH
        
        Args:
            date: 交易日期（格式: "20230101"），当前未使用（ZZ800本地数据未准备）
        """
        logger.info(f"Fetching CSI 800 (ZZ800) components... (date={date})")

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
            
            # 速率限制（Tushare API）
            self.rate_limiter.wait()
            
            # 中证800 = 沪深300 + 中证500
            df300 = self.ts_pro.index_member(index_code='000300.SH')
            
            # 速率限制（Tushare API）
            self.rate_limiter.wait()
            
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
        
        # ── 0. 优先从本地数据库读取（index_constituent 表）───
        if self.local_db_available:
            try:
                logger.debug("Trying local DB for ZZ800 components...")
                import sqlite3
                conn = sqlite3.connect(self.local_db_path)
                cursor = conn.cursor()
                
                # 读取中证800成分股（index_code = '000906.SH'）
                cursor.execute("""
                    SELECT ts_code FROM index_constituent
                    WHERE index_code = '000906.SH'
                """)
                rows = cursor.fetchall()
                conn.close()
                
                if rows and len(rows) > 300:
                    import pandas as pd
                    codes = [r[0].replace('.SH', '').replace('.SZ', '') for r in rows]
                    result = pd.DataFrame({'code': codes, 'name': [''] * len(codes)})
                    logger.info(f"✅ Local DB ZZ800: 获取 {len(result)} 只成分股")
                    return result[['code', 'name']]
                else:
                    logger.warning(f"Local DB ZZ800 成分股数量不足: {len(rows) if rows else 0}")
            except Exception as e:
                logger.warning(f"Local DB failed for ZZ800 components: {e}")
        
        # ─ 1. 尝试 Tushare ─
        if self.tushare_backup_available:
            try:
                logger.debug("Trying Tushare for ZZ800 components...")
                result = _get_zz800_ts()
                if result is not None and len(result) > 0:
                    logger.debug(f"✓ Got {len(result)} ZZ800 components (Tushare)")
                    return result
                else:
                    logger.warning("Tushare returned empty data for ZZ800 components")
            except Exception as e:
                logger.warning(f"Tushare failed for ZZ800 components: {e}")
        
        # 2. Tushare 失败，尝试 AkShare
        if self.akshare_backup_available:
            try:
                logger.debug("Trying AkShare for ZZ800 components...")
                result = _get_zz800_ak()
                if result is not None and len(result) > 0:
                    logger.debug(f"✓ Got {len(result)} ZZ800 components (AkShare)")
                    return result
                else:
                    logger.warning("AkShare returned empty data for ZZ800 components")
            except Exception as e:
                logger.warning(f"AkShare failed for ZZ800 components: {e}")
        
        # 3. 所有数据源都失败
        logger.error("All data sources failed for ZZ800 components")
        raise Exception("Failed to fetch ZZ800 components from all sources")
    
    def get_zz500_components(self, date: str = None) -> pd.DataFrame:
        """
        获取中证500成分股
        指数代码：000905.SH
        
        Args:
            date: 交易日期（格式: "20230101"），如果提供则尝试从本地数据库获取该日期的成分股
            
        Returns:
            包含代码、名称、行业等信息的DataFrame
        """
        logger.info(f"Fetching CSI 500 (ZZ500) components... (date={date})")
        
        # ── 0. 优先从本地数据库读取（index_constituent 表）───
        if self.local_db_available:
            try:
                logger.debug("Trying local DB for ZZ500 components...")
                import sqlite3
                conn = sqlite3.connect(self.local_db_path)
                cursor = conn.cursor()
                
                # 读取中证500成分股（index_code = '000905.SH'）
                cursor.execute("""
                    SELECT ts_code FROM index_constituent
                    WHERE index_code = '000905.SH'
                """)
                rows = cursor.fetchall()
                conn.close()
                
                if rows and len(rows) > 200:
                    import pandas as pd
                    codes = [r[0].replace('.SH', '').replace('.SZ', '') for r in rows]
                    result = pd.DataFrame({'code': codes, 'name': [''] * len(codes)})
                    logger.info(f"✅ Local DB ZZ500: 获取 {len(result)} 只成分股")
                    return result[['code', 'name']]
                else:
                    logger.warning(f"Local DB ZZ500 成分股数量不足: {len(rows) if rows else 0}")
            except Exception as e:
                logger.warning(f"Local DB failed for ZZ500 components: {e}")
        
        # ── 1. 尝试 Tushare ─────────────────────────────────
        if self.tushare_backup_available:
            try:
                logger.debug("Trying Tushare for ZZ500 components...")
                
                # 速率限制（Tushare API）
                self.rate_limiter.wait()
                
                # 使用 index_member API 获取当前成分股
                df = self.ts_pro.index_member(index_code='000905.SH')
                
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
                        logger.info(f"✓ Got {len(result)} ZZ500 components (Tushare)")
                        return result[['code', 'name']]
                
                # 如果 index_member 没有数据，尝试使用 index_weight
                try:
                    # 速率限制（Tushare API）
                    self.rate_limiter.wait()
                    
                    df_weight = self.ts_pro.index_weight(
                        index_code='000905.SH',
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
                        logger.info(f"✓ Got {len(result)} ZZ500 components (Tushare index_weight)")
                        return result[['code', 'name']]
                except:
                    pass
                
                logger.warning("Tushare returned empty data for ZZ500 components")
            except Exception as e:
                logger.warning(f"Tushare failed for ZZ500 components: {e}")
        
        # ── 2. Tushare 失败，尝试 AkShare ─────────────────
        if self.akshare_backup_available:
            try:
                logger.debug("Trying AkShare for ZZ500 components...")
                
                # 使用 AkShare 获取中证500成分股
                # 指数代码：000905 = 中证500
                df = ak.index_stock_cons_csindex(symbol="000905")
                result = df[['成分券代码', '成分券名称']].copy()
                result.columns = ['code', 'name']
                
                if result is not None and len(result) > 0:
                    logger.debug(f"✓ Got {len(result)} ZZ500 components (AkShare)")
                    return result[['code', 'name']]
                else:
                    logger.warning("AkShare returned empty data for ZZ500 components")
            except Exception as e:
                logger.warning(f"AkShare failed for ZZ500 components: {e}")
        
        # ── 3. 所有数据源都失败 ───────────────────────────
        logger.error("All data sources failed for ZZ500 components")
        raise Exception("Failed to fetch ZZ500 components from all sources")
    
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
            
            # 速率限制（Tushare API）
            self.rate_limiter.wait()
            
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

        # 本地数据库失败或不可用，按固定优先级尝试其他数据源
        # 优先级：Tushare（付费，更靠谱） → AkShare（免费，作为备用）
        
        # 1. 优先尝试 Tushare
        if self.tushare_backup_available:
            try:
                logger.debug(f"Trying Tushare for {code} history...")
                result = _get_history_ts()
                if result is not None and len(result) > 0:
                    logger.debug(f"✓ Got {len(result)} days of history for {code} (Tushare)")
                    return result
                else:
                    logger.warning(f"Tushare returned empty data for {code}")
            except Exception as e:
                logger.warning(f"Tushare failed for {code}: {e}")
        
        # 2. Tushare 失败，尝试 AkShare
        if self.akshare_backup_available:
            try:
                logger.debug(f"Trying AkShare for {code} history...")
                result = _get_history_ak()
                if result is not None and len(result) > 0:
                    logger.debug(f"✓ Got {len(result)} days of history for {code} (AkShare)")
                    return result
                else:
                    logger.warning(f"AkShare returned empty data for {code}")
            except Exception as e:
                logger.warning(f"AkShare failed for {code}: {e}")
        
        # 3. 所有数据源都失败
        logger.error(f"All data sources failed for {code} history")
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
                # 重命名列为AkShare格式（与AkShare返回的格式一致）
                df = df.rename(columns={
                    'trade_date': '日期',
                    'open': '开盘',
                    'high': '最高',
                    'low': '最低',
                    'close': '收盘',
                    'pre_close': '前收盘',
                    'change': '涨跌额',
                    'pct_chg': '涨跌幅',
                    'vol': '成交量',
                    'amount': '成交额'
                })
                
                # 添加 adj_close 和 adj_open（回测插件需要）
                df['adj_close'] = df['收盘']
                df['adj_open'] = df['开盘']
                
                logger.debug(f"✓ Got history for {code} from local DB ({len(df)} rows)")
                return df
            
            logger.warning(f"No history data in local DB for {code}")
            return None
        except Exception as e:
            logger.error(f"Error reading history from local DB for {code}: {e}")
            return None
    
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
                end_date,
                ts_code,
                eps,
                roe,
                roa,
                netprofit_yoy,
                or_yoy,
                op_yoy,
                tr_yoy,
                debt_to_assets,
                current_ratio,
                assets_turn,
                grossprofit_margin,
                ocf_to_debt,
                ocfps,
                revenue_ps,
                roic
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
                trade_date,
                ts_code,
                pe,
                pb,
                ps,
                ps_ttm,
                dv_ratio,
                dv_ttm,
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
            
            # 速率限制（Tushare API）
            self.rate_limiter.wait()
            
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
    
    def get_financial_history(self, code: str, years: int = 3) -> Optional[pd.DataFrame]:
        """
        获取股票最近 years 年的财务指标历史（年报+季报），用于 PEG 增长稳定性护栏。
        按 end_date 降序返回，含 end_date / netprofit_yoy / or_yoy / op_yoy / roe / eps。
        """
        try:
            ts_code = self._convert_code_to_ts_format(code)
            conn = sqlite3.connect(self.local_db_path)
            limit = years * 4 + 2
            query = """
            SELECT end_date, ts_code, netprofit_yoy, or_yoy, op_yoy, roe, eps
            FROM fina_indicator
            WHERE ts_code = ?
            ORDER BY end_date DESC
            LIMIT ?
            """
            df = pd.read_sql_query(query, conn, params=(ts_code, limit))
            conn.close()
            if df is not None and len(df) > 0:
                logger.debug(f"✓ Got financial history for {code} ({len(df)} rows)")
                return df
            return None
        except Exception as e:
            logger.error(f"Error reading financial history from local DB for {code}: {e}")
            return None

    def get_financial_data_single(self, code: str) -> Optional[pd.DataFrame]:
        """
        获取单只股票财务指标数据（最新一期）
        优先从本地数据库读取，失败后用AkShare/Tushare
        注意：与批量版 get_financial_data(stock_codes, date) 区分，避免同名覆盖。
        
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
            
            # 速率限制（Tushare API）
            self.rate_limiter.wait()
            
            # 转换股票代码格式
            ts_code = self._convert_code_to_ts_format(code)
            
            # 获取最近一年的财务指标（不指定period，默认返回季度数据）
            from datetime import datetime, timedelta
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
            
            try:
                # Tushare fina_indicator 接口
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
        
        # 转换代码格式（添加 .SZ 或 .SH 后缀）
        ts_code = self._convert_code_to_ts_format(code)
        logger.debug(f"  Converted code: {code} -> {ts_code}")
        
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
            """, (ts_code, trade_date))
            
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

    
    def get_index_returns(self, index_code: str, start_date: str, end_date: str) -> np.ndarray:
        """
        获取指数收益率序列（用于计算 Beta）
        
        Args:
            index_code: 指数代码（如 "000300.SH"）
            start_date: 开始日期（格式: "20230101"）
            end_date: 结束日期（格式: "20241231"）
            
        Returns:
            指数收益率序列（numpy array），失败返回空数组
        """
        if not self.local_db_available:
            logger.warning("Local DB not available, cannot fetch index returns")
            return np.array([])
        
        try:
            conn = sqlite3.connect(self.local_db_path)
            
            # 查询指数历史数据
            query = """
                SELECT trade_date, close
                FROM index_daily
                WHERE ts_code = ?
                  AND trade_date BETWEEN ? AND ?
                ORDER BY trade_date ASC
            """
            
            df = pd.read_sql_query(query, conn, params=(index_code, start_date, end_date))
            conn.close()
            
            if df is None or len(df) == 0:
                logger.warning(f"No index data found for {index_code}")
                return np.array([])
            
            # 计算收益率
            close_prices = df["close"].values
            returns = np.diff(close_prices) / close_prices[:-1]
            
            logger.debug(f"Got {len(returns)} days of index returns for {index_code}")
            return returns
            
        except Exception as e:
            logger.error(f"Error fetching index returns for {index_code}: {e}")
            return np.array([])
    
    # ===== 价值投资策略 - 财务数据获取方法 =====
    
    def get_market_cap_all_a(self, date: str) -> pd.DataFrame:
        """
        获取全A股市值数据（用于计算中位数/平均值）
        
        Args:
            date: 交易日期（格式: "YYYYMMDD"）
            
        Returns:
            DataFrame with columns: ['ts_code', 'total_mv']
            total_mv单位：万元
        """
        logger.debug(f"Fetching market cap for all A-shares on {date}...")
        
        # 方法1：从本地数据库获取（daily_basic表）
        if self.local_db_available:
            try:
                conn = sqlite3.connect(self.local_db_path)
                
                query = """
                    SELECT ts_code, total_mv
                    FROM daily_basic
                    WHERE trade_date = ?
                """
                
                df = pd.read_sql_query(query, conn, params=(date,))
                conn.close()
                
                if df is not None and len(df) > 0:
                    # 删除total_mv为NULL的行
                    df = df.dropna(subset=['total_mv'])
                    
                    if len(df) > 0:
                        logger.info(f"✓ Got {len(df)} stocks' market cap from local DB (date={date})")
                        return df
                    else:
                        logger.warning(f"All market cap data is NULL for date={date}")
                else:
                    logger.warning(f"No market cap data found in local DB for date={date}")
                    
            except Exception as e:
                logger.warning(f"Failed to fetch market cap from local DB: {e}")
        
        # 方法2：从Tushare API获取（备用）
        if self.tushare_backup_available and self.ts_pro is not None:
            try:
                # 速率限制（Tushare API）
                self.rate_limiter.wait()
                
                df = self.ts_pro.daily_basic(
                    trade_date=date,
                    fields='ts_code,total_mv'
                )
                
                if df is not None and len(df) > 0:
                    logger.info(f"✓ Got {len(df)} stocks' market cap from Tushare (date={date})")
                    return df
                else:
                    logger.warning(f"No market cap data found from Tushare for date={date}")
                    
            except Exception as e:
                logger.warning(f"Failed to fetch market cap from Tushare: {e}")
        
        # 所有数据源都失败
        logger.error(f"All data sources failed for market cap (date={date})")
        return pd.DataFrame(columns=['ts_code', 'total_mv'])
    
    
    def get_current_ratio_all_a(self, date: str) -> pd.DataFrame:
        """
        获取全A股流动比率数据
        
        Args:
            date: 报告期（格式: "YYYYMMDD"，通常是季度末日期如 "20231231"）
            
        Returns:
            DataFrame with columns: ['ts_code', 'current_ratio']
        """
        logger.debug(f"Fetching current ratio for all A-shares (end_date={date})...")
        
        # 方法1：从本地数据库获取（fina_indicator表）
        if self.local_db_available:
            try:
                conn = sqlite3.connect(self.local_db_path)
                
                # 查询指定报告期的数据（使用end_date字段）
                query = """
                    SELECT ts_code, current_ratio
                    FROM fina_indicator
                    WHERE end_date = ?
                """
                
                df = pd.read_sql_query(query, conn, params=(date,))
                conn.close()
                
                if df is not None and len(df) > 0:
                    # 删除current_ratio为NULL的行
                    df = df.dropna(subset=['current_ratio'])
                    
                    if len(df) > 0:
                        logger.info(f"✓ Got {len(df)} stocks' current ratio from local DB (end_date={date})")
                        return df
                    else:
                        logger.warning(f"All current ratio data is NULL for end_date={date}")
                else:
                    logger.warning(f"No current ratio data found in local DB for end_date={date}")
                    
            except Exception as e:
                logger.warning(f"Failed to fetch current ratio from local DB: {e}")
        
        # 方法2：从Tushare API获取（备用）
        if self.tushare_backup_available and self.ts_pro is not None:
            try:
                # 速率限制（Tushare API）
                self.rate_limiter.wait()
                
                df = self.ts_pro.fina_indicator(
                    end_date=date,
                    fields='ts_code,current_ratio'
                )
                
                if df is not None and len(df) > 0:
                    logger.info(f"✓ Got {len(df)} stocks' current ratio from Tushare (end_date={date})")
                    return df
                else:
                    logger.warning(f"No current ratio data found from Tushare for end_date={date}")
                    
            except Exception as e:
                logger.warning(f"Failed to fetch current ratio from Tushare: {e}")
        
        # 所有数据源都失败
        logger.error(f"All data sources failed for current ratio (end_date={date})")
        return pd.DataFrame(columns=['ts_code', 'current_ratio'])
    
    
    def get_roe_all_a(self, date: str) -> pd.DataFrame:
        """
        获取全A股ROE数据
        
        Args:
            date: 报告期（格式: "YYYYMMDD"，通常是季度末日期如 "20231231"）
            
        Returns:
            DataFrame with columns: ['ts_code', 'roe']
        """
        logger.debug(f"Fetching ROE for all A-shares (end_date={date})...")
        
        # 方法1：从本地数据库获取（fina_indicator表）
        if self.local_db_available:
            try:
                conn = sqlite3.connect(self.local_db_path)
                
                # 查询指定报告期的数据（使用end_date字段）
                query = """
                    SELECT ts_code, roe
                    FROM fina_indicator
                    WHERE end_date = ?
                """
                
                df = pd.read_sql_query(query, conn, params=(date,))
                conn.close()
                
                if df is not None and len(df) > 0:
                    # 删除roe为NULL的行
                    df = df.dropna(subset=['roe'])
                    
                    if len(df) > 0:
                        logger.info(f"✓ Got {len(df)} stocks' ROE from local DB (end_date={date})")
                        return df
                    else:
                        logger.warning(f"All ROE data is NULL for end_date={date}")
                else:
                    logger.warning(f"No ROE data found in local DB for end_date={date}")
                    
            except Exception as e:
                logger.warning(f"Failed to fetch ROE from local DB: {e}")
        
        # 方法2：从Tushare API获取（备用）
        if self.tushare_backup_available and self.ts_pro is not None:
            try:
                # 速率限制（Tushare API）
                self.rate_limiter.wait()
                
                df = self.ts_pro.fina_indicator(
                    end_date=date,
                    fields='ts_code,roe'
                )
                
                if df is not None and len(df) > 0:
                    logger.info(f"✓ Got {len(df)} stocks' ROE from Tushare (end_date={date})")
                    return df
                else:
                    logger.warning(f"No ROE data found from Tushare for end_date={date}")
                    
            except Exception as e:
                logger.warning(f"Failed to fetch ROE from Tushare: {e}")
        
        # 所有数据源都失败
        logger.error(f"All data sources failed for ROE (end_date={date})")
        return pd.DataFrame(columns=['ts_code', 'roe'])
    
    
    def get_financial_data(self, stock_codes: list, date: str) -> pd.DataFrame:
        """
        获取指定股票的财务数据（核心方法）
        
        从 daily_basic 和 fina_indicator 表获取所有需要的财务数据
        
        Args:
            stock_codes: 股票代码列表（如 ["000001.SZ", "600000.SH"]）
            date: 报告期（格式: "YYYYMMDD"）
                 对于daily_basic表使用trade_date，对于fina_indicator表使用end_date
            
        Returns:
            DataFrame with columns:
            ['ts_code', 'name', 'total_mv', 'current_ratio', 'roe', 
             'fcff', 'op_yoy', 'eps']
        """
        logger.info(f"🔍 [get_financial_data] 开始获取 {len(stock_codes)} 只股票的财务数据 (date={date})...")
        logger.info(f"🔍 [get_financial_data] 股票代码示例: {stock_codes[:5]}")
        
        # 将股票代码转换为Tushare格式（添加 .SZ 或 .SH 后缀）
        logger.info(f"🔍 [get_financial_data] 转换股票代码格式...")
        ts_codes = [self._convert_code_to_ts_format(code) for code in stock_codes]
        logger.info(f"🔍 [get_financial_data] 转换后示例: {ts_codes[:5]}")
        
        # 初始化结果DataFrame（使用转换后的代码）
        result_df = pd.DataFrame({'ts_code': ts_codes})
        logger.info(f"🔍 [get_financial_data] 初始化 result_df: {len(result_df)} 行")
        
        # ========== 1. 获取股票名称 ==========
        result_df['name'] = result_df['ts_code'].apply(lambda x: self._get_stock_name(x))
        
        # ========== 2. 获取市值数据（daily_basic表）==========
        if self.local_db_available:
            try:
                conn = sqlite3.connect(self.local_db_path)
                
                # 查询最接近date的交易日的市值数据
                # 因为daily_basic表使用trade_date（交易日），而不是报告期
                # 注意：必须选择 trade_date 列，否则后续无法按它排序/去重
                placeholders = ','.join(['?'] * len(ts_codes))
                query = f"""
                    SELECT ts_code, trade_date, total_mv
                    FROM daily_basic
                    WHERE ts_code IN ({placeholders})
                      AND trade_date <= ?
                    ORDER BY trade_date DESC
                """
                
                # 对每个股票取最近一个交易日的数据
                df_mv = pd.read_sql_query(
                    query, conn, 
                    params=ts_codes + [date]
                )
                conn.close()
                
                logger.info(f"🔍 [get_financial_data] daily_basic 查询返回: {len(df_mv)} 行")
                if len(df_mv) > 0:
                    logger.info(f"🔍 [get_financial_data] df_mv 列: {list(df_mv.columns)}")
                    logger.info(f"🔍 [get_financial_data] df_mv 前3行:\n{df_mv.head(3).to_string()}")
                
                if df_mv is not None and len(df_mv) > 0:
                    # 对每个股票，取trade_date最大的那条记录
                    # 方法：按ts_code分组，取trade_date最大的行
                    df_mv = df_mv.sort_values(['ts_code', 'trade_date'], ascending=[True, False])
                    df_mv = df_mv.drop_duplicates(subset=['ts_code'], keep='first')
                    df_mv = df_mv[['ts_code', 'total_mv']]
                    
                    # 合并到结果DataFrame
                    result_df = result_df.merge(df_mv, on='ts_code', how='left')
                    
                    logger.info(f"✓ Got market cap for {len(df_mv)} stocks from local DB")
                else:
                    logger.warning(f"No market cap data found in local DB for given stocks")
                    result_df['total_mv'] = np.nan
                    
            except Exception as e:
                logger.warning(f"Failed to fetch market cap from local DB: {e}")
                result_df['total_mv'] = np.nan
        else:
            result_df['total_mv'] = np.nan
        
        # ========== 3. 获取财务指标数据（fina_indicator表）==========
        if self.local_db_available:
            try:
                conn = sqlite3.connect(self.local_db_path)
                
                # 查询指定报告期的财务数据
                placeholders = ','.join(['?'] * len(ts_codes))
                query = f"""
                    SELECT ts_code, current_ratio, roe, fcff, op_yoy, eps
                    FROM fina_indicator
                    WHERE ts_code IN ({placeholders})
                      AND end_date = ?
                """
                
                df_fin = pd.read_sql_query(
                    query, conn,
                    params=ts_codes + [date]
                )
                conn.close()
                
                logger.info(f"🔍 [get_financial_data] fina_indicator 查询返回: {len(df_fin)} 行")
                if len(df_fin) > 0:
                    logger.info(f"🔍 [get_financial_data] df_fin 列: {list(df_fin.columns)}")
                    logger.info(f"🔍 [get_financial_data] df_fin 前3行:\n{df_fin.head(3).to_string()}")
                
                if df_fin is not None and len(df_fin) > 0:
                    # 合并到结果DataFrame
                    # 注意：只有部分股票有数据，所以使用 how='left'
                    result_df = result_df.merge(df_fin, on='ts_code', how='left')
                    
                    logger.info(f"✓ Got financial indicators for {len(df_fin)} stocks from local DB")
                    logger.info(f"  {len(df_fin)}/{len(stock_codes)} stocks have financial data")
                else:
                    logger.warning(f"No financial data found in local DB for given stocks")
                    # 添加空列（全设为NaN）
                    for col in ['current_ratio', 'roe', 'fcff', 'op_yoy', 'eps']:
                        result_df[col] = np.nan
                    
            except Exception as e:
                logger.warning(f"Failed to fetch financial data from local DB: {e}")
                for col in ['current_ratio', 'roe', 'fcff', 'op_yoy', 'eps']:
                    result_df[col] = np.nan
        else:
            for col in ['current_ratio', 'roe', 'fcff', 'op_yoy', 'eps']:
                result_df[col] = np.nan
        
        # ========== 4. 如果本地DB数据不完整，尝试Tushare API（备用）==========
        # 检查是否有NaN值
        nan_count = result_df.isnull().sum().sum()
        if nan_count > 0 and self.tushare_backup_available and self.ts_pro is not None:
            logger.info(f"Found {nan_count} NaN values, trying Tushare API backup...")
            
            try:
                # 速率限制（Tushare API）
                self.rate_limiter.wait()
                
                # 高效方法：只调用1次API获取*所有*股票在end_date的数据
                logger.info(f"  🔍 [get_financial_data] 调用 Tushare API (所有股票, end_date={date})...")
                
                df_tushare_all = self.ts_pro.fina_indicator(
                    end_date=date,
                    fields='ts_code,current_ratio,roe,fcff,op_yoy,eps'
                )
                
                if df_tushare_all is not None and len(df_tushare_all) > 0:
                    logger.info(f"  🔍 [get_financial_data] Tushare API 返回: {len(df_tushare_all)} 行 (所有股票)")
                    
                    # 本地过滤：只保留我们需要的股票
                    ts_codes_set = set(result_df['ts_code'].tolist())
                    df_tushare = df_tushare_all[df_tushare_all['ts_code'].isin(ts_codes_set)]
                    
                    logger.info(f"  🔍 [get_financial_data] 本地过滤后: {len(df_tushare)} 行 (我们只需要的股票)")
                    logger.info(f"  🔍 [get_financial_data] df_tushare 列: {list(df_tushare.columns)}")
                    logger.info(f"  🔍 [get_financial_data] df_tushare 前3行:\n{df_tushare.head(3).to_string()}")
                    
                    # 去重（保留最后一条）
                    df_tushare = df_tushare.drop_duplicates(subset=['ts_code'], keep='last')
                    logger.info(f"  🔍 [get_financial_data] 去重后: {len(df_tushare)} 行")
                    
                    # 合并到结果DataFrame（只填充NaN值）
                    for col in ['current_ratio', 'roe', 'fcff', 'op_yoy', 'eps']:
                        if col in df_tushare.columns:
                            # 创建映射字典：ts_code -> col值
                            tushare_dict = df_tushare.set_index('ts_code')[col].to_dict()
                            logger.info(f"  🔍 [get_financial_data] tushare_dict[{col}] 示例（前5个）: {list(tushare_dict.items())[:5]}")
                            
                            # 只填充 NaN 值
                            mask = result_df[col].isnull()
                            logger.info(f"  🔍 [get_financial_data] {col} 列有 {mask.sum()} 个 NaN 值需要填充")
                            
                            # 使用 map 填充
                            mapped_values = result_df.loc[mask, 'ts_code'].map(tushare_dict)
                            logger.info(f"  🔍 [get_financial_data] map() 后的值示例（前5个）: {mapped_values.head(5).tolist()}")
                            
                            result_df.loc[mask, col] = mapped_values
                            logger.info(f"  🔍 [get_financial_data] 填充后 {col} 列仍有 {result_df[col].isnull().sum()} 个 NaN 值")
                    
                    logger.info(f"✓ Got financial data from Tushare API backup")
                else:
                    logger.warning(f"  🔍 [get_financial_data] Tushare API 返回空数据 (end_date={date})")
                
            except Exception as e:
                logger.warning(f"Failed to fetch financial data from Tushare: {e}")
                import traceback
                logger.warning(f"Traceback: {traceback.format_exc()}")
        
        # ========== 5. 最终检查 ==========
        final_nan_count = result_df.isnull().sum().sum()
        if final_nan_count > 0:
            logger.warning(f"Final result has {final_nan_count} NaN values")
            logger.warning(f"🔍 [get_financial_data] result_df 列: {list(result_df.columns)}")
            logger.warning(f"🔍 [get_financial_data] result_df NaN 统计:\n{result_df.isnull().sum()}")
            logger.warning(f"🔍 [get_financial_data] result_df 前3行:\n{result_df.head(3).to_string()}")
        
        logger.info(f"✓ Financial data fetching complete: {len(result_df)} stocks")
        return result_df
    
    
    def _get_stock_name(self, ts_code: str) -> str:
        """
        获取单个股票的名称（辅助方法）
        
        Args:
            ts_code: Tushare格式的股票代码（如 "000001.SZ"）
            
        Returns:
            股票名称（如 "平安银行"）
        """
        # 转换格式：000001.SZ -> 000001
        simple_code = ts_code.split('.')[0]
        
        # 先从缓存查找
        if hasattr(self, '_stock_name_cache') and simple_code in self._stock_name_cache:
            return self._stock_name_cache[simple_code]
        
        # 从本地数据库查找
        if self.local_db_available:
            try:
                conn = sqlite3.connect(self.local_db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT name FROM stock_basic WHERE ts_code = ? LIMIT 1
                """, (ts_code,))
                
                row = cursor.fetchone()
                conn.close()
                
                if row is not None:
                    name = row[0]
                    # 缓存结果
                    if not hasattr(self, '_stock_name_cache'):
                        self._stock_name_cache = {}
                    self._stock_name_cache[simple_code] = name
                    return name
                    
            except Exception as e:
                logger.debug(f"Error fetching stock name for {ts_code}: {e}")
        
        # 默认返回空字符串
        return ""
    
    
    # ===== 价值投资策略 - 历史自由现金流检查 =====
    
    def get_historical_fcff(self, stock_codes: list, end_date: str, years: int = 5) -> pd.DataFrame:
        """
        获取股票过去N年的FCFF数据，并检查是否都为正
        
        只使用年报数据（end_date以12-31结尾），因为：
        1. 年报是经过审计的，数据最权威
        2. 避免季度数据的季节性波动
        3. "近N年自由现金流为正"通常指年报
        
        Args:
            stock_codes: 股票代码列表（如 ["000001.SZ", "600000.SH"]）
            end_date: 结束报告期（格式: "YYYYMMDD"，如 "20231231"）
            years: 检查过去N年（默认5年）
            
        Returns:
            DataFrame with columns:
            ['ts_code', 'fcff_years_all_positive', 'fcff_details']
            
            fcff_years_all_positive: bool (是否所有年份FCFF都>0)
            fcff_details: dict (详细的每年FCFF值，如 {2023: 100, 2022: 200, ...})
        """
        logger.info(f"Checking historical FCFF for {len(stock_codes)} stocks (past {years} years, end_date={end_date})...")
        
        # 计算起始年份
        end_year = int(end_date[:4])
        start_year = end_year - years + 1
        
        # 构建需要查询的年报报告期列表
        # 例如：years=5, end_date=20231231 → [20231231, 20221231, ..., 20191231]
        report_dates = [f"{year}1231" for year in range(start_year, end_year + 1)]
        
        logger.debug(f"Will check FCFF for report dates: {report_dates}")
        
        # 初始化结果DataFrame
        result_df = pd.DataFrame({'ts_code': stock_codes})
        result_df['fcff_years_all_positive'] = False
        result_df['fcff_details'] = None  # 将存储dict
        
        # ========== 从本地数据库获取历史FCFF数据 ==========
        if self.local_db_available:
            try:
                conn = sqlite3.connect(self.local_db_path)
                
                # 批量查询：获取所有股票在指定报告期的FCFF数据
                # 使用参数化查询，避免SQL注入
                query = """
                    SELECT ts_code, end_date, fcff
                    FROM fina_indicator
                    WHERE ts_code IN ({})
                      AND end_date IN ({})
                      AND fcff IS NOT NULL
                    ORDER BY ts_code, end_date
                """.format(
                    ','.join(['?'] * len(stock_codes)),
                    ','.join(['?'] * len(report_dates))
                )
                
                params = stock_codes + report_dates
                df_fcff = pd.read_sql_query(query, conn, params=params)
                conn.close()
                
                if df_fcff is not None and len(df_fcff) > 0:
                    logger.info(f"✓ Got {len(df_fcff)} FCFF records from local DB")
                    
                    # 对每个股票，检查是否所有年份的FCFF都>0
                    for ts_code in stock_codes:
                        stock_data = df_fcff[df_fcff['ts_code'] == ts_code]
                        
                        if len(stock_data) == 0:
                            # 没有数据，标记为False
                            continue
                        
                        # 构建详情dict：{年份: FCFF值}
                        details = {}
                        all_positive = True
                        
                        for _, row in stock_data.iterrows():
                            year = int(row['end_date']) // 10000  # 20231231 → 2023
                            fcff_val = row['fcff']
                            details[year] = fcff_val
                            
                            if fcff_val <= 0:
                                all_positive = False
                        
                        # 检查是否所有需要的年份都有数据
                        available_years = set(details.keys())
                        required_years = set(range(start_year, end_year + 1))
                        
                        if available_years >= required_years:
                            # 所有需要的年份都有数据
                            result_df.loc[result_df['ts_code'] == ts_code, 'fcff_years_all_positive'] = all_positive
                            result_df.loc[result_df['ts_code'] == ts_code, 'fcff_details'] = [details]
                        else:
                            # 数据不完整，标记为False
                            logger.debug(f"  {ts_code}: FCFF data incomplete (got {sorted(available_years)}, need {sorted(required_years)})")
                            result_df.loc[result_df['ts_code'] == ts_code, 'fcff_years_all_positive'] = False
                            result_df.loc[result_df['ts_code'] == ts_code, 'fcff_details'] = [details]
                    
                else:
                    logger.warning(f"No FCFF data found in local DB for given stocks and report dates")
                    
            except Exception as e:
                logger.warning(f"Failed to fetch historical FCFF from local DB: {e}")
        
        # ========== 如果本地DB数据不完整，尝试Tushare API（备用）==========
        # 检查是否有股票需要补充数据（完全没数据 或 数据不完整）
        needs_backup = []
        for idx, row in result_df.iterrows():
            ts_code = row['ts_code']
            details = row['fcff_details']
            
            if details is None:
                # 完全没数据
                needs_backup.append(ts_code)
            elif isinstance(details, dict):
                # 有数据但不完整
                available_years = set(details.keys())
                required_years = set(range(start_year, end_year + 1))
                if available_years < required_years:
                    needs_backup.append(ts_code)
        
        if len(needs_backup) > 0 and self.tushare_backup_available and self.ts_pro is not None:
            logger.info(f"Found {len(needs_backup)} stocks needing FCFF data backup, trying Tushare API...")
            
            try:
                # 从Tushare API获取（注意：Tushare有速率限制，需要循环）
                for ts_code in needs_backup:  # 不再限制10只
                    self.rate_limiter.wait()
                    
                    df_api = self.ts_pro.fina_indicator(
                        ts_code=ts_code,
                        start_date=f"{start_year}0101",
                        end_date=end_date,
                        fields='ts_code,end_date,fcff'
                    )
                    
                    if df_api is not None and len(df_api) > 0:
                        # 筛选年报数据
                        df_annual = df_api[df_api['end_date'].str.endswith('1231')]
                        
                        if len(df_annual) > 0:
                            # 获取已有的details（可能来自本地DB）
                            existing_details = result_df.loc[result_df['ts_code'] == ts_code, 'fcff_details'].iloc[0]
                            if existing_details is None:
                                existing_details = {}
                            
                            # 合并数据：用API数据补充/覆盖
                            for _, row in df_annual.iterrows():
                                year = int(row['end_date']) // 10000
                                fcff_val = row['fcff']
                                if fcff_val is not None:
                                    existing_details[year] = fcff_val
                            
                            # 检查是否所有需要的年份都有数据
                            available_years = set(existing_details.keys())
                            required_years = set(range(start_year, end_year + 1))
                            
                            if available_years >= required_years:
                                # 所有需要的年份都有数据，检查是否都>0
                                all_positive = all(v > 0 for v in existing_details.values() if v is not None)
                                result_df.loc[result_df['ts_code'] == ts_code, 'fcff_years_all_positive'] = all_positive
                            
                            # 更新details（即使不完整也更新）
                            result_df.loc[result_df['ts_code'] == ts_code, 'fcff_details'] = [existing_details]
                
                logger.info(f"✓ Tushare API backup complete for {len(needs_backup)} stocks")
                    
            except Exception as e:
                logger.warning(f"Failed to fetch FCFF from Tushare API: {e}")
        
        # ========== 最终统计 ==========
        positive_count = result_df['fcff_years_all_positive'].sum()
        logger.info(f"✓ Historical FCFF check complete: {positive_count}/{len(stock_codes)} stocks have all positive FCFF in past {years} years")
        
        return result_df[['ts_code', 'fcff_years_all_positive', 'fcff_details']]
