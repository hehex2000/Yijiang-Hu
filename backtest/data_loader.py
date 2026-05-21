"""
数据加载模块 - 从 SQLite 数据库加载股票数据（前复权价格 + MA计算）
支持 Tushare API 作为备用数据源
"""

import sqlite3
import pandas as pd
import tushare as ts
from loguru import logger
from typing import List, Dict, Optional


# 数据库路径
DB_PATH = "D:/tu-shareData/astock_daily.db"

# Tushare token（从配置文件读取）
TUSHARE_TOKEN = ""


class DataLoader:
    """数据加载器：从 SQLite 加载股票数据，计算前复权价格和均线"""
    
    def __init__(self, db_path: str = DB_PATH, tushare_token: str = ""):
        """
        初始化数据加载器
        
        Args:
            db_path: SQLite 数据库路径
            tushare_token: Tushare API token（可选，用于API备用）
        """
        self.db_path = db_path
        self.tushare_token = tushare_token
        self.db_available = False
        
        # 测试数据库连接
        try:
            conn = sqlite3.connect(self.db_path)
            conn.close()
            self.db_available = True
            logger.info(f"DataLoader initialized: db_path={db_path} (DB available)")
        except Exception as e:
            logger.warning(f"SQLite database not available: {e}")
            logger.warning("Will use Tushare API as backup")
            self.db_available = False
    
    def _convert_code_to_ts_format(self, code: str) -> str:
        """
        将股票代码转换为 Tushare 格式
        
        Args:
            code: 股票代码（支持简单格式如 "000001", "600000" 
                  或 Tushare 格式如 "000001.SZ", "600000.SH"）
            
        Returns:
            Tushare 格式（如 "000001.SZ", "600000.SH"）
        """
        code = str(code).strip().upper()
        # 已经是 Tushare 格式（含后缀），直接返回
        if code.endswith(('.SZ', '.SH')):
            return code
        # 简单格式，需要转换
        if code.startswith('6'):
            return f"{code}.SH"
        elif code.startswith(('0', '3')):
            return f"{code}.SZ"
        else:
            return code
    
    def get_adjusted_prices(self, code: str, 
                              start_date: str, 
                              end_date: str,
                              ma_short: int = 10,
                              ma_long: int = 60,
                              channel_period: int = 30) -> Optional[pd.DataFrame]:
        """
        获取前复权价格数据（考虑分红送股）+ 计算 MA 和海龟通道
        
        Args:
            code: 股票代码（简单格式，如 "000063"）
            start_date: 开始日期（格式: "20250101"）
            end_date: 结束日期（格式: "20260501"）
            ma_short: 短期均线周期（默认10日）
            ma_long: 长期均线周期（默认60日）
            channel_period: 海龟通道周期（默认30日）
            
        Returns:
            DataFrame with columns: trade_date, adj_close, ma_short, ma_long, signal, signal_change, turtle_signal, turtle_signal_change, upper_channel, lower_channel
            如果无数据，返回 None
        """
        logger.debug(f"Loading adjusted prices for {code} from {start_date} to {end_date}")
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # 转换代码格式
            ts_code = self._convert_code_to_ts_format(code)
            
            # 获取价格（open, high, low, close）和复权因子
            query = f"""
                SELECT d.trade_date, d.open, d.high, d.low, d.close, af.adj_factor
                FROM daily d
                JOIN adj_factor af ON d.ts_code = af.ts_code 
                                    AND d.trade_date = af.trade_date
                WHERE d.ts_code = '{ts_code}'
                  AND d.trade_date BETWEEN '{start_date}' AND '{end_date}'
                ORDER BY d.trade_date
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if len(df) == 0:
                logger.warning(f"No data for {code}")
                return None
            
            # 计算前复权价格（以最新价格为基准）
            latest_adj = df['adj_factor'].iloc[-1]
            df['adj_open'] = df['open'] * (df['adj_factor'] / latest_adj)
            df['adj_high'] = df['high'] * (df['adj_factor'] / latest_adj)
            df['adj_low'] = df['low'] * (df['adj_factor'] / latest_adj)
            df['adj_close'] = df['close'] * (df['adj_factor'] / latest_adj)
            
            # 计算短期和长期均线
            df[f'ma{ma_short}'] = df['adj_close'].rolling(window=ma_short).mean()
            df[f'ma{ma_long}'] = df['adj_close'].rolling(window=ma_long).mean()
            
            # 计算海龟通道（channel_period日最高/最低价，当前日之前）
            df['upper_channel'] = df['adj_close'].rolling(window=channel_period).max().shift(1)
            df['lower_channel'] = df['adj_close'].rolling(window=channel_period).min().shift(1)
            
            # 生成双均线信号：1=买入（金叉），-1=卖出（死叉），0=持有
            df['signal'] = 0
            df.loc[df[f'ma{ma_short}'] > df[f'ma{ma_long}'], 'signal'] = 1
            df.loc[df[f'ma{ma_short}'] < df[f'ma{ma_long}'], 'signal'] = -1
            
            # 生成海龟信号：1=买入（突破上限），-1=卖出（跌破下限），0=持有
            df['turtle_signal'] = 0
            df.loc[df['adj_close'] > df['upper_channel'], 'turtle_signal'] = 1
            df.loc[df['adj_close'] < df['lower_channel'], 'turtle_signal'] = -1
            
            # 双均线信号变化：只在信号变化时触发交易
            df['signal_change'] = df['signal'].diff().fillna(0)
            
            # 海龟信号变化
            df['turtle_signal_change'] = df['turtle_signal'].diff().fillna(0)
            
            logger.debug(f"✓ Got {len(df)} days of adjusted prices for {code}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load adjusted prices for {code}: {e}")
            return None
    
    def get_monthly_trading_days(self, df: pd.DataFrame) -> List[str]:
        """
        获取每月第一个交易日
        
        Args:
            df: 包含 trade_date 列的 DataFrame
            
        Returns:
            每月第一个交易日的日期列表（字符串格式：YYYYMMDD）
        """
        # 提取 YYYYMM 格式的年-月
        df = df.copy()
        df['year_month'] = df['trade_date'].str[:6]
        
        # 获取每月第一个交易日
        monthly_first = df.groupby('year_month')['trade_date'].min().tolist()
        
        logger.debug(f"Found {len(monthly_first)} monthly trading days")
        return monthly_first
    
    def get_benchmark_returns(self, code: str,
                                start_date: str,
                                end_date: str) -> Optional[pd.DataFrame]:
        """
        获取基准收益（买入持有策略）
        
        Args:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame with columns: trade_date, adj_close, cumulative_return
        """
        df = self.get_adjusted_prices(code, start_date, end_date)
        
        if df is None:
            return None
        
        # 计算累计收益（假设第一天买入并持有）
        initial_price = df['adj_close'].iloc[0]
        df['cumulative_return'] = (df['adj_close'] - initial_price) / initial_price
        
        return df[['trade_date', 'adj_close', 'cumulative_return']]
    
    def get_index_data(self, index_code: str,
                      start_date: str,
                      end_date: str) -> Optional[pd.DataFrame]:
        """
        获取指数数据（如沪深300：000300.SH）
        优先从本地 SQLite index_daily 表加载，失败则使用 Tushare API

        Args:
            index_code: 指数代码（如 "000300.SH"）
            start_date: 开始日期（格式: "20200101"）
            end_date: 结束日期（格式: "20241231"）

        Returns:
            DataFrame with columns: trade_date, close, cumulative_return
        """
        logger.debug(f"Loading index data for {index_code} from {start_date} to {end_date}")

        # 优先从本地 SQLite index_daily 表加载
        try:
            conn = sqlite3.connect(self.db_path)

            query = f"""
                SELECT trade_date, close
                FROM index_daily
                WHERE ts_code = '{index_code}'
                  AND trade_date BETWEEN '{start_date}' AND '{end_date}'
                ORDER BY trade_date
            """

            df = pd.read_sql_query(query, conn)
            conn.close()

            if len(df) > 0:
                df['close'] = pd.to_numeric(df['close'], errors='coerce')
                df = df.dropna(subset=['close']).reset_index(drop=True)
                df['cumulative_return'] = (df['close'] / df['close'].iloc[0]) - 1
                logger.info(f"✓ Got {len(df)} days of index data for {index_code} from local SQLite")
                return df[['trade_date', 'close', 'cumulative_return']]
            else:
                logger.debug(f"No index data for {index_code} in SQLite, trying API...")

        except Exception as e:
            logger.debug(f"SQLite index_daily load failed: {e}, trying Tushare API...")

        # SQLite 无数据，使用 Tushare API
        try:
            import yaml
            import os

            config_file = os.path.join(os.path.dirname(__file__), '..', 'config', 'backtest_config.yaml')
            tushare_token = ""
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                tushare_token = config.get('tushare_token', '')
            if not tushare_token:
                tushare_token = self.tushare_token

            if not tushare_token:
                logger.warning("Tushare token not found, skipping index data")
                return None

            ts.set_token(tushare_token)
            pro = ts.pro_api()

            df = pro.index_daily(
                ts_code=index_code,
                start_date=start_date,
                end_date=end_date
            )

            if len(df) == 0:
                logger.warning(f"No index data for {index_code} from Tushare API")
                return None

            df = df.sort_values('trade_date')
            df['cumulative_return'] = (df['close'] / df['close'].iloc[0]) - 1

            logger.info(f"✓ Got {len(df)} days of index data for {index_code} from Tushare API")
            return df[['trade_date', 'close', 'cumulative_return']]

        except Exception as e:
            logger.error(f"Failed to load index data for {index_code}: {e}")
            return None
