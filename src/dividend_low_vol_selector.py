"""
红利低波选股器
====================
双重排序法：红利因子（高股息率 dv_ttm）+ 低波因子（低波动率）
集成到 multi_factor_selection 主平台

选股逻辑（来自 value_selection/run_dividend_lowvol.py）：
1. 中证800成分股
2. 估值过滤：0 < PE_TTM < 50，0 < PB < 10，DV_TTM > 0
3. 个GU MACD金叉过滤
4. 双重排序：股息率降序 + 波动率升序 → 综合得分
5. 返回 TOP_N 股票
"""
import sqlite3
import pandas as pd
import numpy as np
from typing import Optional, Dict, List

DB_PATH = "D:/tu-shareData/astock_daily.db"


class DividendLowVolSelector:
    """
    红利低波选股器
    用法：
        selector = DividendLowVolSelector(config, data_fetcher)
        stocks = selector.select_stocks(date="20240102")
    """

    def __init__(self, config: dict, data_fetcher=None):
        """
        Args:
            config: DIVIDEND_LOW_VOL 配置节
            data_fetcher: DataFetcher 实例（可选，用于兼容接口）
        """
        self.config = config
        self.data_fetcher = data_fetcher

        # 选股参数
        self.date = config.get("date", "20240102")
        self.stock_pool = config.get("stock_pool", "zz800")
        self.top_n = config.get("top_n", 5)

        # 因子阈值
        self.dividend_yield_min = config.get("dividend_yield_min", 0.0)   # 股息率下限（dv_ttm）
        self.pe_min = config.get("pe_min", 0)
        self.pe_max = config.get("pe_max", 50)
        self.pb_min = config.get("pb_min", 0)
        self.pb_max = config.get("pb_max", 10)
        self.volatility_window = config.get("volatility_window", 120)
        self.macd_filter = config.get("macd_filter", True)  # 是否启用MACD金叉过滤

        # 输出
        self.output_dir = config.get("output_dir", "data/results/dividend_low_vol")
        self.output_file = config.get("output_file", "dividend_low_vol_{date}.csv")

        # 缓存（用于股票池查询）
        self._hs300_cache = None
        self._zz500_cache = None
        self._zz800_cache = None
        self._zz1000_cache = None

    # ------------------------------------------------------------------ #
    #  内部工具
    # ------------------------------------------------------------------ #
    def _get_conn(self):
        return sqlite3.connect(DB_PATH)

    def _get_zz800_constituents(self) -> set:
        """获取中证800成分股（缓存）"""
        if self._zz800_cache is not None:
            return self._zz800_cache
        conn = self._get_conn()
        df = pd.read_sql_query(
            "SELECT ts_code FROM index_constituent WHERE index_code = '000906.SH'",
            conn,
        )
        conn.close()
        self._zz800_cache = set(df["ts_code"].tolist()) if len(df) > 0 else set()
        return self._zz800_cache

    def _get_hs300_constituents(self) -> set:
        """获取沪深300成分股（缓存）"""
        if self._hs300_cache is not None:
            return self._hs300_cache
        conn = self._get_conn()
        df = pd.read_sql_query(
            "SELECT ts_code FROM index_constituent WHERE index_code = '000300.SH'",
            conn,
        )
        conn.close()
        self._hs300_cache = set(df["ts_code"].tolist()) if len(df) > 0 else set()
        return self._hs300_cache

    def _get_zz500_constituents(self) -> set:
        """获取中证500成分股（缓存）"""
        if self._zz500_cache is not None:
            return self._zz500_cache
        conn = self._get_conn()
        df = pd.read_sql_query(
            "SELECT ts_code FROM index_constituent WHERE index_code = '000905.SH'",
            conn,
        )
        conn.close()
        self._zz500_cache = set(df["ts_code"].tolist()) if len(df) > 0 else set()
        return self._zz500_cache

    def _get_zz1000_constituents(self) -> set:
        """获取中证1000成分股（缓存）"""
        if self._zz1000_cache is not None:
            return self._zz1000_cache
        conn = self._get_conn()
        df = pd.read_sql_query(
            "SELECT ts_code FROM index_constituent WHERE index_code = '000852.SH'",
            conn,
        )
        conn.close()
        self._zz1000_cache = set(df["ts_code"].tolist()) if len(df) > 0 else set()
        return self._zz1000_cache

    def _get_all_stocks(self) -> set:
        """获取全A股股票列表（从 stock_basic 表）"""
        conn = self._get_conn()
        df = pd.read_sql_query(
            "SELECT ts_code FROM stock_basic WHERE ts_code NOT LIKE '%.BJ'",
            conn,
        )
        conn.close()
        return set(df["ts_code"].tolist()) if len(df) > 0 else set()

    def _get_prev_trading_day(self, date_str: str) -> str:
        """查询 date_str 之前最近一个交易日"""
        conn = self._get_conn()
        row = pd.read_sql_query(
            "SELECT MAX(trade_date) AS d FROM daily WHERE trade_date < ?",
            conn, params=(date_str,),
        )
        conn.close()
        if row.iloc[0, 0] is not None:
            return str(row.iloc[0, 0])
        return date_str

    def _calc_volatility(self, ts_code: str, trade_date: str, window: int = None) -> Optional[float]:
        """计算年化波动率 = std(日收益率) * sqrt(252)"""
        if window is None:
            window = self.volatility_window
        conn = self._get_conn()
        df = pd.read_sql_query(
            """
            SELECT close FROM daily
            WHERE ts_code = ? AND trade_date <= ?
            ORDER BY trade_date DESC
            LIMIT ?
            """,
            conn, params=(ts_code, trade_date, window + 1),
        )
        conn.close()
        if len(df) < max(int(window * 0.6), 60):
            return None
        closes = df["close"].values[::-1]  # 升序
        returns = np.diff(closes) / closes[:-1]
        return float(np.std(returns) * np.sqrt(252))

    def _is_macd_golden(self, ts_code: str, trade_date: str, is_index: bool = False) -> bool:
        """判断 MACD 是否金叉（DIF > DEA）"""
        table = "index_daily" if is_index else "daily"
        conn = self._get_conn()
        df = pd.read_sql_query(
            f"""
            SELECT close FROM {table}
            WHERE ts_code = ? AND trade_date <= ?
            ORDER BY trade_date DESC
            LIMIT 200
            """,
            conn, params=(ts_code, trade_date),
        )
        conn.close()
        if len(df) < 26 + 9:
            return False
        closes = df["close"].values[::-1]
        ema_fast = pd.Series(closes).ewm(span=12, adjust=False).mean().values
        ema_slow = pd.Series(closes).ewm(span=26, adjust=False).mean().values
        dif = ema_fast - ema_slow
        dea = pd.Series(dif).ewm(span=9, adjust=False).mean().values
        return float(dif[-1]) > float(dea[-1])

    # ------------------------------------------------------------------ #
    #  选股主函数
    # ------------------------------------------------------------------ #
    def select_stocks(self, date: str = None) -> pd.DataFrame:
        """
        执行红利低波选股，返回 DataFrame（含 ts_code, score, dv_ttm, volatility 等列）

        Args:
            date: 选股基准日期（YYYYMMDD），默认用 self.date

        Returns:
            DataFrame，按 score 降序排列，已截取 TOP_N
        """
        if date is None:
            date = self.date

        # 向前填充：如果 daily_basic 当天无数据，取前一个交易日
        conn = self._get_conn()
        actual_date = date
        while True:
            cnt = pd.read_sql_query(
                "SELECT COUNT(*) AS n FROM daily_basic WHERE trade_date = ?",
                conn, params=(actual_date,),
            ).iloc[0, 0]
            if cnt > 0:
                break
            prev = pd.read_sql_query(
                "SELECT MAX(trade_date) AS d FROM daily_basic WHERE trade_date < ?",
                conn, params=(actual_date,),
            )
            if prev.iloc[0, 0] is None:
                conn.close()
                return pd.DataFrame()
            actual_date = str(prev.iloc[0, 0])
        if actual_date != date:
            print(f"  [红利低波] daily_basic 无 {date} 数据，使用 {actual_date} 替代")

        # ---- 第一步：估值 + 分红过滤 ----
        df = pd.read_sql_query(
            """
            SELECT ts_code, pe_ttm, pb, dv_ttm, total_mv
            FROM daily_basic
            WHERE trade_date = ?
              AND pe_ttm > ? AND pe_ttm < ?
              AND pb > ? AND pb < ?
              AND dv_ttm > ?
              AND total_mv > 0
            """,
            conn, params=(actual_date,
                         self.pe_min, self.pe_max,
                         self.pb_min, self.pb_max,
                         self.dividend_yield_min),
        )

        # ---- 第二步：股票池过滤 ----
        if self.stock_pool == "all":
            # 全A股模式：不过滤股票池
            print(f"  [红利低波] 股票池: 全A股（不限制）")
            print(f"  [红利低波] 估值过滤后：{len(df)} 只")
        else:
            # 指数成分股模式：根据 stock_pool 动态选择
            if self.stock_pool == "hs300":
                constituents = self._get_hs300_constituents()
                pool_name = "沪深300"
            elif self.stock_pool == "zz500":
                constituents = self._get_zz500_constituents()
                pool_name = "中证500"
            elif self.stock_pool == "zz1000":
                constituents = self._get_zz1000_constituents()
                pool_name = "中证1000"
            else:  # "zz800" 或其他
                constituents = self._get_zz800_constituents()
                pool_name = "中证800"

            df = df[df["ts_code"].isin(constituents)]
            print(f"  [红利低波] 估值+{pool_name}过滤后：{len(df)} 只")

        # ---- 第三步：排除 ST ----
        st_df = pd.read_sql_query(
            "SELECT ts_code FROM stock_basic WHERE name LIKE '%ST%' OR name LIKE '%*%'",
            conn,
        )
        conn.close()
        if len(st_df) > 0:
            st_set = set(st_df["ts_code"].tolist())
            df = df[~df["ts_code"].isin(st_set)]
        print(f"  [红利低波] 排除ST后：{len(df)} 只")

        if df.empty:
            return pd.DataFrame()

        # ---- 第四步：MACD金叉过滤（个GU）----
        if self.macd_filter:
            print(f"  [红利低波] MACD金叉过滤中...")
            macd_ok = []
            for ts_code in df["ts_code"]:
                if self._is_macd_golden(ts_code, actual_date, is_index=False):
                    macd_ok.append(ts_code)
            df = df[df["ts_code"].isin(macd_ok)]
            print(f"  [红利低波] MACD金叉过滤后：{len(df)} 只")

        if df.empty:
            return pd.DataFrame()

        # ---- 第五步：计算波动率 ----
        print(f"  [红利低波] 计算 {len(df)} 只股票波动率（窗口={self.volatility_window}）...")
        vol_dict = {}
        for ts_code in df["ts_code"]:
            vol = self._calc_volatility(ts_code, actual_date)
            if vol is not None:
                vol_dict[ts_code] = vol
        df = df[df["ts_code"].isin(set(vol_dict.keys()))]
        df["volatility"] = df["ts_code"].map(vol_dict)
        print(f"  [红利低波] 波动率计算完成：{len(df)} 只")

        if df.empty:
            return pd.DataFrame()

        # ---- 第六步：双重排序 ----
        df["dv_rank"] = df["dv_ttm"].rank(pct=True, ascending=False)   # 股息率越高越好
        df["vol_rank"] = df["volatility"].rank(pct=True, ascending=True)  # 波动率越低越好
        df["score"] = (df["dv_rank"] + df["vol_rank"]) / 2

        # ---- 第七步：获取股票名称 ----
        conn = self._get_conn()
        name_map = {}
        for ts_code in df["ts_code"].tolist():
            r = pd.read_sql_query(
                "SELECT name FROM stock_basic WHERE ts_code = ? LIMIT 1",
                conn, params=(ts_code,),
            )
            name_map[ts_code] = r.iloc[0, 0] if len(r) > 0 else ts_code
        conn.close()
        df["name"] = df["ts_code"].map(name_map)

        # ---- 第八步：排序 + 截取 TOP N ----
        df = df.sort_values("score", ascending=False).head(self.top_n)
        df["code"] = df["ts_code"].str.extract(r"(\d{6})", expand=False)

        # 格式化输出：代码 + 名称
        stocks_list = ', '.join([f"{row['ts_code']}({row['name']})" for _, row in df.iterrows()])
        print(f"  [红利低波] 最终选出 {len(df)} 只：{stocks_list}")
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    #  导出 CSV（与 value_stock_selector 接口保持一致）
    # ------------------------------------------------------------------ #
    def export_to_csv(self, df: pd.DataFrame, filename: str = None, output_dir: str = None) -> str:
        """保存选股结果到 CSV，返回文件路径"""
        if filename is None:
            filename = self.output_file.format(date=self.date)
        if output_dir is None:
            output_dir = self.output_dir

        import os
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False, encoding="utf-8-sig")
        print(f"  [红利低波] 结果已保存：{filepath}")
        return filepath
