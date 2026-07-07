"""
狗股策略（Dogs of the Market）A股版选股器
============================================
参考：UP主Jim视频《凯利公式——只看一个指标，每年操作一次》

核心理念（来自《Beating the Dow》1991）：
- 高股息率本身不是选股标准，它只是一个信号——提醒你关注这只股票为什么便宜了
- 连续分红 + 低市净率 = 真正被错杀的公司
- 不需要复杂财报分析，用股息率定位跌多了的股票

选股逻辑：
1. 指数成分股过滤
2. 股息率高于市场中位数（被错杀信号）
3. 市净率低于市场中位数（排除股价虚高）
4. 连续3年以上分红记录（排除硬撑分红陷阱）
5. 排除ST
6. 双重排序：股息率降序 + PB升序 → 综合得分
7. 返回 TOP_N

与红利低波策略的差异：
- 红利低波：高股息 + 低波动 → 防御型配置
- 狗股策略：高股息 + 低PB + 持续分红 → 价值回归预期（均值回归）
"""
import sqlite3
import pandas as pd
import numpy as np
from typing import Optional

DB_PATH = "D:/tu-shareData/astock_daily.db"


class DogsOfMarketSelector:
    """
    狗股策略选股器

    用法：
        selector = DogsOfMarketSelector(config, data_fetcher)
        stocks = selector.select_stocks(date="20240102")
    """

    def __init__(self, config: dict, data_fetcher=None):
        """
        Args:
            config: DOGS_OF_MARKET 配置节
            data_fetcher: DataFetcher 实例（可选）
        """
        self.config = config
        self.data_fetcher = data_fetcher

        # 选股参数
        self.date = config.get("date", "20240102")
        self.stock_pool = config.get("stock_pool", "zz800")
        self.top_n = config.get("top_n", 5)

        # 因子阈值
        self.dividend_yield_percentile = config.get("dividend_yield_percentile", 0.5)
        self.pb_percentile = config.get("pb_percentile", 0.5)
        self.min_dividend_years = config.get("min_dividend_years", 3)
        self.dividend_lookback_years = config.get("dividend_lookback_years", 3)
        self.volatility_window = config.get("volatility_window", 120)

        # 输出
        self.output_dir = config.get("output_dir", "data/results/dogs_of_market")
        self.output_file = config.get("output_file", "dogs_of_market_{date}.csv")

        # 缓存（按 (pool, date) 区分，支持真正的时点成分股查询）
        self._constituent_cache = {}
        self._constituent_warned = False

    # ------------------------------------------------------------------ #
    #  内部工具
    # ------------------------------------------------------------------ #
    def _get_conn(self):
        return sqlite3.connect(DB_PATH)

    def _get_constituents(self, pool_name: str, trade_date: str = None) -> set:
        """通用成分股获取（带缓存，真正的「时点」查询）

        Args:
            pool_name: 股票池名称 (hs300/zz500/zz800/zz1000/all)
            trade_date: 调仓日期（YYYYMMDD），取该日期之前最近一次发布的成分股快照

        时点语义：index_constituent 每个 trade_date 是一份完整快照。
            取 MAX(trade_date) <= 调仓日 的那一份，即为当时真实成分股，
            可避免「用未来才知道的成分股」(前视偏差) 与「只含幸存者」(生存偏差)。
            若数据库仅含单时点快照(如只有最新一期)，则该查询会退化为「每年都用
            同一份当前名单」——此时打印偏差警告，结果仅供参考。
        """
        cache_key = f"{pool_name}|{trade_date}"
        if cache_key in self._constituent_cache:
            return self._constituent_cache[cache_key]

        pool_to_index = {
            "hs300": "000300.SH",
            "zz500": "000905.SH",
            "zz800": "000906.SH",
            "zz1000": "000852.SH",
            "zz2000": "932000.SH",
        }
        index_code = pool_to_index.get(pool_name)
        if index_code is None:
            # all模式：全市场（排除北交所）
            conn = self._get_conn()
            df = pd.read_sql_query(
                "SELECT ts_code FROM stock_basic WHERE ts_code NOT LIKE '%.BJ'",
                conn,
            )
            conn.close()
            result = set(df["ts_code"].tolist()) if len(df) > 0 else set()
        else:
            conn = self._get_conn()
            # 1) 该调仓日之前最近的一次快照日期
            #    CAST 确保 trade_date(INTEGER 列) 与参数(可能含字符串)比较时类型一致
            snap_row = pd.read_sql_query(
                "SELECT MAX(CAST(trade_date AS INTEGER)) AS d FROM index_constituent "
                "WHERE index_code = ? AND CAST(trade_date AS INTEGER) <= CAST(? AS INTEGER)",
                conn, params=(index_code, trade_date or "99999999"),
            )
            # ⚠️ 关键：snap 来自 pandas → numpy.int64，直接作为 sqlite 参数会绑定失败
            #    （numpy 类型无法被 sqlite3 正确转换，导致 = 匹配 0 行，回退全市场）。
            #    必须转成原生 int 再用于后续查询参数。
            snap = int(snap_row.iloc[0, 0]) if (len(snap_row) > 0 and snap_row.iloc[0, 0] is not None) else None
            look_ahead = False
            if snap is None:
                # 没有任何 <= 调仓日的快照 → 取最早一期（实为前视，需警告）
                snap_row = pd.read_sql_query(
                    "SELECT MIN(CAST(trade_date AS INTEGER)) AS d FROM index_constituent WHERE index_code = ?",
                    conn, params=(index_code,),
                )
                snap = int(snap_row.iloc[0, 0]) if (len(snap_row) > 0 and snap_row.iloc[0, 0] is not None) else None
                look_ahead = True

            result = set()
            if snap is not None:
                df = pd.read_sql_query(
                    "SELECT ts_code FROM index_constituent WHERE index_code = ? AND CAST(trade_date AS INTEGER) = CAST(? AS INTEGER)",
                    conn, params=(index_code, snap),
                )
                result = set(df["ts_code"].tolist()) if len(df) > 0 else set()

            # 2) 偏差检测：若整个表只有 1 个时点快照，时点查询退化 → 警告
            nsnap = pd.read_sql_query(
                "SELECT COUNT(DISTINCT CAST(trade_date AS INTEGER)) AS n FROM index_constituent WHERE index_code = ?",
                conn, params=(index_code,),
            ).iloc[0, 0]
            conn.close()

            if not self._constituent_warned:
                if nsnap <= 1:
                    print(f"  [⚠️ 偏差警告] index_constituent 仅含单时点快照({snap})，"
                          f"所有年度将使用同一份「当前成分股」→ 存在生存偏差/前视偏差，"
                          f"回测收益偏高，仅供参考。\n            → 建议运行 "
                          f"download_index_constituents.py 回补历史成分股快照。")
                    self._constituent_warned = True
                elif look_ahead:
                    print(f"  [⚠️ 偏差警告] 调仓日 {trade_date} 之前无成分股快照，"
                          f"已退化为使用最早快照({snap})，含前视偏差。")
                    self._constituent_warned = True

        self._constituent_cache[cache_key] = result
        return result

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

    def _check_dividend_history(self, ts_code: str, trade_date: str, years: int = 3) -> bool:
        """
        检查股票是否连续 years 年都有分红记录
        每年检查两个时间点（4月底 + 8月底），任一满足即算该年有分红
        避免分红在下半年的股票被漏判
        """
        trade_year = int(trade_date[:4])

        # 每年检查两个时间点：4月底（年报出完）+ 8月底（中报出完）
        check_months = []
        for y in range(years):
            year = trade_year - y
            check_months.append((f"{year:04d}0430", f"{year:04d}0830"))
        # 当年使用选股日期本身
        check_months[0] = (trade_date, trade_date)

        conn = self._get_conn()
        try:
            for apr_date, aug_date in check_months:
                # 查4月底前后是否有 dv_ttm > 0
                found = False
                for check_date in [apr_date, aug_date]:
                    row = pd.read_sql_query(
                        "SELECT MAX(trade_date) AS d FROM daily_basic WHERE trade_date <= ?",
                        conn, params=(check_date,),
                    )
                    if row.iloc[0, 0] is None:
                        continue
                    actual_date = str(row.iloc[0, 0])
                    dv = pd.read_sql_query(
                        "SELECT dv_ttm FROM daily_basic WHERE ts_code = ? AND trade_date = ?",
                        conn, params=(ts_code, actual_date),
                    )
                    if not dv.empty and dv.iloc[0, 0] is not None and dv.iloc[0, 0] > 0:
                        found = True
                        break
                if not found:
                    return False
            return True
        finally:
            conn.close()

    def _calc_volatility(self, ts_code: str, trade_date: str) -> Optional[float]:
        """计算年化波动率（用于日志展示，不参与排序）"""
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
        closes = df["close"].values[::-1]
        returns = np.diff(closes) / closes[:-1]
        return float(np.std(returns) * np.sqrt(252))

    # ------------------------------------------------------------------ #
    #  选股主函数
    # ------------------------------------------------------------------ #
    def select_stocks(self, date: str = None) -> pd.DataFrame:
        """
        执行狗股策略选股

        Args:
            date: 选股基准日期（YYYYMMDD）

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
            print(f"  [狗股策略] daily_basic 无 {date} 数据，使用 {actual_date} 替代")

        # ---- 第一步：确定股票池范围 ----
        pool_stocks = None
        if self.stock_pool == "all":
            print(f"  [狗股策略] 股票池: 全A股（不限制）")
        else:
            constituents = self._get_constituents(self.stock_pool, trade_date=actual_date)
            pool_names = {"hs300": "沪深300", "zz500": "中证500",
                          "zz800": "中证800", "zz1000": "中证1000"}
            pool_name = pool_names.get(self.stock_pool, self.stock_pool)
            if constituents:
                pool_stocks = sorted(constituents)
                print(f"  [狗股策略] {pool_name}成分股：{len(pool_stocks)} 只")
            else:
                print(f"  [狗股策略] 警告: 无{pool_name}成分股数据，使用全市场")
                pool_stocks = None

        # ---- 第二步：在股票池范围内获取估值数据 ----
        if pool_stocks is not None:
            # 只查中证800范围内的股票
            placeholders = ",".join(["?"] * len(pool_stocks))
            df = pd.read_sql_query(
                f"""
                SELECT ts_code, pe_ttm, pb, dv_ttm, total_mv
                FROM daily_basic
                WHERE trade_date = ?
                  AND ts_code IN ({placeholders})
                  AND pe_ttm > 0
                  AND pb > 0
                  AND dv_ttm > 0
                  AND total_mv > 0
                """,
                conn, params=[actual_date] + pool_stocks,
            )
        else:
            # 全A股模式
            df = pd.read_sql_query(
                """
                SELECT ts_code, pe_ttm, pb, dv_ttm, total_mv
                FROM daily_basic
                WHERE trade_date = ?
                  AND pe_ttm > 0
                  AND pb > 0
                  AND dv_ttm > 0
                  AND total_mv > 0
                """,
                conn, params=(actual_date,),
            )

        if df.empty:
            conn.close()
            print(f"  [狗股策略] 无满足基础条件的股票")
            return pd.DataFrame()

        print(f"  [狗股策略] 基础条件过滤后：{len(df)} 只（PE>0, PB>0, DV_TTM>0）")

        if df.empty:
            conn.close()
            return pd.DataFrame()

        # ---- 第三步：股息率高于市场中位数 ----
        dv_median = df["dv_ttm"].median()
        dv_threshold = df["dv_ttm"].quantile(1 - self.dividend_yield_percentile)
        df = df[df["dv_ttm"] >= dv_threshold]
        print(f"  [狗股策略] 股息率>={dv_threshold:.2f}%（中位数={dv_median:.2f}%）：{len(df)} 只")

        if df.empty:
            conn.close()
            return pd.DataFrame()

        # ---- 第四步：市净率低于市场中位数 ----
        pb_median = df["pb"].median()
        pb_threshold = df["pb"].quantile(self.pb_percentile)
        df = df[df["pb"] <= pb_threshold]
        print(f"  [狗股策略] PB<={pb_threshold:.2f}（中位数={pb_median:.2f}）：{len(df)} 只")

        if df.empty:
            conn.close()
            return pd.DataFrame()

        # ---- 第五步：排除ST ----
        st_df = pd.read_sql_query(
            "SELECT ts_code FROM stock_basic WHERE name LIKE '%ST%' OR name LIKE '%*%'",
            conn,
        )
        if len(st_df) > 0:
            st_set = set(st_df["ts_code"].tolist())
            df = df[~df["ts_code"].isin(st_set)]
        print(f"  [狗股策略] 排除ST后：{len(df)} 只")

        if df.empty:
            conn.close()
            return pd.DataFrame()

        conn.close()

        # ---- 第六步：连续分红检查 ----
        print(f"  [狗股策略] 检查连续{self.min_dividend_years}年分红记录（{len(df)} 只）...")
        div_ok = []
        for ts_code in df["ts_code"]:
            if self._check_dividend_history(ts_code, actual_date, self.min_dividend_years):
                div_ok.append(ts_code)
        df = df[df["ts_code"].isin(div_ok)]
        print(f"  [狗股策略] 连续分红过滤后：{len(df)} 只")

        if df.empty:
            return pd.DataFrame()

        # ---- 第七步：计算波动率（仅用于日志展示）----
        vol_found = 0
        vol_list = []
        for ts_code in df["ts_code"]:
            vol = self._calc_volatility(ts_code, actual_date)
            if vol is not None:
                vol_list.append(ts_code)
                vol_found += 1
        print(f"  [狗股策略] 波动率计算完成：{vol_found} 只")

        # ---- 第八步：双重排序 ----
        # 狗股策略排序：股息率越高越好（均值回归潜力大）+ PB越低越好（价值洼地）
        df["dv_rank"] = df["dv_ttm"].rank(pct=True, ascending=False)  # 股息率降序
        df["pb_rank"] = df["pb"].rank(pct=True, ascending=True)        # PB升序（低PB更好）
        df["score"] = (df["dv_rank"] + df["pb_rank"]) / 2

        # ---- 第九步：获取股票名称 ----
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

        # ---- 第十步：排序 + 截取 TOP N ----
        df = df.sort_values("score", ascending=False).head(self.top_n)
        df["code"] = df["ts_code"].str.extract(r"(\d{6})", expand=False)

        # 格式化输出
        stocks_list = ', '.join([f"{row['ts_code']}({row['name']})" for _, row in df.iterrows()])
        print(f"  [狗股策略] 最终选出 {len(df)} 只：{stocks_list}")
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    #  导出 CSV
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
        print(f"  [狗股策略] 结果已保存：{filepath}")
        return filepath
