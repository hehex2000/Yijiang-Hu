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


# ════════════════════════════════════════════════════════════════
#  统一价值选股逻辑（破净价值）
#  供 [1] 选股+回测（ValueStockSelector）与 [6] 月度调仓（run_monthly_rebalance）
#  共用同一份实现，消除重复逻辑与口径分歧。
# ════════════════════════════════════════════════════════════════
_VSEL_POOL_INDEX = {
    "hs300": "000300.SH",
    "zz500": "000905.SH",
    "zz800": "000906.SH",
    "zz1000": "000852.SH",
}
_VSEL_DB_PATH = None


def _vsel_db_path():
    global _VSEL_DB_PATH
    if _VSEL_DB_PATH is None:
        try:
            from config import DATA
            p = DATA.get("local_db_path")
        except Exception:
            p = None
        if not p:
            p = "D:/tu-shareData/astock_daily.db"
        _VSEL_DB_PATH = p
    return _VSEL_DB_PATH


def _vsel_conn():
    return sqlite3.connect(_vsel_db_path())


def _vsel_pool_ts_set(pool, conn, trade_date):
    """返回股票池 ts_code 集合（时点快照）；None 表示全A股不过滤。"""
    if pool in ("all", None):
        return None
    idx = _VSEL_POOL_INDEX.get(pool)
    if not idx:
        return None
    # 时点快照：取 <= trade_date 最近一期，避免前视/生存偏差
    snap = pd.read_sql_query(
        "SELECT MAX(CAST(trade_date AS INTEGER)) AS d FROM index_constituent "
        "WHERE index_code=? AND CAST(trade_date AS INTEGER)<=CAST(? AS INTEGER)",
        conn, params=(idx, trade_date),
    )
    sd = snap.iloc[0, 0]
    if sd is not None:
        df = pd.read_sql_query(
            "SELECT ts_code FROM index_constituent "
            "WHERE index_code=? AND ts_code NOT LIKE '688%' AND CAST(trade_date AS INTEGER)=CAST(? AS INTEGER)",
            conn, params=(idx, int(sd)),   # int() 避免 numpy 类型绑定失败
        )
        return set(df["ts_code"].tolist())
    # 退化：无快照则取全部历史成分股
    df = pd.read_sql_query(
        "SELECT ts_code FROM index_constituent WHERE index_code=? AND ts_code NOT LIKE '688%'", conn, params=(idx,)
    )
    return set(df["ts_code"].tolist())


def select_value_stocks(trade_date, top_n=5, stock_pool="zz800"):
    """
    统一价值选股（破净价值）逻辑，[1] 选股+回测 与 [6] 月度调仓 共用：

        1. 0 < PE_TTM < 30
        2. 0 < PB < 1.0（破净）
        3. ROE > 8%（fina_indicator 最新可得财报，ann_date<=trade_date 防前视）
        4. 流动比率 >= 1.2（同上）
        5. 股票池成分股（时点快照过滤）
        6. 按 PB 升序取前 top_n 只

    返回 DataFrame：ts_code, code(6位), name, pb, pe_ttm, roe, current_ratio, selection_date
    """
    conn = _vsel_conn()

    # daily_basic 可用性回退：若当日无数据，向前取最近交易日
    actual_date = trade_date
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
            return pd.DataFrame(columns=["ts_code", "code", "name", "pb", "pe_ttm",
                                         "roe", "current_ratio", "selection_date"])
        actual_date = str(prev.iloc[0, 0])

    # ---- 估值初筛 ----
    df = pd.read_sql_query(
        """
        SELECT ts_code, pe_ttm, pb, total_mv
        FROM daily_basic
        WHERE trade_date = ?
          AND pe_ttm > 0 AND pe_ttm < 30
          AND pb > 0 AND pb < 1.0
          AND total_mv > 0
        """,
        conn, params=(actual_date,),
    )
    if df.empty:
        conn.close()
        return pd.DataFrame(columns=["ts_code", "code", "name", "pb", "pe_ttm",
                                     "roe", "current_ratio", "selection_date"])

    # ---- 股票池过滤（时点快照）----
    pool_set = _vsel_pool_ts_set(stock_pool, conn, actual_date)
    if pool_set is not None:
        df = df[df["ts_code"].isin(pool_set)]
    # 全局剔除科创板(688)与北交所(.BJ后缀)，对散户更友好
    df = df[~df["ts_code"].str.startswith("688") & ~df["ts_code"].str.endswith(".BJ")]
    if df.empty:
        conn.close()
        return pd.DataFrame(columns=["ts_code", "code", "name", "pb", "pe_ttm",
                                     "roe", "current_ratio", "selection_date"])

    # ---- ROE>8% 且 流动比率>=1.2（fina_indicator 最新可得财报）----
    cand = df["ts_code"].tolist()
    placeholders = ",".join("?" * len(cand))
    fin = pd.read_sql_query(
        f"""
        SELECT f.ts_code, f.roe, f.current_ratio
        FROM fina_indicator f
        INNER JOIN (
            SELECT ts_code, MAX(end_date) AS mx
            FROM fina_indicator
            WHERE ts_code IN ({placeholders})
              AND end_date <= ?
              AND (ann_date IS NULL OR ann_date <= ?)
            GROUP BY ts_code
        ) m ON f.ts_code = m.ts_code AND f.end_date = m.mx
        """,
        conn, params=cand + [actual_date, actual_date],
    )
    conn.close()

    # ---- 质量门槛（fina_indicator 最新可得财报，防前视）----
    # 分级回退：避免某些时段（如 2018 初）无股票同时满足
    # ROE>8% 与 流动比率>=1.2 时整个选股落空、回测无法运行。
    #   Tier1: ROE>8% 且 流动比率>=1.2   （用户首选口径，质量最高）
    #   Tier2: ROE>8% 或  流动比率>=1.2 （至少满足一项质量门槛）
    #   Tier3: 仅 破净+低PE               （价值兜底，保证有标的可选）
    # 注：fina_indicator 部分历史期次 roe/current_ratio 为 NULL，
    #     视为“未通过该门槛”，自然落入更低 Tier。
    if fin.empty:
        df["roe"] = float("nan")
        df["current_ratio"] = float("nan")
    else:
        df = df.merge(fin[["ts_code", "roe", "current_ratio"]], on="ts_code", how="left")

    t1 = df[(df["roe"] > 8) & (df["current_ratio"] >= 1.2)]
    t2 = df[(df["roe"] > 8) | (df["current_ratio"] >= 1.2)]
    if not t1.empty:
        chosen, tier = t1, 1
    elif not t2.empty:
        chosen, tier = t2, 2
    else:
        chosen, tier = df, 3
    logger.info(f"[价值选股] 质量门槛分级 Tier{tier} "
                f"(双条件 {len(t1)} / 单条件 {len(t2)} / 兜底 {len(df)}) @ {actual_date}")

    # ---- 按 PB 升序取 top_n ----
    if top_n and top_n > 0:
        chosen = chosen.sort_values("pb", ascending=True).head(top_n)
    else:
        chosen = chosen.sort_values("pb", ascending=True)
    df = chosen

    # ---- 补全名称与 6 位代码 ----
    conn = _vsel_conn()
    names = {}
    for tc in df["ts_code"].tolist():
        r = pd.read_sql_query(
            "SELECT name FROM stock_basic WHERE ts_code = ? LIMIT 1",
            conn, params=(tc,),
        )
        names[tc] = r.iloc[0, 0] if len(r) > 0 else tc
    conn.close()
    df["name"] = df["ts_code"].map(names)
    df["code"] = df["ts_code"].str.extract(r"(\d{6})", expand=False)
    df.insert(0, "selection_date", trade_date)
    return df[["ts_code", "code", "name", "pb", "pe_ttm", "roe",
               "current_ratio", "selection_date"]]


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

    # [已移除] _get_kcb_cyb_constituents：科创板+创业板(高风险) 股票池已按需求删除

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
        执行选股（主流程）。

        现统一委托给模块级 select_value_stocks（破净价值逻辑），
        使 [1] 选股+回测 与 [6] 月度调仓 共用同一份价值选股逻辑。
        """
        date = date or self.date
        top_n = top_n if top_n is not None else self.top_n
        return select_value_stocks(date, top_n, self.stock_pool)
    
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
