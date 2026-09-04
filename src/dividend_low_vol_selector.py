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

        # ── 红利质量复合（意愿维度）──
        self.payout_ratio_max = config.get("payout_ratio_max", 1.00)   # 分红比例上限（>1 不可持续）
        self.payout_ratio_min = config.get("payout_ratio_min", 0.00)   # 分红比例下限（0=不限）
        self.use_dividend_growth = config.get("use_dividend_growth", True)
        self.div_growth_min_years = config.get("div_growth_min_years", 0)
        self.div_growth_min_yoy = config.get("div_growth_min_yoy", 0.0)

        # ── 红利质量复合（文档《红利个股DIY》能力维度）──
        self.forward_yield_min = config.get("forward_yield_min", 0.036)
        self.consecutive_div_years_min = config.get("consecutive_div_years_min", 5)
        self.div_drop_max = config.get("div_drop_max", 0.30)
        self.ocf_positive_years = config.get("ocf_positive_years", 5)
        self.ocf_to_profit_min = config.get("ocf_to_profit_min", 0.20)
        self.roe_stability_max_drop = config.get("roe_stability_max_drop", 0.20)
        self.lev_debt_to_assets_max = config.get("lev_debt_to_assets_max", 0.0)
        self.quality_mode = config.get("quality_mode", "hard")  # "hard" 硬过滤 | "soft" 软打分 | "official" 官方编制法
        self.quality_soft_weight = config.get("quality_soft_weight", 0.40)  # 软打分中质量维度权重(默认40%)
        # ── 官方编制法（中证红利低波 930955）参数 ──
        self.final_top_n = config.get("final_top_n", self.top_n)  # 最终持仓数（区别于缓冲 top_n）
        self.yield_keep_frac = config.get("yield_keep_frac", 0.75)  # 3y均息排名保留前 N%
        self.vol_keep_frac = config.get("vol_keep_frac", 0.50)      # 1y低波保留最低 N%
        self.official_bank_cap = config.get("official_bank_cap", 0)  # 银行行业上限(0=不限制)
        self.industry_cap = config.get("industry_cap", 0)            # 通用行业上限(0=不限制；落地版用2做行业中性)

        self._div_cache = {}     # ts_code -> 分红档案(dict)
        self._fina_cache = {}    # ts_code -> 财务档案(dict)

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

    # [已移除] _get_kcb_cyb_constituents：科创板+创业板(高风险) 股票池已按需求删除
    # （科创板投资门槛对散户不友好，本回测平台统一屏蔽北交所(.BJ)）

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

    def _get_dividend_profile(self, ts_code: str, asof: str):
        """从 dividend_detail 表算分红档案（asof 时点之前已实施的）。
        返回 dict：yoy(最新同比) / streak(连续增长年数) / consec(连续分红年数)
                  / max_drop(单年最大降幅) / fwd3(近三年平均每股股利)。
        无表/无数据则全部为 None/0（降级，不抛出）。"""
        if ts_code in self._div_cache:
            return self._div_cache[ts_code]
        prof = {"yoy": None, "streak": None, "consec": 0,
                "max_drop": None, "fwd3": None}
        try:
            conn = self._get_conn()
            df = pd.read_sql_query(
                "SELECT end_date, ex_date, cash_div FROM dividend_detail "
                "WHERE ts_code = ? AND div_proc = '实施' AND cash_div > 0 "
                "AND ex_date <= ?",
                conn, params=(ts_code, asof),
            )
            conn.close()
            if len(df) >= 1:
                df["year"] = df["end_date"].str[:4]
                # 每年取最大每股股利（多次实施取最高），按年升序
                yr = df.groupby("year")["cash_div"].max().sort_index()
                # 连续分红年数（从最新往前数，现金股利>0 才算）
                consec = 0
                for i in range(len(yr) - 1, -1, -1):
                    if yr.iloc[i] > 0:
                        consec += 1
                    else:
                        break
                prof["consec"] = consec
                if len(yr) >= 2:
                    # 单年最大降幅 = max((上年-本年)/上年)
                    drops = []
                    for i in range(1, len(yr)):
                        prev = yr.iloc[i - 1]
                        if prev > 0:
                            drops.append((prev - yr.iloc[i]) / prev)
                    prof["max_drop"] = max(drops) if drops else 0.0
                    # 最新同比与连续增长年数
                    latest, prev = yr.iloc[-1], yr.iloc[-2]
                    prof["yoy"] = (latest - prev) / prev if prev > 0 else None
                    streak = 0
                    for i in range(len(yr) - 1, 0, -1):
                        if yr.iloc[i] > yr.iloc[i - 1]:
                            streak += 1
                        else:
                            break
                    prof["streak"] = streak
                # 前瞻股息率分子：近三年平均每股股利
                last3 = yr.tail(3).values
                prof["fwd3"] = float(np.mean(last3)) if len(last3) >= 1 else None
        except Exception:
            pass  # 表不存在等 → 降级
        self._div_cache[ts_code] = prof
        return prof

    def _get_fina_profile(self, ts_code: str, asof: str):
        """从 fina_indicator（最新年报 1231）算能力维度档案。
        返回 dict：ocf_pos_years(经营现金流为正连续年数) / ocf_to_profit(ocfps/eps)
                  / roe_stable(ROE 3年最大同比降幅) / debt_to_assets(最新负债率)。"""
        if ts_code in self._fina_cache:
            return self._fina_cache[ts_code]
        prof = {"ocf_pos_years": 0, "ocf_to_profit": None,
                "roe_stable": None, "debt_to_assets": None}
        try:
            conn = self._get_conn()
            df = pd.read_sql_query(
                "SELECT end_date, roe, ocfps, eps, debt_to_assets FROM fina_indicator "
                "WHERE ts_code = ? AND end_date LIKE '%1231' AND end_date <= ? "
                "ORDER BY end_date DESC LIMIT 5",
                conn, params=(ts_code, asof),
            )
            conn.close()
            if len(df) > 0:
                # 经营现金流为正连续年数（从最新往前）
                ocf_pos = 0
                for v in df["ocfps"].values:
                    if v is not None and v > 0:
                        ocf_pos += 1
                    else:
                        break
                prof["ocf_pos_years"] = ocf_pos
                # 盈余质量 = 每股经营现金流 / 每股收益（最新年）
                ocf = df["ocfps"].iloc[0]
                eps = df["eps"].iloc[0]
                if ocf is not None and eps is not None and eps > 0:
                    prof["ocf_to_profit"] = float(ocf) / float(eps)
                prof["debt_to_assets"] = df["debt_to_assets"].iloc[0]
                # ROE 稳定性：近 3 年最大同比降幅
                roes = df["roe"].dropna().values
                if len(roes) >= 2:
                    drops = []
                    for i in range(1, min(len(roes), 3)):
                        if roes[i - 1] is not None and roes[i - 1] > 0:
                            drops.append((roes[i - 1] - roes[i]) / roes[i - 1])
                    prof["roe_stable"] = max(drops) if drops else 0.0
        except Exception:
            pass
        self._fina_cache[ts_code] = prof
        return prof

    def _quality_soft_rank(self, df: pd.DataFrame) -> pd.Series:
        """软打分：把六维质量算成 0~1 百分排名后取均值（NaN 填 0.5 中性）。
        仅当 quality_mode=='soft' 且配置了质量维度时调用。"""
        dims = [
            ("fwd_yield", False),        # 前瞻股息率：越高越好
            ("div_consec_years", False),  # 连续分红年数：越多越好
            ("div_max_drop", True),       # 单年最大降幅：越低越好
            ("ocf_pos_years", False),     # 经营现金流转正年数：越多越好
            ("ocf_to_profit", False),     # 盈余质量 ocf/eps：越高越好
            ("roe_max_drop", True),       # ROE 3年最大降幅：越低越好
        ]
        ranks = []
        for col, lower_better in dims:
            if col not in df.columns:
                continue
            s = df[col].astype(float)
            if lower_better:
                s = -s
            med = s.median()
            s = s.fillna(med if pd.notna(med) else 0.5)
            ranks.append(s.rank(pct=True, ascending=False))
        if not ranks:
            return pd.Series(0.5, index=df.index)
        return sum(ranks) / len(ranks)

    def _cap_banks(self, df: pd.DataFrame, bank_set, cap: int) -> pd.DataFrame:
        """官方编制法可选：银行行业最多保留 cap 只（取波动率最低者），缓解集中度。"""
        if cap is None or cap <= 0 or bank_set is None or len(bank_set) == 0:
            return df
        bank_rows, other_rows = [], []
        for _, r in df.iterrows():
            if r["ts_code"] in bank_set:
                bank_rows.append(r)
            else:
                other_rows.append(r)
        bank_rows.sort(key=lambda r: r["volatility"])
        kept = bank_rows[:cap] + other_rows
        return pd.DataFrame(kept).reset_index(drop=True) if kept else df

    def _cap_industry(self, df: pd.DataFrame, cap: int, sort_key: str = "fwd_yield") -> pd.DataFrame:
        """官方编制法可选：单行业最多保留 cap 只，行业中性。
        与 _cap_banks 互补：前者只限制银行，本方法对全行业生效（落地版用 cap=2 防银行/公用独大）。
        行业取自 stock_basic.industry（首次调用缓存）。

        sort_key（2026-09-03 新增，opt-in，默认 fwd_yield = 旧行为）：
          "fwd_yield"  → 按股息率**降序**取前 cap（历史行为）
          "volatility" → 按波动率**升序**取前 cap（🔴 官方 930955 口径）
        🔴 为什么这很关键：官方 930955 的选样是「股息率前 300 → **波动率升序**取前 100」，
           最后一段筛子的排序键是**波动率**（慢变量，年度间高度自相关 → 留存率高、换手低）；
           我们原实现在波动率筛出的候选池上又按 **fwd_yield 重排**取前 top_n，
           把排序键换成了股息率（快变量，随股价与分红剧烈变化）→ 每期大翻、换手高。
        """
        if cap is None or cap <= 0 or len(df) == 0:
            return df
        if sort_key not in ("fwd_yield", "volatility"):
            raise ValueError(f"sort_key 只能是 fwd_yield / volatility，收到 {sort_key!r}")
        if not hasattr(self, "_ind_map") or self._ind_map is None:
            conn = self._get_conn()
            im = pd.read_sql_query("SELECT ts_code, industry FROM stock_basic", conn)
            conn.close()
            self._ind_map = {
                str(r["ts_code"]): (str(r["industry"]) if pd.notna(r["industry"]) else "其他")
                for _, r in im.iterrows()
            }
        d = df.copy()
        d["ts_code"] = d["ts_code"].astype(str)
        d["_ind"] = d["ts_code"].map(self._ind_map).fillna("其他")
        d = d.sort_values(sort_key, ascending=(sort_key != "fwd_yield"))
        capped = [g.head(cap) for _, g in d.groupby("_ind")]
        out = pd.concat(capped).sort_values(sort_key, ascending=(sort_key != "fwd_yield")) if capped else d
        return out.reset_index(drop=True)

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

        # ---- 第一步之附：红利质量·分红比例过滤 ----
        # 分红比例 = 股息率(%) / 100 × 市盈率 = 每股股息 / 每股收益（免新增数据）
        # 作用：排除"分红超过盈利"(>100%，不可持续)与"几乎不分红"的伪红利
        df["payout_ratio"] = df["dv_ttm"] / 100.0 * df["pe_ttm"]
        if self.payout_ratio_max > 0:
            before = len(df)
            df = df[df["payout_ratio"] <= self.payout_ratio_max]
            print(f"  [红利质量] 分红比例≤{self.payout_ratio_max:.0%} 过滤：{before}→{len(df)} 只")
        if self.payout_ratio_min > 0:
            before = len(df)
            df = df[df["payout_ratio"] >= self.payout_ratio_min]
            print(f"  [红利质量] 分红比例≥{self.payout_ratio_min:.0%} 过滤：{before}→{len(df)} 只")

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

        # 全局屏蔽科创板(688开头)与北交所(.BJ后缀)：投资门槛对散户不友好，本平台统一剔除
        df = df[~df["ts_code"].str.endswith(".BJ")]

        # ---- 第二步之附：红利质量·能力+意愿+前瞻（视频《红利个股DIY》框架）----
        _any_new = (self.forward_yield_min > 0 or self.consecutive_div_years_min > 0
                    or self.div_drop_max > 0 or self.ocf_positive_years > 0
                    or self.ocf_to_profit_min > 0 or self.roe_stability_max_drop > 0
                    or self.lev_debt_to_assets_max > 0)
        if (_any_new or self.quality_mode == "official") and len(df) > 0:
            if self.quality_mode == "official":
                # 全A 预筛：仅保留「有 >=3 次实施分红(且 asof 前已除权)」的股票。
                # 既符合官方"连续3年分红"门禁精神，又避免对海量无分红数据股做无效建档
                # （dividend_detail 全市场仅覆盖 ~589 只，而全A dv_ttm>0 有 ~3530 只）。
                dd = pd.read_sql_query(
                    "SELECT ts_code FROM dividend_detail "
                    "WHERE div_proc='实施' AND cash_div>0 AND ex_date <= ? "
                    "GROUP BY ts_code HAVING COUNT(1) >= 3",
                    conn, params=(actual_date,))
                dd_set = set(str(x) for x in dd["ts_code"].tolist())
                _before_dd = len(df)
                df = df[df["ts_code"].isin(dd_set)]
                print(f"  [红利低波] 官方·全A预筛(>=3次实施分红): {_before_dd}→{len(df)} 只")
            codes = df["ts_code"].tolist()
            ph = ",".join("?" for _ in codes)
            close_df = pd.read_sql_query(
                f"""SELECT ts_code, close FROM (
                      SELECT ts_code, close,
                             ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY trade_date DESC) rn
                      FROM daily WHERE trade_date <= ? AND ts_code IN ({ph})
                    ) WHERE rn = 1""",
                conn, params=(actual_date, *codes))
            close_map = dict(zip(close_df["ts_code"], close_df["close"]))
            # 金融行业（银行/证券/保险/信托/多元金融）豁免杠杆过滤（会计口径差异）
            fin_df = pd.read_sql_query(
                f"SELECT ts_code FROM stock_basic WHERE ts_code IN ({ph}) "
                f"AND (industry LIKE '%银行%' OR industry LIKE '%证券%' "
                f"OR industry LIKE '%保险%' OR industry LIKE '%信托%' "
                f"OR industry LIKE '%多元金融%')",
                conn, params=codes)
            fin_set = set(fin_df["ts_code"].tolist())
            bank_df = pd.read_sql_query(
                f"SELECT ts_code FROM stock_basic WHERE ts_code IN ({ph}) AND industry LIKE '%银行%'",
                conn, params=codes)
            bank_set = set(bank_df["ts_code"].tolist())
            dvp = {c: self._get_dividend_profile(c, actual_date) for c in codes}
            fnp = {c: self._get_fina_profile(c, actual_date) for c in codes}
            df["fwd_yield"] = df["ts_code"].map(
                lambda c: (dvp[c]["fwd3"] / close_map[c])
                if (dvp[c]["fwd3"] and close_map.get(c)) else None)
            df["div_consec_years"] = df["ts_code"].map(lambda c: dvp[c]["consec"])
            df["div_max_drop"] = df["ts_code"].map(lambda c: dvp[c]["max_drop"])
            df["ocf_pos_years"] = df["ts_code"].map(lambda c: fnp[c]["ocf_pos_years"])
            df["ocf_to_profit"] = df["ts_code"].map(lambda c: fnp[c]["ocf_to_profit"])
            df["roe_max_drop"] = df["ts_code"].map(lambda c: fnp[c]["roe_stable"])
            df["debt_to_assets"] = df["ts_code"].map(lambda c: fnp[c]["debt_to_assets"])

            if self.quality_mode == "soft":
                # 软打分：保留全部候选，质量维度留待第七步并入综合分（不剔除）
                print(f"  [红利低波] 红利质量·软打分（不剔除，{len(df)} 只参与评分）")
            elif self.quality_mode == "official":
                # 官方编制法：载入分红/财务档案但不做硬剔除，留待第七步分段筛选
                print(f"  [红利低波] 官方编制法：载入分红/财务档案（{len(df)} 只候选，不硬剔除）")
            else:
                if self.forward_yield_min > 0:
                    before = len(df)
                    df = df[df["fwd_yield"].isna() | (df["fwd_yield"] >= self.forward_yield_min)]
                    print(f"  [红利质量] 前瞻股息率≥{self.forward_yield_min:.1%} 过滤：{before}→{len(df)} 只")
                if self.consecutive_div_years_min > 0:
                    before = len(df)
                    df = df[df["div_consec_years"].fillna(0) >= self.consecutive_div_years_min]
                    print(f"  [红利质量] 连续分红≥{self.consecutive_div_years_min}年 过滤：{before}→{len(df)} 只")
                if self.div_drop_max > 0:
                    before = len(df)
                    df = df[df["div_max_drop"].fillna(0) <= self.div_drop_max]
                    print(f"  [红利质量] 单年降幅≤{self.div_drop_max:.0%} 过滤：{before}→{len(df)} 只")
                if self.ocf_positive_years > 0:
                    before = len(df)
                    df = df[df["ocf_pos_years"].fillna(0) >= self.ocf_positive_years]
                    print(f"  [红利质量] 经营现金流为正≥{self.ocf_positive_years}年 过滤：{before}→{len(df)} 只")
                if self.ocf_to_profit_min > 0:
                    before = len(df)
                    df = df[df["ocf_to_profit"].fillna(0) >= self.ocf_to_profit_min]
                    print(f"  [红利质量] 盈余质量(ocfps/eps)≥{self.ocf_to_profit_min:.0%} 过滤：{before}→{len(df)} 只")
                if self.roe_stability_max_drop > 0:
                    before = len(df)
                    df = df[df["roe_max_drop"].fillna(0) <= self.roe_stability_max_drop]
                    print(f"  [红利质量] ROE 3年降幅≤{self.roe_stability_max_drop:.0%} 过滤：{before}→{len(df)} 只")
                if self.lev_debt_to_assets_max > 0:
                    before = len(df)
                    df = df[df["ts_code"].isin(fin_set)
                           | (df["debt_to_assets"].fillna(0) <= self.lev_debt_to_assets_max)]
                    print(f"  [红利质量] 资产负债率≤{self.lev_debt_to_assets_max:.0f}%(金融豁免) 过滤：{before}→{len(df)} 只")
                print(f"  [红利低波] 红利质量门禁后：{len(df)} 只")
        if df.empty:
            return pd.DataFrame()

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
        if self.macd_filter and self.quality_mode != "official":
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

        # ---- 第六步：红利质量·分红增长（意愿维度）----
        if self.use_dividend_growth:
            grows, streaks = [], []
            for code in df["ts_code"]:
                p = self._get_dividend_profile(code, actual_date)
                grows.append(p["yoy"])
                streaks.append(p["streak"])
            df["div_growth_yoy"] = grows
            df["div_growth_years"] = streaks
            # 硬筛（默认均关闭，仅排序）
            if self.div_growth_min_years > 0:
                before = len(df)
                df = df[df["div_growth_years"].apply(lambda x: x is not None and x >= self.div_growth_min_years)]
                print(f"  [红利质量] 连续增长≥{self.div_growth_min_years}年 过滤：{before}→{len(df)} 只")
            if self.div_growth_min_yoy > 0:
                before = len(df)
                df = df[df["div_growth_yoy"].apply(lambda x: x is not None and x >= self.div_growth_min_yoy)]
                print(f"  [红利质量] 同比增长≥{self.div_growth_min_yoy:.0%} 过滤：{before}→{len(df)} 只")
            # 排序：有数据按增长降序；无数据填中位数(不拖累)
            has = df["div_growth_yoy"].notna().sum()
            fill = df["div_growth_yoy"].median() if has > 0 else 0.0
            # 口径同上：分红增长越高越好 → rank 越大（2026-08-01 随 score 方向统一修正）
            df["div_growth_rank"] = df["div_growth_yoy"].fillna(fill).rank(pct=True, ascending=True)
            print(f"  [红利质量] 分红增长维度：{has}/{len(df)} 只有数据")
            if df.empty:
                return pd.DataFrame()

        # ---- 第七步：官方编制法 OR 双重排序 + 红利质量复合 ----
        if self.quality_mode == "official":
            # 官方编制法（中证红利低波 930955）：
            # 门禁(连续分红≥3年 & 3年移动平均股息率>0) → 3y均息降序取前N% → 1y波动率升序取最低N% → TOP_N
            before = len(df)
            df = df[df["div_consec_years"].fillna(0) >= 3]
            df = df[df["fwd_yield"].fillna(0) > 0]
            if len(df) >= self.final_top_n:
                df = df.sort_values("fwd_yield", ascending=False)
                keep_y = max(self.final_top_n, int(round(len(df) * self.yield_keep_frac)))
                df = df.head(keep_y)
                df = df.sort_values("volatility", ascending=True)
                keep_v = max(self.final_top_n, int(round(len(df) * self.vol_keep_frac)))
                df = df.head(keep_v)
                if self.official_bank_cap and self.official_bank_cap > 0:
                    df = self._cap_banks(df, bank_set, self.official_bank_cap)
            df["score"] = -df["volatility"]  # 占位：最终按波动率升序取 TOP_N
            print(f"  [红利低波] 官方编制法：门禁(连分≥3 & 3y均息>0) {before}→{len(df)} 只")
        else:
            # ---- 第七步：双重排序 + 红利质量复合（legacy / soft）----
            # ⚠️ 2026-08-01 修复方向性 bug（与 dogs_of_market_selector 同源问题）：
            #   原写法 dv_rank=rank(ascending=False) / vol_rank=rank(ascending=True) 让
            #   「高股息」「低波动」都拿到【最小】pct 值 → score 越小越优；
            #   但第八步按 sort_values(ascending=False) 取头部 → 实际选出低股息+高波动的反向组合。
            #   实测 2024-01-02 hs300：建设银行(息5.94%/波0.198)被排最后，九州通(息3.35%/波0.68)排第二。
            # 统一口径：所有 rank 一律「越大越好」，与降序取头部一致。
            df["dv_rank"] = df["dv_ttm"].rank(pct=True, ascending=True)        # 股息率越高 → rank 越大
            df["vol_rank"] = df["volatility"].rank(pct=True, ascending=False)  # 波动率越低 → rank 越大
            if self.quality_mode == "soft" and _any_new:
                df["quality_rank"] = self._quality_soft_rank(df)
                w = self.quality_soft_weight  # 质量维度权重
                if self.use_dividend_growth and "div_growth_rank" in df.columns and df["div_growth_yoy"].notna().any():
                    df["score"] = (1 - w) * (df["dv_rank"] + df["vol_rank"] + df["div_growth_rank"]) / 3 + w * df["quality_rank"]
                    print(f"  [红利质量] 软打分复合 = {1-w:.0%}(股息率+低波+增长) + {w:.0%}质量")
                else:
                    df["score"] = (1 - w) * (df["dv_rank"] + df["vol_rank"]) / 2 + w * df["quality_rank"]
                    print(f"  [红利质量] 软打分复合 = {1-w:.0%}(股息率+低波) + {w:.0%}质量")
            elif self.use_dividend_growth and "div_growth_rank" in df.columns and df["div_growth_yoy"].notna().any():
                df["score"] = (df["dv_rank"] + df["vol_rank"] + df["div_growth_rank"]) / 3
                print(f"  [红利质量] 复合打分 = (股息率 + 低波 + 分红增长) / 3")
            else:
                df["score"] = (df["dv_rank"] + df["vol_rank"]) / 2
                if self.use_dividend_growth:
                    print(f"  [红利质量] dividend_detail 无数据，降级为 (股息率 + 低波) / 2")

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
