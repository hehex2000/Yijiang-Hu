# -*- coding: utf-8 -*-
"""
小市值轮动策略 —— 选股引擎 (v2, 手册对齐版)
==========================================
规则（基于《小市值量化策略投研手册》2026-07-08 精确化）：

1. 股票池（宇宙）：全市场（沪深两市，剔除北交所/老三板）在快照日 D 有正常交易的股票，
   按流通市值升序取最小约 2000 只作为「中证2000 风格」候选宇宙，再持有其中最小 N 只。
   - 用「D 当天有 daily 行且成交额>0」界定宇宙 = point-in-time，且**通过 LEFT JOIN
     stock_basic 天然包含已退市股**（daily 里 5796 只 vs stock_basic 5534 只，
     262 只退市/遗留股在内）。这是规避幸存者偏差（手册陷阱2）的关键：
     之前用 INNER JOIN 会把这 262 只排除，等于只看活下来的票。
   - 不使用 index_constituent(399006/932000)：本地库指数成分仅 1 个 2026 快照，套用会前视，
     故采用「point-in-time 全市场按流通市值排序取最小 2000」重建宇宙，等价于中证2000 方法学。
2. 过滤链（全部用快照日 D 的数据，杜绝前视）：
   - 剔除停牌股：D 日 amount<=0（无成交 = 停牌/无流动）
   - 剔除 ST / *ST：仅当 stock_basic 有记录时按 name 过滤（退市股 name 为 NULL，保留）
   - 剔除上市 < 1 年次新股：仅当 stock_basic 有 list_date 时过滤（YYYYMMDD 阈值）
   - 剔除股价 < 2 元（仙股）或 > 100 元（绝对低价）
   - 剔除当日涨停 / 跌停（日期感知 ±10%/±20%）
   - 剔除日均成交额 < 3000 万（手册陷阱7 流动性硬门槛，20 日均值，单位千元→30000）
   - [可选] 基本面过滤：最近年报(快照日及之前) eps>0 即盈利（防壳/亏损公司，手册 step3，默认关）。
    注：底层利润表 income 本地库为空(下载脚本仅填充 fina_indicator 且 token 为占位符)，
        故此处改用已填充 20万+ 行的 fina_indicator。eps>0 等价于原"净利润>0"意图；
        营收>5亿 因 fina_indicator 无 revenue 字段无法校验，已省略。
        关键防御：财务数据缺失的股票保留(无法证伪即不排除)，仅剔除"有数据且 eps<=0"者，
        绝不因底层表为空而把整个候选池清零。
3. 选股：按流通市值 circ_mv 升序 → 取最小 N 只（默认 7）满仓等权持有。
   - 流通市值前向填充：daily_basic 有零散日期缺口，取 <= D 最近有效 circ_mv。
4. 无前视（手册陷阱1）：select_stocks(snapshot_date) 用 D-1 的快照排序选股，
   回测引擎在 D（次日/本周二）开盘成交，信号整体前移 1 天。
"""

import os
import sys
import sqlite3
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA

DB_PATH = DATA.get("local_db_path", "D:/tu-shareData/astock_daily.db")

# 流动性门槛：日均成交额 >= 3000 万。daily.amount 单位千元 → 30000。
MIN_AVG_AMOUNT_K = 30000.0
# 仙股下限 / 绝对高价上限（元）
PRICE_FLOOR = 2.0
PRICE_CEIL = 100.0


def _prefix_of(ts_code):
    """取代码前 3 位板块前缀。"""
    return ts_code[:3] if len(ts_code) >= 3 else ts_code


def limit_up_ratio(ts_code, trade_date):
    """涨停价比例（close/pre_close 上限）。按板块区分涨跌幅：
    - 创业板 300/301：2020-08-24 注册制改革前 ±10%，之后 ±20%
    - 科创板 688：±20%（注册制，始终）
    - 主板 600/601/603/605/000/001/002/003：±10%
    - 北交所 8xx/4xx/920：±30%（本策略已排除，兜底用 ±10%）
    """
    p = _prefix_of(ts_code)
    if p in ("300", "301"):
        return 1.10 if trade_date < "20200824" else 1.20
    if p == "688":
        return 1.20
    return 1.10  # 主板及兜底


def limit_down_ratio(ts_code, trade_date):
    """跌停价比例（close/pre_close 下限）。"""
    p = _prefix_of(ts_code)
    if p in ("300", "301"):
        return 0.90 if trade_date < "20200824" else 0.80
    if p == "688":
        return 0.80
    return 0.90  # 主板及兜底


# 选股宇宙模式：涨跌停逻辑在 Python 层按 ts_code 前缀自动区分（主板±10%/创业板·科创板±20%），
# 此处 pool_mode 仅控制「选股宇宙范围」与候选池大小/偏移。
#   pool  = 候选池只数（按流通市值升序取最小 N 只）
#   offset= 跳过最小的 N 只（用于剔除最微盘尾，模拟中证1000「市值排名801-1800」）
# 重要：本策略「取候选池内流通市值最小 N 只」，故合并两个指数(如1000+2000)对 min-选择是空操作
#       ——合并池的最小 N 只恒等于其中较小那一个指数，因此只保留两种有意义的"全市场"口径：
#       含微盘尾(zz2000=最小2000只) 与 剔除微盘尾(zz1000=排名801-1800)。
# 注：本地库 index_constituent 仅中证2000的1个2026快照(前视不可用)，故「中证XXX风格」均为
#     按 circ_mv 排名的 point-in-time 近似，比套官方成分股更适合回测。
POOL_MODES = {
    "cyb":     {"desc": "创业板 (300/301)",                            "universe": "cyb",     "pool": 2000, "offset": 0},
    "zz2000":  {"desc": "中证2000风格 (流通市值最小2000只·含微盘尾·剔除北交所)", "universe": "all",     "pool": 2000, "offset": 0},
    "zz1000":  {"desc": "中证1000风格 (市值排名801-1800·剔除微盘尾·剔除北交所)", "universe": "all",     "pool": 1000, "offset": 800},
}
POOL_DESC = {k: v["desc"] for k, v in POOL_MODES.items()}


class SmallCapRotationSelector:
    def __init__(
        self,
        hold_count=7,
        selection_pool=None,
        min_avg_amount_k=MIN_AVG_AMOUNT_K,
        fundamental_filter=False,
        exclude_delisted=False,
        pool_mode="zz2000",
        pool_offset=None,
    ):
        if pool_mode not in POOL_MODES:
            raise ValueError(f"未知 pool_mode={pool_mode!r}，可选: {list(POOL_MODES)}")
        preset = POOL_MODES[pool_mode]
        self.hold_count = hold_count          # 最终持有只数
        # selection_pool / pool_offset：优先用显式参数，否则取 pool_mode 预设
        self.selection_pool = selection_pool if selection_pool is not None else preset["pool"]
        self.pool_offset = pool_offset if pool_offset is not None else preset["offset"]
        self.min_avg_amount_k = min_avg_amount_k  # 日均成交额下限(千元)
        self.fundamental_filter = fundamental_filter  # 可选基本面过滤
        # 剔除已退市股（仅保留 stock_basic 中当前上市的票）→ 用于量化幸存者偏差
        self.exclude_delisted = exclude_delisted
        self.pool_mode = pool_mode            # 选股宇宙模式
        self.universe = preset["universe"]    # cyb / all

    def select_stocks(self, snapshot_date):
        """返回快照日 snapshot_date 选出的 ts_code 列表（流通市值最小 hold_count 只）。

        无前视约定：snapshot_date 应为「实际成交日的前一个交易日」，由回测引擎传入，
        使选股信号整体前移 1 天（手册陷阱1 自检：信号 shift(1)）。

        Args:
            snapshot_date: 'YYYYMMDD' 字符串，库内真实交易日。
        """
        conn = sqlite3.connect(DB_PATH)
        dt = datetime.strptime(snapshot_date, "%Y%m%d") - timedelta(days=365)
        ipo_thr = dt.strftime("%Y%m%d")   # 次新股阈值：上市日 <= D-1年

        fund_join = ""
        fund_cond = ""
        # ── 参数顺序约定（务必与下方 SQL 的 ? 一一对应）──
        # 无基本面过滤(默认)：9 个 ?
        #   1) circ_mv 子查询 trade_date<=?       → snapshot_date
        #   2) d.trade_date = ?                  → snapshot_date
        #   3) s.list_date <= ?                  → ipo_thr
        #   4) d.close >= ?                      → PRICE_FLOOR
        #   5) d.close <= ?                      → PRICE_CEIL
        #   6) 流动性子查询 trade_date<=?         → snapshot_date
        #   7) 流动性门槛 >= ?                   → min_avg_amount_k
        #   8) LIMIT ?                           → selection_pool
        #   9) OFFSET ?                          → pool_offset
        # 开启基本面过滤：多 1 个 ?（fina_indicator end_date<=?，位于 circ_mv 子查询之后、d.trade_date 之前）
        #   即第 2 个 ? 变为 fina_indicator 的 end_date<=?，原 d.trade_date 顺延为第 3 个 ?
        if self.fundamental_filter:
            # 防壳/亏损公司：取「快照日当日及之前」最新一份年报(end_date 以 1231 结尾)，要求 eps>0(盈利)。
            # 用 fina_indicator(本地库已填充 20万+ 行)，不用空表 income。
            # 关键防御：财务数据缺失的股票 fin.ts_code IS NULL → 保留(无法证伪就不排除)，
            #           仅剔除「有数据且 eps<=0」者，绝不因底层表为空而把整个候选池清零(旧 bug)。
            # 另加 end_date <= ? 防前视：只用快照日已发布的年报，不用未来财报。
            fund_join = """
            LEFT JOIN fina_indicator fin ON fin.ts_code = d.ts_code
               AND fin.end_date = (
                   SELECT MAX(end_date) FROM fina_indicator
                   WHERE ts_code = d.ts_code AND end_date LIKE '%1231' AND end_date <= ?
               )
            """
            fund_cond = "AND (fin.ts_code IS NULL OR fin.eps > 0)"
            params = [
                snapshot_date,            # 1) circ_mv 子查询 trade_date<=
                snapshot_date,            # 2) fina_indicator end_date <= (NEW)
                snapshot_date,            # 3) d.trade_date = ?
                ipo_thr,                  # 4) s.list_date <= ?
                PRICE_FLOOR,              # 5) d.close >= ?
                PRICE_CEIL,               # 6) d.close <= ?
                snapshot_date,            # 7) 流动性子查询 trade_date<=?
                self.min_avg_amount_k,    # 8) 流动性门槛 >= ?
                self.selection_pool,      # 9) LIMIT ?
                self.pool_offset,         # 10) OFFSET ?
            ]
        else:
            params = [
                snapshot_date,            # 1) circ_mv 子查询 trade_date<=
                snapshot_date,            # 2) d.trade_date = ?
                ipo_thr,                  # 3) s.list_date <= ?
                PRICE_FLOOR,              # 4) d.close >= ?
                PRICE_CEIL,               # 5) d.close <= ?
                snapshot_date,            # 6) 流动性子查询 trade_date<=?
                self.min_avg_amount_k,    # 7) 流动性门槛 >= ?
                self.selection_pool,      # 8) LIMIT ?
                self.pool_offset,         # 9) OFFSET ?
            ]

        # 剔除已退市股：改用 INNER JOIN（只保留 stock_basic 中当前上市的票）
        join_sql = "INNER JOIN" if self.exclude_delisted else "LEFT JOIN"

        # 选股宇宙：按 pool_mode 切换（涨跌停逻辑在 Python 层按代码前缀自动区分，此处无需区分）
        if self.universe == "cyb":
            universe_cond = "AND (d.ts_code LIKE '300%' OR d.ts_code LIKE '301%')"
        elif self.universe == "cyb_kcb":  # [已移除] 创业板+科创板组合已按需求删除（科创板门槛对散户不友好）
            universe_cond = "AND (d.ts_code LIKE '300%' OR d.ts_code LIKE '301%' OR d.ts_code LIKE '688%')"
        else:  # all：全市场最小N只，剔除北交所/老三板/科创板(688对散户不友好)
            universe_cond = ("AND d.ts_code NOT LIKE '8%' AND d.ts_code NOT LIKE '4%' "
                             "AND d.ts_code NOT LIKE '920%' AND d.ts_code NOT LIKE '688%'")

        q = f"""
        SELECT d.ts_code, db.circ_mv, d.close, d.pre_close
        FROM daily d
        {join_sql} stock_basic s ON d.ts_code = s.ts_code
        JOIN daily_basic db ON db.ts_code = d.ts_code
            AND db.trade_date = (
                SELECT MAX(trade_date) FROM daily_basic
                WHERE ts_code = d.ts_code AND trade_date <= ?
            )
        {fund_join}
        WHERE d.trade_date = ?
          {universe_cond}
          AND d.amount > 0                                        -- 剔除停牌(无成交)
          AND (s.name IS NULL OR s.name NOT LIKE '%ST%')          -- 剔除ST(退市股name=NULL则保留)
          AND (s.list_date IS NULL OR s.list_date <= ?)           -- 剔除次新股(有记录才判)
          AND d.close >= ? AND d.close <= ?                       -- 2元<=价<=100元
          AND db.circ_mv > 0                                      -- 有流通市值
          AND (SELECT AVG(amount) FROM (
                  SELECT amount FROM daily
                  WHERE ts_code = d.ts_code AND trade_date <= ?
                  ORDER BY trade_date DESC LIMIT 20
              )) >= ?                                             -- 日均成交额门槛
          {fund_cond}
        ORDER BY db.circ_mv ASC
        LIMIT ? OFFSET ?
        """
        rows = conn.execute(q, params).fetchall()
        conn.close()

        # 无前视 + 涨跌停判定（按板块前缀区分幅度，仅用开盘/收盘价，不参考盘中）：
        # 对「流通市值最小的候选池」逐个检查，跳过涨停(买不进)/跌停(卖不出续持)，取前 hold_count 只。
        codes = []
        for r in rows:
            ts_code, circ_mv, close, pre_close = r[0], r[1], r[2], r[3]
            if pre_close and pre_close > 0:
                up_r = limit_up_ratio(ts_code, snapshot_date)
                down_r = limit_down_ratio(ts_code, snapshot_date)
                if close / pre_close >= up_r - 1e-9:    # 涨停，买不进
                    continue
                if close / pre_close <= down_r + 1e-9:  # 跌停，卖不出(续持)
                    continue
            codes.append(ts_code)
            if len(codes) >= self.hold_count:
                break
        return codes
