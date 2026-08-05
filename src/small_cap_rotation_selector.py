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
   - [可选] 基本面过滤 fundamental_filter：最近年报(快照日及之前) eps>0 即盈利（防壳/亏损，默认关）。
    注：底层利润表 income 本地库为空，改用已填充 20万+ 行的 fina_indicator。eps>0 等价于"净利润>0"意图；
        关键防御：财务数据缺失的股票保留(无法证伪即不排除)，仅剔除"有数据且 eps<=0"者。
   - [可选·A档] 质量门禁 quality_filter（默认关，与 fundamental_filter 互斥时优先）：
        roe>0 & bps>0(未资不抵债) & debt_to_assets<70(非高杠杆雷) & ocfps>0(经营现金流为正)。
        对应视频《小盘股为什么跑出超额》避雷#2(基本面有雷)/#3(只看市值不看质量)；缺失数据保留。
   - [可选·B档] 成长倾斜 growth_tilt（默认关）：在"最小市值桶(hold_count*growth_pool_mult, 默认3倍)"内
        按 (netprofit_yoy>0 优先, roe 降序) 重排取前 N —— size 为底 + 成长/质量增强(视频称成长溢价为超额主因)。
3. 选股：按流通市值 circ_mv 升序 → 取最小 N 只（默认 7）满仓等权持有。
   - 流通市值前向填充：daily_basic 有零散日期缺口，取 <= D 最近有效 circ_mv。
4. 无前视（手册陷阱1）：select_stocks(snapshot_date) 用 D-1 的快照排序选股，
   回测引擎在 D（次日/本周二）开盘成交，信号整体前移 1 天。
"""

import os
import sys
import sqlite3
from datetime import datetime, timedelta

import numpy as np

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
        quality_filter=False,
        growth_tilt=False,
        growth_pool_mult=3,
        vol_filter=False,
        vol_window=60,
        vol_quantile=0.95,
        order="ASC",
    ):
        if pool_mode not in POOL_MODES:
            raise ValueError(f"未知 pool_mode={pool_mode!r}，可选: {list(POOL_MODES)}")
        if order not in ("ASC", "DESC"):
            raise ValueError(f"order 仅支持 ASC/DESC，收到 {order!r}")
        preset = POOL_MODES[pool_mode]
        self.hold_count = hold_count          # 最终持有只数
        # selection_pool / pool_offset：优先用显式参数，否则取 pool_mode 预设
        self.selection_pool = selection_pool if selection_pool is not None else preset["pool"]
        self.pool_offset = pool_offset if pool_offset is not None else preset["offset"]
        self.min_avg_amount_k = min_avg_amount_k  # 日均成交额下限(千元)
        self.fundamental_filter = fundamental_filter  # 可选基本面过滤(eps>0)
        # A档 质量门禁升级：roe>0 & bps>0(未资不抵债) & debt_to_assets<70 & ocfps>0
        #   —— 直接对应视频"避雷#2 基本面有雷 / #3 只看市值不看质量"；数据缺失保留(不证伪不排除)。
        #   与 fundamental_filter 互斥时以 quality_filter 优先(它是 eps>0 的强化超集)。
        self.quality_filter = quality_filter
        # B档 成长倾斜：在"最小市值桶(hold_count*growth_pool_mult)"内按 netprofit_yoy>0 优先 + roe 降序
        #   重排后取前 N —— size 为底 + 成长/质量增强(视频称成长溢价为超额主因)。
        self.growth_tilt = growth_tilt
        self.growth_pool_mult = max(1, int(growth_pool_mult))
        # 维度3·极端波动过滤（默认关）：近 vol_window 日收益率方差(point-in-time)，
        #   剔除最高 vol_quantile 分位的票——文章"质量过滤"明确点名"波动特别极端"需剔除，
        #   否则"超额收益"可能只是承担了极端波动风险的伪超额。方差与标准差单调，用方差排序即可，
        #   规避 SQLite 无 sqrt 依赖。数据不足者保留(无法证伪不排除)。
        self.vol_filter = vol_filter
        self.vol_window = max(10, int(vol_window))
        self.vol_quantile = min(0.99, max(0.5, float(vol_quantile)))
        # 剔除已退市股（仅保留 stock_basic 中当前上市的票）→ 用于量化幸存者偏差
        self.exclude_delisted = exclude_delisted
        self.pool_mode = pool_mode            # 选股宇宙模式
        self.universe = preset["universe"]    # cyb / all
        self.order = order                    # 排序方向：ASC=最小市值优先 / DESC=最大市值优先(用于分位组对照的大桶)

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

        # need_fin：只要开启 基本面(eps) / A档质量 / B档成长倾斜 任一，就 LEFT JOIN fina_indicator。
        need_fin = self.fundamental_filter or self.quality_filter or self.growth_tilt

        # 波动率过滤（维度3·剔除极端波动）：近 vol_window 日收益率方差(point-in-time <= snapshot)。
        # 方差与标准差单调，用方差排序剔除最高 vol_quantile 即可（避免 SQLite 无 sqrt 依赖）。
        vol_select = ""
        if self.vol_filter:
            vol_select = (
                ", (SELECT AVG(r*r) - AVG(r)*AVG(r) FROM ("
                "   SELECT (close*1.0/pre_close - 1) AS r FROM daily vd"
                f"   WHERE vd.ts_code = d.ts_code AND vd.trade_date <= ? AND vd.pre_close > 0"
                f"   ORDER BY vd.trade_date DESC LIMIT {int(self.vol_window)}"
                ")) AS _vol"
            )

        fund_join = ""
        fund_cond = ""
        extra_cols = ""
        if need_fin:
            # 取「快照日当日及之前」最新一份年报(end_date 以 1231 结尾)，防前视(end_date<=?)。
            # 用 fina_indicator(本地库已填充 20万+ 行，roe/bps/debt_to_assets/ocfps 填充率 90%~100%)。
            # 关键防御：财务数据缺失的股票 fin.ts_code IS NULL → 保留(无法证伪就不排除)，
            #           绝不因底层缺数据而把候选池清零。
            fund_join = """
            LEFT JOIN fina_indicator fin ON fin.ts_code = d.ts_code
               AND fin.end_date = (
                   SELECT MAX(end_date) FROM fina_indicator
                   WHERE ts_code = d.ts_code AND end_date LIKE '%1231' AND end_date <= ?
               )
            """
            # A档质量门禁优先于旧的 eps 过滤(它是 eps>0 的强化超集)：
            #   roe>0(盈利) & bps>0(未资不抵债) & debt_to_assets<70(非高杠杆雷) & ocfps>0(经营现金流为正)
            if self.quality_filter:
                fund_cond = ("AND (fin.ts_code IS NULL OR "
                             "(fin.roe > 0 AND fin.bps > 0 AND fin.debt_to_assets < 70 AND fin.ocfps > 0))")
            elif self.fundamental_filter:
                fund_cond = "AND (fin.ts_code IS NULL OR fin.eps > 0)"
            # B档成长倾斜需要 netprofit_yoy / roe 参与 Python 层重排
            if self.growth_tilt:
                extra_cols = ", fin.netprofit_yoy, fin.roe"

        # 剔除已退市股：改用 INNER JOIN（只保留 stock_basic 中当前上市的票）
        join_sql = "INNER JOIN" if self.exclude_delisted else "LEFT JOIN"

        # 选股宇宙：按 pool_mode 切换（涨跌停逻辑在 Python 层按代码前缀自动区分，此处无需区分）
        if self.universe == "cyb":
            universe_cond = "AND (d.ts_code LIKE '300%' OR d.ts_code LIKE '301%')"
        else:  # all：全市场最小N只，剔除北交所/老三板/科创板(688对散户不友好)
            universe_cond = ("AND d.ts_code NOT LIKE '8%' AND d.ts_code NOT LIKE '4%' "
                             "AND d.ts_code NOT LIKE '920%' AND d.ts_code NOT LIKE '688%'")

        # ── 参数按 SQL 文本出现顺序收集（保证 ? 顺序 === SQL 中 ? 出现顺序）──
        params = []
        if self.vol_filter:
            params.append(snapshot_date)            # vol 子查询 trade_date <=
        params.append(snapshot_date)                # circ_mv 子查询 trade_date <=
        if need_fin:
            params.append(snapshot_date)            # fina_indicator end_date <=
        params.append(snapshot_date)                # d.trade_date = ?
        params.append(ipo_thr)                      # s.list_date <= ?
        params.append(PRICE_FLOOR)                  # d.close >= ?
        params.append(PRICE_CEIL)                   # d.close <= ?
        params.append(snapshot_date)                # 流动性子查询 trade_date <=
        params.append(self.min_avg_amount_k)        # 流动性门槛 >= ?
        params.append(self.selection_pool)          # LIMIT ?
        params.append(self.pool_offset)             # OFFSET ?

        q = f"""
        SELECT d.ts_code, db.circ_mv, d.close, d.pre_close{vol_select}{extra_cols}
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
        ORDER BY db.circ_mv {self.order}
        LIMIT ? OFFSET ?
        """
        rows = conn.execute(q, params).fetchall()
        conn.close()

        # ── 维度3·极端波动过滤：剔除 _vol(近 vol_window 日收益率方差) 最高的 vol_quantile 分位 ──
        # 先滤波动，再在"低波动小票"里做成长倾斜——确保超额来自小市值本身而非极端波动伪超额。
        if self.vol_filter and rows:
            vols = [r[4] for r in rows if r[4] is not None]
            if vols:
                thr = float(np.quantile(vols, self.vol_quantile)) if len(vols) > 1 else max(vols)
                rows = [r for r in rows if r[4] is None or r[4] <= thr]

        # ── B档 成长倾斜：size 为底 + 成长/质量增强 ──
        # rows 已按 circ_mv ASC 排序 → 前 M=hold_count*growth_pool_mult 只即"最小市值桶"。
        # 桶内按 (netprofit_yoy>0 优先, roe 降序) 重排，使 size 仍是硬门槛(只在最小的一批里挑)，
        # 但优先命中"正增长/高质量"的成长股(视频称成长溢价为超额主因)。数据缺失者排最后但不剔除。
        if self.growth_tilt and rows:
            M = max(self.hold_count * self.growth_pool_mult, self.hold_count)
            bucket = rows[:M]
            # vol 列插入后(索引4)，fina 的 netprofit_yoy/roe 整体后移 1 位
            _base = 5 if self.vol_filter else 4

            def _gscore(r):
                npy = r[_base] if len(r) > _base and r[_base] is not None else -1e9  # netprofit_yoy
                roe = r[_base + 1] if len(r) > _base + 1 and r[_base + 1] is not None else -1e9  # roe
                return (1 if npy > 0 else 0, roe)

            ranked = sorted(bucket, key=_gscore, reverse=True)
        else:
            ranked = rows

        # 无前视 + 涨跌停判定（按板块前缀区分幅度，仅用开盘/收盘价，不参考盘中）：
        # 对候选序列逐个检查，跳过涨停(买不进)/跌停(卖不出续持)，取前 hold_count 只。
        codes = []
        for r in ranked:
            ts_code, close, pre_close = r[0], r[2], r[3]
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
