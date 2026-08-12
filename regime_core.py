# -*- coding: utf-8 -*-
"""
Regime 识别 + β 兜底核心模块（独立，被 run_etf_rotation_v6_merged 的 --regime 开关调用）
=================================================================================

信号（默认，与 run_regime_validate.py 历史验证口径一致，MA_LEN=200）：
  信号1 趋势（沪深300 长期 MA）：
        BULL_trend = 沪深300收盘 > MA(沪深300收盘, MA_LEN)
  信号2 宽度（市场广度）：
        代理口径(proxy, 默认)：8 大指数(沪深300/中证500/中证1000/创业板/科创50/
                              上证50/深成指/中证全指)中 收盘>各自MA20 的占比。
        真实口径(full, 可选)  ：daily 表全部个股 收盘>各自MA20 的占比。
        BULL_breadth = breadth >= 阈值(breadth_thr, 默认0.25)
  合成规则（保守，防御优先）：
        RuleA (AND)       : BULL = 趋势牛 AND 宽度>=0.50
        RuleB (趋势为主)  : BULL = 趋势牛 AND 宽度>=0.25  ← 默认，最平衡
        RuleC (仅趋势)    : BULL = 趋势牛
  + 1 月滞后确认：连续 min_consecutive(=2) 月读数为 BULL 才实际触发 β 兜底，
    降 whipsaw 误触（如 2025-01 / 2026-07 宽度一度 0% 的掉线）。

β 兜底（apply_beta_floor）：
  BULL 时若目标持仓中宽基 β ETF 权重 < BETA_FLOOR(默认40%)，把排名最高的 β ETF
  顶补到 BETA_FLOOR，削减最投机的行业/主题仓；其余仍由动量排名决定。
  熊市不兜底，策略原样跑（保留货基避险/防御），保护 2022 优势。

数据源：index_daily（沪深300 + 8 代理指数）、daily（full 口径个股）。全部用 get_conn 同源。
所有信号均在「调仓决策日(prev_td)」读取，不引入未来函数。
"""

import sqlite3
import numpy as np
import pandas as pd

# ── β 标的（平台池已含，无需新增数据）────────────────────────
BETA_ETFS = {
    "510300.SH": "沪深300ETF",
    "515800.SH": "中证800ETF",
    "512910.SH": "中证800ETF",
    "510500.SH": "中证500ETF",
}

# ── 宽度代理 8 大指数 ───────────────────────────────────────
_PROXY_IDX = ['000300.SH', '000905.SH', '000852.SH', '399006.SZ',
              '000688.SH', '000016.SH', '399001.SZ', '000985.SH']


def _get_conn():
    try:
        from run_monthly_rebalance import get_conn
        return get_conn()
    except Exception:
        return sqlite3.connect(r'D:/tu-shareData/astock_daily.db')


class RegimeDetector:
    """ regime 信号探测器：预载趋势/宽度序列，按调仓日读数，带滞后确认。 """

    def __init__(self, ma_len=200, breadth_thr=0.25, rule='B',
                 breadth_mode='proxy', min_consecutive=2):
        self.ma_len = int(ma_len)
        self.breadth_thr = float(breadth_thr)
        self.rule = rule
        self.breadth_mode = breadth_mode
        self.min_consecutive = int(min_consecutive)
        self._streak = 0
        self._history = []      # [(date, raw, eff)] 调仓日读数历史
        self._trend = {}        # {date(int): bool}
        self._breadth = {}      # {date(int): float}  (proxy 预载)
        self._breadth_full_cache = {}  # {date(int): float}  (full 惰性)
        self._conn = _get_conn()
        self._load()

    # ── 预载 ──
    def _load(self):
        conn = self._conn
        # 沪深300 趋势
        hs = pd.read_sql_query(
            "SELECT trade_date,close FROM index_daily WHERE ts_code='000300.SH' ORDER BY trade_date",
            conn)
        if len(hs):
            hs['trade_date'] = hs['trade_date'].astype(int)
            hs = hs.set_index('trade_date')['close'].astype(float)
            ma = hs.rolling(self.ma_len).mean()
            self._trend = {int(d): bool(hs[d] > ma[d])
                           for d in hs.index if not pd.isna(ma[d])}
        # 宽度
        if self.breadth_mode == 'proxy':
            self._load_proxy_breadth(conn)
        # full 口径惰性计算，不在此预载

    def _load_proxy_breadth(self, conn):
        frames = {}
        for c in _PROXY_IDX:
            df = pd.read_sql_query(
                f"SELECT trade_date,close FROM index_daily WHERE ts_code='{c}' ORDER BY trade_date",
                conn)
            if len(df):
                s = df.set_index(df['trade_date'].astype(int))['close'].astype(float)
                frames[c] = s
        if not frames:
            return
        all_dates = sorted(set().union(*[set(s.index) for s in frames.values()]))
        br = {}
        for d in all_dates:
            cnt = tot = 0
            for c, s in frames.items():
                if d not in s.index:
                    continue
                # 注意：整数索引下 s[:d] 会被当成【位置】切片（取前 d 行=全部），
                # 必须用 .loc[:d] 做【标签】切片（含 d 及之前），否则 MA20 取到未来。
                w = s.loc[:d]
                if len(w) < 20:
                    continue
                tot += 1
                if s[d] > w.iloc[-20:].mean():
                    cnt += 1
            if tot:
                br[d] = cnt / tot
        self._breadth = br

    def _breadth_full(self, td):
        """惰性计算单日全A宽度（个股 收盘>MA20 占比），缓存。 """
        td = int(td)
        if td in self._breadth_full_cache:
            return self._breadth_full_cache[td]
        conn = self._conn
        try:
            row = pd.read_sql_query(
                "SELECT a.ts_code, a.close AS c, "
                "(SELECT AVG(b.close) FROM daily b WHERE b.ts_code=a.ts_code "
                "AND b.trade_date<=a.trade_date ORDER BY b.trade_date DESC LIMIT 20) AS ma20 "
                "FROM daily a WHERE a.trade_date=?",
                conn, params=(td,))
            row = row.dropna(subset=['c', 'ma20'])
            val = float((row['c'] > row['ma20']).mean()) if len(row) else 0.0
        except Exception:
            val = 0.0
        self._breadth_full_cache[td] = val
        return val

    # ── 读数 ──
    def reading(self, td):
        """原始读数（无滞后）：'BULL' / 'BEAR'。 """
        td = int(td)
        trend = self._trend.get(td, False)
        if self.breadth_mode == 'proxy':
            br = self._breadth.get(td, 0.0)
        else:
            br = self._breadth_full(td)
        if self.rule == 'A':
            bull = trend and (br >= 0.50)
        elif self.rule == 'B':
            bull = trend and (br >= self.breadth_thr)
        else:  # 'C'
            bull = trend
        return 'BULL' if bull else 'BEAR'

    def update(self, td):
        """按调仓日推进，返回带滞后确认的有效 regime。 """
        raw = self.reading(td)
        if raw == 'BULL':
            self._streak += 1
        else:
            self._streak = 0
        self._last_raw = raw
        eff = 'BULL' if self._streak >= self.min_consecutive else 'BEAR'
        self._last_eff = eff
        self._history.append((int(td), raw, eff))
        return eff

    @property
    def last_raw(self):
        return getattr(self, '_last_raw', 'BEAR')

    @property
    def last_eff(self):
        return getattr(self, '_last_eff', 'BEAR')


def apply_beta_floor(scored, targets, beta_floor=0.40, beta_etfs=None):
    """ β 兜底：保证目标持仓中宽基 β ETF 合计权重 >= beta_floor。

    Args:
        scored:  score_assets 返回的 4 元组列表 (code,name,combined,base)，按 combined 降序
        targets: select_targets 返回的 (code,name,weight) 列表，权重和为 1
        beta_floor: 牛市强制最低 β 权重（默认 0.40）
        beta_etfs:  β 标的代码集合
    Returns:
        新的 targets 列表（权重和为 1）。若已满足或无可补 β 标的则原样返回。
    """
    if beta_etfs is None:
        beta_etfs = set(BETA_ETFS.keys())
    beta_set = {c for c, n, _, _ in scored if c in beta_etfs}
    if not beta_set:
        return targets
    beta_w = sum(w for c, _, w in targets if c in beta_set)
    if beta_w >= beta_floor - 1e-9:
        return targets  # 已满足，不破坏动量选择

    beta_lead = next((c for c, n, _, _ in scored if c in beta_etfs), None)
    if beta_lead is None:
        return targets
    lead_name = next((n for c, n, _, _ in scored if c == beta_lead), beta_lead)
    rank = {c: i for i, (c, n, _, _) in enumerate(scored)}

    others = [(c, n, w) for c, n, w in targets if c != beta_lead]
    if beta_lead in {c for c, _, _ in targets}:
        # 已在目标中：仅把其权重抬到 floor，其余按比例缩到 (1-floor)
        rem = 1.0 - beta_floor
        k = len(others)
        others = [(c, n, rem / k) for c, n, _ in others] if k else []
    else:
        # 不在目标中：剔除排名最低的 other（最投机的行业/主题），顶补 beta_lead
        if others:
            others.sort(key=lambda x: rank.get(x[0], 1e9))
            others = others[:-1]
        rem = 1.0 - beta_floor
        k = len(others)
        others = [(c, n, rem / k) for c, n, _ in others] if k else []
    return [(beta_lead, lead_name, beta_floor)] + others


def build_regime_hook(beta_floor=0.40, rule='B', ma_len=200, breadth_thr=0.25,
                      breadth_mode='proxy', min_consecutive=2):
    """构造 regime_hook(scored, targets, prev_td) -> targets，供 merged 主循环调用。

    返回值为闭包；闭包附带 .state 属性（'BULL'/'BEAR' 有效态）与 .raw 原始读数，
    供 verbose 打印。 """
    det = RegimeDetector(ma_len=ma_len, breadth_thr=breadth_thr, rule=rule,
                         breadth_mode=breadth_mode, min_consecutive=min_consecutive)

    def hook(scored, targets, prev_td):
        eff = det.update(prev_td)
        hook.state = eff
        hook.raw = det.last_raw
        hook.breadth = (det._breadth.get(int(prev_td), None)
                        if det.breadth_mode == 'proxy' else det._breadth_full(int(prev_td)))
        hook.trend = det._trend.get(int(prev_td), False)
        if eff == 'BULL':
            return apply_beta_floor(scored, targets, beta_floor=beta_floor)
        return targets

    hook.state = 'BEAR'
    hook.raw = 'BEAR'
    hook.breadth = None
    hook.trend = False
    hook.detector = det
    hook.beta_floor = beta_floor
    return hook
