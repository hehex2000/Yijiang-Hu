"""
利弗莫尔关键点突破策略 - 插件化版本（继承 BaseStrategy，逐股）
================================================================

适配「选股与回测」逐股对比框架。把视频四步法翻译成逐股可执行的规则：

  步骤1 市场环境(水流) : 沪深300 站上 MA(market_ma) → 才允许开仓（熊市整批关信号）
  步骤2 板块强度(板块靠前): 逐股近似 = 个股 N 日动量 > 沪深300 同期动量
                            （即"个股跑赢大盘"，对应视频"不碰弱势股/选板块靠前的"）
                            ⚠️ 平台逐股框架不便做行业内横截面排名，故用"跑赢指数"做代理；
                               完整横截面版本见独立脚本 run_livermore_breakout.py
  步骤3 关键点(被越过) : 收盘价创 lookback 日新高（前高突破）即触发
  步骤4 失效退出(跌回区间): 收盘跌回突破位(关键点) 或 跌破 MA(ma_period)
                            或市场转熊（整批清仓）

无未来函数: 信号用 T-1 数据判定（close[T-1] > 前N日最高价），T 开盘执行；
           退出信号用 T-1 收盘判定，T 开盘执行。

成本模型与框架一致: 复用 BaseStrategy 的 buy/sell（佣金万2+最低5元，卖出另加千1印花税）。
末日按市值计价（与买入持有口径对齐）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.base_strategy import BaseStrategy


# ── 指数缓存（沪深300 用于市场环境门控 + 相对强度代理）──
_idx_cache = {}   # (start, end) -> dict(date_str -> close)


def _load_index(start, end):
    key = (start, end)
    if key in _idx_cache:
        return _idx_cache[key]
    try:
        from run_monthly_rebalance import get_conn
        conn = get_conn()
        df = pd.read_sql_query(
            "SELECT trade_date, close FROM index_daily WHERE ts_code='000300.SH' "
            "AND trade_date BETWEEN ? AND ? ORDER BY trade_date",
            conn, params=(str(start), str(end)))
        conn.close()
        d = dict(zip(df["trade_date"].astype(str).tolist(), df["close"].astype(float).tolist()))
        _idx_cache[key] = d
    except Exception:
        _idx_cache[key] = {}
    return _idx_cache[key]


class LivermorePlugin(BaseStrategy):
    """
    利弗莫尔关键点突破策略（市场环境 + 相对强度 + 关键点突破 + 失效退出）
    """

    def __init__(self, capital: float, cfg: dict):
        super().__init__(
            name=cfg.get("name", "利弗莫尔关键点突破策略"),
            capital=capital,
            cfg=cfg,
        )
        self.lookback = int(cfg.get("lookback", 60))
        self.mom_lookback = int(cfg.get("mom_lookback", 60))
        self.ma_period = int(cfg.get("ma_period", 20))
        self.market_ma = int(cfg.get("market_ma", 60))
        self.stop_loss = float(cfg.get("stop_loss", 0.0))
        self.exit_pct = float(cfg.get("exit_pct", 0.0))
        self.market_exit = bool(cfg.get("market_exit", True))
        self.entry_key = np.nan       # 入场时的突破位（失效退出对照，非滚动关键点）
        self.daily_values = []
        self.trades = []

    def run(self, df: pd.DataFrame, start_idx: int = 0) -> dict:
        self.daily_values = []
        self.trades = []
        self.cash = self.capital
        self.position = 0
        self.avg_cost = 0.0

        if df is None or len(df) == 0:
            return {"returns": 0.0, "trades": [], "daily_values": []}

        data = df.copy()
        if "trade_date" not in data.columns:
            raise KeyError("缺少必要列: trade_date")
        data = data.sort_values("trade_date").reset_index(drop=True)

        if "adj_close" not in data.columns:
            data["adj_close"] = data.get("close", pd.Series(dtype=float))
        if "adj_open" not in data.columns:
            data["adj_open"] = data.get("open", data["adj_close"])
        if "adj_high" not in data.columns:
            data["adj_high"] = data.get("high", data["adj_close"])

        close = np.asarray(data["adj_close"], dtype=float)
        high = np.asarray(data["adj_high"], dtype=float)
        open_p = np.asarray(data["adj_open"], dtype=float)
        dates = data["trade_date"].astype(str).tolist()
        n = len(close)

        # ── 指数（市场环境 + 相对强度）──
        idx_map = _load_index(dates[0], dates[-1])
        idx_close = np.array([idx_map.get(d, np.nan) for d in dates], dtype=float)
        # 指数 MA
        idx_valid = ~np.isnan(idx_close)
        idx_ma = np.full(n, np.nan)
        if idx_valid.any():
            s = pd.Series(idx_close)
            m = s.rolling(self.market_ma, min_periods=self.market_ma).mean().values
            idx_ma = m
        bull = np.zeros(n, dtype=bool)
        for t in range(n):
            if t < self.market_ma or np.isnan(idx_close[t]) or np.isnan(idx_ma[t]):
                bull[t] = False
            else:
                bull[t] = idx_close[t] > idx_ma[t]

        # ── 关键点（前 lookback 日最高价, ≤ T-1）──
        key_level = np.full(n, np.nan)
        for t in range(n):
            lo = t - self.lookback
            if lo < 0:
                continue
            seg = high[max(0, lo):t]   # [t-lookback, t-1]
            if len(seg) >= self.lookback:
                key_level[t] = seg.max()
        # 突破信号: close[t] > key_level[t]（用 ≤ t-1 的高点，无未来函数）
        breakout = np.zeros(n, dtype=bool)
        for t in range(n):
            if not np.isnan(key_level[t]) and close[t] > key_level[t]:
                breakout[t] = True

        # ── 相对强度代理: 个股动量 > 指数动量 ──
        rs_pass = np.zeros(n, dtype=bool)
        for t in range(self.mom_lookback, n):
            if np.isnan(close[t]) or np.isnan(close[t - self.mom_lookback]):
                continue
            sm = close[t] / close[t - self.mom_lookback] - 1.0
            ic = idx_close[t]
            ic0 = idx_close[t - self.mom_lookback]
            if np.isnan(ic) or np.isnan(ic0) or ic0 == 0:
                continue
            im = ic / ic0 - 1.0
            rs_pass[t] = bool(sm > im)

        # ── MA（失效退出用）──
        ma = pd.Series(close).rolling(self.ma_period, min_periods=self.ma_period).mean().values

        # ── 主循环: 信号 T-1 判定, T 开盘执行（与 macd 插件口径一致）──
        last_valid = close[0] if n > 0 and not np.isnan(close[0]) else 0.0
        for i in range(n):
            date = dates[i]
            co = open_p[i]
            cc = close[i]

            if i < start_idx:
                self.daily_values.append({"date": date, "portfolio_value": self.capital})
                if not np.isnan(cc):
                    last_valid = cc
                continue

            if np.isnan(co) or np.isnan(cc):
                if not np.isnan(cc):
                    last_valid = cc
                cv = self.cash + (self.position * last_valid if self.position > 0 else 0.0)
                self.daily_values.append({"date": date, "portfolio_value": cv})
                continue

            prev = i - 1
            # 退出（T-1 收盘判定 → T 开盘卖出；对照"入场突破位"，非滚动关键点，避免次日被误判）
            if self.position > 0 and prev >= 0:
                exit_now = False
                if not np.isnan(self.entry_key) and close[prev] < self.entry_key * (1 - self.exit_pct):
                    exit_now = True
                elif not np.isnan(ma[prev]) and close[prev] < ma[prev]:
                    exit_now = True
                elif self.stop_loss > 0 and not np.isnan(self.avg_cost) and close[prev] < self.avg_cost * (1 - self.stop_loss):
                    exit_now = True
                elif self.market_exit and (not bull[prev]):
                    exit_now = True
                if exit_now:
                    self.sell(date, co, None, "利弗莫尔失效退出")
            # 开仓（T-1 突破 + 市场环境 + 相对强度 → T 开盘买入）
            elif self.position == 0 and prev >= 0 and breakout[prev] and rs_pass[prev] and bull[prev]:
                budget = self.cash * 0.98
                sh = int(budget / co / 100) * 100
                if sh > 0:
                    self.entry_key = key_level[prev]   # 记录入场突破位
                    self._kelly_buy(date, co, sh, "利弗莫尔关键点突破买入")

            if not np.isnan(cc):
                last_valid = cc
            cv = self.cash + (self.position * cc if self.position > 0 else 0.0)
            self.daily_values.append({"date": date, "portfolio_value": cv})

        returns = self.calc_returns()
        return {"returns": returns, "trades": self.trades, "daily_values": self.daily_values}
