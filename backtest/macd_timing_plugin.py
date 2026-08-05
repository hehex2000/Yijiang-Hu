"""
MACD 趋势跟随择时策略 - 插件化版本（继承 BaseStrategy）
============================================================

适配「选股与回测」逐股对比框架的 MACD 择时。

语义（与独立脚本 run_macd_timing.py 的核心一致，但跑在「已选出的个股」上）：
  - 多头区 = DIF > DEA（MACD 柱为正，即 DIF 在信号线上方 = 动量向上）。
    入场 = 状态由 False→True（即金叉），出场 = 状态由 True→False（死叉）。
  - 用 T-1 收盘判定、T 开盘执行，杜绝未来函数。
  - 可选零轴过滤（zero_line=True）：多头区额外要求 DIF>0，减少假死叉 whipshaw。
    指数 MA 门控（regime gate）在独立脚本里有，但逐股插件不便于取指数序列，
    故此处不实现；需要全市场 MACD 择时请用 run_macd_timing.py。

成本模型与框架其它插件一致：复用 BaseStrategy 的 buy/sell（佣金万2+最低5元，
卖出另加千1印花税）。逐日 portfolio_value = 现金 + 持仓×复权收盘，
末日按市值计价（不强制平仓，与买入持有口径对齐）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.base_strategy import BaseStrategy


class MacdTimingPlugin(BaseStrategy):
    """
    MACD 趋势跟随择时策略（金叉买入 · 死叉卖出）
    """

    def __init__(self, capital: float, cfg: dict):
        super().__init__(
            name=cfg.get("name", "MACD趋势跟随择时策略"),
            capital=capital,
            cfg=cfg,
        )

        # ── MACD 参数 ──
        self.fast = int(cfg.get("fast", 12))
        self.slow = int(cfg.get("slow", 26))
        self.signal = int(cfg.get("signal", 9))

        # ── 零轴过滤（可选）──
        self.zero_line = bool(cfg.get("zero_line", False))

        # ── 状态 ──
        self.daily_values = []
        self.trades = []

    @staticmethod
    def _macd(close, fast, slow, signal):
        """返回 (dif, dea) 序列（与 talib.MACD 同口径：EMA adjust=False）。"""
        ema_fast = pd.Series(close).ewm(span=fast, adjust=False).mean().values
        ema_slow = pd.Series(close).ewm(span=slow, adjust=False).mean().values
        dif = ema_fast - ema_slow
        dea = pd.Series(dif).ewm(span=signal, adjust=False).mean().values
        return dif, dea

    def run(self, df: pd.DataFrame, start_idx: int = 0) -> dict:
        """运行策略（逐股 MACD 择时）"""
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

        # ── 列名映射（与框架一致：优先用复权价，缺失则回退原始价）──
        if "adj_close" not in data.columns:
            data["adj_close"] = data.get("close", pd.Series(dtype=float))
        if "adj_open" not in data.columns:
            data["adj_open"] = data.get("open", data["adj_close"])
        if "adj_high" not in data.columns:
            data["adj_high"] = data.get("high", data["adj_close"])
        if "adj_low" not in data.columns:
            data["adj_low"] = data.get("low", data["adj_close"])

        close = np.asarray(data["adj_close"], dtype=float)
        open_p = np.asarray(data["adj_open"], dtype=float)
        n = len(close)

        # ── 计算 MACD（全窗口，含回溯期，保证 EMA 连续性）──
        dif, dea = self._macd(close, self.fast, self.slow, self.signal)

        # ── 多头状态（T-1 收盘判定，避免未来函数）──
        warm = self.slow + self.signal  # DIF/DEA 需足够 bar 才可信（约 35 根）
        long_state = np.zeros(n, dtype=bool)
        prev = False
        for t in range(1, n):
            d, e = dif[t - 1], dea[t - 1]
            if t < warm or np.isnan(d) or np.isnan(e):
                long_state[t] = prev  # 回溯期内维持现状（=False）
                continue
            s = bool(d > e)
            if self.zero_line:
                s = s and (d > 0)
            long_state[t] = s
            prev = s

        # ── 主循环 ──
        prev_long = False  # 回测起点前视为空仓，仅接受金叉后的首次入场
        last_valid_close = close[0] if n > 0 else 0.0
        for i in range(n):
            date = data["trade_date"].iloc[i]
            co = open_p[i]
            cc = close[i]

            if i < start_idx:
                # 回溯/预热期：平曲线
                self.daily_values.append(
                    {"date": date, "portfolio_value": self.capital})
                prev_long = False
                if not np.isnan(cc):
                    last_valid_close = cc
                continue

            # 缺数据日：维持现状估值
            if np.isnan(co) or np.isnan(cc):
                if not np.isnan(cc):
                    last_valid_close = cc
                cv = self.cash + (self.position * last_valid_close
                                  if self.position > 0 else 0.0)
                self.daily_values.append({"date": date, "portfolio_value": cv})
                continue

            long_now = long_state[i]
            if long_now and not prev_long:
                # 金叉 → 开盘买入（留 2% 现金缓冲）
                if self.position == 0 and self.cash > 0:
                    budget = self.cash * 0.98
                    sh = int(budget / co / 100) * 100
                    if sh > 0:
                        self.buy(date, co, sh, "MACD金叉买入")
            elif (not long_now) and prev_long:
                # 死叉 → 开盘卖出（清仓）
                if self.position > 0:
                    self.sell(date, co, None, "MACD死叉卖出")
            prev_long = long_now

            if not np.isnan(cc):
                last_valid_close = cc
            cv = self.cash + (self.position * cc if self.position > 0 else 0.0)
            self.daily_values.append({"date": date, "portfolio_value": cv})

        returns = self.calc_returns()
        return {"returns": returns, "trades": self.trades,
                "daily_values": self.daily_values}
