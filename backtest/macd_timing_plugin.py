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

# 方向状态机三态（与双均线 Jim 框架同构）
UP, DOWN, UNCLEAR = "UP", "DOWN", "UNCLEAR"


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

        # ── Jim 状态机增强（可选，默认关；由 macd_jim 子类强制开启）──
        self.state_machine = bool(cfg.get("state_machine", False))
        self.slope_lookback = int(cfg.get("slope_lookback", 10))
        self.slope_thresh = float(cfg.get("slope_thresh", 0.0))

        # ── 评估节奏门控（anti-whipsaw）──
        # daily    = 每日判定金叉/死叉（横盘期可能月内反复买卖=磨损）
        # monthly  = 仅月末重估状态，中间日沿用上次决策（过滤月内噪声）★ 现默认
        # quarterly= 仅季末重估
        self.eval_freq = str(cfg.get("eval_freq", "monthly")).lower()
        self._eval_mask = None  # 在 run()/_run_state_machine() 内按 trade_date 构建

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

    @staticmethod
    def _build_eval_mask(trade_dates, freq):
        """评估边界掩码：True=该日重估 MACD 状态。
        daily=每日；monthly=每月末；quarterly=每季末。非边界日沿用上次决策→过滤月内噪声。
        """
        n = len(trade_dates)
        if freq == "daily":
            return np.ones(n, dtype=bool)
        mask = np.zeros(n, dtype=bool)
        keys = []
        for d in trade_dates:
            y, m = d // 10000, (d // 100) % 100
            keys.append((y, (m - 1) // 3) if freq == "quarterly" else (y, m))
        seen = {}
        for i, k in enumerate(keys):
            if k not in seen or trade_dates[i] > trade_dates[seen[k]]:
                seen[k] = i
        for i in seen.values():
            mask[i] = True
        return mask

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

        # ── Jim 状态机增强分支（方向状态机 + 信号漏斗 + 失效≠反转）──
        if self.state_machine:
            return self._run_state_machine(df, start_idx, data, close, open_p, dif, dea)

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

        # ── 评估节奏门控（anti-whipsaw）──
        self._eval_mask = self._build_eval_mask(
            data["trade_date"].astype(int).tolist(), self.eval_freq)

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

            # 节奏门控：非边界日沿用上次决策，仅边界日重估 long_state（过滤月内噪声磨损）
            long_now = long_state[i] if self._eval_mask[i] else prev_long
            if long_now and not prev_long:
                # 金叉 → 开盘买入（留 2% 现金缓冲）
                if self.position == 0 and self.cash > 0:
                    budget = self.cash * 0.98
                    sh = int(budget / co / 100) * 100
                    if sh > 0:
                        self._kelly_buy(date, co, sh, "MACD金叉买入")
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

    # ── Jim 状态机增强分支 ──────────────────────────────────────────────
    def _state_of_macd(self, dif, dea, slope):
        """方向状态机（与双均线 Jim 框架同构）：
        UP   = DIF>DEA 且 DIF>0(零轴上方·趋势确认) 且 DIF斜率向上  → 多头确认
        DOWN = DIF<DEA 且 DIF<0(零轴下方) 且 DIF斜率向下          → 空头确认
        UNCLEAR = 柱近零/零轴附近纠缠/方向不明（看不清，不出手）   → 合法空仓
        零轴上方约束使空头市场的假金叉(反弹)被挡在 UNCLEAR，构成信号漏斗。
        """
        if np.isnan(dif) or np.isnan(dea) or np.isnan(slope):
            return UNCLEAR
        if dif > dea and dif > 0 and slope > self.slope_thresh:
            return UP
        if dif < dea and dif < 0 and slope < -self.slope_thresh:
            return DOWN
        return UNCLEAR

    def _run_state_machine(self, df, start_idx, data, close, open_p, dif, dea):
        """Jim 增强主循环：
        入场 = 金叉(上升沿) 且 状态==UP（金叉硬触发 + 方向确认，信号漏斗做减法）
        出场 = 状态退出 UP（prev_state==UP 且 state!=UP，失效≠反转先空仓，不裸空）
        """
        n = len(close)
        warm = self.slow + self.signal
        # 评估节奏门控掩码
        self._eval_mask = self._build_eval_mask(
            data["trade_date"].astype(int).tolist(), self.eval_freq)
        # 金叉判定（T-1 收盘，复用 zero_line 逻辑）
        long_state = np.zeros(n, dtype=bool)
        prev = False
        for t in range(1, n):
            d, e = dif[t - 1], dea[t - 1]
            if t < warm or np.isnan(d) or np.isnan(e):
                long_state[t] = prev
                continue
            s = bool(d > e)
            if self.zero_line:
                s = s and (d > 0)
            long_state[t] = s
            prev = s
        # DIF 斜率（L 期差分）
        L = max(1, self.slope_lookback)
        slope = np.zeros(n)
        for i in range(L, n):
            slope[i] = dif[i] - dif[i - L]

        prev_long = False
        prev_state = UNCLEAR
        last_valid_close = close[0] if n > 0 else 0.0
        for i in range(n):
            date = data["trade_date"].iloc[i]
            co = open_p[i]
            cc = close[i]
            if i < start_idx:
                self.daily_values.append({"date": date, "portfolio_value": self.capital})
                prev_long = False
                prev_state = UNCLEAR
                if not np.isnan(cc):
                    last_valid_close = cc
                continue
            if np.isnan(co) or np.isnan(cc):
                if not np.isnan(cc):
                    last_valid_close = cc
                cv = self.cash + (self.position * last_valid_close if self.position > 0 else 0.0)
                self.daily_values.append({"date": date, "portfolio_value": cv})
                continue
            long_now = long_state[i] if self._eval_mask[i] else prev_long
            golden = bool(long_now and not prev_long)
            if i < warm or np.isnan(dif[i]) or np.isnan(dea[i]):
                state = UNCLEAR
            else:
                state = self._state_of_macd(dif[i], dea[i], slope[i])
            # 入场：金叉 + 方向 UP（漏斗：金叉硬触发，方向过滤只做减法）
            if golden and state == UP and self.cash > 0 and self.position == 0:
                budget = self.cash * 0.98
                sh = int(budget / co / 100) * 100
                if sh > 0:
                    self._kelly_buy(date, co, sh, "MACD金叉+方向确认买入")
            # 出场：状态退出 UP（失效≠反转，先空仓，不裸空）
            elif prev_state == UP and state != UP and self.position > 0:
                self.sell(date, co, None, "状态退出卖出")
            prev_long = long_now
            prev_state = state
            if not np.isnan(cc):
                last_valid_close = cc
            cv = self.cash + (self.position * cc if self.position > 0 else 0.0)
            self.daily_values.append({"date": date, "portfolio_value": cv})
        returns = self.calc_returns()
        return {"returns": returns, "trades": self.trades,
                "daily_values": self.daily_values}
