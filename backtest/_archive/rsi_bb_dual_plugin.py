# -*- coding: utf-8 -*-
"""
RSI超买超卖 + 布林带双确认择时策略
==========================================
参考视频：《【干货】RSI超买超卖，为什么用了老是被打脸？》

核心理念：
1. RSI的本质是震荡市指标——跌多了买，涨多了卖
   在趋势市中RSI会钝化（70后继续涨到90），必须结合布林带过滤

2. 双确认原则：
   - RSI说"跌够了" + 价格还没碰布林带下轨 → 再等等（假信号）
   - RSI说"涨多了" + 价格还没碰上轨 → 别急着卖（还能涨）
   - RSI和布林带同时指向极端区域 → 信号可信度大幅提升

3. A股参数适配：
   - RSI周期: 9天（比Wilder默认14更敏感，适合A股震荡节奏）
   - RSI阈值: 30/70（标准设置）
   - 布林带周期: 20天, 标准差: 2.0

4. 半凯利公式仓位管理
   参考《凯利公式——一个教我们每次买票买多少的数学公式》

信号基于前一日数据（shift(1)），避免未来函数
买入/卖出均用次日开盘价执行
"""
import pandas as pd
import numpy as np
import talib as ta
from backtest.base_strategy import BaseStrategy
from backtest.atr_stop_loss import ATRStopLoss
from backtest.kelly_sizer import KellySizer
from loguru import logger


class RsiBbDualStrategyPlugin(BaseStrategy):
    """RSI超买超卖 + 布林带双确认择时策略（半凯利）"""

    def __init__(self, capital: float, cfg: dict):
        super().__init__("RsiBbDualStrategyPlugin", capital, cfg)

        # ── RSI参数 ----
        self.rsi_period = cfg.get("rsi_period", 9)            # A股适配：9天比14更敏感
        self.rsi_oversold = cfg.get("rsi_oversold", 30)        # 超卖阈值
        self.rsi_overbought = cfg.get("rsi_overbought", 70)    # 超买阈值
        # RSI中轴：上穿50确认反弹，下穿50确认回落
        self.rsi_center = cfg.get("rsi_center", 50)

        # ── 布林带参数 ----
        self.bb_period = cfg.get("bb_period", 20)
        self.bb_std = cfg.get("bb_std", 2.0)

        # ── 止损止盈 ----
        self.stop_loss = cfg.get("stop_loss", 0.05)      # 5%硬止损
        self.take_profit = cfg.get("take_profit", 0.20)   # 20%止盈

        # ── ATR 动态止损 ----
        self.use_atr_sl = cfg.get("atr_stop_loss", True)
        self.atr_sl = ATRStopLoss(
            atr_period=cfg.get("atr_period", 14),
            atr_mult=cfg.get("atr_mult", 2.0),
            trail_mult=cfg.get("trail_mult", 2.0),
        )

        # ── 凯利公式仓位 ----
        self.use_kelly = cfg.get("use_kelly", True)
        if self.use_kelly:
            self.kelly = KellySizer(
                estimated_win_rate=cfg.get("kelly_win_rate", 0.53),
                estimated_win_loss_ratio=cfg.get("kelly_win_loss_ratio", 1.5),
                kelly_fraction=cfg.get("kelly_fraction", 0.5),
                max_position_pct=cfg.get("kelly_max_position", 0.20),
                min_position_pct=cfg.get("kelly_min_position", 0.05),
                safety_discount=cfg.get("kelly_safety_discount", 0.8),
            )
        else:
            self.position_mode = cfg.get("position_mode", "half")

        logger.info(
            f"RsiBbDualStrategyPlugin initialized: "
            f"RSI({self.rsi_period}) o/s={self.rsi_oversold}/{self.rsi_overbought}, "
            f"BB({self.bb_period},{self.bb_std}), "
            f"kelly={'enabled' if self.use_kelly else 'disabled'}"
        )

    def _calculate_bollinger(self, close: pd.Series) -> pd.DataFrame:
        """计算布林带指标（TA-Lib）"""
        upper, middle, lower = ta.BBANDS(
            close.values,
            timeperiod=self.bb_period,
            nbdevup=self.bb_std,
            nbdevdn=self.bb_std,
            matype=0  # SMA
        )
        return pd.DataFrame({
            'middle': middle, 'upper': upper, 'lower': lower
        })

    def _calculate_rsi(self, close: pd.Series) -> pd.Series:
        """计算RSI指标（TA-Lib）"""
        return pd.Series(ta.RSI(close.values, timeperiod=self.rsi_period), index=close.index)

    def run(self, df: pd.DataFrame, start_idx: int = 0) -> dict:
        """
        运行RSI+布林带双确认策略

        Returns:
            {"returns": float, "trades": list, "daily_values": list}
        """
        logger.info(f"Running RsiBbDualStrategyPlugin on {len(df)} days, start_idx={start_idx}")
        self.trades = []
        self.daily_values = []
        self.position = 0
        self.cash = self.capital

        if len(df) == 0:
            return {"returns": 0.0, "trades": [], "daily_values": []}

        df = df.sort_values('trade_date').reset_index(drop=True)

        if start_idx < 0 or start_idx >= len(df):
            start_idx = 0

        # 列名映射
        data = df.copy()
        if "adj_close" not in data.columns:
            if "close" in data.columns:
                data["adj_close"] = data["close"]
            else:
                logger.error("缺少必要列: close / adj_close")
                return {"returns": 0.0, "trades": [], "daily_values": []}
        if "adj_open" not in data.columns:
            if "open" in data.columns:
                data["adj_open"] = data["open"]
            else:
                logger.error("缺少必要列: open / adj_open")
                return {"returns": 0.0, "trades": [], "daily_values": []}

        # === 计算指标 ===
        bb = self._calculate_bollinger(data['adj_close'])
        data = pd.concat([data, bb], axis=1)
        data['rsi'] = self._calculate_rsi(data['adj_close'])

        # === ATR ===
        if self.use_atr_sl:
            high_col = 'high' if 'high' in data.columns else 'adj_close'
            low_col = 'low' if 'low' in data.columns else 'adj_close'
            atr_arr = self.atr_sl.calc_atr(data[high_col].values, data[low_col].values, data['adj_close'].values)
        else:
            atr_arr = np.zeros(len(data))

        # === 信号生成（基于前一日数据，避免未来函数）===
        prev_close = data['adj_close'].shift(1)
        prev_lower = data['lower'].shift(1)
        prev_upper = data['upper'].shift(1)
        prev_rsi = data['rsi'].shift(1)

        # 买入信号：RSI超卖 + 价格接近布林带下轨（双确认）
        # RSI < 超卖阈值：动量耗尽
        # 价格 <= 下轨×1.02：处于布林带底部区域
        buy_signal = (
            (prev_rsi < self.rsi_oversold).fillna(False) &
            (prev_close <= prev_lower * 1.02).fillna(False)
        )

        # 卖出信号：RSI超买 + 价格接近布林带上轨（双确认）
        sell_signal = (
            (prev_rsi > self.rsi_overbought).fillna(False) &
            (prev_close >= prev_upper * 0.98).fillna(False)
        )

        data['buy_signal'] = buy_signal
        data['sell_signal'] = sell_signal

        # === 遍历数据执行交易 ===
        for i in range(start_idx, len(data)):
            row = data.iloc[i]
            date = row['trade_date']
            open_price = float(row['adj_open'])
            close_price = float(row['adj_close'])

            # 跳过早期无指标数据
            if pd.isna(row['middle']) or pd.isna(row['rsi']):
                v = self.cash + self.position * close_price if self.position > 0 else self.cash
                self.daily_values.append({'date': date, 'portfolio_value': v})
                continue

            # 前一日收盘价（用于止损判断）
            prev_close_val = float(data.iloc[i - 1]['adj_close']) if i >= 1 else None

            # === 买入逻辑 ===
            if self.position == 0 and bool(data.iloc[i]['buy_signal']):
                # 计算仓位
                if self.use_kelly:
                    position_pct = self.kelly.get_position_pct()
                else:
                    position_pct = 0.95 if self.position_mode == 'full' else 0.50

                if position_pct <= 0:
                    v = self.cash
                    self.daily_values.append({'date': date, 'portfolio_value': v})
                    continue

                buy_amount = self.cash * position_pct
                shares = int(buy_amount / open_price / 100) * 100
                if shares > 0:
                    rsi_val = float(prev_rsi.iloc[i]) if i < len(prev_rsi) else 0
                    entry_reason = (
                        f"RSI超卖+下轨(RSI={rsi_val:.1f},"
                        f"仓位={position_pct:.1%})"
                    )
                    success = self.buy(date, open_price, shares, reason=entry_reason)
                    if success and self.use_atr_sl:
                        self.atr_sl.on_entry(entry_price=open_price, atr_val=atr_arr[i])

            # === 卖出逻辑 ===
            elif self.position > 0:
                # ── ATR 追踪止损 ----
                if self.use_atr_sl:
                    high_val = float(data.iloc[i].get('high', close_price))
                    self.atr_sl.update(high_price=high_val, atr_val=atr_arr[i])
                    should_stop, stop_price, atr_reason = self.atr_sl.check_stop(close_price=close_price)
                    if should_stop:
                        self.sell(date, open_price, reason=atr_reason)
                        v = self.cash + self.position * close_price if self.position > 0 else self.cash
                        self.daily_values.append({'date': date, 'portfolio_value': v})
                        continue

                should_sell = False
                sell_reason = ""

                # 1. 出场信号：RSI超买 + 价格近上轨（双确认）
                if bool(data.iloc[i]['sell_signal']):
                    should_sell = True
                    rsi_val = float(prev_rsi.iloc[i]) if i < len(prev_rsi) else 0
                    sell_reason = f"RSI超买+上轨(RSI={rsi_val:.1f})"

                # 2. 止盈（基于前一日收盘价）
                if not should_sell and self.avg_cost > 0 and prev_close_val is not None:
                    profit_pct = (prev_close_val - self.avg_cost) / self.avg_cost
                    if profit_pct >= self.take_profit:
                        should_sell = True
                        sell_reason = f"止盈({profit_pct:.1%})"

                # 3. 止损（ATR启用时跳过固定止损）
                if not should_sell and not self.use_atr_sl:
                    if self.avg_cost > 0 and prev_close_val is not None:
                        loss_pct = (prev_close_val - self.avg_cost) / self.avg_cost
                        if loss_pct <= -self.stop_loss:
                            should_sell = True
                            sell_reason = f"硬止损({loss_pct:.1%})"

                if should_sell and self.position > 0:
                    self.sell(date, open_price, reason=sell_reason)

            # 记录每日资产
            v = self.cash + self.position * close_price if self.position > 0 else self.cash
            self.daily_values.append({'date': date, 'portfolio_value': v})

        # 回测结束平仓
        if self.position > 0:
            last_date = data.iloc[-1]['trade_date']
            last_price = float(data.iloc[-1]['adj_close'])
            self.sell(last_date, last_price, reason="回测结束平仓")
            self.daily_values[-1]['portfolio_value'] = self.cash

        ret = self.calc_returns()
        logger.info(f"RsiBbDualStrategyPlugin finished: returns={ret:.2f}%, trades={len(self.trades)}")
        return {"returns": ret, "trades": self.trades, "daily_values": self.daily_values}
