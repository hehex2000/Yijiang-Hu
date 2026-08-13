#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
均值回归策略插件（继承 BaseStrategy）
基于视频《量化交易入门——均值回归策略》实现

核心逻辑：
1. 三把尺子：布林带（均值）、波动率（正常范围）、Z-Score（偏离程度）
2. 双重共振入场：价格触及下轨 AND RSI < 30（动能彻底耗尽）
3. 布林带宽度过滤：喇叭口张开（趋势启动）时禁止入场
4. 出场：Z > 2 OR RSI > 70
5. 硬止损：3%（防止均值永久下移）

信号基于前一日数据（shift(1)），避免未来函数
买入/卖出均用次日开盘价执行
"""
import pandas as pd
import numpy as np
import talib as ta
from backtest.base_strategy import BaseStrategy
from backtest.atr_stop_loss import ATRStopLoss
from loguru import logger


class MeanReversionStrategyPlugin(BaseStrategy):
    """均值回归策略（视频版）"""

    def __init__(self, capital: float, cfg: dict):
        super().__init__("MeanReversionStrategyPlugin", capital, cfg)
        self.bb_period = cfg.get("bb_period", 20)
        self.bb_std = cfg.get("bb_std", 2.0)
        self.rsi_period = cfg.get("rsi_period", 14)
        self.rsi_oversold = cfg.get("rsi_oversold", 30)
        self.rsi_overbought = cfg.get("rsi_overbought", 70)
        self.zscore_threshold = cfg.get("zscore_threshold", 2.0)
        self.band_width_ma = cfg.get("band_width_ma", 20)
        self.stop_loss = cfg.get("stop_loss", 0.03)
        # ── ATR 动态止损 ──
        self.use_atr_sl = cfg.get("atr_stop_loss", False)
        self.atr_sl = ATRStopLoss(
            atr_period=cfg.get("atr_period", 14),
            atr_mult=cfg.get("atr_mult", 3.0),
            trail_mult=cfg.get("trail_mult", 3.0),
        )
        self.position_mode = cfg.get("position_mode", "half")
        logger.info(
            f"MeanReversionStrategyPlugin initialized: "
            f"bb_period={self.bb_period}, bb_std={self.bb_std}, "
            f"rsi_oversold={self.rsi_oversold}, rsi_overbought={self.rsi_overbought}, "
            f"zscore_threshold={self.zscore_threshold}, stop_loss={self.stop_loss}"
        )

    def _calculate_bollinger(self, close: pd.Series) -> pd.DataFrame:
        """计算布林带指标（使用 TA-Lib）"""
        upper, middle, lower = ta.BBANDS(
            close.values,
            timeperiod=self.bb_period,
            nbdevup=self.bb_std,
            nbdevdn=self.bb_std,
            matype=0  # 0 = SMA
        )
        return pd.DataFrame({
            'middle': middle, 'upper': upper, 'lower': lower
        })

    def _calculate_rsi(self, close: pd.Series) -> pd.Series:
        """计算RSI指标（使用 TA-Lib）"""
        return pd.Series(ta.RSI(close.values, timeperiod=self.rsi_period), index=close.index)

    def run(self, df: pd.DataFrame, start_idx: int = 0) -> dict:
        """
        运行均值回归策略
        返回: {"returns": float, "trades": list, "daily_values": list}
        """
        logger.info(f"Running MeanReversionStrategyPlugin on {len(df)} days of data, start_idx={start_idx}...")
        self.trades = []
        self.daily_values = []
        self.position = 0
        self.cash = self.capital

        if len(df) == 0:
            return {"returns": 0.0, "trades": [], "daily_values": []}

        df = df.sort_values('trade_date').reset_index(drop=True)

        # 确保 start_idx 在有效范围内
        if start_idx < 0 or start_idx >= len(df):
            start_idx = 0

        # 列名映射（兼容有无复权列）
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

        # 布林带宽度（检测喇叭口）
        data['band_width'] = (data['upper'] - data['lower']) / data['middle']
        data['band_width_ma'] = data['band_width'].rolling(window=self.band_width_ma).mean()
        # 喇叭口张开：宽度超过MA 10% = 趋势启动，禁止入场
        data['band_widening'] = data['band_width'] > data['band_width_ma'] * 1.1

        # Z-Score（偏离程度）
        data['std'] = data['adj_close'].rolling(window=self.bb_period).std()
        data['z_score'] = (data['adj_close'] - data['middle']) / data['std']

        # ── ATR 计算 ──
        if self.use_atr_sl:
            high_col = 'high' if 'high' in data.columns else 'adj_close'
            low_col = 'low' if 'low' in data.columns else 'adj_close'
            atr_arr = self.atr_sl.calc_atr(data[high_col].values, data[low_col].values, data['adj_close'].values)
        else:
            atr_arr = np.zeros(len(data))

        # === 信号生成（避免未来函数：用 shift(1)）===
        prev_close = data['adj_close'].shift(1)
        prev_z = data['z_score'].shift(1)
        prev_rsi = data['rsi'].shift(1)
        prev_widening = data['band_widening'].shift(1)
        prev_lower = data['lower'].shift(1)
        prev_upper = data['upper'].shift(1)

        # 处理 NaN：NaN 视为不满足条件（保守做法）
        # prev_z < -threshold：NaN → False（不入场）
        cond_z_low = (prev_z < -self.zscore_threshold).fillna(False)
        # prev_rsi < oversold：NaN → False（不入场）
        cond_rsi_low = (prev_rsi < self.rsi_oversold).fillna(False)
        # prev_widening：NaN → True（未知时保守视为布林带张开，禁止入场）
        # 然后用 ~ 取反（此时已无 NaN，~ 可安全使用）
        cond_not_widening = ~(prev_widening.fillna(True))

        # 入场信号：Z < -2 AND RSI < 30 AND 布林带未张开（双重共振）
        buy_signal = cond_z_low & cond_rsi_low & cond_not_widening

        # 出场信号：Z > 2 OR RSI > 70
        # NaN → False（不触发出场）
        cond_z_high = (prev_z > self.zscore_threshold).fillna(False)
        cond_rsi_high = (prev_rsi > self.rsi_overbought).fillna(False)
        sell_signal = cond_z_high | cond_rsi_high

        data['buy_signal'] = buy_signal
        data['sell_signal'] = sell_signal

        # === 遍历数据执行交易 ===
        for i in range(start_idx, len(data)):
            row = data.iloc[i]
            date = row['trade_date']
            open_price = float(row['adj_open'])
            close_price = float(row['adj_close'])

            # 跳过早期无指标数据
            if pd.isna(row['middle']) or pd.isna(row['z_score']):
                v = self.cash + self.position * close_price if self.position > 0 else self.cash
                self.daily_values.append({'date': date, 'portfolio_value': v})
                continue

            # 前一日收盘价（用于止损判断，避免未来函数）
            prev_close_val = float(data.iloc[i - 1]['adj_close']) if i >= 1 else None

            # === 买入逻辑 ===
            if self.position == 0 and bool(data.iloc[i]['buy_signal']):
                buy_amount = self.cash * (0.95 if self.position_mode == 'full' else 0.50)
                shares = int(buy_amount / open_price / 100) * 100
                if shares > 0:
                    z_val = float(prev_z.iloc[i]) if i < len(prev_z) else 0
                    rsi_val = float(prev_rsi.iloc[i]) if i < len(prev_rsi) else 0
                    success = self._kelly_buy(
                        date, open_price, shares,
                        reason=f"均值回归入场(Z={z_val:.2f},RSI={rsi_val:.1f})"
                    )
                    # ATR初始化（仅当买入成功时）
                    if success and self.use_atr_sl:
                        self.atr_sl.on_entry(entry_price=open_price, atr_val=atr_arr[i])

            # === 卖出逻辑 ===
            elif self.position > 0:
                # ── ATR 追踪止损（每日更新，独立于信号）──
                if self.use_atr_sl:
                    high_val = float(data.iloc[i].get('high', close_price))
                    self.atr_sl.update(high_price=high_val, atr_val=atr_arr[i])
                    should_stop, stop_price, atr_reason = self.atr_sl.check_stop(close_price=close_price)
                    if should_stop:
                        self.sell(date, open_price, reason=atr_reason)
                        v = self.cash + self.position * close_price
                        self.daily_values.append({'date': date, 'portfolio_value': v})
                        continue

                should_sell = False
                sell_reason = ""

                # 1. 出场信号（Z > 2 或 RSI > 70）
                if bool(data.iloc[i]['sell_signal']):
                    should_sell = True
                    z_val = float(prev_z.iloc[i]) if i < len(prev_z) else 0
                    rsi_val = float(prev_rsi.iloc[i]) if i < len(prev_rsi) else 0
                    sell_reason = f"均值回归出场(Z={z_val:.2f},RSI={rsi_val:.1f})"

                # 2. 止损（用前一日收盘价判断，避免未来函数）
                #    启用ATR时使用ATR动态止损，否则用固定3%硬止损
                if not self.use_atr_sl:
                    if prev_close_val is not None and self.avg_cost > 0:
                        loss_pct = (prev_close_val - self.avg_cost) / self.avg_cost
                        if loss_pct <= -self.stop_loss and not should_sell:
                            should_sell = True
                            sell_reason = f"硬止损({(loss_pct * 100):.1f}%)"

                if should_sell and self.position > 0:
                    self.sell(date, open_price, reason=sell_reason)

            # 记录每日资产
            v = self.cash + self.position * close_price
            self.daily_values.append({'date': date, 'portfolio_value': v})

        # 回测结束平仓
        if self.position > 0:
            last_date = data.iloc[-1]['trade_date']
            last_price = float(data.iloc[-1]['adj_close'])
            self.sell(last_date, last_price, reason="回测结束平仓")
            # 重新记录最后一条每日资产（含卖出手续费）
            v_final = self.cash
            self.daily_values[-1]['portfolio_value'] = v_final

        ret = self.calc_returns()
        logger.info(f"MeanReversionStrategyPlugin finished: returns={ret:.2f}%, trades={len(self.trades)}")
        return {"returns": ret, "trades": self.trades, "daily_values": self.daily_values}
