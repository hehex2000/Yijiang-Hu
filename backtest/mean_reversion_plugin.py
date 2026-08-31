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

═══════════════════════════════════════════════════════════════
⚠️ 布林带改进候选验证状态（2026-08-31，全部默认关闭）
   完整结论见 docs/bollinger_enhancement_report.md（唯一权威）
───────────────────────────────────────────────────────────────
四个候选经四道检验（单变量 A/B → regime-gate+真实成本两关 →
walk-forward → 随机入场暴露度对照），最终判定 **0 个有 alpha**：

  headfake_filter   ❌ 证伪（Δ −1.53pp）
  mfi_filter        ❌ 证伪（Δ −8.30pp 简单 / −7.96pp 真实）
  pctb_divergence   ❌ 无 alpha（暴露度对齐后 ≈ −1.8pp，跑输随机）
  obv_divergence    ❌ 无 alpha（暴露度对齐后 ≈ 0.00pp，等于随机）

🔑 教训：后两者在全样本 A/B 上「看似为正」（+1.79 / +4.61pp），
   但那只是「交易更多 = 在市更久 = 吃到长牛 beta」。
   regime-gate 与 walk-forward 都洗不掉 beta —— 它们只检验稳定性，
   而暴露度在每个子样本里被同等放大。
   → 任何「额外买入触发」型新信号，上线前必须用 random_entry 做暴露度对照。

✅ 真正赚钱的是本文件的**出场/风控框架**（Z>2 或 RSI>70 出场 + 止损）：
   随机入场控制组用同一套出场逻辑也能拿到同等收益，证明正期望来自出场侧。
═══════════════════════════════════════════════════════════════
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
        # ── Head-fake 假突破过滤（opt-in，默认关闭）──
        # 视频 BV1FP876bEAo 进阶篇：收口期假突破(向下击穿下轨洗止损)后，
        # 需随后 1-2 根 K 线重新站回下轨(顺轨站稳)才确认买入；连续贴轨外下行=真破位陷阱，取消。
        self.headfake_filter = cfg.get("headfake_filter", False)
        # ── %B 背离额外买入（opt-in，默认关闭）──
        # 视频 BV1FP876bEAo：价格创 N 日新低但 %B 未同步创新低 = 动能衰减/抛压枯竭的看涨背离，
        # 作为均值回归的额外买入触发。固定滞后 K 比较（t 与 t-K 均属过去），无未来函数。
        self.pctb_divergence = cfg.get("pctb_divergence", False)
        self.pctb_lookback = cfg.get("pctb_lookback", 10)
        # ── MFI 量能过滤闸门（opt-in，默认关闭）──
        # MFI=量加权 RSI，< 超卖阈值表示下跌伴随放量抛压(恐慌性抛售/capitulation)。
        # 作为均值回归买入的确认闸门：仅在 MFI 也超卖时才允许入场，
        # 过滤"无量阴跌/死水"式假超卖信号。纯量能确认，不改变 %B/OBV 逻辑。
        self.mfi_filter = cfg.get("mfi_filter", False)
        self.mfi_period = cfg.get("mfi_period", 14)
        self.mfi_oversold = cfg.get("mfi_oversold", 20)
        # ── OBV 量能背离额外买入（opt-in，默认关闭）──
        # OBV 累积量能：价格创 K 日新低但 OBV 未同步创 K 日新低 = 下跌未获量能确认
        # (抛压枯竭/暗中吸筹) 的看涨量能背离，作为均值回归额外买入触发。
        # 与 %B 背离机制平行，但信号源在成交量而非布林带相对位置。
        self.obv_divergence = cfg.get("obv_divergence", False)
        self.obv_lookback = cfg.get("obv_lookback", 10)
        # ── 随机入场控制组（opt-in，默认关闭；仅用于暴露度/beta 对照实验）──
        # 目的：判定"额外买入触发(如 OBV)的增量"是真 alpha，还是纯粹
        #       "交易更多 → 在市时间更长 → 吃到长牛 beta"的机械结果。
        # 机制：与 OBV 完全相同的准入条件（not_widening + market_allowed + 空仓），
        #       仅把"何时入场"换成伯努利随机触发（概率 random_entry_p）。
        #       出场逻辑与基线完全一致（Z>2 或 RSI>70 + 止损），不做任何改动。
        # 对照方法：扫描 p 得到"入场笔数→收益"响应曲线，再与 OBV 在同等
        #       入场笔数下比较。若随机组收益≈OBV，则 OBV 无超越暴露度的增量。
        self.random_entry = cfg.get("random_entry", False)
        self.random_entry_p = cfg.get("random_entry_p", 0.03)
        self.random_entry_seed = cfg.get("random_entry_seed", 42)
        # ── 市场环境门控（opt-in，默认关闭）──
        # 大盘处于下行趋势(基准指数 t-1 收盘 < MA(regime_ma))时禁止一切新开仓，
        # 消除"%B 背离/均值回归在熊市接飞刀"的告警。C0/C1 同等施加，隔离单一变量 pctb。
        self.market_regime_gate = cfg.get("market_regime_gate", False)
        self.regime_idx = cfg.get("regime_idx", "000300.SH")
        self.regime_ma = cfg.get("regime_ma", 60)
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

    def _load_market_regime_mask(self, data: pd.DataFrame) -> list[bool]:
        """构建大盘趋势门控掩码（与 data 行对齐，无未来函数）。

        规则：t-1 基准指数收盘 >= 其 MA(regime_ma) → 允许开仓；否则禁止。
        早期(MA 未成形/缺失)默认允许（保守）。
        """
        import sqlite3
        import config
        dates = [str(d) for d in data["trade_date"].tolist()]
        if not dates:
            return [True]
        lo, hi = dates[0], dates[-1]
        try:
            conn = sqlite3.connect(config.DATA["local_db_path"])
            df = pd.read_sql_query(
                "SELECT trade_date, close FROM index_daily "
                "WHERE ts_code=? AND trade_date BETWEEN ? AND ? ORDER BY trade_date",
                conn, params=(self.regime_idx, lo, hi))
            conn.close()
        except Exception:  # noqa BLE001
            return [True] * len(data)
        if df.empty:
            return [True] * len(data)
        df["trade_date"] = df["trade_date"].astype(str)
        s = df.set_index("trade_date")["close"].astype(float)
        s = s.reindex(dates).ffill()
        closes = s.values
        ma = pd.Series(closes).rolling(self.regime_ma).mean().values
        allowed = []
        for i in range(len(data)):
            if i == 0 or pd.isna(ma[i - 1]) or pd.isna(closes[i - 1]):
                allowed.append(True)
            else:
                allowed.append(closes[i - 1] >= ma[i - 1])
        return allowed



    def _enter_long(self, date, open_price, z_val, rsi_val, atr_val, reason_suffix=""):
        """执行买入（半/全仓 + 凯利封顶 + ATR 初始化），供即时买入与 headfake 确认复用。"""
        if self.position > 0:
            return False
        position_pct = 0.95 if self.position_mode == 'full' else 0.50
        buy_amount = self.cash * position_pct
        shares = int(buy_amount / open_price / 100) * 100
        if shares <= 0:
            return False
        reason = f"均值回归入场(Z={z_val:.2f},RSI={rsi_val:.1f}){reason_suffix}"
        success = self._kelly_buy(date, open_price, shares, reason=reason)
        if success and self.use_atr_sl:
            self.atr_sl.on_entry(entry_price=open_price, atr_val=atr_val)
        return success

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
        self._pending = None  # headfake 待确认状态（每次 run 复位）
        self._cancelled_count = 0  # 被判为假突破陷阱而取消的信号数
        self._confirmed_count = 0  # 经 headfake 确认成交的次数
        self._pctb_entries = 0  # %B 背离触发的买入次数
        self._pctb_overlap = 0  # 其中与 buy_signal 同日的重叠次数（归因用）
        self._mfi_suppressed = 0  # MFI 闸门过滤掉的 buy_signal 次数（归因）
        self._obv_entries = 0  # OBV 背离触发的买入次数
        self._obv_overlap = 0  # 其中与 buy_signal 同日的重叠次数（归因用）
        self._rand_entries = 0  # 随机入场控制组触发的买入次数
        self._rng = np.random.default_rng(self.random_entry_seed)  # 随机流（每次 run 复位，可复现）

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
        # ── MFI / OBV 量能指标（opt-in 量能确认用，仅当开启时计算）──
        high_c = data['adj_high'] if 'adj_high' in data.columns else data['adj_close']
        low_c = data['adj_low'] if 'adj_low' in data.columns else data['adj_close']
        vol = data['volume'] if 'volume' in data.columns else None
        if vol is not None:
            mfi_arr = ta.MFI(high_c.values, low_c.values, data['adj_close'].values,
                             vol.values, timeperiod=self.mfi_period)
            data['mfi'] = mfi_arr
            obv_arr = ta.OBV(data['adj_close'].values, vol.values)
            data['obv'] = obv_arr
        else:
            data['mfi'] = np.nan
            data['obv'] = np.nan

        prev_close = data['adj_close'].shift(1)
        prev_z = data['z_score'].shift(1)
        prev_rsi = data['rsi'].shift(1)
        prev_widening = data['band_widening'].shift(1)
        prev_lower = data['lower'].shift(1)
        prev_upper = data['upper'].shift(1)
        prev_mfi = data['mfi'].shift(1) if 'mfi' in data.columns else pd.Series(np.nan, index=data.index)
        prev_obv = data['obv'].shift(1) if 'obv' in data.columns else pd.Series(np.nan, index=data.index)

        # 处理 NaN：NaN 视为不满足条件（保守做法）
        # prev_z < -threshold：NaN → False（不入场）
        cond_z_low = (prev_z < -self.zscore_threshold).fillna(False)
        # prev_rsi < oversold：NaN → False（不入场）
        cond_rsi_low = (prev_rsi < self.rsi_oversold).fillna(False)
        # prev_widening：NaN → True（未知时保守视为布林带张开，禁止入场）
        # 然后用 ~ 取反（此时已无 NaN，~ 可安全使用）
        cond_not_widening = ~(prev_widening.fillna(True))
        data['cond_not_widening'] = cond_not_widening  # 供随机入场控制组复用同一准入条件

        # === %B 指标与看涨背离（opt-in 买入增强，须放在 cond_not_widening 之后）===
        # %B = (close - lower) / (upper - lower)，∈[0,1] 在下轨~上轨之间
        denom = (data['upper'] - data['lower']).replace(0, np.nan)
        data['pct_b'] = (data['adj_close'] - data['lower']) / denom
        prev_pctb = data['pct_b'].shift(1)
        # 看涨背离：用 t-1 与过去 K 日窗口极值比较（均属过去，无未来函数）。
        # 严格定义（非两点比较）：t-1 收盘创 K 日新低（窗口含 t-1），
        # 但 t-1 的 %B 未同步创 K 日新低 = 价格更便宜而布林带相对位置未更低
        # = 抛压枯竭、动能衰减预警。
        K = self.pctb_lookback
        prev_close_div = data['adj_close'].shift(1)
        price_lower = prev_close_div <= prev_close_div.rolling(K).min()
        pctb_higher = prev_pctb > prev_pctb.rolling(K).min()
        pctb_bull_div = (price_lower & pctb_higher & cond_not_widening).fillna(False)

        # === MFI 量能闸门 & OBV 量能背离（opt-in，须放在 cond_not_widening 之后）===
        # MFI 超卖确认闸门：下跌需伴随放量抛压(MFI<阈值)才确认"真超卖"，
        # 过滤无量阴跌。仅当 mfi_filter 开启时计入 buy_signal。
        cond_mfi_low = (prev_mfi < self.mfi_oversold).fillna(False)
        # OBV 看涨量能背离：价格创 K 日新低但 OBV 未同步创 K 日新低
        # (下跌未获量能确认=抛压枯竭/暗中吸筹) = 量能版看涨背离，平行 %B 背离。
        K_obv = self.obv_lookback
        obv_price_lower = prev_close_div <= prev_close_div.rolling(K_obv).min()
        obv_higher = prev_obv > prev_obv.rolling(K_obv).min()
        obv_bull_div = (obv_price_lower & obv_higher & cond_not_widening).fillna(False)

        # 入场信号：Z < -2 AND RSI < 30 AND 布林带未张开（双重共振）
        base_buy = cond_z_low & cond_rsi_low & cond_not_widening
        if self.mfi_filter:
            # 仅当 MFI 也超卖才允许入场；记录被过滤掉的基数（归因）
            buy_signal = base_buy & cond_mfi_low
            self._mfi_suppressed = int((base_buy & ~cond_mfi_low).sum())
        else:
            buy_signal = base_buy
            self._mfi_suppressed = 0

        # 出场信号：Z > 2 OR RSI > 70
        # NaN → False（不触发出场）
        cond_z_high = (prev_z > self.zscore_threshold).fillna(False)
        cond_rsi_high = (prev_rsi > self.rsi_overbought).fillna(False)
        sell_signal = cond_z_high | cond_rsi_high

        data['buy_signal'] = buy_signal
        data['sell_signal'] = sell_signal
        data['pctb_bull_div'] = pctb_bull_div
        data['obv_bull_div'] = obv_bull_div

        # ── 大盘趋势门控掩码（opt-in）──
        self.market_allowed = [True] * len(data)
        if self.market_regime_gate:
            self.market_allowed = self._load_market_regime_mask(data)

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
            # ── Head-fake 假突破确认（opt-in）──
            # 信号触发(价格击穿下轨)不立即买，改用「前一根」收盘是否重新站回下轨判定：
            #   站回下轨(顺轨站稳) → 假突破洗盘完成，确认买入；
            #   连续 2 根仍贴轨外下行 → 真破位陷阱，取消信号。
            # 用 i-1 的收盘/下轨判定、在 i 开盘成交，无未来函数。
            if self.headfake_filter and self._pending is not None and self.position == 0:
                wait = i - self._pending["start_i"]
                p_close = data.iloc[i - 1]['adj_close'] if i >= 1 else np.nan
                p_lower = data.iloc[i - 1]['lower'] if i >= 1 else np.nan
                if not pd.isna(p_lower) and not pd.isna(p_close):
                    if p_close >= p_lower:
                        success = self._enter_long(date, open_price, self._pending["z_val"],
                                        self._pending["rsi_val"], atr_arr[i],
                                        reason_suffix="[headfake确认]")
                        if success:
                            self._confirmed_count += 1
                        self._pending = None
                    elif wait >= 2:
                        self._cancelled_count += 1  # 陷阱：连续贴轨外，取消
                        self._pending = None

            # ── %B 看涨背离额外买入（opt-in，独立即时入场，不经 headfake 闸门）──
            if self.pctb_divergence and self.position == 0 and self._pending is None \
                    and self.market_allowed[i] and bool(data.iloc[i]['pctb_bull_div']):
                success = self._enter_long(date, open_price, 0.0, 0.0, atr_arr[i],
                                reason_suffix="[%B背离买]")
                if success:
                    self._pctb_entries += 1
                    if bool(data.iloc[i]['buy_signal']):
                        self._pctb_overlap += 1

            # ── OBV 看涨量能背离额外买入（opt-in，独立即时入场，不经 headfake 闸门）──
            if self.obv_divergence and self.position == 0 and self._pending is None \
                    and self.market_allowed[i] and bool(data.iloc[i]['obv_bull_div']):
                success = self._enter_long(date, open_price, 0.0, 0.0, atr_arr[i],
                                reason_suffix="[OBV背离买]")
                if success:
                    self._obv_entries += 1
                    if bool(data.iloc[i]['buy_signal']):
                        self._obv_overlap += 1

            # ── 随机入场控制组（opt-in，暴露度/beta 对照实验用）──
            # 与 OBV 相同准入条件（空仓 + 未待确认 + 大盘允许 + 布林带未张开），
            # 仅把入场时机换成伯努利随机触发；出场逻辑与基线完全一致。
            if self.random_entry and self.position == 0 and self._pending is None \
                    and self.market_allowed[i] and bool(data.iloc[i]['cond_not_widening']) \
                    and self._rng.random() < self.random_entry_p:
                success = self._enter_long(date, open_price, 0.0, 0.0, atr_arr[i],
                                reason_suffix="[随机入场]")
                if success:
                    self._rand_entries += 1

            if self.position == 0 and self._pending is None and self.market_allowed[i] \
                    and bool(data.iloc[i]['buy_signal']):
                z_val = float(prev_z.iloc[i]) if i < len(prev_z) else 0
                rsi_val = float(prev_rsi.iloc[i]) if i < len(prev_rsi) else 0
                if self.headfake_filter:
                    # 仅登记待确认，不立即成交
                    self._pending = {"start_i": i, "z_val": z_val, "rsi_val": rsi_val}
                else:
                    self._enter_long(date, open_price, z_val, rsi_val, atr_arr[i])

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
