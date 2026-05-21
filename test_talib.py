"""
测试 TA-Lib 功能
"""
import talib
import numpy as np
import pandas as pd

print("\n" + "="*70)
print("测试 TA-Lib 功能")
print("="*70 + "\n")

# 生成模拟价格数据
np.random.seed(42)
close_prices = np.cumsum(np.random.randn(100)) + 100  # 随机游走
high_prices = close_prices + np.random.uniform(0, 2, 100)
low_prices = close_prices - np.random.uniform(0, 2, 100)

print("[1/6] 计算 SMA（简单移动平均线）...")
sma_20 = talib.SMA(close_prices, timeperiod=20)
print(f"  SMA(20) 最后一个值: {sma_20[-1]:.2f}")
print(f"  [OK] SMA 计算成功\n")

print("[2/6] 计算 EMA（指数移动平均线）...")
ema_20 = talib.EMA(close_prices, timeperiod=20)
print(f"  EMA(20) 最后一个值: {ema_20[-1]:.2f}")
print(f"  [OK] EMA 计算成功\n")

print("[3/6] 计算 RSI...")
rsi = talib.RSI(close_prices, timeperiod=14)
print(f"  RSI(14) 最后一个值: {rsi[-1]:.2f}")
print(f"  [OK] RSI 计算成功\n")

print("[4/6] 计算 MACD...")
macd, macd_signal, macd_hist = talib.MACD(
    close_prices, 
    fastperiod=12, 
    slowperiod=26, 
    signalperiod=9
)
print(f"  MACD: {macd[-1]:.2f}")
print(f"  Signal: {macd_signal[-1]:.2f}")
print(f"  Histogram: {macd_hist[-1]:.2f}")
print(f"  [OK] MACD 计算成功\n")

print("[5/6] 计算布林带...")
upper, middle, lower = talib.BBANDS(
    close_prices, 
    timeperiod=20, 
    nbdevup=2, 
    nbdevdn=2, 
    matype=0
)
print(f"  上轨: {upper[-1]:.2f}")
print(f"  中轨: {middle[-1]:.2f}")
print(f"  下轨: {lower[-1]:.2f}")
print(f"  [OK] 布林带计算成功\n")

print("[6/6] 计算 KDJ (Stochastic)...")
slowk, slowd = talib.STOCH(
    high_prices, 
    low_prices, 
    close_prices,
    fastk_period=9,
    slowk_period=3,
    slowk_matype=0,
    slowd_period=3,
    slowd_matype=0
)
print(f"  K值: {slowk[-1]:.2f}")
print(f"  D值: {slowd[-1]:.2f}")
print(f"  [OK] KDJ 计算成功\n")

print("="*70)
print("✅ 所有 TA-Lib 功能测试通过！")
print("="*70 + "\n")

print("常用函数列表：")
print("  移动平均线: SMA, EMA, WMA, DEMA, TEMA, TRIMA")
print("  动量指标: MACD, RSI, STOCH (KDJ), CCI, ROC, MOM")
print("  成交量指标: AD, ADOSC, OBV")
print("  波动率指标: ATR, NATR, TRANGE")
print("  形态识别: CDLENGULFING, CDLHAMMER, ...")
print("")
print("完整函数列表: http://ta-lib.org/functions.html")
print("")
