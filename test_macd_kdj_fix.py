"""
测试 MACD/KDJ 策略修复 - 带调试信息
"""
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backtest'))

from data_loader import DataLoader
from macd_kdj_strategy import MACDKDJStrategy
from metrics import calculate_metrics

# 初始化
DB_PATH = 'D:/tu-shareData/astock_daily.db'
data_loader = DataLoader(db_path=DB_PATH)

# 测试一只股票
code = '300308'
name = '中际旭创'
START_DATE = '20230101'
END_DATE = '20231231'

print(f"\n测试 MACD/KDJ 策略: {code} - {name}")
print("=" * 60)

# 加载数据
df = data_loader.get_adjusted_prices(
    code, START_DATE, END_DATE,
    ma_short=10, ma_long=60, channel_period=30
)

if df is None or len(df) == 0:
    print(f"无数据: {code}")
    sys.exit(1)

print(f"✓ 加载数据: {len(df)} 个交易日")
print(f"  价格范围: {df['adj_close'].min():.2f} - {df['adj_close'].max():.2f}")

# 手动计算指标，检查信号
strategy = MACDKDJStrategy(total_capital=200000.0)
dif, dea, macd_hist = strategy.calculate_macd(df['adj_close'])
k_value, d_value, j_value = strategy.calculate_kdj(
    df['adj_high'], df['adj_low'], df['adj_close']
)

# 生成信号
macd_golden = (dif > dea) & (dif.shift(1) <= dea.shift(1))
macd_death = (dif < dea) & (dif.shift(1) >= dea.shift(1))

kdj_golden = (k_value > d_value) & (k_value.shift(1) <= d_value.shift(1))
kdj_death = (k_value < d_value) & (k_value.shift(1) >= d_value.shift(1))

# 买入信号：MACD金叉 OR KDJ金叉 OR J值 < 20（放松条件）
buy_signal = macd_golden | kdj_golden | (j_value < 20)

# 卖出信号：MACD死叉 OR KDJ死叉 OR J值 > 80
sell_signal = macd_death | kdj_death | (j_value > 80)

# 统计信号
print(f"\n信号统计:")
print(f"  MACD金叉: {macd_golden.sum()} 次")
print(f"  MACD死叉: {macd_death.sum()} 次")
print(f"  KDJ金叉: {kdj_golden.sum()} 次")
print(f"  KDJ死叉: {kdj_death.sum()} 次")
print(f"  J值 < 20: {(j_value < 20).sum()} 天")
print(f"  J值 > 80: {(j_value > 80).sum()} 天")
print(f"  买入信号: {buy_signal.sum()} 次")
print(f"  卖出信号: {sell_signal.sum()} 次")

# 检查前5个买入信号
buy_indices = df.index[buy_signal].tolist()
if len(buy_indices) > 0:
    print(f"\n前5个买入信号:")
    for i, idx in enumerate(buy_indices[:5], 1):
        row = df.iloc[idx]
        print(f"  {i}. {row['trade_date']} "
              f"MACD金叉={macd_golden.iloc[idx]} "
              f"KDJ金叉={kdj_golden.iloc[idx]} "
              f"J值={j_value.iloc[idx]:.2f}")
else:
    print(f"\n✗ 没有买入信号！策略永远不会买入！")
    print(f"  可能原因：")
    print(f"    1. MACD金叉条件太严格")
    print(f"    2. KDJ金叉条件太严格")
    print(f"    3. J值 < 20 的条件太严格")
    print(f"  建议：放松买入条件")

# 运行策略
print(f"\n运行策略...")
orders = strategy.run(df)

print(f"✓ 策略运行完成")
print(f"  交易次数: {len(orders)}")
print(f"  daily_values 长度: {len(strategy.daily_values)}")

# 计算指标
if len(strategy.daily_values) > 0:
    metrics = calculate_metrics(orders, pd.DataFrame(strategy.daily_values))
    print(f"\n✓ 性能指标:")
    print(f"  总收益率: {metrics['total_return']:.2%}")
    print(f"  年化收益率: {metrics['annualized_return']:.2%}")
    print(f"  最大回撤: {metrics['max_drawdown']:.2%}")
    print(f"  夏普比率: {metrics['sharpe_ratio']:.2f}")
    print(f"  交易次数: {metrics['num_trades']}")
    print(f"  胜率: {metrics['win_rate']:.1%}")
else:
    print("\n✗ 错误: daily_values 为空，无法计算指标")
