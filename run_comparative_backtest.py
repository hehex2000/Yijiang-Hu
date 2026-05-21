"""
多策略对比回测脚本 - 集成4种策略
- MACD/RSI组合策略
- MACD/KDJ组合策略  
- 双均线策略(MA5/20)
- 海龟策略(N=40)

输出：策略对比表 + 详细交易记录
"""

import pandas as pd
import numpy as np
import yaml
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backtest.macd_rsi_strategy import MACDRSIStrategy
from backtest.macd_kdj_strategy import MACDKDJStrategy
from backtest.dual_ma_strategy import DualMAStrategy
from backtest.turtle_strategy import TurtleStrategy
from backtest.data_loader import DataLoader


def load_config(config_path='config/backtest_config.yaml'):
    """加载配置文件"""
    config_file = project_root / config_path
    
    if not config_file.exists():
        print(f"警告: 配置文件不存在 {config_file}，使用默认配置")
        return get_default_config()
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def get_default_config():
    """获取默认配置"""
    return {
        'macd_rsi': {
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'take_profit': 0.25,
            'stop_loss': 0.10
        },
        'macd_kdj': {
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'kdj_n': 9,
            'kdj_m1': 3,
            'kdj_m2': 3,
            'kdj_overbought': 80,
            'kdj_oversold': 20,
            'take_profit': 0.25,
            'stop_loss': 0.10
        },
        'dual_ma': {
            'short_period': 5,
            'long_period': 20,
            'take_profit': 0.25,
            'stop_loss': 0.10
        },
        'turtle': {
            'n_period': 40,
            'take_profit': 0.25,
            'stop_loss': 0.10,
            'trailing_stop': True,
            'trailing_pct': 0.20
        }
    }


def calculate_metrics(orders: List[Dict], initial_cash: float = 100000) -> Dict:
    """
    根据交易记录计算策略指标
    
    Args:
        orders: 交易记录列表
        initial_cash: 初始资金
        
    Returns:
        指标字典 {'total_return', 'win_rate', 'trade_count'}
    """
    if not orders:
        return {
            'total_return': 0.0,
            'win_rate': 0.0,
            'trade_count': 0
        }
    
    # 计算最终资金
    cash = initial_cash
    position = 0
    
    for order in orders:
        if order['action'] == 'buy':
            cost = order['price'] * order['shares'] * 1.0002  # 手续费
            cash -= cost
            position += order['shares']
        elif order['action'] == 'sell':
            revenue = order['price'] * order['shares'] * 0.9998  # 扣除手续费和印花税
            cash += revenue
            position = 0
    
    # 如果还有持仓，按最后价格计算
    if position > 0:
        last_price = orders[-1]['price']
        cash += position * last_price * 0.9998
    
    final_value = cash
    total_return = (final_value - initial_cash) / initial_cash
    
    # 计算胜率
    buy_orders = [o for o in orders if o['action'] == 'buy']
    sell_orders = [o for o in orders if o['action'] == 'sell']
    
    win_count = 0
    for i, sell_order in enumerate(sell_orders):
        if i < len(buy_orders):
            buy_order = buy_orders[i]
            if sell_order['price'] > buy_order['price']:
                win_count += 1
    
    win_rate = win_count / len(sell_orders) if sell_orders else 0.0
    
    return {
        'total_return': total_return,
        'win_rate': win_rate,
        'trade_count': len(orders)
    }


def run_backtest_for_symbol(symbol, config, start_date, end_date, data_loader):
    """对单个股票运行所有策略的回测"""
    print(f"\n{'='*60}")
    print(f"回测股票: {symbol}")
    print(f"回测区间: {start_date} ~ {end_date}")
    print(f"{'='*60}")
    
    # 加载数据
    df = data_loader.get_adjusted_prices(
        symbol, 
        start_date.replace('-', ''), 
        end_date.replace('-', ''),
        ma_short=5,
        ma_long=20,
        channel_period=40
    )
    
    if df is None or df.empty:
        print(f"错误: 无法加载股票 {symbol} 的数据")
        return None
    
    # 转换数据格式以适配策略类
    df_strategy = df.copy()
    df_strategy.index = pd.to_datetime(df_strategy['trade_date'])
    df_strategy['close'] = df_strategy['adj_close']
    if 'adj_high' in df_strategy.columns:
        df_strategy['high'] = df_strategy['adj_high']
        df_strategy['low'] = df_strategy['adj_low']
        df_strategy['open'] = df_strategy['adj_open']
    
    # 过滤日期（策略不接受 end_date 参数）
    df_strategy = df_strategy[df_strategy.index <= pd.Timestamp(end_date)]
    
    # 初始化策略
    strategies = {
        'MACD/RSI': MACDRSIStrategy(config),
        'MACD/KDJ': MACDKDJStrategy(config),
        '双均线(MA5/20)': DualMAStrategy(config),
        '海龟(N=40)': TurtleStrategy(config)
    }
    
    results = {}
    
    # 运行每个策略
    for strategy_name, strategy in strategies.items():
        print(f"\n--- {strategy_name} 策略 ---")
        
        try:
            # 调用策略的 run() 方法（只接受 df 参数）
            orders = strategy.run(df_strategy)
            
            # 计算指标
            metrics = calculate_metrics(orders)
            
            # 输出结果
            print(f"  总收益率: {metrics['total_return']*100:.2f}%")
            print(f"  胜率: {metrics['win_rate']*100:.2f}%")
            print(f"  交易次数: {metrics['trade_count']}")
            
            # 输出交易记录（前5笔）
            if orders and len(orders) > 0:
                print(f"  交易记录（前5笔）:")
                for i, order in enumerate(orders[:5], 1):
                    date_str = order['date'].strftime('%Y-%m-%d') if hasattr(order['date'], 'strftime') else str(order['date'])
                    print(f"    {i}. {date_str} | {order['action']} | "
                          f"价格:{order['price']:.2f} | 原因:{order['reason']}")
            
            results[strategy_name] = {
                'orders': orders,
                'metrics': metrics,
                'daily_values': strategy.daily_values if hasattr(strategy, 'daily_values') else []
            }
            
        except Exception as e:
            print(f"  错误: 策略运行失败 - {e}")
            import traceback
            traceback.print_exc()
            results[strategy_name] = None
    
    return results


def generate_comparison_table(all_results, output_dir=None):
    """生成策略对比表"""
    print(f"\n{'='*60}")
    print("生成策略对比表")
    print(f"{'='*60}")
    
    # 创建对比表
    comparison_data = []
    
    for symbol, results in all_results.items():
        if results is None:
            continue
        
        row = {'股票代码': symbol}
        
        for strategy_name, result in results.items():
            if result is None:
                row[f"{strategy_name}_收益率"] = "N/A"
                row[f"{strategy_name}_胜率"] = "N/A"
                row[f"{strategy_name}_交易次数"] = "N/A"
            else:
                metrics = result['metrics']
                row[f"{strategy_name}_收益率"] = f"{metrics['total_return']*100:.2f}%"
                row[f"{strategy_name}_胜率"] = f"{metrics['win_rate']*100:.2f}%"
                row[f"{strategy_name}_交易次数"] = metrics['trade_count']
        
        comparison_data.append(row)
    
    # 转换为DataFrame
    df_comparison = pd.DataFrame(comparison_data)
    
    # 输出到控制台
    print("\n策略对比表:")
    print(df_comparison.to_string(index=False))
    
    # 保存到文件
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_file = output_dir / f"backtest_comparison_{timestamp}.csv"
        txt_file = output_dir / f"backtest_comparison_{timestamp}.txt"
        
        df_comparison.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"\n对比表已保存到: {csv_file}")
        
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("多策略回测对比表\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
            f.write(df_comparison.to_string(index=False))
            f.write("\n")
        print(f"文本格式已保存到: {txt_file}")
    
    return df_comparison


def main():
    """主函数"""
    print("="*60)
    print("多策略回测系统")
    print("="*60)
    
    # 加载配置
    config = load_config()
    
    # 初始化数据加载器
    from backtest.data_loader import DB_PATH
    print(f"\n数据库路径: {DB_PATH}")
    
    if not os.path.exists(DB_PATH):
        print(f"错误: 数据库文件不存在 {DB_PATH}")
        print("请检查配置文件中的数据库路径")
        return
    
    data_loader = DataLoader()
    
    # 测试股票列表
    symbols = ['601919', '601633', '300274', '002304', '600809']
    
    # 回测时间区间
    start_date = '2021-01-01'
    end_date = '2022-12-31'
    
    print(f"\n回测配置:")
    print(f"  股票数量: {len(symbols)}")
    print(f"  回测区间: {start_date} ~ {end_date}")
    print(f"  策略数量: 4 (MACD/RSI, MACD/KDJ, 双均线, 海龟)")
    
    # 运行回测
    all_results = {}
    
    for symbol in symbols:
        results = run_backtest_for_symbol(symbol, config, start_date, end_date, data_loader)
        if results:
            all_results[symbol] = results
    
    # 生成对比表
    if all_results:
        output_dir = project_root / "data" / "results"
        df_comparison = generate_comparison_table(all_results, output_dir)
        
        print(f"\n{'='*60}")
        print("回测完成！")
        print(f"{'='*60}")
    else:
        print("\n错误: 没有成功完成任何回测")


if __name__ == '__main__':
    main()
