"""
回测脚本 - 运行3个策略（双均线MA10/30，海龟N=20，月定投5000）
测试周期：2023年一整年
每支股票初始资金10万，年末清仓卖出
"""

import sys
import os
import pandas as pd
from datetime import datetime
from loguru import logger

# 添加项目根目录到路径
project_root = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from backtest.data_loader import DataLoader
from backtest.dual_ma_strategy import DualMAStrategy
from backtest.turtle_strategy import TurtleStrategy
from backtest.dca_strategy import DCAStrategy
from backtest.metrics import calculate_metrics

# 配置参数
STOCKS = [
    ('002384', '东山精密'),
    ('300308', '中际旭创'),
    ('600522', '中天科技'),
    ('000338', '潍柴动力'),
    ('600183', '生益科技'),
]

START_DATE = '20230101'
END_DATE = '20231231'
TOTAL_CAPITAL = 100000  # 每支股票初始资金10万
DCA_AMOUNT_PER_MONTH = 5000  # 月定投5000元
TURTLE_CHANNEL_PERIOD = 20  # 海龟策略N=20


def run_backtest_for_stock(stock_code: str, stock_name: str):
    """
    对单支股票运行3个策略，返回回测结果
    """
    logger.info("="*60)
    logger.info(f"开始回测：{stock_code} {stock_name}")
    logger.info("="*60)
    
    # 1. 加载数据（双均线用MA10/MA30，海龟用N=20）
    loader = DataLoader()
    
    # 双均线策略数据（MA10/MA30）
    df_dual_ma = loader.get_adjusted_prices(
        stock_code, START_DATE, END_DATE,
        ma_short=10, ma_long=30, channel_period=TURTLE_CHANNEL_PERIOD
    )
    
    if df_dual_ma is None or len(df_dual_ma) < 30:
        logger.warning(f"{stock_code} 数据不足，跳过")
        return None
    
    results = {}
    
    # 2. 运行双均线策略（MA10/MA30）
    logger.info("  运行双均线策略（MA10/MA30）...")
    strategy_dual_ma = DualMAStrategy(
        total_capital=TOTAL_CAPITAL,
        take_profit=0.25,
        stop_loss=0.10
    )
    orders_dual_ma = strategy_dual_ma.run(df_dual_ma)
    
    # 转换 daily_values 为 DataFrame
    daily_df_dual_ma = pd.DataFrame(strategy_dual_ma.daily_values)
    metrics_dual_ma = calculate_metrics(orders_dual_ma, daily_df_dual_ma)
    
    # 年末清仓
    final_price = df_dual_ma.iloc[-1]['adj_close']
    final_value_dual_ma = strategy_dual_ma.cash + strategy_dual_ma.position_shares * final_price
    
    results['双均线(MA10/30)'] = {
        'orders': orders_dual_ma,
        'metrics': metrics_dual_ma,
        'final_value': final_value_dual_ma
    }
    logger.info(f"  ✓ 双均线完成：{len(orders_dual_ma)} 笔交易，最终资产 {final_value_dual_ma:.2f}")
    
    # 3. 运行海龟策略（N=20）
    logger.info(f"  运行海龟策略（N={TURTLE_CHANNEL_PERIOD}）...")
    strategy_turtle = TurtleStrategy(
        total_capital=TOTAL_CAPITAL,
        channel_period=TURTLE_CHANNEL_PERIOD,
        position_mode='half',
        max_positions=3,
        take_profit=0.20,
        stop_loss=0.10
    )
    orders_turtle = strategy_turtle.run(df_dual_ma)
    
    # 转换 daily_values 为 DataFrame
    daily_df_turtle = pd.DataFrame(strategy_turtle.daily_values)
    metrics_turtle = calculate_metrics(orders_turtle, daily_df_turtle)
    
    final_value_turtle = strategy_turtle.cash + strategy_turtle.position_shares * final_price
    
    results['海龟(N=20)'] = {
        'orders': orders_turtle,
        'metrics': metrics_turtle,
        'final_value': final_value_turtle
    }
    logger.info(f"  ✓ 海龟完成：{len(orders_turtle)} 笔交易，最终资产 {final_value_turtle:.2f}")
    
    # 4. 运行月定投策略（每月5000元）
    logger.info(f"  运行月定投策略（每月{DCA_AMOUNT_PER_MONTH}元）...")
    df_dca = df_dual_ma[['trade_date', 'adj_open', 'adj_close']].copy()
    
    strategy_dca = DCAStrategy(
        total_capital=TOTAL_CAPITAL,
        amount_per_month=DCA_AMOUNT_PER_MONTH,
        take_profit=0.30,
        stop_loss=0.10
    )
    orders_dca = strategy_dca.run(df_dca)
    
    # 转换 daily_values 为 DataFrame
    daily_df_dca = pd.DataFrame(strategy_dca.daily_values)
    metrics_dca = calculate_metrics(orders_dca, daily_df_dca)
    
    # 计算DCA策略最终资产
    dca_total_shares = sum(lot['shares'] for lot in strategy_dca.lots)
    final_value_dca = strategy_dca.cash + dca_total_shares * final_price
    
    results['月定投(5000元/月)'] = {
        'orders': orders_dca,
        'metrics': metrics_dca,
        'final_value': final_value_dca
    }
    logger.info(f"  ✓ 定投完成：{len(orders_dca)} 笔交易，最终资产 {final_value_dca:.2f}")
    
    return {
        'stock_code': stock_code,
        'stock_name': stock_name,
        'results': results
    }


def main():
    """主函数：运行所有股票的回测，输出对比结果"""
    logger.info("="*60)
    logger.info("开始2023年回测（双均线MA10/30，海龟N=20，月定投5000）")
    logger.info("="*60)
    
    all_results = []
    
    for stock_code, stock_name in STOCKS:
        result = run_backtest_for_stock(stock_code, stock_name)
        if result:
            all_results.append(result)
    
    # 输出汇总报告
    print("\n" + "="*100)
    print("2023年回测结果汇总")
    print("="*100)
    print(f"{'股票':<15} {'策略':<20} {'交易次数':>8} {'最终资产':>12} {'总收益率':>12} {'年化收益':>12}")
    print("-"*100)
    
    for result in all_results:
        stock_code = result['stock_code']
        stock_name = result['stock_name']
        stock_label = f"{stock_code} {stock_name[:6]}"
        
        for strategy_name, data in result['results'].items():
            metrics = data['metrics']
            num_orders = len(data['orders'])
            final_value = data['final_value']
            total_return = (final_value - TOTAL_CAPITAL) / TOTAL_CAPITAL
            
            print(f"{stock_label:<15} {strategy_name:<20} {num_orders:>8} {final_value:>11.2f} {total_return:>11.2%}")
        
        print("-"*100)
    
    print("="*100)
    logger.info("回测完成！")
    
    return all_results


if __name__ == '__main__':
    main()
