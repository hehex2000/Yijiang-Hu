#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细交易明细输出工具
使用方法：
  python print_order_details.py --code 601808 --name 中海油服 --start-date 20200101 --end-date 20211231 --strategy dca
"""

import sys
import os
import pandas as pd
from datetime import datetime
from loguru import logger
import argparse

# 添加项目根目录到路径
project_root = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from backtest.data_loader import DataLoader
from backtest.dual_ma_strategy import DualMAStrategy
from backtest.turtle_strategy import TurtleStrategy
from backtest.dca_strategy import DCAStrategy
from backtest.metrics import calculate_metrics


# ========== 命令行参数 ==========
parser = argparse.ArgumentParser(description='输出详细交易明细')
parser.add_argument('--code', type=str, required=True, help='股票代码（如：601808）')
parser.add_argument('--name', type=str, required=True, help='股票名称（如：中海油服）')
parser.add_argument('--start-date', type=str, default='20200101', help='回测开始日期（格式:YYYYMMDD）')
parser.add_argument('--end-date', type=str, default='20211231', help='回测结束日期（格式:YYYYMMDD）')
parser.add_argument('--strategy', type=str, default='dca', choices=['dual_ma', 'turtle', 'dca'], help='策略类型：dual_ma, turtle, dca')
parser.add_argument('--ma-short', type=int, default=5, help='双均线短期均线周期（默认5）')
parser.add_argument('--ma-long', type=int, default=20, help='双均线长期均线周期（默认20）')
parser.add_argument('--turtle-period', type=int, default=20, help='海龟策略通道周期N（默认20）')
parser.add_argument('--capital', type=int, default=100000, help='初始资金（默认10万）')
parser.add_argument('--dca-amount', type=int, default=5000, help='月定投金额（默认5000元）')
args = parser.parse_args()


def run_backtest():
    """运行回测并返回交易记录"""
    logger.info(f"开始回测：{args.code} {args.name}")
    logger.info(f"周期：{args.start_date} 至 {args.end_date}")
    logger.info(f"策略：{args.strategy}")
    
    # 加载数据
    loader = DataLoader()
    df = loader.get_adjusted_prices(
        args.code, 
        args.start_date, 
        args.end_date,
        ma_short=args.ma_short,
        ma_long=args.ma_long,
        channel_period=args.turtle_period
    )
    
    if df is None or len(df) < 2:
        logger.error(f"{args.code} 数据不足")
        return None
    
    logger.info(f"  ✓ 加载了 {len(df)} 天的数据")
    
    # 根据策略类型运行回测
    if args.strategy == 'dual_ma':
        strategy = DualMAStrategy(
            capital=args.capital,
            ma_short=args.ma_short,
            ma_long=args.ma_long,
            take_profit=0.25,
            stop_loss=0.1
        )
        orders = strategy.run(df)
        strategy_name = f'双均线(MA{args.ma_short}/{args.ma_long})'
        
    elif args.strategy == 'turtle':
        strategy = TurtleStrategy(
            capital=args.capital,
            channel_period=args.turtle_period,
            position_mode='half',
            max_positions=3,
            take_profit=0.2,
            stop_loss=0.1
        )
        orders = strategy.run(df)
        strategy_name = f'海龟(N={args.turtle_period})'
        
    else:  # dca
        strategy = DCAStrategy(
            total_capital=args.capital,
            amount_per_month=args.dca_amount,
            take_profit=0.30,
            stop_loss=0.10
        )
        orders = strategy.run(df)
        strategy_name = f'月定投({args.dca_amount}元/月)'
    
    logger.info(f"  ✓ {strategy_name} 完成：{len(orders)} 笔交易")
    
    return {
        'code': args.code,
        'name': args.name,
        'strategy_name': strategy_name,
        'orders': orders,
        'df': df
    }


def print_order_details(result):
    """打印详细的交易明细"""
    code = result['code']
    name = result['name']
    strategy_name = result['strategy_name']
    orders = result['orders']
    
    print("\n" + "="*160)
    print(f"{code} {name} - {strategy_name} 交易明细")
    print("="*160)
    print(f"{'序号':<6} {'日期':<12} {'操作':<6} {'原因':<15} {'价格':>8} {'数量':>8} {'金额':>12} {'手续费':>10} {'印花税':>10} {'盈亏':>12}")
    print("-"*160)
    
    buy_count = 0
    sell_count = 0
    stop_profit_count = 0
    stop_loss_count = 0
    end_sell_count = 0
    
    for i, order in enumerate(orders, 1):
        date = order['date']
        action = order['action']
        reason = order.get('reason', 'N/A')
        price = order['price']
        shares = order.get('shares', 0)
        
        # 转换reason为中文
        reason_cn = {
            'monthly_dca': '定投买入',
            'stop_profit': '止盈卖出',
            'stop_loss': '止损卖出',
            'end_of_backtest': '清仓卖出',
            'golden_cross': '金叉买入',
            'death_cross': '死叉卖出',
            'breakout': '突破买入',
            'breakdown': '跌破卖出'
        }.get(reason, reason)
        
        if action == 'buy':
            amount = order['amount']
            fee = 0
            tax = 0
            profit = ''
            buy_count += 1
            print(f"{i:<6} {date:<12} {'买入':<6} {reason_cn:<15} {price:>8.2f} {shares:>8} {amount:>12.2f} {fee:>10.2f} {tax:>10.2f} {profit:>12}")
        else:  # sell
            amount = order.get('amount', 0)
            fee = order.get('trading_fee', 0)
            tax = order.get('stamp_duty', 0)
            profit = order.get('profit', 0)
            profit_str = f"{profit:.2f}" if profit else ''
            sell_count += 1
            
            # 统计卖出原因
            if reason == 'stop_profit':
                stop_profit_count += 1
            elif reason == 'stop_loss':
                stop_loss_count += 1
            elif reason == 'end_of_backtest':
                end_sell_count += 1
            
            print(f"{i:<6} {date:<12} {'卖出':<6} {reason_cn:<15} {price:>8.2f} {shares:>8} {amount:>12.2f} {fee:>10.2f} {tax:>10.2f} {profit_str:>12}")
    
    print("-"*160)
    print(f"买入笔数：{buy_count}，卖出笔数：{sell_count}")
    print(f"止盈卖出：{stop_profit_count}，止损卖出：{stop_loss_count}，到期清仓：{end_sell_count}")
    print("="*160 + "\n")


def save_to_file(result):
    """保存详细交易明细到文件"""
    code = result['code']
    name = result['name']
    strategy_name = result['strategy_name']
    orders = result['orders']
    
    filename = f"trade_details_{code}_{name}_{args.start_date}_to_{args.end_date}.txt"
    filepath = os.path.join(project_root, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"{code} {name} - {strategy_name} 交易明细\n")
        f.write("="*160 + "\n\n")
        f.write(f"{'序号':<6} {'日期':<12} {'操作':<6} {'原因':<15} {'价格':>8} {'数量':>8} {'金额':>12} {'手续费':>10} {'印花税':>10} {'盈亏':>12}\n")
        f.write("-"*160 + "\n")
        
        buy_count = 0
        sell_count = 0
        stop_profit_count = 0
        stop_loss_count = 0
        end_sell_count = 0
        
        for i, order in enumerate(orders, 1):
            date = order['date']
            action = order['action']
            reason = order.get('reason', 'N/A')
            price = order['price']
            shares = order.get('shares', 0)
            
            # 转换reason为中文
            reason_cn = {
                'monthly_dca': '定投买入',
                'stop_profit': '止盈卖出',
                'stop_loss': '止损卖出',
                'end_of_backtest': '清仓卖出',
                'golden_cross': '金叉买入',
                'death_cross': '死叉卖出',
                'breakout': '突破买入',
                'breakdown': '跌破卖出'
            }.get(reason, reason)
            
            if action == 'buy':
                amount = order['amount']
                fee = 0
                tax = 0
                profit = ''
                buy_count += 1
                f.write(f"{i:<6} {date:<12} {'买入':<6} {reason_cn:<15} {price:>8.2f} {shares:>8} {amount:>12.2f} {fee:>10.2f} {tax:>10.2f} {profit:>12}\n")
            else:  # sell
                amount = order.get('amount', 0)
                fee = order.get('trading_fee', 0)
                tax = order.get('stamp_duty', 0)
                profit = order.get('profit', 0)
                profit_str = f"{profit:.2f}" if profit else ''
                sell_count += 1
                
                # 统计卖出原因
                if reason == 'stop_profit':
                    stop_profit_count += 1
                elif reason == 'stop_loss':
                    stop_loss_count += 1
                elif reason == 'end_of_backtest':
                    end_sell_count += 1
                
                f.write(f"{i:<6} {date:<12} {'卖出':<6} {reason_cn:<15} {price:>8.2f} {shares:>8} {amount:>12.2f} {fee:>10.2f} {tax:>10.2f} {profit_str:>12}\n")
        
        f.write("-"*160 + "\n")
        f.write(f"买入笔数：{buy_count}，卖出笔数：{sell_count}\n")
        f.write(f"止盈卖出：{stop_profit_count}，止损卖出：{stop_loss_count}，到期清仓：{end_sell_count}\n")
        f.write("\n" + "="*160 + "\n")
    
    logger.info(f"✓ 交易明细已保存到：{filepath}")
    return filepath


if __name__ == '__main__':
    result = run_backtest()
    if result:
        print_order_details(result)
        save_to_file(result)
        logger.info("✓ 完成！")
    else:
        logger.error("✗ 回测失败")
