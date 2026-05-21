#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
牛熊市策略表现对比回测
分别统计牛市和熊市中各策略的表现
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
from backtest.macd_rsi_strategy import MACDRSIStrategy
from backtest.macd_kdj_strategy import MACDKDJStrategy
from backtest.buy_and_hold_strategy import BuyAndHoldStrategy
from backtest.dca_strategy import DCAStrategy
from backtest.market_state import MarketStateDetector
from backtest.metrics import calculate_metrics


def split_bull_bear_periods(df: pd.DataFrame, detector: MarketStateDetector) -> dict:
    """
    将回测区间按牛熊市分割成多个子区间
    
    Returns:
        {
            'bull': [(start_date, end_date), ...],
            'bear': [(start_date, end_date), ...]
        }
    """
    periods = {'bull': [], 'bear': []}
    current_state = None
    period_start = None
    
    for i in range(len(df)):
        date_str = str(df.iloc[i]['trade_date'])
        state = detector.get_market_state(date_str)
        
        if current_state is None:
            current_state = state
            period_start = i
        elif state != current_state:
            # 状态切换，保存上一个区间
            if current_state in ['bull', 'bear']:
                periods[current_state].append((period_start, i - 1))
            current_state = state
            period_start = i
    
    # 保存最后一个区间
    if current_state in ['bull', 'bear'] and period_start is not None:
        periods[current_state].append((period_start, len(df) - 1))
    
    return periods


def run_backtest_for_stock(stock_code: str, start_date: str, end_date: str):
    """对单只股票运行5种策略，返回结果"""
    print(f"\n{'='*60}")
    print(f"回测股票: {stock_code}")
    print(f"{'='*60}")
    
    # 加载数据
    data_loader = DataLoader()
    df = data_loader.get_adjusted_prices(
        stock_code, 
        start_date, 
        end_date,
        ma_short=5,
        ma_long=20,
        channel_period=40
    )
    
    if df is None or len(df) == 0:
        print(f"  ✗ 无数据: {stock_code}")
        return None
    
    print(f"  数据加载成功: {len(df)} 天")
    
    # 市场状态检测
    detector = MarketStateDetector(ma_period=200)
    
    results = {}
    
    # 1. 买入持有策略
    print(f"\n  [1/5] 买入持有策略...")
    bah = BuyAndHoldStrategy(
        total_capital=200000,
        trading_fee_rate=0.0002,
        stamp_duty_rate=0.001
    )
    bah_orders = bah.run(df)
    df_daily_bah = pd.DataFrame(bah.daily_values)
    bah_metrics = calculate_metrics(bah_orders, df_daily_bah, 200000)
    results['买入持有'] = {
        'orders': bah_orders,
        'total_return': bah_metrics.get('total_return', 0),
        'win_rate': bah_metrics.get('win_rate', 0),
        'trade_count': len(bah_orders)
    }
    print(f"    总收益率: {bah_metrics.get('total_return', 0)*100:.2f}%")
    print(f"    交易次数: {len(bah_orders)}")
    
    # 2. 月定投策略（无止损止盈）
    print(f"\n  [2/5] 月定投策略（无TP/SL）...")
    dca = DCAStrategy(
        total_capital=200000,
        amount_per_month=5000,
        take_profit=0.30,
        stop_loss=0.20,
        enable_tp_sl=False,  # 关闭止盈止损
        trading_fee_rate=0.0002,
        stamp_duty_rate=0.001
    )
    dca_orders = dca.run(df)
    df_daily_dca = pd.DataFrame(dca.daily_values)
    dca_metrics = calculate_metrics(dca_orders, df_daily_dca, 200000)
    results['月定投(无TP/SL)'] = {
        'orders': dca_orders,
        'total_return': dca_metrics.get('total_return', 0),
        'win_rate': dca_metrics.get('win_rate', 0),
        'trade_count': len(dca_orders)
    }
    print(f"    总收益率: {dca_metrics.get('total_return', 0)*100:.2f}%")
    print(f"    交易次数: {len(dca_orders)}")
    
    # 3. 双均线策略 (MA5/20)
    print(f"\n  [3/5] 双均线策略 (MA5/20)...")
    dual_ma = DualMAStrategy(
        total_capital=200000,
        take_profit=0.25,
        stop_loss=0.10
    )
    dual_ma_orders = dual_ma.run(df)
    df_daily_dual = pd.DataFrame(dual_ma.daily_values)
    dual_ma_metrics = calculate_metrics(dual_ma_orders, df_daily_dual, 200000)
    results['双均线(MA5/20)'] = {
        'orders': dual_ma_orders,
        'total_return': dual_ma_metrics.get('total_return', 0),
        'win_rate': dual_ma_metrics.get('win_rate', 0),
        'trade_count': len(dual_ma_orders)
    }
    print(f"    总收益率: {dual_ma_metrics.get('total_return', 0)*100:.2f}%")
    print(f"    胜率: {dual_ma_metrics.get('win_rate', 0)*100:.2f}%")
    print(f"    交易次数: {len(dual_ma_orders)}")
    
    # 4. MACD/KDJ策略
    print(f"\n  [4/5] MACD/KDJ策略...")
    macd_kdj = MACDKDJStrategy(
        total_capital=200000,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        kdj_n=9,
        kdj_m1=3,
        kdj_m2=3,
        take_profit=0.25,
        stop_loss=0.10
    )
    macd_kdj_result = macd_kdj.run_backtest(df)
    results['MACD/KDJ'] = {
        'orders': macd_kdj_result['trades'],
        'total_return': macd_kdj_result['total_return'],
        'win_rate': macd_kdj_result['win_rate'],
        'trade_count': len(macd_kdj_result['trades'])
    }
    print(f"    总收益率: {macd_kdj_result['total_return']*100:.2f}%")
    print(f"    胜率: {macd_kdj_result['win_rate']*100:.2f}%")
    print(f"    交易次数: {len(macd_kdj_result['trades'])}")
    
    # 5. MACD/RSI策略
    print(f"\n  [5/5] MACD/RSI策略...")
    macd_rsi = MACDRSIStrategy(
        total_capital=200000,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        rsi_period=14,
        rsi_overbought=70,
        rsi_oversold=30,
        take_profit=0.25,
        stop_loss=0.10
    )
    macd_rsi_orders = macd_rsi.run(df)
    df_daily_rsi = pd.DataFrame(macd_rsi.daily_values)
    macd_rsi_metrics = calculate_metrics(macd_rsi_orders, df_daily_rsi, 200000)
    results['MACD/RSI'] = {
        'orders': macd_rsi_orders,
        'total_return': macd_rsi_metrics.get('total_return', 0),
        'win_rate': macd_rsi_metrics.get('win_rate', 0),
        'trade_count': len(macd_rsi_orders)
    }
    print(f"    总收益率: {macd_rsi_metrics.get('total_return', 0)*100:.2f}%")
    print(f"    胜率: {macd_rsi_metrics.get('win_rate', 0)*100:.2f}%")
    print(f"    交易次数: {len(macd_rsi_orders)}")
    
    return results


def main():
    """主函数"""
    print("="*60)
    print("多策略回测 - 牛熊市对比")
    print("="*60)
    
    # 测试股票列表（2022年初选出的沪深300 Top5）
    stocks = ['300498', '600015', '601916', '601229', '600011']
    
    # 回测参数：2022年全年（熊市）
    start_date = '20220101'
    end_date = '20221231'
    
    # 存储所有结果
    all_results = {}
    
    # 对每只股票运行回测
    for stock_code in stocks:
        results = run_backtest_for_stock(stock_code, start_date, end_date)
        if results:
            all_results[stock_code] = results
    
    # 生成比对表
    print(f"\n\n{'='*80}")
    print("回测结果比对表（2022年全年）")
    print(f"{'='*80}\n")
    
    # 表头
    header = f"{'股票':<8} {'策略':<18} {'总收益率':<12} {'胜率':<10} {'交易次数':<10}"
    print(header)
    print("-"*80)
    
    # 表格内容
    strategy_order = ['买入持有', '月定投(无TP/SL)', '双均线(MA5/20)', 'MACD/KDJ', 'MACD/RSI']
    for stock_code in stocks:
        if stock_code in all_results:
            results = all_results[stock_code]
            
            for strategy_name in strategy_order:
                if strategy_name in results:
                    r = results[strategy_name]
                    # 买入持有没有胜率概念，显示 -
                    if strategy_name == '买入持有':
                        win_str = '-'
                    else:
                        win_str = f"{r['win_rate']*100:>8.2f}%"
                    print(f"{stock_code:<8} {strategy_name:<18} {r['total_return']*100:>10.2f}% {win_str:<10} {r['trade_count']:>10}")
            
            print("-"*80)
    
    print(f"\n回测完成！")
    
    # 保存结果到文件（backtest/result 目录）
    result_dir = os.path.join(project_root, 'backtest', 'result')
    os.makedirs(result_dir, exist_ok=True)
    output_file = os.path.join(result_dir, f"backtest_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    print(f"\n结果已保存到: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("多策略回测结果比对表（2022年全年）\n")
        f.write("="*80 + "\n\n")
        
        for stock_code in stocks:
            if stock_code in all_results:
                f.write(f"股票: {stock_code}\n")
                f.write("-"*80 + "\n")
                f.write(f"{'策略':<18} {'总收益率':<12} {'胜率':<10} {'交易次数':<10}\n")
                f.write("-"*80 + "\n")
                
                results = all_results[stock_code]
                for strategy_name in strategy_order:
                    if strategy_name in results:
                        r = results[strategy_name]
                        if strategy_name == '买入持有':
                            win_str = '-'
                        else:
                            win_str = f"{r['win_rate']*100:>8.2f}%"
                        f.write(f"{strategy_name:<18} {r['total_return']*100:>10.2f}% {win_str:<10} {r['trade_count']:>10}\n")
                
                f.write("\n")
    

if __name__ == '__main__':
    main()
