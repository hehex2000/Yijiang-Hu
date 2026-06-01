#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多策略对比回测 - 完整版（2022-2023动态资金分配）
"""
import sys, os, pandas as pd
from datetime import datetime
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest.rsi_trend_strategy import RSITrendStrategy
from backtest.macd_kdj_strategy_fixed import MACDKDJStrategy
from backtest.data_loader import DataLoader
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_backtest(ts_code, strategy_name, start_date, end_date, initial_capital=100000.0):
    """对单只股票运行回测"""
    try:
        loader = DataLoader(
            db_path=config.DATA_FETCHER['local_db_path'],
            tushare_token=config.DATA_FETCHER['tushare_token']
        )
        df = loader.get_adjusted_prices(code=ts_code, start_date=start_date, end_date=end_date)
        
        if df is None or len(df) == 0:
            return {'final_value': initial_capital, 'total_return': 0.0, 'trades': []}
        
        if strategy_name == 'RSI_Trend':
            strategy = RSITrendStrategy(
                total_capital=initial_capital,
                rsi_period=14,
                rsi_center=50,
                take_profit=0.50,
                stop_loss=0.15
            )
        elif strategy_name == 'MACD_KDJ':
            strategy = MACDKDJStrategy(
                total_capital=initial_capital,
                macd_fast=12,
                macd_slow=26,
                macd_signal=9,
                kdj_n=9,
                kdj_m1=3,
                kdj_m2=3,
                take_profit=0.25,
                stop_loss=0.10
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        result = strategy.run(df=df)
        
        if len(result['daily_values']) > 0:
            final_value = result['daily_values'][-1]['portfolio_value']
        else:
            final_value = initial_capital
        
        total_return = (final_value - initial_capital) / initial_capital
        
        return {'final_value': final_value, 'total_return': total_return, 'trades': result['trades']}
    
    except Exception as e:
        logger.error(f"{ts_code} backtest failed: {e}")
        return {'final_value': initial_capital, 'total_return': 0.0, 'trades': [], 'error': str(e)}


def main():
    """主函数"""
    strategies = ['RSI_Trend', 'MACD_KDJ']
    
    for strategy_name in strategies:
        print(f"\n{'='*100}")
        print(f"Strategy: {strategy_name}")
        print(f"{'='*100}\n")
        
        # 加载选股结果
        selection_path = "data/results/selection_results_20220101.csv"
        if not os.path.exists(selection_path):
            print(f"ERROR: Cannot find selection results: {selection_path}")
            continue
        
        df = pd.read_csv(selection_path, encoding='utf-8-sig', dtype=str)
        if 'ts_code' not in df.columns and '股票代码' in df.columns:
            df = df.rename(columns={'股票代码': 'ts_code', '股票名称': 'name'})
        df = df.head(20)
        
        print(f"Loaded selection results: {len(df)} stocks\n")
        
        # ========== 2022年回测 ==========
        print(f"[{strategy_name}] Running 2022 backtest...")
        results_2022 = []
        total_final_2022 = 0.0
        
        for idx, row in df.iterrows():
            ts_code = row['ts_code']
            stock_name = row['name'] if 'name' in row else ts_code
            
            result = run_backtest(
                ts_code=ts_code,
                strategy_name=strategy_name,
                start_date='20220104',
                end_date='20221230',
                initial_capital=100000.0
            )
            
            result['ts_code'] = ts_code
            result['name'] = stock_name
            results_2022.append(result)
            total_final_2022 += result['final_value']
            
            status = "+" if result['total_return'] >= 0 else "-"
            print(f"  {status} {ts_code} {stock_name}: "
                  f"Return={result['total_return']*100:+.2f}%, "
                  f"Final={result['final_value']:,.2f}")
        
        print(f"\n[{strategy_name}] 2022 backtest completed")
        print(f"  Total funds: {total_final_2022:,.2f}")
        print(f"  Return: {(total_final_2022 - 2000000) / 2000000 * 100:+.2f}%\n")
        
        # ========== 2023年动态资金回测 ==========
        print(f"[{strategy_name}] Running 2023 backtest (dynamic fund allocation)...")
        
        # 动态资金分配：如果总资金 < 200万，则均分给20只股票
        if total_final_2022 < 2000000:
            allocated_capital = total_final_2022 / 20
            print(f"  Warning: Insufficient funds ({total_final_2022:,.2f} < 2,000,000)")
            print(f"  Allocating {allocated_capital:,.2f} to each stock\n")
        else:
            allocated_capital = 100000.0  # 足够的话还是每只10万
            print(f"  Sufficient funds ({total_final_2022:,.2f} >= 2,000,000)")
            print(f"  Allocating 100,000 to each stock\n")
        
        results_2023 = []
        total_final_2023 = 0.0
        
        for idx, row in df.iterrows():
            ts_code = row['ts_code']
            stock_name = row['name'] if 'name' in row else ts_code
            
            result = run_backtest(
                ts_code=ts_code,
                strategy_name=strategy_name,
                start_date='20230104',
                end_date='20231229',
                initial_capital=allocated_capital
            )
            
            result['ts_code'] = ts_code
            result['name'] = stock_name
            results_2023.append(result)
            total_final_2023 += result['final_value']
            
            status = "+" if result['total_return'] >= 0 else "-"
            print(f"  {status} {ts_code} {stock_name}: "
                  f"Return={result['total_return']*100:+.2f}%, "
                  f"Final={result['final_value']:,.2f}")
        
        print(f"\n[{strategy_name}] 2023 backtest completed")
        print(f"  Total funds: {total_final_2023:,.2f}")
        print(f"  Return: {(total_final_2023 - total_final_2022) / total_final_2022 * 100:+.2f}%\n")
        
        # ========== 生成完整报告 ==========
        report_path = f"backtest/result/multi_strategy_report_{strategy_name}_2022_2023.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"Strategy: {strategy_name}\n")
            f.write(f"="*100 + "\n\n")
            
            # 2022年结果
            f.write(f"2022 Backtest Results\n")
            f.write(f"{'Rank':<6}{'Code':<12}{'Name':<20}{'Return':<12}{'Final':<15}\n")
            for i, r in enumerate(sorted(results_2022, key=lambda x: x['total_return'], reverse=True)):
                f.write(f"{i+1:<6}{r['ts_code']:<12}{r['name']:<20}"
                       f"{r['total_return']*100:>+10.2f}%  {r['final_value']:>12,.2f}\n")
            f.write(f"\n2022 Total funds: {total_final_2022:,.2f}\n")
            f.write(f"2022 Return: {(total_final_2022 - 2000000) / 2000000 * 100:+.2f}%\n\n")
            
            # 2023年结果
            f.write(f"{'='*100}\n\n")
            f.write(f"2023 Backtest Results (Dynamic Fund Allocation)\n")
            f.write(f"Allocated capital per stock: {allocated_capital:,.2f}\n\n")
            f.write(f"{'Rank':<6}{'Code':<12}{'Name':<20}{'Return':<12}{'Final':<15}\n")
            for i, r in enumerate(sorted(results_2023, key=lambda x: x['total_return'], reverse=True)):
                f.write(f"{i+1:<6}{r['ts_code']:<12}{r['name']:<20}"
                       f"{r['total_return']*100:>+10.2f}%  {r['final_value']:>12,.2f}\n")
            f.write(f"\n2023 Total funds: {total_final_2023:,.2f}\n")
            f.write(f"2023 Return: {(total_final_2023 - total_final_2022) / total_final_2022 * 100:+.2f}%\n\n")
            
            # 总结
            f.write(f"{'='*100}\n\n")
            f.write(f"Summary (2022-2023)\n")
            f.write(f"Initial capital: 2,000,000.00\n")
            f.write(f"Final capital: {total_final_2023:,.2f}\n")
            f.write(f"Total return: {(total_final_2023 - 2000000) / 2000000 * 100:+.2f}%\n")
        
        print(f"Report generated: {report_path}\n")
        
        # 打印总结
        print(f"\n{'='*100}")
        print(f"[{strategy_name}] Summary (2022-2023)")
        print(f"{'='*100}")
        print(f"  Initial capital: 2,000,000.00")
        print(f"  2022 end capital: {total_final_2022:,.2f}")
        print(f"  2023 end capital: {total_final_2023:,.2f}")
        print(f"  Total return: {(total_final_2023 - 2000000) / 2000000 * 100:+.2f}%\n")


if __name__ == '__main__':
    main()
