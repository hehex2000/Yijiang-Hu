"""
统一回测脚本 - 对多因子选股TOP5运行6种策略
对比沪深300指数基准
回测时间：2023年全年
"""

import sys
import os
import pandas as pd
import numpy as np
from loguru import logger
from typing import Dict, List, Tuple
import json

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backtest'))

# 导入策略
from data_loader import DataLoader
from buy_and_hold_strategy import BuyAndHoldStrategy
from dual_ma_strategy import DualMAStrategy
from turtle_strategy import TurtleStrategy
from macd_rsi_strategy import MACDRSIStrategy
from macd_kdj_strategy import MACDKDJStrategy
from dca_strategy import DCAStrategy
from metrics import calculate_metrics

# =========== 配置 ===========
# 选出的TOP5股票（2023-01-01多因子选股结果）
TOP5_STOCKS = [
    ('603986', '兆易创新'),
    ('002384', '东山精密'),
    ('300308', '中际旭创'),
    ('002709', '天赐材料'),
    ('600183', '生益科技')
]

# 回测参数
START_DATE = '20230101'
END_DATE = '20231231'
INITIAL_CAPITAL = 200000.0  # 20万本金

# 沪深300指数代码
HS300_CODE = '000300.SH'

# 数据库路径
DB_PATH = 'D:/tu-shareData/astock_daily.db'

# 输出目录
OUTPUT_DIR = 'backtest/result'


def run_backtest_for_stock(code: str, name: str, data_loader: DataLoader) -> Dict:
    """
    对单只股票运行所有6种策略
    
    Returns:
        策略结果字典：{策略名: 绩效指标}
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"回测股票: {code} - {name}")
    logger.info(f"{'='*60}")
    
    # 加载数据
    df = data_loader.get_adjusted_prices(
        code, START_DATE, END_DATE,
        ma_short=10, ma_long=60, channel_period=30
    )
    
    if df is None or len(df) == 0:
        logger.warning(f"无数据: {code}")
        return {}
    
    logger.info(f"✓ 加载数据: {len(df)} 个交易日")
    
    results = {}
    
    # 策略1: 买入持有
    logger.info("\n[1/6] 买入持有策略...")
    strategy1 = BuyAndHoldStrategy(total_capital=INITIAL_CAPITAL)
    orders1 = strategy1.run(df)
    metrics1 = calculate_metrics(orders1, pd.DataFrame(strategy1.daily_values))
    results['买入持有'] = {
        'metrics': metrics1,
        'orders': orders1,
        'daily_values': strategy1.daily_values
    }
    logger.info(f"  总收益: {metrics1['total_return']:.2%}, 夏普: {metrics1['sharpe_ratio']:.2f}")
    
    # 策略2: 双均线
    logger.info("\n[2/6] 双均线策略...")
    strategy2 = DualMAStrategy(total_capital=INITIAL_CAPITAL)
    orders2 = strategy2.run(df)
    metrics2 = calculate_metrics(orders2, pd.DataFrame(strategy2.daily_values))
    results['双均线'] = {
        'metrics': metrics2,
        'orders': orders2,
        'daily_values': strategy2.daily_values
    }
    logger.info(f"  总收益: {metrics2['total_return']:.2%}, 夏普: {metrics2['sharpe_ratio']:.2f}")
    
    # 策略3: 海龟
    logger.info("\n[3/6] 海龟策略...")
    strategy3 = TurtleStrategy(total_capital=INITIAL_CAPITAL)
    orders3 = strategy3.run(df)
    metrics3 = calculate_metrics(orders3, pd.DataFrame(strategy3.daily_values))
    results['海龟'] = {
        'metrics': metrics3,
        'orders': orders3,
        'daily_values': strategy3.daily_values
    }
    logger.info(f"  总收益: {metrics3['total_return']:.2%}, 夏普: {metrics3['sharpe_ratio']:.2f}")
    
    # 策略4: MACD/RSI
    logger.info("\n[4/6] MACD/RSI策略...")
    strategy4 = MACDRSIStrategy(total_capital=INITIAL_CAPITAL)
    orders4 = strategy4.run(df)
    metrics4 = calculate_metrics(orders4, pd.DataFrame(strategy4.daily_values))
    results['MACD/RSI'] = {
        'metrics': metrics4,
        'orders': orders4,
        'daily_values': strategy4.daily_values
    }
    logger.info(f"  总收益: {metrics4['total_return']:.2%}, 夏普: {metrics4['sharpe_ratio']:.2f}")
    
    # 策略5: MACD/KDJ
    logger.info("\n[5/6] MACD/KDJ策略...")
    strategy5 = MACDKDJStrategy(total_capital=INITIAL_CAPITAL)
    orders5 = strategy5.run(df)
    metrics5 = calculate_metrics(orders5, pd.DataFrame(strategy5.daily_values))
    results['MACD/KDJ'] = {
        'metrics': metrics5,
        'orders': orders5,
        'daily_values': strategy5.daily_values
    }
    logger.info(f"  总收益: {metrics5['total_return']:.2%}, 夏普: {metrics5['sharpe_ratio']:.2f}")
    
    # 策略6: 月定投
    logger.info("\n[6/6] 月定投策略...")
    strategy6 = DCAStrategy(total_capital=INITIAL_CAPITAL, amount_per_month=5000)
    orders6 = strategy6.run(df)
    metrics6 = calculate_metrics(orders6, pd.DataFrame(strategy6.daily_values))
    results['月定投'] = {
        'metrics': metrics6,
        'orders': orders6,
        'daily_values': strategy6.daily_values
    }
    logger.info(f"  总收益: {metrics6['total_return']:.2%}, 夏普: {metrics6['sharpe_ratio']:.2f}")
    
    return results


def get_hs300_benchmark(data_loader: DataLoader) -> Dict:
    """
    获取沪深300指数基准收益
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"获取沪深300指数基准收益...")
    logger.info(f"{'='*60}")
    
    df = data_loader.get_index_data(HS300_CODE, START_DATE, END_DATE)
    
    if df is None or len(df) == 0:
        logger.warning("无沪深300指数数据")
        return {}
    
    # 计算基准收益（买入持有）
    initial_close = df['close'].iloc[0]
    final_close = df['close'].iloc[-1]
    total_return = (final_close - initial_close) / initial_close
    
    # 计算年化收益
    days = len(df)
    annualized_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
    
    # 计算最大回撤
    cumulative_max = df['close'].cummax()
    drawdown = (df['close'] - cumulative_max) / cumulative_max
    max_drawdown = abs(drawdown.min())
    
    # 计算夏普比率（简化版）
    daily_returns = df['close'].pct_change().dropna()
    annualized_vol = daily_returns.std() * np.sqrt(252)
    sharpe = (annualized_return - 0.03) / annualized_vol if annualized_vol > 0 else 0
    
    benchmark_metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe,
        'num_trades': 1,  # 基准只有一次买入
        'win_rate': 1.0 if total_return > 0 else 0.0,
        'avg_holding_days': days,
        'max_profit': total_return,
        'max_loss': 0.0
    }
    
    logger.info(f"✓ 沪深300基准:")
    logger.info(f"  总收益: {total_return:.2%}")
    logger.info(f"  年化收益: {annualized_return:.2%}")
    logger.info(f"  最大回撤: {max_drawdown:.2%}")
    logger.info(f"  夏普比率: {sharpe:.2f}")
    
    return {
        'metrics': benchmark_metrics,
        'daily_values': df[['trade_date', 'close']].rename(columns={'close': 'portfolio_value'})
    }


def format_results(all_results: Dict, benchmark: Dict) -> str:
    """
    格式化回测结果为Markdown表格
    """
    output = []
    output.append("# 多因子选股回测报告")
    output.append(f"\n**回测时间**: {START_DATE} 至 {END_DATE}")
    output.append(f"**初始资金**: {INITIAL_CAPITAL:,.0f} 元")
    output.append(f"**选股日期**: 2023-01-01")
    output.append(f"**选股策略**: 价值 + 成长 + 质量 + 动量 + 技术 + 低波动 (6大类31因子)\n")
    
    # 表格头部
    output.append("## 回测结果汇总\n")
    output.append("| 股票 | 策略 | 总收益率 | 年化收益率 | 最大回撤 | 夏普比率 | 交易次数 | 胜率 |")
    output.append("|------|------|----------|------------|----------|----------|----------|------|")
    
    # 填充数据
    for code_name, strategies in all_results.items():
        code, name = code_name.split(' - ')
        
        for strategy_name, result in strategies.items():
            metrics = result['metrics']
            output.append(
                f"| {code} {name} | {strategy_name} | "
                f"{metrics['total_return']:.2%} | "
                f"{metrics['annualized_return']:.2%} | "
                f"{metrics['max_drawdown']:.2%} | "
                f"{metrics['sharpe_ratio']:.2f} | "
                f"{metrics['num_trades']} | "
                f"{metrics['win_rate']:.1%} |"
            )
    
    # 基准
    if benchmark:
        bm_metrics = benchmark['metrics']
        output.append(
            f"| **沪深300** | **基准** | "
            f"**{bm_metrics['total_return']:.2%}** | "
            f"**{bm_metrics['annualized_return']:.2%}** | "
            f"**{bm_metrics['max_drawdown']:.2%}** | "
            f"**{bm_metrics['sharpe_ratio']:.2f}** | "
            f"{bm_metrics['num_trades']} | "
            f"{bm_metrics['win_rate']:.1%} |"
        )
    
    output.append("\n---\n")
    
    # 最佳策略分析
    output.append("## 最佳策略分析\n")
    
    # 按总收益率排序
    all_strategies = []
    for code_name, strategies in all_results.items():
        for strategy_name, result in strategies.items():
            all_strategies.append({
                'code_name': code_name,
                'strategy': strategy_name,
                'total_return': result['metrics']['total_return'],
                'sharpe_ratio': result['metrics']['sharpe_ratio']
            })
    
    # 排序
    all_strategies.sort(key=lambda x: x['total_return'], reverse=True)
    
    output.append("### TOP 5 策略（按总收益率排序）\n")
    output.append("| 排名 | 股票 | 策略 | 总收益率 | 夏普比率 |")
    output.append("|------|------|------|----------|----------|")
    
    for i, s in enumerate(all_strategies[:5], 1):
        code, name = s['code_name'].split(' - ')
        output.append(
            f"| {i} | {code} {name} | {s['strategy']} | "
            f"{s['total_return']:.2%} | {s['sharpe_ratio']:.2f} |"
        )
    
    output.append("\n---\n")
    
    # 策略平均表现
    output.append("## 策略平均表现\n")
    output.append("| 策略 | 平均收益率 | 平均夏普比率 | 正收益占比 |")
    output.append("|------|------------|----------------|------------|")
    
    strategy_stats = {}
    for code_name, strategies in all_results.items():
        for strategy_name, result in strategies.items():
            if strategy_name not in strategy_stats:
                strategy_stats[strategy_name] = {
                    'returns': [],
                    'sharpes': []
                }
            strategy_stats[strategy_name]['returns'].append(result['metrics']['total_return'])
            strategy_stats[strategy_name]['sharpes'].append(result['metrics']['sharpe_ratio'])
    
    for strategy_name, stats in strategy_stats.items():
        avg_return = np.mean(stats['returns'])
        avg_sharpe = np.mean(stats['sharpes'])
        positive_rate = sum(1 for r in stats['returns'] if r > 0) / len(stats['returns'])
        
        output.append(
            f"| {strategy_name} | {avg_return:.2%} | {avg_sharpe:.2f} | {positive_rate:.1%} |"
        )
    
    return '\n'.join(output)


def main():
    """主函数"""
    logger.info("\n" + "="*60)
    logger.info("多因子选股回测系统 - 统一回测脚本")
    logger.info("="*60 + "\n")
    
    # 初始化数据加载器
    data_loader = DataLoader(db_path=DB_PATH)
    
    # 存储所有结果
    all_results = {}
    benchmark_result = {}
    
    # 对每只股票运行回测
    for code, name in TOP5_STOCKS:
        key = f"{code} - {name}"
        all_results[key] = run_backtest_for_stock(code, name, data_loader)
    
    # 获取沪深300基准
    benchmark_result = get_hs300_benchmark(data_loader)
    
    # 格式化结果
    report = format_results(all_results, benchmark_result)
    
    # 保存报告
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, 'backtest_report_2023.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✓ 回测完成！报告已保存: {report_path}")
    logger.info(f"{'='*60}\n")
    
    # 打印报告
    print("\n" + "="*60)
    print(report)
    print("="*60 + "\n")
    
    return all_results, benchmark_result


if __name__ == '__main__':
    main()
