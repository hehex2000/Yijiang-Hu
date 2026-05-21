"""
对比 TOP 20 股票组合与沪深300指数
计算等权配置的组合收益，并生成对比报告
"""

import sys
import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-5s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

DB_PATH = "D:/tu-shareData/astock_daily.db"


def load_stock_data(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    从本地数据库加载股票前复权价格
    """
    # 转换代码格式
    code_str = str(code).zfill(6)
    if code_str.startswith('6'):
        ts_code = f"{code_str}.SH"
    else:
        ts_code = f"{code_str}.SZ"
    
    conn = sqlite3.connect(DB_PATH)
    try:
        # 读取日线数据，计算前复权价格
        sql = """
        SELECT 
            trade_date,
            open,
            high,
            low,
            close,
            vol,
            amount
        FROM daily
        WHERE ts_code = ?
          AND trade_date BETWEEN ? AND ?
        ORDER BY trade_date
        """
        df = pd.read_sql(sql, conn, params=(ts_code, int(start_date), int(end_date)))
        
        if df.empty:
            return None
        
        # 读取复权因子
        sql_adj = """
        SELECT trade_date, adj_factor
        FROM adj_factor
        WHERE ts_code = ?
          AND trade_date BETWEEN ? AND ?
        ORDER BY trade_date
        """
        df_adj = pd.read_sql(sql_adj, conn, params=(ts_code, int(start_date), int(end_date)))
        
        if df_adj.empty:
            # 没有复权因子，用收盘价
            df['adj_close'] = df['close']
        else:
            # 合并复权因子，计算前复权价
            df = pd.merge(df, df_adj, on='trade_date', how='left')
            # 最新一天的复权因子
            latest_adj = df_adj.iloc[-1]['adj_factor']
            df['adj_close'] = df['close'] * df['adj_factor'] / latest_adj
        
        return df
    except Exception as e:
        logger.warning(f"Error loading {code}: {e}")
        return None
    finally:
        conn.close()


def load_hs300_index(start_date: str, end_date: str) -> pd.DataFrame:
    """
    从CSV加载沪深300指数数据
    """
    hs300_file = os.path.join(os.path.dirname(__file__), 'data', 'hs300_index_daily.csv')
    
    if not os.path.exists(hs300_file):
        logger.error(f"HS300 index file not found: {hs300_file}")
        return None
    
    df = pd.read_csv(hs300_file)
    # trade_date 是 int64 格式，直接过滤
    start = int(start_date)
    end = int(end_date)
    df = df[(df['trade_date'] >= start) & (df['trade_date'] <= end)]
    
    if df.empty:
        logger.error(f"No HS300 data found for date range {start_date} to {end_date}")
        return None
    
    logger.info(f"✓ Loaded HS300 index: {len(df)} days")
    return df


def calculate_equal_weight_portfolio(stock_codes: list, start_date: str, end_date: str, initial_capital: float = 200000) -> dict:
    """
    计算等权组合收益
    """
    all_returns = {}  # code -> daily return series
    
    for code in stock_codes:
        df = load_stock_data(code, start_date, end_date)
        if df is None or df.empty:
            logger.warning(f"Failed to load data for {code}, skipping")
            continue
        
        # 计算收益率（相对于第一天）
        start_price = df.iloc[0]['adj_close']
        df['return'] = df['adj_close'] / start_price - 1
        
        all_returns[code] = df[['trade_date', 'return']].copy()
    
    if not all_returns:
        logger.error("No valid stock data loaded")
        return None
    
    logger.info(f"Calculating equal-weight portfolio return for {len(all_returns)} stocks...")
    
    # 合并所有股票的每日收益率
    base_code = list(all_returns.keys())[0]
    portfolio_df = all_returns[base_code].copy()
    portfolio_df = portfolio_df.rename(columns={'return': base_code})
    
    for code in list(all_returns.keys())[1:]:
        temp_df = all_returns[code].copy()
        temp_df = temp_df.rename(columns={'return': code})
        portfolio_df = pd.merge(portfolio_df, temp_df, on='trade_date', how='outer')
    
    # 按日期排序，向前填充缺失值
    portfolio_df = portfolio_df.sort_values('trade_date').reset_index(drop=True)
    return_cols = [c for c in portfolio_df.columns if c != 'trade_date']
    portfolio_df[return_cols] = portfolio_df[return_cols].fillna(method='ffill')
    
    # 计算等权组合收益率（每天的平均收益率）
    portfolio_df['portfolio_return'] = portfolio_df[return_cols].mean(axis=1)
    
    # 计算组合市值
    portfolio_df['portfolio_value'] = initial_capital * (1 + portfolio_df['portfolio_return'])
    
    # 总收益率
    total_return = portfolio_df.iloc[-1]['portfolio_return']
    total_value = initial_capital * (1 + total_return)
    
    logger.info(f"✓ Portfolio return calculated: {total_return:.2%}")
    
    # 转换为 daily_values 格式
    daily_values = portfolio_df[['trade_date', 'portfolio_value']].to_dict('records')
    
    # 计算每只股票的收益率
    stock_returns = []
    for code in all_returns:
        ret = all_returns[code].iloc[-1]['return']
        stock_returns.append({'code': code, 'return': ret})
    
    return {
        'total_return': total_return,
        'total_value': total_value,
        'initial_capital': initial_capital,
        'stock_returns': stock_returns,
        'daily_values': daily_values,
        'num_stocks': len(stock_returns)
    }


def main():
    # 配置
    start_date = "20230101"
    end_date = "20231231"
    initial_capital = 200000  # 20万
    
    logger.info("=" * 70)
    logger.info("TOP 20 股票组合 vs 沪深300 对比")
    logger.info("=" * 70)
    logger.info(f"回测区间: {start_date} 至 {end_date}")
    logger.info(f"初始资金: {initial_capital:,.0f} 元")
    logger.info("")
    
    # 1. 读取 TOP 20 股票
    selection_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                'data', 'results', 'selection_results.csv')
    
    if not os.path.exists(selection_file):
        logger.error(f"Selection results file not found: {selection_file}")
        logger.error("Please run run_selection.py first")
        return
    
    df = pd.read_csv(selection_file)
    top20_codes = df['股票代码'].astype(str).tolist()[:20]
    
    logger.info(f"✓ Loaded TOP 20 stocks from {selection_file}")
    logger.info("")
    
    # 2. 计算组合收益
    logger.info("正在计算 TOP 20 等权组合收益...")
    portfolio_result = calculate_equal_weight_portfolio(
        stock_codes=top20_codes,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital
    )
    
    if portfolio_result is None:
        logger.error("Failed to calculate portfolio return")
        return
    
    logger.info(f"✓ 组合收益计算完成")
    logger.info(f"  组合数: {portfolio_result['num_stocks']} 只股票")
    logger.info(f"  组合总收益率: {portfolio_result['total_return']:.2%}")
    logger.info("")
    
    # 3. 计算沪深300收益
    logger.info("正在计算沪深300指数收益...")
    hs300_df = load_hs300_index(start_date, end_date)
    
    if hs300_df is None or hs300_df.empty:
        logger.error("Failed to load HS300 index data")
        return
    
    hs300_start = hs300_df.iloc[0]['close']
    hs300_end = hs300_df.iloc[-1]['close']
    hs300_return = (hs300_end - hs300_start) / hs300_start
    
    logger.info(f"✓ 沪深300收益计算完成")
    logger.info(f"  开始点位: {hs300_start:.2f}")
    logger.info(f"  结束点位: {hs300_end:.2f}")
    logger.info(f"  指数收益率: {hs300_return:.2%}")
    logger.info("")
    
    # 4. 对比结果
    logger.info("=" * 70)
    logger.info("对比结果")
    logger.info("=" * 70)
    logger.info(f"{'指标':<20} {'TOP 20 组合':<20} {'沪深300':<20} {'超额收益':<20}")
    logger.info("-" * 70)
    
    excess_return = portfolio_result['total_return'] - hs300_return
    logger.info(f"{'总收益率':.<20} {portfolio_result['total_return']:>18.2%} {'':<2} {hs300_return:>18.2%} {'':<2} {excess_return:>18.2%}")
    
    # 计算年化收益率
    trading_days = len(hs300_df)
    
    portfolio_annualized = (1 + portfolio_result['total_return']) ** (252 / trading_days) - 1
    hs300_annualized = (1 + hs300_return) ** (252 / trading_days) - 1
    
    logger.info(f"{'年化收益率':.<20} {portfolio_annualized:>18.2%} {'':<2} {hs300_annualized:>18.2%} {'':<2} {portfolio_annualized - hs300_annualized:>18.2%}")
    logger.info("")
    
    # 5. 打印正收益股票数量
    positive_count = sum([1 for r in portfolio_result['stock_returns'] if r['return'] > 0])
    logger.info(f"正收益股票: {positive_count}/{portfolio_result['num_stocks']} ({positive_count/portfolio_result['num_stocks']:.1%})")
    logger.info("")
    
    # 6. 打印 TOP 5 和 BOTTOM 5 股票
    sorted_returns = sorted(portfolio_result['stock_returns'], key=lambda x: x['return'], reverse=True)
    
    logger.info("TOP 5 股票:")
    for i, r in enumerate(sorted_returns[:5], 1):
        logger.info(f"  {i}. {r['code']}: {r['return']:.2%}")
    
    logger.info("")
    logger.info("BOTTOM 5 股票:")
    for i, r in enumerate(sorted_returns[-5:][::-1], 1):
        logger.info(f"  {i}. {r['code']}: {r['return']:.2%}")
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("对比完成!")
    logger.info("=" * 70)
    
    # 7. 保存报告
    report_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            'backtest', 'result')
    os.makedirs(report_dir, exist_ok=True)
    
    report_file = os.path.join(report_dir, f'portfolio_vs_hs300_{start_date}_{end_date}.md')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# TOP 20 股票组合 vs 沪深300 对比报告\n\n")
        f.write(f"**回测区间**: {start_date} 至 {end_date}\n\n")
        f.write(f"**初始资金**: {initial_capital:,.0f} 元\n\n")
        f.write(f"**选股策略**: 价值 + 成长 + 质量 + 动量 + 技术 + 低波动 (6大类31因子)\n\n")
        f.write(f"**选股日期**: 2023-01-01\n\n")
        f.write(f"**选股数量**: TOP 20（沪深300成分股）\n\n")
        f.write("\n\n## 对比结果\n\n")
        f.write("| 指标 | TOP 20 组合 | 沪深300 | 超额收益 |\n")
        f.write("|------|--------------|----------|----------|\n")
        f.write(f"| 总收益率 | {portfolio_result['total_return']:.2%} | {hs300_return:.2%} | {excess_return:.2%} |\n")
        f.write(f"| 年化收益率 | {portfolio_annualized:.2%} | {hs300_annualized:.2%} | {portfolio_annualized - hs300_annualized:.2%} |\n")
        f.write("\n\n## 组合详情\n\n")
        f.write(f"- **组合数**: {portfolio_result['num_stocks']} 只股票\n")
        f.write(f"- **正收益占比**: {positive_count}/{portfolio_result['num_stocks']} ({positive_count/portfolio_result['num_stocks']:.1%})\n")
        f.write("\n\n## TOP 5 股票\n\n")
        f.write("| 排名 | 股票代码 | 收益率 |\n")
        f.write("|------|---------|--------|\n")
        for i, r in enumerate(sorted_returns[:5], 1):
            f.write(f"| {i} | {r['code']} | {r['return']:.2%} |\n")
        
        f.write("\n\n## BOTTOM 5 股票\n\n")
        f.write("| 排名 | 股票代码 | 收益率 |\n")
        f.write("|------|---------|--------|\n")
        for i, r in enumerate(sorted_returns[-5:][::-1], 1):
            f.write(f"| {i} | {r['code']} | {r['return']:.2%} |\n")
    
    logger.info(f"\n报告已保存到: {report_file}")


if __name__ == "__main__":
    main()
