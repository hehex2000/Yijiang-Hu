"""
TOP 20 股票组合 vs 沪深300 对比（简化版）
直接计算等权组合收益，用已知的沪深300收益率对比
"""

import sys
import os
import pandas as pd
import numpy as np
import sqlite3
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-5s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

DB_PATH = "D:/tu-shareData/astock_daily.db"


def load_stock_return(code: str, start_date: str, end_date: str) -> float:
    """
    计算单只股票的收益率（前复权）
    """
    code_str = str(code).zfill(6)
    if code_str.startswith('6'):
        ts_code = f"{code_str}.SH"
    else:
        ts_code = f"{code_str}.SZ"
    
    conn = sqlite3.connect(DB_PATH)
    try:
        # 读取日线数据
        sql = """
        SELECT trade_date, close
        FROM daily
        WHERE ts_code = ?
          AND trade_date BETWEEN ? AND ?
        ORDER BY trade_date
        LIMIT 1
        """
        df_start = pd.read_sql(sql, conn, params=(ts_code, int(start_date), int(end_date)))
        
        sql = """
        SELECT trade_date, close
        FROM daily
        WHERE ts_code = ?
          AND trade_date BETWEEN ? AND ?
        ORDER BY trade_date DESC
        LIMIT 1
        """
        df_end = pd.read_sql(sql, conn, params=(ts_code, int(start_date), int(end_date)))
        
        if df_start.empty or df_end.empty:
            logger.warning(f"No data for {code}")
            return None
        
        start_price = df_start.iloc[0]['close']
        end_price = df_end.iloc[0]['close']
        
        # 读取复权因子
        sql_adj = """
        SELECT adj_factor
        FROM adj_factor
        WHERE ts_code = ?
          AND trade_date BETWEEN ? AND ?
        ORDER BY trade_date
        """
        df_adj_start = pd.read_sql(sql_adj, conn, params=(ts_code, int(start_date), int(end_date)))
        
        sql_adj_end = """
        SELECT adj_factor
        FROM adj_factor
        WHERE ts_code = ?
          AND trade_date BETWEEN ? AND ?
        ORDER BY trade_date DESC
        """
        df_adj_end = pd.read_sql(sql_adj_end, conn, params=(ts_code, int(start_date), int(end_date)))
        
        if not df_adj_start.empty and not df_adj_end.empty:
            # 用复权因子计算前复权价
            adj_start = df_adj_start.iloc[0]['adj_factor']
            adj_end = df_adj_end.iloc[0]['adj_factor']
            start_adj = start_price * adj_end / adj_start
            end_adj = end_price * adj_end / adj_start
            ret = (end_adj - start_adj) / start_adj
        else:
            # 没有复权因子，用收盘价
            ret = (end_price - start_price) / start_price
        
        return ret
        
    except Exception as e:
        logger.warning(f"Error loading {code}: {e}")
        return None
    finally:
        conn.close()


def main():
    start_date = "20230101"
    end_date = "20231231"
    initial_capital = 200000
    
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
        return
    
    df = pd.read_csv(selection_file)
    top20_codes = df['股票代码'].astype(str).tolist()[:20]
    
    logger.info(f"✓ Loaded TOP 20 stocks from {selection_file}")
    logger.info("")
    
    # 2. 计算每只股票的收益率
    logger.info("正在计算每只股票的收益率...")
    stock_returns = []
    for code in top20_codes:
        ret = load_stock_return(code, start_date, end_date)
        if ret is not None:
            stock_returns.append({'code': code, 'return': ret})
            logger.info(f"  {code}: {ret:.2%}")
    
    if not stock_returns:
        logger.error("No valid stock returns calculated")
        return
    
    logger.info(f"✓ 已计算 {len(stock_returns)} 只股票的收益率")
    logger.info("")
    
    # 3. 计算等权组合收益
    portfolio_return = np.mean([r['return'] for r in stock_returns])
    portfolio_value = initial_capital * (1 + portfolio_return)
    
    logger.info(f"等权组合总收益率: {portfolio_return:.2%}")
    logger.info(f"组合期末市值: {portfolio_value:,.2f} 元")
    logger.info("")
    
    # 4. 沪深300收益（已知：-11.75%）
    hs300_return = -0.1175  # 2023年沪深300收益率
    
    logger.info(f"沪深300收益率（2023年）: {hs300_return:.2%}")
    logger.info("")
    
    # 5. 对比
    excess_return = portfolio_return - hs300_return
    
    logger.info("=" * 70)
    logger.info("对比结果")
    logger.info("=" * 70)
    logger.info(f"{'指标':<20} {'TOP 20 组合':<20} {'沪深300':<20} {'超额收益':<20}")
    logger.info("-" * 70)
    logger.info(f"{'总收益率':.<20} {portfolio_return:>18.2%} {'':<2} {hs300_return:>18.2%} {'':<2} {excess_return:>18.2%}")
    
    # 年化收益率
    trading_days = 242  # 2023年A股交易日约242天
    portfolio_annualized = (1 + portfolio_return) ** (252 / trading_days) - 1
    hs300_annualized = (1 + hs300_return) ** (252 / trading_days) - 1
    
    logger.info(f"{'年化收益率':.<20} {portfolio_annualized:>18.2%} {'':<2} {hs300_annualized:>18.2%} {'':<2} {portfolio_annualized - hs300_annualized:>18.2%}")
    logger.info("")
    
    # 6. 正收益股票
    positive_count = sum([1 for r in stock_returns if r['return'] > 0])
    logger.info(f"正收益股票: {positive_count}/{len(stock_returns)} ({positive_count/len(stock_returns):.1%})")
    logger.info("")
    
    # 7. TOP 5 和 BOTTOM 5
    sorted_returns = sorted(stock_returns, key=lambda x: x['return'], reverse=True)
    
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
    
    # 8. 保存报告
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
        f.write(f"| 总收益率 | {portfolio_return:.2%} | {hs300_return:.2%} | {excess_return:.2%} |\n")
        f.write(f"| 年化收益率 | {portfolio_annualized:.2%} | {hs300_annualized:.2%} | {portfolio_annualized - hs300_annualized:.2%} |\n")
        f.write("\n\n## 组合详情\n\n")
        f.write(f"- **组合数**: {len(stock_returns)} 只股票\n")
        f.write(f"- **正收益占比**: {positive_count}/{len(stock_returns)} ({positive_count/len(stock_returns):.1%})\n")
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
