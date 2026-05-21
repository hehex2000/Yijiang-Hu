"""
回测脚本 - 2020年
策略：
1. 双均线策略（MA10/30）
2. 海龟策略（N=20）
3. 月定投策略（每月8000元）

测试周期：2020年一整年
每支股票初始资金10万，年末清仓卖出
新增：最大回撤风险指标 + 沪深300基准对比
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
    # 将由选股结果填充
]

START_DATE = '20200101'
END_DATE = '20201231'
TOTAL_CAPITAL = 100000  # 每支股票初始资金10万
DCA_AMOUNT_PER_MONTH = 8000  # 月定投8000元
TURTLE_CHANNEL_PERIOD = 20  # 海龟策略N=20

# 双均线组合
MA_COMBINATIONS = [
    (10, 30, 'MA10/30'),
]

# 沪深300指数代码
HS300_CODE = '000300.SH'


def get_hs300_benchmark():
    """
    获取沪深300指数2020年数据作为基准
    返回：沪深300指数收益率
    """
    try:
        import tushare as ts
        import os
        
        # 从Tushare-Downloader的config.py读取token
        config_path = os.path.join('C:', os.sep, 'Users', '99395', 'WorkBuddy', 'Tushare-Downloader', 'config.py')
        
        if not os.path.exists(config_path):
            logger.warning(f"配置文件不存在：{config_path}")
            return None
        
        # 读取token（简单解析）
        with open(config_path, 'r') as f:
            for line in f:
                if line.startswith('TUSHARE_TOKEN'):
                    token = line.split('=')[1].strip().strip('"').strip("'")
                    break
            else:
                logger.warning("配置文件中未找到 TUSHARE_TOKEN")
                return None
        
        ts.set_token(token)
        pro = ts.pro_api()
        
        logger.info("正在下载沪深300指数数据...")
        df_hs300 = pro.index_daily(ts_code='000300.SH', 
                                   start_date=START_DATE, 
                                   end_date=END_DATE)
        
        if df_hs300 is None or len(df_hs300) == 0:
            logger.warning("沪深300指数数据下载失败，跳过基准对比")
            return None
        
        # 按日期排序（从早到晚）
        df_hs300 = df_hs300.sort_values('trade_date')
        
        # 计算收益率
        start_close = df_hs300.iloc[0]['close']
        end_close = df_hs300.iloc[-1]['close']
        hs300_return = (end_close - start_close) / start_close
        
        logger.info(f"✓ 沪深300指数2020年收益率：{hs300_return:.2%}")
        logger.info(f"  起始点位：{start_close:.2f}（{df_hs300.iloc[0]['trade_date']}）")
        logger.info(f"  结束点位：{end_close:.2f}（{df_hs300.iloc[-1]['trade_date']}）")
        
        return {
            'return': hs300_return,
            'start_close': start_close,
            'end_close': end_close,
            'start_date': df_hs300.iloc[0]['trade_date'],
            'end_date': df_hs300.iloc[-1]['trade_date'],
            'data': df_hs300
        }
        
    except Exception as e:
        logger.warning(f"获取沪深300指数失败：{e}")
        return None


def run_backtest_for_stock(stock_code: str, stock_name: str):
    """
    对单支股票运行多个策略，返回回测结果
    """
    logger.info("="*60)
    logger.info(f"开始回测：{stock_code} {stock_name}")
    logger.info("="*60)
    
    # 加载数据
    loader = DataLoader()
    
    results = {}
    
    # 1. 运行双均线策略（多个MA组合）
    for ma_short, ma_long, ma_label in MA_COMBINATIONS:
        logger.info(f"  运行双均线策略（{ma_label}）...")
        
        # 获取对应MA的数据
        df_dual_ma = loader.get_adjusted_prices(
            stock_code, START_DATE, END_DATE,
            ma_short=ma_short, ma_long=ma_long, 
            channel_period=TURTLE_CHANNEL_PERIOD
        )
        
        if df_dual_ma is None or len(df_dual_ma) < ma_long:
            logger.warning(f"{stock_code} 数据不足（需要至少{ma_long}天），跳过{ma_label}")
            continue
        
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
        
        results[f'双均线({ma_label})'] = {
            'orders': orders_dual_ma,
            'metrics': metrics_dual_ma,
            'final_value': final_value_dual_ma,
            'max_drawdown': metrics_dual_ma.get('max_drawdown', 0)
        }
        logger.info(f"  ✓ 双均线{ma_label}完成：{len(orders_dual_ma)} 笔交易，最终资产 {final_value_dual_ma:.2f}，最大回撤 {metrics_dual_ma.get('max_drawdown', 0):.2%}")
    
    # 2. 运行海龟策略（N=20）
    logger.info(f"  运行海龟策略（N={TURTLE_CHANNEL_PERIOD}）...")
    
    # 使用MA10/30的数据（包含channel_high/low）
    df_turtle = loader.get_adjusted_prices(
        stock_code, START_DATE, END_DATE,
        ma_short=10, ma_long=30, 
        channel_period=TURTLE_CHANNEL_PERIOD
    )
    
    if df_turtle is not None and len(df_turtle) >= TURTLE_CHANNEL_PERIOD:
        strategy_turtle = TurtleStrategy(
            total_capital=TOTAL_CAPITAL,
            channel_period=TURTLE_CHANNEL_PERIOD,
            position_mode='half',
            max_positions=3,
            take_profit=0.20,
            stop_loss=0.10
        )
        orders_turtle = strategy_turtle.run(df_turtle)
        
        # 转换 daily_values 为 DataFrame
        daily_df_turtle = pd.DataFrame(strategy_turtle.daily_values)
        metrics_turtle = calculate_metrics(orders_turtle, daily_df_turtle)
        
        final_price = df_turtle.iloc[-1]['adj_close']
        final_value_turtle = strategy_turtle.cash + strategy_turtle.position_shares * final_price
        
        results[f'海龟(N={TURTLE_CHANNEL_PERIOD})'] = {
            'orders': orders_turtle,
            'metrics': metrics_turtle,
            'final_value': final_value_turtle,
            'max_drawdown': metrics_turtle.get('max_drawdown', 0)
        }
        logger.info(f"  ✓ 海龟完成：{len(orders_turtle)} 笔交易，最终资产 {final_value_turtle:.2f}，最大回撤 {metrics_turtle.get('max_drawdown', 0):.2%}")
    else:
        logger.warning(f"{stock_code} 数据不足，跳过海龟策略")
    
    # 3. 运行月定投策略（每月8000元）
    logger.info(f"  运行月定投策略（每月{DCA_AMOUNT_PER_MONTH}元）...")
    
    # 使用任意一个DataFrame即可（只需要trade_date, adj_open, adj_close）
    if df_turtle is not None:
        df_dca = df_turtle[['trade_date', 'adj_open', 'adj_close']].copy()
        
        strategy_dca = DCAStrategy(
            total_capital=TOTAL_CAPITAL,
            amount_per_month=DCA_AMOUNT_PER_MONTH,  # 已改为8000
            take_profit=0.30,
            stop_loss=0.10
        )
        orders_dca = strategy_dca.run(df_dca)
        
        # 转换 daily_values 为 DataFrame
        daily_df_dca = pd.DataFrame(strategy_dca.daily_values)
        metrics_dca = calculate_metrics(orders_dca, daily_df_dca)
        
        # 计算DCA策略最终资产
        dca_total_shares = sum(lot['shares'] for lot in strategy_dca.lots)
        final_price = df_turtle.iloc[-1]['adj_close']
        final_value_dca = strategy_dca.cash + dca_total_shares * final_price
        
        # 计算DCA策略的实际总投入和收益率
        dca_total_invested = TOTAL_CAPITAL + sum(lot['cost_total'] for lot in strategy_dca.lots)
        dca_total_return = (final_value_dca - dca_total_invested) / dca_total_invested if dca_total_invested > 0 else 0
        
        results[f'月定投({DCA_AMOUNT_PER_MONTH}元/月)'] = {
            'orders': orders_dca,
            'metrics': metrics_dca,
            'final_value': final_value_dca,
            'max_drawdown': metrics_dca.get('max_drawdown', 0),
            'total_invested': dca_total_invested,
            'actual_return': dca_total_return
        }
        logger.info(f"  ✓ 定投完成：{len(orders_dca)} 笔交易，最终资产 {final_value_dca:.2f}，最大回撤 {metrics_dca.get('max_drawdown', 0):.2%}")
    else:
        logger.warning(f"{stock_code} 数据不足，跳过定投策略")
    
    return {
        'stock_code': stock_code,
        'stock_name': stock_name,
        'results': results
    }


def main(selected_stocks: list):
    """主函数：运行所有股票的回测，输出对比结果（含沪深300基准）"""
    global STOCKS
    STOCKS = selected_stocks
    
    logger.info("="*60)
    logger.info(f"开始2020年回测（含沪深300基准对比）")
    logger.info("="*60)
    
    # 获取沪深300指数基准
    hs300_data = get_hs300_benchmark()
    
    all_results = []
    stock_prices = {}  # 存储每支股票的期初和期末价格
    
    for stock_code, stock_name in STOCKS:
        result = run_backtest_for_stock(stock_code, stock_name)
        if result:
            # 获取该股票的期初和期末价格
            loader = DataLoader()
            df = loader.get_adjusted_prices(stock_code, START_DATE, END_DATE, 
                                            ma_short=10, ma_long=30, 
                                            channel_period=TURTLE_CHANNEL_PERIOD)
            if df is not None and len(df) >= 2:
                start_price = df.iloc[0]['adj_close']
                end_price = df.iloc[-1]['adj_close']
                stock_change = (end_price - start_price) / start_price
                stock_prices[stock_code] = {
                    'start_price': start_price,
                    'end_price': end_price,
                    'change': stock_change
                }
            
            all_results.append(result)
    
    # 输出汇总报告
    print("\n" + "="*160)
    print("2020年回测结果汇总（含沪深300基准对比 + 股票本身涨跌）")
    print("="*160)
    
    if hs300_data:
        print(f"沪深300指数基准：{hs300_data['return']:.2%}  （{hs300_data['start_date']}：{hs300_data['start_close']:.2f} -> {hs300_data['end_date']}：{hs300_data['end_close']:.2f}）")
        print("-"*160)
    
    print(f"{'股票':<15} {'策略':<25} {'交易次数':>8} {'最终资产':>12} {'总收益率':>12} {'股票涨跌':>12} {'最大回撤':>12} {'vs HS300':>12}")
    print("-"*160)
    
    for result in all_results:
        stock_code = result['stock_code']
        stock_name = result['stock_name']
        stock_label = f"{stock_code} {stock_name[:6]}"
        
        # 获取该股票的涨跌幅度
        stock_change = stock_prices.get(stock_code, {}).get('change', 0) if stock_code in stock_prices else 0
        
        for strategy_name, data in result['results'].items():
            metrics = data['metrics']
            num_orders = len(data['orders'])
            final_value = data['final_value']
            
            # 使用正确的收益率计算
            if 'actual_return' in data:
                # DCA策略：使用实际收益率
                total_return = data['actual_return']
            else:
                # 其他策略：使用初始资金计算
                total_return = (final_value - TOTAL_CAPITAL) / TOTAL_CAPITAL
            
            max_dd = data.get('max_drawdown', 0)
            
            # 计算相对沪深300的超额收益
            if hs300_data:
                excess_return = total_return - hs300_data['return']
                vs_hs300 = f"{excess_return:+.2%}"
            else:
                vs_hs300 = "N/A"
            
            print(f"{stock_label:<15} {strategy_name:<25} {num_orders:>8} {final_value:>11.2f} {total_return:>11.2%} {stock_change:>11.2%} {max_dd:>11.2%} {vs_hs300:>12}")
        
        print("-"*160)
    
    # 保存结果到文件
    output_file = os.path.join(project_root, 'backtest_result_2020.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("2020年回测结果汇总（含沪深300基准对比）\n")
        f.write("="*140 + "\n\n")
        
        if hs300_data:
            f.write(f"沪深300指数基准：{hs300_data['return']:.2%}\n")
            f.write(f"  起始点位：{hs300_data['start_close']:.2f}（{hs300_data['start_date']}）\n")
            f.write(f"  结束点位：{hs300_data['end_close']:.2f}（{hs300_data['end_date']}）\n\n")
        
        for result in all_results:
            stock_code = result['stock_code']
            stock_name = result['stock_name']
            f.write(f"{stock_code} {stock_name}\n")
            
            # 写入该股票的涨跌幅度
            if stock_code in stock_prices:
                stock_change = stock_prices[stock_code]['change']
                f.write(f"  股票2020年涨跌：{stock_change:.2%}\n")
            
            f.write("-"*140 + "\n")
            
            for strategy_name, data in result['results'].items():
                metrics = data['metrics']
                num_orders = len(data['orders'])
                final_value = data['final_value']
                
                # 使用正确的收益率计算
                if 'actual_return' in data:
                    total_return = data['actual_return']
                else:
                    total_return = (final_value - TOTAL_CAPITAL) / TOTAL_CAPITAL
                
                max_dd = data.get('max_drawdown', 0)
                
                # 计算相对沪深300的超额收益
                if hs300_data:
                    excess_return = total_return - hs300_data['return']
                    vs_hs300 = f"{excess_return:+.2%}"
                else:
                    vs_hs300 = "N/A"
                
                # 计算相对股票本身的超额收益
                if stock_code in stock_prices:
                    stock_change = stock_prices[stock_code]['change']
                    vs_stock = f"{total_return - stock_change:+.2%}"
                else:
                    vs_stock = "N/A"
                
                f.write(f"  策略：{strategy_name}\n")
                f.write(f"    交易次数：{num_orders}\n")
                f.write(f"    最终资产：{final_value:.2f}\n")
                f.write(f"    总收益率：{total_return:.2%}\n")
                
                # 如果有股票涨跌数据，写入
                if stock_code in stock_prices:
                    f.write(f"    股票涨跌：{stock_change:.2%}\n")
                    f.write(f"    相对股票本身：{vs_stock}\n")
                
                f.write(f"    最大回撤：{max_dd:.2%}\n")
                f.write(f"    相对HS300：{vs_hs300}\n")
                f.write(f"    年化收益：{metrics.get('annualized_return', 0):.2%}\n")
                f.write(f"    夏普比率：{metrics.get('sharpe_ratio', 0):.2f}\n")
                f.write(f"    胜率：{metrics.get('win_rate', 0):.2%}\n")
                f.write("\n")
            
            f.write("\n")
    
    logger.info(f"✓ 回测完成！结果已保存到：{output_file}")
    
    return all_results


if __name__ == '__main__':
    # 需要从选股结果导入
    import sys
    import os
    
    if len(sys.argv) < 2:
        print("用法: python run_backtest_2020.py <选股结果文件>")
        print("  选股结果文件: Excel文件，包含'code'和'name'列")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"文件不存在: {input_file}")
        sys.exit(1)
    
    # 读取选股结果
    df = pd.read_excel(input_file)
    
    # 提取股票代码和名称
    selected_stocks = []
    for _, row in df.iterrows():
        code = str(row['code']).zfill(6)  # 补齐前导零
        name = row['name'] if 'name' in df.columns else ''
        selected_stocks.append((code, name))
    
    print(f"从 {input_file} 读取了 {len(selected_stocks)} 只股票")
    
    # 运行回测
    main(selected_stocks)
