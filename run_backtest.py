#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
回测系统主程序 - 运行主动量化（双均线/海龟）和被动量化（定投）策略
生成 Excel 报告（多个 Sheet）和可视化图表
支持通过配置文件驱动回测参数
支持规则选股和ML选股
"""

import sys
import os
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger
from typing import Dict, List, Any

# 添加项目根目录到路径
project_root = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from backtest.data_loader import DataLoader
from backtest.dual_ma_strategy import DualMAStrategy
from backtest.turtle_strategy import TurtleStrategy
from backtest.dca_strategy import DCAStrategy
from backtest.weekly_dca_strategy import WeeklyDCAStrategy
from backtest.metrics import calculate_metrics, compare_strategies, _empty_metrics


# 回测参数（将从配置文件读取，默认值如下）
CONFIG_FILE = os.path.join(project_root, 'config', 'backtest_config.yaml')
STOCKS = []  # 将从多因子选股结果填充
START_DATE = '20200101'
END_DATE = '20241231'
TOTAL_CAPITAL = 600000  # 每支股票60万（3个策略各20万）
STRATEGY_CAPITAL = 200000  # 每个策略20万

# 双均线策略参数（从配置文件读取）
DUAL_MA_MA_SHORT = 15  # 短期均线周期（默认15日）
DUAL_MA_MA_LONG = 60   # 长期均线周期（默认60日）
DUAL_MA_TAKE_PROFIT = 0.25  # 止盈线（25%）
DUAL_MA_STOP_LOSS = 0.10    # 止损线（10%）

# 海龟策略参数
CHANNEL_PERIOD = 30  # 通道周期（日前）
POSITION_MODE = 'half'  # 仓位模式：'full'=全仓，'half'=半仓
MAX_POSITIONS = 3  # 最大加仓次数
TAKE_PROFIT = 0.20  # 止盈线（20%）
STOP_LOSS = 0.10  # 止损线（10%）

# 定投策略参数（从配置文件读取）
DCA_TAKE_PROFIT = 0.30  # 止盈线（30%）
DCA_STOP_LOSS = 0.10    # 止损线（10%）


def load_config(config_file: str = CONFIG_FILE) -> Dict:
    """
    加载配置文件
    
    Args:
        config_file: 配置文件路径
        
    Returns:
        配置字典
    """
    if not os.path.exists(config_file):
        logger.warning(f"配置文件不存在: {config_file}，使用默认参数")
        return {}
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"✓ 已加载配置文件: {config_file}")
    return config if config else {}


def apply_config(config: Dict):
    """
    应用配置参数到全局变量
    
    Args:
        config: 配置字典
    """
    global STOCKS, START_DATE, END_DATE, TOTAL_CAPITAL, STRATEGY_CAPITAL
    global CHANNEL_PERIOD, POSITION_MODE, MAX_POSITIONS, TAKE_PROFIT, STOP_LOSS
    global DUAL_MA_MA_SHORT, DUAL_MA_MA_LONG, DUAL_MA_TAKE_PROFIT, DUAL_MA_STOP_LOSS
    global DCA_TAKE_PROFIT, DCA_STOP_LOSS
    
    # 数据配置
    if 'data' in config:
        START_DATE = config['data'].get('start_date', START_DATE)
        END_DATE = config['data'].get('end_date', END_DATE)
        stocks_file = config['data'].get('stocks_file', '')
        
        if stocks_file and os.path.exists(stocks_file):
            # 读取CSV，兼容中英文列名，code列强制文本格式保留前导零
            df = pd.read_csv(stocks_file, encoding='utf-8-sig', dtype={'code': str})
            
            # 兼容中文列名
            column_mapping = {}
            if '股票代码' in df.columns:
                column_mapping['股票代码'] = 'code'
            if '股票名称' in df.columns:
                column_mapping['股票名称'] = 'name'
            
            if column_mapping:
                df = df.rename(columns=column_mapping)
            
            if 'code' not in df.columns:
                logger.error(f"CSV文件缺少'code'或'股票代码'列: {stocks_file}")
            else:
                STOCKS = df['code'].astype(str).tolist()
                logger.info(f"✓ 已从 {stocks_file} 加载 {len(STOCKS)} 只股票")
    
    # 资金配置
    if 'capital' in config:
        TOTAL_CAPITAL = config['capital'].get('total', TOTAL_CAPITAL)
        STRATEGY_CAPITAL = config['capital'].get('per_strategy', STRATEGY_CAPITAL)
    
    # 双均线策略参数
    DUAL_MA_MA_SHORT = 15
    DUAL_MA_MA_LONG = 60
    DUAL_MA_TAKE_PROFIT = 0.25
    DUAL_MA_STOP_LOSS = 0.10
    if 'dual_ma' in config:
        DUAL_MA_MA_SHORT = config['dual_ma'].get('ma_short', DUAL_MA_MA_SHORT)
        DUAL_MA_MA_LONG = config['dual_ma'].get('ma_long', DUAL_MA_MA_LONG)
        DUAL_MA_TAKE_PROFIT = config['dual_ma'].get('take_profit', DUAL_MA_TAKE_PROFIT)
        DUAL_MA_STOP_LOSS = config['dual_ma'].get('stop_loss', DUAL_MA_STOP_LOSS)
    
    # 海龟策略参数
    if 'turtle' in config:
        CHANNEL_PERIOD = config['turtle'].get('channel_period', CHANNEL_PERIOD)
        POSITION_MODE = config['turtle'].get('position_mode', POSITION_MODE)
        MAX_POSITIONS = config['turtle'].get('max_positions', MAX_POSITIONS)
        TAKE_PROFIT = config['turtle'].get('take_profit', TAKE_PROFIT)
        STOP_LOSS = config['turtle'].get('stop_loss', STOP_LOSS)
    
    # 定投策略参数
    DCA_TAKE_PROFIT = 0.30
    DCA_STOP_LOSS = 0.10
    if 'dca' in config:
        DCA_TAKE_PROFIT = config['dca'].get('take_profit', DCA_TAKE_PROFIT)
        DCA_STOP_LOSS = config['dca'].get('stop_loss', DCA_STOP_LOSS)
    
    logger.info(f"✓ 配置参数已应用: START_DATE={START_DATE}, END_DATE={END_DATE}")
    logger.info(f"  策略资金: {STRATEGY_CAPITAL}")
    logger.info(f"  双均线止盈: {DUAL_MA_TAKE_PROFIT}, 止损: {DUAL_MA_STOP_LOSS}")
    logger.info(f"  海龟止盈: {TAKE_PROFIT}, 止损: {STOP_LOSS}")
    logger.info(f"  定投止盈: {DCA_TAKE_PROFIT}, 止损: {DCA_STOP_LOSS}")


def _cleanup_old_results():
    """清理旧的回测结果文件（.xlsx）"""
    import glob
    
    # 结果目录
    results_dir = os.path.join(project_root, 'data', 'results')
    
    # 查找所有xlsx文件
    pattern = os.path.join(results_dir, '*.xlsx')
    old_files = glob.glob(pattern)
    
    if len(old_files) > 0:
        print(f"[0/6] 清理旧的回测结果文件...")
        for f in old_files:
            try:
                os.remove(f)
                logger.debug(f"Deleted old result file: {f}")
            except Exception as e:
                logger.warning(f"Failed to delete {f}: {e}")
        
        print(f"✓ 已删除 {len(old_files)} 个旧文件\n")
    else:
        print(f"[0/6] 没有旧的回测结果文件需要清理\n")


def _load_stocks_from_config() -> list:
    """从配置文件的 stocks_file 读取股票列表，找不到就直接报错"""
    print(f"[1/6] 从配置文件读取股票列表...")
    config = load_config()
    stocks_file = config.get('data', {}).get('stocks_file', '')
    if not stocks_file:
        logger.error("配置文件中未设置 data.stocks_file，无法获取股票列表")
        raise SystemExit("请在 config/backtest_config.yaml 中设置 data.stocks_file")
    # 支持相对路径（相对于项目根目录）
    if not os.path.isabs(stocks_file):
        stocks_file = os.path.join(project_root, stocks_file)
    if not os.path.exists(stocks_file):
        logger.error(f"股票列表文件不存在: {stocks_file}")
        raise SystemExit(f"股票列表文件不存在: {stocks_file}")
    df = pd.read_csv(stocks_file, encoding='utf-8-sig', dtype={'code': str, '股票代码': str})
    # 兼容中英文列名
    code_col = None
    for col in ('code', '股票代码'):
        if col in df.columns:
            code_col = col
            break
    if code_col is None:
        logger.error(f"CSV文件缺少'code'或'股票代码'列: {stocks_file}")
        raise SystemExit(f"CSV文件缺少'code'或'股票代码'列: {stocks_file}")
    codes = df[code_col].astype(str).str.zfill(6).tolist()
    print(f"✓ 已读取股票列表: {len(codes)} 只股票 ({stocks_file})\n")
    return codes


def main():
    """主程序"""
    global STOCKS  # 声明使用全局变量
    
    print("\n" + "="*70)
    print("回测系统 - 主动量化 vs 被动量化")
    print("="*70 + "\n")
    
    # 0. 加载配置文件
    print("[0/7] 加载配置文件...")
    config = load_config()
    apply_config(config)
    print(f"✓ 配置加载完成\n")
    
    # 1. 清理旧的回测结果文件
    _cleanup_old_results()
    
    # 1. 初始化日志
    logger.remove()
    logger.add(sys.stdout, level="INFO", 
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")
    logger.add("backtest/backtest.log", rotation="100 MB", level="INFO", encoding="utf-8")
    
    logger.info("="*50)
    logger.info("回测系统启动")
    logger.info("="*50)
    
    try:
        # 2. 从配置文件读取股票列表（stocks_file）
        if len(STOCKS) == 0:
            STOCKS = _load_stocks_from_config()
        else:
            print(f"\n[1/6] 使用配置文件中的股票列表: {len(STOCKS)} 只股票\n")
        
        # 3. 初始化数据加载器
        print("[2/6] 初始化数据加载器...")
        data_loader = DataLoader()
        print("✓ 数据加载器初始化完成\n")
        
        # 3.5 读取沪深300涨跌幅（从配置文件）
        print("[2.5/6] 读取沪深300基准涨跌幅...")
        hs300_return = config.get('hs300_return', 0.0)
        hs300_start = config.get('hs300_start', START_DATE)
        hs300_end = config.get('hs300_end', END_DATE)
        print(f"✓ 沪深300 ({hs300_start}~{hs300_end}) 涨跌幅: {hs300_return:.2%}\n")
        
        # 3. 对每只股票运行回测
        print("[2/5] 运行回测（5只股票）...")
        all_results = []
        
        for stock_code in STOCKS:
            print(f"\n  处理股票: {stock_code}...")
            
            # 加载数据（使用关键字参数传递 MA 周期）
            df = data_loader.get_adjusted_prices(
                stock_code, 
                START_DATE, 
                END_DATE,
                ma_short=DUAL_MA_MA_SHORT,
                ma_long=DUAL_MA_MA_LONG,
                channel_period=CHANNEL_PERIOD
            )
            
            if df is None:
                logger.warning(f"跳过 {stock_code}: 无数据")
                continue
            
            # 主动量化：双均线策略
            print(f"    主动量化（双均线）...")
            dual_ma = DualMAStrategy(
                total_capital=STRATEGY_CAPITAL,
                take_profit=DUAL_MA_TAKE_PROFIT,
                stop_loss=DUAL_MA_STOP_LOSS,
                trading_fee_rate=0.0002,
                stamp_duty_rate=0.001
            )
            dual_ma_orders = dual_ma.run(df)
            dual_ma_metrics = _calculate_strategy_metrics(dual_ma, df, dual_ma_orders)
            
            # 主动量化：海龟策略
            print(f"    主动量化（海龟）...")
            turtle = TurtleStrategy(
                total_capital=STRATEGY_CAPITAL,
                channel_period=CHANNEL_PERIOD,
                position_mode=POSITION_MODE,
                max_positions=MAX_POSITIONS,
                take_profit=TAKE_PROFIT,
                stop_loss=STOP_LOSS
            )
            turtle_orders = turtle.run(df)
            turtle_metrics = _calculate_strategy_metrics(turtle, df, turtle_orders)
            
            # 被动量化：月定投策略（跳过）
            print(f"    被动量化（月定投）... [跳过]")
            dca_orders = []
            dca_metrics = _empty_metrics()
            
            # 被动量化：周定投策略（跳过）
            print(f"    被动量化（周定投）... [跳过]")
            weekly_dca_orders = []
            weekly_dca_metrics = _empty_metrics()
            
            # 对比策略
            comparison = compare_strategies(dual_ma_metrics, dca_metrics)
            turtle_comparison = compare_strategies(turtle_metrics, dca_metrics)
            
            # 计算股票涨跌幅（开始价 vs 结束价）
            start_price = df['adj_close'].iloc[0]
            end_price = df['adj_close'].iloc[-1]
            
            # 处理异常情况
            if pd.isna(start_price) or pd.isna(end_price) or start_price == 0:
                stock_price_change = 0.0
            else:
                stock_price_change = (end_price - start_price) / start_price
                # 检查是否为 nan 或 inf
                if pd.isna(stock_price_change) or np.isinf(stock_price_change):
                    stock_price_change = 0.0
            
            # 保存结果
            result = {
                'code': stock_code,
                'name': df.iloc[0].get('name', stock_code) if 'name' in df.columns else stock_code,
                'start_price': start_price,
                'end_price': end_price,
                'stock_price_change': stock_price_change,
                'hs300_return': hs300_return,
                'dual_ma_orders': dual_ma_orders,
                'turtle_orders': turtle_orders,
                'dca_orders': dca_orders,
                'weekly_dca_orders': weekly_dca_orders,
                'dual_ma_metrics': dual_ma_metrics,
                'turtle_metrics': turtle_metrics,
                'dca_metrics': dca_metrics,
                'weekly_dca_metrics': weekly_dca_metrics,
                'comparison': comparison,
                'turtle_comparison': turtle_comparison,
                'dual_ma_daily': dual_ma.daily_values,
                'turtle_daily': turtle.daily_values,
                'dca_daily': [],
                'weekly_dca_daily': []
            }
            all_results.append(result)
            
            print(f"    ✓ {stock_code} 回测完成")
        
        print(f"\n✓ 回测完成: {len(all_results)} 只股票\n")
        
        # 4. 生成 Excel 报告
        print("[3/6] 生成 Excel 报告...")
        excel_path = _generate_excel_report(all_results)
        print(f"✓ Excel 报告已保存: {excel_path}\n")
        
        # 5. 生成可视化图表
        print("[4/5] 生成可视化图表...")
        chart_paths = _generate_charts(all_results)
        print(f"✓ 图表已保存: {len(chart_paths)} 张\n")
        
        # 6. 打印汇总
        print("[5/6] 打印汇总结果...")
        _print_summary(all_results)
        
        # 7. 选股得分与收益相关性分析
        print("\n[6/6] 分析选股得分与收益相关性...")
        _analyze_score_return_correlation(all_results)
        
        logger.info("="*50)
        logger.info("回测系统运行完成!")
        logger.info("="*50 + "\n")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        raise
    
    print("="*70)
    print("程序执行完成!")
    print("="*70 + "\n")


def _calculate_strategy_metrics(strategy, df: pd.DataFrame, orders: List[Dict]) -> Dict:
    """
    计算策略的绩效指标
    
    Args:
        strategy: 策略对象（DualMAStrategy / TurtleStrategy / DCAStrategy）
        df: 价格数据 DataFrame
        orders: 交易记录
        
    Returns:
        绩效指标字典
    """
    # 使用策略运行过程中记录的每日市值
    if hasattr(strategy, 'daily_values') and len(strategy.daily_values) > 0:
        daily_df = pd.DataFrame(strategy.daily_values)
    else:
        # 兼容旧代码：重新计算
        daily_values = []
        for idx, row in df.iterrows():
            current_price = row['adj_close']
            portfolio_value = strategy.get_portfolio_value(current_price)
            daily_values.append({
                'date': row['trade_date'],
                'portfolio_value': portfolio_value
            })
        daily_df = pd.DataFrame(daily_values)
    
    # 计算绩效指标
    metrics = calculate_metrics(orders, daily_df)
    
    return metrics


def _generate_excel_report(all_results: List[Dict]) -> str:
    """
    生成 Excel 报告（多个 Sheet）
    
    Args:
        all_results: 所有股票的回测结果
        
    Returns:
        Excel 文件路径
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"backtest_{timestamp}.xlsx"
    output_path = os.path.join(project_root, 'data', 'results', filename)
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 创建 Excel Writer
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: 交易记录
        _write_trading_records_sheet(writer, all_results)
        
        # Sheet 2: 绩效汇总
        _write_performance_summary_sheet(writer, all_results)
        
        # Sheet 3: 每日市值曲线
        _write_daily_values_sheet(writer, all_results)
        
        # Sheet 4: 策略深度对比
        _write_strategy_comparison_sheet(writer, all_results)
    
    logger.info(f"✓ Excel report generated: {output_path}")
    
    return output_path


def _write_trading_records_sheet(writer, all_results: List[Dict]):
    """写入交易记录 Sheet"""
    rows = []
    
    for result in all_results:
        code = result['code']
        
        # 主动量化交易记录（双均线）
        for order in result['dual_ma_orders']:
            rows.append({
                '股票代码': code,
                '策略类型': '主动量化（双均线）',
                '日期': order['date'],
                '操作': order['action'],
                '价格': order['price'],
                '数量': order['shares'],
                '金额': order['amount'],
                '盈亏': order.get('profit'),
                '收益率': order.get('return_pct'),
                '备注': order.get('reason')
            })
        
        # 主动量化交易记录（海龟）
        for order in result['turtle_orders']:
            rows.append({
                '股票代码': code,
                '策略类型': '主动量化（海龟）',
                '日期': order['date'],
                '操作': order['action'],
                '价格': order['price'],
                '数量': order['shares'],
                '金额': order['amount'],
                '盈亏': order.get('profit'),
                '收益率': order.get('return_pct'),
                '备注': order.get('reason')
            })
        
        # 被动量化交易记录
        for order in result['dca_orders']:
            rows.append({
                '股票代码': code,
                '策略类型': '被动量化（月定投）',
                '日期': order['date'],
                '操作': order['action'],
                '价格': order['price'],
                '数量': order['shares'],
                '金额': order['amount'],
                '盈亏': order.get('profit'),
                '收益率': order.get('return_pct'),
                '备注': order.get('reason')
            })
    
    df = pd.DataFrame(rows)
    df.to_excel(writer, sheet_name='交易记录', index=False)


def _write_performance_summary_sheet(writer, all_results: List[Dict]):
    """写入绩效汇总 Sheet"""
    rows = []
    
    for result in all_results:
        code = result['code']
        name = result.get('name', code)
        stock_change = result.get('stock_price_change', 0.0)
        hs300_return = result.get('hs300_return', 0.0)
        dual_ma = result['dual_ma_metrics']
        turtle = result['turtle_metrics']
        dca = result['dca_metrics']
        
        # 双均线
        rows.append({
            '股票代码': code,
            '股票名称': name,
            '策略类型': '双均线',
            '股票涨跌幅': f"{stock_change:.2%}",
            '沪深300涨跌幅': f"{hs300_return:.2%}",
            '总收益率': f"{dual_ma['total_return']:.2%}",
            '年化收益率': f"{dual_ma['annualized_return']:.2%}",
            '最大回撤': f"{dual_ma['max_drawdown']:.2%}",
            '夏普比率': f"{dual_ma['sharpe_ratio']:.2f}",
            '胜率': f"{dual_ma['win_rate']:.2%}",
            '交易次数': dual_ma['num_trades']
        })
        
        # 海龟
        rows.append({
            '股票代码': code,
            '股票名称': name,
            '策略类型': '海龟',
            '股票涨跌幅': f"{stock_change:.2%}",
            '沪深300涨跌幅': f"{hs300_return:.2%}",
            '总收益率': f"{turtle['total_return']:.2%}",
            '年化收益率': f"{turtle['annualized_return']:.2%}",
            '最大回撤': f"{turtle['max_drawdown']:.2%}",
            '夏普比率': f"{turtle['sharpe_ratio']:.2f}",
            '胜率': f"{turtle['win_rate']:.2%}",
            '交易次数': turtle['num_trades']
        })
        
        # 定投
        rows.append({
            '股票代码': code,
            '股票名称': name,
            '策略类型': '定投',
            '股票涨跌幅': f"{stock_change:.2%}",
            '沪深300涨跌幅': f"{hs300_return:.2%}",
            '总收益率': f"{dca['total_return']:.2%}",
            '年化收益率': f"{dca['annualized_return']:.2%}",
            '最大回撤': f"{dca['max_drawdown']:.2%}",
            '夏普比率': f"{dca['sharpe_ratio']:.2f}",
            '胜率': f"{dca['win_rate']:.2%}",
            '交易次数': dca['num_trades']
        })
    
    df = pd.DataFrame(rows)
    df.to_excel(writer, sheet_name='绩效汇总', index=False)


def _write_daily_values_sheet(writer, all_results: List[Dict]):
    """写入每日市值曲线 Sheet"""
    rows = []
    
    for result in all_results:
        code = result['code']
        
        # 主动量化每日市值（双均线）
        if 'dual_ma_daily' in result and len(result['dual_ma_daily']) > 0:
            for dv in result['dual_ma_daily']:
                rows.append({
                    '股票代码': code,
                    '日期': dv['date'],
                    '双均线市值': dv['portfolio_value'],
                    '海龟市值': None,
                    '定投市值': None
                })
        
        # 主动量化每日市值（海龟）
        if 'turtle_daily' in result and len(result['turtle_daily']) > 0:
            for dv in result['turtle_daily']:
                # 找到对应的行（如果已存在）
                found = False
                for row in rows:
                    if row['股票代码'] == code and row['日期'] == dv['date']:
                        row['海龟市值'] = dv['portfolio_value']
                        found = True
                        break
                
                if not found:
                    rows.append({
                        '股票代码': code,
                        '日期': dv['date'],
                        '双均线市值': None,
                        '海龟市值': dv['portfolio_value'],
                        '定投市值': None
                    })
        
        # 被动量化每日市值（定投）
        if 'dca_daily' in result and len(result['dca_daily']) > 0:
            for dv in result['dca_daily']:
                # 找到对应的行（如果已存在）
                found = False
                for row in rows:
                    if row['股票代码'] == code and row['日期'] == dv['date']:
                        row['定投市值'] = dv['portfolio_value']
                        found = True
                        break
                
                if not found:
                    rows.append({
                        '股票代码': code,
                        '日期': dv['date'],
                        '双均线市值': None,
                        '海龟市值': None,
                        '定投市值': dv['portfolio_value']
                    })
    
    if len(rows) > 0:
        df = pd.DataFrame(rows)
        df.to_excel(writer, sheet_name='每日市值曲线', index=False)


def _write_strategy_comparison_sheet(writer, all_results: List[Dict]):
    """写入策略深度对比 Sheet"""
    rows = []
    
    for result in all_results:
        code = result['code']
        dual_ma = result['dual_ma_metrics']
        turtle = result['turtle_metrics']
        dca = result['dca_metrics']
        
        # 收益能力对比 - 双均线
        rows.append({
            '股票代码': code,
            '对比维度': '总收益率',
            '双均线': f"{dual_ma['total_return']:.2%}",
            '海龟': f"{turtle['total_return']:.2%}",
            '定投': f"{dca['total_return']:.2%}",
            '最优策略': '双均线' if dual_ma['total_return'] == max(dual_ma['total_return'], turtle['total_return'], dca['total_return']) else ('海龟' if turtle['total_return'] == max(dual_ma['total_return'], turtle['total_return'], dca['total_return']) else '定投')
        })
        
        # 收益能力对比 - 海龟
        rows.append({
            '股票代码': code,
            '对比维度': '年化收益率',
            '双均线': f"{dual_ma['annualized_return']:.2%}",
            '海龟': f"{turtle['annualized_return']:.2%}",
            '定投': f"{dca['annualized_return']:.2%}",
            '最优策略': '双均线' if dual_ma['annualized_return'] == max(dual_ma['annualized_return'], turtle['annualized_return'], dca['annualized_return']) else ('海龟' if turtle['annualized_return'] == max(dual_ma['annualized_return'], turtle['annualized_return'], dca['annualized_return']) else '定投')
        })
        
        # 风险特征对比
        rows.append({
            '股票代码': code,
            '对比维度': '最大回撤',
            '双均线': f"{dual_ma['max_drawdown']:.2%}",
            '海龟': f"{turtle['max_drawdown']:.2%}",
            '定投': f"{dca['max_drawdown']:.2%}",
            '最优策略': '双均线' if dual_ma['max_drawdown'] == min(dual_ma['max_drawdown'], turtle['max_drawdown'], dca['max_drawdown']) else ('海龟' if turtle['max_drawdown'] == min(dual_ma['max_drawdown'], turtle['max_drawdown'], dca['max_drawdown']) else '定投')
        })
        
        rows.append({
            '股票代码': code,
            '对比维度': '夏普比率',
            '双均线': f"{dual_ma['sharpe_ratio']:.2f}",
            '海龟': f"{turtle['sharpe_ratio']:.2f}",
            '定投': f"{dca['sharpe_ratio']:.2f}",
            '最优策略': '双均线' if dual_ma['sharpe_ratio'] == max(dual_ma['sharpe_ratio'], turtle['sharpe_ratio'], dca['sharpe_ratio']) else ('海龟' if turtle['sharpe_ratio'] == max(dual_ma['sharpe_ratio'], turtle['sharpe_ratio'], dca['sharpe_ratio']) else '定投')
        })
        
        # 交易行为对比
        rows.append({
            '股票代码': code,
            '对比维度': '胜率',
            '双均线': f"{dual_ma['win_rate']:.2%}",
            '海龟': f"{turtle['win_rate']:.2%}",
            '定投': f"{dca['win_rate']:.2%}",
            '最优策略': '双均线' if dual_ma['win_rate'] >= max(turtle['win_rate'], dca['win_rate']) else ('海龟' if turtle['win_rate'] >= dca['win_rate'] else '定投')
        })
        
        # 计算最优策略（交易次数越少越好）
        min_trades = min(dual_ma['num_trades'], turtle['num_trades'], dca['num_trades'])
        best_trades = '双均线' if dual_ma['num_trades'] == min_trades else ('海龟' if turtle['num_trades'] == min_trades else '定投')
        
        rows.append({
            '股票代码': code,
            '对比维度': '交易次数',
            '双均线': dual_ma['num_trades'],
            '海龟': turtle['num_trades'],
            '定投': dca['num_trades'],
            '最优策略': best_trades
        })
    
    df = pd.DataFrame(rows)
    df.to_excel(writer, sheet_name='策略深度对比', index=False)


def _generate_charts(all_results: List[Dict]) -> List[str]:
    """
    生成可视化图表
    
    Args:
        all_results: 所有股票的回测结果
        
    Returns:
        生成的图表文件路径列表
    """
    # TODO: 实现可视化图表生成
    return []


def _print_summary(all_results: List[Dict]):
    """打印汇总结果"""
    print("\n" + "="*70)
    print("回测汇总结果")
    print("="*70)
    
    for result in all_results:
        code = result['code']
        dual_ma = result['dual_ma_metrics']
        turtle = result['turtle_metrics']
        dca = result['dca_metrics']
        weekly_dca = result['weekly_dca_metrics']
        
        print(f"\n股票: {code}")
        print(f"  主动量化（双均线）:")
        print(f"    总收益率: {dual_ma['total_return']:.2%}")
        print(f"    夏普比率: {dual_ma['sharpe_ratio']:.2f}")
        print(f"    最大回撤: {dual_ma['max_drawdown']:.2%}")
        print(f"  主动量化（海龟）:")
        print(f"    总收益率: {turtle['total_return']:.2%}")
        print(f"    夏普比率: {turtle['sharpe_ratio']:.2f}")
        print(f"    最大回撤: {turtle['max_drawdown']:.2%}")
        print(f"  被动量化（月定投）:")
        print(f"    总收益率: {dca['total_return']:.2%}")
        print(f"    夏普比率: {dca['sharpe_ratio']:.2f}")
        print(f"    最大回撤: {dca['max_drawdown']:.2%}")
        print(f"  被动量化（周定投）:")
        print(f"    总收益率: {weekly_dca['total_return']:.2%}")
        print(f"    夏普比率: {weekly_dca['sharpe_ratio']:.2f}")
        print(f"    最大回撤: {weekly_dca['max_drawdown']:.2%}")
    
    print("\n" + "="*70)


def _analyze_score_return_correlation(all_results: List[Dict]):
    """
    分析选股得分与收益的相关性
    
    读取top10_stocks.csv中的选股得分，与各策略的收益进行相关性分析
    """
    print("\n" + "="*70)
    print("选股得分与收益相关性分析")
    print("="*70 + "\n")
    
    # 读取选股得分
    selection_file = os.path.join(project_root, 'data', 'results', 'top10_stocks.csv')
    if not os.path.exists(selection_file):
        logger.warning("选股结果文件不存在，跳过相关性分析")
        return
    
    df_scores = pd.read_csv(selection_file, encoding='utf-8-sig', dtype={'股票代码': str})
    
    # 清理代码格式（保留前导零，确保6位代码）
    df_scores['code'] = df_scores['股票代码'].astype(str).str.strip().str.zfill(6)
    
    # 构建分析结果
    analysis_data = []
    
    for result in all_results:
        code = str(result['code']).strip().zfill(6)
        score_row = df_scores[df_scores['code'] == code]
        
        if len(score_row) == 0:
            logger.warning(f"未找到股票 {code} 的选股得分")
            continue
        
        total_score = score_row['总得分'].iloc[0]
        
        analysis_data.append({
            'code': code,
            'selection_score': total_score,
            'dual_ma_return': result['dual_ma_metrics']['total_return'],
            'turtle_return': result['turtle_metrics']['total_return'],
            'dca_return': result['dca_metrics']['total_return'],
            'weekly_dca_return': result['weekly_dca_metrics']['total_return']
        })
    
    if len(analysis_data) < 2:
        print(f"数据不足（仅 {len(analysis_data)} 条），无法进行相关性分析\n")
        return
    
    df_analysis = pd.DataFrame(analysis_data)
    
    # 计算相关系数
    correlations = {
        '双均线策略': df_analysis['selection_score'].corr(df_analysis['dual_ma_return']),
        '海龟策略': df_analysis['selection_score'].corr(df_analysis['turtle_return']),
        '月定投策略': df_analysis['selection_score'].corr(df_analysis['dca_return']),
        '周定投策略': df_analysis['selection_score'].corr(df_analysis['weekly_dca_return'])
    }
    
    # 打印相关性分析结果
    print("相关系数矩阵（选股得分 vs 策略收益）:")
    print(f"  双均线策略: {correlations['双均线策略']:.4f}")
    print(f"  海龟策略: {correlations['海龟策略']:.4f}")
    print(f"  月定投策略: {correlations['月定投策略']:.4f}")
    print(f"  周定投策略: {correlations['周定投策略']:.4f}")
    
    print("\n详细数据:")
    print(df_analysis.to_string(index=False, formatters={
        'selection_score': '{:.2f}'.format,
        'dual_ma_return': '{:.2%}'.format,
        'turtle_return': '{:.2%}'.format,
        'dca_return': '{:.2%}'.format,
        'weekly_dca_return': '{:.2%}'.format
    }))
    
    print("\n解读:")
    for strategy, corr in correlations.items():
        if corr > 0.5:
            print(f"  {strategy}: 强正相关 ({corr:.4f}) - 选股得分越高，收益越高")
        elif corr > 0.2:
            print(f"  {strategy}: 弱正相关 ({corr:.4f}) - 得分略有影响")
        elif corr > -0.2:
            print(f"  {strategy}: 几乎无关 ({corr:.4f}) - 选股模型对该策略无指导意义")
        elif corr > -0.5:
            print(f"  {strategy}: 弱负相关 ({corr:.4f}) - 得分高反而收益低")
        else:
            print(f"  {strategy}: 强负相关 ({corr:.4f}) - 选股模型对该策略有反向指导意义")
    
    print("\n" + "="*70)
    
    # 保存到Excel
    _save_correlation_analysis(df_analysis, correlations)


def _save_correlation_analysis(df_analysis: pd.DataFrame, correlations: Dict):
    """保存相关性分析结果到Excel"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"correlation_analysis_{timestamp}.xlsx"
    output_path = os.path.join(project_root, 'data', 'results', filename)
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: 详细数据
        df_analysis.to_excel(writer, sheet_name='相关性数据', index=False)
        
        # Sheet 2: 相关系数
        corr_df = pd.DataFrame(list(correlations.items()), columns=['策略', '相关系数'])
        corr_df.to_excel(writer, sheet_name='相关系数', index=False)
    
    logger.info(f"✓ 相关性分析报告已保存: {output_path}")


if __name__ == "__main__":
    main()
