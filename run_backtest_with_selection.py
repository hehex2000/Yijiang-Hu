"""
用选股结果运行回测
读取 selection_results.csv 中的 TOP N 股票，运行多策略回测
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger

# 读取配置
try:
    from config import BACKTEST, DATA_FETCHER, STRATEGIES_TO_TEST
except ImportError:
    BACKTEST = {"start_date": "2023-01-01", "end_date": "2023-12-31", "total_capital": 100000}
    DATA_FETCHER = {"local_db_path": "D:/tu-shareData/astock_daily.db"}
    STRATEGIES_TO_TEST = []  # 空 = 测试所有策略

# 初始化日志
log_file = BACKTEST.get("log_file", "backtest.log")
logger.add(log_file, rotation="500 MB", level="INFO")

print("\n" + "="*70)
print("多策略回测 - 使用选股结果")
print("="*70 + "\n")

# 1. 读取选股结果
print("[1/5] 读取选股结果...")
try:
    df = pd.read_csv("data/results/selection_results.csv", dtype={'股票代码': str})
    # 确保股票代码是6位格式（补前导零）
    df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)
    print(f"[OK] 读取到 {len(df)} 只股票")
    print(f"  股票代码: {df['股票代码'].tolist()[:10]}...")
except Exception as e:
    print(f"[FAIL] 读取选股结果失败: {e}")
    raise

# 2. 初始化回测模块
print("\n[2/5] 初始化回测模块...")
try:
    from backtest.data_loader import DataLoader
    from backtest.metrics import calculate_metrics
    
    # 动态导入策略类（根据配置）
    STRATEGY_CLASSES = {}
    
    # 总是导入买入持有（基础策略）
    from backtest.buy_and_hold_strategy import BuyAndHoldStrategy
    STRATEGY_CLASSES["买入持有"] = BuyAndHoldStrategy
    
    # 根据配置导入其他策略
    if not STRATEGIES_TO_TEST or "双均线" in STRATEGIES_TO_TEST:
        from backtest.dual_ma_strategy import DualMAStrategy
        STRATEGY_CLASSES["双均线"] = DualMAStrategy
    
    if not STRATEGIES_TO_TEST or "海龟" in STRATEGIES_TO_TEST:
        from backtest.turtle_strategy import TurtleStrategy
        STRATEGY_CLASSES["海龟"] = TurtleStrategy
    
    if not STRATEGIES_TO_TEST or "MACD/RSI" in STRATEGIES_TO_TEST:
        from backtest.macd_rsi_strategy import MACDRSIStrategy
        STRATEGY_CLASSES["MACD/RSI"] = MACDRSIStrategy
    
    if not STRATEGIES_TO_TEST or "MACD/KDJ" in STRATEGIES_TO_TEST:
        from backtest.macd_kdj_strategy import MACDKDJStrategy
        STRATEGY_CLASSES["MACD/KDJ"] = MACDKDJStrategy
    
    if not STRATEGIES_TO_TEST or "月定投" in STRATEGIES_TO_TEST:
        from backtest.dca_strategy import DCAStrategy
        STRATEGY_CLASSES["月定投"] = DCAStrategy
    
    # 如果配置了策略列表，只保留配置的
    if STRATEGIES_TO_TEST:
        STRATEGY_CLASSES = {k: v for k, v in STRATEGY_CLASSES.items() if k in STRATEGIES_TO_TEST}
    
    data_loader = DataLoader(db_path=DATA_FETCHER.get("local_db_path", "D:/tu-shareData/astock_daily.db"))
    print(f"[OK] 回测模块初始化完成（{len(STRATEGY_CLASSES)} 种策略）")
except Exception as e:
    print(f"[FAIL] 初始化回测模块失败: {e}")
    raise

# 3. 运行回测
# 导入 QuantStats 报告生成器
try:
    from backtest.quantstats_reporter import prepare_returns, generate_report
    ENABLE_QUANTSTATS = True
    print("[✓] QuantStats 报告模块已加载")
except ImportError:
    ENABLE_QUANTSTATS = False
    print("[WARN] QuantStats 未安装，跳过图表生成")

print("\n[3/5] 运行回测...")
print(f"  回测期间: {BACKTEST.get('start_date', '2023-01-01')} 至 {BACKTEST.get('end_date', '2023-12-31')}")
print(f"  策略数量: {len(STRATEGY_CLASSES)} 种")
print(f"  股票数量: {len(df)} 只\n")

results = []
failed = []
daily_values_dict = {}  # 保存每日市值数据用于生成报告
output_dir = "backtest/result"  # 输出目录（提前定义供 QuantStats 使用）

for _, row in df.iterrows():
    code = str(row['股票代码'])
    name = row.get('股票名称', code)
    
    print(f"📊 处理 {code} ({name})...")
    
    # 加载历史数据
    try:
        # 转换日期格式（从 "2023-01-01" 到 "20230101"）
        start_date_fmt = BACKTEST.get('start_date', '2023-01-01').replace('-', '')
        end_date_fmt = BACKTEST.get('end_date', '2023-12-31').replace('-', '')
        
        print(f"  正在加载 {code} 从 {start_date_fmt} 至 {end_date_fmt} 的数据...")
        
        df_hist = data_loader.get_adjusted_prices(
            code=code,
            start_date=start_date_fmt,
            end_date=end_date_fmt
        )
        
        print(f"  数据加载完成: 类型={type(df_hist)}, ", end="")
        if df_hist is None:
            print("返回 None")
        elif isinstance(df_hist, pd.DataFrame):
            print(f"DataFrame 行数={len(df_hist)}")
        else:
            print("未知类型")
        
        if df_hist is None or len(df_hist) == 0:
            print(f"  ⚠️ 无历史数据")
            failed.append(code)
            continue
        
        # 运行每个策略
        for strategy_name, StrategyClass in STRATEGY_CLASSES.items():
            try:
                strategy = StrategyClass(total_capital=BACKTEST.get("total_capital", 200000))
                result = strategy.run(df_hist)
                
                # 计算指标
                daily_values_list = result.get('daily_values', [])
                daily_values_df = pd.DataFrame(daily_values_list) if daily_values_list else pd.DataFrame()
                
                # 保存每日市值数据（用于生成 QuantStats 报告）
                if ENABLE_QUANTSTATS and not daily_values_df.empty:
                    key = f"{code}_{strategy_name}"
                    daily_values_dict[key] = {
                        'strategy_name': strategy_name,
                        'code': code,
                        'name': name,
                        'daily_values': daily_values_df
                    }
                
                metrics = calculate_metrics(
                    orders=result.get('trades', []),
                    daily_values=daily_values_df,
                    risk_free_rate=0.03
                )
                
                results.append({
                    '股票代码': code,
                    '股票名称': name,
                    '策略': strategy_name,
                    '总收益率': metrics.get('total_return', 0.0),
                    '年化收益率': metrics.get('annualized_return', 0.0),
                    '最大回撤': metrics.get('max_drawdown', 0.0),
                    '夏普比率': metrics.get('sharpe_ratio', 0.0),
                    '交易次数': metrics.get('num_trades', 0),
                    '胜率': metrics.get('win_rate', 0.0)
                })
            except Exception as e:
                print(f"  ❌ {strategy_name} 策略失败: {e}")
                failed.append(f"{code}_{strategy_name}")
                continue
        
        print(f"  ✅ 完成 ({len(STRATEGY_CLASSES)} 种策略)")
        
    except Exception as e:
        print(f"  ❌ 处理失败: {e}")
        failed.append(code)
        continue

print(f"\n[OK] 回测完成: {len(results)} 条结果, {len(failed)} 个失败")

# 3.5 生成 QuantStats 图表和报告
if ENABLE_QUANTSTATS and daily_values_dict:
    print("\n[3.5/5] 生成 QuantStats 报告...")
    
    # 创建输出目录
    qs_output_dir = os.path.join(output_dir, "quantstats")
    os.makedirs(qs_output_dir, exist_ok=True)
    
    # 导入 prepare_returns
    from backtest.quantstats_reporter import prepare_returns, generate_report
    
    # 按收益率排序，只生成 TOP 10 的报告（避免生成过多）
    top_results = sorted(results, key=lambda x: x['总收益率'], reverse=True)[:10]
    
    generated = 0
    for res in top_results:
        code = res['股票代码']
        strategy_name = res['策略']
        key = f"{code}_{strategy_name}"
        
        if key in daily_values_dict:
            dv = daily_values_dict[key]['daily_values']
            if not dv.empty:
                try:
                    # 转换为收益率序列
                    returns = prepare_returns(dv)
                    if returns.empty or len(returns) < 2:
                        continue
                    
                    # 生成报告（传入 returns 而非 daily_values）
                    report_result = generate_report(
                        strategy_name=f"{code}_{res['股票名称']}_{strategy_name}",
                        returns=returns,
                        output_dir=qs_output_dir
                    )
                    if report_result and 'html_path' in report_result:
                        generated += 1
                except Exception as e:
                    print(f"  ⚠️ 生成报告失败 {code}_{strategy_name}: {e}")
    
    print(f"[OK] 已生成 {generated} 份 QuantStats 报告到: {qs_output_dir}")

# 4. 保存和显示结果
print("\n[4/5] 保存和显示结果...\n")

if results:
    # 转换为 DataFrame
    results_df = pd.DataFrame(results)
    
    # 保存到 CSV
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"backtest_report_{datetime.now().strftime('%Y%m%d')}.csv")
    
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"[OK] 结果已保存到: {output_path}")
    
    # 打印汇总
    print("\n" + "="*70)
    print("回测结果汇总 (TOP 20 股票)")
    print("="*70 + "\n")
    
    # 按总收益率排序
    results_df_sorted = results_df.sort_values('总收益率', ascending=False)
    
    # 打印 TOP 20
    print("📈 TOP 20 策略 (按总收益率排序):")
    for i, (_, row) in enumerate(results_df_sorted.head(20).iterrows(), 1):
        total_ret = row['总收益率']
        annual_ret = row['年化收益率']
        sharpe = row['夏普比率']
        print(f"  {i}. {row['股票代码']} {row['股票名称']} - {row['策略']}: {total_ret:+.2%} (年化: {annual_ret:+.2%}, 夏普: {sharpe:.2f})")
    
    # 打印 BOTTOM 20
    print("\n📉 BOTTOM 20 策略 (按总收益率排序):")
    for i, (_, row) in enumerate(results_df_sorted.tail(20).iterrows(), 1):
        total_ret = row['总收益率']
        annual_ret = row['年化收益率']
        sharpe = row['夏普比率']
        print(f"  {i}. {row['股票代码']} {row['股票名称']} - {row['策略']}: {total_ret:+.2%} (年化: {annual_ret:+.2%}, 夏普: {sharpe:.2f})")
    
    # 打印策略平均表现
    print("\n📈 策略平均表现:")
    strategy_avg = results_df.groupby('策略').agg({
        '总收益率': 'mean',
        '年化收益率': 'mean',
        '夏普比率': 'mean',
        '最大回撤': 'mean',
        '胜率': 'mean'
    }).sort_values('总收益率', ascending=False)
    
    for strategy, row in strategy_avg.iterrows():
        avg_ret = row['总收益率']
        sharpe = row['夏普比率']
        win_rate = row['胜率']
        print(f"  {strategy}: 平均收益 {avg_ret:+.2%} (夏普: {sharpe:.2f}, 胜率: {win_rate:.1%})")
    
else:
    print("[WARN] 无回测结果")

print("\n" + "="*70)
print("回测完成!")
print("="*70 + "\n")
