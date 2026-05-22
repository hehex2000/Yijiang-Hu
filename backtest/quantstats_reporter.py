"""
QuantStats 整合模块 - 生成专业回测报告
使用 QuantStats 库生成策略绩效分析和可视化报告
"""

import pandas as pd
import numpy as np
import quantstats as qs
from loguru import logger
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免显示问题
import re
from typing import Dict


# ========= 英文→中文翻译词典 =========
TRANSLATIONS = {
    # 标题和章节
    "Strategy Analysis": "策略分析",
    "Strategy Report": "策略报告",
    "Performance": "绩效表现",
    "Risk": "风险指标",
    "Monthly Returns": "月度收益率",
    "Yearly Returns": "年度收益率",
    "Returns Distribution": "收益率分布",
    "Rolling Statistics": "滚动统计",
    
    # 指标名称
    "Total Return": "累计收益",
    "CAGR": "年化收益率",
    "Sharpe": "夏普比率",
    "Sortino": "索提诺比率",
    "Calmar": "卡玛比率",
    "Max Drawdown": "最大回撤",
    "Volatility": "波动率",
    "Win Rate": "胜率",
    "Loss Rate": "亏损率",
    "Avg Win": "平均盈利",
    "Avg Loss": "平均亏损",
    "Profit Factor": "盈利因子",
    "Expectancy": "期望收益",
    "SQN": "系统质量指数",
    
    # 基准对比
    "vs Benchmark": "vs 基准",
    "Outperformance": "跑赢基准",
    "Underperformance": "跑输基准",
    
    # 时间相关
    "Start Date": "开始日期",
    "End Date": "结束日期",
    "Duration": "持续时间",
    
    # 风险指标
    "Value at Risk": "风险价值 (VaR)",
    "Expected Shortfall": "预期损失",
    "Tail Ratio": "尾部比率",
    "Gain to Pain": "盈亏比",
    
    # 滚动指标
    "Rolling Sharpe": "滚动夏普比率",
    "Rolling Sortino": "滚动索提诺比率",
    "Rolling CAGR": "滚动年化收益率",
    "Rolling Volatility": "滚动波动率",
    "Rolling Max Drawdown": "滚动最大回撤",
    
    # 表格标题
    "Metric": "指标",
    "Strategy": "策略",
    "Benchmark": "基准",
    
    # 月份
    "Jan": "1月",
    "Feb": "2月",
    "Mar": "3月",
    "Apr": "4月",
    "May": "5月",
    "Jun": "6月",
    "Jul": "7月",
    "Aug": "8月",
    "Sep": "9月",
    "Oct": "10月",
    "Nov": "11月",
    "Dec": "12月",
    
    # 其他
    "Annual": "年度",
    "Monthly": "月度",
    "Weekly": "周度",
    "Daily": "日度",
    "Positive Days": "盈利天数",
    "Negative Days": "亏损天数",
    "Positive Months": "盈利月数",
    "Negative Months": "亏损月数",
    "Best Day": "最佳单日收益",
    "Worst Day": "最差单日收益",
    "Best Month": "最佳单月收益",
    "Worst Month": "最差单月收益",
    "Recovery Time": "恢复时间",
    "Skewness": "偏度",
    "Kurtosis": "峰度",
}


def translate_html(html_content: str) -> str:
    """
    将 QuantStats HTML 报告中的英文翻译为中文
    
    Args:
        html_content: 原始 HTML 内容（英文）
        
    Returns:
        翻译后的 HTML 内容（中文）
    """
    translated = html_content
    
    # 按术语长度排序（长的优先，避免部分匹配）
    sorted_translations = sorted(
        TRANSLATIONS.items(), 
        key=lambda x: len(x[0]), 
        reverse=True
    )
    
    # 使用简单的字符串替换（最可靠的方法）
    for en, zh in sorted_translations:
        # 直接替换（忽略大小写）
        pattern = re.compile(re.escape(en), re.IGNORECASE)
        translated = pattern.sub(zh, translated)
    
    logger.info(f"✓ HTML 翻译完成（替换 {len(TRANSLATIONS)} 个术语）")
    
    return translated


def translate_text(text: str) -> str:
    """
    翻译纯文本内容（用于图表标题等）
    """
    translated = text
    for en, zh in TRANSLATIONS.items():
        translated = translated.replace(en, zh)
    return translated


def prepare_returns(daily_values: pd.DataFrame) -> pd.Series:
    """
    将每日市值数据转换为收益率序列（QuantStats 所需格式）
    """
    df = daily_values.copy()
    
    # 兼容不同列名格式
    date_col = None
    value_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'date' in col_lower or 'time' in col_lower:
            date_col = col
        if 'portfolio' in col_lower or 'value' in col_lower or 'equity' in col_lower:
            value_col = col
    
    if date_col is None or value_col is None:
        logger.warning(f"无法识别列名: {df.columns.tolist()}")
        return pd.Series(dtype=float)
    
    # 设置日期索引
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    
    # 去除重复索引
    df = df[~df.index.duplicated(keep='first')]
    
    # 计算每日收益率
    returns = df[value_col].pct_change().dropna()
    
    return returns


def prepare_benchmark(start_date: str, end_date: str, 
                      benchmark_code: str = '000300.SH') -> pd.Series:
    """
    准备基准指数收益率数据（从数据库读取沪深300）
    """
    try:
        import sqlite3
        db_path = "D:/tu-shareData/astock_daily.db"
        conn = sqlite3.connect(db_path)
        
        query = """
            SELECT trade_date, close 
            FROM index_daily 
            WHERE ts_code = ? 
            AND trade_date BETWEEN ? AND ?
            ORDER BY trade_date
        """
        df = pd.read_sql_query(query, conn, params=(benchmark_code, start_date, end_date))
        conn.close()
        
        if df.empty:
            logger.warning(f"未找到基准数据: {benchmark_code}")
            return None
            
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        df = df.set_index('trade_date').sort_index()
        benchmark_returns = df['close'].pct_change().dropna()
        
        logger.info(f"✓ 加载基准数据: {benchmark_code}, {len(benchmark_returns)} 条")
        return benchmark_returns
        
    except Exception as e:
        logger.warning(f"加载基准数据失败: {e}")
        return None


def generate_report(strategy_name: str,
                   returns: pd.Series,
                   output_dir: str = "backtest/result/quantstats",
                   benchmark_returns: pd.Series = None) -> dict:
    """
    生成 QuantStats 回测报告（HTML + 图表）
    
    Args:
        strategy_name: 策略名称
        returns: 策略每日收益率序列
        output_dir: 输出目录
        benchmark_returns: 基准收益率序列（可选）
        
    Returns:
        dict: 生成结果
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 清理文件名中的非法字符（Windows/Linux 通用）
    safe_name = re.sub(r'[\\/*?:"<>|]', '_', strategy_name)
    
    if returns.empty or len(returns) < 2:
        logger.warning(f"{strategy_name}: 收益率数据不足，跳过报告生成")
        return {}
    
    result = {'strategy': strategy_name}
    
    try:
        # 扩展 pandas（添加量化分析函数）
        qs.extend_pandas()
        
        # 对齐基准数据（如果有）
        aligned_returns = returns
        aligned_benchmark = None
        if benchmark_returns is not None:
            common_index = returns.index.intersection(benchmark_returns.index)
            if len(common_index) > 1:
                aligned_returns = returns.loc[common_index]
                aligned_benchmark = benchmark_returns.loc[common_index]
        
        # 1. 生成 HTML 报告
        try:
            html_path = output_path / f"report_{safe_name}.html"
            
            if aligned_benchmark is not None and len(aligned_benchmark) > 1:
                qs.reports.html(aligned_returns, 
                               benchmark=aligned_benchmark,
                               output=str(html_path),
                               title=f"{strategy_name} - 策略分析报告")
            else:
                qs.reports.html(aligned_returns, 
                               output=str(html_path),
                               title=f"{strategy_name} - 策略分析报告")
            
            # 翻译 HTML 报告（英文→中文）
            try:
                with open(html_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                translated_content = translate_html(html_content)
                
                # 保存翻译后的 HTML（覆盖原文件）
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(translated_content)
                
                logger.info(f"✓ HTML报告已翻译: {html_path}")
            except Exception as e:
                logger.warning(f"HTML翻译失败: {e}")
            
            result['html_path'] = str(html_path)
            logger.info(f"✓ HTML报告已生成: {html_path}")
        except Exception as e:
            logger.warning(f"HTML报告生成失败: {e}")
        
        # 2. 生成关键图表（保存为PNG）
        charts = {}
        
        try:
            # 累积收益图
            fig = qs.plots.returns(aligned_returns, 
                                  benchmark=aligned_benchmark)
            if fig:
                chart_path = output_path / f"{safe_name}_cum_returns.png"
                fig.savefig(chart_path, dpi=100, bbox_inches='tight')
                charts['cum_returns'] = str(chart_path)
                matplotlib.pyplot.close(fig)
        except Exception as e:
            logger.warning(f"累积收益图生成失败: {e}")
        
        try:
            # 回撤图
            fig = qs.plots.drawdown(aligned_returns)
            if fig:
                chart_path = output_path / f"{safe_name}_drawdown.png"
                fig.savefig(chart_path, dpi=100, bbox_inches='tight')
                charts['drawdown'] = str(chart_path)
                matplotlib.pyplot.close(fig)
        except Exception as e:
            logger.warning(f"回撤图生成失败: {e}")
        
        try:
            # 月度收益热力图
            fig = qs.plots.monthly_heatmap(aligned_returns)
            if fig:
                chart_path = output_path / f"{safe_name}_monthly_heatmap.png"
                fig.savefig(chart_path, dpi=100, bbox_inches='tight')
                charts['monthly_heatmap'] = str(chart_path)
                matplotlib.pyplot.close(fig)
        except Exception as e:
            logger.warning(f"月度收益热力图生成失败: {e}")
        
        result['charts'] = charts
        if charts:
            logger.info(f"✓ 图表已生成: {len(charts)} 张")
        
        # 3. 计算关键指标
        try:
            metrics = {
                'total_return': qs.stats.comp(aligned_returns),
                'cagr': qs.stats.cagr(aligned_returns),
                'max_drawdown': qs.stats.max_drawdown(aligned_returns),
                'sharpe': qs.stats.sharpe(aligned_returns),
                'sortino': qs.stats.sortino(aligned_returns),
                'win_rate': qs.stats.win_rate(aligned_returns),
                'volatility': qs.stats.volatility(aligned_returns),
            }
            result['metrics'] = metrics
            logger.info(f"✓ 指标计算完成: 总收益={metrics['total_return']:.2%}, 夏普={metrics['sharpe']:.2f}")
        except Exception as e:
            logger.warning(f"指标计算失败: {e}")
        
    except Exception as e:
        logger.error(f"生成报告失败: {e}")
        import traceback
        traceback.print_exc()
    
    return result


def generate_comparison_report(results_dict: dict,
                              output_dir: str = "backtest/result/quantstats",
                              benchmark_code: str = '000300.SH') -> str:
    """
    生成多策略对比报告（简单HTML）
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    html_path = output_path / "comparison_report.html"
    
    try:
        html_content = f"""
        <html>
        <head>
            <meta charset="utf-8">
            <title>策略对比报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>策略对比报告</h1>
            <p>生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <table>
                <tr>
                    <th>策略</th>
                    <th>总收益率</th>
                    <th>CAGR</th>
                    <th>夏普比率</th>
                    <th>最大回撤</th>
                </tr>
        """
        
        for name, returns in results_dict.items():
            if returns.empty or len(returns) < 2:
                continue
            
            html_content += f"""
                <tr>
                    <td>{name}</td>
                    <td>{qs.stats.comp(returns):.2%}</td>
                    <td>{qs.stats.cagr(returns):.2%}</td>
                    <td>{qs.stats.sharpe(returns):.2f}</td>
                    <td>{qs.stats.max_drawdown(returns):.2%}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"✓ 对比报告已生成: {html_path}")
        return str(html_path)
        
    except Exception as e:
        logger.error(f"生成对比报告失败: {e}")
        return ""


if __name__ == '__main__':
    # 测试代码
    print("QuantStats 整合模块")
    print(f"QuantStats 版本: {qs.__version__}")
    print("=" * 60)
