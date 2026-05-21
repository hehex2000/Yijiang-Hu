"""
绩效指标计算模块 - 计算策略的绩效指标
收益率、年化收益、最大回撤、夏普比率等
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from loguru import logger


def calculate_metrics(orders: List[Dict], 
                    daily_values: pd.DataFrame,
                    risk_free_rate: float = 0.03) -> Dict:
    """
    计算策略的绩效指标
    
    Args:
        orders: 交易记录列表
        daily_values: 每日市值 DataFrame（包含 date, portfolio_value）
        risk_free_rate: 无风险利率（默认3%）
        
    Returns:
        绩效指标字典
    """
    if len(orders) == 0:
        logger.warning("No orders to calculate metrics")
        return _empty_metrics()
    
    # 计算各项指标
    total_return = _calculate_total_return(orders, daily_values)
    annualized_return = _calculate_annualized_return(daily_values)
    max_drawdown = _calculate_max_drawdown(daily_values)
    sharpe_ratio = _calculate_sharpe_ratio(daily_values, risk_free_rate)
    win_rate = _calculate_win_rate(orders)
    num_trades = len(orders)  # 统计所有订单（买入+卖出）
    avg_holding_days = _calculate_avg_holding_days(orders)
    max_profit, max_loss = _calculate_max_profit_loss(orders)
    
    metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'win_rate': win_rate,
        'num_trades': num_trades,
        'avg_holding_days': avg_holding_days,
        'max_profit': max_profit,
        'max_loss': max_loss
    }
    
    logger.info(f"✓ Metrics calculated: total_return={total_return:.2%}, sharpe={sharpe_ratio:.2f}")
    return metrics


def _empty_metrics() -> Dict:
    """返回空的绩效指标"""
    return {
        'total_return': 0.0,
        'annualized_return': 0.0,
        'max_drawdown': 0.0,
        'sharpe_ratio': 0.0,
        'win_rate': 0.0,
        'num_trades': 0,
        'avg_holding_days': 0.0,
        'max_profit': 0.0,
        'max_loss': 0.0
    }


def _calculate_total_return(orders: List[Dict], 
                          daily_values: pd.DataFrame) -> float:
    """计算总收益率"""
    if len(daily_values) == 0:
        return 0.0
    
    # DCA策略：用总投入作为分母计算收益率
    if 'total_invested' in daily_values.columns:
        total_invested = daily_values['total_invested'].iloc[-1]
        final_value = daily_values['portfolio_value'].iloc[-1]
        if total_invested == 0:
            return 0.0
        return (final_value - total_invested) / total_invested
    
    # 其他策略：用首日市值作为初始值
    initial_value = daily_values['portfolio_value'].iloc[0]
    final_value = daily_values['portfolio_value'].iloc[-1]
    
    if initial_value == 0:
        # 找第一个非零市值
        non_zero = daily_values[daily_values['portfolio_value'] > 0]
        if len(non_zero) == 0:
            return 0.0
        initial_value = non_zero['portfolio_value'].iloc[0]
    
    return (final_value - initial_value) / initial_value


def _calculate_annualized_return(daily_values: pd.DataFrame) -> float:
    """计算年化收益率"""
    if len(daily_values) < 2:
        return 0.0
    
    total_return = _calculate_total_return([], daily_values)
    
    # 处理极端情况：如果 total_return <= -1，年化收益无意义
    if total_return <= -1:
        return -1.0  # 返回 -100%
    
    days = len(daily_values)
    
    # 年化收益率 = (1 + 总收益)^(252/交易日数) - 1
    # A股年交易日约252天
    try:
        annualized = (1 + total_return) ** (252 / days) - 1
        # 检查是否为 nan 或 inf
        if pd.isna(annualized) or np.isinf(annualized):
            return 0.0
        return annualized
    except (ValueError, ZeroDivisionError):
        return 0.0


def _calculate_max_drawdown(daily_values: pd.DataFrame) -> float:
    """计算最大回撤"""
    if len(daily_values) < 2:
        return 0.0
    
    # 计算累计最大值
    cumulative_max = daily_values['portfolio_value'].cummax()
    
    # 计算回撤
    drawdown = (daily_values['portfolio_value'] - cumulative_max) / cumulative_max
    
    # 最大回撤（负数）
    max_drawdown = drawdown.min()
    
    return abs(max_drawdown)


def _calculate_sharpe_ratio(daily_values: pd.DataFrame,
                             risk_free_rate: float = 0.03) -> float:
    """
    计算夏普比率（标准方法）
    
    公式： Sharpe = Mean(日超额收益) / Std(日超额收益) * sqrt(252)
    其中：日超额收益 = 日收益率 - 日无风险利率
    """
    if len(daily_values) < 2:
        return 0.0
    
    # 计算每日收益率
    daily_returns = daily_values['portfolio_value'].pct_change().dropna()
    
    # 移除 inf 和 -inf 值（DCA策略可能出现）
    daily_returns = daily_returns[~daily_returns.isin([np.inf, -np.inf])]
    
    if len(daily_returns) == 0:
        return 0.0
    
    # 标准夏普比率计算
    # 1. 计算日无风险利率
    daily_rf = risk_free_rate / 252
    
    # 2. 计算日超额收益
    excess_returns = daily_returns - daily_rf
    
    # 3. 计算超额收益的均值和标准差
    mean_excess = excess_returns.mean()
    std_excess = excess_returns.std()
    
    if std_excess == 0 or pd.isna(std_excess) or pd.isna(mean_excess):
        return 0.0
    
    # 4. 年化夏普比率
    sharpe = mean_excess / std_excess * np.sqrt(252)
    
    # 检查 sharpe 是否为 nan 或 inf
    if pd.isna(sharpe) or np.isinf(sharpe):
        return 0.0
    
    # 限制夏普比率范围（避免极端值）
    if sharpe > 10 or sharpe < -10:
        return 0.0
    
    return sharpe


def _calculate_win_rate(orders: List[Dict]) -> float:
    """计算胜率（盈利交易 / 总交易）"""
    sell_orders = [o for o in orders if o['action'] == 'sell']
    
    if len(sell_orders) == 0:
        return 0.0
    
    win_count = len([o for o in sell_orders if o.get('profit', 0) > 0])
    
    return win_count / len(sell_orders)


def _calculate_avg_holding_days(orders: List[Dict]) -> float:
    """计算平均持仓天数"""
    # 简化：返回 0（实际需要匹配买卖订单）
    return 0.0


def _calculate_max_profit_loss(orders: List[Dict]) -> tuple:
    """计算最大单笔盈利和亏损"""
    sell_orders = [o for o in orders if o['action'] == 'sell']
    
    if len(sell_orders) == 0:
        return 0.0, 0.0
    
    profits = [o.get('profit', 0) for o in sell_orders]
    returns = [o.get('return_pct', 0) for o in sell_orders]
    
    max_profit = max(returns) if returns else 0.0
    max_loss = min(returns) if returns else 0.0
    
    return max_profit, max_loss


def compare_strategies(dual_ma_metrics: Dict, 
                      dca_metrics: Dict) -> Dict:
    """
    对比主动量化和被动量化策略
    
    Args:
        dual_ma_metrics: 双均线策略的绩效指标
        dca_metrics: 定投策略的绩效指标
        
    Returns:
        对比结果字典
    """
    comparison = {
        'total_return_diff': dual_ma_metrics['total_return'] - dca_metrics['total_return'],
        'annualized_return_diff': dual_ma_metrics['annualized_return'] - dca_metrics['annualized_return'],
        'max_drawdown_diff': dual_ma_metrics['max_drawdown'] - dca_metrics['max_drawdown'],
        'sharpe_diff': dual_ma_metrics['sharpe_ratio'] - dca_metrics['sharpe_ratio'],
        'win_rate_diff': dual_ma_metrics['win_rate'] - dca_metrics['win_rate'],
        'num_trades_diff': dual_ma_metrics['num_trades'] - dca_metrics['num_trades']
    }
    
    return comparison
