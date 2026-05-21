"""
QF-Lib 轻量整合模块（方案A）
==========================
借用 QF-Lib 的分析能力，不碰回测引擎。

功能：
1. 用 TimeseriesAnalysis 交叉验证当前 metrics.py 的计算结果
2. 生成增强版分析报告

依赖：pip install qf-lib
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

from qf_lib.containers.series.qf_series import QFSeries
from qf_lib.containers.series.prices_series import PricesSeries
from qf_lib.common.enums.frequency import Frequency
from qf_lib.analysis.timeseries_analysis.timeseries_analysis import TimeseriesAnalysis
from qf_lib.common.utils.logging.qf_parent_logger import qf_logger

logger = qf_logger.getChild(__name__)


def daily_values_to_qf_series(daily_values: pd.DataFrame,
                              value_col: str = "portfolio_value",
                              date_col: str = "date") -> Tuple[Optional[PricesSeries], Optional[QFSeries]]:
    """
    将每日市值数据转换为 QF-Lib 所需的序列。

    QF-Lib 的 TimeseriesAnalysis 期望接收 PricesSeries（价格序列），
    它会内部调用 to_simple_returns() 转换为收益率。

    Args:
        daily_values: 包含日期和市值的 DataFrame
        value_col: 市值列名
        date_col: 日期列名

    Returns:
        (PricesSeries 价格序列, QFSeries 日收益率序列) 或 (None, None)
    """
    if daily_values is None or daily_values.empty:
        logger.warning("daily_values 为空，无法转换")
        return None, None

    df = daily_values.copy()

    # 兼容不同列名
    if date_col not in df.columns:
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                date_col = col
                break

    if value_col not in df.columns:
        for col in df.columns:
            if 'value' in col.lower() or 'equity' in col.lower() or 'portfolio' in col.lower():
                value_col = col
                break

    if date_col not in df.columns or value_col not in df.columns:
        logger.warning(f"无法识别列名，可用列: {df.columns.tolist()}")
        return None, None

    # 转换日期（兼容整数格式 YYYYMMDD）
    if pd.api.types.is_numeric_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col].astype(str), format='%Y%m%d')
    else:
        df[date_col] = pd.to_datetime(df[date_col])

    df = df.sort_values(date_col).set_index(date_col)

    # 价格序列（PricesSeries）—— 传给 TimeseriesAnalysis
    portfolio_values = df[value_col].astype(float)
    prices_series = PricesSeries(portfolio_values)

    # 日收益率序列（QFSeries）—— 用于其他计算
    daily_returns = portfolio_values.pct_change().dropna()
    daily_returns_series = QFSeries(daily_returns)

    return prices_series, daily_returns_series


def analyze_strategy_returns(prices_series: PricesSeries,
                            benchmark_series: Optional[PricesSeries] = None,
                            freq: Frequency = Frequency.DAILY) -> Dict:
    """
    使用 QF-Lib TimeseriesAnalysis 分析策略收益率。

    注意：TimeseriesAnalysis 的成员变量是直接访问的（不是方法）：
      ta.total_return    （不是 ta.total_cumulative_return()）
      ta.cagr            （不是 ta.annualized_return()）
      ta.annualised_vol  （不是 ta.volatility_annualised()）
      ta.max_drawdown   （不是 ta.max_drawdown()）
      ta.sharpe_ratio    （不是 ta.sharpe_ratio()）
      ta.sorino_ratio   （注意拼写：sorino 不是 sortino）
      ta.calmar_ratio    （不是 ta.calmar_ratio()）

    Args:
        prices_series: 策略价格序列（PricesSeries）
        benchmark_series: 基准价格序列（可选）
        freq: 数据频率

    Returns:
        包含各项指标的字典
    """
    if prices_series is None or len(prices_series) < 2:
        logger.warning("价格序列数据不足，跳过 QF-Lib 分析")
        return {}

    try:
        ta = TimeseriesAnalysis(prices_series, freq)

        # 注意：这些是成员变量，不是方法！不要用括号！
        result = {
            "qf_累计收益": float(ta.total_return),
            "qf_年化收益": float(ta.cagr),
            "qf_波动率": float(ta.annualised_vol),
            "qf_最大回撤": float(ta.max_drawdown),
            "qf_夏普比率": float(ta.sharpe_ratio),
            "qf_索提诺比率": float(ta.sorino_ratio),  # 注意拼写：sorino
            "qf_卡玛比率": float(ta.calmar_ratio),
        }

        logger.info(f"[QF-Lib] 分析完成: 年化={result.get('qf_年化收益', 0):.2%}, "
                    f"夏普={result.get('qf_夏普比率', 0):.2f}, "
                    f"最大回撤={result.get('qf_最大回撤', 0):.2%}")

        return result

    except Exception as e:
        logger.error(f"[QF-Lib] 分析失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {}


def compare_metrics(current_metrics: Dict, qf_metrics: Dict) -> Dict:
    """
    对比当前 metrics.py 计算结果与 QF-Lib 计算结果，用于交叉验证。
    """
    comparison = {
        "match": {},
        "diff": {},
        "discrepancy": []
    }

    # 映射关系：当前平台 metrics.py 的英文 key -> qf_ 前缀的 QF-Lib 指标
    # 注意：metrics.py 返回的是英文 key
    key_map = {
        "total_return": "qf_total_return",
        "annualized_return": "qf_annualized_return",
        "sharpe_ratio": "qf_sharpe_ratio",
        "max_drawdown": "qf_max_drawdown",
    }

    for curr_key, qf_key in key_map.items():
        if curr_key in current_metrics and qf_key in qf_metrics:
            curr_val = current_metrics[curr_key]
            qf_val = qf_metrics[qf_key]

            # 当前平台存的是小数（如 2.9450 表示 294.50%）
            curr_val_norm = float(curr_val)

            diff = abs(curr_val_norm - qf_val)
            rel_diff = diff / (abs(qf_val) + 1e-10)

            comparison["match"][curr_key] = {
                "current": curr_val,
                "qf_lib": qf_val,
                "diff": diff,
                "rel_diff": rel_diff,
                "consistent": rel_diff < 0.05
            }

            if rel_diff >= 0.05:
                comparison["discrepancy"].append({
                    "metric": curr_key,
                    "current": curr_val,
                    "qf_lib": qf_val,
                    "rel_diff_percent": f"{rel_diff * 100:.1f}%"
                })

    return comparison


def print_comparison(comparison: Dict, strategy_name: str, stock_name: str):
    """打印对比结果"""
    print(f"\n{'='*60}")
    print(f"[QF-Lib 交叉验证] {strategy_name} - {stock_name}")
    print(f"{'='*60}")

    if not comparision["match"]:
        print("⚠️ 无匹配指标可对比")
        return

    print(f"\n{'指标':<12} {'当前平台':>12} {'QF-Lib':>12} {'差异':>10} {'状态':>8}")
    print("-" * 60)

    for metric, data in comparision["match"].items():
        status = "✅ 一致" if data["consistent"] else "⚠️ 差异"
        print(f"{metric:<12} {data['current']:>12.4f} {data['qf_lib']:>12.4f} "
              f"{data['rel_diff']:>9.2%} {status:>8}")

    if comparision["discrepancy"]:
        print(f"\n⚠️ 发现差异较大的指标：")
        for item in comparision["discrepancy"]:
            print(f"  - {item['metric']}: 当前={item['current']:.4f}, "
                  f"QF-Lib={item['qf_lib']:.4f}, 差异={item['rel_diff_percent']}")

    print(f"{'='*60}\n")


def analyze_backtest_results(results: List[Dict],
                           daily_values_dict: Dict) -> pd.DataFrame:
    """
    对回测结果进行 QF-Lib 增强分析。
    
    Args:
        results: 当前平台计算的回测结果列表
        daily_values_dict: 每日市值数据字典 {key: {"daily_values": DataFrame, ...}}
    
    Returns:
        增强后的结果 DataFrame（追加 qf_ 前缀的指标）
    """
    enhanced_results = []

    for res in results:
        code = res.get("股票代码", "")
        strategy = res.get("策略", "")
        key = f"{code}_{strategy}"

        if key not in daily_values_dict:
            enhanced_results.append(res)
            continue

        dv = daily_values_dict[key].get("daily_values")
        if dv is None or dv.empty:
            enhanced_results.append(res)
            continue

        # 转换为 QF-Lib 格式
        prices_series, _ = daily_values_to_qf_series(dv)

        if prices_series is None:
            enhanced_results.append(res)
            continue

        # 分析
        qf_metrics = analyze_strategy_returns(prices_series, freq=Frequency.DAILY)

        # 合并结果
        enhanced = res.copy()
        enhanced.update(qf_metrics)

        # 对比
        comparision = compare_metrics(res, qf_metrics)
        if comparision["discrepancy"]:
            enhanced["qf_discrepancy"] = comparision["discrepancy"]

        enhanced_results.append(enhanced)

        # 打印对比结果
        if comparision["discrepancy"]:
            print_comparison(comparision, strategy, res.get("股票名称", code))

    # 转换为 DataFrame
    return pd.DataFrame(enhanced_results)


# ========= 使用示例 =========
if __name__ == "__main__":
    # 测试：用中际旭创的数据
    import sys
    sys.path.insert(0, ".")

    from qf_lib.common.enums.frequency import Frequency
    from backtest.data_loader import DataLoader
    from backtest.metrics import calculate_metrics
    from config import BACKTEST

    # 1. 加载数据
    loader = DataLoader()
    df_hist = loader.get_adjusted_prices("300308", "20230101", "20231231")

    if df_hist is not None and not df_hist.empty:
        # 2. 运行买入持有策略
        from backtest.buy_and_hold_strategy import BuyAndHoldStrategy
        strategy = BuyAndHoldStrategy(total_capital=200000)
        result = strategy.run(df_hist)

        # 3. 当前平台计算指标
        daily_values_df = pd.DataFrame(result.get("daily_values", []))
        metrics = calculate_metrics(
            orders=result.get("trades", []),
            daily_values=daily_values_df,
            risk_free_rate=0.03
        )
        print("[当前平台] 指标:")
        for k in ["总收益率", "年化收益率", "夏普比率", "最大回撤"]:
            if k in metrics:
                print(f"  {k}: {metrics[k]:.4f}")

        # 4. QF-Lib 分析
        prices_series, _ = daily_values_to_qf_series(daily_values_df)
        if prices_series is not None:
            qf_metrics = analyze_strategy_returns(prices_series, freq=Frequency.DAILY)
            print("\n[QF-Lib] 指标:")
            for k, v in qf_metrics.items():
                print(f"  {k}: {v:.4f}")

            # 5. 对比
            comparision = compare_metrics(metrics, qf_metrics)
            print_comparison(comparision, "买入持有", "中际旭创")
    else:
        print("⚠️ 无法加载 300308 数据")
