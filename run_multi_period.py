#!/usr/bin/env python3
"""
多周期批量回测脚本
不修改 config.py 文件，直接在内存中覆盖配置参数
避免过拟合：使用同一套策略参数测试多个年份
"""
import sys
import os
import importlib
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入配置（作为模块对象，可以直接修改其属性）
import config

# 保存原始配置（用于恢复）
ORIGINAL_CONFIG = {
    "BACKTEST_start": config.BACKTEST["start_date"],
    "BACKTEST_end": config.BACKTEST["end_date"],
    "SELECTION_date": config.SELECTION["date"],
}

# 测试周期配置
TEST_PERIODS = [
    {"year": "2017", "start": "20170104", "end": "20171229", "market": "牛市", "desc": "2017年白马股大牛市"},
    {"year": "2018", "start": "20180102", "end": "20181228", "market": "熊市", "desc": "2018年全面下跌"},
    {"year": "2019", "start": "20190102", "end": "20191231", "market": "牛市", "desc": "2019年春季行情+科技牛"},
]


def run_period(period):
    """运行单个周期的回测（不修改文件，只改内存）"""
    year = period["year"]
    start = period["start"]
    end = period["end"]
    market = period["market"]
    desc = period["desc"]

    print(f"\n{'='*70}")
    print(f"  [{year}年] {market} - {desc}")
    print(f"  回测区间: {start} → {end}")
    print(f"{'='*70}")

    # ── 在内存中修改 config 模块的属性 ──────────────────
    config.BACKTEST["start_date"] = start
    config.BACKTEST["end_date"] = end
    config.SELECTION["date"] = end
    # ─────────────────────────────────────────────────────────

    # 重新导入 run_backtest（让修改后的 config 生效）
    import run_backtest
    importlib.reload(run_backtest)

    # 执行选股
    print(f"\n>>> [{year}年] 执行选股...")
    stocks = run_backtest.run_selection()

    if stocks is None or len(stocks) == 0:
        print(f"  ⚠️ {year}年 选股失败，跳过回测")
        return None

    # 执行回测
    print(f"\n>>> [{year}年] 执行回测...")
    run_backtest.run_backtest(stocks)

    print(f"\n{'='*70}")
    print(f"  {year}年 ({market}) 回测完成")
    print(f"{'='*70}")

    return True


def restore_config():
    """恢复原始配置"""
    config.BACKTEST["start_date"] = ORIGINAL_CONFIG["BACKTEST_start"]
    config.BACKTEST["end_date"] = ORIGINAL_CONFIG["BACKTEST_end"]
    config.SELECTION["date"] = ORIGINAL_CONFIG["SELECTION_date"]


def main():
    print("\n" + "="*70)
    print("  多周期批量回测 - 避免过拟合")
    print("  使用同一套策略参数，测试多个年份")
    print("  不修改 config.py 文件，仅修改内存中的配置")
    print("="*70)

    results = []
    for i, period in enumerate(TEST_PERIODS):
        print(f"\n进度: {i+1}/{len(TEST_PERIODS)}")
        success = run_period(period)
        if success:
            results.append(period["year"])

    # 恢复原始配置
    restore_config()
    print(f"\n已恢复 config.py 原始配置: {ORIGINAL_CONFIG['BACKTEST_start']} → {ORIGINAL_CONFIG['BACKTEST_end']}")

    print("\n" + "="*70)
    print("  多周期回测完成")
    print(f"  已完成年份: {', '.join(results) if results else '无'}")
    print("  请查看各年份的回测结果，对比策略在不同市场环境下的表现")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
