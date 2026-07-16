"""网格策略多场景扫描（用户指定区间 2022-01-02 ~ 2025-12-31，长熊短牛）。

目标：
  [方案1验证] 非对称 2/8% + 趋势过滤，在 10万/30万/50万 三档资金下表现，
              验证“该方案是否只在大资金(30-50万)才有效”。
  [方案2]     非对称 2/4% + 趋势过滤 @10万（缩卖距 8%->4%），实测表现。

基线复刻：非对称 2/8% + 过滤 @10万（应复现用户报告的“买11卖0”）。
"""
import sys, os, io, contextlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_grid_backtest import run_grid_backtest

SD, ED = "20220102", "20251231"

SCEN = [
    # label, ts_code, grid_pct, mode, sell_pct, trend_filter, capital
    ("基线  2/8% +过滤 @10万",  "510300.SH", 0.02, "asymmetric", 0.08, True, 100000),
    ("方案1 2/8% +过滤 @30万",  "510300.SH", 0.02, "asymmetric", 0.08, True, 300000),
    ("方案1 2/8% +过滤 @50万",  "510300.SH", 0.02, "asymmetric", 0.08, True, 500000),
    ("方案2 2/4% +过滤 @10万",  "510300.SH", 0.02, "asymmetric", 0.04, True, 100000),
    # 对照：515800 同口径，确认是否 ETF 选择导致差异
    ("基线  2/8% +过滤 @10万",  "515800.SH", 0.02, "asymmetric", 0.08, True, 100000),
    ("方案2 2/4% +过滤 @10万",  "515800.SH", 0.02, "asymmetric", 0.04, True, 100000),
]

KEYS = ["初始资金", "最终资产", "总盈亏", "总收益率", "年化收益率", "最大回撤",
        "夏普比率", "网格交易次数", "胜率", "涨幅", "超额收益"]

for label, code, gp, mode, sp, tf, cap in SCEN:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = run_grid_backtest(
            code, SD, ED,
            grid_pct=gp, mode=mode, sell_pct=sp,
            trend_filter=tf, initial_capital=cap,
            per_grid_cash=5000, init_position_pct=0.5,
        )
    out = buf.getvalue()
    summary = [l.strip() for l in out.splitlines()
               if any(k in l for k in KEYS)]
    print("=" * 64)
    print(f"{label}  |  {code}  |  {SD}~{ED}")
    print("=" * 64)
    for l in summary:
        print(l)
    print()
