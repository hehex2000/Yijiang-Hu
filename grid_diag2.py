"""根因隔离：为什么 方案1(加钱) 和 方案2(缩卖距) 在 2022-2025 都无效？

对照 510300，20220102~20251231：
  F1  asym 2/8% 无过滤 @10万   —— 隔离「趋势过滤」是否封锁卖出
  F2  sym 4%  +过滤 @10万      —— 对称网格对照组（应正常买卖）
  F3  sym 4%   无过滤 @10万     —— 对称无过滤对照
  F4  asym 2/4% 无过滤 @10万    —— 方案2卖距缩到4%且无过滤
"""
import sys, os, io, contextlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_grid_backtest import run_grid_backtest

SD, ED = "20220102", "20251231"
SCEN = [
    ("F1 asym 2/8% 无过滤 @10万", "510300.SH", 0.02, "asymmetric", 0.08, False, 100000),
    ("F2 sym  4%   +过滤 @10万", "510300.SH", 0.04, "symmetric",  None, True,  100000),
    ("F3 sym  4%    无过滤 @10万", "510300.SH", 0.04, "symmetric",  None, False, 100000),
    ("F4 asym 2/4%  无过滤 @10万", "510300.SH", 0.02, "asymmetric", 0.04, False, 100000),
]
KEYS = ["初始资金","最终资产","总收益率","年化收益率","最大回撤","夏普比率","网格交易次数","胜率","涨幅","超额收益"]
for label, code, gp, mode, sp, tf, cap in SCEN:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_grid_backtest(code, SD, ED, grid_pct=gp, mode=mode, sell_pct=sp,
                          trend_filter=tf, initial_capital=cap, per_grid_cash=5000, init_position_pct=0.5)
    print("="*64); print(f"{label}"); print("="*64)
    for l in [x.strip() for x in buf.getvalue().splitlines() if any(k in x for k in KEYS)]:
        print(l)
    print()
