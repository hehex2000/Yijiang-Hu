#!/usr/bin/env bash
# 网格策略对比脚本：对称放大间距 vs 非对称
cd /c/Users/99395/WorkBuddy/multi_factor_selection
PY=/c/Users/99395/.workbuddy/binaries/python/envs/default/Scripts/python.exe
TS=510300.SH
S=20180102
E=20260703

run() {
  local name="$1"; shift
  echo "=================================================="
  echo "  $name"
  echo "=================================================="
  $PY run_grid_backtest.py "$TS" "$S" "$E" "$@" 2>&1 | grep -E "模式：|总收益率：|年化收益率：|最大回撤：|夏普比率：|网格交易次数：|胜率：|涨幅：|超额收益：|最终资产："
  echo
}

run "A. 对称 2%（基准）"              --grid-pct 0.02
run "B. 对称 4%（放大间距）"          --grid-pct 0.04
run "C. 对称 5%（放大间距）"          --grid-pct 0.05
run "D. 非对称 买2%/卖5%"            --grid-pct 0.02 --mode asymmetric
run "E. 非对称 买2%/卖8%"            --grid-pct 0.02 --mode asymmetric --sell-pct 0.08
