#!/usr/bin/env bash
# 单调性验证：F>=6 / F>=9 各跑 全窗口2010-2025 + 子窗口2020-2025
# 统一落指纹 f197a65e9f9a（与已有 GATE7/GATE8 同口径）
# 输出全量落盘，不用 grep 管道，避免 SIGPIPE 杀 python
set -u
cd /c/Users/99395/WorkBuddy/multi_factor_selection
PY=./venv_ml/Scripts/python.exe
LOG=piotroski_gate69.log
: > "$LOG"

run() {
  local tag="$1"; shift
  echo "========== CONFIG: $tag ==========" >> "$LOG"
  "$PY" -u run_monthly_rebalance.py "$@" >> "$LOG" 2>&1
  echo "========== DONE: $tag ==========" >> "$LOG"
}

run GATE6-2010   20100101 20251231 --stock-pool zz800 --piotroski-gate 6
run GATE9-2010   20100101 20251231 --stock-pool zz800 --piotroski-gate 9
run GATE6-2020   20200101 20251231 --stock-pool zz800 --piotroski-gate 6
run GATE9-2020   20200101 20251231 --stock-pool zz800 --piotroski-gate 9

echo "ALL_DONE" >> "$LOG"
