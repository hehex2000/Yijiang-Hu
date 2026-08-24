#!/usr/bin/env bash
# Piotroski 连续加权(--piotroski-blend) 双窗口验证
# 协议：blend 须在 2010-2019(train) 与 2020-2025(holdout) 两窗口都跑赢 OFF 才算稳健质量 overlay
# 注意：每个组合完整输出落盘(不管道 grep，规避 SIGPIPE 误杀)；跑完用 grep 抓关键数
set -u
cd "$(dirname "$0")"
PY=./venv_ml/Scripts/python.exe
LOG=piotroski_blend.log
: > "$LOG"

run() {
  local tag="$1"; shift
  echo "================ CONFIG: $tag ================" >> "$LOG"
  "$PY" run_monthly_rebalance.py "$@" >> "$LOG" 2>&1
  echo "================ DONE: $tag ================" >> "$LOG"
}

# —— 核心 w=0.5（全窗口 + 两子窗口）——
run "BLEND50-2010" 20100101 20251231 --stock-pool zz800 --piotroski-blend 0.5
run "BLEND50-2010t" 20100101 20191231 --stock-pool zz800 --piotroski-blend 0.5
run "BLEND50-2020" 20200101 20251231 --stock-pool zz800 --piotroski-blend 0.5

# —— 稳健性 w=0.25 / w=0.75（仅两子窗口，证伪"w=0.5 是刀刃"）——
run "BLEND25-2010t" 20100101 20191231 --stock-pool zz800 --piotroski-blend 0.25
run "BLEND25-2020" 20200101 20251231 --stock-pool zz800 --piotroski-blend 0.25
run "BLEND75-2010t" 20100101 20191231 --stock-pool zz800 --piotroski-blend 0.75
run "BLEND75-2020" 20200101 20251231 --stock-pool zz800 --piotroski-blend 0.75

echo "================ ALL_DONE ================" >> "$LOG"
