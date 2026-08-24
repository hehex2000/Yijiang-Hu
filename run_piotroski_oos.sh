#!/bin/bash
set -u
LOG=piotroski_oos.log
: > "$LOG"
echo "CONFIG: OFF-train-2010" >> "$LOG"
./venv_ml/Scripts/python.exe run_monthly_rebalance.py 20100101 20191231 --stock-pool zz800 >> "$LOG" 2>&1
echo "DONE: OFF-train-2010" >> "$LOG"
echo "CONFIG: GATE6-train-2010" >> "$LOG"
./venv_ml/Scripts/python.exe run_monthly_rebalance.py 20100101 20191231 --stock-pool zz800 --piotroski-gate 6 >> "$LOG" 2>&1
echo "DONE: GATE6-train-2010" >> "$LOG"
echo "ALL_DONE" >> "$LOG"
