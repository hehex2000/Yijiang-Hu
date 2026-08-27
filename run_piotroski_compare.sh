cd /c/Users/99395/WorkBuddy/multi_factor_selection
PY=./venv_ml/Scripts/python.exe
LOG=piotroski_compare_v2.log
: > "$LOG"
run() {
  local label="$1"; local start="$2"; local end="$3"; local flags="$4"
  echo "===== CONFIG: $label ($start-$end $flags) =====" >> "$LOG"
  $PY run_monthly_rebalance.py "$start" "$end" --stock-pool zz800 $flags >> "$LOG" 2>&1
  echo "" >> "$LOG"
}
run "OFF-2010"     20100101 20251231 ""
run "GATE8-2010"   20100101 20251231 "--piotroski-gate 8"
run "DISTRESS-2010" 20100101 20251231 "--piotroski-distress"
run "GATE7-2010"   20100101 20251231 "--piotroski-gate 7"
run "OFF-2020"     20200101 20251231 ""
run "GATE8-2020"   20200101 20251231 "--piotroski-gate 8"
run "DISTRESS-2020" 20200101 20251231 "--piotroski-distress"
run "GATE7-2020"   20200101 20251231 "--piotroski-gate 7"
echo "ALL_DONE" >> "$LOG"
