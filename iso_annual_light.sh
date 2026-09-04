#!/usr/bin/env bash
set -u
cd /c/Users/99395/WorkBuddy/multi_factor_selection
PY=venv_ml/Scripts/python.exe
OUT=data/results/dividend_low_vol
NAV="$OUT/bt_quality_nav_20200101_20211231_official_compact_all_12_hfq.csv"
SEL="$OUT/bt_quality_sel_OFFICIAL_OFFICIAL_COMPACT_all_12_20200101_20211231.csv"
PAR="$OUT/_official_official_compact_all_12_bk0_20200101_20211231_partial.csv"

echo "=== [1] RUN 1 (annual default, 2020-2021, 2 periods) ==="
"$PY" run_dividend_low_vol_quality_bt.py --mode official_compact --pool all --start 20200101 --end 20211231 > /tmp/dlq_lr1.log 2>&1
echo "RUN1_EXIT=$?"
ls -la "$NAV" "$SEL" 2>/dev/null || echo "RUN1 missing output"
cp -v "$NAV" /tmp/nav_lr1.csv; cp -v "$SEL" /tmp/sel_lr1.csv
mv -v "$PAR" /tmp/partial_lr1.csv 2>/dev/null || echo "no partial (ok)"

echo "=== [2] RUN 2 (same) ==="
"$PY" run_dividend_low_vol_quality_bt.py --mode official_compact --pool all --start 20200101 --end 20211231 > /tmp/dlq_lr2.log 2>&1
echo "RUN2_EXIT=$?"
ls -la "$NAV" "$SEL" 2>/dev/null || echo "RUN2 missing output"

echo "=== [3] md5 compare (run1 vs run2) ==="
md5sum /tmp/nav_lr1.csv "$NAV"
md5sum /tmp/sel_lr1.csv "$SEL"
echo "=== DONE ==="
