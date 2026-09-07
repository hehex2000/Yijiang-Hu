#!/usr/bin/env bash
set -u
cd /c/Users/99395/WorkBuddy/multi_factor_selection
PY=venv_ml/Scripts/python.exe
OUT=data/results/dividend_low_vol
BAK=_pre_annual_landing_20260904
mkdir -p "$BAK"
NAV="$OUT/bt_quality_nav_20200101_20260723_official_compact_all_12_hfq.csv"
SEL="$OUT/bt_quality_sel_OFFICIAL_OFFICIAL_COMPACT_all_12_20200101_20260723.csv"
PAR="$OUT/_official_official_compact_all_12_bk0_20200101_20260723_partial.csv"

echo "=== [0] backup old quarterly default (if any) + move stale partial ==="
[ -f "$NAV" ] && cp -v "$NAV" "$BAK/" || echo "no old NAV"
[ -f "$SEL" ] && cp -v "$SEL" "$BAK/" || echo "no old SEL"
[ -f "$PAR" ] && mv -v "$PAR" "$BAK/partial_stale.csv" || echo "no stale partial"

echo "=== [1] RUN 1 (fresh, annual default) ==="
"$PY" run_dividend_low_vol_quality_bt.py --mode official_compact --pool all > /tmp/dlq_r1.log 2>&1
echo "RUN1_EXIT=$?"
ls -la "$NAV" "$SEL" 2>/dev/null || echo "RUN1 missing output"
cp -v "$NAV" /tmp/nav_r1.csv
cp -v "$SEL" /tmp/sel_r1.csv

echo "=== [2] move RUN1 partial aside so RUN2 re-selects from scratch ==="
mv -v "$PAR" /tmp/partial_r1.csv || echo "no partial to move (ok if RUN1 wrote none)"

echo "=== [3] RUN 2 (fresh, annual default) ==="
"$PY" run_dividend_low_vol_quality_bt.py --mode official_compact --pool all > /tmp/dlq_r2.log 2>&1
echo "RUN2_EXIT=$?"
ls -la "$NAV" "$SEL" 2>/dev/null || echo "RUN2 missing output"

echo "=== [4] md5 compare (run1 vs run2) ==="
md5sum /tmp/nav_r1.csv "$NAV"
md5sum /tmp/sel_r1.csv "$SEL"
echo "=== DONE ==="
