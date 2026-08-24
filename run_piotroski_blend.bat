@echo off
cd /d "%~dp0"
set PY=.\venv_ml\Scripts\python.exe
set LOG=piotroski_blend.log
if exist "%LOG%" del "%LOG%"

echo ================ CONFIG: BLEND50-2010 ================ >> "%LOG%"
"%PY%" run_monthly_rebalance.py 20100101 20251231 --stock-pool zz800 --piotroski-blend 0.5 >> "%LOG%" 2>&1
echo ================ DONE: BLEND50-2010 ================ >> "%LOG%"

echo ================ CONFIG: BLEND50-2010t ================ >> "%LOG%"
"%PY%" run_monthly_rebalance.py 20100101 20191231 --stock-pool zz800 --piotroski-blend 0.5 >> "%LOG%" 2>&1
echo ================ DONE: BLEND50-2010t ================ >> "%LOG%"

echo ================ CONFIG: BLEND50-2020 ================ >> "%LOG%"
"%PY%" run_monthly_rebalance.py 20200101 20251231 --stock-pool zz800 --piotroski-blend 0.5 >> "%LOG%" 2>&1
echo ================ DONE: BLEND50-2020 ================ >> "%LOG%"

echo ================ CONFIG: BLEND25-2010t ================ >> "%LOG%"
"%PY%" run_monthly_rebalance.py 20100101 20191231 --stock-pool zz800 --piotroski-blend 0.25 >> "%LOG%" 2>&1
echo ================ DONE: BLEND25-2010t ================ >> "%LOG%"

echo ================ CONFIG: BLEND25-2020 ================ >> "%LOG%"
"%PY%" run_monthly_rebalance.py 20200101 20251231 --stock-pool zz800 --piotroski-blend 0.25 >> "%LOG%" 2>&1
echo ================ DONE: BLEND25-2020 ================ >> "%LOG%"

echo ================ CONFIG: BLEND75-2010t ================ >> "%LOG%"
"%PY%" run_monthly_rebalance.py 20100101 20191231 --stock-pool zz800 --piotroski-blend 0.75 >> "%LOG%" 2>&1
echo ================ DONE: BLEND75-2010t ================ >> "%LOG%"

echo ================ CONFIG: BLEND75-2020 ================ >> "%LOG%"
"%PY%" run_monthly_rebalance.py 20200101 20251231 --stock-pool zz800 --piotroski-blend 0.75 >> "%LOG%" 2>&1
echo ================ DONE: BLEND75-2020 ================ >> "%LOG%"

echo ================ ALL_DONE ================ >> "%LOG%"
