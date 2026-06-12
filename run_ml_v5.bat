@echo off
cd /d C:\Users\99395\WorkBuddy\multi_factor_selection
call venv_ml\Scripts\activate.bat

echo ================================
echo 开始训练 v5 模型（超参数调优 + 多模型对比）...
echo ================================
python ml_stock_selector_v5.py

echo ================================
echo 训练完成！
echo ================================
pause
