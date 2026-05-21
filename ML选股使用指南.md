# ML选股模块使用指南

## ✅ 已完成的功能

### 1. 核心模块
- `ml_stock_selector.py` - ML选股核心类（MLStockSelector）
- `ml_stock_selector_v2.py` - 简化版ML选股脚本
- `ml_stock_selector_v3.py` - 多时间点训练版（推荐）

### 2. 功能特性
- ✅ 支持随机森林（Random Forest）和XGBoost
- ✅ 自动计算多因子（技术因子 + 基本面因子）
- ✅ 使用未来收益率作为训练标签
- ✅ 自动标准化因子
- ✅ 保存/加载训练好的模型
- ✅ 输出TOP N股票列表

---

## 🚀 快速开始

### 方法1：使用已训练好的模型（推荐）

如果已经有训练好的模型文件（`data/models/randorm_forest_model.pkl` 和/或 `xgboost_model.pkl`），直接运行预测：

```bash
cd C:\Users\99395\WorkBuddy\multi_factor_selection
python ml_stock_selector_v3.py
```

### 方法2：从头训练模型

#### 步骤1：安装依赖

```bash
pip install scikit-learn xgboost pandas numpy sqlite3 loguru
```

#### 步骤2：准备训练数据

修改 `ml_stock_selector_v3.py` 中的参数：

```python
# 在 `run_ml_stock_selection()` 函数中
X, y = prepare_training_data_multi_period(
    end_date='20191231',  # 使用2019年及之前的数据
    periods=6              # 使用6个时间点（每6个月一个）
)
```

#### 步骤3：训练模型

```bash
python ml_stock_selector_v3.py
```

训练完成后，模型会自动保存到 `data/models/` 目录。

#### 步骤4：使用模型预测

模型训练完成后，脚本会自动运行预测并输出TOP 10股票。

---

## 📁 文件说明

### 1. `ml_stock_selector.py`
- **功能**：ML选股核心类
- **类**：`MLStockSelector`
- **主要方法**：
  - `prepare_training_data()` - 准备训练数据
  - `train_models()` - 训练ML模型
  - `select_stocks_with_ml()` - 使用ML模型选股
  - `save_models()` - 保存模型
  - `load_models()` - 加载模型

### 2. `ml_stock_selector_v2.py`
- **功能**：简化版ML选股脚本（单次训练）
- **适用场景**：快速测试，小规模数据

### 3. `ml_stock_selector_v3.py`（推荐）
- **功能**：多时间点训练版
- **优势**：使用多个时间点生成训练数据，样本量更大
- **适用场景**：生产环境，完整训练

---

## 📊 因子说明

### 技术因子（实时计算）
1. **动量因子**（momentum_20）- 过去20日收益率
2. **波动率因子**（volatility）- 过去20日收益率标准差（负值，越小越好）
3. **均线因子**（ma_ratio）- 当前价格与MA60的比率（越大越好）

### 基本面因子（来自年报）
1. **市盈率**（pe）- 越低越好（负值）
2. **市净率**（pb）- 越低越好（负值）
3. **市销率**（ps）- 越低越好（负值）
4. **股息率**（dv_ratio）- 越高越好

---

## 🔧 配置参数

### 在 `ml_stock_selector_v3.py` 中修改：

```python
# 训练参数
train_end_date = '20191231'   # 训练数据截止日期
periods = 6                   # 使用过去N个时间点
test_size = 0.2               # 测试集比例

# 预测参数
pred_date = '20200101'        # 预测基准日期
top_n = 10                  # 选择TOP N股票

# 模型类型
model_type = 'both'           # 'random_forest', 'xgboost', 'both'
```

---

## 📈 输出文件

### 1. 模型文件
- `data/models/randorm_forest_model.pkl`
- `data/models/xgboost_model.pkl`

### 2. 选股结果
- `data/results/top10_stocks_ml.csv`

格式：
```csv
code,ml_score
600188,0.1523
600738,0.1421
...
```

---

## 🔗 与现有系统集成

### 方法1：替换选股模块

修改 `run_backtest.py`，使用ML选股结果：

```python
# 在 `run_backtest.py` 的 `main()` 函数中
# 替换原来的选股逻辑
from ml_stock_selector_v3 import run_ml_stock_selection

# 使用ML选股
STOCKS = run_ml_stock_selection(
    train_end_date='20191231',
    pred_date='20200101',
    top_n=10,
    model_type='both'
)
```

### 方法2：对比规则选股和ML选股

```python
# 1. 运行规则选股
python select_top5_stocks.py

# 2. 运行ML选股
python ml_stock_selector_v3.py

# 3. 对比结果
# 规则选股结果：data/results/top10_stocks.csv
# ML选股结果：data/results/top10_stocks_ml.csv
```

---

## ❓ 常见问题

### 1. 训练数据不足
**错误信息**：`训练数据不足！仅获得 0 条有效数据`

**原因**：
- 股票在 `train_end_date` 前上市不足60个交易日
- 股票在 `train_end_date` 后20日没有交易数据
- 股票缺少基本面数据（PE/PB/PS/股息率）

**解决方法**：
- 增加 `periods` 参数（使用更多时间点）
- 增加 `sample_size` 参数（使用更多股票）
- 检查数据库是否有完整数据

### 2. 模型文件未找到
**错误信息**：`未找到模型文件！请先运行训练脚本`

**解决方法**：
```bash
# 先运行训练脚本
python ml_stock_selector_v3.py

# 再运行预测
python -c "from ml_stock_selector_v3 import select_stocks_with_ml; print(select_stocks_with_ml())"
```

### 3. 安装依赖失败
**解决方法**：
```bash
# 安装scikit-learn
pip install scikit-learn

# 安装xgboost
pip install xgboost

# 如果安装慢，使用清华镜像
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scikit-learn xgboost
```

---

## 📞 下一步优化

### 1. 添加更多因子
- 动量因子（5日、10日、60日）
- 成交量因子
- 技术指标（RSI、MACD、KDJ等）

### 2. 优化训练数据
- 使用滚动时间窗口（如过去3年）
- 平衡样本（上涨/下跌样本比例）
- 去除异常值

### 3. 模型优化
- 调参（GridSearchCV）
- 集成学习（Voting、Stacking）
- 深度学习模型（LSTM、Transformer）

### 4. 回测验证
- 使用ML选股结果运行回测
- 对比规则选股和ML选股的收益
- 生成对比报告

---

## 📧 联系作者

如有问题或建议，请联系：
- 作者：huyijiang
- 项目路径：`C:\Users\99395\WorkBuddy\multi_factor_selection\`

---

**最后更新**：2026-05-15
