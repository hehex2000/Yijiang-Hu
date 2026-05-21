# 多因子选股回测系统

## 📖 项目简介

这是一个基于多因子选股的量化回测系统，支持多种交易策略（双均线、海龟、定投等），可以对比不同策略在相同股票池上的表现。

**核心功能**：
- ✅ 多因子选股（价值、成长、质量、动量、技术五类因子）
- ✅ 多种交易策略回测（主动量化 + 被动量化）
- ✅ 自动计算交易手续费和印花税
- ✅ 生成Excel报告和Markdown分析报告
- ✅ 选股得分与收益相关性分析

---

## 📁 项目结构

```
multi_factor_selection/
├── config/
│   └── backtest_config.yaml      # 回测配置文件（重要！）
├── data/
│   ├── results/                  # 输出目录（选股结果、回测报告）
│   │   ├── top10_stocks.csv      # 多因子选股结果
│   │   ├── backtest_*.xlsx      # 回测Excel报告
│   │   └── correlation_analysis_*.xlsx  # 相关性分析报告
│   └── stock_daily.db           # SQLite数据库（需自行创建）
├── backtest/
│   ├── data_loader.py           # 数据加载器（从SQLite加载股票价格）
│   ├── dual_ma_strategy.py     # 双均线策略（MA20/MA60）
│   ├── turtle_strategy.py       # 海龟策略（channel_period=60）
│   ├── dca_strategy.py         # 月度定投策略
│   ├── weekly_dca_strategy.py  # 周度定投策略
│   ├── metrics.py              # 绩效指标计算（夏普比率、最大回撤等）
│   └── backtest.log           # 回测日志
├── run_backtest.py             # 主程序（运行回测）
├── select_top10_stocks.py      # 多因子选股脚本（可选）
└── README.md                  # 本文件
```

---

## 🔧 安装依赖

### 1. Python环境
- Python 3.9.5 或更高版本

### 2. 安装Python包
```bash
pip install pandas numpy sqlite3 loguru pyyaml openpyxl xlsxwriter
```

### 3. 准备SQLite数据库
- 将A股日线数据放入SQLite数据库
- 默认路径：`D:/tu-shareData/astock_daily.db`
- 表结构：
  - `daily`: 日线数据（ts_code, trade_date, open, close）
  - `adj_factor`: 复权因子（ts_code, trade_date, adj_factor）

---

## ⚙️ 配置文件说明

### `config/backtest_config.yaml`

```yaml
# 数据配置
data:
  start_date: "20200103"  # 回测开始日期
  end_date: "20221231"    # 回测结束日期
  stocks_file: "data/results/top10_stocks.csv"  # 股票列表文件

# 资金配置
capital:
  total: 600000      # 每只股票总资金（元）
  per_strategy: 200000  # 每个策略分配资金（元）

# 双均线策略参数
dual_ma:
  enabled: true
  ma_short: 20         # 短期均线周期（MA20）
  ma_long: 60          # 长期均线周期（MA60）
  take_profit: 0.30    # 止盈线（30%）
  stop_loss: 0.10      # 止损线（10%）

# 海龟策略参数
turtle:
  enabled: true
  channel_period: 60    # 通道周期（60日）
  position_mode: "half" # 仓位模式：'full'=全仓，'half'=半仓
  max_positions: 3      # 最大加仓次数
  take_profit: 0.30     # 止盈线（30%）
  stop_loss: 0.10      # 止损线（10%）
  trading_fee_rate: 0.0002  # 交易手续费率（万分之2）
  stamp_duty_rate: 0.001   # 印花税率（千分之1，仅卖出收取）

# 定投策略参数
dca:
  enabled: true
  amount_per_month: 5000  # 每月定投金额（元）
  take_profit: 0.30       # 止盈线（30%）
  stop_loss: 0.10         # 止损线（10%）
```

---

## 🚀 如何运行回测

### 方法1：使用默认配置（推荐）
```bash
cd C:\Users\99395\WorkBuddy\multi_factor_selection
python run_backtest.py
```

### 方法2：修改配置文件后再运行
1. 编辑 `config/backtest_config.yaml`
2. 修改回测日期、策略参数等
3. 保存文件
4. 运行 `python run_backtest.py`

---

## 📊 输出文件说明

### 1. Excel报告（`backtest_YYYYMMDD_HHMMSS.xlsx`）
包含以下Sheet：
- **交易记录**：所有策略的买卖记录（日期、价格、数量、盈亏）
- **绩效汇总**：总收益率、年化收益率、最大回撤、夏普比率、胜率
- **每日市值曲线**：用于绘制资金曲线图
- **策略深度对比**：不同策略的绩效对比

### 2. 相关性分析报告（`correlation_analysis_YYYYMMDD_HHMMSS.xlsx`）
- **相关性数据**：选股得分与各策略收益
- **相关系数**：选股模型对策略的指导意义

### 3. Markdown报告（`backtest_comparison_report.md`）
- 本次回测配置变化
- 与上次回测的结果对比
- 策略表现排名
- 选股模型优化建议

---

## 📈 策略说明

### 主动量化策略

#### 1. 双均线策略（Dual MA）
- **原理**：MA20上穿MA60买入，下穿卖出
- **仓位**：买入50%仓，卖出全仓
- **止盈止损**：30%止盈，10%止损
- **交易频率**：中等

#### 2. 海龟策略（Turtle）
- **原理**：突破60日最高价买入，跌破60日最低价卖出
- **仓位**：半仓模式（可调整为全仓）
- **加仓**：最多加仓3次
- **止盈止损**：30%止盈，10%止损
- **交易频率**：较低

### 被动量化策略

#### 3. 月度定投策略（Monthly DCA）
- **原理**：每月第一个交易日买入固定金额
- **买入规则**：优先买1000股，买不起买100股
- **止盈止损**：30%止盈，10%止损
- **交易频率**：低（每月一次）
- **特点**：最稳健，夏普比率最高

#### 4. 周度定投策略（Weekly DCA）
- **原理**：每周第一个交易日（周一）买入固定股数
- **买入规则**：每周买入100股
- **止盈止损**：30%止盈，10%止损
- **交易频率**：中等（每周一次）

---

## 🔧 参数调整指南

### 1. 修改止盈止损线
编辑 `config/backtest_config.yaml`：
```yaml
dual_ma:
  take_profit: 0.30  # 止盈线（30%）
  stop_loss: 0.10    # 止损线（10%）
```

### 2. 修改回测时间范围
```yaml
data:
  start_date: "20200103"  # 修改为你的开始日期
  end_date: "20221231"    # 修改为你的结束日期
```

### 3. 修改策略资金分配
```yaml
capital:
  total: 600000        # 每只股票总资金
  per_strategy: 200000  # 每个策略分配资金
```

### 4. 修改海龟策略通道周期
```yaml
turtle:
  channel_period: 60  # 改为20、30、60等
```

### 5. 修改双均线周期
**注意**：需要同时修改 `backtest/data_loader.py` 和 `config/backtest_config.yaml`

在 `data_loader.py` 中：
```python
df['ma20'] = df['adj_close'].rolling(window=20).mean()  # 修改这里的数字
df['ma60'] = df['adj_close'].rolling(window=60).mean()  # 修改这里的数字
```

在 `config/backtest_config.yaml` 中：
```yaml
dual_ma:
  ma_short: 20  # 与data_loader.py中的window一致
  ma_long: 60   # 与data_loader.py中的window一致
```

---

## 📋 选股结果文件格式

### `data/results/top10_stocks.csv`

```csv
排名,股票代码,股票名称,市值,总得分
1,600738,丽尚国潮,41.23,21.34
2,600507,方大特钢,145.07,10.02
...
```

**注意**：
- 股票代码必须是6位数字（前导零不能省略）
- 如果使用其他列名，需要在 `run_backtest.py` 中修改 `apply_config()` 函数

---

## 🔍 常见问题

### 1. 报错：`No data for XXXXX`
**原因**：SQLite数据库中没有该股票的数据
**解决**：
- 检查股票代码格式是否正确（是否需要加`.SH`或`.SZ`后缀）
- 检查数据库中是否有该股票的数据

### 2. 报错：`UnboundLocalError: local variable 'STOCKS' referenced before assignment`
**原因**：在 `run_backtest.py` 的 `main()` 函数中修改全局变量 `STOCKS` 时，没有声明 `global STOCKS`
**解决**：确保在 `main()` 函数开头添加 `global STOCKS`

### 3. 所有策略都亏损？
**可能原因**：
- 回测时间段市场整体下跌
- 策略参数不适合当前市场环境
- 选股模型需要优化

**建议**：
- 尝试不同的回测时间段
- 调整策略参数（如止盈止损线、均线周期）
- 优化选股模型（添加动量、成长性因子）

### 4. 如何添加新策略？
1. 在 `backtest/` 目录下创建新的策略文件（参考 `dca_strategy.py`）
2. 在 `run_backtest.py` 中导入新策略
3. 在 `run_backtest.py` 的 `main()` 函数中添加新策略的回测代码
4. 修改Excel报告生成函数，添加新策略的Sheet

---

## 📈 性能优化建议

### 1. 数据库优化
- 为 `daily` 表的 `ts_code` 和 `trade_date` 字段添加索引
- 定期清理无用数据

### 2. 回测速度优化
- 减少不必要的日志记录（修改 `backtest.log` 的日志级别）
- 使用 `pandas` 的向量化操作，避免循环

### 3. 结果分析优化
- 使用 `matplotlib` 或 `plotly` 生成可视化图表
- 添加多股票组合回测功能

---

## 📞 联系作者

如有问题或建议，请联系：
- 作者：huyijiang
- Email：xxxxxx（请自行填写）
- 项目路径：`C:\Users\99395\WorkBuddy\multi_factor_selection\`

---

## 📄 许可证

（请自行添加许可证信息）

---

**最后更新**：2026-05-15
