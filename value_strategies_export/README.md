# 价值投资选股策略包（A股多因子回测平台抽取）

本目录是从 `multi_factor_selection` 回测平台中抽取的 **价值投资相关** 选股策略集合，
连同其运行所需的共享引擎、因子选择器与配置，**自洽可独立运行**（仅需自备行情数据库）。

> 抽取日期：2026-08-01
> 适用市场：A股（中证800 / 沪深300 等成分股池）
> 仅供量化研究与学习，不构成任何投资建议。

---

## 一、包含的策略（9 个）

| 脚本 | 策略 | 说明 |
|---|---|---|
| `run_value_selection.py` | 价值选股 | 破净 + ROE + 现金流质量筛选，价值因子内核（BP / DY 正交） |
| `run_dogs_annual.py` | 狗股策略（Dogs of the Market） | 高股息率低估值组合，年度调仓；复用神奇公式资格过滤 |
| `run_magic_formula.py` | 神奇公式 | Greenblatt ROC + EY 排名，月度/季度调仓 |
| `run_magic_v2.py` | 神奇公式 v2 | 原版改进，复用 `run_magic_formula` 缓存与资格过滤 |
| `run_dividend_low_vol_quality_bt.py` | 红利低波质量复合 | 官方编制法口径（中证红利低波 930955），季度调仓 + 股息率加权 + 单行业上限 |
| `run_ep_neutral.py` | EP 中性策略 | 纯行业中性 Earnings Yield 月度策略（已验证 +64.79%） |
| `run_multifactor.py` | 多因子价值 | 复合多因子选股，复用 EP 价格缓存与基础资料 |
| `run_value_backtest.py` | 价值回测 | 基于 `ValueStockSelector` 的回测入口 |
| `run_apb_backtest.py` | APB 价值内核回测 | 账面市值比（APB）调整因子回测，直接读 SQLite |

## 二、目录结构

```
value_strategies_export/
├── config.py                      # 全局配置（GLOBAL / SELECTION / BACKTEST / VALUE_STRATEGY / DIVIDEND_LOW_VOL / DOGS_OF_MARKET）
├── config_tushare.py              # 数据源配置（local_db_path 指向本地 SQLite）
├── run_monthly_rebalance.py       # 共享回测引擎（get_conn / calc_fee / 调仓日 / 价格缓存等）
├── run_value_selection.py ...    # 9 个价值策略脚本（见上表）
├── src/                           # 价值因子选择器包
│   ├── data_fetcher.py            # 共享数据层（读 SQLite）
│   ├── stock_selector.py          # 选股基类
│   ├── value_stock_selector.py    # 价值选股
│   ├── dogs_of_market_selector.py # 狗股
│   ├── dividend_low_vol_selector.py # 红利低波
│   ├── factor_calculator.py       # 因子计算
│   └── factor_processor.py        # 因子处理
├── requirements.txt
└── README.md
```

## 三、依赖与运行环境

```bash
pip install -r requirements.txt
# TA-Lib 需先装系统二进制；Windows 可用预编译 whl
```

- Python 3.11+（本机验证：pandas 3.0.3 / numpy 2.4.6 / loguru / TA-Lib）
- 不需要联网即可回测（数据来自本地 SQLite）

## 四、数据要求

回测读取本地 SQLite 数据库，路径在 `config_tushare.py` 的 `DATA.local_db_path`
（默认 `D:/tu-shareData/astock_daily.db`）。所需主要表：

- `daily` / `daily_basic`（行情、PE/PB/市值）
- `stock_basic`（行业、上市日期）
- `index_constituent` / `index_daily`（成分股、指数行情）
- `fina_indicator` / `income` / `balance_sheet`（财务质量）
- `dividend_detail`（分红记录，红利类策略用）

> 注意：该数据库不在本包内（体积大且含原始行情），需自行用 Tushare 下载或从此前备份获取。

## 五、运行方式

1. 修改 `config.py` 中的 `GLOBAL`（股票池 `stock_pool`、选股数 `top_n`、回测区间、
   初始资金）与各策略配置节。
2. 在**本目录根**运行（config.py / run_*.py / src/ 同级的布局）：

```bash
# 价值选股
python run_value_selection.py

# 狗股（年度）
python run_dogs_annual.py

# 神奇公式
python run_magic_formula.py

# 红利低波质量复合（季度）
python run_dividend_low_vol_quality_bt.py

# EP 中性
python run_ep_neutral.py
```

## 六、已知注意点

- `run_value_backtest.py` 使用 `from data_fetcher import ...`（顶层导入，不带 `src.` 前缀），
  与多数脚本的 `from src.xxx import` 风格不同。若从本目录根直接运行报导入错误，
  请在该文件顶部加 `import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))`，
  或进入 `src/` 目录运行（确保 config.py 仍在可导入路径）。
- `run_apb_backtest.py` 直接读 SQLite，不依赖 `src/` 与 `config` 顶层导入。

## 七、免责声明

本包所有策略与回测结果均来自历史数据，仅供量化方法研究与学习使用，
不预示未来收益，不构成任何投资建议。使用时请自行评估风险。
