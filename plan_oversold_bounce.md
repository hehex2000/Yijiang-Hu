# 超跌反弹策略 开发计划 (run_oversold_bounce.py)

> 来源视频：BV1Qiuz6WE6Y ｜ UP：跟着Jim学量化 ｜ 白名单 #4（方法论诚实，B+）
> 制定日期：2026-08-21 ｜ 性质：复现 + 显式参数填补 + anti-overfitting 验证
> 关联技能：backtest-lookahead-check（前视自检）、bilibili-critical-summary（白名单已记）

## 0. 定位与可行性结论（结论先行）

- **视频本身结论**：三状态漏斗（超跌→止跌→反弹）+ 全成本计入后，**无稳定优势**；弱市抗跌、强市踏空。
- **本实现目标**：不是"做成能赚钱的策略"，而是——

  1. 把视频的模糊语义（双维度超跌 / 止跌三条件 / 反弹确认）**忠实落成可计算规则**；
  2. 用**显式参数**填补视频空白（跌幅阈值、下跌力度定义、持有天数、止盈止损视频全没给）；
  3. 套 **anti-overfitting + 前视自检**纪律，看它到底有没有 edge。

- **诚实边界（重要）**：

  - 复现若也"无稳定优势"＝**预期内**，与视频结论一致，如实写报告；
  - 复现若"暴赚"＝**先怀疑过拟合/前视**（P0-1 同日、复权假新低、挑标的），**不报喜**；
  - 视频"无参数"留白必须由我们显式决定并**在报告中披露**，不可暗调参数求优。

## 1. 策略语义精确化（视频模糊 → 可计算规则）

视频概念 → 计算定义 → **待定参数（我们决定，需披露）**。

| 状态 | 视频语义 | 计算定义（提议） | 待定参数 |
|---|---|---|---|
| **超跌** | 近期跌幅 + 下跌力度双维度异常 | (a) N1 日收益率 ≤ −X%；(b) N1 日内下跌日占比 ≥ P% 且区间最大回撤 ≥ D% | N1, X, P, D |
| **止跌** | 低点不再下移 + 收盘稳定 + 量能缩小 | (a) 近 N2 日最低价不再创 N3 日新低；(b) 近 N2 日收盘标准差小/收盘在近 N2 高位；(c) 近 N2 日成交量 < 前段均值 × Q | N2, N3, Q |
| **反弹确认** | 止跌后股价真实向上修复 | 收盘价较止跌低点反弹 ≥ Y%（或站上 M 日均线） | Y, M |
| **入场** | 反弹确认后买入 | 反弹确认成立当根 K 线**收盘后**，下一交易日**开盘**买入（严禁同日成交，见 §3 P0-1） | — |
| **出场** | 视频**未给** → 我们定 | 三方案待选（默认 A，B 可选）：A 固定持有 H 日；B 止盈 +T% / 止损 −S%；C 反转退出（重新新低/止跌破坏） | H, T, S |

> 参数初值建议（仅起点，不在全样本上调优）：N1=20, X=15, P=60, D=12, N2=5, N3=10, Q=0.8, Y=5, M=5, H=10/20 两档对照。

## 2. 数据与复权口径（critical）

- **股票池**：大盘股 → 提议 **沪深300**（point-in-time 成分，用 index_weight 快照）。
  - 核查：hs300 成分股快照 2010–2015 缺口（已知）→ 回测起点建议 **≥2015-01-05**（与 adj_factor 起点对齐）或先补快照。
- **行情表**：`astock_daily.db` daily 表。需核查列：`open/high/low/close/pct_chg/vol/amount` + `adj_factor`。
- **复权口径（防失真）**：
  - 信号计算（跌幅 / 低点 / 量能）必须用**后复权**价 → 避免除权除息造"假新低 / 假放量"导致止跌误判；
  - 执行成交用**实际可交易价**（前复权或 raw 经 adj_factor 还原），买入价取信号根次交易日 `open`（复用引擎 `get_open_price`）。
- **量能**：daily 表需有 `vol/amount`；若无 → 按"缺数据优先补全"原则，给 download 脚本命令补（不绕过）。

## 3. 前视偏差自检（backtest-lookahead-check 四问 + 主循环 6 失真）

本策略对全宇宙应用**统一规则**（非 ex-ante 挑 N 只赢家），selection look-ahead 风险集中在以下四点 + 主循环失真：

- **问井（样本/幸存者）**：用 point-in-time 沪深300，不夹带存活标的；起点前加载足够历史（`load_start = start − 1年`）供滚动特征。
- **数活口（样本量）**：统计**每年**触发信号个股数；若任一年 < K 只（提议 K=20）则结论不可外推，报告如实标注。
- **剥出身（信号前视）**：每笔信号只用 t 及之前收盘；入场在 **t+1 open**；严禁用 t 收盘同时赚 t 收益（P0-1）。
- **防过拟合**：参数不在全样本上调；walk-forward 按年切；宽成本对照；扩展候选池（不止选"看起来好"的）。

**回测主循环 6 项失真逐项过**：

| 项 | 风险 | 本策略处置 |
|---|---|---|
| P0-1 信号与收益同日 | t 收盘信号赚 t 收益 | 入场 t+1 open；`bisect_left` 风格持旧仓 |
| P0-2 空仓期稀释 CAGR | 启动前空仓计入 NAV | NAV/基准只从 `START_STRAT` 起算 |
| P1-3 pct_chg ffill | 停牌日幻影收益 | 对 close ffill 后重算收益率 |
| P1-4 成本不随换手 | 换 1/3 与全换同费 | 按实际换手比例计费（复用 `calc_fee`） |
| P1-5 因子除零 | `(upc−dnc)/cnt` 传播 NaN | 除前 `cnt.replace(0, nan)` |
| P1-6 涨跌停阈值不分板块 | 9.5% 只抓主板 | `limit = 5 if ST else (20 if code[:2] in ('30','68') else 10)`；ST 按 `stock_basic.name` 判，非 `ts_code` 前缀 |

## 4. 实现结构（复用平台引擎）

- **新脚本**：`run_oversold_bounce.py`
- **import 共享引擎** `run_monthly_rebalance`：`get_conn` / `calc_fee` / `get_price` / `get_open_price` / `INIT_CAPITAL` / `COMMISSION_RATE` / `COMMISSION_MIN` / `STAMP_DUTY_RATE` / `SLIPPAGE_RATE`。
- **非月度引擎**：自建**逐日事件循环**（per stock 扫描三态信号 → 命中后 t+1 open 建仓 → 持有/退出规则驱动平仓 → 聚合 NAV）。
- **CLI**：`--universe hs300 --start 20150105 --end 20260731 --oversold-pct --decline-force --bounce-pct --hold-days --cost narrow|wide --walkforward`

## 5. 成本建模（真实分科目）

- 买入：佣金(≥MIN) + 过户费 + 滑点；卖出：佣金 + 印花 + 过户费 + 滑点。复用 `calc_fee`，**禁止单参数 `cost_one_way`**。

## 6. 验证与反过拟合（mandatory）

- walk-forward 年度切段（前段估参 / 后段测试）；
- 宽成本（narrow vs wide）对照；
- 扩展池 vs 精选对比（看生存者偏差）；
- 基准：沪深300 买入持有（同起点）；
- **报告指标**：总收益 / 年化 / 最大回撤 / 夏普 / 胜率 / 盈亏比 / 平均持有天数 / **每年信号数**；
- **诚实声明**：选股是否 ex-ante free + 前视 vs ex-ante 对照（若涉及挑标的则给表）。

## 7. 开发步骤（task 拆解）

1. **数据核查**：daily 表列 + adj_factor 覆盖 + hs300 快照数 → 补全命令（若有缺口）。
2. **信号函数** `build_signal()`：超跌/止跌/反弹三态，后复权价。
3. **事件循环 + 执行**：t+1 open 建仓，复用 `calc_fee`。
4. **NAV 聚合 + 指标 + 基准对照**（只从 START_STRAT 起算）。
5. **CLI + walk-forward + 宽成本 + 扩展池**开关。
6. **本地跑验证**（用户本机），填 `oversold_bounce_report.md`。

## 8. 交付与验证命令（用户本机自跑）

```bat
cd C:\Users\99395\WorkBuddy\multi_factor_selection

rem 基准变体（沪深300, 后复权信号, t+1 open 入场）
.\venv_ml\Scripts\python.exe run_oversold_bounce.py 20150105 20260731 --universe hs300 --cost narrow

rem 宽成本 + walk-forward（反过拟合）
.\venv_ml\Scripts\python.exe run_oversold_bounce.py 20150105 20260731 --universe hs300 --cost wide --walkforward

rem 持有期对照（H=20）
.\venv_ml\Scripts\python.exe run_oversold_bounce.py 20150105 20260731 --universe hs300 --hold-days 20 --cost narrow
```

## 9. 红旗 / 边界

- 视频已自承"无稳定优势"；复现若也无优势＝预期内，不视为失败。
- 若出现高收益：先查 P0-1 同日、前视选股、复权假新低 → **不急着报喜**。
- 白名单 Jim #4，方法可信；但其"无参数"留白须由我们显式决定并披露，不可暗调参数求优。
- 与既有策略矩阵关系：本策略独立于月度调仓引擎（事件驱动），落地后**不并入**红利低波/狗股等月度体系，单独评估。
