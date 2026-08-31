# %B 背离买入增强 —— 设计说明 + A/B 归因报告（%B 细节附录）

> ⚠️ **本文件已并入总报告：[`bollinger_enhancement_report.md`](bollinger_enhancement_report.md)**
> 总报告是**唯一权威结论文档**（覆盖 Head-fake / %B / MFI / OBV 四候选 + walk-forward）。
> 本文件保留为 **%B 背离的机制细节与 review 记录**（代码行号、Bug 1–4 修复史、定义修正对照），供追溯用。

> **脚本**: `run_mean_reversion_pctb_divergence_ablation.py`  
> **策略插件**: `backtest/mean_reversion_plugin.py`（`pctb_divergence` 开关）  
> **视频来源**: B站 BV1FP876bEAo（布林带进阶篇）  
> **日期**: 2026-08-28  
> **状态**: 代码 review 已修复（Bug 1–4 全修），本文档已同步到修复后状态，含严格 K 日极值定义的重跑结果。

---

## 1. 实验目的

验证视频 BV1FP876bEAo 提出的 **%B 看涨背离** 能否给现有均值回归策略带来真增量。

控制单一变量 `pctb_divergence`，其余参数全同 `config.STRATEGIES["mean_reversion"]`：

| 配置 | `pctb_divergence` | 说明 |
|------|-------------------|------|
| **C0 基线** | `False` | 当前默认，不使用 %B 背离 |
| **C1 +背离** | `True` | 开启 %B 看涨背离作为额外买入触发 |

> 纪律：视频说法无回测支撑，仅作候选点验证。验证出正贡献才考虑进 opt-in 默认，不静默改全局默认。

---

## 2. %B 看涨背离逻辑

### 2.1 %B 指标

```
%B = (close - lower_band) / (upper_band - lower_band)
```

- %B ∈ [0, 1]，0 = 贴在下轨，1 = 贴在上轨
- 当 `upper == lower`（布林带极度收口）时分母为 0，代码用 `replace(0, np.nan)` 处理

### 2.2 看涨背离定义（修复后：严格 K 日滚动极值）

**视频口径**：价格在 K 日窗口内创出新低，但 %B 未同步创新低 → 抛压枯竭、动能衰减 → 看涨预警。

**代码实现**（`mean_reversion_plugin.py` 第 177–187 行）：

```python
data['pct_b'] = (data['adj_close'] - data['lower']) / denom        # 第 177 行
prev_pctb = data['pct_b'].shift(1)                                  # 第 178 行
K = self.pctb_lookback                                             # 默认 10
prev_close_div = data['adj_close'].shift(1)                         # t-1 收盘（与 prev_close 同值，命名沿用历史）
price_lower = prev_close_div <= prev_close_div.rolling(K).min()     # 第 185 行：t-1 收盘 ≤ K 日窗口极值（创 K 日新低）
pctb_higher = prev_pctb > prev_pctb.rolling(K).min()                # 第 186 行：t-1 的 %B > K 日窗口 %B 极值（未同步创新低）
pctb_bull_div = (price_lower & pctb_higher & cond_not_widening).fillna(False)  # 第 187 行
```

- 比较全部基于 **t-1 及更早的滚动极值**（不含 t 当日），**无未来函数**。
- 额外要求 `cond_not_widening`：布林带未喇叭口张开（趋势启动时禁止入场）。
- ⚠️ **定义修正史**：初版曾用两点比较 `prev_close_div < prev_close_div.shift(K)`（即"t-1 < t-1-K"两点），与"创 K 日新低"语义不符。经 review 改为 `rolling(K).min()` 严格极值，并重跑（见第 5 节对照，净收益由 +2.91pp → +1.79pp 更保守）。

### 2.3 入场执行（修复后）

% B 背离是独立即时入场，不经 headfake 闸门（`mean_reversion_plugin.py` 第 241–248 行）：

```python
if self.pctb_divergence and self.position == 0 and self._pending is None \
        and self.market_allowed[i] and bool(data.iloc[i]['pctb_bull_div']):  # 第 290–291 行
    success = self._enter_long(date, open_price, 0.0, 0.0, atr_arr[i],
                               reason_suffix="[%B背离买]")            # 第 292 行
    if success:                                                      # 第 293 行（修复 Bug 1）
        self._pctb_entries += 1
    if success and bool(data.iloc[i]['buy_signal']):                 # 第 296 行（修复 Bug 4：重叠归因）
        self._pctb_overlap += 1
```

- 买入用 t 日开盘价执行（`open_price`），信号由 t-1 数据生成
- 传入 `z_val=0.0, rsi_val=0.0`（背离入场不依赖 Z/RSI 阈值，仅记录用）
- `_pctb_entries` / `_pctb_overlap` 计数器（第 107–108 行复位）跟踪触发次数与"与 `buy_signal` 同日重叠"次数

### 2.4 优先级

同一天如果 `pctb_bull_div` 和 `buy_signal` 同时为 True：
1. %B 背离先触发买入（position > 0）
2. `buy_signal` 分支条件 `self.position == 0` 为 False，不再触发

→ %B 背离优先于正常信号，reason 标记为 `[%B背离买]`。重叠次数单独由 `_pctb_overlap` 记录（实测仅占 5.6%，见第 5 节）。

---

## 3. 实验设置

| 参数 | 值 |
|------|----|
| 回测区间 | `20190101` ~ `20260731` |
| 股票池 | 沪深 300 成分股，as-of `20190101` 快照前 40 只（按代码排序） |
| 每支本金 | 100,000 元 |
| 仓位模式 | half（半仓 50%） |
| 基准指数 | 000300.SH（沪深 300） |
| 布林带参数 | period=20, std=2.0 |
| RSI 参数 | period=14, oversold=30, overbought=70 |
| Z-Score 阈值 | 2.0 |
| 硬止损 | 3% |
| ATR 追踪止损 | `use_atr_sl=True`（plugin 默认） |
| %B 背离回看 K | 10（`pctb_lookback` 默认值，config 未覆盖） |
| 手续费 | 佣金万分之二（最低 5 元）+ 印花税卖出单边千分之一 |
| 复权方式 | 前复权（adj_factor） |

### 3.1 股票池构建

```python
def build_universe(n=40):
    df = _get_index_constituents_from_db("000300.SH", as_of_date=START)   # 第 50 行
    codes = sorted(df["code"].tolist())[:n]
    return codes
```

- 使用 **as-of 起始日快照**，消除幸存者偏差（吸取 headfake ablation 教训）
- 复用 `run_backtest._get_index_constituents_from_db`，含数据边界回退
- 按代码排序取前 40 只，确定性且可复现

### 3.2 单只回测流程

```
load_stock_prices(code, START, END, conn, lookback_days=250)
  → 返回前复权 OHLCV + adj_factor，含 START 前 250 日回溯期
df = df.reset_index(drop=True)                                       # 第 80 行：保证索引 0-based 连续
start_idx = 第一个 trade_date >= START 的行号
  → 回测从 start_idx 开始，回溯期用于计算指标（布林带/RSI/Z-Score）
strat = MeanReversionStrategyPlugin(CAPITAL, cfg)
res = strat.run(df, start_idx)
  → 返回 {"returns", "trades", "daily_values"}
```

---

## 4. 输出指标

### 4.1 逐股记录（`ablation_C0.csv` / `ablation_C1.csv`）

| 字段 | 说明 |
|------|------|
| `code` | 股票代码（6 位数字） |
| `ret` | 收益率（%） |
| `win_rate` | 胜率（FIFO 匹配买卖对计算） |
| `trades` | 交易笔数 |
| `max_dd` | 最大回撤（%） |
| `pctb_entries` | %B 背离触发买入次数（已按 `success` 计数，修复 Bug 1） |
| `pctb_overlap` | 其中与 `buy_signal` 同日的重叠次数（归因用，修复 Bug 4） |
| `final_val` | 最终净值 = CAPITAL × (1 + ret/100) |

### 4.2 汇总统计（`summarize()` 第 108 行）

| 指标 | 计算方式 |
|------|----------|
| `mean` | 各股收益算术平均 |
| `median` | 各股收益中位数 |
| `n_pos` | 正收益股票数 |
| `n_beat` | 跑赢沪深 300 的股票数 |
| `win_rate_mean` | 各股胜率均值 |
| `trades_mean` | 各股交易笔数均值 |
| `max_dd_mean` | 各股最大回撤均值 |
| `pctb_entries_mean` | 各股 %B 背离触发次数均值（第 121 行） |
| `pctb_overlap_mean` | 各股重叠触发次数均值（第 122 行，修复 Bug 4） |

### 4.3 落盘文件

```
data/results/mean_reversion_pctb_divergence_ablation/
  ├── ablation_compare.csv   # 两组汇总对比
  ├── ablation_C0.csv         # C0 逐股记录
  └── ablation_C1.csv         # C1 逐股记录
```

---

## 5. A/B 对照结果（严格 K 日极值定义，修复后重跑）

| 配置 | 均值% | 中位% | 正收益 | 跑赢指 | 胜率% | 均交易 | 均回撤% | %B买入/只 | 重叠/只 |
|---|---|---|---|---|---|---|---|---|---|
| C0 基线(无%B背离) | +10.74 | +8.87 | 38/40 | 0/40 | 63.70 | 26.6 | 5.72 | 0.0 | — |
| C1 +%B背离买入 | +12.53 | +12.45 | 36/40 | 0/40 | 56.04 | 74.7 | 8.74 | 31.9 | 1.8 |
| 沪深300基准 | +54.51 | — | — | — | — | — | — | — | — |
| Δ(C1−C0) | **+1.79** | **+3.58** | −2 | 0 | −7.66 | +48.0 | +3.02 | +31.9 | — |

### 5.1 定义修正对照（诚实标注口径变化）

| 指标 | 两点比较（旧，初版） | 严格 K 日极值（新，本版） | 变化 |
|---|---|---|---|
| 均值% Δ | +2.91 | +1.79 | **−1.12pp** |
| 中位% Δ | +4.10 | +3.58 | −0.52pp |
| 胜率% Δ | −5.10 | −7.66 | −2.56pp |
| 均交易 Δ | +46.1 | +48.0 | +1.9 |
| 均回撤% Δ | +2.26 | +3.02 | +0.76pp |
| %B买入/只 | 28.5 | 31.9 | +3.4 |

→ 严格定义下净收益改善更小、代价更明显，结论更保守也更稳健：**两种定义下 %B 背离均为正增量**，方向不变，仅幅度不同。

### 5.2 诊断结论

1. **正贡献但结构清晰**：净值 +1.79pp、中位 +3.58pp（中位改善>均值 → 普惠而非个别大票拉动）。与 headfake 过滤**机制镜像**——headfake 是"少做交易省回撤但少赚"，%B 背离是"多做交易赚更多但回撤升"。
2. **代价明确**：交易数 26.6→74.7（+180%）、胜率 63.70→56.04（−7.66pp）、回撤 5.72→8.74（+3.02pp）。%B 看涨背离在趋势中途"回调衰竭"点频繁触发买入，吃到更多中段反弹，也吃到更多假衰竭。
3. **归因干净（回应 review Bug 4）**：与 `buy_signal` 同日的重叠触发仅 **1.8/只（占 31.9 的 5.6%）**，绝大多数 %B 买入是纯背离独立触发，不存在"把正常信号算进背离"的显著偏差。
4. **换手成本敏感**：当前为单票 10 万简化手续费口径；真实多票组合里 48 笔/只×40 只年化换手会显著拉高摩擦成本，+1.79pp 净收益中有多少被手续费吃掉需复核（平台 MA5 经验 ~−4%/年换手成本，提示该增量可能部分被摩擦吞噬）。
5. **与 headfake 互补而非替代**：headfake 减少假突破亏损、%B 背离增加回调衰竭买点，作用于不同环节；但二者同时开启的交互未验证，当前 A/B 各自单开。
6. **策略域本身跑不赢指数**：C0/C1 均 0/40 跑赢沪深300（+54.51%），属均值回归域在长牛的通病，与 %B 背离无关。

### 5.3 两关复核（regime-gate + 真实分科目成本）

纪律要求新因子须过「大盘趋势门控」+「真实分科目成本」两关，确认不是牛市样本内过拟合、且净收益在真实摩擦下仍成立。三组各隔离单一变量 `pctb_divergence`，其余配置全同。

| 配置 | 均值% | 中位% | 正收益 | 跑赢指 | 胜率% | 均交易 | 均回撤% | %B买入/只 |
|---|---|---|---|---|---|---|---|---|
| C0 基线(简单成本) | +10.74 | +8.87 | 38/40 | 0/40 | 63.70 | 26.6 | 5.72 | 0.0 |
| C1 +%B背离(简单) | +12.53 | +12.45 | 36/40 | 0/40 | 56.04 | 74.7 | 8.74 | 31.9 |
| C0g regime门控 | +1.48 | +1.36 | 26/40 | 0/40 | 50.28 | 7.3 | 2.87 | 0.0 |
| C1g +%B+regime | +3.28 | +3.85 | 30/40 | 0/40 | 56.02 | 36.2 | 6.57 | 16.7 |
| C0r 真实成本 | +10.27 | +8.26 | 37/40 | 0/40 | 63.70 | 26.6 | 5.81 | 0.0 |
| C1r +%B+成本 | +11.22 | +11.46 | 35/40 | 0/40 | 56.04 | 74.7 | 9.01 | 31.9 |
| 沪深300基准 | +54.51 | — | — | — | — | — | — | — |

**三对照 Δ(C1 − C0)**（隔离变量 pctb_divergence）：

| 对照 | Δ均值 | Δ中位 | Δ胜率 | Δ交易 | Δ回撤 |
|---|---|---|---|---|---|
| ① 简单成本（基线） | +1.79 | +3.58 | −7.66 | +48.0 | +3.02 |
| ② 大盘趋势门控（消除熊市接飞刀告警） | **+1.80** | +2.49 | +5.75 | +28.9 | +3.70 |
| ③ 真实分科目成本（佣万2.5+滑点0.1%+过户0.001%+日期感知印花税） | **+0.96** | **+3.20** | −7.66 | +48.0 | +3.20 |

**解读**：
- **regime-gate 关（②）通过**：C1−C0 边缘 +1.80pp ≈ 基线 +1.79pp，**几乎不变**，证明 %B 背离增量不是纯牛市样本内过拟合，在「大盘低于 MA60 时禁止开仓」的严格过滤下依然成立。「熊市接飞刀」告警基本可消除。
  - 注意：门控本身把 C0 绝对收益从 +10.74 压到 +1.48（−9.3pp），说明均值回归本就在「 Correction 期（市场低于 MA60）」最赚钱，门控把这些最佳买点也禁了。这是门控与该策略域的天然冲突，**不代表 pctb 边缘有问题**——我们只用它证明 pctb 边缘与牛熊无关。
  - ② 的胜率 Δ +5.75pp 是门控下交易数骤减（C0g 仅 7.3 笔）导致的小样本噪声，不单独解读；核心看 Δ均值稳定。
- **真实成本关（③）通过但腰斩**：边缘从 +1.79pp（简单）压到 **+0.96pp（均值）/ +3.20pp（中位）**。此前担心的「+48 笔/只换手把增量吃光」未发生——因简单口径本就含万2佣金+千1印花税，新增主要是滑点0.1%/双边；边缘仍正，但幅度约为原来的一半。**诚实结论：真实净增量约 +1pp（均值口径），并非免费午餐。**
- **综合**：两关均显示正增量 → 满足「过两关才以 opt-in 形式保留」的门槛；但真实净增量温和（+0.96pp 均值），代价是换手 +180%、回撤 +3pp。属「边际 opt-in」，非默认增强。

> 真实成本口径说明（base_strategy 新增 opt-in `real_cost`）：开时 buy/sell 用 `_real_fee_buy/_real_fee_sell` = 佣金万2.5 + 滑点0.1%（双边）+ 过户费0.001%（双边）+ 日期感知印花税（2023-08-28 前0.1%/后0.05%）；关时维持原简单口径（佣金万2 + 卖出印花税0.1% 固定，无滑点/过户费），与历史数字完全一致。regime-gate 用基准 000300.SH 日线 MA(60)，t-1 低于 MA 则禁开仓（无未来函数）。两开关均默认关，不影响其他插件。

---

## 6. Code Review 结论（本轮已全修）

### 6.1 已修复（对比 headfake ablation）

| 问题 | headfake ablation | 本脚本 |
|------|-------------------|--------|
| 幸存者偏差 | 用最新快照 | ✅ as-of START 快照（第 50 行） |
| 索引连续性 | 隐式依赖 | ✅ `reset_index(drop=True)`（第 80 行） |
| headfake `_enter_long` 返回值 | 忽略 | ✅ 已检查 `success` |

### 6.2 本轮修复清单（4 项，均已落地）

| # | 问题 | 处置 | 代码位置 |
|---|---|---|---|
| Bug 1 | `_pctb_entries` 计数虚高：`_enter_long` 返回值被忽略，无条件 +1 | **修**：检查 `success` 才 +1 | plugin 第 243–246 行 |
| Bug 2 | "创 K 日新低"实为两点比较（`shift(K)`），与注释/语义不符 | **修**：改为严格 K 日滚动极值（`rolling(K).min()`），文档同步 | plugin 第 185–186 行 |
| Bug 3 | 同日背离+信号重叠被算进 `_pctb_entries` | **修（归因）**：新增 `_pctb_overlap` 计数器（第 108/248 行）；实测重叠仅 1.8/只，影响极小 | plugin 第 247–248 行 / 脚本 103/122 行 |
| Bug 4 | `cancel_or_ignore` 残留（取 headfake 的 `_cancelled_count`，%B 路径恒为 0） | **删**：移除该字段，新增 `pctb_overlap` 落盘 | 脚本 102–103 行 |

> ⚠️ 注意：旧版本文档曾将 Bug 1–4 列为"已知问题/待修复"，但代码已全部修复完成，现改为"已修复"。以此权威版为准。

### 6.3 非问题澄清

| 项 | 说明 |
|----|------|
| `REPLACE(trade_date,'-','')` | `index_constituent` 表已为纯数字格式，REPLACE 无害 |
| `prev_close_div` 命名 | 与 `prev_close` 重复 shift(1)，命名沿用历史但值正确（t-1 收盘） |
| `{s['n_pos']:>{6}}` 格式 | f-string 中 `:>{6}` 合法，`{6}` 被解析为字面量宽度 |

---

## 7. 运行方式

```bash
cd /c/Users/99395/WorkBuddy/multi_factor_selection
venv_ml/Scripts/python.exe run_mean_reversion_pctb_divergence_ablation.py
```

无命令行参数，所有配置在脚本顶部硬编码：
- `START` / `END` / `CAPITAL` / `TOP_N` / `BENCH`
- `CONFIGS` 字典定义两组对照配置（`pctb_divergence=False/True`）

输出到 `data/results/mean_reversion_pctb_divergence_ablation/`。

---

## 8. 依赖关系

```
run_mean_reversion_pctb_divergence_ablation.py
  ├── run_backtest.py
  │   ├── load_stock_prices()        # 加载前复权行情
  │   ├── calc_win_rate_from_trades()  # FIFO 胜率计算
  │   ├── max_drawdown_with_dates()  # 最大回撤
  │   ├── _get_index_constituents_from_db()  # as-of 成分股快照（第 50 行调用）
  │   └── DB_PATH                   # 本地数据库路径
  ├── backtest.mean_reversion_plugin.MeanReversionStrategyPlugin
  │   ├── %B 计算 (第 177–178 行)
  │   ├── 看涨背离信号 (第 182–187 行)
  │   ├── 大盘趋势门控掩码 (第 85 行定义 / 第 249 行装载)
  │   ├── 背离入场执行 (第 290–298 行，含 market_allowed[i] 门控)
  │   └── _pctb_entries / _pctb_overlap 计数器 (第 107–108 行复位)
  └── config.STRATEGIES["mean_reversion"]  # 基线参数
```

---

## 9. 改进路线

1. ~~接 `--regime-gate` 重跑~~ ✅ **已完成（§5.3 ②）**：边缘 +1.80pp，与基线持平，证明非牛市过拟合。
2. ~~真实分科目成本复核~~ ✅ **已完成（§5.3 ③）**：边缘 +0.96pp（均值）/ +3.20pp（中位），腰斩但仍正。
3. **参数敏感性扫描**：`pctb_lookback` ∈ {5, 10, 15, 20}，观察最优 K。
4. **与 headfake 组合消融**：`pctb_divergence=True & headfake_filter=True`，观察叠加效果（交互未验证）。
5. **样本扩展**：补下跌市 / 全市场对照，提升结论鲁棒性。

---

## 10. 纪律声明与最终结论

- 视频 BV1FP876bEAo 的 %B 背离说法无回测支撑，本次仅作候选点验证。
- **两关复核结论**：regime-gate 下边缘 +1.80pp（稳健）、真实成本下 +0.96pp 均值 / +3.20pp 中位（腰斩仍正）。满足"过两关才以 opt-in 形式保留"的门槛 → **可保留为 opt-in（默认关）**。
- **但非默认增强**：真实净增量温和（均值口径约 +1pp），代价是换手 +180%、回撤 +3pp、胜率 −7.7pp。属"边际 opt-in"——是否启用由用户按风险预算决定，不静默改全局默认、不进价值/红利等默认策略。
- `pctb_divergence` / `market_regime_gate` / `real_cost` 三个开关均默认 False，互不干扰、不影响其他插件。
- 布林带真增量候选剩 **MFI/OBV 量能确认**，待用户指定下一步。
