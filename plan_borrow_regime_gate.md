# 借鉴另一 agent 的 regime gate 思路修改我们策略 —— 修改计划（不执行）

> 状态：计划文档。今天（2026-08-27）**不写代码、不跑测试**，仅产出此计划。工具②（run_activity_tax_check.py）收尾顺延至明天。
> 红线：另一 agent 的 `run_regime_gate_ab.py` / `run_left_right_regime.py` 为非本会话所写，**只读不碰不删**。

---

## 1. 结论先行（TL;DR）

- 他的思路 = **布林带宽分位 + 站上 MA20** 合成 0/0.5/1.0 目标仓位，做 A/B 对照。
- 他的 A/B 实证的教训：**波动率维度门控对均值回归/价值类策略是负贡献**（红利低波 ON +10% vs OFF +38%；价值 ON +35% vs OFF +52%，且回撤反恶化）。
- 因此**不能把他的 gate 照搬进我们的 价值/红利低波/PEG/神奇公式**。这些策略在低波动（挤压/squeeze）环境本来就该满仓赚稳定收益，用带宽减仓等于砍掉最赚钱的段。
- 真正值得借鉴的有两点（均与"具体数字"无关）：
  1. **隔离式 A/B 门控开关纪律**：`--use-gate` 前后对照、不证明正贡献绝不并入。我们已有 `--ic-mode`/`--timing-gate`/`--sector-gate`，可统一成 `--regime-gate <维度>`。
  2. **维度教训**：波动率维度无效、价格/广度维度 overlay 有效。我们平台**已有**有效的价格+广度维度 overlay，所以代码改动极小。

## 2. 他的思路（只读 review，文件红线不动）

文件：`run_regime_gate_ab.py`（另一 agent 写，READ-ONLY）

- 核心函数 `regime_target_ratio(idx_code, trade_date, squeeze_th=0.25, bb_win=20, bb_lookback=120)`：
  - 调 `_bb_width_pct`（布林带宽分位，复用 run_monthly_rebalance 的）判震荡/趋势；
  - 叠加 `price > MA20` 判多头排列；
  - 返回目标仓位 `0.0 / 0.5 / 1.0`。
- 回测里用 `target_stock_value = equity * target_ratio` 缩放买入，**震荡半仓、不确定空仓**。
- 主流程 `main()` 跑 A（OFF 满仓）vs B（ON 门控）两组对照，输出收益/回撤/夏普/超额(vs300) 差值。
- 价值：提供了**干净的可复用 A/B 报告框架**（对照汇总 + 结论自动判断）。

## 3. 我们的现状（已验证有效，勿动）

| 模块 | 维度 | 关键函数 | 状态 |
|---|---|---|---|
| `market_timing_overlay.py` | **广度**（站上MA20/60/200占比 + AD% + NH-NL） | `compute_breadth_oscillator` / `position_cap(osc,boil,ice,floor)` / `build_gate_series` | ✅ 非对称只卖不买闸门，已接 Kara / ETF轮动（ETF轮动 +75.59%、Kara 减震 -37→-34.42） |
| `sector_state_machine.py` | **价格结构**（MA 对齐判右侧） | `classify_state(close_asc,...)` → ACCEL_BOTTOM/RIGHT_TREND/TREND_ACCEL/UNKNOWN；`GATE_PASS=(RIGHT_TREND,TREND_ACCEL,UNKNOWN)` | ✅ 已接 ETF轮动 `--sector-gate`（仅进右侧+趋势加速） |

这两条都是**价格/广度维度**，非波动率维度，已证有效。结论：**我们已有"对的维度"，缺的是"统一开关 + 维度红线"。**

## 4. 维度对照表（他的 gate vs 我们的 overlay）

| 维度 | 他的 gate | 我们现有 | 对均值回归(价值/红利/PEG) | 对趋势(动量/ETF) |
|---|---|---|---|---|
| 波动率(BB带宽) | ✅ 用 | ❌ 无 | **负贡献**（砍掉最赚段） | 中性/略负 |
| 价格/右侧(MA对齐) | 轻量(above MA20) | ✅ `sector_state_machine.classify_state` | 不适用（本就满仓） | ✅ 有效（已 --sector-gate） |
| 广度(站上MA占比/NH-NL) | ❌ 无 | ✅ `market_timing_overlay.position_cap` | ✅ 有效（下行保护） | ✅ 有效 |

## 5. 最小改动修改计划

### P0（明天收尾）工具② `run_activity_tax_check.py`
- 唯一卡点：`ep_neutral`(12030 笔) `fetch_close` 逐标的开 SQLite 连接超时。
- 修复：把逐标的 `fetch_close` 改为**单次批量查询**（`WHERE ts_code IN (...)` 一次拉全样本 close），内存拼 `close_mat`。
- 验证输出：8 策略 × {年化 / 最大回撤 / 活跃税·年 / 总成本} 四列表 + 是否超 `BENCH_TAX=0.065` 阈值。

### P1 共享引擎增加统一 `--regime-gate <维度>` 开关（run_monthly_rebalance.py）
- 新增统一入口 `apply_regime_gate(positions, gate_dim, trade_date, bench_idx, ...)`：
  - `gate_dim='breadth'` → 复用 `market_timing_overlay`（**预计算 gate series**，模式照抄 `run_etf_rotation_v6_merged.py` 的预计算，勿按日重算 1GB 矩阵），`position_cap` 缩放 equity。
  - `gate_dim='price'` → 复用 `sector_state_machine.classify_state`，状态不在 `GATE_PASS` 则 `cap=0`（或 floor）。
  - `gate_dim=None`（默认）→ 不改原策略（满仓调仓）。
  - `gate_dim='volatility'` → **显式拒绝/告警**：对均值回归策略预期负贡献，打印警告并退出，不并入。
- 保持默认 `None`：绝不改变现有任何策略的默认行为。

### P2 各策略接线 + 维度红线
- 价值/红利低波/PEG/神奇公式 `run_*.py`：默认无门控；仅允许 `--regime-gate breadth` 做 A/B；**禁 `--regime-gate volatility`**。
- 动量/ETF轮动/趋势类 `run_*.py`：已有 `--sector-gate`（price 维度）；补 `--regime-gate breadth` 同上对照。

### P3 借鉴他的"隔离 A/B 报告" + 沉淀维度教训
- 把他的 `main()` A/B 汇总框架（对照表 + 自动结论）抽象成共享 helper，供所有 `--regime-gate` / `--ic-mode` 实验复用。
- 维度红线写入 `docs/classic_strategy_demystify_checklist.md`（或新建 `docs/regime_gate_policy.md`）：**波动率维度门控禁用于均值回归；新门控并入前必须隔离 A/B 且正贡献。**
- 同步进 `bilibili-critical-summary` SKILL.md §5：悦悦笔记"环境判断器"的量化落地结论 = 波动率维度对均值回归负贡献、价格/广度维度有效。

## 6. 一句话结论

**不照搬他的波动率门控（对价值/红利是毒药），只借鉴他的"隔离 A/B + 维度分清"方法论；我们平台已有有效的价格/广度 overlay，明天只需收尾工具② + 把门控维度纪律固化进 `--regime-gate` 开关（volatility 维度对均值回归显式禁用）。**

## 7. 明天执行顺序（checklist）
1. `run_activity_tax_check.py`：fetch_close 改批量查询 → 跑通 8 策略 → 出四列表。
2. `run_monthly_rebalance.py`：加 `apply_regime_gate` + `--regime-gate` CLI（breadth/price/None，volatility 拒绝）。
3. 价值/红利类接 `--regime-gate breadth` 跑 A/B（预期中性/小幅下行保护，不得宣称增益）。
4. 抽象 A/B 报告 helper；写 `regime_gate_policy.md` 维度红线；SKILL.md §5 补结论。
5. 统一 commit（不 push）。
