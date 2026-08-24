# P0 实施计划：Piotroski F-score 质量因子（单因子 OOS 验证）

> 来源：B站视频《从沪深300筛到8只股票》批判式总结的 P0 落地项（§5.22 / bilibili-critical-summary 技能库）。
> 目标：验证 Piotroski F-score 9 项在**我们自己的 A股数据 + 严格前视防护 + 真实成本**下到底有没有 edge，再决定是否接入选股引擎。
> 原则：**先单因子验证，再入组合**（与平台方法论一致：单因子先行 → 消融式确认正贡献 → 再组合）。不重写 RS-R-S（已证伪，见 §5.17）。

---

## 0. 结论先行（一句话）

F-score 是平台**已缺失的经典质量因子**（现有 `quality_filter` 仅 4 项：roe>0&bps>0&负债率<70%&ocfps>0），学术 documented 正期望，且 PIT 机制与 OOS 范式平台已现成；**唯一硬门槛是数据补全**——本地 `fina_indicator` 缺 `roa/gross_margin/asset_turn/股本变化` 等 F-score 必需字段，须先按"缺数据优先补全"原则补表，再跑 OOS，不绕降级。

---

## 1. Piotroski F-score 9 项计算口径（经典 Piotroski 2000, JFE）

F-score = 9 个 0/1 二元指标之和（0~9，越高越"财务健康"）。分三组：

| # | 维度 | 指标 | 方向 | 平台数据来源 | 本地库现状 |
|---|---|---|---|---|---|
| 1 | 盈利 | ROA = 净利润/总资产 > 0 | +1 if >0 | `fina_indicator.roa` | ⚠️ 缺列（仅 `roe`，需补 `roa`） |
| 2 | 盈利 | 经营现金流 CFO > 0（用 `ocfps>0` 代理） | +1 if >0 | `fina_indicator.ocfps` | ✅ 已有列 |
| 3 | 盈利 | ΔROA = 本期ROA − 去年同期ROA > 0 | +1 if >0 | `fina_indicator.roa` 两期同比 | ⚠️ 依赖 roa |
| 4 | 盈利质量 | 应计利润低：CFO/总资产 > ROA（现金流不被账面利润粉饰） | +1 if 真 | `ocfps` vs `roa` | ⚠️ 依赖 roa |
| 5 | 杠杆 | 长期负债率下降（Δ长期负债/资产 < 0） | +1 if 下降 | `fina_balance.lt_borr` 或 `fina_indicator.debt_to_assets` 同比近似 | ⚠️ 用 `debt_to_assets` 同比近似（已有列） |
| 6 | 流动性 | 流动比率改善（Δcurrent_ratio > 0） | +1 if 上升 | `fina_indicator.current_ratio` 两期同比 | ✅ 已有列 |
| 7 | 稀释 | 当年无增发（总股本未显著增加） | +1 if 未增 | `daily_basic.total_share` 或 `fina_balance.total_share` 同比 | ⚠️ 需补股本同比（无现成列） |
| 8 | 运营效率 | 毛利率改善（Δgross_margin > 0） | +1 if 上升 | `fina_indicator.gross_margin` 两期同比 | ⚠️ 缺列（需补 `gross_margin`） |
| 9 | 运营效率 | 资产周转率改善（Δasset_turn > 0） | +1 if 上升 | `fina_indicator.asset_turn` 两期同比 | ⚠️ 缺列（需补 `asset_turn`） |

**计算口径约定（与平台既有质量逻辑统一）：**
- 全部用**年报口径**（`end_date LIKE '%1231'`，防季报 NULL/老数据坑，同 `value_stock_selector.py:217`）。
- 同比 = 本期年报 vs 上一期年报（如 2023-12-31 vs 2022-12-31）。
- 第 4 项（应计质量）口径：**`ocfps / (总资产每股)` 用 `ocf_to_debt` 不足时，简化为 `ocfps > eps`**（平台 `value_stock_selector.py:289` 已用 `ocfps/eps` 作盈余质量代理，可复用）。
- 第 7 项（无增发）：A股 F-score 研究常用「当年 `total_share` 同比增幅 < 阈值（如 5%）」判定未显著稀释；需补 `total_share` 时点序列（从 `daily_basic` 已有 `total_share` 取年报日即可，无需新表）。

---

## 2. 数据补全（P0 第一优先级，遵循"缺数据先补全"纪律）

**现状**：`download_financial_data_tushare.py:99` 建表仅 `(current_ratio, roe, fcff, op_yoy, eps)`；`value_stock_selector.py` 实际用到 `ocfps/debt_to_assets/ocf_to_debt/ar_turn`，证明表已被 `backfill_fina_2013_2014.py` 等 ALTER 加列。**但 F-score 必需的 `roa/gross_margin/asset_turn` 仍缺，第7项股本需从 `daily_basic` 取。**

### 2.1 补全 fina_indicator 扩展字段
- Tushare `fina_indicator` API 支持字段：`roa, roa_yoy, gross_margin, gross_margin_yoy, asset_turn`（及既有 `ocfps, debt_to_assets, ocf_to_debt, current_ratio, eps`）。
- 步骤：
  1. `ALTER TABLE fina_indicator ADD COLUMN roa REAL; ADD COLUMN gross_margin REAL; ADD COLUMN asset_turn REAL;`（`backfill_fina_2013_2014.py:55` 的 `get_table_cols` 动态对齐模式可直接复用写回填脚本）。
  2. 新建 `backfill_fina_fscore.py`：复用 `backfill_fina_2013_2014.py` 的 `get_table_cols/upsert` 逻辑，按 `ts_pro.fina_indicator(fields="...,roa,gross_margin,asset_turn")` 回填全市场历史（2010 起）。
  3. 校验填充率（参考 `src/small_cap_rotation_selector.py:198` 的 90%~100% 填充率口径）。
- 第 7 项（无增发）：**不建新表**，直接从 `daily_basic.total_share` 按年报日 `end_date` 取时点值做同比（PIT 已在 `daily_basic` 层面实现）。

### 2.2 补全后校验
- 跑 `SELECT COUNT(*) FROM fina_indicator WHERE roa IS NOT NULL` 确认填充率；
- 抽 3 只票手工核对 9 项得分（视频第5步建议，也是平台"抽历史调仓日手工核对"习惯）。

---

## 3. Point-in-Time 处理（严格防前视，复用平台现成机制）

**核心原则**：调仓日 `t` 只能看到 `ann_date < t` 的财报（盘后公告不能当日用，同 `run_kara_factors.py:212` P1-5 严格 `<`）。

### 3.1 复用设施
- `mine_kara_factors.py:107` `build_pit_map(sql, valcol, denom=None)`：返回 `{ts_code: (ann_dates[], vals[])}`，已按 `ann_date` 排序。
- `run_kara_factors.py:213` `pit_get(m, code, t)`：`bisect_left(anns, t)-1` 取 `< t` 最近一期 → **现成 PIT 取值器，F-score 9 项逐字段调用即可**。
- ST 过滤：`run_kara_factors.py:148` `load_st_intervals`（namechange 落库 + 时点 ST 判定）直接复用，杜绝"当前名含 ST"前视。
- IPO 过滤：`run_kara_factors.py:206` `list_tidx` 上市天数门槛，复用。

### 3.2 F-score 的 PIT 取值
- 对 9 项所需字段各建一个 PIT map（`roa`, `ocfps`, `current_ratio`, `debt_to_assets`, `gross_margin`, `asset_turn`, `eps`, `total_share`）。
- 同比项（3/5/6/8/9）：取 `pit_get` 当前期 + 前一期（同一 map 中 `ann_dates` 往前再退一格），两期都非空才判分，否则该指标记 0（保守，不夸大）。
- 年报对齐：PIT map 查询时 `WHERE end_date LIKE '%1231'` 限制年报，避免混合季报。

---

## 4. 单因子 OOS 测试脚本骨架（run_kara_factors.py 范式）

**新建 `run_piotroski_oos.py`**，骨架（复用既有范式，不重复造轮子）：

```
# 1. 复用 build_pit_map / pit_get / ST过滤 / IPO过滤（from run_kara_factors import 或复制）
# 2. 逐月（rebal 为月度调仓日序列）：
#    for t in rebal[:-1]:
#        elig = [上市≥N日 且 非ST 且 市值≥阈值]
#        fscore = {c: compute_fscore(c, t) for c in elig}   # 调 3 节 PIT 取值
#        fwd_ret[t] = {c: 前向1月复权收益 (非ffill, 退市→NaN剔除)}  # 同 run_kara_factors.py:254
#        comp_dates.append(t); store[t] = fscore
# 3. calc_ic(fscore_store, fwd_ret):  # 同 run_kara_factors.py:292
#        逐月截面 rank IC = corr(fscore, fwd_ret, method='spearman')
#        ICIR = mean(IC)/std(IC); IC>0 占比
# 4. TopN 多空：每月按 fscore 降序取前 TOPN(多) vs 后 TOPN(空)，等权，
#    真实成本（佣金万2.5/印花万5/过户万0.1/滑点10bp，同 run_monthly_rebalance）。
# 5. walk-forward 分训练/测试（平台约定 train_end=20221231 / test_start=20230101，
#    同 factor_eval 配置）：训练段做参数敏感性（阈值 7/8、年报 vs 季报），
#    测试段报净收益/夏普/最大回撤，不掺训练信息。
```

**OOS 输出指标**（对齐平台报告模板）：
| 指标 | 含义 | 通过线（参考） |
|---|---|---|
| Rank IC | 因子与次月收益的截面相关 | > 0.03 且显著 |
| ICIR | IC 信息比（稳定性） | > 0.5 |
| IC>0 占比 | 月度胜率 | > 55% |
| TopN 多空净收益 | 前N − 后N 等权 | 跑赢中证800 |
| 参数敏感性 | 阈值 7/8、年报/季报切换 | 方向不反转 |

---

## 5. 接入选股引擎设计（OOS 通过后再做，不在 P0 范围）

仅当 OOS 显示 F-score 在 A股有显著正 edge 时，再接 `run_monthly_rebalance.py`：
- `select_stocks` / `select_by_method` 加 `--piotroski-gate`（默认关）：`F >= 7`（视频口径）或 `F >= 8`（更严）。
- 作为**质量增强层叠加在现有 value/pure_bm 选股之后**（先价值初筛 → 再 F-score 门槛 → 按 BM 排序取 top_n），不替代现有 4 项 `quality_filter`。
- 接入时复用 `select_value_stocks` 已有的 fina_indicator 查询块（`value_stock_selector.py:191-240`），在同一次 SQL 里取 F-score 9 项，避免重复查库。

---

## 6. 验收标准（P0 完成定义）

1. ✅ `fina_indicator` 补全 `roa/gross_margin/asset_turn` 且填充率 ≥ 90%（抽 3 票手工核对）。
2. ✅ `run_piotroski_oos.py` 跑通 2013(或2016起)~2026，输出 Rank IC / ICIR / IC>0占比 / TopN 多空净收益。
3. ✅ 训练段(≤20221231) + 测试段(≥20230101) 分别报告，测试段 F-score 多空净收益为正且跑赢基准。
4. ✅ 参数敏感性：阈值 7 vs 8、年报 vs 季报，方向一致不反转。
5. ✅ 无未来函数：ST 过滤 + `ann_date < t` 严格 PIT（同 run_kara_factors P1-5）审计通过。

---

## 7. 风险与坑（来自视频自检清单 + 平台教训）

- **未来函数**：财报必须 `ann_date < t`（盘后公告当日不可用）。视频重点警示，平台 PIT 机制已覆盖，但 F-score 新接入时务必复用 `pit_get` 不手写。
- **排序方向**：F-score 高=好，但 A股"高质量"可能已被定价（quality minus junk 因子拥挤）。OOS 用多空双向验证，不只看多头。
- **小样本/生存偏差**：用 `index_constituent` 时点快照做候选池（同 `value_stock_selector.py:71` `_vsel_pool_ts_set`），不用当前成分股（防未来成分股泄漏）。
- **参数过度拟合**：18/600/0.7/8 这类"魔法数字"在视频里被批。F-score 阈值 7/8 是经典固定值（非调参），但年报 vs 季报、ROA 代理口径需做敏感性，不盲信单点。
- **集中风险**：视频"8只等权=单只12.5%"被批。即使 OOS 通过，接入选股时 F-score 仅作门槛，最终持仓仍由 `top_n`(约5) + 等权控制，不主动缩到 8 只。

---

## 8. 任务拆解（按 cheapest/most-impact 优先）

| 顺序 | 任务 | 依赖 | 成本 | 影响 |
|---|---|---|---|---|
| 1 | `backfill_fina_fscore.py` 补 `roa/gross_margin/asset_turn` 列 + 回填 | 无 | 中（全市场下载慢） | 🔴 阻塞项 |
| 2 | `compute_fscore(code, t)` 函数（9项 PIT 取值 + 同比） | 1 | 低 | 核心 |
| 3 | `run_piotroski_oos.py` 骨架（复用 build_pit_map/calc_ic/ST过滤） | 2 | 低 | 核心 |
| 4 | OOS 跑通 + 验收指标（IC/ICIR/TopN多空 + walk-forward） | 3 | 低 | 决策依据 |
| 5 | （可选）接入 `select_by_method --piotroski-gate` | 4 通过 | 中 | 落地 |

> P0 只做 1–4；5 待 OOS 结论再定。数据补全（1）是唯一的硬阻塞，先做。

---

## 9. 与平台既有因子的关系（避免重复）

- `src/factor_processor.py` 已有质量因子 `QF1_asset_liab_ratio/QF2_current_ratio/QF3_asset_turnover/QF4_cash_flow_quality/QF5_cash_flow_to_revenue` + 成长 `GF5_gross_margin_growth`——这些是**连续值打分**，F-score 是**离散 0/1 九维求和**，口径不同、不冲突，可并存对照。
- F-score 的独特价值：① 经典学术背书 ② 离散门槛（非连续 zscore）对"财务健康"二元判定更稳健 ③ 含 dilution(无增发) 维度，平台现有因子无此维。
- macOS 平台记忆提到 `factor_zoo.py`（29 因子），Windows 这边对应 `src/factor_processor.py` / `run_kara_factors.py`；F-score 因子建议落在 `run_piotroski_oos.py` 独立模块（与 kara 因子测试平级），不强行塞进 factor_zoo 命名。
