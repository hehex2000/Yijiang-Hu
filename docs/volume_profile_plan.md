# Volume Profile（密集成交区）分析模块 · 开发计划

> 来源：B站 Kara说量化《Python自动扫描5000只股票，一键找出支撑位压力位》BV1tgVs6SE4C
> 落档：2026-08-27
> 状态：**计划文档，未实施**（按"只写计划、其他别做"）

---

## 0. 一句话定位（结论先行）

视频的"核心算法"是金融里成熟的 **Volume Profile（密集成交区，VPVR）**——K线高低价区间分箱 + 成交量加权 + scipy 平滑 + `find_peaks` 找峰值，标准组件、~30 行能写完。**不是新发明，Kara 只是封装成 Streamlit 仪表盘。**

本计划目标：基于我们**本地日线库 + tushare 付费补全**，自建 Volume Profile 算法层与**因子化接口**，先走单因子验证（复用平台纪律），再按需做可视化。**诚实红线**：日线级 VPVR 精度有限 + 回测必须防前视 + 必须经四问自检；把它定位为"识别关键价位的描述/候选因子工具"，**不预设为已验证 alpha**。

---

## 1. 视频精华提炼 vs 我们取舍

| 视频宣称 | 实质 | 我们对应做法 |
|---|---|---|
| 6 Tab 仪表盘（概览/决策支持/分析图表/市场扫描/板块热力/交易记录） | Streamlit 展示层 | **末位做（P4）**，先算法+因子 |
| 三层架构（数据/算法/应用） | 标准分层 | 对齐，算法层独立成模块 |
| 算法四步：分箱→加权→平滑→检测 | VPVR 标准流程 | 直接复用（numpy/scipy） |
| 防前视三道防线（只用当日前数据/次日开盘成交/T+1） | 回测纪律 | **直接对齐平台既有纪律** |
| 历史胜率追踪 + 守住率星级 + 回测置信度（高/中/低） | 描述性统计 | 保留概念，但**必须标样本量置信度**（见 §6） |
| "5000 只 60 秒扫描" | 批量能力 | 我们全 A 日线已有，扫描可行 |
| baostock 免费数据源 | 数据获取 | **不用**，我们用本地库 + tushare |

**批判去魅点（务必写进 §7 去魅报告）**：
1. 封装成熟技术，价值在"我们自己可复现 + 因子化"，不在"秘方"。
2. 视频暗示"精确支撑压力位"——但**日线高低价区间很宽**，VPVR 真精度来自日内/小时数据；我们只有日线，必须声明是**日线级近似**，峰值偏软、箱宽敏感。
3. 单只股票某区域的"历史挑战次数"可能 <10 次，"守住率星级"置信度低，不能当统计结论。

---

## 2. 数据源映射（our situation）

| 项 | 现状 | 计划用法 |
|---|---|---|
| 本地行情库 | `config.DATA.local_db_path = D:\tu-shareData\astock_daily.db`，日线 OHLCV，`primary_source=local_db` | 主数据源：`get_conn()` 读日线 |
| tushare 付费 | `config.DATA.tushare_token` 已配 | 补全缺口（adj_factor 2015 前、成分股快照等） |
| 行业分类 | `sw_industry_daily`（31 申万一级，1999→，已验证 OHLC 完整） | 板块热力 Tab 直接复用 |
| 复权口径 | 平台双轨 NAV 修复中 | **Volume Profile 用后复权价**（避免分红跳空造假密集成交区），明确声明 |
| baostock | 视频用 | **不用** |

读取接口复用平台共享引擎（`run_monthly_rebalance.py` 导出的 `get_conn` 等），不另起炉灶。

---

## 3. 核心算法（Volume Profile 四步）

输入：单只股票 trailing `N` 日日线 `[low, high, close, vol]`（后复权）。
输出：峰值区列表，每区 `{center, lo, hi, strength, dist_pct}`。

1. **分箱（binning）**：价格轴按 `box_width`（百分比或固定 N 箱）切分。
   - **正确做法**：把每日 `vol` **均匀分摊到 [low, high] 覆盖的所有箱**（range-weighted），而非只堆在 close。这是 VPVR 与"close 直方图"的关键区别，必须在实现里写明。
2. **加权（weighting）**：箱内累计成交量即权重（上一步已含）。
3. **平滑（smoothing）**：`scipy.ndimage.gaussian_filter1d(profile, sigma=σ)` 抑制单箱噪声。
4. **检测（detection）**：`scipy.signal.find_peaks(profile, prominence=..., width=...)` → 峰值即密集成交区中心；`lo/hi` 取峰值两侧落到 `center±k*σ` 或 prominence 谷；`strength = 峰值累计量 / 总量`；`dist_pct = (price - center)/price`。

**关键参数（留待 walk-forward 调，不拍脑袋定）**：`box_width`、`σ`、`prominence`、`width`、`lookback N`（滚动 VP vs 全历史 VP）。

---

## 4. 模块架构（三层，对齐平台）

```
vp_data.py          # 数据层：get_vp_bars(ts_code, start, end) 读本地库 + tushare 补全
volume_profile.py   # 算法层：compute_volume_profile() / detect_zones() / volume_profile_features()
vp_factor.py        # 因子接口：把 features 注册进平台单因子框架（见 §5 P1）
vp_scan.py          # 应用层：全 A 批量算最近密集成交区（市场扫描）
vp_sector.py        # 应用层：sw_industry_daily 聚合（板块热力）
vp_dashboard.py     # 应用层（末位）：Streamlit 仪表盘（6 Tab 展示）
```

- **不新建回测引擎**：若做"支撑反弹"回测，复用平台共享引擎的 `calc_fee` / 次日开盘价执行 / T+1 约束（§6 三道防线）。
- **因子框架文件待确认**：平台记忆提及 `factor_zoo.py` / `factor_ls_backtest.py`，但当前 `multi_factor_selection` 树下未找到（仅 `macd_plugin_validate.py` 在）。**实施后第一步先确认这两文件真实路径；若不存在，新建最小单因子回测（次日开盘、宽成本、walk-forward）接入 `volume_profile_features`。**

---

## 5. 功能清单与优先级（对标 6 Tab，按平台纪律排序）

| 优先级 | 功能 | 对标 Tab | 说明 |
|---|---|---|---|
| **P0** | 算法层 `volume_profile.py` + 数据层 `vp_data.py` | — | 核心，先跑通单只 |
| **P1** | 因子化 + 单因子验证 | 交易记录/回测 | `volume_profile_features` → 平台单因子框架，walk-forward + 宽成本 |
| **P2** | 市场扫描 `vp_scan.py` | Tab4 | 全 A 批量算最近区，输出支撑/压力附近 + R 最优排序 |
| **P3** | 板块热力 `vp_sector.py` | Tab5 | 复用 `sw_industry_daily` 聚合 |
| **P4** | Streamlit 仪表盘 `vp_dashboard.py` | Tab1/2/3 | 展示层，**末位做**；6 张卡片/决策支持/8 图按需 |
| P1 伴生 | 去魅报告 `docs/volume_profile_demystify.md` | — | 四问自检 + 日线精度/置信度声明 |

**因子候选（P1 产出，供单因子验证）**：
- `VP_DIST_NEAREST` = (price − 最近区中心)/price
- `VP_NEAREST_STRENGTH` = 最近区 strength
- `VP_POS_IN_DIST` = 当前价以下累计量 / 总量（分布位置）
- `VP_ZONE_COUNT` = 近期显著区数量（结构复杂度）

---

## 6. 回测 / 验证纪律（复用平台四问自检 + 反过拟合三件套）

- **防前视三道防线（与视频一致，强制）**：①只用 `t` 日前数据算区；②交易用 `t+1` 开盘价；③T+1 约束。
- **单因子验证先行 → 消融 → 组合**（用户既有纪律）：不跳过单因子直接做策略。
- **反过拟合三件套**：walk-forward 切窗 + 宽成本建模（含 20bp 基金级对照 3bp ETF 级）+ 扩展候选池（全 A 非幸存者偏差）。
- **置信度必须标注样本量**：支撑/压力"守住率星级"若基于 <30 次历史挑战，评级强制降为"低/不可信"。
- **诚实预期**：日线 VPVR 大概率是个**弱因子/描述工具**，非强 alpha；验证未过四问前不得写入 `live_strategies/`。

---

## 7. 开发里程碑（cheapest / most-impact 优先）

| 里程碑 | 内容 | 验收标准 |
|---|---|---|
| **M1 spike** | 定位/创建 `spike_volume_profile.py`，单只股票跑通 `compute_volume_profile` 打印峰值区 | 单只输出合理区列表；防前视（只用当日前数据） |
| **M2 算法+数据层** | `volume_profile.py` + `vp_data.py`（本地库读取 + tushare 补全 + 后复权口径） | 全 A 任意标的可算；箱宽/σ/prominence/lookback 参数化 |
| **M3 单因子验证** | `volume_profile_features` 接入平台单因子框架，walk-forward + 宽成本 | 出 IC/IR/分组收益；明确"是否弱因子"结论 |
| **M4 市场扫描** | `vp_scan.py` 全 A 批量，输出支撑/压力附近 + R 最优排序 | 与 Tab4 五张卡片对齐，可落 CSV |
| **M5 板块热力** | `vp_sector.py` 复用 `sw_industry_daily` 聚合 | 行业偏多/偏空标签 + 触战比 |
| **M6 仪表盘** | `vp_dashboard.py` Streamlit 6 Tab（展示层末位） | 本地起服务可交互；非必须 |
| **M7 去魅报告** | `docs/volume_profile_demystify.md` | 四问自检 + 日线精度/置信度结论固化 |

---

## 8. 风险与开放问题

1. **日线精度**：高低价区间宽 → 箱宽敏感、峰值软；真 VPVR 需日内数据（我们暂无）。定位"日线级近似"。
2. **除权跳空**：必须用后复权价，否则分红造成假密集成交区。
3. **样本量陷阱**：单股单区挑战次数少 → 星级/守住率置信度低，强制标注。
4. **参数过拟合**：`box_width/σ/prominence/lookback` 四旋钮必须 walk-forward 隔离，不得跑后调参。
5. **因子冗余**：需与平台已有 35 因子做相关性，确认是否增量信息。
6. **spike 文件位置**：用户称 `spike_volume_profile.py` 已起头，但当前树未找到 → M1 先确认（可能在 `.openclaw` 工作区）或重建。

---

## 9. 与本次板块轮动去魅的关联

同一套**回测欺诈面自检框架**（问井/数活口/剥出身/防过拟合）直接套用：Volume Profile 若自报高胜率，先问井（是否前视/幸存者）、数活口（walk-forward 是否存续）、剥出身（是否只是描述工具）、防过拟合（参数是否运气）。结论先行、机制后述，不预设 alpha。
