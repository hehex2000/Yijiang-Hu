# 交接文档：weekly_highdiv_vol 修复重跑（2026-09-02 收尾，明日续）

> 状态：**主网格重跑已全部完成**，还差稳健性窗口核验、活跃税重算、sensitivity 对照表、README 四步收尾。

---

## 1. 本轮干了什么（已闭环）

### 1.1 根因（比 9/2 上午结论更深一层）
- 现象：`n10` 主窗口引擎净值 5.5 年单调阴跌 100000→47202（−52.8%），n15 同期 +214%。
- **真根因：`run_weekly_highdiv_vol.py` 的前复权公式写反了**：
  - 原公式（错）：`qfq = raw × fac_latest / fac_t`
  - 正确公式：`qfq = raw × fac_t / fac_latest`（tushare adj_factor 随时间增长）
  - 反写导致**每个除权日把分红再扣一遍**（双重惩罚），高股息组合被分红拖死。
  - 9/2 上午"trades 存 hfq、引擎按 raw 估值"的结论不完整——引擎内部其实自洽，只是自洽在错误尺度里。
- 自证三点：600519 2016 raw 210.02→qfq 166.56 ✓；锚定日 qfq==raw ✓；000012.SZ 20240716 除权日 raw −5.78%→qfq −1.01%（分红正确加回）✓。

### 1.2 修复与提速（均通过行为自证）
- 修复：`qfq_close`/`qfq_open` 各改 1 行。
- 提速：逐日行情预加载（`_DAILY_CACHE`）+ 财务映射内存化（`_load_fi`/`_load_bs`），33 分钟→约 10 分钟/次（6 并发下约 20 分钟）。
- 行为自证：缓存版 vs 逐笔 SQL 版净值 CSV **diff 为空**；财务映射 6 个日期全部一致。
- 顺带修复 bug：`_report` 文件名用了模块常量 `TOP_N`/`DIV_PCT` 而非 CLI 实参 → n5/n15/d10/d40 产物全部写进 `n10_d25` 文件名互相覆盖。已参数化修复并重跑 5 个受影响任务。

### 1.3 重跑结果（qfq 一致口径，20210104-20260710，全部 ✅ 已落位）

| 变体 | 总收益 | 年化 | 最大回撤 | Sharpe | 交易次数 |
|---|---|---|---|---|---|
| **n10 d25（基准）** | **+195.87%** | **+22.70%** | **−10.55%** | **2.09** | 1548 |
| n5 d25 | +213.41% | +24.04% | −12.76% | 1.72 | 788 |
| n15 d25 | +192.88% | +22.47% | −9.02% | 2.33 | 2310 |
| n10 d10（股息前10%） | +272.40% | +28.15% | −8.96% | 2.62 | 1608 |
| n10 d40（股息前40%） | +272.27% | +28.14% | −10.02% | 2.32 | 1606 |
| n10 zero-fee（无成本） | +291.53% | +29.36% | −9.60% | 2.65 | — |

- **成本吃掉约 96pp 总收益**（291.53→195.87）——周度换手摩擦巨大，与活跃税主线互相印证。
- 对比 7/14 旧 raw 档 sensitivity（n10 基准 +267%/mdd −3.26%）：qfq 口径下收益略低但同量级，回撤 −10.55% vs −3.26% 差异待明日核对（旧档 mdd 可疑地低）。
- 伪影版 −52.8% 已被推翻，永不引用。

### 1.4 稳健性窗口（第二波 w7-w12，12/12 已跑完）
- 窗口：2020-2026 / 2021 变体起点 / 2022 / 2023 / 2018-2022 / 2026H1，均为 n10_d25 参数。
- 产物文件名不冲突（窗口不同），**但日志汇总指标尚未提取核对**（`/tmp/w7.log`~`/tmp/w12.log`，/tmp 重启会丢，见 §3 行动 1）。

---

## 2. 未完成事项（明日按序做）

1. **提取 w7-w12 汇总指标**（趁 /tmp 日志还在；丢了就用产物 CSV 重算）：
   `grep -E "总收益率|年化收益|最大回撤|夏普" /tmp/w{7..12}.log`
2. **活跃税重算**：对 6 个新 qfq trades 文件跑
   `run_activity_tax_check.py --scan data/results/weekly_highdiv_vol --glob "trades_n*_20210104_20260710.csv" --out data/results/weekly_highdiv_vol/activity_tax_20260902_v2.csv`
   → 回答"全平台 10/115 超 B&O 6.5%"在修复口径后还剩几个（round_trip_cost 0.15~0.23% 单边的方向性结论预计不变）。
3. **重写 sensitivity.csv**：修复口径 6 变体 + 6 稳健窗口 + 7/14 旧 raw 档三源对照；核对旧档 mdd −3.26% 是否可信。
4. **写/更新 README.md**：口径红线（qfq 公式方向）+ 本次修复史 + 新对照表；参照 `daily20_divlow/README.md` 格式。
5. **可选项（需拍板）**：`rb0`/`rb0_lag5/lag10` 变体（关闭 20 日黑名单开关）在恢复版脚本里不存在，未擅自加参；若要复现需先加 `--no-blacklist` 类开关。
6. **清理**：`data/results/_bak_whv_20260902/`（42 文件旧档）确认无用后删；/tmp 的 r*.log w*.log 备份关键行。

## 3. 环境与红线（明日必读）

- venv：`./venv_ml/Scripts/python.exe`；DB：`D:\tu-shareData\astock_daily.db`（7.6GB）。
- 重跑必带 `PYTHONHASHSEED=0`（`pending_sell` 是 set，否则 trades 行序不可复现；只影响行序不影响 NAV——已实测净值 CSV 逐行一致）。
- 🔴 **qfq 公式方向红线**：`raw × fac_t / fac_latest`。凡"红利策略在含除权引擎下单调阴跌"先查公式方向。
- 🔴 **cl_args 必须进产物文件名**：本次 n5/n15/d10/d40 被覆盖的教训（`_report` 已参数化，新脚本注意同类问题）。
- 并发注意：另一 agent 在往 DB 并发补数据（今日股票池 300→303 只退市股变动），**md5 对不上先怀疑数据变了，不是代码 bug**。
- Git 约束：另一 agent 在 commit/push，**零 git 写操作**，等用户通知。脚本已恢复在工作区，勿 commit。

## 4. 关键文件索引

| 文件 | 说明 |
|---|---|
| `run_weekly_highdiv_vol.py` | 修复版脚本（qfq 正向 + 缓存提速 + 文件名参数化），694→~720 行 |
| `data/results/weekly_highdiv_vol/backtest_*.csv` | 6 个主网格产物（17:45-17:47 新鲜落位）+ zero + 6 个稳健窗口 |
| `data/results/_bak_whv_20260902/` | 重跑前全目录备份（42 文件，含 7/14 raw 档与 7/16 伪影档） |
| `data/results/weekly_highdiv_vol/activity_tax_diag_20260902.csv` | 修复前 15 文件口径诊断（伪影证据，留档） |
| 今日日志 `2026-09-02.md` §③④⑤ | 完整证据链与三个插曲（ann_date REAL 亲和、set 遍历序、并发补数据） |

---

## 追加存档（17:57，最后状态）

### 今日已完成（交接文档 §2 行动 1-3）
1. ✅ **稳健窗口 w7-w12 全部提取**（日志已备份 `_bak_whv_20260902/w7~w12.log`）：全窗口 Sharpe 1.76~3.40、年化 18.75%~35.40%，全口径稳健；w8 与 w1 仅起点差一天（+197.38% vs +195.87%），一致性极好。
2. ✅ **sensitivity.csv 已重写**（三源对照：qfq 新档 / 7-16 伪影档 / 7-14 旧 raw 档；旧 raw 档 mdd −3.26% 因原文件被覆盖无法逐位复核，表中已标注）。
3. ⚠️ **活跃税重算跑了但结果失真，明天第一件事处理**：`activity_tax_20260902_v2.csv` 中新 qfq trades 重建值与引擎对不上（n10 重建 mdd −68.2% vs 引擎 −10.55%；zero 档重建 +190.3% vs 引擎 +291.5%；2026H1 重建 −56.9% vs 引擎 +11.4%）。
   **根因方向（已定位未修）**：新 qfq 价 = raw×fac_t/fac_latest，既不等于 raw 也不等于 hfq 锚定版；`nav_recon_util.reconstruct` 的现金流（trades price）与估值（closes）必须同口径。**修复方案：给 nav_recon_util 加第三种口径 `qfq_engine`（closes 也用 raw×fac_t/fac_latest 同公式），或让 run_activity_tax_check 检测到引擎内 qfq 时直接跳过并标注"以引擎 backtest CSV 为准"**。
   两个入口文件当前状态：`nav_recon_util.py`（有 detect_price_mode，缺 qfq_engine 分支）、`run_activity_tax_check.py`（hobo 按文件选 raw/hfq closes）。

### 明日行动清单（更新版）
1. 修活跃税工具的 qfq 口径（方案见上，二选一），重跑 `activity_tax_check --scan data/results/weekly_highdiv_vol` → 验证"10/115 超 B&O 6.5%"还剩几个。
2. 写 `weekly_highdiv_vol/README.md`（含 qfq 修复全过程、四 bug 链、三源对照表、成本吃 ~96pp 结论）。
3. rb0 变体需 `--no-blacklist` 开关，由用户拍板是否加。
4. 全部完成后清理 `_bak_whv_20260902/`（先确认 README 与 sensitivity 落定）。

### 今日四 bug 链（全部已修，均在 run_weekly_highdiv_vol.py）
① qfq 公式反写（raw×ref/fac_t → raw×fac_t/ref，分红双重惩罚是 −52.8% 伪影根因）；② 逐笔 SQL 慢 → 内存缓存（33min→10min，净值 diff 为空行为中性）；③ 财务映射内存化（ann_date REAL 亲和虚惊，无前视）；④ `_report` 文件名用模块常量而非 CLI 实参 → n5/n15/d10/d40 互相覆盖（已参数化修复并重跑落位）。

### 纪律状态
- 零 git 写操作；脚本与产物改动全部在工作区未 commit，等用户通知。
- 另一 agent 正在并发改 DB（退市股 300→303 只）与其它策略文件，无冲突。
