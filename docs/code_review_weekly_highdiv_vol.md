# 代码审查报告：`run_weekly_highdiv_vol.py`

> **审查日期**：2026-07-14  
> **审查范围**：`run_weekly_highdiv_vol.py` 全文 + 复用引擎层（`run_monthly_rebalance.py`）+ 数据库实查  
> **回测区间**：2021-01-04 ~ 2026-07-10  

---

## 审查结论概览

| 编号 | 严重等级 | 问题 | 预期影响 |
|:---:|:---:|---|---|
| #1 | 🔴 致命 | 财务数据 100% 前视偏差（MLEV 因子用未来数据） | 回测结果可能完全失效 |
| #2 | 🔴 致命 | 幸存者偏差 — 256 只退市股被静默剔除 | 系统性高估收益 |
| #3 | 🟡 中等 | 最终资产未扣除清仓手续费 | 收益率被轻微高估 |
| #4 | 🟡 中等 | 首日选股用了当日数据（前视） | 影响有限（仅首日一笔） |
| #5 | 🟢 低 | 跳过股票的现金不重新分配（现金拖累） | 设计选择，非 bug |
| #6 | 🟢 低 | 死代码（L676） | 无影响 |

---

## 🔴 致命漏洞 #1：财务数据 100% 前视偏差

### 问题定位

- `_quarter_end_str()`（L243-267）
- `_liab_map()`（L191-209）
- `_fina_debt_map()`（L212-230）
- `get_leverage_value()`（L280-283）

以上四处均用 `end_date <= 季度末` 来过滤财报，但 `end_date` 是**报告期截止日**，不是**公告日**。`balance_sheet` 表里有 `ann_date` 列，代码完全没用。

### 代码片段

```python
# _liab_map() L198-204（当前写法 — 前视）
df = pd.read_sql_query(
    "WITH ranked AS ("
    "  SELECT ts_code, total_liab,"
    "         ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY end_date DESC) rn"
    "  FROM balance_sheet WHERE end_date <= ?)"          # ← end_date 是报告期，不是公告日
    "SELECT ts_code, total_liab FROM ranked WHERE rn = 1",
    conn, params=(qe,))
```

### 数据库实查验证

在 `D:\tu-shareData\astock_daily.db` 上执行验证：

| 调仓日 | `_quarter_end_str` 返回 | 查询使用 | 该季报尚未公告的比例 |
|---|---|---|---|
| 2021-07-15 | 20210630 | `end_date <= 20210630` | **100%**（4811/4812） |
| 2021-01-15 | 20201231 | `end_date <= 20201231` | **100%**（10244/10244） |

各报告期的公告延迟分布：

```
end_date      total  ann>30d  ann>60d  ann>90d
20251231       5611     5611     5611     5611
20250930       5922     5922     5922     5800
20250630       5946     5946     5946     5944
20250331       6445     6445     6445     6124
20241231       5876     5876     5876     5876
20240930       5554     5554     5554     5440
20240630       5435     5435     5435     5432
20240331       6029     6029     6029     5662
...
```

**几乎所有财报的公告日都比报告期截止日晚 30 天以上。**

### 全年影响时间线

| 月份 | 使用的季报 | 法定公告截止日 | 前视天数 |
|---|---|---|---|
| 1-3月 | Q4（上年年报） | 4月30日 | **最多 120 天** |
| 4月 | Q1（一季报） | 4月30日 | ~30 天 |
| 5-6月 | Q1 | 已公告 | ✅ 无 |
| 7-8月 | Q2（半年报） | 8月31日 | **最多 62 天** |
| 9月 | Q2 | 已公告 | ✅ 无 |
| 10月 | Q3（三季报） | 10月31日 | ~31 天 |
| 11-12月 | Q3 | 已公告 | ✅ 无 |

**一年 12 个月中有 7 个月在用未公告的财务数据。** MLEV 因子每时每刻都在"偷看未来"。

### 为什么这能造假收益

策略选"低负债"股票。用了未来财报后，它能**提前知道哪家公司即将降杠杆**（还债/资产增长），从而在公告前买入；也能**避开即将升杠杆**的公司。这种"选股 alpha"在现实中不可能存在。

### 修复方向

`_liab_map` 和 `_fina_debt_map` 的 SQL 应改为：

```python
# 修复前（前视）：
"FROM balance_sheet WHERE end_date <= ?"

# 修复后（用公告日）：
"FROM balance_sheet WHERE ann_date <= ? AND ann_date IS NOT NULL AND ann_date != ''"
```

同理 `get_leverage_value()`（L280-283）也要改。`_quarter_end_str` 函数可以删除——不再需要猜季度末，直接用 `ann_date <= trade_date` 即可。

---

## 🔴 致命漏洞 #2：幸存者偏差 — 退市股被静默剔除

### 问题定位

L346-370 的选股流程：

```python
# L346-349: 从 daily 取当日交易的股票（时点存在性 — 这步没问题）
rows = pd.read_sql_query(
    "SELECT DISTINCT ts_code FROM daily WHERE trade_date = ?", conn,
    params=(rebalance_date,))
trading = set(rows["ts_code"].tolist())

# L353-355: 用 stock_basic 过滤
for c in trading:
    info = basic.get(c)       # ← 退市股不在 stock_basic 里
    if info is None:          # ← None
        continue              # ← 静默跳过，退市股被剔除
```

### 数据库实查验证

| 统计项 | 数值 |
|---|---|
| `daily` 表中的 distinct ts_code | 5,800 |
| `stock_basic` 表中的行数 | 5,534 |
| 在 `daily` 有数据但不在 `stock_basic` 的退市股（排除 688/BJ） | **256 只** |

部分退市股示例：

| 代码 | 最后交易日 |
|---|---|
| 300344.SZ | 2026-04-21 |
| 000638.SZ | 2026-04-13 |
| 600355.SH | 2026-04-03 |
| 002231.SZ | 2026-01-29 |
| 601989.SH | 2025-08-12 |
| 300208.SZ | 2025-07-18 |

### 影响

这 256 只退市股在 `daily` 表中有历史交易数据（它们在退市前确实在交易），但因为不在 `stock_basic` 中，被 L353 的 `basic.get(c) → None → continue` 静默过滤掉了。

退市股通常是**基本面恶化、股价暴跌**的股票。将它们从候选池中剔除 = 只在"最终活下来的股票"里选股 = 系统性高估收益。

对于小市值策略（因子4），影响更大——退市股多为小盘股。

### 修复方向

从 `daily` 表或包含退市信息的扩展 `stock_basic` 构建股票信息（名称/上市日），而非依赖仅含存续股的 `stock_basic`：

```python
# 方案 A：导入时用 list_status='D,L,P,S' 下载含退市股的 stock_basic
# 方案 B：从 daily 表历史数据补充退市股的名称和上市日
```

---

## 🟡 中等漏洞 #3：最终资产未扣除清仓手续费

### 问题定位

L630-636（结束平仓）+ L676-678（绩效计算）：

```python
# L630-636: 循环结束后平仓（扣了手续费，更新了 cash）
if trade_dates:
    last = trade_dates[-1]
    for code in list(positions.keys()):
        px = qfq_close(code, last)
        if px is not None:
            do_sell(code, last, "backtest_end", price=px)  # ← cash 已更新

# L676-678: 但绩效用的是 daily_vals[-1]，这是平仓前的值！
final_value = daily_vals[-1]["value"] if daily_vals else INIT_CAPITAL
```

`daily_vals[-1]` 是主循环最后一次 `append` 的值，记录于 L603-608：

```python
# L603-608（主循环收盘记账）
total = cash
for code, pos in list(positions.items()):
    px = qfq_close(code, td)
    if px is not None:
        total += pos["shares"] * px
daily_vals.append({"date": td, "value": total})   # ← 没扣清仓手续费
```

### 对比正确实现

`run_monthly_rebalance.py`（L2082-2089）的正确做法：

```python
# 正确版：平仓后追加一行
for code in list(positions.keys()):
    p = get_price(code, trade_dates[-1])
    cash += positions[code]["shares"] * p - calc_fee('sell', p, positions[code]["shares"])
    del positions[code]
daily_vals.append({"date": trade_dates[-1], "value": cash})  # ← 平仓后追加
```

### 修复

在 L636 之后、L638 之前加一行：

```python
daily_vals.append({"date": last, "value": cash})
```

---

## 🟡 中等漏洞 #4：首日选股用了当日数据（前视）

### 问题定位

L541：

```python
prev_td = trade_dates[i - 1] if i > 0 else td
```

当 `i=0`（回测第一天）时，`prev_td = td`。如果第一天恰好是周度调仓日（如 `20210104` 是周一，必然是），则 `select_highdiv_vol_stocks(td, ...)` 用当天 `daily_basic` 数据选股，但执行用的是当天开盘价——**开盘时还没有当天的 `daily_basic`（它是收盘后的数据）**。

### 修复

```python
prev_td = trade_dates[i - 1] if i > 0 else None
if prev_td is None:
    # 第一天跳过调仓，从第二周开始
    continue
```

---

## 🟢 低风险 #5：跳过股票的现金不重新分配（现金拖累）

### 问题定位

L566-600：

```python
cash_per = cash / len(to_buy)   # ← 一次性算好每股配额
for code in to_buy:
    if code in blacklist: continue       # 跳过 → 配额浪费
    if row["vol"] == 0: continue         # 停牌 → 配额浪费
    if pct_chg <= -thr: continue         # 一字跌停 → 配额浪费
    ...
```

被跳过的股票的配额不会重新分配给剩余股票。如果 10 只候选中 3 只被跳过，30% 现金闲置。

> **性质**：设计选择，非 bug。但可考虑改为"可用现金 / 可买入股票数"动态分配。

---

## 🟢 低风险 #6：死代码

L676：

```python
final_value = trades[-1]["price"] * 0 if False else None  # ← 永远是 None，死代码
```

> **修复**：删除此行。

---

## 三个重点区域审查结论

### 1. L346 候选池取数：幸存者偏差是真的

`daily WHERE trade_date = ?` 本身没有问题（时点存在性查询），**但紧接着 L353 用 `stock_basic` 做名称/上市日过滤时，256 只退市股因不在 `stock_basic` 中被静默剔除**。这就是幸存者偏差的入口。

### 2. qfq_open / qfq_close 的 ref 基准：无造利，确认安全

```python
qfq_price(t) = raw(t) × ref / factor(t)    # ref = 最新因子
```

买卖同口径，`ref` 在收益率比值中**完全抵消**：

$$\frac{qfq(t_{sell})}{qfq(t_{buy})} = \frac{raw(t_{sell}) \,/\, factor(t_{sell})}{raw(t_{buy}) \,/\, factor(t_{buy})}$$

`ref` 不影响收益率。前复权正确地内含了除息日调整（持仓期间分红被自动再投到价格里）。**这里没有 bug。**

### 3. do_sell 黑名单与清仓逻辑：已修复，但有残留问题

- **黑名单逻辑本身正确**：`RBUY_BLACKLIST_DAYS=0` 时止盈后可再入选，不会导致长期空仓。
- **残留问题**：最终清仓手续费未计入 `daily_vals`（漏洞 #3），导致最大回撤和夏普比率的计算也略偏乐观（最后一个数据点偏高）。

---

## 修复优先级

| 优先级 | 漏洞 | 修复要点 | 预期影响 |
|:---:|:---:|---|---|
| **P0** | #1 财务前视 | `end_date <= ?` → `ann_date <= ?` | MLEV 因子选股能力可能大幅下降 |
| **P0** | #2 幸存者偏差 | 补入退市股的 stock_basic 信息 | 小市值因子 alpha 可能显著缩水 |
| **P1** | #3 清仓费 | 平仓后追加 `daily_vals` | 收益率高估约 0.1-0.2% |
| **P2** | #4 首日前视 | `i=0` 时跳过调仓 | 影响极小（1 笔交易） |
| **P3** | #5 现金拖累 | 动态重分配现金 | 可改善但不影响正确性 |
| **P3** | #6 死代码 | 删除 L676 | 无影响 |

> **关键判断**：P0 的两个漏洞修完后，如果此前的高额收益大幅缩水甚至消失，就说明之前的超额收益主要来自这两个信息泄漏。
