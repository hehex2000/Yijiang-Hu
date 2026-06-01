# 修复报告：多因子选股平台数据源配置

## 问题根源

**`config.py` 中的 `primary_source` 被错误设置为 `"akshare"`**，导致所有数据获取方法都优先使用 AkShare API（网络连接不稳定，容易失败），而不是本地数据库（稳定、快速）。

## 已完成修复

### 1. 修改 `config.py`

**文件**: `C:\Users\99395\WorkBuddy\multi_factor_selection\config.py`

**修改内容**:

| 配置项 | 修改前 | 修改后 | 说明 |
|--------|--------|--------|------|
| `primary_source` | `"akshare"` | `"local_db"` | **关键修复**：优先使用本地数据库 |
| `use_tushare_backup` | `False` | `True` | 启用 Tushare 作为备用数据源（付费账户，数据更全） |
| `use_akshare_backup` | `True` | `True` | 保持 AkShare 作为备用数据源 |

**修改原因**:
- 你的本地数据库 `D:\tu-sharedata\astock_daily.db` 有 **6.5GB**，包含几乎所有需要的数据（`daily`, `daily_basic`, `fina_indicator`, `income`, `balancesheet`, `cashflow` 等表）
- 优先使用本地数据库可以避免网络连接问题（你遇到的 `Connection aborted` 错误）
- Tushare 是付费账户，数据质量更高，应该作为备用数据源

### 2. 修改 `data_fetcher.py` 的 `get_stock_info()` 方法

**文件**: `C:\Users\99395\WorkBuddy\multi_factor_selection\src\data_fetcher.py`

**修改内容**:
- **优先从本地数据库 `stock_basic` 表读取股票信息**（名称、市值等）
- 如果本地数据库没有，再尝试 Tushare 缓存
- 如果 Tushare 缓存没有，再尝试 AkShare 单只查询
- 如果所有数据源都失败，返回空信息（不中断流程）

**修改原因**:
- 原代码优先使用 Tushare 缓存和 AkShare 查询，完全没有使用本地数据库
- 本地数据库 `stock_basic` 表应该包含股票基本信息（名称、市值等），应该优先使用

## 数据获取方法优先级

修改后，所有数据获取方法都会**优先使用本地数据库**，只有本地数据库没有数据时，才会尝试网络数据源：

| 方法 | 本地数据库表 | 备用数据源 |
|------|--------------|------------|
| `get_stock_info()` | `stock_basic` | Tushare 缓存 → AkShare |
| `get_stock_history()` | `daily` | AkShare → Tushare |
| `get_financial_data()` | `fina_indicator` | Tushare → AkShare |
| `get_valuation_data()` | `daily_basic` | Tushare → AkShare |
| `get_industry_momentum_factor()` | `industry_momentum` | 无（本地数据库必须有数据） |

## 下一步验证

### 1. 运行测试脚本

```bash
cd C:\Users\99395\WorkBuddy\multi_factor_selection
python test_local_db.py
```

**预期结果**:
- 所有测试方法都从本地数据库读取数据（日志显示 `✓ Got ... from local DB`）
- 没有网络连接错误

### 2. 运行完整回测

```bash
cd C:\Users\99395\WorkBuddy\multi_factor_selection
run_backtest.bat
```

**预期结果**:
- 选股阶段不再有 `Connection aborted` 错误
- 因子计算速度明显加快（本地数据库读取速度 >> 网络API调用）
- 回测结果合理（不再有 +2160523% 这样的荒谬收益率）

## 注意事项

1. **本地数据库必须包含最新数据**：
   - 如果你的本地数据库数据不是最新的（比如缺少最近一个月的数据），那么因子计算可能会失败
   - 建议定期运行 `Tushare-Downloader` 项目更新本地数据库

2. **Tushare 付费账户的使用**：
   - 你已经配置了 `tushare_token`，并且 `use_tushare_backup = True`
   - 当本地数据库没有数据时，会优先使用 Tushare（付费账户，数据更全）
   - AkShare 作为最后备用（免费，但可能有限流）

3. **如果还有 `Connection aborted` 错误**：
   - 说明某些数据本地数据库没有，正在尝试网络数据源
   - 可以检查本地数据库的表，看看是否缺少某些数据
   - 或者临时禁用网络数据源（`use_akshare_backup = False`, `use_tushare_backup = False`），只允许从本地数据库读取

## 总结

| 修复项 | 状态 |
|--------|------|
| 修改 `config.py` 的 `primary_source` | ✅ 已完成 |
| 修改 `config.py` 的 `use_tushare_backup` | ✅ 已完成 |
| 修改 `get_stock_info()` 优先使用本地数据库 | ✅ 已完成 |
| 验证 `get_stock_history()` 优先使用本地数据库 | ✅ 已确认（代码已实现） |
| 验证 `get_financial_data()` 优先使用本地数据库 | ✅ 已确认（代码已实现） |
| 验证 `get_valuation_data()` 优先使用本地数据库 | ✅ 已确认（代码已实现） |
| 验证 `get_industry_momentum_factor()` 优先使用本地数据库 | ✅ 已确认（代码已实现） |
| 创建测试脚本 `test_local_db.py` | ✅ 已完成 |
| 创建修复报告 `FIX_REPORT_DATA_SOURCE.md` | ✅ 已完成（本文件） |

**下一步**: 运行 `test_local_db.py` 验证修复是否成功！
