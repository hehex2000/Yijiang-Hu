# -*- coding: utf-8 -*-
"""
ETF轮动策略【升级版 V6_merged】 - 平台框架 + V6 独有 RSRS 质量分
===============================================================
本文件 = 平台 run_etf_rotation（架构/池/风控/成本已齐） + V6 独有 RSRS 质量分（第4评分因子）。

升级来源（对照 run_etf_rotation_v6 的过拟合审计结论，平台全面占优，仅 RSRS 为 V6 独有增量）：
  · 信号：保留平台双动量(ROC20/ROC60+波动惩罚)+MA60+追高保护，并叠加 V6 的 RSRS 质量分
          （25日归一化收盘价对时间序号一元回归 slope×R²），横截面 z 后与平台基分加权合并。
  · 风控：完整继承平台（现金缓冲+防抖+VaR缩放+折溢价闸门+货币基金避险），替换 V6 简陋的
          止损+冷却+100%满仓。
  · 成本：平台 calc_etf_fee（佣0.025%+最低5元+滑点0.1%+ETF免印花税），替换 V6 简化双档。
  · 架构：import 共享引擎 run_monthly_rebalance（get_conn/calc_fee 同源），不再是 V6 硬编码孤岛。
  · 池  ：平台 20 只诚实广覆盖池（明确不含红利），替换 V6 后视镜 4 只（MC 99 分位）。

RSRS 权重默认 0.30（平台基分占 0.70）；可用 --rsrs-weight 0 退化为纯平台、
--rsrs-weight 0.6 复刻 V6 风格，做 A/B 对照。
数据来源：etf_daily 表（真实ETF价格）。RSRS 采用原始 close（ETF 25 日复权差可忽略）。
"""

import sys, os, argparse, numpy as np, pandas as pd

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── 导入现有工具函数 ──────────────────────────────────────
from run_monthly_rebalance import get_conn, get_monthly_5th_trading_days, COMMISSION_RATE, COMMISSION_MIN, SLIPPAGE_RATE, compute_reality_discounts

# ── 折溢价过滤（可选，缺失不影响主流程）────────────────────
try:
    from backtest import etf_premium_filter as _epf
except Exception:
    _epf = None

# ── 板块三态状态机（B 项前置闸门，纯函数模块）──────────────
# sector_state_machine 为纯函数（无 DB 依赖，可单测）；DB 取数 + 闸门封装在本文件。
from sector_state_machine import (
    classify_state, STATE_CN, GATE_PASS, UNKNOWN, HIST_LEN as SSM_HIST_LEN,
)

# ── 市场择时 overlay（A 项振荡器闸门，纯函数模块）────────────
from market_timing_overlay import compute_breadth_oscillator, position_cap

# ── 常量 ──────────────────────────────────────────────────
STAMP_DUTY_RATE_ETF = 0.0       # ETF 免印花税
INITIAL_CAPITAL = 100000        # 初始总资金

# ── 手册精华参数 ──────────────────────────────────────────
SWITCH_THRESHOLD = 0.05         # 最小切换阈值：新标的领先5%以上才切换（防抖动）
CASH_BUFFER_PCT = 0.10          # 现金缓冲：保留10%现金（手册建议10-20%）
MOMENTUM_WARNING_THRESHOLD = 0.40  # 追高警告：60日涨幅>40%时警惕见顶
MONEY_FUND_CODE = "511990.SH"   # 货币基金代码（熊市避险）

# ── RSRS 质量分（V6 独有增量）─────────────────────────────
RSRS_WEIGHT = 0.30   # RSRS 在合并评分中的权重（z 单位），其余 1-RSRS_WEIGHT 给平台双动量基分
RSRS_WINDOW = 25     # RSRS 回归窗口（天），与 V6 一致

# ── 标的池定义（真实ETF·扩大版）───────────────────────────────
#  分类：宽基 / 行业 / 跨境 / 商品债券 / 货币
ETF_UNIVERSE = [
    # ── 宽基（防守层：大盘价值/蓝筹）──────────────────────
    {"code": "510300.SH", "name": "沪深300ETF", "desc": "大盘价值",   "cat": "宽基", "layer": "防守"},
    {"code": "510050.SH", "name": "上证50ETF",  "desc": "超大盘蓝筹", "cat": "宽基", "layer": "防守"},
    {"code": "515800.SH", "name": "中证800ETF", "desc": "大中盘",     "cat": "宽基", "layer": "防守"},
    {"code": "510980.SH", "name": "上证指数ETF","desc": "上证全指",   "cat": "宽基", "layer": "防守"},
    {"code": "510500.SH", "name": "中证500ETF", "desc": "中盘成长",   "cat": "宽基", "layer": "防守"},
    # ── 宽基（进攻层：高成长/小盘）────────────────────────
    {"code": "512100.SH", "name": "中证1000ETF","desc": "小盘",       "cat": "宽基", "layer": "进攻"},
    {"code": "159915.SZ", "name": "创业板ETF",  "desc": "科技成长",   "cat": "宽基", "layer": "进攻"},
    {"code": "159949.SZ", "name": "创业板50ETF","desc": "创业板龙头", "cat": "宽基", "layer": "进攻"},
    {"code": "588000.SH", "name": "科创50ETF",  "desc": "科创板",     "cat": "宽基", "layer": "进攻"},
    # ── 行业 ──────────────────────────────────────────────
    {"code": "512480.SH", "name": "半导体ETF",  "desc": "芯片半导体", "cat": "行业", "layer": "进攻"},
    {"code": "515030.SH", "name": "新能源车ETF","desc": "新能源",     "cat": "行业", "layer": "进攻"},
    {"code": "512010.SH", "name": "医药ETF",    "desc": "医药生物",   "cat": "行业", "layer": "防守"},
    {"code": "159928.SZ", "name": "消费ETF",    "desc": "大消费",     "cat": "行业", "layer": "防守"},
    {"code": "512880.SH", "name": "证券ETF",    "desc": "券商金融",   "cat": "行业", "layer": "进攻"},
    # ── 跨境 ──────────────────────────────────────────────
    {"code": "159920.SZ", "name": "恒生ETF",    "desc": "港股宽基",   "cat": "跨境", "layer": "进攻"},
    {"code": "513100.SH", "name": "纳指ETF",    "desc": "美股科技",   "cat": "跨境", "layer": "进攻"},
    # ── 商品/债券（避险层：负相关资产）────────────────────
    {"code": "518880.SH", "name": "黄金ETF",     "desc": "商品避险",   "cat": "商品", "layer": "避险"},
    {"code": "501018.SH", "name": "原油LOF",    "desc": "商品能源",   "cat": "商品", "layer": "避险"},
    {"code": "511010.SH", "name": "国债ETF",     "desc": "利率债",     "cat": "债券", "layer": "避险"},
    # ── 货币（避险层：熊市保命）──────────────────────────
    {"code": "511990.SH", "name": "华宝添益",    "desc": "货币基金",   "cat": "货币", "layer": "避险"},
]

BENCHMARK_CODE = "510300.SH"   # 基准（沪深300ETF）
CASH_NAME = "货币基金"          # 现金避险名称

# ── 行业池（--pool industry）─────────────────────────────
# 12 只 2018 年前上市的行业旗舰 + 科技板块 2 只（2019 起上市，历史不足时自动跳过）
# 纯行业、不掺宽基/跨境——"钱在板块间搬家"的赛道（行业动量视频灵感）
INDUSTRY_UNIVERSE = [
    {"code": "512880.SH", "name": "证券ETF",     "desc": "券商",       "cat": "行业", "layer": "进攻"},
    {"code": "512660.SH", "name": "军工ETF",     "desc": "国防军工",   "cat": "行业", "layer": "进攻"},
    {"code": "512800.SH", "name": "银行ETF",     "desc": "银行",       "cat": "行业", "layer": "防守"},
    {"code": "512400.SH", "name": "有色ETF",     "desc": "有色金属",   "cat": "行业", "layer": "进攻"},
    {"code": "512200.SH", "name": "房地产ETF",   "desc": "地产",       "cat": "行业", "layer": "进攻"},
    {"code": "159928.SZ", "name": "消费ETF",     "desc": "大消费",     "cat": "行业", "layer": "防守"},
    {"code": "512010.SH", "name": "医药ETF",     "desc": "医药生物",   "cat": "行业", "layer": "防守"},
    {"code": "159930.SZ", "name": "能源ETF",     "desc": "能源",       "cat": "行业", "layer": "防守"},
    {"code": "512330.SH", "name": "信息技术ETF", "desc": "中证信息",   "cat": "行业", "layer": "进攻"},
    {"code": "512580.SH", "name": "环保ETF",     "desc": "环保",       "cat": "行业", "layer": "进攻"},
    {"code": "512070.SH", "name": "证券保险ETF", "desc": "非银金融",   "cat": "行业", "layer": "进攻"},
    {"code": "512980.SH", "name": "传媒ETF",     "desc": "传媒",       "cat": "行业", "layer": "进攻"},
    # ── 科技板块（2019 起上市，回测早期数据不足会被自动跳过）──
    {"code": "515000.SH", "name": "科技ETF",     "desc": "科技龙头",   "cat": "科技", "layer": "进攻"},
    {"code": "512480.SH", "name": "半导体ETF",   "desc": "芯片半导体", "cat": "科技", "layer": "进攻"},
    # ── 货币（避险层：全弱时转入）──
    {"code": "511990.SH", "name": "华宝添益",    "desc": "货币基金",   "cat": "货币", "layer": "避险"},
]

MIXED_UNIVERSE = ETF_UNIVERSE   # 保留原 20 只混合池引用（--pool mixed 默认）


# ── 数据表路由 ────────────────────────────────────────────

def _get_data_table(ts_code):
    """判断从哪个表读取数据

    - etf_daily: 真实ETF价格（51xxxx/58xxxx.SH, 159xxxx.SZ）
    - index_daily: 指数价格（用于非ETF代码的fallback）
    """
    if ts_code.endswith('.SH') and len(ts_code) == 9 and ts_code[:2] in ('51', '58'):
        return 'etf_daily'
    if ts_code.endswith('.SZ') and len(ts_code) == 9 and ts_code[:3] == '159':
        return 'etf_daily'
    return 'index_daily'


# ── 价格辅助函数 ──────────────────────────────────────────

def _query_price(ts_code, trade_date, field="close", fallback_field="close"):
    """从 etf_daily 或 index_daily 表查询价格

    内部函数：先用精确日期匹配，fallback到最近交易日。
    返回原始数据库中的数值（ETF实际价格 / 指数原始点数）。
    """
    table = _get_data_table(ts_code)
    conn = get_conn()

    row = pd.read_sql_query(
        f"SELECT {field} FROM {table} WHERE ts_code = ? AND trade_date = ?",
        conn, params=(ts_code, trade_date)
    )
    if len(row) > 0 and row.iloc[0][field] is not None:
        conn.close()
        return float(row.iloc[0][field])

    # fallback: 最近交易日
    row2 = pd.read_sql_query(
        f"SELECT {fallback_field} FROM {table} WHERE ts_code = ? "
        f"AND trade_date < ? ORDER BY trade_date DESC LIMIT 1",
        conn, params=(ts_code, trade_date)
    )
    conn.close()
    if len(row2) > 0 and row2.iloc[0][fallback_field] is not None:
        return float(row2.iloc[0][fallback_field])
    return None


def get_etf_price(ts_code, trade_date):
    """获取ETF收盘价（真实价格，无需缩放）"""
    return _query_price(ts_code, trade_date, field="close")


def get_etf_open(ts_code, trade_date):
    """获取ETF交易执行价格

    2026-07-06起：盘后30分钟可用收盘价定价交易（非未来函数）
    之前：正常开盘价交易
    """
    td = int(trade_date) if isinstance(trade_date, str) else trade_date
    if td >= 20260706:
        # 收盘盘后定价交易
        return _query_price(ts_code, trade_date, field="close")
    # 正常盘中交易：开盘价
    return _query_price(ts_code, trade_date, field="open", fallback_field="close")


# ── ETF 费用计算（免印花税）────────────────────────────────

def calc_etf_fee(buy_or_sell, price, shares):
    """计算ETF交易费用（免印花税）"""
    amount = price * shares
    commission = max(amount * COMMISSION_RATE, COMMISSION_MIN)
    slippage = amount * SLIPPAGE_RATE
    return commission + slippage


# ── 胜率计算 ─────────────────────────────────────────────

def calc_win_rate(trades):
    """从交易记录计算胜率（FIFO匹配买卖对）"""
    if not trades:
        return 0.0, 0, 0
    pending = {}
    win = 0
    total = 0
    for t in trades:
        code = t["code"]
        action = t["action"]
        shares = t["shares"]
        price = t["price"]
        if action.startswith("BUY"):
            if code not in pending:
                pending[code] = []
            pending[code].append({"price": price, "shares": shares})
        elif action.startswith("SELL"):
            remaining = shares
            while remaining > 0 and code in pending and pending[code]:
                first = pending[code][0]
                pnl_shares = min(first["shares"], remaining)
                pnl = (price - first["price"]) * pnl_shares
                total += 1
                if pnl > 0:
                    win += 1
                first["shares"] -= pnl_shares
                remaining -= pnl_shares
                if first["shares"] <= 0:
                    pending[code].pop(0)
    wr = (win / total * 100) if total > 0 else 0.0
    return wr, win, total


# ── 技术指标函数 ──────────────────────────────────────────

def _query_history(ts_code, trade_date, field, limit):
    """从 etf_daily 或 index_daily 查询历史序列

    返回: 降序排列的 float 列表，或 None
    """
    table = _get_data_table(ts_code)
    conn = get_conn()
    rows = pd.read_sql_query(
        f"SELECT {field} FROM {table} WHERE ts_code = ? AND trade_date <= ? "
        f"ORDER BY trade_date DESC LIMIT ?",
        conn, params=(ts_code, trade_date, limit)
    )
    conn.close()
    if len(rows) < 1:
        return None
    return [float(r) for r in rows[field].values]


def calc_roc(ts_code, trade_date, period=20):
    """ROC涨跌幅：close / close_N_days_ago - 1"""
    closes = _query_history(ts_code, trade_date, "close", period + 5)
    if closes is None or len(closes) < period + 1:
        return None
    return closes[0] / closes[period] - 1.0


def calc_ma(ts_code, trade_date, period=60):
    """计算移动平均线"""
    closes = _query_history(ts_code, trade_date, "close", period + 5)
    if closes is None or len(closes) < period:
        return None
    return np.mean(closes[:period])


def calc_rsrs(ts_code, trade_date, window=RSRS_WINDOW):
    """RSRS 质量分（V6 独有增量）：对最近 window 日收盘价 min-max 归一化，
    对时间序号 1..window 做一元回归，返回 slope×R²。
    与 run_etf_rotation_v6.rsrs_quality 同口径；采用原始 close（ETF 25 日复权差可忽略）。
    _query_history 返回降序（[0]=当日），翻转成升序（旧→新）再回归。
    """
    closes = _query_history(ts_code, trade_date, "close", window + 5)
    if closes is None or len(closes) < window:
        return None
    x = np.array(closes[:window], dtype=float)[::-1]  # 升序：旧→新
    xmin, xmax = x.min(), x.max()
    if xmax - xmin < 1e-12:
        return None
    y = (x - xmin) / (xmax - xmin)
    t = np.arange(1, len(y) + 1, dtype=float)
    b, a = np.polyfit(t, y, 1)
    yhat = a + b * t
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return float(b * r2)


def _etf_basket_var(codes, trade_date, conf=0.95, lookback=120, method="hist"):
    """等权 ETF 篮子的单日 VaR 损失比例（正数小数）；数据不足返回 None

    与 run_monthly_rebalance.estimate_basket_var 同口径，但走 etf_daily 表。
    """
    if not codes:
        return None
    series = []
    for code in codes:
        closes = _query_history(code, trade_date, "close", int(lookback) + 1)
        if closes and len(closes) >= 2:
            arr = np.array(closes[::-1], dtype=float)  # 升序
            r = np.diff(arr) / arr[:-1]
            series.append(r)
    if not series:
        return None
    n = min(len(s) for s in series)
    if n < 10:
        return None
    basket = np.mean(np.array([s[-n:] for s in series]), axis=0)
    if method == "param":
        mu, sigma = basket.mean(), basket.std(ddof=1)
        z = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}.get(conf, 1.645)
        var_ret = mu - z * sigma
    else:
        q = max(0.0, min(1.0, 1.0 - conf))
        var_ret = float(np.quantile(basket, q))
    loss = -var_ret
    return float(loss) if loss > 0 else 0.0


def _etf_var_invest_ratio(codes, trade_date, var_control, var_maxdd, var_n,
                          var_lookback=120, var_method="hist"):
    """VaR 反解投入比例（0~1）。var_control<=0 → 1.0（满仓不缩放）

    持有期VaR = 日VaR×√21（月度调仓）；预算 = 目标回撤/N；比例 = min(1, 预算/持有期VaR)
    """
    if not (var_control and var_control > 0) or not codes:
        return 1.0
    bvar = _etf_basket_var(codes, trade_date, conf=var_control / 100.0,
                           lookback=var_lookback, method=var_method)
    if not bvar or bvar <= 0:
        return 1.0
    hold_var = bvar * (21 ** 0.5)
    risk_budget = (var_maxdd / 100.0) / max(1, var_n)
    return min(1.0, risk_budget / hold_var)


def calc_volatility(ts_code, trade_date, window=20, annualized=True):
    """计算波动率

    Args:
        annualized: True=年化波动率（用于报告），False=原始日波动率（用于打分）
    打分时使用非年化波动率，与ROC收益同量级，避免惩罚项过大。
    """
    closes = _query_history(ts_code, trade_date, "close", window + 5)
    if closes is None or len(closes) < window + 1:
        return None
    closes = closes[::-1]  # 升序
    returns = np.diff(np.log(closes))
    vol = float(np.std(returns))
    return vol * np.sqrt(252) if annualized else vol


# ── 交易日期 ──────────────────────────────────────────────

def get_trade_dates(start_date, end_date):
    """获取回测区间的所有交易日（从 index_daily 表获取）"""
    conn = get_conn()
    rows = pd.read_sql_query(
        "SELECT DISTINCT trade_date FROM index_daily WHERE "
        "trade_date BETWEEN ? AND ? ORDER BY trade_date",
        conn, params=(start_date, end_date)
    )
    conn.close()
    return [int(r) for r in rows["trade_date"].values]


# ── 打分与选标逻辑 ────────────────────────────────────────

def score_assets(trade_date, method="dual", roc_period=20, ma_period=60, universe=None,
                 rsrs_weight=RSRS_WEIGHT, rsrs_window=RSRS_WINDOW):
    """对标的池中所有标的打分，返回排序后的列表

    合并评分 = (1-W)·z(平台基分) + W·z(RSRS质量分)，横截面 z 后加权。
    平台基分 = 原 dual/single/ma_filter 评分（已含 MA60 过滤 + 追高保护 + 波动惩罚）。
    RSRS 为 V6 独有增量；W 默认 0.30（可由 --rsrs-weight 调）。W=0 即纯平台。

    universe: 标的池列表（默认 ETF_UNIVERSE 混合池）
    返回:
        [ (code, name, score), ... ] 按得分降序
        或 [] 表示全部走弱应空仓
    """
    if universe is None:
        universe = ETF_UNIVERSE
    rows = []  # (code, name, base, rsrs)
    for etf in universe:
        code = etf["code"]
        # 货币基金不参与打分（作为熊市避险资产单独处理）
        if code == MONEY_FUND_CODE:
            continue
        close_price = get_etf_price(code, trade_date)
        if close_price is None:
            continue

        # ── 计算 MA60 位置 ──
        ma60 = calc_ma(code, trade_date, period=ma_period)

        # ── 计算 ROC（双动量用两个周期） ──
        if method == "single":
            roc_val = calc_roc(code, trade_date, roc_period)
            if roc_val is None:
                continue
            # 单动量：MA60 非必须（无过滤），但可用于标记
            base = roc_val

        elif method == "dual":
            roc_short = calc_roc(code, trade_date, roc_period)
            roc_long = calc_roc(code, trade_date, roc_period * 3)
            vol = calc_volatility(code, trade_date, roc_period, annualized=False)
            if roc_short is None or roc_long is None or vol is None:
                continue
            # MA60 过滤：跌破MA60的不参与排名
            if ma60 is not None and close_price < ma60:
                continue
            # 手册推荐权重：w1=0.4(近月动量), w2=0.4(近季动量), w3=0.2(波动惩罚)
            base = roc_short * 0.4 + roc_long * 0.4 - vol * 0.2
            # 追高保护：60日涨幅超过阈值时扣分，避免追涨见顶
            if roc_long > MOMENTUM_WARNING_THRESHOLD:
                excess = roc_long - MOMENTUM_WARNING_THRESHOLD
                base -= excess * 0.5

        elif method == "ma_filter":
            roc_val = calc_roc(code, trade_date, roc_period)
            if roc_val is None:
                continue
            # MA60 过滤：跌破MA60的不买入
            if ma60 is not None and close_price < ma60:
                continue
            base = roc_val

        else:
            raise ValueError(f"未知的调仓方法: {method}")

        # V6 独有 RSRS 质量分（历史不足返回 None，z 时按均值0处理）
        rsrs_val = calc_rsrs(code, trade_date, window=rsrs_window)
        rows.append((code, etf["name"], base, rsrs_val))

    if not rows:
        return []

    # ── 横截面 z 标准化：基分与 RSRS 分别 z，再加权合并 ──
    bases = np.array([r[2] for r in rows], dtype=float)
    rsrss = np.array([(0.0 if r[3] is None else r[3]) for r in rows], dtype=float)
    zb = (bases - bases.mean()) / (bases.std(ddof=0) + 1e-12)
    zr = (rsrss - rsrss.mean()) / (rsrss.std(ddof=0) + 1e-12)
    w = max(0.0, min(1.0, float(rsrs_weight)))
    combined = (1.0 - w) * zb + w * zr

    # 返回 4 元组 (code, name, combined, base)：combined 用于选股排序，base(平台原始基分)用于防抖比较
    results = [(rows[i][0], rows[i][1], float(combined[i]), float(rows[i][2])) for i in range(len(rows))]
    # 按得分降序排列
    results.sort(key=lambda x: x[2], reverse=True)
    return results


def select_targets(scored_list, method="dual", top_n=2):
    """根据调仓方法从打分列表中选择目标持仓

    返回:
        [ (code, name, weight_pct), ... ]
        weight_pct 为分配权重（小数，如 0.5=50%）
        空列表表示空仓（全部转货币基金避险）
    """
    if not scored_list:
        return []

    if method == "single":
        return [(scored_list[0][0], scored_list[0][1], 1.0)]

    elif method == "dual":
        # 双动量法：选Top N等权持有（手册推荐Top 2-3）
        n = min(top_n, len(scored_list))
        return [(scored_list[i][0], scored_list[i][1], 1.0 / n) for i in range(n)]

    elif method == "ma_filter":
        # 均线过滤法：选Top 1-2等权（已在打分时过滤MA60以下的）
        n = min(2, len(scored_list))
        return [(scored_list[i][0], scored_list[i][1], 1.0 / n) for i in range(n)]

    return []


# ── 折溢价闸门（ETF专属风控）──────────────────────────────
#
# 【实证依据】对本池 2018-2026 共 104 个调仓日、1755 个样本的事件研究：
#   · 境内ETF 溢价>5%：后20日均值 -6.18%、胜率20%  → 有真实负预测力
#     但 14/15 样本来自 501018(原油LOF·QDII额度受限)，宽基ETF因做市商
#     实时套利，91.5% 的样本溢价都在 ±1% 内，根本触发不了阈值。
#   · 跨境ETF 溢价>3%：后20日均值 +1.16%、胜率53.8% → 无负预测力
#     纳指ETF的溢价主要反映"境外隔夜已涨、NAV尚未更新"，是动量的代理
#     而非泡沫。对它设更严阈值会误杀（实测拖累收益，见 --premium-filter 对比）。
# 【结论】默认 off。仅在池中含申赎受限品种(QDII商品/QDII额度紧张)时建议开启 qdii 模式。
#
QDII_LIMITED = {          # 申赎受限 → 套利机制失灵 → 溢价会均值回归
    "501018.SH": "原油LOF",
    "513100.SH": "纳指ETF",
}
QDII_HARD_THRESHOLD = 0.08   # 受限品种的硬阈值（实证最优区间 8%+）
MAX_NAV_STALE_DAYS = 7       # 净值滞后超此天数 → 读数不可信 → 放行（不拦截）

# ── rolling 模式参数（推荐）──────────────────────────────
# 固定阈值的缺陷：QDII额度紧张会让溢价【结构性】抬升（纳指ETF 2026年平均
# 溢价+5.48%），固定5%线会把它永久屏蔽60%的时间——这不是择时，是误杀。
# 改用"相对自身过去1年的分位数"，自动区分结构性溢价与情绪泡沫。
# 实证(N=1741)：P95分位 且 绝对溢价≥2% → 后20日均值 -6.67%、胜率20.8%，
#               仅命中1.4%的样本；同期纳指ETF在2026年的P90拦截率为0%。
ROLLING_LOOKBACK = 252       # 分位数回看窗口（约1年）
ROLLING_MIN_HIST = 60        # 历史观测不足则放行
ROLLING_PCT = 0.95           # 分位阈值
ROLLING_MIN_PREM = 0.02      # 同时要求绝对溢价下限，避免低溢价品种被乱拦


class PremiumGate:
    """
    调仓日折溢价闸门：把高溢价标的从候选中剔除，让 top_n 自动顺延到下一名。

    mode:
      off     不启用（默认，保证历史回测可复现）
      uniform 统一硬阈值 5%（不区分跨境）
      strict  跨境更严（block 3% / 境内 5%）—— 实证偏差，仅供对比
      qdii    仅对申赎受限品种(QDII_LIMITED)用 8% 硬阈值
      rolling 自适应：溢价处于自身过去1年 P95 且绝对值≥2% —— 推荐

    溢价基准：统一使用「收盘确认净值 NAV」而非 IOPV(盘中参考净值)。
      IOPV 在标的停牌 / 交投清淡 / 跨境休市时会失真(沿用旧价)，NAV 为收盘确认值最可靠
      （依据 同UP主 BV1YP326jE7S《一只ETF同时出现两个价格》）。跨境标的额外用 NAV 滞后
      天数识别时滞造成的"假溢价拉宽"。

    安全默认：净值缺失 / 滞后过久 / 计价单位不一致 → 一律放行，绝不误杀。
    """

    def __init__(self, universe, mode="off", conn=None):
        self.mode = mode
        self.enabled = (mode != "off") and (_epf is not None)
        self.nav = {}        # {code: {date: unit_nav}}
        self.pxdates = {}    # {code: [排序的交易日]}，用于二分找 <=day 的最近价
        self.px = {}         # {code: {date: close}}
        self.blocked = []    # [(date, code, name, premium)]
        self.skipped_nodata = 0
        if not self.enabled:
            return
        c = conn or get_conn()
        for e in universe:
            code = e["code"]
            try:
                nrows = c.execute(
                    "SELECT nav_date, unit_nav FROM etf_nav WHERE ts_code=? "
                    "ORDER BY nav_date", (code,)).fetchall()
            except Exception:
                nrows = []
            self.nav[code] = {str(d): float(v) for d, v in nrows if v and v > 0}
            try:
                prows = c.execute(
                    "SELECT trade_date, close FROM etf_daily WHERE ts_code=? "
                    "ORDER BY trade_date", (code,)).fetchall()
            except Exception:
                prows = []
            self.px[code] = {str(d): float(v) for d, v in prows if v and v > 0}
            self.pxdates[code] = sorted(self.px[code])
        # rolling 模式：预构造【同日配对】的溢价序列，用于滚动分位数
        self.prem_series = {}   # {code: [(date, prem), ...]}
        if mode == "rolling":
            for code in self.px:
                ser = []
                navmap = self.nav.get(code, {})
                for d in self.pxdates[code]:
                    nav = navmap.get(d)
                    if not nav or nav <= 0:
                        continue
                    p = self.px[code][d]
                    if p / nav > _epf.PRICE_NAV_RATIO_CAP:
                        continue        # 货币ETF等计价单位不一致
                    ser.append((d, (p - nav) / nav))
                self.prem_series[code] = ser

    def _rolling_pct(self, code, day, prem):
        """当前溢价在自身过去 ROLLING_LOOKBACK 个观测中的分位；不可得返回 None。"""
        ser = self.prem_series.get(code) or []
        lo, hi, pos = 0, len(ser) - 1, -1
        while lo <= hi:
            m = (lo + hi) // 2
            if ser[m][0] <= day:
                pos = m
                lo = m + 1
            else:
                hi = m - 1
        if pos < 0:
            return None
        hist = [v for _, v in ser[max(0, pos - ROLLING_LOOKBACK):pos]]
        if len(hist) < ROLLING_MIN_HIST:
            return None
        return sum(1 for v in hist if v <= prem) / len(hist)

    def _price_on_or_before(self, code, day):
        """<= day 的最近交易日及其收盘价 (date, close)；无则 None。

        注意：不能用 get_etf_price()，它会无限回溯 fallback，
        导致把数月前的价格与当期净值配对，产生虚假溢价。
        """
        arr = self.pxdates.get(code) or []
        lo, hi, best = 0, len(arr) - 1, None
        while lo <= hi:
            mid = (lo + hi) // 2
            if arr[mid] <= day:
                best = arr[mid]
                lo = mid + 1
            else:
                hi = mid - 1
        return (best, self.px[code][best]) if best else None

    def hard_threshold(self, code):
        """该标的的 block 阈值；返回 None 表示不设限。

        rolling 模式返回绝对溢价下限（真正的判定还要叠加分位数条件）。
        """
        if self.mode == "uniform":
            return _epf.PREMIUM_HARD
        if self.mode == "strict":
            extra = _epf.CROSSBORDER_EXTRA if _epf.is_crossborder(code) else 0.0
            return _epf.PREMIUM_HARD - extra
        if self.mode == "qdii":
            return QDII_HARD_THRESHOLD if code in QDII_LIMITED else None
        if self.mode == "rolling":
            return ROLLING_MIN_PREM
        return None

    def check(self, code, day):
        """
        返回 (allow: bool, premium: float)。
        折溢价必须【同一天】的市价与净值配对，否则读数无意义。
        任何数据不确定的情形都返回 allow=True（宁可不拦，不可误杀）。
        """
        if not self.enabled:
            return True, float("nan")
        thr = self.hard_threshold(code)
        if thr is None:
            return True, float("nan")
        day = str(day)
        pr = self._price_on_or_before(code, day)
        if not pr:
            self.skipped_nodata += 1
            return True, float("nan")
        price_date, price = pr
        # 价格过于陈旧（该标的行情未更新）→ 放行
        if _epf.staleness_days(price_date, day) > MAX_NAV_STALE_DAYS:
            self.skipped_nodata += 1
            return True, float("nan")
        # 严格同日净值：找不到就放行，绝不用邻近日期凑
        nav = self.nav.get(code, {}).get(price_date)
        if not nav or nav <= 0:
            self.skipped_nodata += 1
            return True, float("nan")
        # 计价单位不一致（如货币ETF 净值1.0/市价100）→ 折溢价无意义
        if price / nav > _epf.PRICE_NAV_RATIO_CAP:
            return True, float("nan")
        prem = (price - nav) / nav
        if self.mode == "rolling":
            # 双条件：绝对溢价不低 + 处于自身历史高位（自适应结构性溢价）
            if prem < ROLLING_MIN_PREM:
                return True, prem
            pct = self._rolling_pct(code, price_date, prem)
            if pct is None:
                self.skipped_nodata += 1
                return True, prem
            return (pct < ROLLING_PCT), prem
        return (prem < thr), prem

    def filter_scored(self, scored, day, universe, verbose=False):
        """对打分列表逐个过滤，返回保留下来的列表（顺序不变）。"""
        if not self.enabled or not scored:
            return scored
        kept = []
        for item in scored:
            code = item[0]
            allow, prem = self.check(code, day)
            if allow:
                kept.append(item)
            else:
                name = next((e["name"] for e in universe if e["code"] == code), code)
                self.blocked.append((day, code, name, prem))
                if verbose:
                    why = (f"处于自身1年P{ROLLING_PCT*100:.0f}高位"
                           if self.mode == "rolling"
                           else f"≥ {self.hard_threshold(code):.0%}")
                    print(f"    [折溢价拦截] {name}({code}) 溢价 {prem:+.2%} {why}"
                          f" → 剔除候选")
        return kept


# ── 板块三态状态机前置闸门（B 项）──────────────────────────
#
# 【设计原则】「状态确认」而非「强度排序」（§5.12 已证板块强度=负贡献）：
#   · 横截面动量强弱排序 → 容易买到「最强=刚见顶」的标的（动量均值回归）。
#   · 本闸门不重排候选，只对每个候选逐期判定其趋势「状态」——
#       加速见底(左侧)  → 价格在长期趋势线(MA60)下方，或站上 MA60 但 MA60 斜率仍
#                         向下（死猫反弹）→ 结构未反转，不抄底、不接下落刀 → 剔除。
#       右侧趋势        → 价格>MA60 且 MA60 斜率转正，动量平稳/温和减速 → 可持有。
#       趋势加速        → 右侧基础上短期动量相对长期动量加速(roc20>roc60) → 突破主升。
#   · 只保留 右侧趋势+趋势加速，剔除 加速见底。排序仍由策略自身 RSRS+双动量决定，
#     二者解耦，避免把状态机变成又一个强度排序器。
#   · 数据不足(UNKNOWN)保守保留，不误杀。
#
def classify_etf_state(code, trade_date, hist_len=SSM_HIST_LEN):
    """用本地库价格历史判定某 ETF 当期趋势状态（接 sector_state_machine 纯函数）。

    返回 STATE_CN 的 key：ACCEL_BOTTOM / RIGHT_TREND / TREND_ACCEL / UNKNOWN。
    """
    closes = _query_history(code, trade_date, "close", hist_len)
    if closes is None or len(closes) < SSM_HIST_LEN:
        return UNKNOWN
    asc = np.array(closes[::-1], dtype=float)  # 降序→升序(旧→新)
    st, _det = classify_state(asc)
    return st


class SectorStateGate:
    """板块三态状态机前置闸门：与 PremiumGate 同接口，用于调仓前过滤候选。

    只进 右侧趋势+趋势加速，剔除 加速见底（左侧/下落刀）。
    """

    def __init__(self, universe, mode="off"):
        self.mode = mode
        self.enabled = (mode == "on")
        self.dropped = []   # [(date, code, name, state)]
        self.kept = []      # [(date, code, name, state)] 诊断用

    def filter_scored(self, scored, day, universe, verbose=False):
        """对打分列表逐个按三态过滤，返回保留列表（顺序不变）。"""
        if not self.enabled or not scored:
            return scored
        kept = []
        for item in scored:
            code = item[0]
            st = classify_etf_state(code, day)
            if st in GATE_PASS:
                kept.append(item)
                self.kept.append((day, code, st))
            else:
                name = next((e["name"] for e in universe if e["code"] == code), code)
                self.dropped.append((day, code, name, st))
                if verbose:
                    print(f"    [三态闸门剔除] {name}({code}) 状态={STATE_CN.get(st, st)} → 不接左侧")
        return kept


# ── 主回测循环 ────────────────────────────────────────────

# ── 市场择时 overlay 辅助：加载全市场收盘价矩阵（供广度振荡器）──
def _load_market_close_p(start_date, end_date):
    """从 daily 表取全市场收盘价，pivot 成 (trade_date × ts_code) 矩阵。
    回溯 300 日以确保 MA200 有足够 min_periods。返回 DataFrame(index=int日期)。
    """
    import datetime
    sy, sm, sd = int(start_date[:4]), int(start_date[4:6]), int(start_date[6:8])
    lo = (datetime.date(sy, sm, sd) - datetime.timedelta(days=300)).strftime("%Y%m%d")
    con = get_conn()
    try:
        df = pd.read_sql(
            "SELECT trade_date, ts_code, close FROM daily "
            "WHERE trade_date>=? AND trade_date<=?",
            con, params=(lo, end_date))
    finally:
        con.close()
    if df.empty:
        return pd.DataFrame()
    px = df.pivot(index="trade_date", columns="ts_code", values="close").sort_index()
    px.index = pd.to_numeric(px.index).astype(int)
    return px


def _build_timing_gate_caps(rebalance_dates, start_date, end_date,
                            boil, ice, floor, verbose=True):
    """对调仓日计算择时 cap ∈ [floor,1]（非对称，只减仓）。返回 {int_date: cap}。"""
    px = _load_market_close_p(start_date, end_date)
    if px.empty:
        if verbose:
            print("  [择时overlay] 警告：daily 表为空，闸门未生效（退化为满仓）")
        return {}
    osc_all = compute_breadth_oscillator(px).dropna()
    caps = {}
    for d in sorted(rebalance_dates):
        caps[d] = float(position_cap(float(osc_all[d]), boil, ice, floor)) \
            if d in osc_all.index else 1.0   # 早期无信号 → 满仓
    return caps


def run_etf_rotation(start_date="20200101", end_date="20251231",
                     method="dual", roc_period=20, ma_period=60,
                     capital=INITIAL_CAPITAL, verbose=True,
                     top_n=2, switch_threshold=SWITCH_THRESHOLD,
                     cash_buffer_pct=CASH_BUFFER_PCT,
                     pool="mixed",
                     var_control=0, var_maxdd=15.0, var_n=5,
                     var_lookback=120, var_method="hist",
                     interrupt_start=None, interrupt_months=0, interrupt_pct=0.0,
                     premium_filter="off",
                     rsrs_weight=RSRS_WEIGHT, rsrs_window=RSRS_WINDOW,
                     regime_hook=None,
                     sector_gate="off",
                     timing_gate=False, boil=80, ice=55, floor=0.0):
    """ETF轮动策略主回测

    Args:
        start_date: 回测开始日期 YYYYMMDD
        end_date:   回测结束日期 YYYYMMDD
        method:     调仓方法 "single" / "dual" / "ma_filter"
        roc_period: ROC 计算周期
        ma_period:  MA 计算周期
        capital:    初始资金
        verbose:    是否打印日志
        top_n:      双动量法持仓数量（默认2，手册推荐2-3）
        switch_threshold: 最小切换阈值（默认5%，防抖动）
        cash_buffer_pct:  现金缓冲比例（默认10%）

    Returns:
        dict: 回测结果
    """
    # ── 标的池选择 ──
    universe = INDUSTRY_UNIVERSE if pool == "industry" else ETF_UNIVERSE
    pool_names = {"mixed": "混合池(20只·宽基+行业+跨境+商品)", "industry": "行业池(14只·纯行业+科技)"}

    # ── 获取交易日和调仓日 ──
    trade_dates = get_trade_dates(start_date, end_date)
    if len(trade_dates) < 60:
        print(f"[ERR] 交易日数据不足：{len(trade_dates)} 天 (需至少 60 天)")
        return None

    rebalance_dates = get_monthly_5th_trading_days(trade_dates)
    # 调仓日集合，只包含回测区间内的
    rebalance_dates = {d for d in rebalance_dates if start_date <= str(d) <= end_date}

    method_names = {"single": "单动量法", "dual": "双动量法", "ma_filter": "均线过滤法"}

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  ETF轮动策略回测")
        print(f"{'=' * 70}")
        print(f"  标的池：{pool_names.get(pool, pool)}")
        print(f"  成员：{' | '.join(e['name'] for e in universe)}")
        print(f"  调仓方法：{method_names.get(method, method)}")
        print(f"  RSRS质量分：权重 {rsrs_weight:.2f}（窗口{RSRS_WINDOW}天）| 其余 {1-rsrs_weight:.2f} 给平台双动量基分")
        if rsrs_weight <= 0:
            print(f"    （--rsrs-weight 0 → 纯平台，RSRS 未参与）")
        if var_control and var_control > 0:
            print(f"  VaR仓位缩放：开启 置信{var_control}% | 目标回撤{var_maxdd}% | N={var_n}（未投部分留现金）")
        print(f"  回测区间：{start_date} ~ {end_date}")
        print(f"  初始资金：{capital:,.2f}")
        print(f"  交易日：{len(trade_dates)} 天 | 调仓日：{len(rebalance_dates)} 次")
        print()

    # ── 市场择时 overlay（A 项振荡器闸门，默认关，不破坏审计链）──
    timing_caps = {}
    n_timing_trig = 0
    if timing_gate:
        timing_caps = _build_timing_gate_caps(
            rebalance_dates, start_date, end_date, boil, ice, floor, verbose=verbose)
        n_timing_trig = sum(1 for c in timing_caps.values() if c < 1.0)
        if verbose:
            print(f"  市场择时 overlay：开启（沸点>={boil}清仓 / 冰点<={ice}满仓 / floor={floor:.2f}）")
            print(f"    触发减仓或清仓的调仓月 = {n_timing_trig}/{len(timing_caps)}")
            print()

    # ── 初始化 ──
    cash = float(capital)
    positions = {}  # { code: {"shares": int, "buy_price": float} }
    trades = []
    daily_vals = []

    # ── 折溢价闸门（默认 off，不影响历史结果）──
    gate = PremiumGate(universe, mode=premium_filter)
    if gate.enabled and verbose:
        mode_desc = {"uniform": "统一5%硬阈值", "strict": "跨境更严(3%/5%)",
                     "qdii": f"仅限申赎受限品种({QDII_HARD_THRESHOLD:.0%})",
                     "rolling": "自适应分位数(P95+abs≥2%)"}
        print(f"  折溢价过滤：开启 [{premium_filter}] {mode_desc.get(premium_filter, '')}")
        print()

    # ── 板块三态状态机前置闸门（默认 off，不破坏审计链）──
    sector_state_g = SectorStateGate(universe, mode=sector_gate)
    if sector_state_g.enabled and verbose:
        print(f"  板块三态闸门：开启（只进 右侧趋势+趋势加速，剔除 加速见底/左侧）")
        print()

    # ── Regime β 兜底开关提示（仅 on 时）──
    if regime_hook is not None and verbose:
        _det = getattr(regime_hook, "detector", None)
        _rule = getattr(_det, "rule", "?") if _det else "?"
        _ma = getattr(_det, "ma_len", "?") if _det else "?"
        _bf = getattr(_det, "breadth_thr", 0.25) if _det else 0.25
        _mode = getattr(_det, "breadth_mode", "?") if _det else "?"
        _mc = getattr(_det, "min_consecutive", 2) if _det else 2
        try:
            from regime_core import BETA_ETFS as _BETA_ETFS
            _bnames = ",".join(_BETA_ETFS.values())
        except Exception:
            _bnames = ""
        print(f"  Regime β 兜底：开启 [Rule{_rule}] MA_LEN={_ma} | "
              f"β底线={regime_hook.beta_floor:.0%} | 宽度口径={_mode} | "
              f"滞后确认={_mc}月 | β标的={_bnames}")
        print()

    # ── 逐日循环 ──
    for i, td_str in enumerate(trade_dates):
        td = int(td_str)

        # 前一天日期
        prev_td = int(trade_dates[i - 1]) if i > 0 else td

        # ── 调仓日：执行轮动（先调仓，再记录净值）──
        if td in rebalance_dates:
            # 用前一天收盘价打分
            scored = score_assets(prev_td, method=method,
                                  roc_period=roc_period, ma_period=ma_period,
                                  universe=universe,
                                  rsrs_weight=rsrs_weight, rsrs_window=rsrs_window)
            # 折溢价闸门：剔除高溢价标的，top_n 自动顺延到下一名
            scored = gate.filter_scored(scored, prev_td, universe, verbose=verbose)
            # 板块三态闸门：剔除 加速见底(左侧)，只留 右侧趋势+趋势加速
            scored = sector_state_g.filter_scored(scored, prev_td, universe, verbose=verbose)
            targets = select_targets(scored, method=method, top_n=top_n)

            # ── Regime β 兜底（--regime 开关，默认关闭，不破坏审计链）──
            # 仅在 BULL（带滞后确认）时把排名最高 β ETF 顶补到 BETA_FLOOR，削减最投机仓；
            # 熊市不兜底，策略原样跑（保留货基避险/防御）。regime_hook=None 时完全跳过。
            if regime_hook is not None:
                targets = regime_hook(scored, targets, prev_td)

            # ── VaR 仓位缩放（可选）：按目标篮子历史估 VaR，反解本期投入比例 ──
            var_ratio = 1.0
            if var_control and var_control > 0 and targets:
                var_ratio = _etf_var_invest_ratio(
                    [c for c, _, _ in targets], prev_td,
                    var_control, var_maxdd, var_n, var_lookback, var_method)

            # ── 最小切换阈值（防抖动）──
            # 当前持仓得分与新目标接近时，不切换，避免临界来回换手
            # 防抖比较用平台原始基分 base（尺度不变，权重=0 时与纯平台逐笔一致）；
            # 选股排序用 combined(z 加权)，二者解耦避免 z 分数拉伸破坏 anti-jitter 阈值。
            score_lookup = {c: b for c, _, _, b in scored}
            target_codes = {c for c, _, _ in targets}
            protected = set()
            if switch_threshold > 0 and positions and targets:
                for code in list(positions.keys()):
                    if code in target_codes or code not in score_lookup:
                        continue
                    current_score = score_lookup[code]
                    target_scores = [score_lookup.get(c, 0.0) for c, _, _ in targets]
                    if target_scores:
                        worst_target = min(target_scores)
                        if worst_target > 0 and current_score > 0 and \
                           current_score >= worst_target * (1 - switch_threshold):
                            protected.add(code)

            if verbose:
                score_str = ", ".join(f"{n}({s:.1%})" for _, n, s, _b in scored[:3])
                if targets:
                    target_str = ", ".join(f"{n}({w:.0%})" for _, n, w in targets)
                else:
                    target_str = f"{CASH_NAME}(避险)"
                prot_names = [next((e["name"] for e in universe if e["code"] == p), p) for p in protected]
                prot_str = f" | 保留={','.join(prot_names)}" if protected else ""
                var_str = f" | VaR投入{var_ratio:.0%}" if (var_control and var_control > 0 and targets and var_ratio < 1.0) else ""
                regime_str = ""
                if regime_hook is not None:
                    br = regime_hook.breadth
                    br_s = (f"宽度={br:.0%}" if br is not None else "宽度=N/A")
                    regime_str = f" | Regime={regime_hook.state}({regime_hook.raw}·趋势{regime_hook.trend}·{br_s})"
                print(f"  调仓日 {td}：得分前三={score_str} | 目标={target_str}{prot_str}{var_str}{regime_str}")

            # ── A 项市场择时 overlay：非对称减仓（只减不增）──
            # cap=1（正常市）→ 完全走原基线轮动；cap<1（过热）→ 按比例 trim 全部权益、不轮换不抄底
            cap = 1.0
            if timing_gate and td in timing_caps:
                cap = timing_caps[td]

            if cap < 1.0 - 1e-9:
                # ── 闸门触发：非对称减仓，按 cap 比例 trim 全部权益持仓（货币基金不动）──
                cur_equity = 0.0
                for _c, _p in positions.items():
                    if _c == MONEY_FUND_CODE:
                        continue
                    _px = get_etf_price(_c, prev_td) or _p.get("buy_price") or 0.0
                    cur_equity += _p["shares"] * _px
                if cur_equity > 0:
                    scale = cap
                    for code in list(positions.keys()):
                        if code == MONEY_FUND_CODE:
                            continue
                        open_price = get_etf_open(code, td)
                        if open_price is None or open_price <= 0:
                            continue
                        pos = positions[code]
                        keep_shares = int(pos["shares"] * scale / 100) * 100
                        sell_shares = pos["shares"] - keep_shares
                        if sell_shares >= 100:
                            proceeds = sell_shares * open_price
                            fee = calc_etf_fee('sell', open_price, sell_shares)
                            cash += proceeds - fee
                            trades.append({
                                "date": td, "action": "SELL", "code": code,
                                "name": next((e["name"] for e in universe if e["code"] == code), code),
                                "price": open_price, "shares": sell_shares,
                                "reason": f"timing_trim(cap={cap:.2f})"
                            })
                            pos["shares"] = keep_shares
                            if keep_shares == 0:
                                del positions[code]
                if verbose:
                    print(f"    → 择时闸门触发(cap={cap:.2f})：权益按比例 trim 至 {cap*100:.0f}%，"
                          f"本月不轮换、不抄底")
            else:
                # ── 常规轮动（与基线完全一致）──
                # ── 卖出不在目标中的旧持仓（跳过protected和货币基金）──
                for code in list(positions.keys()):
                    if code in target_codes or code in protected:
                        continue
                    # 熊市模式下保留货币基金
                    if not targets and code == MONEY_FUND_CODE:
                        continue
                    open_price = get_etf_open(code, td)
                    if open_price is None or open_price <= 0:
                        continue
                    pos = positions[code]
                    proceeds = pos["shares"] * open_price
                    fee = calc_etf_fee('sell', open_price, pos["shares"])
                    cash += proceeds - fee
                    trades.append({
                        "date": td, "action": "SELL", "code": code,
                        "name": next((e["name"] for e in universe if e["code"] == code), code),
                        "price": open_price, "shares": pos["shares"], "reason": "rotation"
                    })
                    if verbose:
                        etf_name = next((e["name"] for e in universe if e["code"] == code), code)
                        print(f"    → 卖出 {etf_name}：{pos['shares']}份 @ {open_price:.3f}")
                    del positions[code]

                # ── 买入新的目标持仓（等权分配，保留现金缓冲）──
                if targets:
                    new_to_buy = [(c, n, w) for c, n, w in targets if c not in positions]
                    if new_to_buy:
                        # VaR 缩放：现金缓冲后再乘投入比例，未投部分留现金（凶策略同款口径）
                        investable = cash * (1 - cash_buffer_pct) * var_ratio
                        cash_per_target = investable / len(new_to_buy)
                        for code, name, weight in new_to_buy:
                            open_price = get_etf_open(code, td)
                            if open_price is None or open_price <= 0:
                                continue
                            alloc = cash_per_target * weight * len(new_to_buy)
                            alloc = min(alloc, cash)
                            alloc_after_fee = alloc * 0.998
                            max_shares = int(alloc_after_fee / open_price / 100) * 100
                            if max_shares < 100:
                                continue
                            cost = max_shares * open_price
                            fee = calc_etf_fee('buy', open_price, max_shares)
                            if cost + fee <= cash:
                                cash -= cost + fee
                                positions[code] = {"shares": max_shares, "buy_price": open_price}
                                trades.append({
                                    "date": td, "action": "BUY", "code": code,
                                    "name": name, "price": open_price,
                                    "shares": max_shares, "reason": "rotation"
                                })
                                if verbose:
                                    print(f"    → 买入 {name}：{max_shares}份 @ {open_price:.3f}")
                else:
                    # ── 熊市避险：全部走弱时买入货币基金 ──
                    if MONEY_FUND_CODE not in positions:
                        open_price = get_etf_open(MONEY_FUND_CODE, td)
                        if open_price and open_price > 0:
                            investable = cash * (1 - cash_buffer_pct)
                            alloc_after_fee = investable * 0.998
                            max_shares = int(alloc_after_fee / open_price / 100) * 100
                            if max_shares >= 100:
                                cost = max_shares * open_price
                                fee = calc_etf_fee('buy', open_price, max_shares)
                                if cost + fee <= cash:
                                    cash -= cost + fee
                                    positions[MONEY_FUND_CODE] = {"shares": max_shares, "buy_price": open_price}
                                    mf_name = next((e["name"] for e in universe if e["code"] == MONEY_FUND_CODE), CASH_NAME)
                                    trades.append({
                                        "date": td, "action": "BUY", "code": MONEY_FUND_CODE,
                                        "name": mf_name, "price": open_price,
                                        "shares": max_shares, "reason": "bear_market_safe_haven"
                                    })
                                    if verbose:
                                        print(f"    → 熊市避险：买入 {mf_name}：{max_shares}份 @ {open_price:.3f}")

        # ── 每日市值记录（调仓后，反映当日实际持仓的收盘价）──
        total_value = cash
        for code, pos in list(positions.items()):
            price = get_etf_price(code, td)
            if price is not None:
                total_value += pos["shares"] * price
        daily_vals.append({"date": td, "value": total_value})

    # ── 回测结束：平仓 ──
    if trade_dates:
        last_date = trade_dates[-1]
        for code in list(positions.keys()):
            price = get_etf_price(code, last_date)
            if price is not None:
                pos = positions[code]
                proceeds = pos["shares"] * price
                fee = calc_etf_fee('sell', price, pos["shares"])
                cash += proceeds - fee
                trades.append({
                    "date": last_date, "action": "SELL", "code": code,
                    "name": next((e["name"] for e in universe if e["code"] == code), code),
                    "price": price, "shares": pos["shares"], "reason": "backtest_end"
                })
                del positions[code]

    # ── 计算绩效 ──
    final_value = cash
    total_return = (final_value / capital - 1) * 100
    days = len(trade_dates)
    years = days / 252.0
    annual_return = ((final_value / capital) ** (1 / years) - 1) * 100 if years > 0 else 0

    vals = np.array([d["value"] for d in daily_vals])
    cummax = np.maximum.accumulate(vals)
    safe_cummax = np.where(cummax == 0, 1, np.array(cummax, dtype=float))
    drawdowns = (vals - cummax) / safe_cummax
    max_dd = float(np.min(drawdowns)) * 100

    rets = np.diff(vals) / np.where(vals[:-1] == 0, 1, vals[:-1])
    if len(rets) > 1 and np.std(rets) > 0:
        sharpe = (np.mean(rets) * 252 - 0.025) / (np.std(rets) * np.sqrt(252))
    else:
        sharpe = 0.0

    # ── 基准收益 ──
    b_start = get_etf_price(BENCHMARK_CODE, trade_dates[0])
    b_end = get_etf_price(BENCHMARK_CODE, trade_dates[-1])
    idx_return = (b_end / b_start - 1) * 100 if b_start and b_end else 0

    # ── 输出 ──
    profit_amount = final_value - capital
    print(f"\n{'=' * 70}")
    print(f"  ETF轮动策略回测结果（{method_names.get(method, method)}）")
    print(f"{'=' * 70}")
    print(f"  初始资金：{capital:,.2f}")
    print(f"  最终资产：{final_value:,.2f}")
    print(f"  总盈亏：{profit_amount:+,.0f} 元")
    print(f"  总收益率：{total_return:+.2f}%")
    print(f"  年化收益率：{annual_return:+.2f}%")
    print(f"  最大回撤：{max_dd:.2f}%")
    print(f"  夏普比率：{sharpe:.2f}")
    print(f"  交易次数：{len(trades)}（买{sum(1 for t in trades if t['action']=='BUY')}次 / "
          f"卖{sum(1 for t in trades if t['action']=='SELL')}次）")
    win_rate, win_cnt, tot_cnt = calc_win_rate(trades)
    if tot_cnt > 0:
        print(f"  胜率：{win_rate:.1f}%（{win_cnt}/{tot_cnt}）")
    print(f"  基准（沪深300）涨幅：{idx_return:+.2f}%")
    print(f"  超额收益：{total_return - idx_return:+.2f}%")
    if sector_state_g.enabled:
        print(f"  板块三态闸门：剔除 {len(sector_state_g.dropped)} 次候选(加速见底/左侧) ｜ "
              f"放行 {len(sector_state_g.kept)} 次")
    if timing_gate:
        print(f"  市场择时 overlay：触发减仓/清仓 {n_timing_trig} 个调仓月 ｜ "
              f"沸点>={boil}清仓 / 冰点<={ice}满仓 / floor={floor:.2f}")

    # ── 现实折扣三件套（扣通胀 / 定投拖累 / 中断模拟）──
    disc = compute_reality_discounts(
        daily_vals, capital,
        interrupt_start=interrupt_start,
        interrupt_months=interrupt_months,
        interrupt_pct=interrupt_pct,
    )
    if "real_total_return" in disc:
        print(f"  ── 现实折扣（预期管理，不改收益计算）──")
        print(f"  扣通胀真实总收益：{disc['real_total_return']:+.2f}% ｜ 真实年化：{disc['real_annual_return']:+.2f}%")
    if "dca_drag_pct" in disc:
        print(f"  定投对比(DCA)：一次性建仓较分12月定投 {disc['dca_drag_pct']:+.2f}%"
              f"（正=一次性占优·负=定投占优）｜ 终值 一次性 {disc['dca_lump_final']:,.0f} / 定投 {disc['dca_dca_final']:,.0f}")
    if "interrupt_loss_pct" in disc:
        print(f"  中断模拟：{interrupt_start}起撤{interrupt_pct*100:.0f}%持有{interrupt_months}月，"
              f"终值损失 {disc['interrupt_loss_pct']:+.2f}%（终值 {disc['interrupt_final']:,.0f}）")

    # ── VaR 报告行（开启 VaR 缩放时输出实际权益曲线的风险水平）──
    if var_control and var_control > 0:
        try:
            from run_monthly_rebalance import equity_curve_var
            _ev = equity_curve_var([d["value"] for d in daily_vals],
                                   capital=final_value, conf_levels=(0.95, 0.99), method="hist")
            print(f"  VaR(95%)单日：{_ev[0.95]['hist_loss']*100:.2f}% | "
                  f"VaR(99%)单日：{_ev[0.99]['hist_loss']*100:.2f}%（历史法·实际权益曲线）")
        except Exception:
            pass

    # ── 成交明细 CSV 导出（含 reason 列，调试友好）──
    if trades:
        _trades_df = pd.DataFrame(trades)[
            ["date", "action", "code", "name", "price", "shares", "reason"]
        ]
        _trades_df.to_csv("etf_rotation_trades.csv", index=False, encoding="utf-8-sig")
        print(f"  成交明细：已导出 etf_rotation_trades.csv（{len(_trades_df)} 笔，含 reason 列）")

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "trades": len(trades),
        "idx_return": idx_return,
        "final_value": final_value,
        "pool": pool,
        "var_control": var_control,
        "premium_filter": premium_filter,
        "premium_blocked": list(gate.blocked),
        "timing_gate": timing_gate,
        "timing_triggered": n_timing_trig,
        "timing_boil": boil, "timing_ice": ice, "timing_floor": floor,
    }


# ── CLI 入口 ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETF轮动策略回测")
    parser.add_argument("start_date", nargs="?", default="20200101", help="开始日期 YYYYMMDD")
    parser.add_argument("end_date", nargs="?", default="20251231", help="结束日期 YYYYMMDD")
    parser.add_argument("--method", default="dual",
                        choices=["single", "dual", "ma_filter"],
                        help="调仓方法（默认 dual=双动量法）")
    parser.add_argument("--pool", default="mixed",
                        choices=["mixed", "industry"],
                        help="标的池：mixed=原20只混合池（默认）| industry=行业池14只(12行业+2科技)")
    parser.add_argument("--var-control", type=int, default=0,
                        help="VaR仓位缩放置信度：0=关闭（默认）| 95 | 99")
    parser.add_argument("--var-maxdd", type=float, default=15.0,
                        help="VaR目标最大回撤上限%%（默认15）")
    parser.add_argument("--var-n", type=int, default=5,
                        help="回撤预算分摊系数N（默认5）")
    parser.add_argument("--roc-period", type=int, default=20, help="ROC计算周期（默认20）")
    parser.add_argument("--ma-period", type=int, default=60, help="MA计算周期（默认60）")
    parser.add_argument("--capital", type=int, default=INITIAL_CAPITAL, help="初始资金（默认100000）")
    parser.add_argument("--top-n", type=int, default=2, help="双动量法持仓数量（默认2）")
    parser.add_argument("--switch-threshold", type=float, default=SWITCH_THRESHOLD,
                        help=f"最小切换阈值（默认{SWITCH_THRESHOLD}，防抖动）")
    parser.add_argument("--cash-buffer", type=float, default=CASH_BUFFER_PCT,
                        help=f"现金缓冲比例（默认{CASH_BUFFER_PCT}）")
    parser.add_argument("--premium-filter", type=str, default="off",
                        choices=["off", "uniform", "strict", "qdii", "rolling"],
                        help="ETF折溢价过滤：off=关(默认) | uniform=统一5%% | "
                             "strict=跨境更严(3%%/5%%) | qdii=仅申赎受限品种8%% | "
                             "rolling=自适应分位数(P95+绝对值≥2%%，推荐)")
    parser.add_argument("--sector-gate", action="store_true",
                        help="板块三态状态机前置闸门：只进 右侧趋势+趋势加速，剔除 加速见底(左侧/下落刀)。"
                             "默认关(不破坏审计链)")
    parser.add_argument("--timing-gate", action="store_true",
                        help="A项市场择时 overlay：全市场广度振荡器(沸点>=boil清仓/冰点<=ice满仓)非对称闸门，"
                             "只减仓不抄底。默认关(不破坏审计链)。阈值见 --boil/--ice/--floor")
    parser.add_argument("--boil", type=float, default=80.0,
                        help="择时 overlay 沸点阈值(默认80，振荡器>=此值清仓)")
    parser.add_argument("--ice", type=float, default=20.0,
                        help="择时 overlay 冰点阈值(默认20，振荡器<=此值满仓)")
    parser.add_argument("--floor", type=float, default=0.0,
                        help="择时 overlay 清仓时最低保留仓位(默认0=全清；设0.2=最多降到2成)")
    parser.add_argument("--quiet", action="store_true", help="静默模式")
    parser.add_argument("--interrupt-start", type=str, default=None,
                        help="现实折扣-中断模拟：从 YYYYMM 起撤出部分资金（配合 --interrupt-months/--interrupt-pct）")
    parser.add_argument("--interrupt-months", type=int, default=0,
                        help="中断模拟持续月数（默认0=关闭）")
    parser.add_argument("--interrupt-pct", type=float, default=0.0,
                        help="中断模拟撤出比例(0~1，如 0.5=撤一半)，默认0")
    parser.add_argument("--rsrs-weight", type=float, default=RSRS_WEIGHT,
                        help=f"RSRS质量分权重(0~1，默认{RSRS_WEIGHT})；0=纯平台，0.6=复刻V6风格")
    parser.add_argument("--rsrs-window", type=int, default=RSRS_WINDOW,
                        help=f"RSRS回归窗口天(默认{RSRS_WINDOW})")
    # ── Regime β 兜底开关（默认 off，不破坏原版审计链）──
    parser.add_argument("--regime", type=str, default="off",
                        choices=["off", "on"],
                        help="Regime β 兜底开关：off=关(默认,保留原版审计链) | on=开(牛市强制≥β底线宽基β)")
    parser.add_argument("--beta-floor", type=float, default=0.40,
                        help="牛市强制最低宽基β权重(默认0.40=40%%，配合 --regime on)")
    parser.add_argument("--regime-rule", type=str, default="B",
                        choices=["A", "B", "C"],
                        help="Regime合成规则：A=AND(趋势&宽度≥0.5) | B=趋势为主(趋势&宽度≥0.25,默认) | C=仅趋势")
    parser.add_argument("--ma-len", type=int, default=200,
                        help="沪深300 趋势均线长度(默认200日)")
    parser.add_argument("--breadth-thr", type=float, default=0.25,
                        help="宽度触发阈值(默认0.25，仅 RuleA/B 用)")
    parser.add_argument("--breadth-mode", type=str, default="proxy",
                        choices=["proxy", "full"],
                        help="宽度口径：proxy=8大指数代理(默认,已验证) | full=全A个股(较重,需复验)")
    parser.add_argument("--min-consecutive", type=int, default=2,
                        help="BULL 滞后确认月数(默认2=连续2月BULL才生效，降whipsaw误触)")
    args = parser.parse_args()

    # ── 构造 regime hook（仅 --regime on 时惰性导入，off 时完全不触发）──
    regime_hook = None
    if args.regime == "on":
        from regime_core import build_regime_hook
        regime_hook = build_regime_hook(
            beta_floor=args.beta_floor, rule=args.regime_rule,
            ma_len=args.ma_len, breadth_thr=args.breadth_thr,
            breadth_mode=args.breadth_mode, min_consecutive=args.min_consecutive)

    run_etf_rotation(
        start_date=args.start_date,
        end_date=args.end_date,
        method=args.method,
        roc_period=args.roc_period,
        ma_period=args.ma_period,
        capital=args.capital,
        verbose=not args.quiet,
        top_n=args.top_n,
        switch_threshold=args.switch_threshold,
        cash_buffer_pct=args.cash_buffer,
        pool=args.pool,
        var_control=args.var_control,
        var_maxdd=args.var_maxdd,
        var_n=args.var_n,
        interrupt_start=args.interrupt_start,
        interrupt_months=args.interrupt_months,
        interrupt_pct=args.interrupt_pct,
        premium_filter=args.premium_filter,
        rsrs_weight=args.rsrs_weight,
        rsrs_window=args.rsrs_window,
        regime_hook=regime_hook,
        sector_gate="on" if args.sector_gate else "off",
        timing_gate=args.timing_gate,
        boil=args.boil, ice=args.ice, floor=args.floor,
    )
