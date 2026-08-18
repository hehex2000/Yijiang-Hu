# -*- coding: utf-8 -*-
"""
RSRS 宽基 ETF 轮动 (BV1MS3d6qEZx「14倍」) 忠实复现 + 开关版回测
================================================================

视频核心 (转写逐字):
  - 8 只宽基 ETF 轮动, 月度选排名第 1 的 1 只持有
  - 模块① 双均线过滤: MA5 > MA20 才允许买入 (多头排列过滤)
  - 模块② RSRS 复合排序: 20日涨跌幅(动量) 与 RSRS(阻力支撑相对强度) 按 6:4 加权
                        (z-score 标准化后加总排序); RSRS 权重 0→100% 扫过, 40% 最优
  - 模块③ ATR 自适应止损: 每支 ETF 独立阈值 (波动大放宽)
      黄金/纳指/沪深300/恒生=5% | 国证2000=5.4% | 中证1000=5.9% | 创业板50=7.5% | 科创100=8.7%
  - 宣称: 2017-01~2026-07 累计1388%(≈14×) / 年化32.6% / 回撤30.7% / 夏普1.25 / 卡玛1.06
  - 原始策略 (纯20日动量选最强1只): 年化25.4% / 累计772%(≈8×)
  - 保守替代: 无过滤+0%RS+8%固定止损: 年化27% / 回撤25%

本脚本目标: faithful 还原这套方法, 并把每个模块做成 **开关** (argparse) 便于后续
反过拟合电池. **不追求复现14倍** —— 那是 32 组网格 in-sample 调参的产物
(见 plan_etf_rsrs_rotation.md). 阶段0 只做 faithful 复现 + 上市日闸门 + 基准对比.

诚实边界 (必读):
  * 候选池事前固定 8 只宽基 ETF (不得为美化回测改池子).
  * 🔴 上市日闸门 (本计划头号纪律): 科创100(2023-09-15)/国证2000(2022-07-18) 上市晚,
    必须在其首交易日之后才纳入候选池 —— 否则 2017 年就凭空持有 2023 才上市的标的 = 真·前视.
  * RSRS 实现 = 经典光大证券(2017)定义: 对近 N 日 (high, low) 做 OLS 回归 low~high, 斜率β=RSRS
    (β>1 支撑强于阻力, 趋势易延续), 横截面 z-score 标准化.
    ⚠️ 注意: 平台 V6 的 rsrs_quality 是"归一化收盘价对时间回归的斜率×R²"(趋势质量代理),
    并非经典阻力支撑相对强度. 本脚本按 UP 自陈定义实现经典 low~high RSRS, 不与 V6 混用.
  * 调仓: 月度第5交易日开盘 (平台范式). UP 未明说频率, 标注假设.
  * RSRS 窗口 / 动量窗口(20日已给) 中 RSRS 窗口 UP 未给 -> 默认20, 阶段2做敏感性.
  * QDII/商品LOF 缺 etf_adj_factor -> 沿用刘不牛已验证的 pct_chg 累乘复权, 避开份额折算断崖.
  * 成本: 平台真实模型 佣0.025%+最低5元+滑点0.1%(双边), ETF免印花税. 默认开启;
    --no-cost 关成本以对照 UP 的"裸收益"(UP 视频未提成本).

数据: etf_daily(OHLC+pct_chg) + index_daily(基准) | 本地离线, 无需 tushare
"""

import sqlite3
import argparse
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_monthly_rebalance import (
    COMMISSION_RATE, COMMISSION_MIN, SLIPPAGE_RATE,
    get_monthly_5th_trading_days,
)

DB = r'D:/tu-shareData/astock_daily.db'

# ── 事前固定候选池 (8只宽基 ETF, 不得为美化回测改动) ──
# code          name        类
POOL = {
    '510300.SH': ('沪深300',  'A'),
    '512100.SH': ('中证1000', 'A'),
    '159628.SZ': ('国证2000', 'A'),
    '159949.SZ': ('创业板50',  'A'),
    '588030.SH': ('科创100',  'A'),
    '159920.SZ': ('恒生',     'G'),
    '513100.SH': ('纳指',     'G'),
    '518880.SH': ('黄金',     'G'),
}
ALL_CODES = list(POOL.keys())

# 🔴 上市日闸门: 首交易日前该 ETF 不得进入候选池 (来自 etf_daily MIN(trade_date), 已核验)
LIST_DATE = {
    '510300.SH': '2012-05-28',
    '512100.SH': '2016-11-04',
    '159628.SZ': '2022-07-18',
    '159949.SZ': '2016-07-22',
    '588030.SH': '2023-09-15',
    '159920.SZ': '2012-10-22',
    '513100.SH': '2013-05-15',
    '518880.SH': '2013-07-29',
}
LIST_TS = {c: pd.Timestamp(d) for c, d in LIST_DATE.items()}

# 模块③ ATR 自适应止损: 每支 ETF 独立阈值 (UP 自陈, 按波动率设定)
STOP_THR = {
    '518880.SH': 0.05,   # 黄金
    '513100.SH': 0.05,   # 纳指
    '510300.SH': 0.05,   # 沪深300
    '159920.SZ': 0.05,   # 恒生
    '159628.SZ': 0.054,  # 国证2000
    '512100.SH': 0.059,  # 中证1000
    '159949.SZ': 0.075,  # 创业板50
    '588030.SH': 0.087,  # 科创100
}

INIT = 100000.0
START = '2017-01-01'
END = '2026-07-01'


# ── 成本模型 (与 v6 / 刘不牛 / 平台完全一致) ──
def calc_etf_fee(buy_or_sell, price, shares):
    amount = price * shares
    commission = max(amount * COMMISSION_RATE, COMMISSION_MIN)
    slippage = amount * SLIPPAGE_RATE
    return commission + slippage


# ── 数据加载: pct_chg 累乘构造复权净值 (规避 QDII/商品LOF 缺 adj_factor 的折算跳变) ──
def load_etf(ts_code):
    c = sqlite3.connect(DB)
    df = pd.read_sql_query(
        "SELECT trade_date, open, high, low, close, pre_close, pct_chg FROM etf_daily WHERE ts_code=? ORDER BY trade_date",
        c, params=(ts_code,))
    c.close()
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    df = df.set_index('trade_date').sort_index()
    ret = df['pct_chg'].fillna(0).astype(float) / 100.0
    nav = (1 + ret).cumprod()
    base = df['close'].iloc[0]
    df['close_adj'] = nav * (base / nav.iloc[0])
    for col, ac in [('open', 'open_adj'), ('high', 'high_adj'), ('low', 'low_adj')]:
        df[ac] = df['close_adj'] * (df[col] / df['close'])
    return df


# ── 经典光大 RSRS: 近 N 日 low~high OLS 斜率 β (β>1 支撑强于阻力) ──
# 向量化: beta = cov(low,high)/var(high)
def rsrs_df(high_df, low_df, W):
    hw = high_df.rolling(W)
    cov = (high_df * low_df).rolling(W).mean() - high_df.rolling(W).mean() * low_df.rolling(W).mean()
    var = high_df.rolling(W).var()
    beta = cov / var
    return beta


def zscore_rows(df):
    """横截面 z-score (每行对列): 可用列(非NaN)参与; std=0 或单列 -> 该行星等权置0;
    不可用列(原始NaN, 即未上市/历史不足) 保持 NaN 不被选。"""
    mu = df.mean(axis=1)
    sd = df.std(axis=1)  # ddof=1
    out = df.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)
    out = out.where(df.notna(), np.nan)              # 保留未上市 NaN
    out = out.mask(df.notna() & out.isna(), 0.0)     # 已上市但 std=0 -> 中性 0
    return out


def build_features(args, cal):
    """构建动量/RSRS/双均线特征。cal = A股共同交易日历。"""
    etf = {}
    for code in ALL_CODES:
        d = load_etf(code).reindex(cal).ffill()   # 未上市段保持 NaN (无 backfill)
        etf[code] = d
    close = pd.DataFrame({c: etf[c]['close_adj'] for c in ALL_CODES})
    openp = pd.DataFrame({c: etf[c]['open_adj'] for c in ALL_CODES})
    low = pd.DataFrame({c: etf[c]['low_adj'] for c in ALL_CODES})
    high = pd.DataFrame({c: etf[c]['high_adj'] for c in ALL_CODES})

    # 20日动量 = pct_change(20)
    mom = close.pct_change(args.momentum_window)
    # 经典 RSRS (low~high 斜率), 窗口 RSRS_W
    rsrs = rsrs_df(high, low, args.rsrs_window)
    # 双均线
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()

    # 横截面 z-score (未上市/历史不足 -> NaN 自然排除)
    z_mom = zscore_rows(mom)
    z_rsrs = zscore_rows(rsrs)

    return dict(close=close, open=openp, low=low,
                mom=mom, rsrs=rsrs, z_mom=z_mom, z_rsrs=z_rsrs,
                ma5=ma5, ma20=ma20)


def compute_target(F, date, current, args):
    """返回 target code (None=空仓)。含: 上市日闸门 + 双均线过滤 + RSRS复合排序。"""
    # 上市日闸门 (显式, 审计用)
    avail = [c for c in ALL_CODES if date >= LIST_TS[c]]
    if not avail:
        return None
    z_mom = F['z_mom'].loc[date]
    z_rsrs = F['z_rsrs'].loc[date]
    w_rsrs = args.rsrs_weight
    w_mom = 1.0 - w_rsrs

    cands = {}
    for c in avail:
        m = z_mom.get(c, np.nan)
        r = z_rsrs.get(c, np.nan)
        if np.isnan(m) or np.isnan(r):
            continue
        # 双均线过滤 (MA5 > MA20 才允许入选)
        if args.ma_filter:
            m5 = F['ma5'].loc[date, c]
            m20 = F['ma20'].loc[date, c]
            if np.isnan(m5) or np.isnan(m20) or not (m5 > m20):
                continue
        cands[c] = w_mom * m + w_rsrs * r

    if not cands:
        return None
    return max(cands, key=lambda c: cands[c])


def next_trading_day(cal, d):
    i = cal.get_loc(d)
    return cal[i + 1] if i + 1 < len(cal) else d


def backtest(F, args, trade_dates):
    """月度调仓: 每月第5交易日开盘成交(T-1信号)。单只持仓+日级止损(每支独立阈值)。"""
    monthly_5th = set(get_monthly_5th_trading_days(trade_dates))
    s0, e0 = START.replace('-', ''), END.replace('-', '')
    dates = [d for d in trade_dates if s0 <= d.strftime('%Y%m%d') <= e0]
    dates = pd.Index(sorted(dates))
    valid0 = F['close'].loc[dates].dropna(how='all').index[0]
    dates = dates[dates >= valid0]

    current = None
    cash = INIT
    shares = 0.0
    entry_price = 0.0
    nav = {}
    pos = {}
    trades = []
    pending = {}

    stop_mode = args.stop_mode
    for d in dates:
        if d in pending:
            tgt = pending.pop(d)
            if current is not None:
                p = F['open'].loc[d, current]
                if not np.isnan(p) and p > 0:
                    fee = calc_etf_fee('sell', p, shares)
                    cash = shares * p - fee
                    trades.append(('sell', current, d, p))
                shares = 0.0
                current = None
            if tgt is not None and cash > 0:
                p = F['open'].loc[d, tgt]
                if not np.isnan(p) and p > 0:
                    shares = cash / (p * (1 + COMMISSION_RATE + SLIPPAGE_RATE))
                    fee = calc_etf_fee('buy', p, shares)
                    cash = cash - (shares * p + fee)
                    current = tgt
                    entry_price = p
                    trades.append(('buy', tgt, d, p))
        if current is not None:
            cd = F['close'].loc[d, current]
            nav[d] = cash + (shares * cd if not np.isnan(cd) else 0.0)
        else:
            nav[d] = cash
        pos[d] = current
        # 日级止损 (最低价触发, 次日开盘卖)
        if current is not None and stop_mode != 'off':
            if stop_mode == 'atr':
                thr = STOP_THR.get(current, args.fixed_stop)
            else:  # fixed
                thr = args.fixed_stop
            low = F['low'].loc[d, current]
            if not np.isnan(low) and low <= entry_price * (1 - thr):
                pending[next_trading_day(dates, d)] = None
        # 月度决策
        if d in monthly_5th:
            tgt = compute_target(F, d, current, args)
            if tgt != current:
                pending[next_trading_day(dates, d)] = tgt

    nav_s = pd.Series(nav).sort_index().dropna()
    pos_s = pd.Series(pos).sort_index()
    return nav_s, trades, pos_s


def metrics(nav_s):
    ret = nav_s.iloc[-1] / nav_s.iloc[0] - 1
    yrs = (nav_s.index[-1] - nav_s.index[0]).days / 365.25
    ann = (nav_s.iloc[-1] / nav_s.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else np.nan
    daily = nav_s.pct_change().dropna()
    rf_daily = 0.02 / 252
    excess = daily - rf_daily
    sharpe = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0
    peak = nav_s.cummax()
    dd = (nav_s - peak) / peak
    maxdd = dd.min()
    calmar = ann / abs(maxdd) if maxdd < 0 else np.nan
    yearly = nav_s.resample('YE').last().pct_change().dropna()
    return dict(total=ret, ann=ann, sharpe=sharpe, maxdd=maxdd, calmar=calmar, yrs=yrs, yearly=yearly)


def benchmark_buyhold(ts_code, trade_dates):
    c = sqlite3.connect(DB)
    if ts_code[0] in '51':
        df = pd.read_sql_query("SELECT trade_date,close,pct_chg FROM etf_daily WHERE ts_code=? ORDER BY trade_date",
                               c, params=(ts_code,))
    else:
        df = pd.read_sql_query("SELECT trade_date,close,pct_chg FROM index_daily WHERE ts_code=? ORDER BY trade_date",
                               c, params=(ts_code,))
    c.close()
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    df = df.set_index('trade_date').sort_index()
    ret = df['pct_chg'].fillna(0).astype(float) / 100.0
    nav = (1 + ret).cumprod()
    nav = nav.reindex(trade_dates).ffill().loc[START:END].dropna()
    nav = nav / nav.iloc[0] * INIT
    return metrics(nav)


def benchmark_equal_weight(trade_dates):
    c = sqlite3.connect(DB)
    rets = {}
    for code in ALL_CODES:
        d = pd.read_sql_query("SELECT trade_date,pct_chg FROM etf_daily WHERE ts_code=? ORDER BY trade_date",
                              c, params=(code,))
        d['trade_date'] = pd.to_datetime(d['trade_date'], format='%Y%m%d')
        rets[code] = d.set_index('trade_date').sort_index()['pct_chg'].fillna(0).astype(float) / 100.0
    c.close
    rdf = pd.DataFrame(rets).reindex(trade_dates).ffill().loc[START:END].fillna(0)
    ret = rdf.mean(axis=1)
    nav = (1 + ret).cumprod() * INIT
    return metrics(nav)


def load_trade_dates():
    c = sqlite3.connect(DB)
    ts = pd.read_sql_query("SELECT DISTINCT trade_date FROM daily ORDER BY trade_date", c)['trade_date']
    c.close()
    return [pd.Timestamp(d) for d in ts]


def parse_args():
    p = argparse.ArgumentParser(description='RSRS 宽基 ETF 轮动 (BV1MS3d6qEZx)')
    p.add_argument('--momentum-window', type=int, default=20, help='动量窗口(日)')
    p.add_argument('--rsrs-window', type=int, default=20, help='RSRS窗口(日, UP未给, 自由参数)')
    p.add_argument('--rsrs-weight', type=float, default=0.40, help='RSRS权重(动量=1-此值); 0=纯动量')
    p.add_argument('--ma-filter', action='store_true', default=True, help='双均线 MA5>MA20 过滤')
    p.add_argument('--no-ma-filter', dest='ma_filter', action='store_false')
    p.add_argument('--stop-mode', choices=['atr', 'fixed', 'off'], default='atr',
                   help='atr=每支独立阈值(UP); fixed=统一阈值; off=关止损')
    p.add_argument('--fixed-stop', type=float, default=0.08, help='fixed 模式止损阈值')
    p.add_argument('--no-cost', action='store_true', help='关成本(对照UP裸收益)')
    p.add_argument('--start', default=START)
    p.add_argument('--end', default=END)
    p.add_argument('--baseline', action='store_true', help='额外打印"原始策略"(纯20日动量无过滤无止损)')
    return p.parse_args()


def fmt(m):
    return (f"累计 {m['total']*100:+.1f}% | 年化 {m['ann']*100:+.1f}% | 夏普 {m['sharpe']:.2f} | "
            f"回撤 {m['maxdd']*100:+.1f}% | 卡玛 {m['calmar']:.2f}")


def run_config(args, trade_dates, cal, F, label):
    nav_s, trades, pos_s = backtest(F, args, trade_dates)
    m = metrics(nav_s)
    empty = (pos_s.isna()).mean() if len(pos_s) else 0
    # 上市日闸门核对: 每支 ETF 首次被持有的日期 应 >= 其上市日
    first_hold = {}
    for d, c in pos_s.items():
        if pd.notna(c) and c not in first_hold:
            first_hold[c] = d
    gate_viol = {c: str(first_hold[c].date()) for c in first_hold
                 if first_hold[c] < LIST_TS[c]}
    print(f'\n[{label}] ' + fmt(m) + f' | 空仓率 {empty*100:.1f}% | 交易 {len(trades)}次 | 买入 {sum(1 for t in trades if t[0]=="buy")}次')
    if gate_viol:
        print(f'  ⚠️ 上市日闸门违规: {gate_viol}')
    else:
        print(f'  ✅ 上市日闸门 OK (8只首次持有均不早于上市日)')
    # 持仓分布
    dist = pos_s.value_counts(dropna=False)
    print('  持仓分布:', ' '.join(
        f"{POOL.get(k, ('空仓',))[0] if pd.notna(k) else '空仓'}:{v/len(pos_s)*100:.1f}%" for k, v in dist.items()))
    return m, first_hold


def main():
    global START, END
    args = parse_args()
    START, END = args.start, args.end

    trade_dates = load_trade_dates()
    cal = pd.Index(sorted(trade_dates))
    F = build_features(args, cal)

    print('=' * 72)
    print('RSRS 宽基 ETF 轮动 (faithful 复现)')
    print(f'  池子={len(ALL_CODES)}只(事前固定) | 起点 {START} 终点 {END}')
    print(f'  动量{args.momentum_window}日 | RSRS窗口{args.rsrs_window} | RSRS权重={args.rsrs_weight:.0%}(动量{1-args.rsrs_weight:.0%})')
    print(f'  双均线MA5>MA20={"开" if args.ma_filter else "关"} | 止损={args.stop_mode}'
          + (f'(统一{args.fixed_stop:.0%})' if args.stop_mode == "fixed" else "")
          + f' | 成本={"关" if args.no_cost else "开(佣0.025%+滑0.1%)"}')
    print('=' * 72)

    # faithful 主配置
    m, fh = run_config(args, trade_dates, cal, F, 'faithful(双均线+RSRS60/40+ATR止损)')

    # 基准
    mb = benchmark_buyhold('000300.SH', trade_dates)
    print('\n[基准] 沪深300买入持有: ' + fmt(mb))
    me = benchmark_equal_weight(trade_dates)
    print('[基准] 8只等权月度:    ' + fmt(me))

    # 年度
    print('\n年度收益:')
    for y in sorted(set(d.year for d in m['yearly'].index)):
        v = m['yearly'].get(pd.Timestamp(year=y, month=12, day=31))
        if v is not None:
            print(f'  {y}: {v*100:+.1f}%')

    if args.baseline:
        print('\n' + '-' * 72)
        print('原始策略对照 (纯20日动量选最强1只, 无MA过滤/无RSRS/无止损):')
        a0 = argparse.Namespace(**vars(args))
        a0.ma_filter = False
        a0.rsrs_weight = 0.0
        a0.stop_mode = 'off'
        run_config(a0, trade_dates, cal, F, '原始策略(纯动量)')


if __name__ == '__main__':
    main()
