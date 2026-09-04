# ❌ INVALIDATED — 量化刘不牛「173倍」ETF 轮动 (BV1BqMX6TEx2)
# 判定：经严谨回测 + 月度调仓 bug 修复后，策略彻底证伪（关止损 -64.1%，远逊等权 +103.5%；
#       阶段3“池子洗清”结论被推翻为 0.0%ile condemned）。
# 历史：曾因 get_monthly_5th_trading_days bug（每年仅 2 个调仓日）一度给出虚假 +120.7% 被错误采纳，已纠正。
# 处置：本文件按清理惯例标记为无效策略归档，不作为可实盘/可复现参考。详见 etf_rebalance_bugfix_reverify.md。
# ============================================================

# -*- coding: utf-8 -*-
"""
ETF轮动 (刘不牛 / BV1BqMX6TEx2) 忠实复现 + 开关版回测
========================================================
视频核心: 标题写"173倍", 内容在解构这个数为什么不可信 + 公开三个隐性bug +
四步验证法(触发统计/样本外/压力测试/模拟盘)。

本脚本目标: faithful 还原"规则选1只ETF的动量轮动", 并把每个模块做成 **开关**(argparse),
便于后续做模块归因消融与触发统计(反过拟合电池)。**不追求复现173倍**——那是
后视镜选池+样本内过拟合的产物(见 plan_etf_liubuni_rotation.md)。

信号(UP自陈规则):
  - 持仓 1 只 ETF (测过2-3只换仓增收益降)            -> --position-size (默认1)
  - 动量窗口 25日                                     -> --momentum-window (默认25)
  - R² 趋势线拟合度过滤 (窗口未给, 自由参数)           -> --r2-filter / --r2-threshold
  - 保留区间 90%: 当前持仓达第一名90%分数就不换        -> --keep-threshold (0=总换,1=永不换)
  - 止损 5% 固定                                      -> --stop-loss (<-0.9=关)
  - A股走弱(沪深300/小盘/创业板/中证A500 中≥3破10日线) -> 切海外/商品池  --weak-a-share

诚实边界(必读):
  * 候选池事前固定 10 只 (A股2 + 海外4 + 商品2 + 避险2), 不得为美化回测改池子。
  * 日经513520(2019-06)/创业50 159949(2016-07)上市晚 -> 全池诚实起点 2019-07。
  * 纳指513100等 QDII 无 etf_adj_factor -> 用 raw 价 (同 v6 处理)。
  * 调仓频率 UP 未明说 -> 用月度第5交易日开盘 (平台范式, 降摩擦), 标注为假设。
  * R² 阈值 / "切海外池"具体成分 视频未给 -> 默认 r2 关、global 子池=排除A股两只, 标注假设。

数据: etf_daily(OHLC) + etf_adj_factor(后复权) + index_daily(走弱判定)
成本: 平台真实模型 佣0.025%+最低5元+滑点0.1%(双边), ETF免印花税 (复用 run_monthly_rebalance)
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

# ── 事前固定候选池 (10只, 四大类诚实多样化; 不得为美化回测改动) ──
# A股子类(走强时才参与) / 全球子类(走弱时切到这里)
POOL = {
    # code        name        class
    '510880.SH': ('红利',     'A'),
    '159949.SZ': ('创业板50', 'A'),
    '513100.SH': ('纳指',     'G'),
    '513500.SH': ('标普500',  'G'),
    '513520.SH': ('日经',     'G'),
    '159920.SZ': ('恒生',     'G'),
    '501018.SH': ('原油',     'G'),
    '518880.SH': ('黄金',     'G'),
    '511010.SH': ('国债',     'G'),
    '511880.SH': ('货币',     'G'),
}
ALL_CODES = list(POOL.keys())
A_CODES = [c for c, (_, k) in POOL.items() if k == 'A']
G_CODES = [c for c, (_, k) in POOL.items() if k == 'G']

# A股走弱判定用的4只指数 (沪深300 / 小盘[中证1000代] / 创业板指 / 中证A500[中证500代, 库缺A500])
WEAK_INDEX = {
    '000300.SH': '沪深300',
    '000852.SH': '中证1000',   # 代"小盘"
    '399006.SZ': '创业板指',
    '000905.SH': '中证500',     # 代"中证A500"(库缺 A500, 标注)
}
WEAK_MA = 10  # 跌破几日均线判走弱

INIT = 100000.0
START = '2019-07-01'   # 全池诚实共同起点 (日经2019-06上市)
END = '2026-07-01'


# ── 成本模型 (与 v6 / 平台 run_etf_rotation 完全一致) ──
def calc_etf_fee(buy_or_sell, price, shares):
    amount = price * shares
    commission = max(amount * COMMISSION_RATE, COMMISSION_MIN)
    slippage = amount * SLIPPAGE_RATE
    return commission + slippage


# ── 数据加载: 统一用 pct_chg 累乘构造复权净值 (规避 QDII/商品LOF 分红与份额折算跳变) ──
def load_etf(ts_code):
    c = sqlite3.connect(DB)
    df = pd.read_sql_query(
        "SELECT trade_date, open, high, low, close, pre_close, pct_chg FROM etf_daily WHERE ts_code=? ORDER BY trade_date",
        c, params=(ts_code,))
    c.close()
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    df = df.set_index('trade_date').sort_index()
    # 复权净值序列 (pct_chg 已含分红/折算的日收益)
    ret = df['pct_chg'].fillna(0).astype(float) / 100.0
    nav = (1 + ret).cumprod()
    base = df['close'].iloc[0]
    df['close_adj'] = nav * (base / nav.iloc[0])
    # open/high/low 按当日 raw 比例映射到复权净值, 保留日内高低关系
    for col, ac in [('open', 'open_adj'), ('high', 'high_adj'), ('low', 'low_adj')]:
        df[ac] = df['close_adj'] * (df[col] / df['close'])
    return df


def load_index(ts_code):
    c = sqlite3.connect(DB)
    df = pd.read_sql_query(
        "SELECT trade_date, close FROM index_daily WHERE ts_code=? ORDER BY trade_date",
        c, params=(ts_code,))
    c.close()
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    return df.set_index('trade_date').sort_index()['close']


# ── R² 趋势拟合度 (25日归一化收盘价对时间回归的 R²) ──
def r2_only(x):
    if len(x) < 2:
        return np.nan
    xmin, xmax = x.min(), x.max()
    if xmax - xmin < 1e-12:
        return np.nan
    y = (x - xmin) / (xmax - xmin)
    t = np.arange(1, len(y) + 1, dtype=float)
    b, a = np.polyfit(t, y, 1)
    yhat = a + b * t
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


def build_features(args, cal):
    """构建动量/R²/指数状态特征。cal = 共同交易日历 (A股)。"""
    # ETF 价 reindex 到 cal
    etf = {}
    for code in ALL_CODES:
        d = load_etf(code).reindex(cal).ffill()
        etf[code] = d
    close = pd.DataFrame({c: etf[c]['close_adj'] for c in ALL_CODES})
    openp = pd.DataFrame({c: etf[c]['open_adj'] for c in ALL_CODES})
    low = pd.DataFrame({c: etf[c]['low_adj'] for c in ALL_CODES})

    # 25日动量 = close_adj(t)/close_adj(t-W) - 1
    W = args.momentum_window
    mom = close.shift(W) / close - 1.0   # 用 shift(W) 得 W 日收益率
    # R² (可选过滤用, 始终计算供统计)
    r2 = close.rolling(W).apply(r2_only, raw=True)

    # 指数 10日线状态 (走弱判定)
    weak_below = pd.DataFrame({name: (load_index(code) > load_index(code).rolling(WEAK_MA).mean())
                               for code, name in WEAK_INDEX.items()})
    weak_below = weak_below.reindex(cal).ffill().fillna(False).astype(bool)
    # True=站上≥10日线 (未走弱); 跌破数 = 未在10日线上 的指数个数
    weak_count = (~weak_below).sum(axis=1)

    return dict(close=close, open=openp, low=low, mom=mom, r2=r2, weak_count=weak_count)


def compute_target(F, date, current, args):
    """返回 target code (None=空仓)。含 90%保留 / A股走弱切池 / R²过滤 开关。"""
    mom = F['mom'].loc[date]
    r2 = F['r2'].loc[date]
    weak = F['weak_count'].loc[date] if date in F['weak_count'].index else 0

    # A股走弱 -> 候选池排除 A股两只
    if args.weak_a_share and weak >= args.weak_n_down:
        pool = G_CODES
    else:
        pool = ALL_CODES

    # 候选分数: 动量 (R² 过滤为剔除)
    scores = {}
    for code in pool:
        m = mom.get(code, np.nan)
        if np.isnan(m):
            continue
        if args.r2_filter:
            rv = r2.get(code, np.nan)
            if np.isnan(rv) or rv < args.r2_threshold:
                continue
        scores[code] = m

    if not scores:
        return None

    best = max(scores, key=lambda c: scores[c])
    best_score = scores[best]

    # 90% 保留: 当前持仓若达第一名 90% 分数则不换
    if (current is not None and current in scores
            and args.keep_threshold > 0.0):
        cur_score = scores[current]
        if cur_score >= best_score * args.keep_threshold:
            return current
    return best


def next_trading_day(cal, d):
    i = cal.get_loc(d)
    return cal[i + 1] if i + 1 < len(cal) else d


def backtest(F, args, trade_dates):
    """月度调仓: 每月第5交易日开盘成交(T-1信号)。单只持仓+日级止损。"""
    monthly_5th = set(get_monthly_5th_trading_days(trade_dates))
    s0, e0 = START.replace('-', ''), END.replace('-', '')
    dates = [d for d in trade_dates if s0 <= d.strftime('%Y%m%d') <= e0]
    dates = pd.Index(sorted(dates))
    # 仅保留有数据的区间 (close 首行非全 nan)
    valid0 = F['close'].loc[dates].dropna(how='all').index[0]
    dates = dates[dates >= valid0]

    current = None
    cash = INIT
    shares = 0.0
    entry_price = 0.0
    nav = {}
    pos = {}
    trades = []
    pending = {}  # date -> target(None=卖不买)

    stop = args.stop_loss
    for d in dates:
        # 1. 执行 pending (开盘价)
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
        # 2. nav (收盘价估值)
        if current is not None:
            cd = F['close'].loc[d, current]
            nav[d] = cash + (shares * cd if not np.isnan(cd) else 0.0)
        else:
            nav[d] = cash
        pos[d] = current
        # 3. 日级止损 (最低价触发, 次日开盘卖; 当日仍按持仓估值, 次日step1正常卖出记账)
        if current is not None and stop > -0.9:
            low = F['low'].loc[d, current]
            if not np.isnan(low) and low <= entry_price * (1 + stop):
                pending[next_trading_day(dates, d)] = None
        # 4. 月度决策 (T日信号, T+1开盘成交)
        if d in monthly_5th:
            tgt = compute_target(F, d, current, args)
            if tgt != current:   # 同只不重复交易, 挂下一交易日开盘
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
    """指数/ETF 买入持有基准 (pct_chg 复权净值)。"""
    c = sqlite3.connect(DB)
    if ts_code[0] in '51':   # ETF 代码以 5/1 开头; 指数以 0/3 开头
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


def benchmark_equal_weight(trade_dates, args):
    """全池等权月度再平衡基准 (pct_chg 复权净值)。"""
    c = sqlite3.connect(DB)
    rets = {}
    for code in ALL_CODES:
        d = pd.read_sql_query("SELECT trade_date,pct_chg FROM etf_daily WHERE ts_code=? ORDER BY trade_date",
                              c, params=(code,))
        d['trade_date'] = pd.to_datetime(d['trade_date'], format='%Y%m%d')
        rets[code] = d.set_index('trade_date').sort_index()['pct_chg'].fillna(0).astype(float) / 100.0
    c.close()
    rdf = pd.DataFrame(rets).reindex(trade_dates).ffill().loc[START:END].fillna(0)
    ret = rdf.mean(axis=1)
    nav = (1 + ret).cumprod() * INIT
    return metrics(nav)


def load_trade_dates():
    """A股交易日历 (daily 表 distinct trade_date)。"""
    c = sqlite3.connect(DB)
    ts = pd.read_sql_query("SELECT DISTINCT trade_date FROM daily ORDER BY trade_date", c)['trade_date']
    c.close()
    return [pd.Timestamp(d) for d in ts]


def parse_args():
    p = argparse.ArgumentParser(description='刘不牛 ETF 轮动 (开关版)')
    p.add_argument('--momentum-window', type=int, default=25)
    p.add_argument('--r2-filter', action='store_true', help='开启 R² 趋势过滤')
    p.add_argument('--r2-threshold', type=float, default=0.0)
    p.add_argument('--keep-threshold', type=float, default=0.90, help='0=总换仓,1=永不换')
    p.add_argument('--stop-loss', type=float, default=-0.05, help='<=-0.9 关闭')
    p.add_argument('--weak-a-share', action='store_true', default=True, help='A股走弱切全球池')
    p.add_argument('--weak-n-down', type=int, default=3, help='几只指数破10日线判走弱')
    p.add_argument('--position-size', type=int, default=1)
    p.add_argument('--start', default=START)
    p.add_argument('--end', default=END)
    p.add_argument('--sweep', action='store_true', help='跑一组开关消融')
    return p.parse_args()


def fmt(m):
    return (f"累计 {m['total']*100:+.1f}% | 年化 {m['ann']*100:+.1f}% | 夏普 {m['sharpe']:.2f} | "
            f"回撤 {m['maxdd']*100:+.1f}% | 卡玛 {m['calmar']:.2f}")


def main():
    global START, END
    args = parse_args()
    START, END = args.start, args.end

    trade_dates = load_trade_dates()
    cal = pd.Index(sorted(trade_dates))
    F = build_features(args, cal)

    print('=' * 70)
    print('刘不牛 ETF 轮动 (faithful 默认开关全开)')
    print(f'  池子={len(ALL_CODES)}只(事前固定) | 起点 {START} 终点 {END}')
    print(f'  动量{args.momentum_window}日 | R²过滤={"开" if args.r2_filter else "关"} | '
          f'90%保留={args.keep_threshold} | 止损={args.stop_loss} | '
          f'A股走弱切池={"开" if args.weak_a_share else "关"}')
    print('=' * 70)

    nav_s, trades, pos_s = backtest(F, args, trade_dates)
    m = metrics(nav_s)
    empty = (pos_s.isna()).mean() if len(pos_s) else 0
    print('\n[策略] ' + fmt(m) + f' | 空仓率 {empty*100:.1f}% | 交易 {len(trades)}次')

    # 基准
    mb = benchmark_buyhold('000300.SH', trade_dates)
    print('[基准] 沪深300买入持有: ' + fmt(mb))
    me = benchmark_equal_weight(trade_dates, args)
    print('[基准] 全池等权月度:   ' + fmt(me))

    # 年度
    print('\n年度收益:')
    for y in sorted(set(d.year for d in m['yearly'].index)):
        v = m['yearly'].get(pd.Timestamp(year=y, month=12, day=31))
        if v is not None:
            print(f'  {y}: {v*100:+.1f}%')
    # 持仓分布
    dist = pos_s.value_counts(dropna=False)
    print('\n持仓分布:')
    for k, v in dist.items():
        name = POOL.get(k, ('空仓',))[0] if k is not None else '空仓'
        print(f'  {name}: {v/len(pos_s)*100:.1f}%')

    if args.sweep:
        print('\n' + '-' * 70)
        print('开关消融 (累计收益对比):')
        print(f'  {"配置":<26s} {"累计":>9s} {"年化":>8s} {"夏普":>6s} {"空仓率":>8s}')
        base = m['total']
        print(f'  {"faithful(默认)":<26s} {m["total"]*100:>+8.1f}% {m["ann"]*100:>+7.1f}% {m["sharpe"]:>6.2f} {empty*100:>7.1f}%')
        # 关 90% 保留
        a2 = argparse.Namespace(**vars(args)); a2.keep_threshold = 0.0
        n2, _, p2 = backtest(F, a2, trade_dates); e2 = (p2.isna()).mean()
        m2 = metrics(n2)
        print(f'  {"关90%保留":<24s} {m2["total"]*100:>+8.1f}% {m2["ann"]*100:>+7.1f}% {m2["sharpe"]:>6.2f} {e2*100:>7.1f}%')
        # 关 A股走弱切池
        a3 = argparse.Namespace(**vars(args)); a3.weak_a_share = False
        n3, _, p3 = backtest(F, a3, trade_dates); e3 = (p3.isna()).mean()
        m3 = metrics(n3)
        print(f'  {"关A股走弱切池":<22s} {m3["total"]*100:>+8.1f}% {m3["ann"]*100:>+7.1f}% {m3["sharpe"]:>6.2f} {e3*100:>7.1f}%')
        # 关止损
        a4 = argparse.Namespace(**vars(args)); a4.stop_loss = -1.0
        n4, _, p4 = backtest(F, a4, trade_dates); e4 = (p4.isna()).mean()
        m4 = metrics(n4)
        print(f'  {"关止损":<26s} {m4["total"]*100:>+8.1f}% {m4["ann"]*100:>+7.1f}% {m4["sharpe"]:>6.2f} {e4*100:>7.1f}%')


if __name__ == '__main__':
    main()
