"""
ETF轮动 V6 复现回测 (BV1uPu86sENG · UP:三门两月)
忠实复现视频策略 + 反过拟合双成本对比。

信号 : RSRS质量分(25日归一化收盘价对时间回归 slope×R²) + 20日动量
       每日横截面对4只ETF各自 z-score, 组合分 = 0.6×z(RSRS) + 0.4×z(动量)
闸门 : 前二门槛(组合第一 且 两单项均进前二) + 独立均线(各ETF站上自身MA)
风控 : -3%止损(入场价基准, 当日最低价触发, 次日开盘卖出) + cooldown(站回长期均线才买回)
执行 : 周五决策, 下周一开盘成交
成本 : 平台真实模型 佣0.025%+最低5元+滑点0.1%(双边), ETF免印花税 (复刻 run_etf_rotation.calc_etf_fee)

数据 : etf_daily(OHLC) + etf_adj_factor(后复权, 红利/创业50有, 纳指/黄金无→raw)
股票池 : 红利510880 / 创业板50 159949 / 纳指513100 / 黄金518880
"""
import sqlite3
import numpy as np
import pandas as pd

DB = r'D:/tu-shareData/astock_daily.db'

# ── 平台真实成本模型 (复用 run_monthly_rebalance 的费率常量, 与 run_etf_rotation.calc_etf_fee 完全一致) ──
# 佣0.025% + 最低5元 + 滑点0.1%(买/卖双边) + ETF免印花税
from run_monthly_rebalance import COMMISSION_RATE, COMMISSION_MIN, SLIPPAGE_RATE

def calc_etf_fee(buy_or_sell, price, shares):
    """ETF 交易费用: 佣0.025%+最低5元 + 滑点0.1%(双边), ETF免印花税。与平台 run_etf_rotation 一致。"""
    amount = price * shares
    commission = max(amount * COMMISSION_RATE, COMMISSION_MIN)
    slippage = amount * SLIPPAGE_RATE
    return commission + slippage

UNIVERSE = {
    '510880.SH': '红利',
    '159949.SZ': '创业板50',
    '513100.SH': '纳指',
    '518880.SH': '黄金',
}
# 独立均线(视频自陈). 红利给 5+20, 取全部均线均站上; cooldown用最长均线
MA = {'513100.SH': [10], '159949.SZ': [20], '510880.SH': [5, 20], '518880.SH': [3]}
RSRS_W = 25
MOM_W = 20
W_RSRS = 0.6
W_MOM = 0.4
STOP = -0.03
START = '2018-01-01'
END = '2026-07-01'
INIT = 100000.0
# 止损冷却恢复判定: 视频"止损后须重新站上该ETF长期均线才允许买回"→
# 冷却期内不轮动到其他ETF, 强制空仓。恢复均线用统一 RECOVER_MA(游标长度, 见 scan_recover)。
# 视频只说"长期均线"未给长度; 自陈空仓率16.6% 用此长度反校准(0% vs ~40% 两极端之间取中)。
# 持仓置信度门槛(组合分 z 单位): 最强ETF组合分须 > COMBO_MIN 才开仓, 否则空仓。
# 扫描显示此门槛会使空仓率飙升(1.0→62%, ≥1.5→100%), 故默认 0.0(关), 不参与校准。
COMBO_MIN = 0.0
RECOVER_MA = 20  # 冷却恢复判定均线长度(扫描: 3~30 空仓率均~40%, 取"长期均线"解读=20)


def load_etf(ts_code):
    c = sqlite3.connect(DB)
    df = pd.read_sql_query(
        "SELECT trade_date, open, high, low, close, pre_close FROM etf_daily WHERE ts_code=? ORDER BY trade_date",
        c, params=(ts_code,))
    af = pd.read_sql_query(
        "SELECT trade_date, adj_factor FROM etf_adj_factor WHERE ts_code=?", c, params=(ts_code,))
    c.close()
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    df = df.set_index('trade_date').sort_index()
    if not af.empty:
        af['trade_date'] = pd.to_datetime(af['trade_date'], format='%Y%m%d')
        af = af.set_index('trade_date').sort_index()
        df = df.join(af, how='left')
        df['adj_factor'] = df['adj_factor'].ffill().bfill()
        maxaf = df['adj_factor'].max()
        for col in ['open', 'high', 'low', 'close', 'pre_close']:
            df[col + '_adj'] = df[col] * df['adj_factor'] / maxaf
    else:
        for col in ['open', 'high', 'low', 'close', 'pre_close']:
            df[col + '_adj'] = df[col]
    return df


def rsrs_quality(x):
    if len(x) < RSRS_W:
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
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return b * r2


def build_features():
    series = {code: load_etf(code) for code in UNIVERSE}
    cal = pd.Index(sorted(set().union(*[set(df.index) for df in series.values()])))
    feat = {}
    ma_frames = {}
    for code, df in series.items():
        d = df.reindex(cal).ffill()
        d['rsrs'] = d['close_adj'].rolling(RSRS_W).apply(rsrs_quality, raw=True)
        d['mom'] = d['close_adj'].rolling(MOM_W + 1).apply(lambda x: x[-1] / x[0] - 1.0, raw=True)
        feat[code] = d
        for L in MA[code]:
            ma_frames[(code, L)] = d['close_adj'].rolling(L).mean()
        ma_frames[(code, RECOVER_MA)] = d['close_adj'].rolling(RECOVER_MA).mean()
    rsrs_df = pd.DataFrame({c: feat[c]['rsrs'] for c in UNIVERSE})
    mom_df = pd.DataFrame({c: feat[c]['mom'] for c in UNIVERSE})
    close_df = pd.DataFrame({c: feat[c]['close_adj'] for c in UNIVERSE})
    open_df = pd.DataFrame({c: feat[c]['open_adj'] for c in UNIVERSE})
    low_df = pd.DataFrame({c: feat[c]['low_adj'] for c in UNIVERSE})
    z_rsrs = rsrs_df.sub(rsrs_df.mean(axis=1), axis=0).div(rsrs_df.std(axis=1), axis=0).fillna(0)
    z_mom = mom_df.sub(mom_df.mean(axis=1), axis=0).div(mom_df.std(axis=1), axis=0).fillna(0)
    combo = W_RSRS * z_rsrs + W_MOM * z_mom
    rank_combo = combo.rank(axis=1, ascending=False)
    rank_rsrs = z_rsrs.rank(axis=1, ascending=False)
    rank_mom = z_mom.rank(axis=1, ascending=False)
    return dict(cal=cal, close_df=close_df, open_df=open_df, low_df=low_df,
                combo=combo, rank_combo=rank_combo,
                rank_rsrs=rank_rsrs, rank_mom=rank_mom, ma_frames=ma_frames)


def ma_gate_passes(F, code, date):
    for L in MA[code]:
        ma = F['ma_frames'][(code, L)].get(date, np.nan)
        close = F['close_df'].loc[date, code]
        if np.isnan(ma) or np.isnan(close) or close <= ma:
            return False
    return True


def compute_target(F, date, combo_min=COMBO_MIN):
    try:
        combo = F['combo'].loc[date]
        rc = F['rank_combo'].loc[date]
        rr = F['rank_rsrs'].loc[date]
        rm = F['rank_mom'].loc[date]
    except (KeyError, ValueError):
        return None
    if combo.isna().any() or rc.isna().any():
        return None
    best, bestc = None, -np.inf
    for code in UNIVERSE:
        if not ma_gate_passes(F, code, date):
            continue
        if rc[code] == 1 and rr[code] <= 2 and rm[code] <= 2:
            if combo[code] > bestc:
                bestc = combo[code]
                best = code
    # 持仓置信度门槛: 最强ETF组合分需显著高于横截面均值才开仓, 否则空仓
    if best is not None and combo[best] > combo_min:
        return best
    return None


def next_trading_day(cal, d):
    i = cal.get_loc(d)
    return cal[i + 1] if i + 1 < len(cal) else d


def next_monday_open(cal, d):
    cand = [x for x in cal[cal > d] if x.weekday() == 0]
    if cand:
        return cand[0]
    i = cal.get_loc(d)
    return cal[i + 1]


def backtest(F, cost=None, combo_min=COMBO_MIN, force_cash=True):
    """cost=None → 平台真实成本模型(calc_etf_fee); 传标量 → 旧简化双档(供 phase3/survivor 套件复用)。"""
    cal = F['cal']
    dates = cal[(cal >= START) & (cal <= END)]
    cooldown = {c: False for c in UNIVERSE}
    waiting = None  # 当前因止损进入冷却、须等其恢复长均线才允许回补的标的
    current = None
    cash = INIT
    shares = 0.0
    entry_price = 0.0  # 买入成交价, 止损基准(替代pre_close, 修复买入日误触发)
    nav = {}
    pending = {}
    trades = []
    pos = {}
    for d in dates:
        if d in pending:
            tgt = pending.pop(d)
            if current is not None:
                p = F['open_df'].loc[d, current]
                if not np.isnan(p) and p > 0:
                    if not np.isnan(p) and p > 0:
                        if cost is None:
                            fee = calc_etf_fee('sell', p, shares)
                            cash = shares * p - fee
                        else:
                            cash = shares * p * (1 - cost)
                        trades.append(('sell', current, d, p))
                    shares = 0.0
                    current = None
            if tgt is not None and not cooldown.get(tgt, False):
                p = F['open_df'].loc[d, tgt]
                if not np.isnan(p) and p > 0 and cash > 0:
                    if cost is None:
                        shares = cash / (p * (1 + COMMISSION_RATE + SLIPPAGE_RATE))
                        fee = calc_etf_fee('buy', p, shares)
                        cash = cash - (shares * p + fee)
                    else:
                        shares = cash * (1 - cost) / p
                        cash = 0.0
                    current = tgt
                    entry_price = p
                    trades.append(('buy', tgt, d, p))
        if current is not None:
            cd = F['close_df'].loc[d, current]
            nav[d] = cash + (shares * cd if not np.isnan(cd) else 0.0)
        else:
            nav[d] = cash
        pos[d] = current
        if current is not None:
            low = F['low_df'].loc[d, current]
            if not np.isnan(low) and low <= entry_price * (1 + STOP):
                nxt = next_trading_day(cal, d)
                pending[nxt] = None
                cooldown[current] = True
                waiting = current  # 止损后等待该标的恢复长均线才回补
        if d.weekday() == 4:
            if force_cash and waiting is not None and cooldown.get(waiting, False):
                tgt = None  # 冷却期内不轮动到其他ETF, 强制空仓(忠实版)
            else:
                tgt = compute_target(F, d, combo_min)
                if tgt is not None and cooldown.get(tgt, False):
                    tgt = None
            nm = next_monday_open(cal, d)
            pending[nm] = tgt
        for code in UNIVERSE:
            if cooldown.get(code):
                cd = F['close_df'].loc[d, code]
                rma = F['ma_frames'][(code, RECOVER_MA)].get(d, np.nan)
                if not np.isnan(cd) and not np.isnan(rma) and cd > rma:
                    cooldown[code] = False
                    if waiting == code:
                        waiting = None
    nav_s = pd.Series(nav).sort_index().dropna()
    pos_s = pd.Series(pos).sort_index()  # 保留None(空仓日)以统计空仓率
    return nav_s, trades, pos_s


def metrics(nav_s):
    ret = nav_s.iloc[-1] / nav_s.iloc[0] - 1
    yrs = (nav_s.index[-1] - nav_s.index[0]).days / 365.25
    ann = (nav_s.iloc[-1] / nav_s.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else np.nan
    daily = nav_s.pct_change().dropna()
    rf_daily = 0.02 / 252  # 无风险利率~2%/年, 超额收益扣除
    excess = daily - rf_daily
    sharpe = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0
    peak = nav_s.cummax()
    dd = (nav_s - peak) / peak
    maxdd = dd.min()
    calmar = ann / abs(maxdd) if maxdd < 0 else np.nan
    yearly = nav_s.resample('YE').last().pct_change().dropna()
    return dict(total=ret, ann=ann, sharpe=sharpe, maxdd=maxdd,
                calmar=calmar, yrs=yrs, yearly=yearly, n=len(nav_s))


def scan_recover():
    """扫描 RECOVER_MA(冷却恢复均线长度), 找空仓率最接近视频自陈 16.6% 的长度。
    waiting 止损冷却强制空仓机制下, 恢复均线越短→空仓越少。"""
    print('=' * 64)
    print('闸门扫描: RECOVER_MA(冷却恢复均线) vs 空仓率 / 累计收益 (平台真实成本: 佣0.025%+滑点0.1%)')
    print('=' * 64)
    print(f'{"RECOVER_MA":>11} {"空仓率":>9} {"累计收益":>10} {"年化":>8} {"最大回撤":>10} {"夏普":>6}')
    results = []
    for rm in [3, 5, 7, 10, 15, 20, 30]:
        global RECOVER_MA
        RECOVER_MA = rm
        F = build_features()
        nav_s, trades, pos_s = backtest(F)
        m = metrics(nav_s)
        er = float((pos_s.isna()).mean()) if len(pos_s) else 0.0
        results.append((rm, er, m))
        print(f'{rm:>11} {er*100:>8.2f}% {m["total"]*100:>+9.1f}% {m["ann"]*100:>+7.1f}% '
              f'{m["maxdd"]*100:>+9.1f}% {m["sharpe"]:>6.2f}')
    best = min(results, key=lambda r: abs(r[1] - 0.166))
    print(f'\n→ 选 RECOVER_MA={best[0]} (空仓率 {best[1]*100:.2f}%, 最接近 16.6%)')
    return best[0]


def main():
    global RECOVER_MA
    # RECOVER_MA 已锁定=20(扫描证明 3~30 空仓率均~40%, 取"长期均线"解读)。
    # 视频自陈 16.6% 在字面规则下不可忠实战现(0%轮动 / ~40%强制空仓 双峰), 见 scan_recover 证据。
    F = build_features()
    print('\n' + '=' * 64)
    print(f'ETF轮动 V6 复现回测 (4只后视镜池, 2018-2026, RECOVER_MA={RECOVER_MA}, COMBO_MIN={COMBO_MIN})')
    print('=' * 64)
    pos_all = None
    # ── 主角: 平台真实成本模型 ──
    nav_s, trades, pos_s = backtest(F, combo_min=COMBO_MIN)  # cost=None → 平台模型
    m = metrics(nav_s)
    empty_rate = (pos_s.isna()).mean() if len(pos_s) else 0
    pos_all = pos_s
    print(f'\n--- 平台真实成本 (佣0.025%+最低5元+滑点0.1%, ETF免印花税) ---')
    print(f'  累计收益 : {m["total"]*100:+.2f}%  (视频宣称 +1886%)')
    print(f'  年化收益 : {m["ann"]*100:+.2f}%  (视频宣称 +43.56%)')
    print(f'  最大回撤 : {m["maxdd"]*100:+.2f}%  (视频宣称 -21.6%)')
    print(f'  夏普     : {m["sharpe"]:.2f}  (视频宣称 1.49)')
    print(f'  卡玛     : {m["calmar"]:.2f}  (视频宣称 2.02)')
    print(f'  交易次数 : {len(trades)}')
    print(f'  空仓率   : {empty_rate*100:.2f}%  (视频宣称 16.6%)')
    yrs = sorted(set(d.year for d in m["yearly"].index))
    print('  年度收益 :')
    for y in yrs:
        v = m["yearly"].get(pd.Timestamp(year=y, month=12, day=31))
        if v is not None:
            print(f'    {y}: {v*100:+.2f}%')
    # ── 旧双档成本对照 (参考) ──
    for cost, label in [(0.0004, '视频成本0.04%'), (0.0015, '真实成本0.15%')]:
        nav_s, trades, pos_s = backtest(F, cost, combo_min=COMBO_MIN)
        m = metrics(nav_s)
        empty_rate = (pos_s.isna()).mean() if len(pos_s) else 0
        print(f'\n--- {label} (单边{cost*100:.2f}%, 旧简化双档) ---')
        print(f'  累计收益 : {m["total"]*100:+.2f}%')
        print(f'  年化收益 : {m["ann"]*100:+.2f}%')
        print(f'  最大回撤 : {m["maxdd"]*100:+.2f}%')
        print(f'  夏普     : {m["sharpe"]:.2f}')
        print(f'  空仓率   : {empty_rate*100:.2f}%')
    dist = pos_all.value_counts(dropna=False)
    print('  持仓分布 (平台成本):')
    for k, v in dist.items():
        name = UNIVERSE.get(k, '空仓') if k is not None else '空仓'
        print(f'    {name}: {v / len(pos_all) * 100:.1f}%')
    print('\n注: 复权口径 纳指/黄金用raw(无现金分红); 持仓分布与成本无关. 平台成本下未对期末持仓计平仓费(与旧口径一致).')


if __name__ == '__main__':
    main()
