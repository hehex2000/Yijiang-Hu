# -*- coding: utf-8 -*-
"""
backtest_volume_signal.py
事件研究：量价信号对 T+1/T+2/T+3 前向收益的影响
维度：放量/缩量(量比阈值) × 价格位置(低/中/高) × 市场状态(沪深300 regime)
控制：市值中性化 + 行业中性化（控制组合法）
样本：2015-2025 全历史（远大于视频的 2024 至今 2.5 年）

方法论纪律（对标平台"因子先实测再入zoo"）：
- 信号定义只用历史数据（量比=当日额/过去N日均额，shift(1)避免含当日）
- 前向收益是 ex-post 测量，非预测，事件研究标准做法，无前视
- 中性化：用同(交易日,市值档,行业)全部股票的前向收益均值作为控制组合，事件收益减控制=净alpha
- 全部内存化向量计算，单次SQL拉全量，不用逐股循环
"""
import sqlite3, numpy as np, pandas as pd, time, os

DB = 'D:/tu-shareData/astock_daily.db'
START, END = 20150101, 20251231
VOL_WIN = 20        # 量比窗口（过去N日成交额均线）
POS_WIN = 60        # 价格位置窗口（60日 trailing return 分低/中/高）
MKT_WIN = 20        # 市场状态窗口（沪深300 trailing return）
MKT_THR = 0.03      # 市场状态阈值：20日收益>3%上涨市，<-3%下跌市，否则震荡
LIST_CUTOFF = '2014-01-01'   # 仅用 2014 前上市股票，保证有完整历史


def load():
    c = sqlite3.connect(DB, timeout=30)
    daily = pd.read_sql(
        f"SELECT ts_code,trade_date,close,high,amount FROM daily "
        f"WHERE trade_date BETWEEN {START} AND {END}", c)
    dbasic = pd.read_sql(
        f"SELECT ts_code,trade_date,circ_mv FROM daily_basic "
        f"WHERE trade_date BETWEEN {START} AND {END}", c)
    sb = pd.read_sql("SELECT ts_code,name,industry,list_date FROM stock_basic", c)
    hs = pd.read_sql(
        f"SELECT trade_date,close FROM index_daily "
        f"WHERE ts_code='000300.SH' AND trade_date BETWEEN {START} AND {END} "
        f"ORDER BY trade_date", c)
    c.close()
    return daily, dbasic, sb, hs


def build_features(daily, dbasic, sb, hs):
    # 类型
    daily['trade_date'] = daily['trade_date'].astype(int)
    dbasic['trade_date'] = dbasic['trade_date'].astype(int)
    daily = daily.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

    # ST 过滤（name 含 ST，先 fillna 防 NaN）
    sb['is_st'] = sb['name'].fillna('').str.contains('ST')
    st_set = set(sb.loc[sb['is_st'], 'ts_code'])
    daily = daily[~daily['ts_code'].isin(st_set)].copy()

    # 上市时间过滤（保证有完整历史）
    sb['list_date'] = pd.to_datetime(sb['list_date'], errors='coerce')
    ok = set(sb.loc[sb['list_date'] <= pd.Timestamp(LIST_CUTOFF), 'ts_code'])
    daily = daily[daily['ts_code'].isin(ok)].copy()

    g = daily.groupby('ts_code', sort=False)

    # 量比：当日成交额 / 过去 VOL_WIN 日均额（shift(1) 不含当日 → 无前视）
    daily['amt_ma'] = g['amount'].transform(
        lambda x: x.shift(1).rolling(VOL_WIN, min_periods=10).mean())
    daily['vol_ratio'] = daily['amount'] / daily['amt_ma']

    # 当日涨跌（vs 昨收）
    daily['ret'] = g['close'].transform(lambda x: x.pct_change())

    # 前向收益 T+1/2/3（ex-post 测量）
    daily['fwd1'] = g['close'].transform(lambda x: x.shift(-1) / x - 1)
    daily['fwd2'] = g['close'].transform(lambda x: x.shift(-2) / x - 1)
    daily['fwd3'] = g['close'].transform(lambda x: x.shift(-3) / x - 1)

    # 价格位置：60日 trailing return（shift(1) 不含当日）
    daily['pos_ret'] = g['close'].transform(
        lambda x: x.shift(1) / x.shift(1 + POS_WIN) - 1)

    # 横盘突破 proxy：20日收益波动率低(横盘) + 收盘价近20日高(突破)
    daily['ret_std20'] = g['ret'].transform(
        lambda x: x.shift(1).rolling(20, min_periods=10).std())
    daily['hi20'] = g['high'].transform(
        lambda x: x.shift(1).rolling(20, min_periods=10).max())

    # 市值（merge_asof 向后取最近，避免 daily_basic 稀疏导致全 NaN）
    dbasic = dbasic.sort_values('trade_date')
    ds = daily.sort_values('trade_date')
    merged = pd.merge_asof(
        ds, dbasic, on='trade_date', by='ts_code',
        direction='backward', suffixes=('', '_db'))
    daily = daily.merge(
        merged[['ts_code', 'trade_date', 'circ_mv']],
        on=['ts_code', 'trade_date'], how='left')

    # 行业
    daily = daily.merge(
        sb[['ts_code', 'industry']], on='ts_code', how='left')

    # 市值档（每日横截面 tercile）
    daily['size_t'] = daily.groupby('trade_date')['circ_mv'].transform(
        lambda x: pd.qcut(x, 3, labels=[0, 1, 2], duplicates='drop'))

    # 市场状态（沪深300）
    hs['trade_date'] = hs['trade_date'].astype(int)
    hs = hs.sort_values('trade_date')
    hs['mkt_ret'] = hs['close'].shift(1) / hs['close'].shift(1 + MKT_WIN) - 1
    hs['mkt_state'] = np.where(hs['mkt_ret'] > MKT_THR, 'up',
                       np.where(hs['mkt_ret'] < -MKT_THR, 'down', 'side'))
    daily = daily.merge(hs[['trade_date', 'mkt_state']], on='trade_date', how='left')

    # 沪深300 前向收益（市值调整用）
    hs['hf_ret3'] = hs['close'].shift(-3) / hs['close'] - 1
    daily = daily.merge(hs[['trade_date', 'hf_ret3']], on='trade_date', how='left')
    daily['fwd3_adj'] = daily['fwd3'] - daily['hf_ret3']  # 市值调整(市场调整)后

    # 价格位置分档（固定阈值，近似视频"低/中/高"）
    daily['pos_band'] = np.where(daily['pos_ret'] < -0.05, 'low',
                        np.where(daily['pos_ret'] <= 0.15, 'mid', 'high'))

    # 量价组合
    up = daily['ret'] > 0
    vol_up = daily['vol_ratio'] > 2.0
    daily['combo'] = np.select(
        [vol_up & up, (~vol_up) & up, vol_up & (~up), (~vol_up) & (~up)],
        ['放量上涨', '缩量上涨', '放量下跌', '缩量下跌'], default='nan')

    # 横盘突破放量
    daily['range_bound'] = daily['ret_std20'] <= daily.groupby('trade_date')['ret_std20'].transform(
        lambda x: x.quantile(0.33))
    daily['breakout'] = daily['close'] >= 0.98 * daily['hi20']
    daily['chanlun_volup'] = (vol_up & daily['range_bound'] & daily['breakout'])

    # 清洗 NaN
    need = ['vol_ratio', 'fwd1', 'fwd2', 'fwd3', 'pos_band', 'size_t',
            'industry', 'mkt_state', 'fwd3_adj']
    daily = daily.dropna(subset=need)
    daily['industry'] = daily['industry'].fillna('NA')
    return daily


def neutralize(events, full, cols):
    """控制组合法：事件收益减 同(交易日,cols)全样本均值，得净alpha。"""
    ctrl = full.groupby(['trade_date'] + cols)['fwd3'].mean().rename('ctrl')
    ev = events.merge(ctrl.reset_index(), on=['trade_date'] + cols, how='left')
    ev['fwd3_neut'] = ev['fwd3'] - ev['ctrl']
    return ev


def stats(mask, full, label, yearly=False):
    ev = full[mask]
    n = len(ev)
    if n == 0:
        return f"{label:>14s}: n=0"
    f1, f2, f3 = ev['fwd1'].mean(), ev['fwd2'].mean(), ev['fwd3'].mean()
    adj3 = ev['fwd3_adj'].mean()
    evn = neutralize(ev, full, ['size_t', 'industry'])
    neut3 = evn['fwd3_neut'].mean()
    line = (f"{label:>14s}: n={n:>8d} | T+1={f1*100:+.3f}% "
            f"T+2={f2*100:+.3f}% T+3={f3*100:+.3f}% "
            f"| 市值调整T+3={adj3*100:+.3f}% 中性化T+3={neut3*100:+.3f}%")
    if not yearly:
        return line
    # 年份切片
    ev = ev.copy()
    ev['yr'] = (ev['trade_date'] // 10000).astype(str)
    rows = []
    for y, e in ev.groupby('yr'):
        rows.append((y, len(e), e['fwd1'].mean() * 100,
                     e['fwd2'].mean() * 100, e['fwd3'].mean() * 100))
    return line, rows


def main():
    t0 = time.time()
    print(f"[load] 读取 {START}-{END} 全量日线...", flush=True)
    daily, dbasic, sb, hs = load()
    print(f"[feat] 构造特征(向量化)... rows={len(daily)}", flush=True)
    D = build_features(daily, dbasic, sb, hs)
    print(f"[done] 特征构造完成 用时 {time.time()-t0:.1f}s  有效事件池={len(D):,}", flush=True)

    print("\n========== ① 四大量价组合 · 整体（对标视频）==========")
    for combo in ['放量上涨', '缩量上涨', '放量下跌', '缩量下跌']:
        print(stats(D['combo'] == combo, D, combo))

    print("\n========== ② 放量上涨 × 价格位置（视频: 中间最强 +0.71%）==========")
    up_vol = D['combo'] == '放量上涨'
    for band in ['low', 'mid', 'high']:
        print(stats(up_vol & (D['pos_band'] == band), D, f'放量上涨·{band}'))

    print("\n========== ③ 放量上涨 × 市场状态（视频: 上涨市 +1.02%）==========")
    for st in ['up', 'down', 'side']:
        print(stats(up_vol & (D['mkt_state'] == st), D, f'放量上涨·{st}'))

    print("\n========== ④ 放量上涨 × 年份切片（视频: 2024+ / 2025≈0 / 2026−）==========")
    _, rows = stats(up_vol, D, '放量上涨', yearly=True)
    print(f"{'年份':>6s} {'n':>9s} {'T+1':>9s} {'T+2':>9s} {'T+3':>9s}")
    for y, n, a, b, c in rows:
        print(f"{y:>6s} {n:>9d} {a:+8.3f}% {b:+8.3f}% {c:+8.3f}%")

    print("\n========== ⑤ 横盘突破再放量（视频反直觉: T+3 −0.39%）==========")
    print(stats(D['chanlun_volup'], D, '横盘突破放量'))

    print("\n========== ⑥ 放量阈值敏感性（1.5× vs 2.0×）==========")
    for mult in [1.5, 2.0]:
        m = (D['vol_ratio'] > mult) & (D['ret'] > 0)
        print(stats(m, D, f'放量上涨(>{mult}x)'))

    print(f"\n[总用时] {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
