# -*- coding: utf-8 -*-
"""
双动量 ETF 轮动 · 批判式复现与证伪
源视频：BV1FoMk6CEcv《12年5.8倍，最大回撤仅18%！上班族每月操作5分钟的ETF轮动策略》
        UP 主「上班做量化的鳄鱼」（462粉新号，评论"回复厉害私发源码"漏斗）
        自报：12年总收益583%(年化~16%)/最大回撤18.38%/beta0.23/胜率47.2%/盈亏比2.276

本脚本目的：不复刻他的数字，而是用平台可得数据，把他的"组件"拆成 5 个可证伪的对照实验。

五个实验（与 SKILL.md §5.29 待办一一对应）
  E1 诚实时间线：pool=pit（point-in-time 动态上市池，无幸存者偏差）
                 vs pool=fixed（固定全池，从最后一只上市日起算 = 后视选池）
  E2 权重：weight=equal vs weight=riskparity（风险平价/ERC 是否真加分）
  E3 空仓规则：abs_momentum=on vs off（绝对动量是不是回撤 30%→18% 的真来源）
  E4 成本：cost=real（ETF 免印花税 + 佣金万2.5 + 滑点千1）vs cost=zero
  E5 参数高原：--grid 扫 window×topn 的 ±50% 扰动，看是不是孤峰

打分（三种，视频原描述 + 学术标准 + 朴素对照）
  wr_r2    = 净胜率 × R²          ← 视频原话"上涨胜率 × 尼核度系数(稳定性)"
             net_wr = 2*win_rate-1 ∈[-1,1]，可正可负 → 才能支撑"分数>0才持仓"
  slope_r2 = 年化斜率 × R²        ← 学术标准的"风险调整动量"
  ret      = 窗口累计收益率        ← 朴素动量对照

无前视保证
  · 打分窗口严格止于调仓日前一交易日（T-1）收盘
  · 调仓日 T 以 **开盘价** 成交
  · 上市日 = 该 ETF 在 etf_daily 的第一个交易日，且要求已上市 ≥ min_listed 个交易日
  · 复权：adj = close × etf_adj_factor（分红再投口径），成交金额与真实金额严格相等
    （证明：shares_real × raw_open = (shares_adj·adjf) × (adj_open/adjf) = shares_adj × adj_open）

用法
  python run_dual_momentum_etf.py --diagnose
  python run_dual_momentum_etf.py --pool pit --score wr_r2 --abs-momentum on --weight equal --cost real
  python run_dual_momentum_etf.py --grid
"""
import argparse
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_monthly_rebalance import get_conn, INIT_CAPITAL  # noqa: E402

DB = 'D:/tu-shareData/astock_daily.db'

# ETF 免印花税
COMMISSION_RATE = 0.00025
COMMISSION_MIN = 5.0
SLIPPAGE_RATE = 0.001
STAMP_DUTY_ETF = 0.0

# 视频 13 只标的 → 平台可得映射（✗ = 平台无数据）
# 上市日取自 etf_daily 实际首个交易日，是本脚本 E1 的核心证据
POOL = {
    '510300.SH': ('沪深300', '宽基'),
    '159915.SZ': ('创业板', '宽基'),
    '510500.SH': ('中证500', '宽基'),
    '512100.SH': ('中证1000', '宽基'),
    '510050.SH': ('上证50', '宽基'),
    '159901.SZ': ('深证100', '宽基'),
    '510880.SH': ('红利', '宽基'),
    '518880.SH': ('黄金', '商品'),
    '159928.SZ': ('消费', '行业'),
    '512010.SH': ('医药', '行业'),
    '512660.SH': ('军工', '行业'),
    '512760.SH': ('芯片', '行业'),
    '515050.SH': ('通信', '行业'),
    '515030.SH': ('新能车', '行业'),
    '515790.SH': ('光伏', '行业'),
    '512690.SH': ('酒', '行业'),
    '512880.SH': ('证券', '行业'),
    '588000.SH': ('科创50', '宽基'),
}
# 排除：511990 华宝添益(货币ETF，非风险资产)、511010/511260(国债，作为 defensive 选件)
#       510330/515800(与 510300/510500 跟踪同指数)、159949(与 159915 重叠)、
#       563300/588190/159766(上市太短，不足一个周期)

DEFENSIVE = {'none': None, 'bond': '511260.SH'}
BENCH = '000300.SH'


# ── 数据层 ────────────────────────────────────────────────────────────
def load_etf_panel(codes):
    """返回 (adj_close_df, adj_open_df, first_date_map)。adj = price × adj_factor。"""
    con = get_conn() if hasattr(get_conn, '__call__') else None
    if con is None:
        con = sqlite3.connect(DB)
    ph = ','.join('?' * len(codes))
    px = pd.read_sql(
        f"SELECT ts_code, trade_date, open, close FROM etf_daily "
        f"WHERE ts_code IN ({ph}) ORDER BY ts_code, trade_date", con,
        params=tuple(codes))
    af = pd.read_sql(
        f"SELECT ts_code, trade_date, adj_factor FROM etf_adj_factor "
        f"WHERE ts_code IN ({ph}) ORDER BY ts_code, trade_date", con,
        params=tuple(codes))
    try:
        con.close()
    except Exception:
        pass

    px['trade_date'] = px['trade_date'].astype(int)
    af['trade_date'] = af['trade_date'].astype(int)
    df = px.merge(af, on=['ts_code', 'trade_date'], how='left')
    df['adj_factor'] = df.groupby('ts_code')['adj_factor'].ffill().bfill().fillna(1.0)

    close = df.pivot(index='trade_date', columns='ts_code', values='close').sort_index()
    openp = df.pivot(index='trade_date', columns='ts_code', values='open').sort_index()
    factor = df.pivot(index='trade_date', columns='ts_code', values='adj_factor').sort_index()

    adj_close = close * factor
    adj_open = openp * factor
    # 开盘价为 0/NaN 的日期回退到收盘价（不影响信号，只影响成交价估计）
    adj_open = adj_open.where(adj_open > 0).fillna(adj_close)

    first = {}
    traded = {}
    for c in close.columns:
        s = close[c].dropna()
        if len(s):
            first[c] = int(s.index[0])
            traded[c] = len(s)
    return adj_close, adj_open, first, traded


def load_calendar(start, end):
    con = sqlite3.connect(DB)
    cal = pd.read_sql(
        "SELECT DISTINCT trade_date FROM index_daily WHERE ts_code=? "
        "AND trade_date>=? AND trade_date<=? ORDER BY trade_date",
        con, params=(BENCH, start, end))
    bench = pd.read_sql(
        "SELECT trade_date, close FROM index_daily WHERE ts_code=? "
        "AND trade_date>=? AND trade_date<=? ORDER BY trade_date",
        con, params=(BENCH, start, end))
    con.close()
    cal['trade_date'] = cal['trade_date'].astype(int)
    bench['trade_date'] = bench['trade_date'].astype(int)
    return cal['trade_date'].tolist(), bench.set_index('trade_date')['close']


def monthly_first_days(calendar):
    s = pd.Series(calendar)
    df = pd.DataFrame({'d': s})
    df['ym'] = df['d'].astype(str).str[:6]
    return df.groupby('ym')['d'].min().tolist()


# ── 打分 ──────────────────────────────────────────────────────────────
def score_window(win, mode):
    """win: 长度 L 的 adj_close 序列（升序，最后一个是 T-1 收盘）。返回 (score, vol)。"""
    if len(win) < max(5, len(win) // 2):
        return None, None
    r = np.diff(np.log(win))
    if len(r) < 3:
        return None, None
    vol = float(np.std(r, ddof=1)) if len(r) > 1 else None
    if mode == 'ret':
        return float(win[-1] / win[0] - 1.0), vol
    t = np.arange(len(win), dtype=float)
    y = np.log(win)
    slope, intercept = np.polyfit(t, y, 1)
    yhat = slope * t + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    if mode == 'slope_r2':
        return float(slope * 242 * r2), vol     # 年化斜率 × R²
    if mode == 'wr_r2':
        wr = float(np.mean(r > 0))              # 上涨天数占比
        return float((2.0 * wr - 1.0) * r2), vol
    raise ValueError(mode)


# ── 回测 ──────────────────────────────────────────────────────────────
def run_backtest(adj_close, adj_open, first, traded, calendar, rebal_days,
                 bench_close, args):
    # 用面板实际列（而非 POOL 全集）——支持 --exclude 剔除标的
    codes = list(adj_close.columns)
    ac = adj_close.reindex(index=calendar)
    ao = adj_open.reindex(index=calendar)
    ac = ac[codes]

    # 上市可用日（point-in-time）：
    #   回测起点之前就已上市的 → 起点即可用（历史已足够长）
    #   回测起点之后才上市的 → 等其自身成交满 min_listed 个交易日
    # 注意：不能用 calendar 索引计算，否则起点后的"老 ETF"会被误判为未上市满 60 天。
    avail_from = {}
    for c in codes:
        if c not in first:
            avail_from[c] = 10 ** 9
            continue
        if first[c] < calendar[0]:
            avail_from[c] = calendar[0]
        else:
            s = ac[c].dropna()
            if len(s):
                avail_from[c] = int(s.index[min(args.min_listed, len(s) - 1)])
            else:
                avail_from[c] = 10 ** 9

    # fixed 池模式：所有标的从"最后一只上市日"起统一可用（= 后视选池，复刻视频的
    # 幸存者偏差）。净值序列必须同步截断到该日，否则前期纯现金会污染胜率/盈亏比。
    if args.pool == 'fixed':
        last_avail = max(avail_from.values())
        avail_from = {c: last_avail for c in codes}
        keep = [i for i, d in enumerate(calendar) if d >= last_avail]
        if keep:
            k0 = keep[0]
            calendar = calendar[k0:]
            ac = ac.iloc[k0:]
            ao = ao.iloc[k0:]

    cash = float(INIT_CAPITAL)           # 现金账户（空仓时全部停留在此）
    shares = {c: 0.0 for c in codes}
    nav_series = []
    cost_total = 0.0
    turn_total = 0.0
    n_rebal = 0
    hold_counts = []
    trades_rows = []

    date_pos = {d: i for i, d in enumerate(calendar)}
    rebal_valid = [d for d in rebal_days if date_pos.get(d, 0) >= args.window]
    rebal_set = set(rebal_valid)

    for i, d in enumerate(calendar):
        if d in rebal_set:
            pos = date_pos[d]
            # 1) 打分：窗口严格止于 T-1（前 window 个交易日，不含 T）
            lo = pos - args.window
            scores, vols = {}, {}
            for c in codes:
                if avail_from.get(c, 10 ** 9) > d:
                    continue
                seg = ac[c].iloc[lo:pos].dropna().values  # [T-window, T-1]
                if len(seg) < args.window:
                    continue
                s, v = score_window(seg, args.score)
                if s is None or (v is None or v <= 0):
                    continue
                if args.abs_momentum == 'on' and s <= 0:
                    continue
                scores[c] = s
                vols[c] = v

            # 2) 目标权重
            target, sel = {}, []
            if scores:
                ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:args.topn]
                sel = [c for c, _ in ranked]
                if args.weight == 'riskparity':
                    inv = {c: 1.0 / vols[c] for c in sel}
                    tot = sum(inv.values())
                    target = {c: inv[c] / tot for c in sel}
                else:
                    target = {c: 1.0 / len(sel) for c in sel}
            hold_counts.append(len(sel))

            # 3) 调仓日开盘价成交
            px = {c: ao[c].iloc[pos] for c in codes}
            cur_val = {c: (shares[c] * px[c]) if not pd.isna(px[c]) else 0.0
                       for c in codes}
            wealth = cash + sum(cur_val.values())
            amount = 0.0
            for c in codes:
                if pd.isna(px[c]):
                    continue
                delta_val = wealth * target.get(c, 0.0) - cur_val[c]
                if abs(delta_val) < 1e-6:
                    continue
                p = px[c]
                dsh = delta_val / p
                amt = abs(delta_val)
                cst = 0.0
                if args.cost == 'real':
                    cst = (max(amt * COMMISSION_RATE, COMMISSION_MIN)
                           + amt * SLIPPAGE_RATE)
                cost_total += cst
                amount += amt
                shares[c] += dsh
                cash -= delta_val          # 买入付钱 / 卖出收钱
                cash -= cst                # 成本从现金扣
                trades_rows.append({
                    'trade_date': d, 'ts_code': c,
                    'action': 'buy' if delta_val > 0 else 'sell',
                    'price': round(p, 4), 'shares': round(dsh, 2),
                    'amount': round(amt, 2), 'cost': round(cst, 2),
                    'reason': 'rot_in' if delta_val > 0 else 'rot_out',
                })
            if wealth > 0:
                turn_total += amount / wealth
            n_rebal += 1

        mv = sum(shares[c] * ac[c].iloc[i] for c in codes
                 if not pd.isna(ac[c].iloc[i]))
        nav_series.append((d, cash + mv, 1 if mv > 1.0 else 0))

    nav_df = pd.DataFrame(nav_series, columns=['trade_date', 'nav', 'hold']) \
        .set_index('trade_date')
    return {
        'nav': nav_df['nav'], 'hold': nav_df['hold'],
        'cost_total': cost_total,
        'turnover': turn_total / max(n_rebal, 1),
        'n_rebal': n_rebal,
        'avg_hold': float(np.mean(hold_counts)) if hold_counts else 0.0,
        'pct_empty': float(np.mean([h == 0 for h in hold_counts])) if hold_counts else 0.0,
        'trades': pd.DataFrame(trades_rows),
        'start': calendar[0], 'end': calendar[-1],
    }


def metrics(nav, bench_close, res):
    nav = nav.dropna()
    if len(nav) < 10:
        return {}
    nav = nav.copy()
    nav.index = pd.to_datetime(nav.index.astype(str), format='%Y%m%d')
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    total = nav.iloc[-1] / nav.iloc[0] - 1.0
    ann = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1.0 if yrs > 0 else np.nan
    dd = nav / nav.cummax() - 1.0
    mdd = float(dd.min())
    r = nav.pct_change().dropna()
    sharpe = float(r.mean() / r.std() * np.sqrt(242)) if r.std() > 0 else np.nan
    # 月度胜率/盈亏比：仅统计「当月有持仓」的月份。
    # 空仓月份月度收益恒为 0，若计入会被算成"不涨"，系统性拉低胜率（fixed 池尤其严重）。
    hold = res.get('hold')
    md_all = nav.resample('ME').last().pct_change().dropna()
    if hold is not None:
        h = hold.copy()
        h.index = pd.to_datetime(h.index.astype(str), format='%Y%m%d')
        hm = h.resample('ME').mean()
        active = (hm.reindex(md_all.index).fillna(0) > 0.5)
        md = md_all[active]
        n_mon_active = int(active.sum())
    else:
        md, n_mon_active = md_all, len(md_all)
    win = float((md > 0).mean()) if len(md) else np.nan
    gp = float(md[md > 0].sum()) if (md > 0).any() else 0.0
    gl = float(-md[md < 0].sum()) if (md < 0).any() else 0.0
    pl = gp / gl if gl > 0 else np.nan
    # 基准
    b = bench_close.copy()
    b.index = pd.to_datetime(b.index.astype(str), format='%Y%m%d')
    b = b.reindex(nav.index).ffill().dropna()
    b_total = b.iloc[-1] / b.iloc[0] - 1.0 if len(b) > 1 else np.nan
    b_ann = (b.iloc[-1] / b.iloc[0]) ** (1 / yrs) - 1.0 if len(b) > 1 and yrs > 0 else np.nan
    # beta
    br = b.pct_change().dropna()
    j = r.to_frame('r').join(br.to_frame('b'), how='inner')
    beta = float(np.cov(j['r'], j['b'])[0][1] / np.var(j['b'])) if len(j) > 30 and np.var(j['b']) > 0 else np.nan
    return {
        'total': total, 'ann': ann, 'mdd': mdd, 'sharpe': sharpe,
        'win': win, 'pl': pl, 'beta': beta,
        'bench_total': b_total, 'bench_ann': b_ann,
        'excess_ann': (ann - b_ann) if (ann == ann and b_ann == b_ann) else np.nan,
        'cost': res['cost_total'], 'turnover': res['turnover'],
        'n_rebal': res['n_rebal'], 'avg_hold': res['avg_hold'],
        'pct_empty': res['pct_empty'], 'n_mon_active': n_mon_active,
        'start': res['start'], 'end': res['end'], 'years': yrs,
    }


def show(tag, m):
    if not m:
        print(f'{tag:<46} 无有效结果')
        return
    print(f'{tag:<46} {m["start"]}-{m["end"]} {m["years"]:.1f}y | '
          f'总{m["total"]*100:>8.2f}% 年化{m["ann"]*100:>6.2f}% | '
          f'回撤{m["mdd"]*100:>7.2f}% 夏普{m["sharpe"]:>5.2f} | '
          f'超额{m["excess_ann"]*100:>7.2f}pp | '
          f'胜率{m["win"]*100:>5.1f}% 盈亏比{m["pl"]:>5.2f}')


def slice_metrics(res, bench_close, start, end):
    """把一个回测结果的净值/持仓按子区间切片后重算指标。

    注意：分段指标的 cost/turnover 仍是全局值，分段报告中不展示这两列。
    """
    nav = res['nav']
    nav = nav[(nav.index >= start) & (nav.index <= end)]
    if len(nav) < 10:
        return {}
    r2 = dict(res)
    r2['nav'] = nav
    if res.get('hold') is not None:
        r2['hold'] = res['hold'].reindex(nav.index)
    r2['start'] = nav.index[0]
    r2['end'] = nav.index[-1]
    return metrics(nav, bench_close, r2)


# ── 主流程 ─────────────────────────────────────────────────────────────
def diagnose():
    codes = list(POOL.keys())
    con = sqlite3.connect(DB)
    ph = ','.join('?' * len(codes))
    d = pd.read_sql(
        f"SELECT ts_code, MIN(trade_date) d0, MAX(trade_date) d1, COUNT(*) n "
        f"FROM etf_daily WHERE ts_code IN ({ph}) GROUP BY ts_code", con,
        params=tuple(codes))
    con.close()
    d['d0'] = d['d0'].astype(int)
    d['d1'] = d['d1'].astype(int)
    d['名称'] = d['ts_code'].map(lambda c: POOL.get(c, ('?', '?'))[0])
    d['类别'] = d['ts_code'].map(lambda c: POOL.get(c, ('?', '?'))[1])
    d = d.sort_values('d0')
    print('=== E1 关键证据：平台可得标的的「真实上市日」===')
    print(d[['ts_code', '名称', '类别', 'd0', 'd1', 'n']].to_string(index=False))
    print()
    late = d[d['d0'] > 20140101]
    print(f'视频自称「2014 年起 12 年回测」。平台可得标的中，'
          f'2014-01-01 之后才上市的有 {len(late)}/{len(d)} 只：')
    for _, r in late.iterrows():
        print(f"  {r['ts_code']} {r['名称']:<8} 上市日 {r['d0']}  ← 2014 年根本不存在")
    print()
    print('视频 13 只里平台无数据、无法验证的：纳指ETF / 恒生科技ETF / 创新药ETF / 计算机ETF')
    print('（恒生科技指数 2020-07 才发布；A股首只纳指ETF 2013、创新药ETF 2020、计算机ETF 2019）')
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='20140101')
    ap.add_argument('--end', default='20260901')
    ap.add_argument('--window', type=int, default=34)
    ap.add_argument('--topn', type=int, default=3)
    ap.add_argument('--score', default='wr_r2', choices=['wr_r2', 'slope_r2', 'ret'])
    ap.add_argument('--weight', default='equal', choices=['equal', 'riskparity'])
    ap.add_argument('--abs-momentum', default='on', choices=['on', 'off'])
    ap.add_argument('--cost', default='real', choices=['real', 'zero'])
    ap.add_argument('--pool', default='pit', choices=['pit', 'fixed'])
    ap.add_argument('--min-listed', type=int, default=60)
    ap.add_argument('--diagnose', action='store_true')
    ap.add_argument('--grid', action='store_true')
    ap.add_argument('--suite', action='store_true', help='跑完整 5 个对照实验')
    ap.add_argument('--exclude', default='',
                    help='剔除标的，逗号分隔（如 518880.SH 剔除黄金）')
    ap.add_argument('--seg-audit', action='store_true',
                    help='分段审计：等权 vs 风险平价 在三个子区间的表现')
    ap.add_argument('--outdir', default='data/results/dual_momentum_etf')
    args = ap.parse_args()

    if args.diagnose:
        diagnose()
        return

    codes = list(POOL.keys())
    if args.exclude.strip():
        ex = [c.strip() for c in args.exclude.split(',') if c.strip()]
        codes = [c for c in codes if c not in ex]
        print(f'已剔除 {len(ex)} 只：{" ".join(ex)}')
        if not codes:
            print('标的池为空，退出')
            return
    adj_close, adj_open, first, traded = load_etf_panel(codes)
    calendar, bench_close = load_calendar(args.start, args.end)
    rebal_days = monthly_first_days(calendar)
    print(f'标的池 {len(codes)} 只 | 交易日 {len(calendar)} | 调仓次数 {len(rebal_days)}')

    if args.grid:
        print('\n=== E5 参数高原：window × topn 网格（±50% 扰动）===')
        rows = []
        for w in [17, 26, 34, 43, 51]:
            for t in [2, 3, 4, 5]:
                a = argparse.Namespace(**vars(args))
                a.window, a.topn = w, t
                r = run_backtest(adj_close, adj_open, first, traded, calendar,
                                 rebal_days, bench_close, a)
                m = metrics(r['nav'], bench_close, r)
                rows.append({'window': w, 'topn': t, 'ann': m['ann'] * 100,
                             'mdd': m['mdd'] * 100, 'sharpe': m['sharpe'],
                             'total': m['total'] * 100})
        g = pd.DataFrame(rows)
        piv = g.pivot(index='window', columns='topn', values='ann')
        print('\n年化收益(%) 矩阵 window × topn：')
        print(piv.round(2).to_string())
        piv2 = g.pivot(index='window', columns='topn', values='mdd')
        print('\n最大回撤(%) 矩阵：')
        print(piv2.round(2).to_string())
        print(f"\n年化 标准差 {g['ann'].std():.2f}pp | "
              f"最好 {g['ann'].max():.2f}% | 最差 {g['ann'].min():.2f}% | "
              f"极差 {g['ann'].max()-g['ann'].min():.2f}pp")
        os.makedirs(args.outdir, exist_ok=True)
        g.to_csv(os.path.join(args.outdir, 'param_grid.csv'),
                 index=False, encoding='utf-8-sig')
        print(f'\n落盘 {args.outdir}/param_grid.csv')
        return

    if args.seg_audit:
        segs = [('2014-2018', 20140101, 20181231),
                ('2019-2021', 20190101, 20211231),
                ('2022-2026', 20220101, 20260831)]
        print('\n=== A2 分段审计：等权 vs 风险平价（ERC）===')
        print('判据：ERC 必须在每一段都跑赢等权，才算稳健正贡献；'
              '只在某段赢 = 样本依赖，不接。\n')
        runs = {}
        for tag, kw in [('等权', dict(weight='equal')),
                        ('风险平价', dict(weight='riskparity'))]:
            a = argparse.Namespace(**{**vars(args), **kw})
            runs[tag] = run_backtest(adj_close, adj_open, first, traded,
                                     calendar, rebal_days, bench_close, a)
            show(f'全区间 {tag}', metrics(runs[tag]['nav'], bench_close, runs[tag]))

        print()
        rows = []
        for name, s0, s1 in segs:
            r = {'区间': name}
            for tag in ['等权', '风险平价']:
                m = slice_metrics(runs[tag], bench_close, s0, s1)
                if not m:
                    r[f'{tag}年化%'] = np.nan
                    r[f'{tag}回撤%'] = np.nan
                    continue
                r[f'{tag}年化%'] = round(m['ann'] * 100, 2)
                r[f'{tag}回撤%'] = round(m['mdd'] * 100, 2)
                r[f'{tag}夏普'] = round(m['sharpe'], 3)
            if r['等权年化%'] == r['等权年化%']:
                r['ERC年化差pp'] = round(r['风险平价年化%'] - r['等权年化%'], 2)
                # 回撤是负数，用绝对值相减：正 = ERC 跌得更少
                r['ERC回撤改善pp'] = round(
                    abs(r['等权回撤%']) - abs(r['风险平价回撤%']), 2)
                r['ERC夏普差'] = round(r['风险平价夏普'] - r['等权夏普'], 3)
                r['收益胜'] = '✅' if r['ERC年化差pp'] > 0 else '❌'
                r['回撤胜'] = '✅' if r['ERC回撤改善pp'] > 0 else '❌'
                r['夏普胜'] = '✅' if r['ERC夏普差'] > 0 else '❌'
            else:
                r['ERC年化差pp'] = np.nan
                r['收益胜'] = r['回撤胜'] = r['夏普胜'] = '-'
            rows.append(r)
        seg = pd.DataFrame(rows)
        print(seg.to_string(index=False))
        n = len(segs)
        nwin = int((seg['收益胜'] == '✅').sum())
        ndd = int((seg['回撤胜'] == '✅').sum())
        nsp = int((seg['夏普胜'] == '✅').sum())
        print(f'\nERC 收益跑赢 {nwin}/{n} 段 | 回撤更小 {ndd}/{n} 段 | 夏普更高 {nsp}/{n} 段')
        if nwin == n:
            print('→ 每段都赢，稳健正贡献，可接。')
        else:
            print('→ 存在收益跑输的段。需结合回撤/夏普判定：'
                  '回撤全段改善但夏普未改善 = 用收益换回撤的交换，非 alpha。')
        os.makedirs(args.outdir, exist_ok=True)
        tag_suffix = f"_ex{'_'.join(c.strip() for c in args.exclude.split(',') if c.strip())}" \
            if args.exclude.strip() else ''
        p = os.path.join(args.outdir, f'seg_audit{tag_suffix}.csv')
        seg.to_csv(p, index=False, encoding='utf-8-sig')
        print(f'落盘 {p}')
        return

    if args.suite:
        print('\n=== 双动量 ETF 轮动 · 五组对照实验 ===')
        base = dict(start=args.start, end=args.end, window=args.window, topn=args.topn,
                    score=args.score, weight=args.weight, abs_momentum=args.abs_momentum,
                    cost=args.cost, pool=args.pool, min_listed=args.min_listed)
        b = lambda **kw: argparse.Namespace(**{**base, **kw})
        rows = []
        exp = ''

        def run_and_show(tag, kw, note=None):
            a = b(**kw)
            rr = run_backtest(adj_close, adj_open, first, traded, calendar,
                              rebal_days, bench_close, a)
            m = metrics(rr['nav'], bench_close, rr)
            show(tag, m)
            if note:
                print(f'{"":<46} {note(m)}')
            if m:
                rows.append({
                    '实验': exp, '配置': tag,
                    '起': m['start'], '止': m['end'], '年数': round(m['years'], 2),
                    '总收益%': round(m['total'] * 100, 2),
                    '年化%': round(m['ann'] * 100, 2),
                    '最大回撤%': round(m['mdd'] * 100, 2),
                    '夏普': round(m['sharpe'], 3),
                    '年化超额pp': round(m['excess_ann'] * 100, 2),
                    '月度胜率%': round(m['win'] * 100, 1),
                    '盈亏比': round(m['pl'], 3),
                    'beta': round(m['beta'], 3) if m['beta'] == m['beta'] else np.nan,
                    '累计成本元': round(m['cost'], 0),
                    '单次换手%': round(m['turnover'] * 100, 1),
                    '调仓次数': m['n_rebal'],
                    '平均持仓数': round(m['avg_hold'], 2),
                    '空仓月占比%': round(m['pct_empty'] * 100, 1),
                })
            return m

        print('\n--- E1 诚实时间线：PIT 动态池 vs 固定全池（幸存者偏差）---')
        exp = 'E1 时间线'
        for tag, kw in [('PIT 动态上市池（无幸存者偏差）', dict(pool='pit')),
                        ('固定全池自最后一只上市起（后视）', dict(pool='fixed'))]:
            run_and_show(tag, kw)

        print('\n--- E2 权重：等权 vs 风险平价（ERC）---')
        exp = 'E2 权重'
        for tag, kw in [('等权 top3', dict(weight='equal')),
                        ('风险平价 top3', dict(weight='riskparity'))]:
            run_and_show(tag, kw)

        print('\n--- E3 空仓规则：绝对动量 on vs off ---')
        exp = 'E3 空仓'
        for tag, kw in [('绝对动量空仓 ON（分数>0 才买）', dict(abs_momentum='on')),
                        ('绝对动量空仓 OFF（永远满仓 top3）', dict(abs_momentum='off'))]:
            run_and_show(tag, kw, note=lambda m: (
                f'空仓月份占比 {m["pct_empty"]*100:.1f}% | '
                f'平均持仓数 {m["avg_hold"]:.2f}'))

        print('\n--- E4 成本敏感性：真实成本 vs 零成本 ---')
        exp = 'E4 成本'
        for tag, kw in [('真实成本（免印花税+佣金万2.5+滑点千1）', dict(cost='real')),
                        ('零成本（复刻"月频成本可忽略"假设）', dict(cost='zero'))]:
            run_and_show(tag, kw, note=lambda m: (
                f'累计成本 {m["cost"]:,.0f} 元 | '
                f'单次调仓双边换手 {m["turnover"]*100:.1f}%'))

        print('\n--- E5 打分函数对照（视频原描述 vs 学术标准 vs 朴素）---')
        exp = 'E5 打分'
        for tag, kw in [('wr_r2  净胜率×R²（视频原描述）', dict(score='wr_r2')),
                        ('slope_r2 年化斜率×R²（学术标准）', dict(score='slope_r2')),
                        ('ret   纯窗口收益率（朴素动量）', dict(score='ret'))]:
            run_and_show(tag, kw)

        if rows:
            os.makedirs(args.outdir, exist_ok=True)
            p = os.path.join(args.outdir, 'suite_results.csv')
            pd.DataFrame(rows).to_csv(p, index=False, encoding='utf-8-sig')
            print(f'\n落盘 {p}')
        return

    r = run_backtest(adj_close, adj_open, first, traded, calendar,
                     rebal_days, bench_close, args)
    m = metrics(r['nav'], bench_close, r)
    print()
    show('单次运行', m)
    print(f'累计成本 {m["cost"]:,.0f} | 单次换手 {m["turnover"]*100:.1f}% | '
          f'调仓 {m["n_rebal"]} 次 | 平均持仓 {m["avg_hold"]:.2f} | '
          f'空仓月占比 {m["pct_empty"]*100:.1f}%')


if __name__ == '__main__':
    main()
