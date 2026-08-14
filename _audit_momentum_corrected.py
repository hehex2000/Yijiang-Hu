# -*- coding: utf-8 -*-
"""
独立、可审计的"修正版"动量回测 —— 对照原 run_monthly_rebalance 的 +1250% 幻影。

修正点：
1. 宇宙：全A股时点宇宙（每只股用其 list_date 判定在调仓日是否已上市；含退市股，因其 daily 历史保留）。
   不使用 index_constituent（399006 仅 2026-07-06 一个快照 -> fallback 到未来幸存者池）。
2. 价格：后复权 adj_close = close * adj_factor（与 base_strategy.py 一致）。
3. 无前视：信号只用 <= (调仓日-1月) 的数据；调仓日收盘买入，下一调仓日收盘卖出。

假设：每月第一个交易日调仓；Top5 等权；单边费 0.1%（保守，含佣+印+滑）。
动量：跳过最近1月，回看 N 月 = adj_close(end)/adj_close(start) - 1。
基准：创业板指(399006) / 中证800(000906) 买入持有。
"""
import sqlite3, numpy as np, bisect
from datetime import datetime
from dateutil.relativedelta import relativedelta

DB = r"D:/tu-shareData/astock_daily.db"
INIT = 1_000_000.0
FEE = 0.001

def get_conn(): return sqlite3.connect(DB)

def load_stocks():
    """返回 {code: (sorted trade_dates list, adj_close array)}。逐股查询，避免 1500 万行大排序。"""
    con = get_conn(); cur = con.cursor()
    cur.execute("SELECT ts_code FROM stock_basic WHERE name LIKE '%ST%' OR name LIKE '%*%'")
    st = set(r[0] for r in cur.fetchall())
    cur.execute("SELECT DISTINCT ts_code FROM daily")
    codes = [r[0] for r in cur.fetchall()]; con.close()
    series = {}
    n = 0
    for code in codes:
        if code in st or code.endswith('.BJ'):
            continue
        con = get_conn(); cur = con.cursor()
        cur.execute("""SELECT d.trade_date, d.close, a.adj_factor
                       FROM daily d
                       LEFT JOIN adj_factor a ON d.ts_code=a.ts_code AND d.trade_date=a.trade_date
                       WHERE d.ts_code=? ORDER BY d.trade_date""", (code,))
        rows = cur.fetchall(); con.close()
        if not rows:
            continue
        tds = [r[0] for r in rows]
        adjc = np.array([(r[1]*(r[2] if r[2] is not None else 1.0)) for r in rows], dtype=float)
        series[code] = (tds, adjc)
        n += 1
    print(f"  载入 {n} 只有效个股", flush=True)
    return series

def adj_close_at(series, target):
    tds, adjc = series
    idx = bisect.bisect_right(tds, target) - 1
    return adjc[idx] if idx >= 0 else None

def rebalance_dates(all_tds):
    out, last = [], None
    for d in all_tds:
        m = d[:6]
        if m != last:
            out.append(d); last = m
    return out

def run(series, all_tds, rbs, lookback_months):
    cash = INIT; holding = {}; eq = [INIT]
    list_date = {c: series[c][0][0] for c in series}
    for k, rb in enumerate(rbs):
        # 卖出
        if holding:
            for c, sh in list(holding.items()):
                p = adj_close_at(series[c], rb)
                if p: cash += sh*p*(1-FEE)
            holding = {}
        dt = datetime.strptime(rb, "%Y%m%d")
        end_d = (dt - relativedelta(months=1)).strftime("%Y%m%d")
        start_d = (dt - relativedelta(months=lookback_months+1)).strftime("%Y%m%d")
        # 候选：已上市
        cands = [c for c in series if list_date[c] <= rb]
        moms = []
        for c in cands:
            pe = adj_close_at(series[c], end_d); ps = adj_close_at(series[c], start_d)
            if pe is not None and ps is not None and ps > 0:
                moms.append((c, pe/ps - 1.0))
        moms.sort(key=lambda x: x[1], reverse=True)
        top = moms[:5]
        if top:
            w = cash/len(top)
            for c, _ in top:
                p = adj_close_at(series[c], rb)
                if p and p > 0:
                    invest = w*(1-FEE); holding[c] = invest/p; cash -= invest
        val = cash
        for c, sh in holding.items():
            p = adj_close_at(series[c], rb)
            if p: val += sh*p
        eq.append(val)
    eq = np.array(eq)
    total = eq[-1]/eq[0]-1
    yrs = (datetime.strptime(rbs[-1],"%Y%m%d")-datetime.strptime(rbs[0],"%Y%m%d")).days/365.25
    ann = (eq[-1]/eq[0])**(1/yrs)-1
    peak = np.maximum.accumulate(eq); mdd = ((eq-peak)/peak).min()
    rets = np.diff(eq)/eq[:-1]
    sharpe = (np.mean(rets)*252-0.025)/(np.std(rets)*np.sqrt(252)) if np.std(rets)>0 else 0.0
    return dict(total=total, ann=ann, mdd=mdd, sharpe=sharpe, final=eq[-1])

def bench(idx):
    con = get_conn(); cur = con.cursor()
    cur.execute("SELECT close FROM index_daily WHERE ts_code=? AND trade_date>=? ORDER BY trade_date ASC LIMIT 1",(idx,'20150101'))
    s=cur.fetchone()
    cur.execute("SELECT close FROM index_daily WHERE ts_code=? AND trade_date<=? ORDER BY trade_date DESC LIMIT 1",(idx,'20260701'))
    e=cur.fetchone(); con.close()
    return (e[0]/s[0]-1) if s and e else None

if __name__ == "__main__":
    con = get_conn(); cur = con.cursor()
    cur.execute("SELECT DISTINCT trade_date FROM daily ORDER BY trade_date")
    all_tds = [r[0] for r in cur.fetchall()]; con.close()
    rbs = [d for d in rebalance_dates(all_tds) if '20150101' <= d <= '20260701']
    series = load_stocks()
    print(f"{'窗口':<8}{'总收益':>12}{'年化':>10}{'最大回撤':>10}{'夏普':>8}{'最终资产':>16}")
    for lb in (3,6,12):
        r = run(series, all_tds, rbs, lb)
        print(f"{lb}月{''::<4}{r['total']*100:>11.2f}%{r['ann']*100:>9.2f}%{r['mdd']*100:>9.2f}%{r['sharpe']:>8.2f}{r['final']:>15,.0f}")
    cyb = bench('399006.SZ'); zz = bench('000906.SH')
    print(f"\n基准(买入持有) 创业板指399006: {cyb*100:+.2f}%   中证800: {zz*100:+.2f}%")
    print("\n对照：原 399006 路径（未来幸存者池+裸价）3月动量 = +1250.76% 是幻影。")
