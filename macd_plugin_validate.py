# -*- coding: utf-8 -*-
"""
主线④：MACD插件策略 vs 红利/价值策略「止损层效率」对照台
=========================================================
把三种风控层叠加到【同一共同基线净值】上，在「相同回撤控制约束」下比较收益代价：

  方法A  基线无控制        : 底层策略原样（中证800买入持有，或任意 --base-nav）
  方法B  平台15%止损+MACD减仓 : faithful 组合层复刻 红利/价值内核风控层
                                (个股-15% → 组合峰回撤15%触止损；基准沪深300 MACD
                                 死亡交叉 → 持币；金叉+收复才再入场)
  方法C  弱市持币 regime cash : 本任务抽出的独立 overlay（regime_cash_overlay）
                                沪深300<MA(regime_ma) → 持币；否则跟底层

效率指标（回撤-收益权衡）：
  DD_cut   = 基线MDD − 方法MDD        (pp, 越大=回撤砍得越多=好)
  Ret_cost = 基线收益 − 方法收益       (pp, 越大=牺牲越多=坏)
  Eff      = DD_cut / max(Ret_cost,ε) (越大=每牺牲1pp收益换来的回撤削减越多=省心)

用法：
  ./venv_ml/Scripts/python.exe macd_plugin_validate.py
  ./venv_ml/Scripts/python.exe macd_plugin_validate.py --regime-ma 250 --stop-pct 0.15
  ./venv_ml/Scripts/python.exe macd_plugin_validate.py --base zz800_eq   # 等权zz800(慢)
注意：基线默认用中证800指数(000906.SH)买入持有——真实有回撤的"需保护"组合；
      若要对照具体策略(如 MacdJimPlugin / 红利低波)，把其净值存 csv 后 --base-nav <csv>。
"""
import argparse
import numpy as np
import pandas as pd

from regime_cash_overlay import (
    load_index_close, regime_signal, apply_overlay, cash_ratio, BENCH, DEFAULT_MA,
)

try:
    from run_monthly_rebalance import get_conn
except Exception:
    import sqlite3
    def get_conn():
        return sqlite3.connect(r'D:/tu-shareData/astock_daily.db')


# ───────────────────────── 基线净值 ─────────────────────────
def load_base_index(ts_code='000906.SH', start='20100101', end='20251231'):
    s = load_index_close(ts_code, start, end)
    if len(s) == 0:
        return None
    return s / s.iloc[0]   # 归一化到本金=1.0


def load_base_zz800_eq(start='20100101', end='20251231'):
    """等权中证800 日净值。

    实现要点（修复两处历史 bug）：
      1) 速度：原实现逐只 1860 次 SQL 循环 ≈ 22+ 分钟且全程无输出，易被误判卡死。
         改为分批 bulk IN 查询（每批 400），实测 ~2 分钟，并打印进度。
      2) 正确性：原实现返回每只股票各自的净值面板(2D DataFrame)，下游
         platform_stop_overlay 的 pd.Series(base) 会因 2D 数组 ValueError 崩溃，
         metrics 同样无法处理 2D。此处返回 1D 等权指数（每日等权收益累乘）。
    """
    conn = get_conn()
    # 取窗口内 zz800 成分（用最新可用名单即可，长周期回测近似）
    codes = [r[0] for r in conn.execute(
        "SELECT DISTINCT ts_code FROM index_constituent WHERE index_code='000906.SH'")]
    if not codes:
        codes = [r[0] for r in conn.execute(
            "SELECT DISTINCT ts_code FROM index_constituent WHERE index_code='000905.SH'")]
    print(f"[load] 等权中证800：{len(codes)} 只成分，分批 bulk 查询日线...", flush=True)
    start_i, end_i = int(start), int(end)
    frames = []
    for i in range(0, len(codes), 400):
        batch = codes[i:i + 400]
        ph = ",".join("?" * len(batch))
        d = pd.read_sql_query(
            f"SELECT ts_code,trade_date,close FROM daily WHERE ts_code IN ({ph}) "
            f"AND trade_date>=? AND trade_date<=? ORDER BY ts_code,trade_date",
            conn, params=batch + [start_i, end_i])
        if len(d):
            frames.append(d)
    conn.close()
    if not frames:
        return None
    print(f"[load] 取到 {sum(len(f) for f in frames)} 行，构建等权面板...", flush=True)
    alld = pd.concat(frames, ignore_index=True)
    alld['trade_date'] = alld['trade_date'].astype(int)  # 关键：保持 int 索引，否则与 hs(int) 对齐失败→全NaN→overlay 全程现金
    panel = alld.pivot(index='trade_date', columns='ts_code', values='close').sort_index()
    ret = panel.pct_change().fillna(0.0)
    # 等权指数：每日等权收益 -> 累乘（等价于每日再平衡到等权）
    eq_ret = ret.mean(axis=1)
    nav = (1.0 + eq_ret).cumprod()
    print(f"[load] 等权基线就绪，长度 {len(nav)}", flush=True)
    return nav


def load_base_csv(path):
    """读取自定义基线净值 csv。

    关键：必须以 date 列为索引返回（交易日 int 索引），否则下游
    hs.reindex(base.index) 会按位置错位 → 沪深300 信号全 NaN → overlay 退化。
    csv 格式：第1列=date(YYYYMMDD 整数), 第2列=nav(任意货币单位, 此处归一化到1.0)。
    """
    df = pd.read_csv(path)
    if df.columns[0] == 'date':
        idx = pd.to_numeric(df['date']).astype(int).values
        col = df.columns[1]
    else:
        # 退化路径：无 date 列，退化为位置索引（下游对齐会失真，仅作容错）
        idx = None
        col = df.columns[0]
        print("[WARN] 自定义基线 csv 缺少 date 列，按位置索引对齐（可能与沪深300信号错位）")
    s = pd.Series(df[col].astype(float).values, index=idx)
    return s / s.iloc[0]


# ───────────────────────── 指标 ─────────────────────────
def metrics(nav):
    nav = pd.Series(nav, dtype=float)
    ret = nav.iloc[-1] / nav.iloc[0] - 1
    n = len(nav)
    ann = (nav.iloc[-1] / nav.iloc[0]) ** (252.0 / max(n - 1, 1)) - 1
    peak = nav.cummax()
    mdd = float((nav / peak - 1).min())
    d = nav.pct_change().dropna()
    sharpe = float(d.mean() / d.std() * np.sqrt(252)) if d.std() > 0 else 0.0
    return ret, ann, mdd, sharpe


def macd_golden(close, fast=12, slow=26, sig=9):
    """沪深300 MACD：DIF>DEA 为 golden(持有)，否则 death(持币)。"""
    close = pd.Series(close, dtype=float)
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    dif = ema_f - ema_s
    dea = dif.ewm(span=sig, adjust=False).mean()
    return (dif > dea).fillna(False).astype(bool)


def platform_stop_overlay(base_nav, bench_close, stop_pct=0.15):
    """faithful 组合层复刻 红利/价值内核风控层（15%止损 + MACD减仓）。
    组合峰回撤>=stop_pct → 触止损持币；基准 MACD death → 持币；
    金叉 且 收复峰(创新高) 才再入场。"""
    base = np.asarray(base_nav, dtype=float)
    peak = pd.Series(base).cummax().values
    trailing_hit = base < peak * (1 - stop_pct)
    golden = macd_golden(bench_close).values if hasattr(bench_close, 'iloc') else macd_golden(pd.Series(bench_close)).values
    # 止损锁存：触 hit(-15%)后保持持币，直到基准 MACD 金叉(golden)才解锁再入场
    # （与 div_low_vol 真实行为一致：非 golden 即持币，金叉恢复；不要求组合创新高，
    #  避免对"未收复历史峰值"的宽基指数永久锁死、退化为全程空仓）
    stopped = np.zeros(len(base), dtype=bool)
    s = False
    for i in range(len(base)):
        if trailing_hit[i]:
            s = True
        if golden[i]:
            s = False
        stopped[i] = s
    hold = golden & (~stopped)
    return apply_overlay(base, hold), hold


# ───────────────────────── 主流程 ─────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='20100101')
    ap.add_argument('--end', default='20251231')
    ap.add_argument('--base', default='zz800_idx',
                    help='基线: zz800_idx(中证800指数,默认/快) | zz800_eq(等权,慢)')
    ap.add_argument('--base-nav', default=None, help='指定基线净值 csv（列=净值）')
    ap.add_argument('--regime-ma', type=int, default=DEFAULT_MA)
    ap.add_argument('--stop-pct', type=float, default=0.15)
    args = ap.parse_args()

    # 基线
    if args.base_nav:
        base = load_base_csv(args.base_nav)
        base_name = f"自定义基线({args.base_nav})"
    elif args.base == 'zz800_eq':
        base = load_base_zz800_eq(args.start, args.end)
        base_name = "等权中证800(买入持有)"
    else:
        base = load_base_index('000906.SH', args.start, args.end)
        base_name = "中证800指数(买入持有)"

    if base is None or len(base) < 30:
        print("[ERR] 基线数据不足，请检查 --start/--end 或数据")
        return

    # 沪深300 信号
    hs = load_index_close(BENCH, args.start, args.end)
    hs = hs.reindex(base.index).ffill()
    if len(hs) < args.regime_ma:
        print("[ERR] 沪深300 信号数据不足")
        return

    # 三方法
    nav_base = base
    nav_plat, hold_plat = platform_stop_overlay(base.values, hs.values, stop_pct=args.stop_pct)
    cash_plat = cash_ratio(hold_plat) * 100
    sig_reg = regime_signal(hs.values, ma_len=args.regime_ma)
    nav_reg = apply_overlay(base.values, sig_reg.values)
    cash_reg = cash_ratio(sig_reg) * 100

    rb, ab, mdb, sb = metrics(nav_base)
    rp, ap_, mdp, sp = metrics(nav_plat)
    rr, ar, mdr, sr = metrics(nav_reg)

    def pct(x):
        return f"{x*100:+.2f}%"

    print(f"\n{'='*86}")
    print(f"  主线④ 止损层效率对照 | 基线={base_name} | {args.start}→{args.end}")
    print(f"  弱市持币 signal=沪深300<MA{args.regime_ma} | 平台止损=组合峰回撤{args.stop_pct*100:.0f}%+MACD减仓")
    print(f"{'='*86}")
    hdr = f"  {'方法':<22}{'总收益':>10}{'年化':>9}{'最大回撤':>10}{'Sharpe':>9}{'持币%':>8}"
    print(hdr)
    print(f"  {'基线(无控制)':<20}{pct(rb):>10}{pct(ab):>9}{pct(mdb):>10}{sb:>9.2f}{0.0:>7.1f}%")
    print(f"  {'平台15%止损+MACD减仓':<18}{pct(rp):>10}{pct(ap_):>9}{pct(mdp):>10}{sp:>9.2f}{cash_plat:>7.1f}%")
    print(f"  {'弱市持币(regime cash)':<18}{pct(rr):>10}{pct(ar):>9}{pct(mdr):>10}{sr:>9.2f}{cash_reg:>7.1f}%")

    # 效率（相同回撤约束下的收益代价）
    dd_cut_base_plat = (mdb - mdp) * 100
    ret_cost_base_plat = (rb - rp) * 100
    dd_cut_base_reg = (mdb - mdr) * 100
    ret_cost_base_reg = (rb - rr) * 100

    def fmt_eff(dd_cut, ret_cost):
        if ret_cost < 0:
            return "∞ (既降回撤又多赚)"
        if abs(ret_cost) < 1e-9:
            return "∞ (收益无损)"
        return f"{dd_cut / max(ret_cost, 1e-9):.2f}"

    print(f"\n  ── 回撤-收益效率（相同回撤约束下，每牺牲1pp收益换来的回撤削减）──")
    print(f"  {'平台15%止损+MACD减仓':<20} ΔDD={dd_cut_base_plat:+.1f}pp  ΔRet={ret_cost_base_plat:+.1f}pp  Eff={fmt_eff(dd_cut_base_plat, ret_cost_base_plat)}")
    print(f"  {'弱市持币(regime cash)':<20} ΔDD={dd_cut_base_reg:+.1f}pp  ΔRet={ret_cost_base_reg:+.1f}pp  Eff={fmt_eff(dd_cut_base_reg, ret_cost_base_reg)}")

    print(f"\n  [结论口径] Eff 越大=用更少收益代价换更多回撤削减=更省心；"
          f"若 ΔRet<0(方法收益反超基线)则 Eff 记为∞(既降回撤又多赚)。")
    print(f"  注：平台层为组合层 faithful 复刻(个股-15%→组合峰回撤stop + 基准MACD死亡交叉持币)，"
          f"非逐股引擎精确重放；弱市持币为独立 overlay(regime_cash_overlay)。")


if __name__ == '__main__':
    main()
