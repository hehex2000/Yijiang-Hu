"""⑤ 箱+量能市场环境门控，套到 alpha 组合做风控闸验证（测度论视频待办）

把 run_livermore_v2.py 里"箱体压缩(改进D)+量能确认+关键点突破"的入场过滤，
移植成【市场环境门控】：当沪深300处于"窄幅横盘(coil)且无量确认突破"的
不确定性状态时，把整个 alpha 组合降到现金（利弗莫尔"赚大钱是等待"）。

验证目标：这个门控是否真把组合【左尾 CVaR】降下来（戳 item ① 的相关性税盲区）。

做法：
  - 读 alpha_nav.csv（item ① 已对齐好的 沪深300/价值/红利低波/高股息+成长 净值）
  - 另从 index_daily 取沪深300 OHLCV，复刻箱+量能门控信号（避免未来函数：用 T-1 窗口）
  - 组合 = 三 alpha 策略逆波动定权；门控=0 的日子组合收益置现金(0)
  - 用 risk_metrics 对 闸门前后 组合序列算 CVaR99 / VaR99(CF) / 偏度 / 最大回撤 / 总收益
  - 参数敏感性网格(box_len × box_width)看结论是否稳健
落盘：
  - data/results/livermore/gated_portfolio_risk.csv
注意：本脚本只读已落盘的 alpha_nav.csv + index_daily，不重跑策略回测。
"""
import os
import argparse
import numpy as np
import pandas as pd
import run_monthly_rebalance as mr
from risk_metrics import risk_summary

OUT_DIR = "data/results/livermore"
NAV_CSV = os.path.join(OUT_DIR, "alpha_nav.csv")
OUT_CSV = os.path.join(OUT_DIR, "gated_portfolio_risk.csv")

ALPHA = ["价值选股", "红利低波", "高股息+成长"]
BENCH = "沪深300"


# ────────────────────────────────────────────────────────────────────────────
#  取数
# ────────────────────────────────────────────────────────────────────────────
def load_navs():
    """优先读 item ① 落盘的已对齐净值；不存在则回退重跑（慢）。"""
    if os.path.exists(NAV_CSV):
        df = pd.read_csv(NAV_CSV, index_col=0)
        df.index = df.index.astype(str)
        print(f"[load] 读 alpha_nav.csv: {df.shape[0]} 行 / 列={list(df.columns)}")
        return df
    # 回退：复用 _portfolio_risk 的取数
    from _portfolio_risk import get_strategy_navs, NAV_CSV as _NC
    navs = get_strategy_navs()
    df = pd.DataFrame({n: {d: v for d, v in p} for n, p in navs.items()})
    df = df.dropna().sort_index()
    df.index = df.index.astype(str)
    df.to_csv(_NC, index_label="trade_date")
    print(f"[load] 回退重跑取数并落盘: {df.shape[0]} 行")
    return df


def fetch_hs300_ohlcv(start, end):
    conn = mr.get_conn()
    try:
        df = pd.read_sql(
            f"SELECT trade_date, open, high, low, close, vol FROM index_daily "
            f"WHERE ts_code='000300.SH' AND trade_date BETWEEN '{start}' AND '{end}' "
            f"ORDER BY trade_date", conn)
    finally:
        conn.close()
    df["trade_date"] = df["trade_date"].astype(str)
    df = df.set_index("trade_date").sort_index()
    return df


# ────────────────────────────────────────────────────────────────────────────
#  门控信号（复刻 run_livermore_v2 的改进D + 量能确认 + 关键点突破）
# ────────────────────────────────────────────────────────────────────────────
def compute_gate(hs, box_len=30, box_width=0.10, vol_win=5, vol_mult=1.5, look=20):
    """返回布尔 Series（as-of 每日收盘状态）: True=投资 / False=空仓( coil 且无确认突破 )。

    注意：本函数返回的是【截至当日收盘 t 的状态】，所有量价只用 ≤ t 的数据，
    本身无未来函数。应用层(build_portfolios)再用 gate[t-1] 决定第 t 日持仓，
    形成 1 日决策延迟，避免"用 T 日收盘信息决定 T 日收益"的同日未来函数。
    """
    high = hs["high"].astype(float)
    low = hs["low"].astype(float)
    close = hs["close"].astype(float)
    vol = hs["vol"].astype(float)

    # 箱体相对宽度（截至 t 的 box_len 日窗口）
    box_hi = high.rolling(box_len, min_periods=box_len).max()
    box_lo = low.rolling(box_len, min_periods=box_len).min()
    box_mid = close.rolling(box_len, min_periods=box_len).mean()
    box_width_s = (box_hi - box_lo) / box_mid
    coiled = (box_width_s <= box_width) & box_width_s.notna()        # as-of t 横盘

    # 关键点突破（截至 t 的前 look 日最高）
    key_level = high.rolling(look, min_periods=look).max()
    up_breakout = (close > key_level) & key_level.notna()

    # 量能确认（截至 t 的前 vol_win 日均量）
    vol_ma = vol.rolling(vol_win, min_periods=vol_win).mean()
    vol_confirmed = (vol > vol_ma * vol_mult) & vol_ma.notna()

    breakout_confirmed = up_breakout & vol_confirmed

    # 投资条件：非横盘，或（横盘但已确认突破）；横盘且未确认突破 → 空仓
    coiled_f = coiled.fillna(False)
    breakout_f = breakout_confirmed.fillna(False)
    invest = ~(coiled_f & (~breakout_f))                            # as-of t 状态
    return invest


# ────────────────────────────────────────────────────────────────────────────
#  组合构建 + 闸门应用
# ────────────────────────────────────────────────────────────────────────────
def build_portfolios(navs, gate):
    rets = navs[ALPHA].pct_change().dropna()
    # 逆波动定权（与 item ① 一致）
    vols = rets.std(ddof=1).values * np.sqrt(252)
    inv = 1.0 / vols
    w = inv / inv.sum()

    port_ret = (rets.values * w).sum(axis=1)
    port_ret = pd.Series(port_ret, index=rets.index)

    # 1 日决策延迟：用 T-1 收盘状态决定 T 日持仓，杜绝同日未来函数
    g = gate.reindex(port_ret.index).shift(1).fillna(True)
    g = g.astype(bool)
    port_ret_gated = port_ret.where(g, 0.0)         # 空仓日收益=现金(0)

    nav_ung = (1.0 + port_ret).cumprod()
    nav_gat = (1.0 + port_ret_gated).cumprod()
    return dict(w=w, names=ALPHA, ret_ung=port_ret, ret_gat=port_ret_gated,
                nav_ung=nav_ung, nav_gat=nav_gat, gate=g)


def _max_drawdown(nav):
    peak = nav.cummax()
    return float(((nav - peak) / peak).min())


def _total_return(nav):
    return float(nav.iloc[-1] - 1.0)


def evaluate(port):
    mu = risk_summary(returns=port["ret_ung"].values, label="未闸门")
    mg = risk_summary(returns=port["ret_gat"].values, label="箱+量能闸门")
    out = {}
    for tag, m in (("ung", mu), ("gat", mg)):
        out[tag] = dict(
            total_ret=_total_return(port["nav_ung" if tag == "ung" else "nav_gat"]),
            ann_vol=m["ann_vol"],
            skew=m["skew"],
            excess_kurt=m["excess_kurt"],
            var99_cf=m["var_99_cf"],
            cvar99=m["var_99_cvar"],
            tail_ratio=m["tail_ratio"],
            max_dd=_max_drawdown(port["nav_ung" if tag == "ung" else "nav_gat"]),
        )
    out["time_in_cash"] = float((~port["gate"]).mean())
    out["w"] = port["w"]
    out["names"] = port["names"]
    # ── 诚实诊断：闸门是"聪明择时"还是"粗暴少持仓"？──
    ung = port["ret_ung"]
    inv = port["gate"]
    # 投资日(alpha 持仓)的平均日收益 vs 全样本平均日收益
    out["mean_ret_invested"] = float(ung[inv].mean())
    out["mean_ret_all"] = float(ung.mean())
    # 最差 1% 日里，被闸门跳过(空仓)的比例 vs 整体空仓比例
    worst = ung.quantile(0.01)
    worst_days = ung <= worst
    out["skip_rate_all"] = float((~inv).mean())
    out["skip_rate_worst1pct"] = float((~inv[worst_days]).mean()) if worst_days.any() else float("nan")
    return out


# ────────────────────────────────────────────────────────────────────────────
#  主流程
# ────────────────────────────────────────────────────────────────────────────
def run(box_len=30, box_width=0.10, vol_win=5, vol_mult=1.5, look=20,
        grid=False):
    os.makedirs(OUT_DIR, exist_ok=True)
    navs = load_navs()
    start, end = navs.index[0], navs.index[-1]
    # 取沪深300 OHLCV（多取 1 年做滚动窗口 warm-up）
    hs = fetch_hs300_ohlcv(str(int(start) - 10000), end)

    print("\n" + "=" * 92)
    print(f"⑤ 箱+量能市场环境门控 → alpha 组合风控闸（{start}~{end}）")
    print("=" * 92)

    base = compute_gate(hs, box_len, box_width, vol_win, vol_mult, look)
    port = build_portfolios(navs, base)
    res = evaluate(port)

    # ── 打印主结论 ──
    u, g = res["ung"], res["gat"]
    print(f"\n[闸门参数] box_len={box_len} box_width={box_width:.0%} "
          f"vol_win={vol_win} vol_mult={vol_mult} look={look}")
    print(f"[空仓占比] {res['time_in_cash']*100:.1f}% 的交易日在现金")
    print(f"[逆波动权重] " + "  ".join(f"{n}={w*100:.1f}%"
          for n, w in zip(res["names"], res["w"])))

    hdr = f"{'指标':<12}{'未闸门':>16}{'闸门后':>16}{'变化':>14}"
    print("\n" + hdr)
    rows = [
        ("总收益", f"{u['total_ret']*100:.1f}%", f"{g['total_ret']*100:.1f}%",
         f"{(g['total_ret']-u['total_ret'])*100:+.1f}pp"),
        ("年化波动", f"{u['ann_vol']*100:.1f}%", f"{g['ann_vol']*100:.1f}%",
         f"{(g['ann_vol']-u['ann_vol'])*100:+.1f}pp"),
        ("偏度", f"{u['skew']:+.2f}", f"{g['skew']:+.2f}",
         f"{g['skew']-u['skew']:+.2f}"),
        ("VaR99(CF)", f"{u['var99_cf']*100:.2f}%", f"{g['var99_cf']*100:.2f}%",
         f"{(g['var99_cf']-u['var99_cf'])*100:+.2f}pp"),
        ("CVaR99", f"{u['cvar99']*100:.2f}%", f"{g['cvar99']*100:.2f}%",
         f"{(g['cvar99']-u['cvar99'])*100:+.2f}pp"),
        ("尾部比率", f"{u['tail_ratio']:.2f}", f"{g['tail_ratio']:.2f}",
         f"{g['tail_ratio']-u['tail_ratio']:+.2f}"),
        ("最大回撤", f"{u['max_dd']*100:.1f}%", f"{g['max_dd']*100:.1f}%",
         f"{(g['max_dd']-u['max_dd'])*100:+.1f}pp"),
    ]
    for r in rows:
        print(f"{r[0]:<12}{r[1]:>16}{r[2]:>16}{r[3]:>14}")

    cvar_cut = (u['cvar99'] - g['cvar99']) / u['cvar99'] if u['cvar99'] > 0 else float('nan')
    print(f"\n--- 结论 ---")
    print(f"  组合左尾 CVaR99: {u['cvar99']*100:.2f}% → {g['cvar99']*100:.2f}%"
          f"（{(cvar_cut*100):+.1f}%，{'↓ 改善' if cvar_cut>0 else '↑ 恶化'}）")
    # 聪明择时 vs 粗暴少持仓 诊断
    smart = res["skip_rate_worst1pct"] > res["skip_rate_all"] + 0.05
    print(f"  诚实诊断: 整体空仓率 {res['skip_rate_all']*100:.1f}% | "
          f"最差1%日空仓率 {res['skip_rate_worst1pct']*100:.1f}% | "
          f"投资日均值 {res['mean_ret_invested']*100:+.3f}% vs 全样本 {res['mean_ret_all']*100:+.3f}%")
    if smart:
        print("  -> 闸门是聪明的：disproportionately 跳过最差日（真择时）")
    else:
        print("  -> 闸门主要粗暴少持仓降低波动（择时信号边际信息有限）")
    verdict = ("✅ 门控降低了左尾 CVaR，且未牺牲过多收益"
               if (cvar_cut > 0.05 and g['total_ret'] >= u['total_ret'] - 0.10)
               else ("⚠️ 门控降了左尾但显著拖累收益"
                     if cvar_cut > 0 else "❌ 门控未降左尾（反而恶化或持平）"))
    print(f"  判定: {verdict}")

    # ── 落盘 ──
    with open(OUT_CSV, "w", encoding="utf-8") as f:
        f.write(f"# ⑤ 箱+量能门控套 alpha 组合 (box_len={box_len}, box_width={box_width}, "
                f"vol_win={vol_win}, vol_mult={vol_mult}, look={look})\n")
        f.write(f"窗口,{start}~{end}\n")
        f.write(f"空仓占比,{res['time_in_cash']*100:.2f}%\n")
        f.write("指标,未闸门,闸门后,变化\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]}\n")
        f.write(f"CVaR99降幅,{cvar_cut*100:.1f}%\n")
        f.write(f"判定,{verdict}\n")
    print(f"\n已保存: {OUT_CSV}")

    # ── 参数敏感性网格 ──
    if grid:
        print("\n[参数敏感性网格] box_len × box_width（vol_mult=1.5, look=20）")
        print(f"{'box_len':>8}{'box_width':>10}{'空仓%':>8}{'总收益':>10}"
              f"{'CVaR99':>10}{'CVaR降幅':>10}")
        for bl in (20, 30, 60):
            for bw in (0.08, 0.10, 0.15):
                gg = compute_gate(hs, bl, bw, vol_win, vol_mult, look)
                pp = build_portfolios(navs, gg)
                rr = evaluate(pp)
                cu, cg = rr["ung"]["cvar99"], rr["gat"]["cvar99"]
                cut = (cu - cg) / cu if cu > 0 else float("nan")
                print(f"{bl:>8}{bw*100:>9.0f}%{rr['time_in_cash']*100:>7.1f}%"
                      f"{cg_ret(rr):>10}{cg*100:>9.2f}%{cut*100:>9.1f}%")
    return res


def cg_ret(rr):
    return f"{rr['gat']['total_ret']*100:.1f}%"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--box-len", type=int, default=30)
    ap.add_argument("--box-width", type=float, default=0.10)
    ap.add_argument("--vol-win", type=int, default=5)
    ap.add_argument("--vol-mult", type=float, default=1.5)
    ap.add_argument("--look", type=int, default=20)
    ap.add_argument("--grid", action="store_true", help="跑参数敏感性网格")
    a = ap.parse_args()
    run(box_len=a.box_len, box_width=a.box_width, vol_win=a.vol_win,
        vol_mult=a.vol_mult, look=a.look, grid=a.grid)
