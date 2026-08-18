"""组合风险诊断：相关性-aware 组合 VaR（测度论视频待办 ①）

戳 inverse-vol / vol-size 定权的盲区："风险可加性只对互斥(ρ=0)成立"。
当策略间相关性 > 0 时，朴素把组合 VaR 当各头寸 VaR 之和(对角协方差)会
系统性低估真实组合风险。

做法：
  - 取 价值选股 / 红利低波 / 高股息+成长 / 沪深300 的日收益（复用 _risk_alpha_compare 入口）
  - 算两两相关性矩阵 + 平均两两相关性
  - 对"等权"与"逆波动定权"两种权重，分别算：
        naive  = 对角协方差(忽略相关性)下的组合日波动 = sqrt(Σ (w_i·σ_i)²)
        true  = 完整协方差(含相关性)下的组合日波动 = sqrt(wᵀΣw)
        gap%  = (true - naive) / naive   ← 相关性税
  - 额外对比"危机日"(沪深300 后10%日)相关性 vs 全样本，看相关性是否在崩盘时飙升
落盘：
  - data/results/livermore/alpha_nav.csv       （四序列对齐后的净值，inner join）
  - data/results/livermore/portfolio_risk.csv  （相关性矩阵 + 组合VaR对比）

注意：三策略回测较重（价值选股单方约 21 分钟），建议在本地后台跑。
"""
import os
import numpy as np
import pandas as pd
import run_monthly_rebalance as mr
import run_dividend_growth_monthly as dg
from _risk_alpha_compare import hs300_returns, _Sup, _extract_result, START, END

OUT_DIR = "data/results/livermore"
NAV_CSV = os.path.join(OUT_DIR, "alpha_nav.csv")
OUT_CSV = os.path.join(OUT_DIR, "portfolio_risk.csv")

Z99 = 2.326347874  # 标准正态 99% 分位


# ────────────────────────────────────────────────────────────────────────────
#  取数：各序列 (date, value) 对
# ────────────────────────────────────────────────────────────────────────────
def _nav_pairs(res, preferred):
    nav = _extract_result(res, preferred=preferred)
    if nav is None:
        return []
    out = []
    for d in nav:
        if isinstance(d, dict):
            dt = d.get("date"); v = d.get("value")
        elif isinstance(d, (list, tuple)) and len(d) >= 2:
            dt = d[0]; v = d[1]
        else:
            continue
        try:
            out.append((str(dt), float(v)))
        except (TypeError, ValueError):
            pass
    return out


def _hs300_nav():
    conn = mr.get_conn()
    try:
        df = pd.read_sql(
            f"SELECT trade_date, close FROM index_daily "
            f"WHERE ts_code='000300.SH' AND trade_date BETWEEN '{START}' AND '{END}' "
            f"ORDER BY trade_date", conn)
    finally:
        conn.close()
    return [(str(r.trade_date), float(r.close)) for r in df.itertuples()]


def _div_growth_cfg():
    return dict(top_n=10, top_pct=0.10, pe_max=20.0, peg_min=0.08, peg_max=2.0,
                roe_min=3.0, rev_min=5.0, np_min=11.0, stop_loss=0.0,
                atr_stop=0.0, atr_period=14)


def get_strategy_navs():
    navs = {}
    print("[0] 沪深300 基准 ...", flush=True)
    navs["沪深300"] = _hs300_nav()

    print("[1/3] 价值选股（较慢）...", flush=True)
    with _Sup():
        res = mr.run_backtest(START, END, selection_method="value")
    navs["价值选股"] = _nav_pairs(res, "daily_values")

    print("[2/3] 红利低波 ...", flush=True)
    with _Sup():
        res = mr.run_backtest(START, END, selection_method="div_low_vol")
    navs["红利低波"] = _nav_pairs(res, "daily_values")

    print("[3/3] 高股息+成长 ...", flush=True)
    with _Sup():
        res = dg.run_window(START, END, _div_growth_cfg())
    navs["高股息+成长"] = _nav_pairs(res, "nav_raw")

    # 打印各序列长度，便于发现对齐问题
    for k, v in navs.items():
        print(f"  {k}: {len(v)} 行")
    return navs


# ────────────────────────────────────────────────────────────────────────────
#  分析
# ────────────────────────────────────────────────────────────────────────────
def _port_var(w, vols, Sigma):
    """返回 (naive对角波动, true完整协方差波动)，均已年化。"""
    naive = float(np.sqrt(np.sum((w * vols) ** 2)))
    true = float(np.sqrt(w @ Sigma @ w))
    return naive, true


def analyze(navs):
    # 对齐到共同交易日（inner join）
    df = pd.DataFrame({name: {d: v for d, v in pairs} for name, pairs in navs.items()})
    df = df.dropna().sort_index()
    df.index = df.index.astype(str)
    df.to_csv(NAV_CSV, index_label="trade_date")
    n_dates = len(df)

    rets = df.pct_change().dropna()
    names = list(rets.columns)
    R = rets.values
    n = len(names)

    corr = rets.corr()
    avg_corr = float(corr.values[np.triu_indices(n, 1)].mean())

    # 年化波动 / 协方差
    vols = rets.std(ddof=1).values * np.sqrt(252)
    Sigma = np.cov(R, rowvar=False) * 252  # 年化协方差

    # 权重
    w_equal = np.ones(n) / n
    inv = 1.0 / vols
    w_iv = inv / inv.sum()

    # 危机日相关性（沪深300 后 10% 日）
    hs = rets["沪深300"].values
    thr = np.quantile(hs, 0.10)
    crisis = rets[hs <= thr]
    crisis_corr = crisis.corr()
    crisis_avg = float(crisis_corr.values[np.triu_indices(n, 1)].mean()) if len(crisis) > 1 else float("nan")

    # ── 打印 ──
    print("\n" + "=" * 92)
    print(f"组合风险诊断（{START}~{END}，{n_dates} 交易日对齐，日收益口径）")
    print("=" * 92)

    print("\n[相关性矩阵]")
    print(corr.round(3).to_string())

    print(f"\n平均两两相关性(全样本): {avg_corr:+.3f}")
    print(f"平均两两相关性(危机日/沪深300后10%): {crisis_avg:+.3f}"
          + ("  ⚠️ 危机时相关性飙升" if crisis_avg > avg_corr + 0.1 else ""))

    print("\n[组合 VaR99：朴素(对角/忽略相关性) vs 真实(完整协方差)]")
    print(f"{'权重':<10}{'朴素年化VaR99':>16}{'真实年化VaR99':>16}{'相关性税gap':>14}")
    comp_rows = []
    for wname, w in [("等权", w_equal), ("逆波动定权", w_iv)]:
        naive, true = _port_var(w, vols, Sigma)
        v_naive = Z99 * naive
        v_true = Z99 * true
        gap = (true - naive) / naive if naive > 0 else float("nan")
        print(f"{wname:<10}{v_naive*100:>15.2f}%{v_true*100:>15.2f}%"
              f"{(gap*100):>13.1f}%")
        comp_rows.append([wname, v_naive, v_true, gap])

    # 最朴素"互斥可加"：各头寸 VaR 直接相加（ρ=1 上界）
    standalone = Z99 * vols
    sum_standalone = float(standalone.sum())
    true_equal = comp_rows[0][2]
    div_benefit = (sum_standalone - true_equal) / sum_standalone if sum_standalone > 0 else float("nan")
    print(f"\n对照：各头寸 VaR99 直接相加(完全可加/ρ=1) = {sum_standalone*100:.2f}%")
    print(f"      真实等权组合 VaR99 = {true_equal*100:.2f}%  → 分散收益 {(div_benefit*100):.1f}%")

    # ── 落盘 CSV ──
    with open(OUT_CSV, "w", encoding="utf-8") as f:
        f.write("## 相关性矩阵\n")
        corr.to_csv(f)
        f.write("\n## 组合VaR99对比(naive=对角/忽略相关性, true=完整协方差)\n")
        f.write("权重,朴素年化VaR99,真实年化VaR99,相关性税gap\n")
        for wname, vn, vt, gap in comp_rows:
            f.write(f"{wname},{vn*100:.2f}%,{vt*100:.2f}%,{gap*100:.1f}%\n")
        f.write(f"\n平均两两相关性(全样本),{avg_corr:.3f}\n")
        f.write(f"平均两两相关性(危机日),{crisis_avg:.3f}\n")
        f.write(f"各头寸VaR直接相加(ρ=1),{sum_standalone*100:.2f}%\n")
        f.write(f"真实等权组合VaR99,{true_equal*100:.2f}%\n")
        f.write(f"分散收益,{div_benefit*100:.1f}%\n")
    print(f"\n已保存: {NAV_CSV}\n已保存: {OUT_CSV}")

    # ── 一句话结论 ──
    iv_gap = comp_rows[1][3]
    print("\n--- 结论 ---")
    print(f"  平均两两相关性 {avg_corr:+.2f}：三策略并非独立 edge，"
          f"逆波动定权在忽略相关性时把组合 VaR99 低估了约 {iv_gap*100:.0f}%。")
    print(f"  危机日相关性升至 {crisis_avg:+.2f}，相关性税在崩盘时更重——"
          f"这正是'可加性只对互斥成立'的实证。")


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    navs = get_strategy_navs()
    analyze(navs)


if __name__ == "__main__":
    run()
