"""
M2 复验 - 组合层净成本 LS 回测（修正 per-stock 净成本相互抵消的缺陷）

问题：vp_factor_wide.py 对每只股票减同一个常数 RT，LS 利差 = 高组-低组，
常数相减抵消 -> "净"数≈"毛"数，不能反映真实摩擦。

正确做法：在组合层扣。每月 100% 换仓 -> 整本书每月 2 个 round-trip
（多头：买入+平仓；空头：卖开+买平）。portfolio 月净收益 = 毛 LS - 2*RT。

RT（单边 round-trip） = (佣金买0.025% + 滑点0.1%) + (佣金卖0.025% + 印花税0.05% + 滑点0.1%)
                      = 0.125% + 0.175% = 0.30%
月成本 = 2 * RT = 0.60%

从已落盘的 m2_wide_panel.csv 直接重算，无需重跑因子。
"""
import pandas as pd
import numpy as np

PANEL = "data/results/volume_profile/m2_wide_panel.csv"
RT_ROUND = 0.00125 + 0.00175   # = 0.0030 per round-trip
MONTHLY_TURNOVER_COST = 2 * RT_ROUND   # 整本书月换手成本

FACTORS = ["vp_dist_to_poc", "vp_support_dist_pct", "vp_va_pass"]


def portfolio_ls(panel: pd.DataFrame, factor: str, ret_col: str = "fwd_ret"):
    sub = panel[["date", "ts_code", factor, ret_col]].dropna()
    # 与 quantile_ls 对齐方向：按全样本 IC 符号决定多/空腿
    ic_sign = np.sign(sub[factor].corr(sub[ret_col]))
    months = []
    for d, g in sub.groupby("date"):
        if len(g) < 15:
            continue
        g = g.sort_values(factor).reset_index(drop=True)
        k = max(1, len(g) // 5)
        if ic_sign >= 0:
            longs = g.iloc[-k:]      # 高因子
            shorts = g.iloc[:k]      # 低因子
        else:
            longs = g.iloc[:k]       # 低因子（负IC：做多低因子=贴近POC/支撑）
            shorts = g.iloc[-k:]     # 高因子
        gross = longs[ret_col].mean() - shorts[ret_col].mean()
        months.append(gross)
    return pd.Series(months)


def tstat(x: pd.Series):
    x = x.dropna()
    if len(x) < 2 or x.std() == 0:
        return np.nan
    return x.mean() / (x.std() / np.sqrt(len(x)))


def main():
    panel = pd.read_csv(PANEL)
    print("宽宇宙 panel: %d 行 | %d 个月 | %d 票" %
          (len(panel), panel["date"].nunique(), panel["ts_code"].nunique()))
    print("组合层净成本: 月换手 100%% -> 每月整本书 2 round-trip = %.2f%%/月"
          % (MONTHLY_TURNOVER_COST * 100))
    print()
    print("%-20s %8s %7s %8s %7s %8s %7s" %
          ("factor", "LS_g%", "t_g", "LS_net%", "t_net", "年化net", "存活"))
    rows = []
    for f in FACTORS:
        sg = portfolio_ls(panel, f, "fwd_ret")
        sn = sg - MONTHLY_TURNOVER_COST
        lg, tg = sg.mean() * 100, tstat(sg)
        ln, tn = sn.mean() * 100, tstat(sn)
        ann = (1 + sn.mean()) ** 12 - 1
        alive = "Y" if (tn > 2 and ln > 0) else "N"
        print("%-20s %8.3f %7.2f %8.3f %7.2f %8.1f%% %7s" %
              (f, lg, tg, ln, tn, ann * 100, alive))
        rows.append((f, lg, tg, ln, tn, ann * 100, alive))
    out = pd.DataFrame(rows, columns=[
        "factor", "LS_gross_%", "t_gross", "LS_net_%", "t_net", "ann_net", "alive"])
    out.to_csv("data/results/volume_profile/m2_wide_net_portfolio.csv", index=False)
    print()
    print("saved -> data/results/volume_profile/m2_wide_net_portfolio.csv")

    # 附带：dist_to_poc 净 LS 累计曲线（看拐点/回撤）
    sg = portfolio_ls(panel, "vp_dist_to_poc", "fwd_ret")
    sn = sg - MONTHLY_TURNOVER_COST
    nav_g = (1 + sg).cumprod()
    nav_n = (1 + sn).cumprod()
    print()
    print("vp_dist_to_poc 累计净值(起点=1):")
    print("  毛: %.2f  净(扣月成本%.2f%%): %.2f" %
          (nav_g.iloc[-1], MONTHLY_TURNOVER_COST * 100, nav_n.iloc[-1]))
    print("  净最大回撤: %.1f%%" % ((nav_n / nav_n.cummax() - 1).min() * 100))


if __name__ == "__main__":
    main()
