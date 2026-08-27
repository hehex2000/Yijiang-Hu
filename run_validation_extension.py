# -*- coding: utf-8 -*-
"""
run_validation_extension.py —— 波段策略两项延伸验证
=============================================================================
延伸一：新标的样本外验证
  对象：3 只未参与参数设定的标的（002594.SZ比亚迪/601318.SH中国平安/000858.SZ五粮液）
  目的：同一组死参数（N_MOM=20/MIN_SWING=5%/D_CONFIRM=3/K_EXIT=3）在未见过的新标的上
        是否仍成立。这是检验"策略 vs 过拟合"的唯一诚实方法。
  判定标准：
    [PASS] 至少有一版开关的 maxDD 显著低于持有（>3pp）且年化差 < -3pp → 保险有效
    [FAIL] 年化差 < -3pp 且 maxDD 无改善 → 保费白交（与茅台/平安同结论）
    [反例] 任何标的上年化差 > 0 → 需警惕（可能是样本内偶然）

延伸二：转向段专项归因
  对象：茅台+平安+新标的，所有波段策略交易中发生在"转向段"的笔
  目的：量化转向段双杀的成分——
    A. 顶部给回多少才确认离场（lastup_ratio：离场前最后一段上涨/总持仓涨幅）
    B. V反转踏空多少（missed：离场后 20 日内市场最高涨幅）
    C. 确认滞后和追高在转向段 vs 非转向段的差异
  这是对波段策略最大短板的"尸检"，不改参数，只诊断。

回测正确性：全部复用 run_swing_trend.py 已修正的原则与参数。
"""
import sys
import sqlite3
import numpy as np
import pandas as pd

from run_swing_trend import (
    N_MOM, K_EXIT, D_CONFIRM, EXEC_WAIT_MAX, LIMIT_PCT, RF_ANN,
    COMMISSION, STAMP_TAX, SLIPPAGE, INITIAL, START_LOOK,
    compute_conditions, load_symbol, perf, label_regimes, _try_exec,
    run_swing_backtest, classify_exits,
)
from run_downside_switch import run_switch, RULES

# 延伸一：新标的（未参与任何参数设定）
NEW_SYMBOLS = ["002594.SZ", "601318.SH", "000858.SZ"]
ALL_SYMBOLS = ["600519.SH", "000001.SZ"] + NEW_SYMBOLS


def regime_of(regimes: pd.Series, date_str: str) -> str:
    """取某交易日的阶段标签；不在则返回"未知"。"""
    if date_str in regimes.index:
        return regimes.loc[date_str]
    return "未知"


def regime_segment(regimes: pd.Series, i: int, j: int) -> str:
    """[i, j] 区间内出现次数最多的阶段（多数票）。"""
    seg = regimes.iloc[i:j + 1] if j + 1 <= len(regimes) else regimes.iloc[i:]
    vc = seg.value_counts()
    return vc.index[0] if len(vc) else "未知"


def is_near_turning(regimes: pd.Series, idx: int, look: int = 10) -> bool:
    """判断 idx 前 look 日内是否发生阶段转换（如上升→下降）。
    转向段是短暂过渡态，不能用持仓区间的多数票判定；
    正确方法：看入场日前后是否处于阶段交界处。"""
    lo = max(0, idx - look)
    seg = regimes.iloc[lo:idx + 1]
    return "转向段" in seg.values or seg.nunique() > 1


# ==================== 延伸一：新标的样本外验证 ====================
def validate_new_symbols(con):
    print("\n" + "=" * 72)
    print("延伸一：新标的样本外验证（同一组死参数，未见过的新标的）")
    print("=" * 72)
    rows = []
    for code in NEW_SYMBOLS:
        raw = load_symbol(con, code)
        if len(raw) < 300:
            print(f"[skip] {code} 数据不足({len(raw)}行)")
            continue
        cond = compute_conditions(raw)
        warm = max(N_MOM, 25)
        cond = cond.iloc[warm:]
        dates = cond.index
        c = cond["close"].values.astype(float)
        o = cond["open"].values.astype(float)

        # 基准
        sh_h = INITIAL / (o[0] * (1 + SLIPPAGE) * (1 + COMMISSION)) if o[0] == o[0] and o[0] > 0 else 0.0
        hold_nav = pd.Series(sh_h * c, index=dates)
        mh = perf(hold_nav)

        # 波段策略
        swing = run_swing_backtest(cond)
        ms = perf(swing["nav_net"])

        # 两版开关
        switch_res = {}
        for rule in RULES:
            switch_res[rule] = run_switch(cond, rule)

        print(f"\n--- {code} ---")
        print(f"区间 {dates[0]}~{dates[-1]}  交易日 {len(dates)}")
        print(f"  一直持有 : 年化{mh['cagr']*100:6.2f}%  回撤{mh['mdd']*100:7.2f}%  夏普{mh['sharpe']:5.2f}")
        print(f"  波段策略 : 年化{ms['cagr']*100:6.2f}%  回撤{ms['mdd']*100:7.2f}%  夏普{ms['sharpe']:5.2f}")
        for rule in RULES:
            m = perf(switch_res[rule]["nav"])
            dd_diff = (m['mdd'] - mh['mdd']) * 100  # 正=回撤更小
            ca_diff = (m['cagr'] - mh['cagr']) * 100
            tag = ""
            if dd_diff > 3 and ca_diff > -3:
                tag = " → [PASS 保险有效]"
            elif ca_diff < -3 and dd_diff < 3:
                tag = " → [FAIL 保费白交]"
            elif ca_diff > 0:
                tag = " → [反例 需警惕]"
            print(f"  开关{rule}    : 年化{m['cagr']*100:6.2f}%  回撤{m['mdd']*100:7.2f}%  "
                  f"在场{switch_res[rule]['time_in_mkt']*100:4.0f}%  "
                  f"年化差{ca_diff:+6.2f}pp  回撤差{dd_diff:+6.2f}pp{tag}")
            rows.append(dict(标的=code, 策略=f"开关{rule}", 年化=m['cagr'],
                             回撤=m['mdd'], 年化差=ca_diff/100, 回撤差=dd_diff/100,
                             在场比例=switch_res[rule]['time_in_mkt']))

    if rows:
        pd.DataFrame(rows).to_csv("validation_new_symbols.csv", index=False)
        print("\n[save] validation_new_symbols.csv")

    # 判定总结（互斥分类，不重复计数）
    print("\n--- 样本外判定总结 ---")
    n_pass = n_fail = n_warn = 0
    for r in rows:
        if r["回撤差"] > 0.03 and r["年化差"] > -0.03:
            n_pass += 1
        elif r["年化差"] < -0.03 and r["回撤差"] < 0.03:
            n_fail += 1
        elif r["年化差"] > 0:
            n_warn += 1
    n_neutral = len(rows) - n_pass - n_fail - n_warn
    print(f"PASS(保险有效) {n_pass}/{len(rows)}  FAIL(保费白交) {n_fail}/{len(rows)}  "
          f"反例(需警惕) {n_warn}/{len(rows)}  未判定 {n_neutral}/{len(rows)}")
    if n_pass == 0:
        print("→ 与茅台/平安同构：该参数化在新标的上仍不成立。")
    elif n_pass >= 2:
        print("→ 存在新标的样本外支持，策略或非纯过拟合，需进一步检验。")


# ==================== 延伸二：转向段专项归因 ====================
def turning_point_autopsy(con):
    print("\n" + "=" * 72)
    print("延伸二：转向段专项归因（波段策略最大短板的尸检）")
    print("=" * 72)
    all_trades = []
    for code in ALL_SYMBOLS:
        raw = load_symbol(con, code)
        if len(raw) < 300:
            continue
        cond = compute_conditions(raw)
        warm = max(N_MOM, 25)
        cond = cond.iloc[warm:]
        res = run_swing_backtest(cond)
        res["trades"] = classify_exits(cond, res["trades"])
        regimes = label_regimes(cond)
        for tr in res["trades"]:
            tr["code"] = code
            tr["regime"] = regime_segment(regimes, tr["entry_i"], tr["exit_i"])
            # 转向段判定：入场前10日是否发生阶段转换（不能用持仓区间多数票）
            tr["near_turning_entry"] = is_near_turning(regimes, tr["entry_i"])
            tr["near_turning_exit"] = is_near_turning(regimes, tr["exit_i"])
            # 新增转向段专属指标
            ei = tr["entry_i"]; xi = tr["exit_i"]
            c = cond["close"].values.astype(float)
            h = cond["high"].values.astype(float)
            # A. 顶部给回比例：离场前从持仓最高点回撤了多少
            seg_high = np.nanmax(h[ei:xi + 1]) if xi > ei else c[ei]
            tr["top_drawback"] = (c[xi] / seg_high - 1) if seg_high > 0 else np.nan
            # B. V反转踏空：离场后20日内市场最高涨幅
            fwd_end = min(xi + 20, len(c) - 1)
            if fwd_end > xi:
                fwd_high = np.nanmax(h[xi + 1:fwd_end + 1])
                tr["missed"] = (fwd_high / c[xi] - 1) if c[xi] > 0 else np.nan
            else:
                tr["missed"] = np.nan
            all_trades.append(tr)

    if not all_trades:
        print("无交易数据")
        return

    df = pd.DataFrame(all_trades)
    # 转向段交易：入场前10日发生阶段转换（顶部追高入场 或 底部V反踏空）
    turning = df[df["near_turning_entry"] | df["near_turning_exit"]]
    other = df[~(df["near_turning_entry"] | df["near_turning_exit"])]

    print(f"\n总交易 {len(df)} 笔：转向段附近 {len(turning)} 笔 / 非转向段 {len(other)} 笔")

    print("\n--- 转向段 vs 非转向段 对比（均值）---")
    cols = ["ret_g", "ret_n", "lag", "chase", "unclear", "top_drawback", "missed"]
    print(f"{'指标':14s}{'转向段':>10s}{'非转向段':>10s}{'差异':>10s}")
    for col in cols:
        t_val = turning[col].mean() if len(turning) else np.nan
        o_val = other[col].mean() if len(other) else np.nan
        diff = (t_val - o_val) * 100 if t_val == t_val and o_val == o_val else np.nan
        print(f"{col:14s}{t_val*100 if t_val==t_val else 0:9.1f}%{o_val*100 if o_val==o_val else 0:9.1f}%{diff:9.1f}pp")

    print("\n--- 转向段交易明细（按 top_drawback 排序，最惨的在前面）---")
    if len(turning):
        tc = turning.sort_values("top_drawback")
        print(f"{'标的':10s}{'入场':>9s}{'出场':>9s}{'天数':>5s}{'毛收益':>8s}"
              f"{'顶部给回':>9s}{'踏空20d':>9s}{'追高':>7s}  归因")
        for _, tr in tc.iterrows():
            print(f"{tr['code']:10s}{tr['entry_date']:>9s}{tr['exit_date']:>9s}"
                  f"{tr['hold_days']:5d}{tr['ret_g']*100:7.1f}%"
                  f"{tr['top_drawback']*100:8.1f}%{tr['missed']*100:8.1f}%"
                  f"{tr['chase']*100:6.1f}%  {tr['exit_cls']}")

    # 诊断结论
    print("\n--- 诊断结论 ---")
    if len(turning):
        avg_giveback = turning["top_drawback"].mean()
        avg_missed = turning["missed"].mean()
        avg_ret = turning["ret_n"].mean()
        print(f"转向段平均：顶部给回 {avg_giveback*100:.1f}%（离场前从最高点回撤）"
              f"  V反踏空 {avg_missed*100:.1f}%（离场后20日最高涨幅）"
              f"  净收益 {avg_ret*100:.1f}%")
        print(f"→ 双杀机制：确认离场需回撤 {avg_giveback*100:.0f}%（K_EXIT=3 代价），"
              f"离场后 V 反又踏空 {avg_missed*100:.0f}%——"
              f"这是 3 日确认规则的固有代价，非参数可调。")

    df.to_csv("turning_point_autopsy.csv", index=False)
    print("[save] turning_point_autopsy.csv")


def main():
    import config
    con = sqlite3.connect(config.DATA["local_db_path"])
    validate_new_symbols(con)
    turning_point_autopsy(con)
    con.close()


if __name__ == "__main__":
    main()
