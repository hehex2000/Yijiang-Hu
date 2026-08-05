# -*- coding: utf-8 -*-
"""纪律隔离消融驱动：同入场信号，换退出方式，隔离『纪律』价值。

退出模式：
  full     = 五句话纪律（止盈+回撤级联A）
  fixedN   = 固定持有 N 交易日清仓（无纪律、无止盈）
  reversal = 持有至收破 MA5（thesis 失效）或触顶 max_hold
"""
import io, contextlib
import run_ma5_swing as m

START, END, POOL, CAP = "20240903", "20260803", "all", 1000000

# (exit_mode, hold_days, label)
VARIANTS = [
    ("full",     10, "五句话纪律(full)"),
    ("fixedN",    5, "固定持有5日"),
    ("fixedN",   10, "固定持有10日"),
    ("fixedN",   20, "固定持有20日"),
    ("reversal", 60, "收破MA5止(reversal)"),
]


def run_one(mode, hold, zc):
    kw = dict(exit_mode=mode, hold_days=hold,
              max_hold_days=(hold if mode == "reversal" else 60))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = m.run_backtest(START, END, pool=POOL, capital=CAP, zero_cost=zc, **kw)
    return r


print(f"\n{'='*100}")
print(f"  纪律隔离消融 · 池={POOL} 区间={START}~{END} · 入场/成本/黑名单/持仓上限 完全一致，仅退出规则不同")
print(f"{'='*100}")
hdr = (f"{'退出模式':<22}{'口径':<8}{'总收益':>10}{'基准':>10}{'超额':>10}"
       f"{'年化':>9}{'夏普':>8}{'最大回撤':>11}{'成本占比':>11}")
print(hdr)
print(f"{'─'*100}")
for mode, hold, label in VARIANTS:
    for zc in (False, True):
        r = run_one(mode, hold, zc)
        tag = "零成本" if zc else "完整"
        cost_pct = (r["cost"] / CAP) if (not zc and r["cost"]) else 0.0
        print(f"{label:<22}{tag:<8}{r['total']:>+9.2%}{r['bench']:>+9.2%}"
              f"{(r['total']-r['bench']):>+9.2%}{r['annual']:>+8.2%}{r['sharpe']:>8.3f}"
              f"{r['mdd']:>+10.2%}{cost_pct:>+10.2%}")
    print(f"{'─'*100}")

print("\n解读：若 full 显著优于 fixedN/reversal → 纪律(退出规则)本身创造 alpha；")
print("      若三者接近 → 收益主要来自『入场信号+市场』，纪律只是换手与回撤的再分配。")
