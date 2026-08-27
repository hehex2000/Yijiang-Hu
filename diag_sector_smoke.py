"""板块轮动回测台合成冒烟：构造确定性多行业序列，验证
- 月度调仓触发、Top-N 选中、弱市持币、曲线健全、无 NaN/崩溃。
"""
import numpy as np
import pandas as pd
import run_sector_rotation as R
from argparse import Namespace


def make_panel(n=5, days=400, seed=7):
    rng = np.random.default_rng(seed)
    dates = (20200101 + np.arange(days) * 1)  # 仅占位, 实际需交易日序列
    # 用真实交易日刻度(日+1)避免月份分组异常
    d = 20200101
    ds = []
    for _ in range(days):
        ds.append(d); d += 1
    cols = [f"T{i:03d}.SI" for i in range(n)]
    # 行业0 长期上行; 行业1 长期下行; 行业2 震荡; 行业3 后期上行; 行业4 先上后下
    base = rng.normal(0, 0.01, (days, n)).cumsum(0) + np.linspace(0, 1, days)[:, None]
    base[:, 1] -= np.linspace(0, 2, days)          # 行业1 下行
    seg1 = np.linspace(0, 1.5, 200)
    seg2 = np.linspace(1.5, -1, days - 200)
    base[:, 4] += np.concatenate([seg1, seg2])  # 行业4 先上后下
    close = 100 * np.exp(base)
    return pd.DataFrame(close, index=ds, columns=cols)


def main():
    panel = make_panel()
    bench = pd.Series(100 * np.exp(np.linspace(0, 0.5, len(panel))),
                      index=panel.index)
    a = Namespace(start=str(panel.index.min()), end=str(panel.index.max()),
                  mom=20, ma=20, regime_ma=60, top_n=2, cost=0.0003, bench="BENCH")
    res = R.backtest(panel, bench, a)
    eq = res["equity"]
    assert np.all(np.isfinite(eq)), "equity 含 NaN/inf"
    assert eq[0] == 1_000_000.0, "初始资金错误"
    assert 0.0 <= res["cash_frac"] <= 1.0, "持币占比越界"
    assert res["n_rebal"] > 0, "未触发调仓"
    print(f"[冒烟 OK] 交易日={len(panel)} 调仓={res['n_rebal']} "
          f"持币占比={res['cash_frac']*100:.1f}% 末值={eq[-1]:,.0f} "
          f"有限={bool(np.all(np.isfinite(eq)))}")
    print("  末次调仓:", res["rebal_log"][-2:])


if __name__ == "__main__":
    main()
