#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
首板低开均值回归 — 严格 walk-forward 验证
=========================================
目的: 验证 Phase A 网格里的微弱正结果(N=1, 低开[3%,4%), 出场A, +17.55%)
      是否为"幸运子区间 / 多重比较偏差"假象, 而非稳健 edge。

方法 (无前视):
  - 全样本 2016-2026, 共 8 个滚动折(训练3年 / 测试1年, 滚动1年)
  - 每折: 用 TRAIN 段 (N×低开区间 8格网格) 选最优参数 → 在紧邻 OOS 段验证
  - 同时把"全局赢家" (N=1, 低开[3%,4%)) 直接钉死, 逐折测 OOS (不重选)
  - 统计: OOS 正折占比 / 几何链乘收益 / 是否集中在牛市

复用 run_limitup_reversion 的 load_daily/build_events/backtest/metrics。
所有决策仅用 T-1 及更早 + T日 open, 无未来泄露。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_limitup_reversion as R
import pandas as pd, numpy as np

GRID = [(1, 0.02, 0.03), (1, 0.03, 0.04), (1, 0.04, 0.05), (1, 0.02, 0.05),
        (20, 0.02, 0.03), (20, 0.03, 0.04), (20, 0.04, 0.05), (20, 0.02, 0.05)]
# 滚动折: (train_start, train_end, oos_start, oos_end)
FOLDS = [
    ("20160101", "20181231", "20190101", "20191231"),
    ("20170101", "20191231", "20200101", "20201231"),
    ("20180101", "20201231", "20210101", "20211231"),
    ("20190101", "20211231", "20220101", "20221231"),
    ("20200101", "20221231", "20230101", "20231231"),
    ("20210101", "20231231", "20240101", "20241231"),
    ("20220101", "20241231", "20250101", "20251231"),
    ("20230101", "20251231", "20260101", "20260826"),
]
WINNER = (1, 0.03, 0.04)  # 全局赢家(在完整样本上选出, 此处仅做 OOS 应用)

def main():
    os.makedirs(R.RESULT_DIR, exist_ok=True)
    df = R.load_daily()
    print("=" * 78)
    print("首板低开 — 严格 walk-forward (训练选参/紧邻OOS验证, 无前视)")
    print(f"网格 {len(GRID)} 格 | 折数 {len(FOLDS)} | 出场 A(开盘盈利即卖否则收盘)")
    print("=" * 78)

    # 全样本预构建事件(一次), backtest 按 buy_date 窗口切片
    ev_cache = {}
    for key in GRID:
        ev_cache[key] = R.build_events(df, key[0], key[1], key[2], 0.5)
    ev_winner = ev_cache[WINNER]

    fold_rows = []
    winner_rows = []
    for i, (ts, te, os_, oe) in enumerate(FOLDS):
        # --- 训练段选参(仅用 train) ---
        best, best_m = None, None
        for key, ev in ev_cache.items():
            nav, tr = R.backtest(ev, "A", ts, te)
            m = R.metrics(nav, tr)
            if best is None or m[0] > best_m[0]:
                best, best_m = key, m
        # --- OOS 验证(选出的配置) ---
        navO, trO = R.backtest(ev_cache[best], "A", os_, oe)
        mO = R.metrics(navO, trO)
        fold_rows.append(dict(
            fold=i + 1, train=f"{ts[:4]}-{te[:4]}", oos=f"{os_[:4]}-{oe[:4]}",
            sel_N=best[0], sel_lo=best[1], sel_hi=best[2],
            train_ret=best_m[0], train_ntr=best_m[6],
            oos_ret=mO[0], oos_annual=mO[1], oos_mdd=mO[2], oos_sharpe=mO[3],
            oos_win=mO[4], oos_per=mO[5], oos_ntr=mO[6]))
        # --- 全局赢家钉死 OOS ---
        navW, trW = R.backtest(ev_winner, "A", os_, oe)
        mW = R.metrics(navW, trW)
        winner_rows.append(dict(
            fold=i + 1, oos=f"{os_[:4]}-{oe[:4]}",
            oos_ret=mW[0], oos_win=mW[4], oos_ntr=mW[6]))

    fdf = pd.DataFrame(fold_rows)
    wdf = pd.DataFrame(winner_rows)

    # ---- 汇总 ----
    # 选参后 OOS 链乘(几何)
    link = (1 + fdf["oos_ret"]).prod() - 1
    pos_oos = (fdf["oos_ret"] > 0).sum()
    # 全局赢家 OOS 链乘
    link_w = (1 + wdf["oos_ret"]).prod() - 1
    pos_w = (wdf["oos_ret"] > 0).sum()

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("\n--- 每折: 训练选参 → OOS 验证 (出场 A) ---")
    print(fdf[["fold", "train", "oos", "sel_N", "sel_lo", "sel_hi",
              "train_ret", "train_ntr", "oos_ret", "oos_mdd", "oos_win", "oos_ntr"]]
          .to_string(index=False,
                     formatters={c: (lambda x: f"{x:+.2%}") for c in
                                 ["train_ret", "oos_ret", "oos_mdd", "oos_win"]}))
    print("\n--- 全局赢家 N=1, 低开[3%,4%) 逐折 OOS (不重选) ---")
    print(wdf.to_string(index=False,
          formatters={c: (lambda x: f"{x:+.2%}") for c in ["oos_ret", "oos_win"]}))

    print("\n" + "=" * 78)
    print("WALK-FORWARD 汇总")
    print("=" * 78)
    print(f"选参后 OOS: 正折 {pos_oos}/{len(fdf)} | 几何链乘收益 {link:+.2%} | "
          f"平均 OOS 笔数 {fdf['oos_ntr'].mean():.1f}")
    print(f"全局赢家 OOS: 正折 {pos_w}/{len(wdf)} | 几何链乘收益 {link_w:+.2%}")
    print(f"HS300 同期(2019-2026)买入持有参考: 见下方分年")

    # 全局赢家分年(看是否集中在牛市)
    print("\n--- 全局赢家 N=1[3%,4%) 分 OOS 年 (正=红牌前的侥幸) ---")
    for _, r in wdf.iterrows():
        flag = "  <-- 正" if r["oos_ret"] > 0 else ""
        print(f"  {r['oos']}: {r['oos_ret']:+.2%}  胜率{r['oos_win']:.1%}  "
              f"笔数{r['oos_ntr']}{flag}")

    fdf.to_csv(os.path.join(R.RESULT_DIR, "walkforward_folds.csv"), index=False)
    wdf.to_csv(os.path.join(R.RESULT_DIR, "walkforward_winner.csv"), index=False)
    print(f"\n结果已保存: {R.RESULT_DIR}/walkforward_folds.csv, walkforward_winner.csv")

if __name__ == "__main__":
    main()
