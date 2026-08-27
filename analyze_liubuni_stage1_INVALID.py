# -*- coding: utf-8 -*-
"""
阶段1 反过拟合验证 (刘不牛 ETF 轮动, 开关版)
==========================================
复用 run_etf_rotation_liubuni 的 backtest / build_features / metrics。

目标: 验证阶段0 发现的"关止损 + 开 R² 过滤(0.3)"组合是否稳健,
      防止是样本内巧合 (呼应 UP 视频四步验证法之"样本外测试/压力测试")。

三部分:
  A. 组合全周期基线 (关止损 + 开R² 0.3) + 与 faithful 对照
  B. 固定参数 rolling OOS: 自然年 2020..2026 各段盲测 (参数事先固定, 看一致性)
  C. anchored walk-forward: 训练段[起点,t) 选最优 R²阈值, 测试段[t,t+1y] 盲测
     -> 演示"若真做样本内调参, 样本外是否仍有效"
  D. 参数敏感性网格: 动量窗口 × R²阈值 (固定关止损) 全周期累计收益矩阵

输出 CSV: etf_liubuni_stage1_rolling.csv / _walkforward.csv / _grid.csv
"""
import argparse
import numpy as np
import pandas as pd
import run_etf_rotation_liubuni as M

FULL_S, FULL_E = '2019-07-01', '2026-07-01'


def make_args(**kw):
    a = argparse.Namespace(**vars(M.parse_args()))
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def run_window(F, args, s, e, td):
    """临时改模块全局 START/END 跑一段区间, 返回 (metrics, 空仓率, nav)。"""
    old_s, old_e = M.START, M.END
    M.START, M.END = s, e
    try:
        nav, tr, pos = M.backtest(F, args, td)
    finally:
        M.START, M.END = old_s, old_e
    m = M.metrics(nav)
    empty = pos.isna().mean() if len(pos) else 0.0
    return m, empty, nav


def main():
    M.START, M.END = FULL_S, FULL_E
    td = M.load_trade_dates()
    cal = pd.Index(sorted(td))

    # ── A. 组合全周期基线 ──
    print('=' * 78)
    print('A. 组合全周期基线 (2019-07 ~ 2026-07, 平台真实成本)')
    print('=' * 78)
    # faithful
    fa, ea, _ = run_window(M.build_features(make_args(), cal), make_args(), FULL_S, FULL_E, td)
    # 关止损 + 开R2 0.3
    cfg = make_args(r2_filter=True, r2_threshold=0.3, stop_loss=-1.0)
    F25 = M.build_features(cfg, cal)  # 动量25固定
    cb, eb, _ = run_window(F25, cfg, FULL_S, FULL_E, td)
    print(f'  faithful(全开)       : 累计 {fa["total"]*100:+.1f}%  年化 {fa["ann"]*100:+.1f}%  '
          f'夏普 {fa["sharpe"]:.2f}  回撤 {fa["maxdd"]*100:+.1f}%  空仓 {ea*100:.1f}%')
    print(f'  关止损+开R²(0.3)    : 累计 {cb["total"]*100:+.1f}%  年化 {cb["ann"]*100:+.1f}%  '
          f'夏普 {cb["sharpe"]:.2f}  回撤 {cb["maxdd"]*100:+.1f}%  空仓 {eb*100:.1f}%')

    # 基准 (复用)
    mb = M.benchmark_buyhold('000300.SH', td)
    me = M.benchmark_equal_weight(td, cfg)
    print(f'  基准 沪深300买入持有 : 累计 {mb["total"]*100:+.1f}%  夏普 {mb["sharpe"]:.2f}')
    print(f'  基准 全池等权月度   : 累计 {me["total"]*100:+.1f}%  夏普 {me["sharpe"]:.2f}')

    # ── B. 固定参数 rolling OOS (自然年 2020..2026) ──
    print('\n' + '=' * 78)
    print('B. 固定参数 rolling OOS (关止损+开R²0.3, 每年盲测段, 参数事先固定)')
    print('=' * 78)
    rows = []
    for y in range(2020, 2027):
        s, e = f'{y}-01-01', f'{y}-12-31'
        m, em, _ = run_window(F25, cfg, s, e, td)
        yrs = m['yrs'] if m['yrs'] > 0 else np.nan
        ann = m['ann'] * 100 if pd.notna(m['ann']) else float('nan')
        rows.append(dict(year=y, total=m['total']*100, ann=ann, sharpe=m['sharpe'],
                         maxdd=m['maxdd']*100, empty=em*100))
        print(f'  {y}: 累计 {m["total"]*100:+.1f}%  年化 {ann:+.1f}%  夏普 {m["sharpe"]:.2f}  '
              f'回撤 {m["maxdd"]*100:+.1f}%  空仓 {em*100:.1f}%')
    roll = pd.DataFrame(rows)
    roll.to_csv('etf_liubuni_stage1_rolling.csv', index=False)
    pos_years = (roll['total'] > 0).sum()
    print(f'  -> 正收益年份 {pos_years}/{len(roll)}  累计(链乘) {((1+roll["total"]/100).prod()-1)*100:+.1f}%')

    # ── C. anchored walk-forward (训练段选最优R², 测试段盲测) ──
    print('\n' + '=' * 78)
    print('C. anchored walk-forward: 训练段[起点,t)选最优R²阈值, 测试段[t,t+1y]盲测')
    print('   (演示"若真做样本内调参, 样本外是否仍有效")')
    print('=' * 78)
    R2_GRID = [0.0, 0.3, 0.5, 0.7]
    wf_rows = []
    nav_chain = 1.0
    for y in range(2020, 2027):
        train_s, train_e = FULL_S, f'{y}-01-01'
        test_s, test_e = f'{y}-01-01', f'{y}-12-31'
        # 训练段: 在 R2_GRID 里选训练段夏普最高者 (关止损)
        best_thr, best_sh = None, -1e9
        for thr in R2_GRID:
            ta = make_args(r2_filter=True, r2_threshold=thr, stop_loss=-1.0)
            Ftr = M.build_features(ta, cal)
            mt, _, _ = run_window(Ftr, ta, train_s, train_e, td)
            if mt['sharpe'] > best_sh:
                best_sh, best_thr = mt['sharpe'], thr
        # 测试段: 用 best_thr (盲测)
        ca = make_args(r2_filter=True, r2_threshold=best_thr, stop_loss=-1.0)
        Fte = M.build_features(ca, cal)
        mt, et, navt = run_window(Fte, ca, test_s, test_e, td)
        nav_chain *= (1 + mt['total'])
        wf_rows.append(dict(year=y, tuned_r2=best_thr, train_sharpe=best_sh,
                            test_total=mt['total']*100, test_ann=mt['ann']*100,
                            test_sharpe=mt['sharpe'], test_maxdd=mt['maxdd']*100, test_empty=et*100))
        print(f'  {y}: 训练选 R²={best_thr:.1f}(训练夏普{best_sh:.2f}) -> '
              f'测试 累计 {mt["total"]*100:+.1f}%  年化 {mt["ann"]*100:+.1f}%  '
              f'夏普 {mt["sharpe"]:.2f}  回撤 {mt["maxdd"]*100:+.1f}%')
    wf = pd.DataFrame(wf_rows)
    wf.to_csv('etf_liubuni_stage1_walkforward.csv', index=False)
    wf_pos = (wf['test_total'] > 0).sum()
    print(f'  -> 盲测正收益年份 {wf_pos}/{len(wf)}  链乘累计 { (nav_chain-1)*100:+.1f}%  '
          f'平均测试夏普 {wf["test_sharpe"].mean():.2f}')

    # ── D. 参数敏感性网格 (固定关止损) ──
    print('\n' + '=' * 78)
    print('D. 参数敏感性网格 (关止损, 动量窗口 × R²阈值, 全周期累计收益%)')
    print('=' * 78)
    MOM_GRID = [10, 15, 20, 25, 30, 40]
    grid = {}
    for W in MOM_GRID:
        ca = make_args(momentum_window=W, stop_loss=-1.0)
        FW = M.build_features(ca, cal)
        row = {}
        for thr in R2_GRID:
            ta = make_args(momentum_window=W, r2_filter=True, r2_threshold=thr, stop_loss=-1.0)
            mt, _, _ = run_window(FW, ta, FULL_S, FULL_E, td)
            row[thr] = mt['total']*100
        grid[W] = row
        print('  W=%-2d: ' % W + '  '.join(f'R²={thr:.1f}:{row[thr]:+6.1f}%' for thr in R2_GRID))
    gd = pd.DataFrame(grid).T  # index=动量, columns=R²
    gd.columns = [f'R2={c:.1f}' for c in gd.columns]
    gd.index.name = 'mom_W'
    gd.to_csv('etf_liubuni_stage1_grid.csv')
    print('\n  (矩阵已存 etf_liubuni_stage1_grid.csv)')

    print('\n阶段1 完成。')


if __name__ == '__main__':
    main()
