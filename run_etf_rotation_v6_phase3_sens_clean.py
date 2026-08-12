"""
ETF轮动 V6 · Phase 3 参数敏感性 · 干净单变量版 (BV1uPu86sENG)
修复原 phase3 ③ 段的全局状态泄漏: 权重循环(0.7/0.3)污染了 RSRS窗口循环.
本脚本每次只改一个旋钮, 其余全部回锁基线默认, 保证单变量隔离.
基线默认: W_RSRS=0.6 W_MOM=0.4 RSRS_W=25 MOM_W=20 RECOVER_MA=20 COMBO_MIN=0
          UNIVERSE=4只后视镜池  MA: 纳指10/创业50 20/红利5+20/黄金3
"""
import run_etf_rotation_v6 as V6

DEFAULTS = dict(
    UNIVERSE={'510880.SH': '红利', '159949.SZ': '创业板50', '513100.SH': '纳指', '518880.SH': '黄金'},
    MA={'513100.SH': [10], '159949.SZ': [20], '510880.SH': [5, 20], '518880.SH': [3]},
    RSRS_W=25, MOM_W=20, W_RSRS=0.6, W_MOM=0.4,
    STOP=-0.03, START='2018-01-01', END='2026-07-01',
    INIT=100000.0, COMBO_MIN=0.0, RECOVER_MA=20,
)


def reset():
    for k, v in DEFAULTS.items():
        setattr(V6, k, v)


def set_one(**kw):
    reset()
    for k, v in kw.items():
        setattr(V6, k, v)


def run(tag, force_cash, **kw):
    set_one(**kw)
    F = V6.build_features()
    nav_s, trades, pos_s = V6.backtest(F, 0.0004, force_cash=force_cash)
    m = V6.metrics(nav_s)
    er = float((pos_s.isna()).mean()) if len(pos_s) else 0.0
    print(f'  [{tag}|fc={force_cash}] 累计{m["total"]*100:+.1f}% 年化{m["ann"]*100:+.1f}% '
          f'MDD{m["maxdd"]*100:+.1f}% 夏普{m["sharpe"]:.2f} 空仓{er*100:.1f}%')
    return m


def main():
    print('=' * 72)
    print('Phase 3 参数敏感性 · 干净单变量隔离 (基线=视频默认 0.6/0.4/W25/黄金MA3)')
    print('=' * 72)

    print('\n[A] 基线(默认全旋钮)')
    run('base', True)
    run('base', False)

    print('\n[B] 黄金MA (隔离, 其余回锁默认)')
    run('金MA3', True, MA=dict(DEFAULTS['MA']))
    run('金MA20', True, MA={'513100.SH': [10], '159949.SZ': [20], '510880.SH': [5, 20], '518880.SH': [20]})
    run('金MA3', False, MA=dict(DEFAULTS['MA']))
    run('金MA20', False, MA={'513100.SH': [10], '159949.SZ': [20], '510880.SH': [5, 20], '518880.SH': [20]})

    print('\n[C] 权重 W_RSRS/W_MOM (隔离, 其余回锁默认)')
    for wr, wm, lbl in [(0.6, 0.4, '0.6/0.4(基)'), (0.5, 0.5, '0.5/0.5'), (0.7, 0.3, '0.7/0.3')]:
        run(lbl, True, W_RSRS=wr, W_MOM=wm)
        run(lbl, False, W_RSRS=wr, W_MOM=wm)

    print('\n[D] RSRS窗口 (隔离, 其余回锁默认 0.6/0.4/W25)')
    for w, lbl in [(25, 'W25(基)'), (20, 'W20'), (30, 'W30')]:
        run(lbl, True, RSRS_W=w)
        run(lbl, False, RSRS_W=w)

    print('\n[E] 动量窗口 MOM_W (隔离)')
    for w, lbl in [(20, 'M20(基)'), (60, 'M60'), (120, 'M120')]:
        run(lbl, True, MOM_W=w)
        run(lbl, False, MOM_W=w)

    print('\n' + '=' * 72)
    print('注: 全部单变量隔离(每测一项先回锁其余默认); 成本固定0.04%.')


if __name__ == '__main__':
    main()
