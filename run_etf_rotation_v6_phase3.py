"""
ETF轮动 V6 · Phase 3 反过拟合验证 (BV1uPu86sENG · 三门两月)
复用 run_etf_rotation_v6 引擎, 在两条冷却基线上跑三件套:
  基线A(force_cash=True) : 忠实~40%版, 止损后强制空仓等回补
  基线B(force_cash=False): ~0%轮动版, 冷却期可轮动到其他ETF
三件套:
  ① 扩展候选池: 354只2018前上市ETF中取跨资产类代表池, 看是否仍集中纳指/黄金(后视镜选池偏差)
  ② walk-forward: 2018-20训 / 2021-23验 / 2024-26样本外, 固定参数诚实标注
  ③ 参数敏感性: 黄金MA(3/20) / 权重(0.6·0.4,0.5·0.5,0.7·0.3) / RSRS窗口(25/20/30)
"""
import numpy as np
import pandas as pd
import run_etf_rotation_v6 as V6

ORIG_UNIVERSE = dict(V6.UNIVERSE)
ORIG_MA = dict(V6.MA)

# 扩展候选池: 跨资产类, 2018前已上市且全周期有数据(剔除588000/159980因2018缺数据)
EXT_UNIVERSE = {
    '510300.SH': '沪深300', '510500.SH': '中证500', '510050.SH': '上证50',
    '159915.SZ': '创业板', '511010.SH': '国债', '511260.SH': '十年国债',
    '518880.SH': '黄金', '518800.SH': '黄金2', '159934.SZ': '黄金3',
    '513100.SH': '纳指', '513500.SH': '标普', '513660.SH': '恒生',
    '159949.SZ': '创业板50', '512010.SH': '医药', '159928.SZ': '消费',
    '512660.SH': '军工', '512000.SH': '券商', '510880.SH': '红利',
    '501018.SH': '原油', '162411.SZ': '原油2',
}
EXT_MA = {c: [20] for c in EXT_UNIVERSE}  # 扩展池统一MA20趋势闸(一致口径)

WINDOWS = [('train', '2018-01-01', '2020-12-31'),
           ('valid', '2021-01-01', '2023-12-31'),
           ('test ', '2024-01-01', '2026-07-01')]

COSTS = [(0.0004, '0.04%'), (0.0015, '0.15%')]


def set_globals(universe, ma, start=None, end=None, recover=None,
                w_rsrs=None, w_mom=None, rsrs_w=None):
    V6.UNIVERSE = universe
    V6.MA = ma
    if start:
        V6.START = start
    if end:
        V6.END = end
    if recover is not None:
        V6.RECOVER_MA = recover
    if w_rsrs is not None:
        V6.W_RSRS = w_rsrs
    if w_mom is not None:
        V6.W_MOM = w_mom
    if rsrs_w is not None:
        V6.RSRS_W = rsrs_w


# 干净基线参数: run() 开头 reset_defaults() 回锁, 杜绝任何跨段全局泄漏(含"权重循环末值泄漏进RSRS窗口循环")
DEFAULTS = dict(UNIVERSE=ORIG_UNIVERSE, MA=ORIG_MA, RSRS_W=25, MOM_W=20,
                W_RSRS=0.6, W_MOM=0.4, STOP=-0.03, START='2018-01-01',
                END='2026-07-01', INIT=100000.0, COMBO_MIN=0.0, RECOVER_MA=20)


def reset_defaults():
    for k, v in DEFAULTS.items():
        setattr(V6, k, v)


def run(tag, force_cash, universe, ma, cost, start=None, end=None, **kw):
    reset_defaults()  # 每次从干净默认出发, 只应用显式覆盖, 杜绝跨段全局泄漏
    set_globals(universe, ma, start, end, **kw)
    F = V6.build_features()
    nav_s, trades, pos_s = V6.backtest(F, cost, force_cash=force_cash)
    m = V6.metrics(nav_s)
    er = float((pos_s.isna()).mean()) if len(pos_s) else 0.0
    dist = pos_s.value_counts(dropna=False)
    dist_s = ' '.join(f'{V6.UNIVERSE.get(k,"空仓") if k is not None else "空仓"}'
                      f'{v/len(pos_s)*100:.0f}%' for k, v in dist.items())
    print(f'  [{tag}|fc={force_cash}] 累计{m["total"]*100:+.1f}% 年化{m["ann"]*100:+.1f}% '
          f'MDD{m["maxdd"]*100:+.1f}% 夏普{m["sharpe"]:.2f} 空仓{er*100:.1f}%  持仓:{dist_s}')
    return m


def main():
    print('=' * 72)
    print('Phase 3 反过拟合三件套 · 两基线 (A=force_cash True~40% / B=False~0%)')
    print('=' * 72)

    # ① 扩展候选池 (原4只池 vs 扩展22只池), 成本0.04%
    print('\n① 扩展候选池 (成本0.04%, 看持仓是否仍集中纳指/黄金)')
    print('  -- 原4只后视镜池 --')
    run('orig', True, ORIG_UNIVERSE, ORIG_MA, 0.0004)
    run('orig', False, ORIG_UNIVERSE, ORIG_MA, 0.0004)
    print('  -- 扩展22只跨资产池 --')
    run('ext ', True, EXT_UNIVERSE, EXT_MA, 0.0004)
    run('ext ', False, EXT_UNIVERSE, EXT_MA, 0.0004)

    # ② walk-forward (原4只池, 0.04%), 三窗口
    print('\n② walk-forward (原4只池, 成本0.04%, 固定参数)')
    for name, s, e in WINDOWS:
        print(f'  -- {name} {s}~{e} --')
        run(name, True, ORIG_UNIVERSE, ORIG_MA, 0.0004, start=s, end=e)
        run(name, False, ORIG_UNIVERSE, ORIG_MA, 0.0004, start=s, end=e)

    # ③ 参数敏感性 (原4只池, 0.04%, 全周期) — run() 自带 reset_defaults, 单变量隔离无泄漏
    print('\n③ 参数敏感性 (原4只池, 成本0.04%, 全周期, 看收益是否炸裂)')

    def run3(tag, fc, **overrides):
        ma = overrides.pop('MA', ORIG_MA)
        run(tag, fc, ORIG_UNIVERSE, ma, 0.0004,
            start='2018-01-01', end='2026-07-01', **overrides)

    print('  -- 黄金MA(基线3 vs 20) --')
    ma3 = dict(ORIG_MA); ma3['518880.SH'] = [3]
    ma20 = dict(ORIG_MA); ma20['518880.SH'] = [20]
    run3('金MA3', True, MA=ma3)
    run3('金MA20', True, MA=ma20)
    run3('金MA3', False, MA=ma3)
    run3('金MA20', False, MA=ma20)
    print('  -- 权重 W_RSRS/W_MOM --')
    for wr, wm, lbl in [(0.6, 0.4, '0.6/0.4'), (0.5, 0.5, '0.5/0.5'), (0.7, 0.3, '0.7/0.3')]:
        run3(lbl, True, w_rsrs=wr, w_mom=wm)
        run3(lbl, False, w_rsrs=wr, w_mom=wm)
    print('  -- RSRS窗口 --')
    for w, lbl in [(25, 'W25'), (20, 'W20'), (30, 'W30')]:
        run3(lbl, True, rsrs_w=w)
        run3(lbl, False, rsrs_w=w)

    print('\n' + '=' * 72)
    print('注: 纳指/黄金 raw close(无现金分红); 扩展池统一MA20; 全部固定参数未重搜.')


if __name__ == '__main__':
    main()
