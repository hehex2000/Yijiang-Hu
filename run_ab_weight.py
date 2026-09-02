# -*- coding: utf-8 -*-
"""
A 档归因闭环：股息率加权 vs 等权
================================
用 M1 已校验费用/换手的 run_sim 引擎，2015-2025 / 月度首日调仓 / N=5 / 无择时 / 本金100万。
隔离「选股器+池+加权」三变量，做 2×2(+MACD档) 归因：
  选择器:
    NEW    : M1.select_div_low_vol        (zz800, 无逐只MACD过滤)
    OLD-H  : RM.select_dividend_low_vol_stocks (hs300, 无逐只MACD过滤)
    OLD-H-F: 同上但保留逐只MACD金叉过滤
  加权  : equal(等权) | div(股息率加权, 以选股时点 dv_ttm 为权重, 归一化)
关键修正：vol_lookup / macd_state 基于全历史；div 权重用 ex-ante 选股时点 dv_ttm。
注意：平台 run_backtest 默认 sizing='equal'（无股息率加权方案），本脚本的 div 档是
      "若改用股息率加权会怎样"的反事实，用于量化加权口径本身的贡献。
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

import run_daily20_macd as D
import run_monthly_rebalance as RM
import macd_plugin_validate as M
from run_monthly_rebalance import get_trade_dates, get_conn

START, END = '20150101', '20251231'
TOPN, CAP = 5, 1_000_000

# ── NAV 计价口径（2026-09-01 新增；2026-09-02 起默认 hfq，raw 变 opt-in）──
# raw = 不复权（不含分红、不处理送转）；hfq = 后复权（总回报，须配全收益基准）
# 用法：python run_ab_weight.py                    → hfq（默认，推荐）
#       python run_ab_weight.py --price-mode raw   → 旧口径，仅用于复现历史结论
PRICE_MODE = 'hfq'
if '--price-mode' in sys.argv:
    _i = sys.argv.index('--price-mode')
    if _i + 1 < len(sys.argv):
        PRICE_MODE = sys.argv[_i + 1]
if PRICE_MODE not in ('raw', 'hfq'):
    print(f"[错误] --price-mode 只能是 raw/hfq，收到 {PRICE_MODE}")
    sys.exit(1)
_PM_TAG = '_hfq' if PRICE_MODE == 'hfq' else ''

# ── 数据：load_closes 返回全历史(2010-2025) ──
# 信号侧（波动率 + 逐只 MACD 金叉）**恒定锁 raw**：隔离变量，
# 让 raw/hfq 双跑的差异纯粹来自 NAV 口径，否则无法归因。
_, raw_full_cf = D.load_closes(hfq=False)
# NAV 侧按 PRICE_MODE 切换
if PRICE_MODE == 'hfq':
    _, full_cf = D.load_closes(hfq=True)
else:
    full_cf = raw_full_cf
closes = full_cf.loc[(full_cf.index >= int(START)) & (full_cf.index <= int(END))]
closes_ff = closes.ffill()

# 波动率查表（window=120，与 M1 / config 对齐）—— 必须基于全历史
vol_lookup = D.build_vol_lookup(raw_full_cf, window=120)
_vol_dict = {c: vol_lookup[c].dropna().to_dict() for c in vol_lookup.columns}

def _fast_calvol(code, actual_date, window=None):
    d = _vol_dict.get(str(code))
    if not d:
        return None
    cand = [k for k in d if k <= int(actual_date)]
    if not cand:
        return None
    return d[max(cand)]

def _fast_macd_state(ts_code, trade_date, is_index_signal=False, regime_code=None,
                     regime_is_index=False, mode="golden"):
    """基于【全历史】closes 计算逐只 MACD 金叉状态（信号侧，恒定 raw）。"""
    s = raw_full_cf.get(str(ts_code))
    if s is None:
        return "death"
    c = s[s.index <= int(trade_date)].dropna()
    if len(c) < 35:
        return "death"
    v = c.values.astype(float)
    ef = pd.Series(v).ewm(span=12, adjust=False).mean().values
    es = pd.Series(v).ewm(span=26, adjust=False).mean().values
    dif = ef - es
    dea = pd.Series(dif).ewm(span=9, adjust=False).mean().values
    return "golden" if dif[-1] > dea[-1] else "death"

RM.calc_volatility = _fast_calvol
RM.macd_state = _fast_macd_state

# dv 查表（ex-ante：选股时点 dv_ttm，全历史 daily_basic）
def _dv_dict(codes, prev_td):
    if not codes:
        return {}
    conn = get_conn()
    ph = ",".join("?" * len(codes))
    rows = conn.execute(f"""
        SELECT ts_code, dv_ttm FROM daily_basic
        WHERE ts_code IN ({ph})
          AND trade_date = (SELECT MAX(trade_date) FROM daily_basic WHERE trade_date <= ?)
          AND dv_ttm > 0
    """, list(codes) + [prev_td]).fetchall()
    conn.close()
    return {r[0]: float(r[1]) for r in rows}

# ── 月度首日调仓（与 M1 main 完全一致）──
trade_dates = get_trade_dates(START, END)
dates_i = [int(d) for d in trade_dates]
month_starts = set()
prev_ym = None
for d in trade_dates:
    ym = d[:6]
    if ym != prev_ym:
        month_starts.add(d); prev_ym = ym

golden = [True] * len(trade_dates)

# ── 选择器封装：统一返回 dict{code: dv_ttm(可为None)} ──
# 关键：保留【全部】入选 code（缺失 dv→None 由引擎 median 填补），
# 这样 equal 与 div 两档篮子数量恒定、仅权重分布不同，隔离加权变量。
def new_div(prev_td, top_n, vl=None, verbose=False):
    codes = D.select_div_low_vol(prev_td, top_n, vl, verbose=False)
    dv = _dv_dict(codes, prev_td)
    return {c: dv.get(c) for c in codes}

def _old_div(prev_td, top_n, pool, use_filter, vl=None, verbose=False):
    saved = RM.macd_state
    if not use_filter:
        RM.macd_state = lambda *a, **k: "golden"
    try:
        df = RM.select_dividend_low_vol_stocks(prev_td, top_n=top_n,
                                              macd_filter_mode="golden", stock_pool=pool)
    finally:
        RM.macd_state = saved
    if df is None or len(df) == 0:
        return {}
    codes = df['ts_code'].tolist()
    if 'dv_ttm' in df.columns:
        spec = {r['ts_code']: (float(r['dv_ttm']) if (r['dv_ttm'] is not None and r['dv_ttm'] > 0)
                               else None) for _, r in df.iterrows()}
    else:
        spec = _dv_dict(codes, prev_td)
    return {c: spec.get(c) for c in codes}

def old_hs_nofilter_div(prev_td, top_n, vl=None, verbose=False):
    return _old_div(prev_td, top_n, "000300.SH", use_filter=False)

def old_hs_filter_div(prev_td, top_n, vl=None, verbose=False):
    return _old_div(prev_td, top_n, "000300.SH", use_filter=True)

def run_one(sel_fn, tag, weight_mode='equal'):
    t0 = time.time()
    nav, tr, st = D.run_sim(trade_dates, dates_i, golden, closes, closes_ff,
                            TOPN, CAP, sel_fn, vol_lookup,
                            rebal_freq='monthly', month_starts=month_starts,
                            verbose=False, weight_mode=weight_mode)
    rb, ab, mdb, sb = M.metrics(pd.Series(nav))
    msg = (f"{tag}[{weight_mode}]: 总收={rb*100:7.2f}% 年化={ab*100:6.2f}% MDD={mdb*100:7.2f}% "
           f"Sharpe={sb:5.2f} 年化换手={st['turnover']/CAP*100:6.1f}% "
           f"总费={st['total_fee']:>9,.0f}  重选={st['n_reselect']}  ({time.time()-t0:.1f}s)")
    print(msg, flush=True)
    return nav

print("=" * 100, flush=True)
print(f"A/B 加权归因 {START}~{END} | N={TOPN} | 月度首日调仓 | 无择时 | 本金{CAP:,} | equal vs div(dv_ttm)", flush=True)
print(f"【NAV 计价口径】{PRICE_MODE}"
      + ("（后复权·总回报，须与全收益基准比）" if PRICE_MODE == 'hfq' else "（不复权·不含分红，须与价格指数比）"),
      flush=True)
print("=" * 100, flush=True)

res = {}
res['NEW_eq']   = run_one(new_div,              "NEW(zz800,无MACD)")
res['NEW_div']  = run_one(new_div,              "NEW(zz800,无MACD)", 'div')
res['OLD_H_eq'] = run_one(old_hs_nofilter_div,  "OLD-H(hs300,无MACD)")
res['OLD_H_div']= run_one(old_hs_nofilter_div,  "OLD-H(hs300,无MACD)", 'div')
res['OLD_HF_eq']= run_one(old_hs_filter_div,    "OLD-H-F(hs300,含MACD)")
res['OLD_HF_div']=run_one(old_hs_filter_div,    "OLD-H-F(hs300,含MACD)", 'div')

# 基准（口径必须跟随 NAV：raw→价格指数 / hfq→全收益，两端同含或同不含分红）
# 否则 hfq NAV 对比价格基准会把"策略多吃到的股息"误算成选股 alpha。
import bench_index as bi
b800, b300 = None, None
_bmeta = None
for _code, _nm in [('000906.SH', '800'), ('000300.SH', '300')]:
    if PRICE_MODE == 'hfq':
        _df, _meta = bi.load_benchmark(_code, START, END, nav_price_mode='hfq')
        if _df is not None and len(_df) >= 2:
            _s = pd.Series((_df['close'] / float(_df['close'].iloc[0])).values,
                           index=_df['trade_date'])
            if _nm == '800':
                b800, _bmeta = _s, _meta
            else:
                b300 = _s
            continue
    _s = M.load_base_index(_code, START, END)
    if _nm == '800':
        b800 = _s
    else:
        b300 = _s
if _bmeta:
    print(f"  基准口径：{bi.benchmark_meta_label(_bmeta)}（NAV 口径：{PRICE_MODE}）", flush=True)
rb8, ab8, md8, _ = M.metrics(b800)
rb3, ab3, md3, _ = M.metrics(b300)
print(f"基准中证800: 总收={rb8*100:7.2f}% 年化={ab8*100:6.2f}% MDD={md8*100:7.2f}%", flush=True)
print(f"基准沪深300: 总收={rb3*100:7.2f}% 年化={ab3*100:6.2f}% MDD={md3*100:7.2f}%", flush=True)

out = pd.DataFrame({'trade_date': trade_dates,
                    'nav_new_eq': res['NEW_eq'], 'nav_new_div': res['NEW_div'],
                    'nav_old_hs_nf_eq': res['OLD_H_eq'], 'nav_old_hs_nf_div': res['OLD_H_div'],
                    'nav_old_hs_f_eq': res['OLD_HF_eq'], 'nav_old_hs_f_div': res['OLD_HF_div'],
                    'nav_zz800': b800.values, 'nav_hs300': b300.values})
os.makedirs('data/results/daily20_divlow', exist_ok=True)
out.to_csv(f'data/results/daily20_divlow/ab_weight{_PM_TAG}_20150101_20251231.csv',
           index=False, encoding='utf-8-sig')
print(f"\nNAV → data/results/daily20_divlow/ab_weight{_PM_TAG}_20150101_20251231.csv", flush=True)
print("DONE", flush=True)
