# ❌ INVALIDATED — 量化刘不牛「173倍」ETF 轮动 (BV1BqMX6TEx2)
# 判定：经严谨回测 + 月度调仓 bug 修复后，策略彻底证伪（关止损 -64.1%，远逊等权 +103.5%；
#       阶段3“池子洗清”结论被推翻为 0.0%ile condemned）。
# 历史：曾因 get_monthly_5th_trading_days bug（每年仅 2 个调仓日）一度给出虚假 +120.7% 被错误采纳，已纠正。
# 处置：本文件按清理惯例标记为无效策略归档，不作为可实盘/可复现参考。详见 etf_rebalance_bugfix_reverify.md。
# ============================================================

# -*- coding: utf-8 -*-
"""
阶段3：嵌套 Walk-Forward 选池 (selection bias / 池子过拟合检验)
问题：原10只池(及V6池)是否"天生过拟合"——即池子是事先用代码/后视镜挖出来的？
方法：
  选池窗 2019-07 ~ 2022-12：用收益/夏普从 28 只跨类全集中挑 top-N (N=10 liubuni / N=4 V6)
  测试窗 2023-01 ~ 2026-07：固定该池, 跑同一把尺子(关止损R²关轮动), 看 OOS 是否还赢
  对照：
    - 原始池 内样本(2019-2022) vs OOS(2023-2026)
    - 选池(ex-ante) OOS
    - 随机池 OOS 分布(R次) -> 原始池 OOS 分位
  判定：
    若 原始池 OOS ≈ 选池 OOS ≈ 随机 -> 池子只是 regime 红利, 非挖掘(但也无 alpha)
    若 原始池 OOS >> 选池 OOS(选池塌) -> 原始池像后视镜挖出 (过拟合嫌疑成立)
"""
import random
import numpy as np
import pandas as pd
import run_etf_rotation_liubuni as M
import analyze_liubuni_stage2_mc as A   # 复用 fast_build_features/fast_backtest + 28全集 + 跨轮缓存

random.seed(20260814)
R = 200  # 随机池控制次数

SEL_START, SEL_END = '2019-07-01', '2022-12-31'
TEST_START, TEST_END = '2023-01-01', '2026-07-01'

# 全集(列前2019)
UNIV = {c: v for c, v in A.CANDIDATE.items() if A.db_listed_before(c)}
print(f"选池全集: {len(UNIV)} 只 (列前2019-01-01)")

td = M.load_trade_dates()
cal = pd.Index(sorted(td))

# 全面板(close_adj) 一次构建, 供选池指标
panel = pd.DataFrame({c: A._etf_adj(c, cal)['close_adj'] for c in UNIV})

# 配置(同 +120.7% 基线: 关止损 / R²关 / 走弱切池开 / 保留0.9 / 动量25)
args = M.parse_args()
args.sweep = False
args.stop_loss = -1.0
args.r2_filter = False
args.r2_threshold = 0.0
args.weak_a_share = True
args.keep_threshold = 0.90
args.momentum_window = 25


def select_top(n, metric):
    """选池窗内按 metric 挑 top-n。返回 (pool_dict, 全排名rows)。"""
    rows = []
    for c in UNIV:
        sub = panel.loc[SEL_START:SEL_END, c].dropna()
        if len(sub) < 20:
            continue
        if metric == 'ret':
            val = sub.iloc[-1] / sub.iloc[0] - 1
        else:  # sharpe
            r = sub.pct_change().dropna()
            val = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
        if not np.isnan(val):
            rows.append((c, val))
    rows.sort(key=lambda x: -x[1])
    top = rows[:n]
    return {c: UNIV[c] for c, _ in top}, rows


def run_pool(pool_dict, start, end):
    """在 [start,end] 用同一把尺子跑轮动 + 等权基准。"""
    M.POOL = pool_dict
    M.ALL_CODES = list(pool_dict.keys())
    M.A_CODES = [c for c, (_, k) in pool_dict.items() if k == 'A']
    M.G_CODES = [c for c, (_, k) in pool_dict.items() if k == 'G']
    F = M.build_features(args, cal)
    M.START = start
    M.END = end
    nav, _, _ = M.backtest(F, args, td)
    m = M.metrics(nav)
    close = pd.DataFrame({c: A._etf_adj(c, cal)['close_adj'] for c in pool_dict}).loc[start:end]
    ret = close.pct_change().fillna(0).mean(axis=1)
    nav_eq = (1 + ret).cumprod() * M.INIT
    meq = M.metrics(nav_eq)
    return m, meq


# ── 原始池 ──
orig10 = {
    '510880.SH': ('红利', 'A'), '159949.SZ': ('创业板50', 'A'), '513100.SH': ('纳指', 'G'),
    '513500.SH': ('标普500', 'G'), '513520.SH': ('日经', 'G'), '159920.SZ': ('恒生', 'G'),
    '501018.SH': ('原油', 'G'), '518880.SH': ('黄金', 'G'), '511010.SH': ('国债', 'G'),
    '511880.SH': ('货币', 'G'),
}
v6_4 = {
    '510880.SH': ('红利', 'A'), '159949.SZ': ('创业板50', 'A'),
    '513100.SH': ('纳指', 'G'), '518880.SH': ('黄金', 'G'),
}

sel_ret10, _ = select_top(10, 'ret')
sel_sh10, _ = select_top(10, 'sharpe')
sel_ret4, _ = select_top(4, 'ret')
sel_sh4, _ = select_top(4, 'sharpe')

named = {
    '原10只': orig10,
    '选池-收益top10': sel_ret10,
    '选池-夏普top10': sel_sh10,
    'V6原4只': v6_4,
    'V6选池-收益top4': sel_ret4,
    'V6选池-夏普top4': sel_sh4,
}

print("\n=== 选池窗内样本(2019-2022) vs 测试窗OOS(2023-2026) ===")
results = {}
for name, pool in named.items():
    m_is, meq_is = run_pool(pool, SEL_START, SEL_END)
    m_oos, meq_oos = run_pool(pool, TEST_START, TEST_END)
    results[name] = dict(
        is_total=m_is['total'], is_ann=m_is['ann'], is_sharpe=m_is['sharpe'], is_eq=meq_is['total'],
        oos_total=m_oos['total'], oos_ann=m_oos['ann'], oos_sharpe=m_oos['sharpe'], oos_eq=meq_oos['total'],
        n=len(pool))
    print(f"  {name:14s} 内样本={results[name]['is_total']:+.1%} (等权{results[name]['is_eq']:+.1%}) | "
          f"OOS={results[name]['oos_total']:+.1%} (等权{results[name]['oos_eq']:+.1%}) 夏普OOS={results[name]['oos_sharpe']:.2f}")

# ── 随机控制 (OOS) ──
rand10, rand4 = [], []
for k in range(R):
    sub10 = random.sample(list(UNIV.keys()), 10)
    m10, _ = run_pool({c: UNIV[c] for c in sub10}, TEST_START, TEST_END)
    rand10.append(m10['total'])
    sub4 = random.sample(list(UNIV.keys()), 4)
    m4, _ = run_pool({c: UNIV[c] for c in sub4}, TEST_START, TEST_END)
    rand4.append(m4['total'])
    if (k + 1) % 50 == 0:
        print(f"  随机控制 {k+1}/{R} done", flush=True)
rand10 = np.array(rand10)
rand4 = np.array(rand4)

orig_oos = results['原10只']['oos_total']
v6_oos = results['V6原4只']['oos_total']
sel10_oos = results['选池-收益top10']['oos_total']
sel4_oos = results['V6选池-收益top4']['oos_total']

print(f"\n=== 随机池 OOS 分布 (R={R}) ===")
print(f"  随机10只: 均值={rand10.mean():.1%} 中位={np.median(rand10):.1%} P10={np.percentile(rand10,10):.1%} P90={np.percentile(rand10,90):.1%} 最小={rand10.min():.1%} 最大={rand10.max():.1%}")
print(f"  随机4只:  均值={rand4.mean():.1%} 中位={np.median(rand4):.1%}")

print(f"\n=== 一锤定音判定 ===")
p_orig = (rand10 < orig_oos).mean()
p_v6 = (rand4 < v6_oos).mean()
print(f"  原10只 OOS={orig_oos:+.1%} | 在随机10只OOS分布分位={p_orig:.1%} ({(1-p_orig)*100:.1f}%随机池比它差)")
print(f"  V6原4只 OOS={v6_oos:+.1%} | 在随机4只OOS分布分位={p_v6:.1%} ({(1-p_v6)*100:.1f}%随机池比它差)")
print(f"  选池-收益top10 OOS={sel10_oos:+.1%} (ex-ante选)  vs 原10只 OOS={orig_oos:+.1%}  -> 差={orig_oos-sel10_oos:+.1%}")
print(f"  V6选池-收益top4 OOS={sel4_oos:+.1%} (ex-ante选) vs V6原4只 OOS={v6_oos:+.1%}  -> 差={v6_oos-sel4_oos:+.1%}")

# 保存
out = pd.DataFrame(results).T
out.to_csv('etf_liubuni_stage3_nested_wf.csv', encoding='utf-8-sig')
pd.DataFrame({'rand10_oos': rand10, 'rand4_oos': rand4}).to_csv('etf_liubuni_stage3_rand.csv', index=False, encoding='utf-8-sig')
print("\n[已保存] etf_liubuni_stage3_nested_wf.csv / etf_liubuni_stage3_rand.csv")
