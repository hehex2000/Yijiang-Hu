# -*- coding: utf-8 -*-
"""
阶段2：扩展池蒙特卡洛 (选标偏差检验)
问题：关止损+R²关 在"原10只"上得 +120.7%，这是否只是恰好选了这10只的运气？
方法：从"事前固定的合理多资产全集"里每次随机抽10只，跑完全相同的配置，重复 K 次，
      看原10只的收益落在随机池收益分布的哪个分位。
      - 若原10只落在 90%+ 分位 → 是选标运气/后视镜 (UP视频Bug级纪律: 选池须事前固定)
      - 若接近中位 → 策略收益来自规则本身, 普适
全集：内置跨 A股/海外股票/商品/债券/货币/黄金 的代码清单(带A/G分类)，运行时查库校验
      min(trade_date) <= 2019-01-01 才纳入，保证随机池起点一致、可比。
"""
import sqlite3, random, sys
import numpy as np
import pandas as pd
import run_etf_rotation_liubuni as M

DB = M.DB
random.seed(20260814)
K = 300
N_PICK = 10

# ── 内置候选全集 (code: (name, class))  class 'A'=A股, 'G'=全球/非A股 ──
CANDIDATE = {
    # A股宽基/行业
    '510300.SH': ('沪深300', 'A'), '510500.SH': ('中证500', 'A'), '510050.SH': ('上证50', 'A'),
    '159915.SZ': ('创业板', 'A'), '159901.SZ': ('深100', 'A'), '510180.SH': ('上证180', 'A'),
    '159919.SZ': ('沪深300', 'A'), '512100.SH': ('中证1000', 'A'), '515800.SH': ('800ETF', 'A'),
    '512660.SH': ('军工', 'A'), '512010.SH': ('医药', 'A'), '512000.SH': ('券商', 'A'),
    '515030.SH': ('新能源车', 'A'), '512760.SH': ('芯片', 'A'), '159995.SZ': ('芯片', 'A'),
    '515050.SH': ('5G', 'A'), '512690.SH': ('酒', 'A'), '515790.SH': ('光伏', 'A'),
    '510880.SH': ('红利', 'A'), '159949.SZ': ('创业板50', 'A'),
    # 海外/跨境股票
    '513100.SH': ('纳指', 'G'), '513500.SH': ('标普500', 'G'), '513520.SH': ('日经', 'G'),
    '159920.SZ': ('恒生', 'G'), '513090.SH': ('香港证券', 'G'), '513180.SH': ('恒生科技', 'G'),
    '513030.SH': ('德国30', 'G'), '513080.SH': ('法国CAC', 'G'), '164824.SZ': ('印度', 'G'),
    # 商品/黄金
    '518880.SH': ('黄金', 'G'), '159934.SZ': ('黄金', 'G'), '159937.SZ': ('黄金', 'G'),
    '501018.SH': ('原油', 'G'), '159980.SZ': ('有色', 'G'), '159985.SZ': ('豆粕', 'G'),
    # 债券/货币 (避险)
    '511010.SH': ('国债', 'G'), '511260.SH': ('十年国债', 'G'), '511270.SH': ('城投债', 'G'),
    '511880.SH': ('货币', 'G'), '511990.SH': ('货币', 'G'), '159001.SZ': ('货币', 'G'),
}

CASH_BOND = {'511010.SH', '511260.SH', '511270.SH', '511880.SH', '511990.SH', '159001.SZ'}

# ── 跨轮数据缓存: 每只ETF/指数只从DB读一次 (避免300轮×14次连接拖慢+触发沙箱限时) ──
_ETFCACHE, _IDXCACHE = {}, {}


def _etf_adj(code, cal):
    if code not in _ETFCACHE:
        _ETFCACHE[code] = M.load_etf(code).reindex(cal).ffill()
    return _ETFCACHE[code]


def _idx_above_ma(code, cal):
    if code not in _IDXCACHE:
        s = M.load_index(code)
        _IDXCACHE[code] = (s > s.rolling(M.WEAK_MA).mean())
    return _IDXCACHE[code]


def fast_build_features(args, cal):
    """同 M.build_features, 但: (1) r2_filter=False 时跳过昂贵 R²; (2) 用跨轮缓存,
    每只ETF/指数只从DB读一次。蒙特卡洛全程 r2_filter 关, 结果与原函数一致。"""
    etf = {c: _etf_adj(c, cal) for c in M.ALL_CODES}
    close = pd.DataFrame({c: etf[c]['close_adj'] for c in M.ALL_CODES})
    openp = pd.DataFrame({c: etf[c]['open_adj'] for c in M.ALL_CODES})
    low = pd.DataFrame({c: etf[c]['low_adj'] for c in M.ALL_CODES})
    W = args.momentum_window
    mom = close.shift(W) / close - 1.0
    if args.r2_filter:
        r2 = close.rolling(W).apply(M.r2_only, raw=True)
    else:
        r2 = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    weak_below = pd.DataFrame({name: _idx_above_ma(code, cal)
                               for code, name in M.WEAK_INDEX.items()})
    weak_below = weak_below.reindex(cal).ffill().fillna(False).astype(bool)
    weak_count = (~weak_below).sum(axis=1)
    return dict(close=close, open=openp, low=low, mom=mom, r2=r2, weak_count=weak_count)


# 蒙特卡洛全程 r2_filter=False -> 用快速版覆盖 (不改源模块)
M.build_features = fast_build_features


def fast_backtest(F, args, trade_dates):
    """与 M.backtest 逻辑完全一致, 但把逐日 .loc 标签查价改为 numpy 数组下标, 提速~100x。
    用于蒙特卡洛(300轮)。月度量价决策仍用 F(标签, 仅~84次, 开销可忽略)。"""
    cal = F['close'].index
    cal_pos = {d: i for i, d in enumerate(cal)}
    codes = list(F['close'].columns)
    ccol = {c: i for i, c in enumerate(codes)}
    open_a = F['open'].to_numpy()
    close_a = F['close'].to_numpy()
    low_a = F['low'].to_numpy()
    weak_a = np.asarray(F['weak_count'].to_numpy(), dtype=float)
    row_to_date = {i: d for d, i in cal_pos.items()}

    monthly_5th = set(M.get_monthly_5th_trading_days(trade_dates))
    s0, e0 = M.START.replace('-', ''), M.END.replace('-', '')
    dates = [d for d in trade_dates if s0 <= d.strftime('%Y%m%d') <= e0]
    dates = pd.Index(sorted(dates))
    rows_all = np.array([cal_pos[d] for d in dates], dtype=int)
    valid_mask = ~np.isnan(close_a[rows_all]).all(axis=1)
    if not valid_mask.any():
        return pd.Series(dtype=float), [], pd.Series(dtype=object)
    first_valid = int(np.argmax(valid_mask))
    valid0 = dates[first_valid]
    dates = dates[dates >= valid0]
    rows = np.array([cal_pos[d] for d in dates], dtype=int)

    current = None
    cash = M.INIT
    shares = 0.0
    entry_price = 0.0
    nav = {}
    pos = {}
    trades = []
    pending = {}
    stop = args.stop_loss

    for d, ri in zip(dates, rows):
        if d in pending:
            tgt = pending.pop(d)
            if current is not None:
                p = open_a[ri, ccol[current]]
                if not np.isnan(p) and p > 0:
                    fee = M.calc_etf_fee('sell', p, shares)
                    cash = shares * p - fee
                    trades.append(('sell', current, d, p))
                shares = 0.0
                current = None
            if tgt is not None and cash > 0:
                p = open_a[ri, ccol[tgt]]
                if not np.isnan(p) and p > 0:
                    shares = cash / (p * (1 + M.COMMISSION_RATE + M.SLIPPAGE_RATE))
                    fee = M.calc_etf_fee('buy', p, shares)
                    cash = cash - (shares * p + fee)
                    current = tgt
                    entry_price = p
                    trades.append(('buy', tgt, d, p))
        if current is not None:
            cd = close_a[ri, ccol[current]]
            nav[d] = cash + (shares * cd if not np.isnan(cd) else 0.0)
        else:
            nav[d] = cash
        pos[d] = current
        if current is not None and stop > -0.9:
            low = low_a[ri, ccol[current]]
            if not np.isnan(low) and low <= entry_price * (1 + stop):
                nxt = row_to_date.get(ri + 1, d)
                pending[nxt] = None
        if d in monthly_5th:
            tgt = M.compute_target(F, d, current, args)
            if tgt != current:
                nxt = row_to_date.get(ri + 1, d)
                pending[nxt] = tgt

    nav_s = pd.Series(nav).sort_index().dropna()
    pos_s = pd.Series(pos).sort_index()
    return nav_s, trades, pos_s


M.backtest = fast_backtest


def db_listed_before(code, cut='2019-01-01'):
    c = sqlite3.connect(DB)
    n, mn = c.execute(
        "SELECT COUNT(*), MIN(trade_date) FROM etf_daily WHERE ts_code=?", (code,)).fetchone()
    c.close()
    if not n:
        return False
    return mn <= cut.replace('-', '')


def main():
    # 1. 校验全集可用性
    UNIV = {c: v for c, v in CANDIDATE.items() if db_listed_before(c)}
    A_UNIV = {c: v for c, v in UNIV.items() if v[1] == 'A'}
    G_UNIV = {c: v for c, v in UNIV.items() if v[1] == 'G'}
    print(f"可用全集: {len(UNIV)} 只 (A股 {len(A_UNIV)} / 全球 {len(G_UNIV)})")
    print("  A股:", ", ".join(f"{UNIV[c][0]}" for c in A_UNIV))
    print("  全球:", ", ".join(f"{UNIV[c][0]}" for c in G_UNIV))
    assert len(UNIV) >= N_PICK, "全集太小"

    # 2. 配置: 关止损 + R²关 + weak-a-share开 + keep0.9 (=原+120.7%配置)
    args = M.parse_args()
    args.sweep = False
    args.stop_loss = -1.0
    args.r2_filter = False
    args.r2_threshold = 0.0
    args.weak_a_share = True
    args.keep_threshold = 0.90
    args.momentum_window = 25
    M.START = '2019-07-01'
    M.END = '2026-07-01'
    td = M.load_trade_dates()
    cal = pd.Index(sorted(td))

    # 原10只基准 (用原POOL重跑确认)
    M.POOL = M.__dict__.get('POOL', None)
    # 直接用模块原始 POOL
    orig_pool = {
        '510880.SH': ('红利', 'A'), '159949.SZ': ('创业板50', 'A'), '513100.SH': ('纳指', 'G'),
        '513500.SH': ('标普500', 'G'), '513520.SH': ('日经', 'G'), '159920.SZ': ('恒生', 'G'),
        '501018.SH': ('原油', 'G'), '518880.SH': ('黄金', 'G'), '511010.SH': ('国债', 'G'),
        '511880.SH': ('货币', 'G'),
    }
    M.POOL = orig_pool
    M.ALL_CODES = list(orig_pool.keys())
    M.A_CODES = [c for c, (_, k) in orig_pool.items() if k == 'A']
    M.G_CODES = [c for c, (_, k) in orig_pool.items() if k == 'G']
    F = M.build_features(args, cal)
    nav0, _, _ = M.backtest(F, args, td)
    m0 = M.metrics(nav0)
    orig_total = m0['total']
    print(f"\n[基准] 原10只 关止损R²关: 累计={orig_total:.1%} 年化={m0['ann']:.1%} 夏普={m0['sharpe']:.2f}")

    # 等权基准(原10只)
    eq0 = equal_weight(orig_pool.keys(), cal, td)
    print(f"[基准] 原10只 等权月度: 累计={eq0['total']:.1%}")

    # 3. 蒙特卡洛
    recs = []
    errs = []
    for k in range(K):
        subset = random.sample(list(UNIV.keys()), N_PICK)
        pool = {c: UNIV[c] for c in subset}
        M.POOL = pool
        M.ALL_CODES = subset
        M.A_CODES = [c for c in subset if pool[c][1] == 'A']
        M.G_CODES = [c for c in subset if pool[c][1] == 'G']
        try:
            F = M.build_features(args, cal)
            nav, _, _ = M.backtest(F, args, td)
            m = M.metrics(nav)
            eq = equal_weight(subset, cal, td)
            n_g = len(M.G_CODES)
            recs.append(dict(
                trial=k, total=m['total'], ann=m['ann'], sharpe=m['sharpe'], mdd=m['maxdd'],
                n_A=len(M.A_CODES), n_G=n_g,
                has_cashbond=len(CASH_BOND & set(subset)) > 0,
                eq_total=eq['total'],
            ))
        except Exception as e:
            errs.append((k, subset, repr(e)))
        if (k + 1) % 25 == 0:
            print(f"  MC {k+1}/{K} done (ok={len(recs)} err={len(errs)})", flush=True)

    if errs:
        print(f"\n[警告] {len(errs)}/{K} 个随机池报错(已跳过), 例如:", flush=True)
        for k, sub, e in errs[:5]:
            print(f"  trial{k}: {sub} -> {e}", flush=True)
    df = pd.DataFrame(recs)
    df.to_csv('etf_liubuni_stage2_mc.csv', index=False, encoding='utf-8-sig')

    # 4. 统计
    tot = df['total']
    eqv = df['eq_total']
    print(f"\n=== 蒙特卡洛 K={K} (每次随机抽{N_PICK}只, 关止损R²关) ===")
    print(f"随机池策略 累计: 均值={tot.mean():.1%} 中位={tot.median():.1%} "
          f"P10={tot.quantile(.1):.1%} P90={tot.quantile(.9):.1%} 最小={tot.min():.1%} 最大={tot.max():.1%}")
    print(f"随机池等权 累计: 均值={eqv.mean():.1%} 中位={eqv.median():.1%}")
    print(f"正收益占比(策略): {(tot>0).mean():.1%}  超自身等权占比: {(tot>eqv).mean():.1%}")
    # 原10只分位
    pct_rank = (tot < orig_total).mean()  # 原值超过了多少比例随机池
    print(f"\n[关键] 原10只 +{orig_total:.1%} 在随机池分布中的分位 = {pct_rank:.1%} "
          f"(即约 {(1-pct_rank)*100:.1f}% 的随机池比它更好, {pct_rank:.1%} 比它更差)")
    print(f"原10只 vs 随机池均值: {orig_total - tot.mean():+.1%}   原10只 vs 随机池中位: {orig_total - tot.median():+.1%}")

    # 5. 含全球资产 vs 全A股池
    has_g = df[df['n_G'] > 0]
    all_a = df[df['n_A'] == N_PICK]
    if len(all_a) > 5:
        print(f"\n[对照] 全A股池(10只纯A股) n={len(all_a)}: 策略均值={all_a['total'].mean():.1%} 等权均值={all_a['eq_total'].mean():.1%}")
    print(f"[对照] 含全球资产池 n={len(has_g)}: 策略均值={has_g['total'].mean():.1%} 等权均值={has_g['eq_total'].mean():.1%}")

    # 6. 含避险(货基/国债)池 vs 不含
    cb = df[df['has_cashbond']]
    ncb = df[~df['has_cashbond']]
    print(f"[对照] 含避险(货基/国债)池 n={len(cb)}: 策略均值={cb['total'].mean():.1%}")
    print(f"[对照] 不含避险池 n={len(ncb)}: 策略均值={ncb['total'].mean():.1%}")


def equal_weight(codes, cal, td):
    """子集等权月度再平衡基准 (pct_chg 复权净值)。与策略同区间 [START:END], 走跨轮缓存。"""
    close = pd.DataFrame({c: _etf_adj(c, cal)['close_adj'] for c in codes})
    close = close.loc[M.START:M.END]
    ret = close.pct_change().fillna(0).mean(axis=1)
    nav = (1 + ret).cumprod() * M.INIT
    return M.metrics(nav)


if __name__ == '__main__':
    main()
