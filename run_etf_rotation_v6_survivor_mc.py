"""
ETF轮动 V6 — 全池蒙特卡洛幸存者偏差终裁 (BV1uPu86sENG)
=========================================================
核心问题: 视频从 354 只(2018前上市)里挑了 4 只赢家(红利/创业50/纳指/黄金),
复现年化只有 ~18%(干净版), 远不及宣称 43.56%。这 4 只是"后视镜运气"吗?

方法(统计上最干净的幸存者偏差检验):
  1. 取全市场 354 只中"2018前上市且存活至2025-06"的 352 只作为合格池。
  2. 对该池每只 ETF 预计算原始特征(RSRS斜率×R²、20日动量、MA20)。
  3. 每次从 352 只【无放回随机抽 4 只】, 在【这 4 只内部】做横截面 z-score、
     组合分=0.6*z(RSRS)+0.4*z(动量)、前二闸门(组合第一 且 两单项均前二)、
     独立 MA20 闸、统一 RECOVER_MA=20 冷却、周五决策周一开盘 —— 即视频策略的
     【完全同构】引擎, 只是池从"视频挑的4只"换成"随机4只"。
  4. 重复 N=2000 次(随机种子固定可复现), 记录每次累计收益。
  5. 把视频的 4 只赢家也用【完全同一引擎】(统一MA20)跑一遍, 看它在 2000 次
     随机分布里排第几百分位。

为什么统一 MA20 而非视频的定制 MA(红利5+20/黄金3/纳指10/创业50 20):
  随机池没有"为每只调出最优MA"的合法依据 —— 定制 MA 本身就是 per-asset 调参
  (另一种过拟合维度)。统一 MA20 是 unbiased 选择, 且这正是扩展池22只用的口径。
  注: 视频4只在定制MA下干净基线为 +321%(见主报告); 本脚本统一MA20下其值会更低,
  但若仍排在全分布前列, 则"挑这4只"的运气成分更强。

输出: 随机4子集收益分布(均值/中位/分位), 视频4只的累计收益与百分位,
      以及"超越大盘/超越视频基线"的比例。
"""
import sqlite3
import random
import numpy as np
import pandas as pd
import run_etf_rotation_v6 as V6

DB = V6.DB
RSRS_W = V6.RSRS_W
MOM_W = V6.MOM_W
W_RSRS = V6.W_RSRS
W_MOM = V6.W_MOM
RECOVER_MA = 20
START = '2018-01-01'
END = '2026-07-01'
INIT = 100000.0
COST = 0.0004
N = 2000
SEED = 42
WIN = ['510880.SH', '159949.SZ', '513100.SH', '518880.SH']  # 视频4赢家


def build_master():
    c = sqlite3.connect(DB)
    cur = c.cursor()
    cur.execute(
        "SELECT ts_code FROM (SELECT ts_code FROM etf_daily GROUP BY ts_code "
        "HAVING MIN(trade_date)<='20180101' AND MAX(trade_date)>='20250601')")
    pool = [r[0] for r in cur.fetchall()]
    c.close()
    print(f'[MC] 合格池(上市<=2018 & 存活>=2025-06): {len(pool)} 只')
    series = {code: V6.load_etf(code) for code in pool}
    cal = pd.Index(sorted(set().union(*[set(df.index) for df in series.values()])))
    rsrs_raw, mom_raw = {}, {}
    close_adj, open_adj, low_adj, ma20 = {}, {}, {}, {}
    for code, df in series.items():
        d = df.reindex(cal).ffill()
        rsrs_raw[code] = d['close_adj'].rolling(RSRS_W).apply(V6.rsrs_quality, raw=True)
        mom_raw[code] = d['close_adj'].rolling(MOM_W + 1).apply(
            lambda x: x[-1] / x[0] - 1.0, raw=True)
        close_adj[code] = d['close_adj']
        open_adj[code] = d['open_adj']
        low_adj[code] = d['low_adj']
        ma20[code] = d['close_adj'].rolling(20).mean()
    master = dict(
        cal=cal,
        rsrs_df=pd.DataFrame(rsrs_raw),
        mom_df=pd.DataFrame(mom_raw),
        close_df=pd.DataFrame(close_adj),
        open_df=pd.DataFrame(open_adj),
        low_df=pd.DataFrame(low_adj),
        ma_frames={(code, 20): ma20[code] for code in pool},
    )
    print('[MC] 全池特征构建完成')
    return master, pool


def run_subset(master, codes, force_cash=True):
    rs = master['rsrs_df'][codes]
    mo = master['mom_df'][codes]
    zr = rs.sub(rs.mean(axis=1), axis=0).div(rs.std(axis=1), axis=0).replace(
        [np.inf, -np.inf], 0).fillna(0)
    zm = mo.sub(mo.mean(axis=1), axis=0).div(mo.std(axis=1), axis=0).replace(
        [np.inf, -np.inf], 0).fillna(0)
    combo = W_RSRS * zr + W_MOM * zm
    rc = combo.rank(axis=1, ascending=False)
    rr = zr.rank(axis=1, ascending=False)
    rm = zm.rank(axis=1, ascending=False)
    F = dict(cal=master['cal'],
             close_df=master['close_df'][codes],
             open_df=master['open_df'][codes],
             low_df=master['low_df'][codes],
             combo=combo, rank_combo=rc, rank_rsrs=rr, rank_mom=rm,
             ma_frames={(c, 20): master['ma_frames'][(c, 20)] for c in codes})
    V6.UNIVERSE = {c: c for c in codes}
    V6.MA = {c: [20] for c in codes}
    V6.RECOVER_MA = RECOVER_MA
    V6.COMBO_MIN = 0.0
    V6.START = START
    V6.END = END
    V6.INIT = INIT
    nav_s, tr, ps = V6.backtest(F, COST, force_cash=force_cash)
    m = V6.metrics(nav_s)
    er = float((ps.isna()).mean()) if len(ps) else 0.0
    return dict(tot=m['total'] * 100, ann=m['ann'] * 100,
                sharpe=m['sharpe'], maxdd=m['maxdd'] * 100, empty=er * 100)


def main():
    master, pool = build_master()
    random.seed(SEED)
    tot, ann, sharpe, maxdd, empty = [], [], [], [], []
    for i in range(N):
        codes = random.sample(pool, 4)
        r = run_subset(master, codes)
        tot.append(r['tot']); ann.append(r['ann']); sharpe.append(r['sharpe'])
        maxdd.append(r['maxdd']); empty.append(r['empty'])
        if (i + 1) % 250 == 0:
            print(f'[MC] {i + 1}/{N} 完成')
    tot = np.array(tot)
    print('\n' + '=' * 64)
    print(f'蒙特卡洛结果: {N} 次随机4子集 (统一MA20, 成本0.04%, force_cash=True)')
    print('=' * 64)
    print(f'  累计收益%  均值={tot.mean():.1f}  中位={np.median(tot):.1f}  '
          f'最小={tot.min():.1f}  最大={tot.max():.1f}')
    for p in [5, 25, 50, 75, 90, 95, 99]:
        print(f'   P{p:<2} = {np.percentile(tot, p):+.1f}%')
    print(f'  年化%      均值={np.mean(ann):.1f}  中位={np.median(ann):.1f}')
    print(f'  夏普       均值={np.mean(sharpe):.2f}  中位={np.median(sharpe):.2f}')
    print(f'  最大回撤%  中位={np.median(maxdd):.1f}')
    print(f'  空仓率%    中位={np.median(empty):.1f}')
    # 视频4赢家(同引擎, 统一MA20)
    w = run_subset(master, WIN)
    wfc = run_subset(master, WIN, force_cash=False)
    pct = float(np.mean(tot <= w['tot'])) * 100
    print('\n' + '-' * 64)
    print('视频4赢家(统一MA20, 同引擎):')
    print(f"  force_cash=True : 累计 {w['tot']:+.1f}%  年化 {w['ann']:+.1f}%  "
          f"夏普 {w['sharpe']:.2f}  回撤 {w['maxdd']:+.1f}%  空仓 {w['empty']:.1f}%")
    print(f"  force_cash=False: 累计 {wfc['tot']:+.1f}%  年化 {wfc['ann']:+.1f}%")
    print(f'  → 在 {N} 次随机4子集中, 视频4只排第 {pct:.2f} 百分位 '
          f'(即 {(100 - pct):.2f}% 的随机组合不如它)')
    print(f'  随机组合收益 > 视频4只的比例: {(tot > w["tot"]).mean() * 100:.2f}%')
    print(f'  随机组合收益 > 0 的比例(赚钱): {(tot > 0).mean() * 100:.2f}%')
    print(f'  随机组合收益 > +1886%(视频宣称)的比例: {(tot > 1886).mean() * 100:.4f}%')
    print(f'  随机组合年化 > +43.56%(视频宣称)的比例: '
          f'{(np.array(ann) > 43.56).mean() * 100:.4f}%')
    # 与扩展池(22只含4赢家)对照
    print('\n对照: 扩展22只池年化 4.1%(fc=T)/8.1%(fc=F); 本MC中位年化 '
          f'{np.median(ann):.1f}%')


if __name__ == '__main__':
    main()
