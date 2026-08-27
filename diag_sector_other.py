import run_sector_rotation as SR
import sqlite3, pandas as pd, numpy as np

con = sqlite3.connect(SR.DB)
df_raw, sw = SR.load_industry_indices(con); con.close()
piv = SR.pivot_ohlc(df_raw)
close_df = piv['close']; open_df = piv['open']
mask = (close_df.index >= SR.START) & (close_df.index <= SR.END)
close_df = close_df.loc[mask]; open_df = open_df.loc[mask]

mom = SR.compute_momentum(close_df, SR.MOM_W)
ma_t = SR.compute_ma(close_df, SR.MA_TREND)
bn, _ = SR.compute_benchmark(close_df)
ma_b = SR.compute_ma(pd.DataFrame({'bench': bn}), SR.MA_BENCH)['bench']
rd = SR.build_rebal_calendar(close_df.index.tolist(), SR.REBAL_DAY)
warm = close_df.index[SR.MA_TREND]
rd = [d for d in rd if d >= warm]
res = SR.run_backtest(close_df, open_df, rd, mom, ma_t, ma_b)
nav = res['nav']
bnv = res['bench']

print("== 年度收益 (对方引擎 MOM_W=%d/MA_TREND=%d/MA_BENCH=%d/TOP_K=%d) ==" % (SR.MOM_W, SR.MA_TREND, SR.MA_BENCH, SR.TOP_K))
for y in sorted(set(str(d)[:4] for d in nav.index)):
    sub = nav[[str(d)[:4] == y for d in nav.index]]
    bsub = bnv[[str(d)[:4] == y for d in bnv.index]]
    yr = sub.iloc[-1]/sub.iloc[0]-1 if len(sub) > 1 else np.nan
    yb = bsub.iloc[-1]/bsub.iloc[0]-1 if len(bsub) > 1 else np.nan
    print(f"  {y}: 策略 {yr:+.1%} | 基准 {yb:+.1%} | 超额 {yr-yb:+.1%}")

print("\n== 2014 月度持仓 ==")
for d in rd:
    if str(d)[:4] == '2014':
        print(f"  {d}: {res['holdings'].get(d)}")

print("\n== 数据尖刺检查 sw_industry_daily 2014-2015 单日 |ret|>15% ==")
con2 = sqlite3.connect(SR.DB)
gl = pd.read_sql("SELECT ts_code,trade_date,close FROM sw_industry_daily WHERE trade_date>=20140101 AND trade_date<=20151231", con2)
con2.close()
gl['ret'] = gl.groupby('ts_code')['close'].pct_change()
bad = gl[gl['ret'].abs() > 0.15].sort_values('ret')
print(f"  极端天数: {len(bad)}")
print(bad.head(25).to_string(index=False))

print("\n== 全样本单日 |ret|>15% (任何行业, 任意年份) ==")
con3 = sqlite3.connect(SR.DB)
allg = pd.read_sql("SELECT ts_code,trade_date,close FROM sw_industry_daily", con3)
con3.close()
allg['ret'] = allg.groupby('ts_code')['close'].pct_change()
badall = allg[allg['ret'].abs() > 0.15]
print(f"  极端天数: {len(badall)} (共 {len(allg)} 行)")
print(badall.sort_values('ret').head(15).to_string(index=False))
print(badall.sort_values('ret').tail(15).to_string(index=False))
