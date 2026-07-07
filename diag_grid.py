import sqlite3, os
import pandas as pd, numpy as np
DB = os.path.join(r"C:\Users\99395\WorkBuddy\multi_factor_selection","data","tu-sharedata","astock_daily.db")
c = sqlite3.connect(DB)
df = pd.read_sql("SELECT trade_date,open,high,low,close FROM etf_daily WHERE ts_code='510300.SH' AND trade_date BETWEEN '20180102' AND '20260703' ORDER BY trade_date", c)
af = pd.read_sql("SELECT trade_date,adj_factor FROM etf_adj_factor WHERE ts_code='510300.SH' AND trade_date BETWEEN '20180102' AND '20260703' ORDER BY trade_date", c)
c.close()
df['trade_date']=df['trade_date'].astype(str)
af['trade_date']=af['trade_date'].astype(str)
df = df.merge(af, on='trade_date', how='left')
for col in ['open','high','low','close']:
    df[col] = df[col] * df['adj_factor']
df = df.reset_index(drop=True)

GRID=0.04; LOT=100; INIT_CAP=100000; INIT_POS=0.5
PER_GRID=5000
POS_MIN_FRAC=0.3; POS_MAX_FRAC=2.0
MAW=250; TREND=True

base=df.iloc[0]['close']
units=int((INIT_CAP*INIT_POS)/base/ LOT)*LOT
cash=INIT_CAP-units*base
pos_min=units*POS_MIN_FRAC; pos_max=units*POS_MAX_FRAC
ma=df['close'].rolling(MAW).mean()
trades=[]
def fee(side,p,u):
    comm=max(p*u*0.00025,5)
    return comm + (p*u*0.001 if side=='sell' else 0) + p*u*0.001
for i in range(len(df)):
    row=df.iloc[i]
    td=row['trade_date']; o,h,l,cl=row['open'],row['high'],row['low'],row['close']
    pc=df.iloc[i-1]['close'] if i>0 else cl
    allow=True
    if TREND and pd.notna(ma.iloc[i]) and cl>ma.iloc[i]:
        allow=False
    sell_t=pc*(1+GRID); buy_t=pc*(1-GRID)
    acted=False
    sig_sell_raw = (h>=sell_t)
    sig_buy_raw  = (l<=buy_t)
    if (units>pos_min) and allow and sig_sell_raw and not acted:
        su=max(int((PER_GRID/sell_t)/LOT)*LOT,0)
        if su>units-pos_min: su=int((units-pos_min)/LOT)*LOT
        if su>0:
            cash+=su*sell_t-fee('sell',sell_t,su); units-=su; trades.append((td,'SELL',sell_t,su)); acted=True
    if (units<pos_max) and sig_buy_raw and not acted:
        bu=max(int((PER_GRID/buy_t)/LOT)*LOT,0)
        if bu>pos_max-units: bu=int((pos_max-units)/LOT)*LOT
        cost=bu*buy_t+fee('buy',buy_t,bu)
        if bu>0 and cost<=cash:
            cash-=cost; units+=bu; trades.append((td,'BUY',buy_t,bu)); acted=True
    df.at[i,'sig_sell_raw']=sig_sell_raw; df.at[i,'sig_buy_raw']=sig_buy_raw; df.at[i,'allow']=allow

print("=== 复现交易（应与你贴的一致）===")
for td,a,p,u in trades:
    print(f"  {a} {td} @ {p:.2f} {u}")

idx=[df.index[df['trade_date']==t[0]].tolist()[0] for t in trades]
print("\n=== 长空窗分析（>150 交易日无成交）===")
for k in range(1,len(trades)):
    gap=idx[k]-idx[k-1]
    if gap>150:
        seg=df.iloc[idx[k-1]+1:idx[k]+1]
        nb=int(seg['sig_buy_raw'].sum()); ns=int(seg['sig_sell_raw'].sum())
        blocked=int(((seg['sig_sell_raw']) & (~seg['allow'])).sum())
        allow_pct=seg['allow'].mean()*100
        print(f"\n空窗 {trades[k-1][0]} -> {trades[k][0]}  ({gap} 交易日, ~{gap/242:.1f}年)")
        print(f"  期间出现『买信号(单日跌≥4%)』天数: {nb}")
        print(f"  期间出现『卖信号(单日涨≥4%)』天数: {ns}  (其中被趋势过滤屏蔽 {blocked} 天)")
        print(f"  期间处于『站上MA250·卖出被屏蔽』的交易日占比: {(100-allow_pct):.0f}%")
