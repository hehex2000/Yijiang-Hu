import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_darvas as d

pool="000906.SH"; start="20200101"; end="20211231"
codes, st_set, list_date, price, fund, grow = d.load_universe_data(pool, start, end)
tdates = d.get_trade_dates(start, end)
rset = list(d.get_monthly_5th_trading_days(tdates))
didx = {x:i for i,x in enumerate(tdates)}
print("index dtype sample:", type(next(iter(price.values())).index[0]), next(iter(price.values())).index[0])
for td in rset[:8]:
    prev_td = tdates[didx[td]-1] if didx[td]>0 else td
    prev_int = int(prev_td)
    n_val=0; n_gr=0; n_mom=0; n_box=0; n_all=0
    for c in codes:
        if c in st_set: continue
        ld=list_date.get(c)
        if ld is None or int(str(ld)) > prev_int-10000: continue
        pf=price.get(c)
        if pf is None or len(pf) < d.BOX_WIN+d.MOM_LOOKBACK: continue
        tail=pf.loc[:prev_int]
        if len(tail) < d.BOX_WIN+d.MOM_LOOKBACK: continue
        pe=d._asof(fund.get(c),prev_int,"pe_ttm"); pb=d._asof(fund.get(c),prev_int,"pb"); mv=d._asof(fund.get(c),prev_int,"total_mv")
        if pe is None or pb is None or mv is None:
            continue
        if not(0<pe<=d.PE_MAX) or not(0<pb<=d.PB_MAX) or mv<d.MV_MIN_WAN:
            continue
        n_val+=1
        op=d._asof(grow.get(c),prev_int,"op_yoy"); eps=d._asof(grow.get(c),prev_int,"basic_eps_yoy")
        if (op is None or op<=0) and (eps is None or eps<=0): pass
        else: n_gr+=1
        mom=d.momentum_return(pf["close"], prev_int, d.MOM_SKIP, d.MOM_LOOKBACK)
        if mom is not None and mom>0: n_mom+=1
        sig=d.darvas_box(tail, d.BOX_PCT, d.BOX_WIN, d.MIN_BOX_AGE)
        if sig is not None: n_box+=1
        if (not((op is None or op<=0) and (eps is None or eps<=0))) and mom is not None and mom>0 and sig is not None:
            n_all+=1
    print(f"{td}: val={n_val} growth={n_gr} mom>0={n_mom} box={n_box} ALL={n_all}")
