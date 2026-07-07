"""
第三方独立重跑（forensic replay）:
不从 run_grid_backtest 导入任何逻辑，独立按相同规则重放一遍，
交叉验证最终资产是否≈170,144。若一致 → 数字真实、非代码bug。
"""
import sqlite3, os
import pandas as pd, numpy as np

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                  "data", "tu-sharedata", "astock_daily.db")
DB = os.path.abspath(DB)

def load_adj(ts_code, start, end):
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query(
        "SELECT trade_date,open,high,low,close FROM etf_daily "
        "WHERE ts_code=? AND trade_date>=? AND trade_date<=? ORDER BY trade_date",
        conn, params=(ts_code, start, end))
    adj = pd.read_sql_query(
        "SELECT trade_date,adj_factor FROM etf_adj_factor "
        "WHERE ts_code=? AND trade_date>=? AND trade_date<=? ORDER BY trade_date",
        conn, params=(ts_code, start, end))
    conn.close()
    df["trade_date"] = df["trade_date"].astype("int64")
    adj["trade_date"] = adj["trade_date"].astype("int64")
    df = df.merge(adj, on="trade_date", how="left")
    df["adj_factor"] = df["adj_factor"].bfill().ffill()
    base = float(df["adj_factor"].iloc[-1])
    for c in ("open", "high", "low", "close"):
        df[c] = df[c] * df["adj_factor"] / base
    return df

# COMMISSION/SLIPPAGE/STAMP 与 run_monthly_rebalance.calc_fee 一致
COMMISSION_RATE = 2.5e-4
COMMISSION_MIN = 5.0
SLIPPAGE_RATE = 1e-3
STAMP_RATE = 1e-3

def fee(side, price, shares):
    amt = price * shares
    comm = max(amt * COMMISSION_RATE, COMMISSION_MIN)
    slip = amt * SLIPPAGE_RATE
    if side == "buy":
        return comm + slip
    return comm + amt * STAMP_RATE + slip

def replay(ts_code, start, end, buy_gap, sell_gap, init_pct=0.5, cap=100000.0,
           PER=5000.0, POS_MIN_FRAC=0.0, POS_MAX_FRAC=4.0):
    df = load_adj(ts_code, start, end)
    lot = 100
    first_open = float(df.iloc[0]["open"])
    units = int((cap * init_pct / first_open) / lot) * lot
    f0 = fee("buy", first_open, units)
    cash = cap - units * first_open - f0
    base = float(df.iloc[0]["close"])
    pos_min = units * POS_MIN_FRAC
    pos_max = units * POS_MAX_FRAC
    N = 400
    buy_lines = sorted([base * (1 - buy_gap) ** k for k in range(1, N + 1)], reverse=True)
    sell_lines = sorted([base * (1 + sell_gap) ** k for k in range(1, N + 1)])
    prev_close = base
    min_cash = cash
    total_fees = f0
    bc = sc = 0
    for _, row in df.iterrows():
        lo, hi, cl = float(row["low"]), float(row["high"]), float(row["close"])
        allow_sell = prev_close > base  # 非对称: 浮盈区才卖
        if cl <= prev_close:
            for line in buy_lines:
                if lo <= line < prev_close and units < pos_max:
                    bu = PER / line
                    bu = int(bu / lot) * lot
                    if bu > pos_max - units:
                        bu = int((pos_max - units) / lot) * lot
                    if bu > 0:
                        cost = bu * line
                        fe = fee("buy", line, bu)
                        if cost + fe > cash:
                            mu = int(cash / (line * lot + fee("buy", line, lot))) * lot
                            if mu >= lot:
                                bu = mu; cost = bu * line; fe = fee("buy", line, bu)
                            else:
                                bu = 0
                        if bu > 0 and cost + fe <= cash:
                            cash -= cost + fe
                            units += bu
                            total_fees += fe
                            bc += 1
                            min_cash = min(min_cash, cash)
        else:
            if allow_sell and units > pos_min:
                for line in sell_lines:
                    if prev_close < line <= hi and units > pos_min:
                        su = PER / line
                        su = int(su / lot) * lot
                        if su > units - pos_min:
                            su = int((units - pos_min) / lot) * lot
                        if su > 0:
                            proc = su * line
                            fe = fee("sell", line, su)
                            cash += proc - fe
                            units -= su
                            total_fees += fe
                            sc += 1
                            min_cash = min(min_cash, cash)
        prev_close = cl
    last = float(df.iloc[-1]["close"])
    proc = units * last
    fe = fee("sell", last, units)
    cash += proc - fe
    total_fees += fe
    return dict(final=cash, min_cash=min_cash, total_fees=total_fees,
                buys=bc, sells=sc, final_units=units, last_close=last,
                init_units=units)

if __name__ == "__main__":
    r = replay("510300.SH", "20180102", "20260703", buy_gap=0.02, sell_gap=0.08)
    print("=== 独立重跑结果（非对称 买2%/卖8%，无趋势过滤）===")
    for k, v in r.items():
        print(f"  {k}: {v:,.2f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"\n  报告值 final=170,144  | 独立重跑 final={r['final']:,.2f}")
    print(f"  偏差: {r['final']-170144:,.2f} 元 ({ (r['final']/170144-1)*100:+.3f}%)")

    print("\n=== 同一策略(非对称买2%/卖8%) 跑在不同ETF上 —— 检验是否'真本事' ===")
    import sqlite3 as _sq
    # 需要验证各ETF数据是否存在
    conn = _sq.connect(DB)
    have = set(x[0] for x in conn.execute("SELECT DISTINCT ts_code FROM etf_daily"))
    conn.close()
    for code in ["510300.SH", "510500.SH", "515800.SH", "512100.SH"]:
        if code not in have:
            print(f"  {code}: 无数据，跳过")
            continue
        rr = replay(code, "20180102", "20260703", buy_gap=0.02, sell_gap=0.08)
        # 基准: 100%买入持有(同复权序列)
        df = load_adj(code, "20180102", "20260703")
        bh = (float(df.iloc[-1]["close"]) / float(df.iloc[0]["close"]) - 1) * 100
        print(f"  {code}: 网格={rr['final']/100000*100-100:+.1f}%  持有={bh:+.1f}%  "
              f"买{rr['buys']}/卖{rr['sells']} 末仓{rr['final_units']:.0f} 现金最低{rr['min_cash']:.0f}")

