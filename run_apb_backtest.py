# -*- coding: utf-8 -*-
"""
APB5D 买卖压力因子回测（修正版）
===================================
核心公式（跨日版）：
  5日VWAP = 5日sum(amount) / 5日sum(vol) × 单位换算
  5日TWAP = 5日复权收盘均价
  APB5D = (VWAP5D - TWAP5D) / TWAP5D
  买压(APB<0) → 买入最低APB的100只

改进说明：
  - 使用 adj_factor 做前复权
  - 涨跌停过滤 ±9.8%
  - 佣金万2.5(最低5元) + 印花税千1
"""

import sys, os, sqlite3, time, argparse, json
from collections import defaultdict
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DB_PATH = r"D:\tu-shareData\astock_daily.db"
INIT_CAPITAL = 500000
DEFAULT_TOP_N = 100

# ── 工具函数 ─────────────────────────────────────────────

def get_conn():
    return sqlite3.connect(DB_PATH)

def get_trade_dates(start_date, end_date):
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT DISTINCT trade_date FROM daily WHERE trade_date BETWEEN ? AND ? ORDER BY trade_date",
        conn, params=(start_date, end_date))
    conn.close()
    return df["trade_date"].tolist()

def get_index_close(index_code, trade_date):
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT close FROM index_daily WHERE ts_code=? AND trade_date <= ? ORDER BY trade_date DESC LIMIT 1",
        conn, params=(index_code, trade_date))
    conn.close()
    return float(df.iloc[0, 0]) if len(df) > 0 else None

def calc_fee_buy(price, shares):
    amount = price * shares
    return max(amount * 0.00025, 5.0)

def calc_fee_sell(price, shares):
    amount = price * shares
    fee = max(amount * 0.00025, 5.0)
    return fee + amount * 0.001

def get_stock_basic():
    conn = get_conn()
    df = pd.read_sql_query("SELECT ts_code, name, list_date FROM stock_basic", conn)
    conn.close()
    return df

# ── 数据加载（含前复权） ────────────────────────────────

def load_data_adj(start_date, end_date):
    """加载日线 + 前复权，返回 [ts_code, trade_date, close_adj, open, vol, amount, pct_chg, vwap_adj]"""
    print(f"📡 加载数据 {start_date}~{end_date}...")
    conn = get_conn()
    df = pd.read_sql_query("""
        SELECT d.ts_code, d.trade_date, d.open, d.close, d.vol, d.amount, d.pct_chg,
               a.adj_factor
        FROM daily d
        LEFT JOIN adj_factor a ON d.ts_code = a.ts_code AND d.trade_date = a.trade_date
        WHERE d.trade_date BETWEEN ? AND ?
          AND d.vol > 0 AND d.amount > 0 AND d.close IS NOT NULL
        ORDER BY d.ts_code, d.trade_date
    """, conn, params=(start_date, end_date))
    conn.close()

    # 前复权：每只股票按最新adj_factor归一化
    print("   前复权处理...")
    df['adj_factor'] = df['adj_factor'].fillna(1.0)
    latest_adj = df.groupby('ts_code')['adj_factor'].transform('last')
    ratio = df['adj_factor'] / latest_adj
    df['close_adj'] = df['close'] * ratio

    # 复权VWAP（用当日adj_factor复权）
    # VWAP_actual = amount(千元)*1000 / (vol(手)*100) = amount*10/vol
    # VWAP_adj = VWAP_actual * ratio
    df['vwap_adj'] = (df['amount'] * 10.0 / df['vol']) * ratio

    print(f"   {len(df):,} 条, {df['ts_code'].nunique()} 只股票")
    return df

def calc_apb5d_fast(df, window=5):
    """向量化计算跨日APB5D"""
    print("📊 计算跨日APB5D因子...")

    # 按股票分组，每组计算滚动聚合
    df = df.sort_values(['ts_code', 'trade_date'])

    # 5日滚动成交额和成交量（用于VWAP5D）
    df['amount_sum5'] = df.groupby('ts_code')['amount'].rolling(window, min_periods=3).sum().values
    df['vol_sum5'] = df.groupby('ts_code')['vol'].rolling(window, min_periods=3).sum().values

    # 5日VWAP（已复权，用当天ratio）
    latest_adj = df.groupby('ts_code')['adj_factor'].transform('last')
    ratio = df['adj_factor'] / latest_adj
    df['vwap_5d'] = (df['amount_sum5'] * 10.0 / df['vol_sum5']) * ratio

    # 5日TWAP = 5日复权收盘均价
    df['twap_5d'] = df.groupby('ts_code')['close_adj'].rolling(window, min_periods=3).mean().values

    # APB5D = (VWAP5D - TWAP5D) / TWAP5D
    # 买压大 → VWAP<TWAP → APB<0 → 买入最小APB
    df['apb_5d'] = (df['vwap_5d'] - df['twap_5d']) / df['twap_5d']
    df['apb_5d'] = df['apb_5d'].replace([np.inf, -np.inf], np.nan)

    has_val = df['apb_5d'].notna().sum()
    print(f"  完成: {has_val}/{len(df)} 条有值")
    return df

# ── 回测主函数 ──────────────────────────────────────────

def run_apb_backtest(start_date="20180101", end_date="20220107", top_n=DEFAULT_TOP_N):
    print(f"\n{'='*70}")
    print(f"  APB5D × 跨日VWAPvsTWAP × 月度调仓")
    print(f"{'='*70}")
    print(f"  区间: {start_date} ~ {end_date}")
    print(f"  选股: {top_n} 只（买最小APB5D = 买压最大）")
    print(f"  资金: {INIT_CAPITAL:,.0f} | 佣金万2.5+印花税千1")
    print(f"  基准: 沪深300\n")

    trade_dates = get_trade_dates(start_date, end_date)
    print(f"  交易日: {len(trade_dates)}")

    # ── 月度调仓计划 ──
    yearly_monthly = defaultdict(list)
    for td in trade_dates:
        ym = td[:6]
        yearly_monthly[ym].append(td)

    rebalance_schedule = {}
    sorted_months = sorted(yearly_monthly.keys())
    for i, ym in enumerate(sorted_months):
        month_dates = yearly_monthly[ym]
        last_date = month_dates[-1]
        if i + 1 < len(sorted_months):
            next_month = sorted_months[i + 1]
            rebalance_date = yearly_monthly[next_month][0]
        else:
            continue
        rebalance_schedule[rebalance_date] = last_date

    print(f"  调仓次数: {len(rebalance_schedule)}\n")

    # ── 数据加载 → 因子计算 ──
    raw = load_data_adj(start_date, end_date)
    data = calc_apb5d_fast(raw)

    # 构建因子索引 {trade_date: {ts_code: apb_5d}}
    print("   构建因子日期索引...")
    idx_df = data[['ts_code', 'trade_date', 'apb_5d']].dropna().copy()
    idx_df['trade_date'] = idx_df['trade_date'].astype(str)
    apb_by_date = defaultdict(dict)
    for _, row in idx_df.iterrows():
        apb_by_date[row['trade_date']][row['ts_code']] = row['apb_5d']

    # 构建价格查找表
    print("   构建价格查找表...")
    pl = {}
    for _, row in raw.iterrows():
        code, td = row['ts_code'], str(row['trade_date'])
        pl[(code, td)] = {'close': row['close_adj'], 'open': row['open'] * (row['adj_factor'] / row['adj_factor'])}
        # Use adj_factor to get forward-adjusted open price
    latest_adj = raw.groupby('ts_code')['adj_factor'].transform('last')
    raw['ratio'] = raw['adj_factor'] / latest_adj
    price_lookup = {}
    for _, row in raw.iterrows():
        code, td = row['ts_code'], row['trade_date']
        ratio = row['ratio']
        price_lookup[(code, td)] = row['close_adj']
        price_lookup[('open', code, td)] = row['open'] * ratio

    def get_price(code, td):
        return price_lookup.get((code, td))

    def get_open_price(code, td):
        return price_lookup.get(('open', code, td))

    # 涨跌停过滤：获取每月最后一天的pct_chg
    last_day_pct = {}
    last_day_mask = raw.groupby('ts_code')['trade_date'].transform('max') == raw['trade_date']
    for _, row in raw[last_day_mask].iterrows():
        last_day_pct[(row['ts_code'], str(row['trade_date'][:6]))] = row['pct_chg']

    stock_basic = get_stock_basic()
    name_map = dict(zip(stock_basic['ts_code'], stock_basic['name']))
    list_date_map = dict(zip(stock_basic['ts_code'], stock_basic['list_date']))

    # ── 回测主循环 ──
    positions = {}
    cash = INIT_CAPITAL
    daily_vals = []
    trade_log = []

    last_print = 0
    for i, td in enumerate(trade_dates):
        td_str = str(td) if not isinstance(td, str) else td

        # 每日估值
        total_val = cash
        for code, pos in list(positions.items()):
            p = get_price(code, td_str)
            if p:
                total_val += pos['shares'] * p
        daily_vals.append({"date": td_str, "value": total_val})

        # 调仓日
        if td_str in rebalance_schedule:
            calc_date = rebalance_schedule[td_str]
            calc_date_str = str(calc_date) if not isinstance(calc_date, str) else calc_date
            ym = calc_date_str[:6]
            print(f"\n📅 调仓日 {td_str} (因子: {calc_date_str})")

            # 获取因子值
            candidates = apb_by_date.get(calc_date_str, {})
            if not candidates:
                # 向前查找最近的因子日
                all_dates = sorted(apb_by_date.keys())
                for d in reversed(all_dates):
                    if d < td_str:
                        candidates = apb_by_date.get(d, {})
                        if candidates:
                            calc_date_str = d
                            ym = d[:6]
                            break
            if not candidates:
                print("  无因子数据，跳过")
                continue

            # 排序：买最小APB（最大买压）
            sorted_codes = sorted(candidates.items(), key=lambda x: x[1])
            selected = [c for c, v in sorted_codes[:top_n]]

            # 过滤 ST/北交所/次新
            filtered = []
            for code in selected:
                name = name_map.get(code, "")
                if "ST" in name or "退" in name:
                    continue
                if code.endswith(".BJ"):
                    continue
                list_date = list_date_map.get(code, "99999999")
                if list_date and list_date[:8] > str(int(td_str) - 60):
                    continue
                # 涨跌停过滤
                pct_key = (code, ym)
                if pct_key in last_day_pct and abs(last_day_pct[pct_key]) >= 9.8:
                    continue
                filtered.append(code)
            new_codes = filtered[:top_n]

            if not new_codes:
                print("  选股结果为空，跳过")
                continue

            new_set = set(new_codes)
            old_codes = set(positions.keys())

            # 卖出
            to_sell = old_codes - new_set
            if to_sell:
                print(f"  卖出 {len(to_sell)} 只...")
            for code in to_sell:
                pos = positions.pop(code, None)
                if pos:
                    op = get_open_price(code, td_str)
                    if op:
                        rev = pos['shares'] * op
                        fee = calc_fee_sell(op, pos['shares'])
                        cash += rev - fee
                        trade_log.append({"date": td_str, "action": "SELL", "code": code,
                                          "shares": pos['shares'], "price": op})

            # 买入
            kept = old_codes & new_set
            to_buy = [c for c in new_codes if c not in positions]
            slots = top_n - len(kept)

            if slots > 0 and cash > 0:
                per_stock = cash * 0.98 / slots
                bought = 0
                for code in to_buy:
                    op = get_open_price(code, td_str)
                    if not op or op <= 0:
                        continue
                    max_shares = int(per_stock / op / 100) * 100
                    if max_shares <= 0:
                        continue
                    cost = max_shares * op
                    fee = calc_fee_buy(op, max_shares)
                    if cost + fee > cash:
                        continue
                    cash -= cost + fee
                    positions[code] = {"shares": max_shares, "buy_price": op}
                    bought += 1
                    trade_log.append({"date": td_str, "action": "BUY", "code": code,
                                      "shares": max_shares, "price": op})
                print(f"  买入 {bought} 只, 保持 {len(kept)} 只, 共 {len(positions)} 只, 现金 {cash:.0f}")

        # 进度
        pct = (i + 1) / len(trade_dates) * 100
        if int(pct / 10) > int(last_print / 10):
            print(f"\r  进度: {pct:.0f}%", end="")
            sys.stdout.flush()
            last_print = pct

    # ── 平仓结算 ──
    print(f"\n\n{'─'*60}")
    last_date = trade_dates[-1]
    last_date_str = str(last_date) if not isinstance(last_date, str) else last_date
    if positions:
        for code, pos in list(positions.items()):
            cp = get_price(code, last_date_str)
            if cp:
                rev = pos['shares'] * cp
                fee = calc_fee_sell(cp, pos['shares'])
                net = rev - fee
                cash += net
                print(f"  平仓 {code}: {pos['shares']}股 @ {cp:.2f}, 净回收 {net:.0f}")
        positions.clear()
        if daily_vals:
            daily_vals[-1]["value"] = cash

    # ── 计算结果 ──
    final_val = daily_vals[-1]["value"]
    total_ret = (final_val / INIT_CAPITAL - 1) * 100
    days = len(trade_dates)
    years = days / 252
    ann_ret = ((final_val / INIT_CAPITAL) ** (1 / years) - 1) * 100 if years > 0 else 0

    vals = np.array([d["value"] for d in daily_vals])
    cummax = np.maximum.accumulate(vals)
    dd = (vals - cummax) / cummax
    max_dd = float(np.min(dd)) * 100

    rets = np.diff(vals) / vals[:-1]
    sharpe = (np.mean(rets) * 252 - 0.025) / (np.std(rets) * np.sqrt(252)) if len(rets) > 0 else 0

    b_start = get_index_close("000300.SH", daily_vals[0]["date"])
    b_end = get_index_close("000300.SH", last_date_str)
    b_ret = (b_end / b_start - 1) * 100 if b_start and b_end else 0

    # 胜率
    pending = defaultdict(list)
    win = 0
    n_trades = 0
    for t in trade_log:
        code, act, shares, price = t["code"], t["action"], t["shares"], t["price"]
        if act == "BUY":
            pending[code].append({"price": price, "shares": shares})
        else:
            rem = shares
            while rem > 0 and pending.get(code):
                first = pending[code][0]
                match = min(first["shares"], rem)
                pnl = (price - first["price"]) * match
                n_trades += 1
                if pnl > 0:
                    win += 1
                first["shares"] -= match
                rem -= match
                if first["shares"] <= 0:
                    pending[code].pop(0)
    win_rate = (win / n_trades * 100) if n_trades > 0 else 0

    buy_cnt = sum(1 for t in trade_log if t["action"] == "BUY")
    sell_cnt = sum(1 for t in trade_log if t["action"] == "SELL")

    # ── 输出 ──
    print(f"\n{'='*70}")
    print(f"  📊 回测结果")
    print(f"{'='*70}")
    print(f"  初始资金:  {INIT_CAPITAL:>10,.0f}")
    print(f"  最终资产:  {final_val:>10,.0f}")
    print(f"  总盈亏:    {final_val-INIT_CAPITAL:>+10,.0f}")
    print(f"  总收益率:  {total_ret:>+9.2f}%")
    print(f"  年化收益率: {ann_ret:>+9.2f}%")
    print(f"  基准收益:  {b_ret:>+9.2f}%")
    print(f"  超额收益:  {total_ret-b_ret:>+9.2f}%")
    print(f"  最大回撤:  {max_dd:>+9.2f}%")
    print(f"  夏普比率:  {sharpe:>9.4f}")
    print(f"  交易次数:  买入{buy_cnt} / 卖出{sell_cnt}")
    print(f"  胜率:      {win_rate:.1f}% ({win}/{n_trades})")
    print(f"  调仓次数:  {len(rebalance_schedule)}")
    print(f"{'='*70}\n")

    # 保存
    csv_dir = "data/results/apb_backtest"
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = f"{csv_dir}/apb5d_top{top_n}_{start_date}_{end_date}.csv"
    pd.DataFrame(daily_vals).to_csv(csv_path, index=False)
    print(f"  CSV: {csv_path}\n")

    return {"total_return": total_ret, "annual_return": ann_ret, "max_drawdown": max_dd,
            "sharpe": sharpe, "win_rate": win_rate, "benchmark_return": b_ret, "excess_return": total_ret - b_ret}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="20180101")
    parser.add_argument("--end", default="20220107")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    args = parser.parse_args()
    run_apb_backtest(args.start, args.end, args.top_n)
