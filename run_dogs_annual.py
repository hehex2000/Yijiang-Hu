# -*- coding: utf-8 -*-
"""
狗股策略（Dogs of the Market）年度调仓回测
==========================================
参考：《凯利公式——只看一个指标，每年操作一次》视频

策略逻辑：
1. 每年第一个交易日调仓
2. 用 DogsOfMarketSelector 选股（高股息+低PB+连续分红）
3. 卖出旧持仓，等权重买入新股票
4. 当年不再调仓（纯持有）
5. 年终对比基准指数，输出年度收益对比表
6. 最后输出总收益

运行方式：
    python run_dogs_annual.py                    # 默认参数
    python run_dogs_annual.py 20200102 20261231   # 指定时间范围
    python run_dogs_annual.py --top-n 10          # 选10只
"""
import sys, os, sqlite3
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime

from config import DATA, SELECTION, GLOBAL, DOGS_OF_MARKET, BACKTEST

DB_PATH = DATA.get("local_db_path", "D:/tu-shareData/astock_daily.db")
# 每只股票初始资金：默认跟随 config 的 per_stock_capital（与「选股+回测」一致），
# 也可通过 run_backtest(capital=...) 或 CLI --capital 覆盖
INIT_CAPITAL = BACKTEST.get("per_stock_capital", 100000)


def ts_code(code):
    """补全股票代码为 ts_code 格式"""
    c = str(code).strip()
    if len(c) == 6:
        if c.startswith(("6", "9")):
            c += ".SH"
        else:
            c += ".SZ"
    return c.split(".")[0] + "." + c.split(".")[1] if "." in c else c


def get_trade_dates(start_date, end_date):
    """获取交易日列表"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT DISTINCT trade_date FROM daily WHERE trade_date BETWEEN ? AND ? ORDER BY trade_date",
        conn, params=(start_date, end_date),
    )
    conn.close()
    return df["trade_date"].tolist()


def get_first_trading_days(trade_dates):
    """获取每年的第一个交易日"""
    yearly = {}
    for td in trade_dates:
        year = td[:4]
        if year not in yearly:
            yearly[year] = td
    return yearly  # {year: first_trading_day}


def get_stock_price(ts_code, trade_date, price_type="close"):
    """获取单只股票在某日的「原始(未复权)」价格"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        f"SELECT {price_type} FROM daily WHERE ts_code=? AND trade_date=?",
        conn, params=(ts_code, trade_date),
    )
    conn.close()
    return float(df.iloc[0, 0]) if len(df) > 0 else None


def get_hfq_price(ts_code, trade_date, price_type="close"):
    """获取单只股票在某日的「后复权」价格 = 当日原始价 × 复权因子。

    后复权价已内含现金分红与拆/送股，即用「买入并持有、分红再投入」的
    投资者真实总回报口径计价。对狗股这类高股息策略，必须用后复权，
    否则除息日的股价跳降会被记成亏损、而分红却没入账 → 长期收益被低估。

    关键防坑: adj_factor 表相对 daily 存在大量缺口(每只股缺数百~上千天)。
    若用 `... JOIN adj_factor ON trade_date = a.trade_date` 直接关联，缺因子的
    那天会回退成原始价(因子=1)，而相邻有因子的天是后复权价(因子可达 50+)，
    造成同一只股票价格在 4 元↔200 元间反复跳变 → 日净值出现 50 倍假尖峰 →
    收益被算成天文数字。因此这里对因子做「前向填充」: 取 trade_date <= 当日
    最近一条有效 adj_factor。复权因子仅在除权除息日跳变、其余交易日恒定，
    前向填充完全正确，可彻底消除价格断层。
    """
    conn = sqlite3.connect(DB_PATH)
    # 1) 当日(或之前最近)的原始价
    px = pd.read_sql_query(
        f"SELECT {price_type} AS p FROM daily WHERE ts_code = ? AND trade_date <= ? "
        f"ORDER BY trade_date DESC LIMIT 1",
        conn, params=(ts_code, trade_date),
    )
    # 2) 前向填充的复权因子: trade_date <= 当日 最近一条
    fac = pd.read_sql_query(
        "SELECT adj_factor FROM adj_factor WHERE ts_code = ? AND trade_date <= ? "
        "ORDER BY trade_date DESC LIMIT 1",
        conn, params=(ts_code, trade_date),
    )
    conn.close()
    if len(px) == 0 or px.iloc[0]["p"] is None:
        return None
    p = float(px.iloc[0]["p"])
    f = fac.iloc[0]["adj_factor"] if (len(fac) > 0 and fac.iloc[0]["adj_factor"] is not None) else None
    if f is None or f == 0:
        return p  # 无复权因子(如因子数据起始前的日期) → 退化为原始价
    return p * f


def get_index_close(index_code, trade_date):
    """获取指数收盘价（如果当天无数据，自动向前取最近交易日）"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT close FROM index_daily WHERE ts_code=? AND trade_date <= ? ORDER BY trade_date DESC LIMIT 1",
        conn, params=(index_code, trade_date),
    )
    conn.close()
    if len(df) > 0:
        val = float(df.iloc[0, 0])
        # 如果实际取到的日期和请求的日期不同，打印提示
        return val
    return None


def get_stock_pool_index():
    """根据股票池配置返回对应的基准指数代码"""
    pool = SELECTION.get("stock_pool", "zz800")
    pool_map = {
        "hs300": "000300.SH",
        "zz500": "000905.SH",
        "zz800": "000906.SH",
        "zz1000": "000852.SH",
        "zz2000": "932000.SH",  # 中证2000 = 微盘基准
        "all": "000985.SH",   # 中证全指 = 全市场基准（之前误用 000906.SH 中证800）
    }
    return pool_map.get(pool, "000906.SH")


def run_backtest(start_date="20200102", end_date="20261231", top_n=None, select_only=False, capital=None):
    """执行狗股策略年度调仓回测"""
    if top_n is None:
        top_n = SELECTION.get("top_n", 5)

    # 初始资金：优先用传入 capital，否则跟随 config 的 per_stock_capital（与「选股+回测」一致）
    if capital is None:
        capital = BACKTEST.get("per_stock_capital", 100000)
    global INIT_CAPITAL
    INIT_CAPITAL = capital

    DOGS_OF_MARKET["top_n"] = top_n
    DOGS_OF_MARKET["stock_pool"] = SELECTION["stock_pool"]

    # 获取交易日
    trade_dates = get_trade_dates(start_date, end_date)
    if not trade_dates:
        print("[ERROR] 无交易日数据")
        return

    # 每年第一个交易日
    yearly_first = get_first_trading_days(trade_dates)
    years = sorted(yearly_first.keys())
    print(f"\n{'='*70}")
    print(f"  狗股策略年度调仓回测")
    print(f"  区间: {start_date} ~ {end_date}  |  选股: {top_n} 只")
    print(f"  调仓频率: 每年初 (共 {len(years)} 年)")
    print(f"{'='*70}\n")

    benchmark_idx = get_stock_pool_index()
    print(f"  基准指数: {benchmark_idx}")
    print()

    from src.dogs_of_market_selector import DogsOfMarketSelector
    from src.data_fetcher import DataFetcher

    # 初始化选股器
    df_config = {
        "primary_source": DATA.get("primary_source", "local_db"),
        "tushare_token": DATA.get("tushare_token", ""),
        "local_db_path": DB_PATH,
        "use_akshare_backup": False,
        "use_tushare_backup": False,
    }
    fetcher = DataFetcher(**df_config)
    selector = DogsOfMarketSelector(DOGS_OF_MARKET, fetcher)

    # 回测状态
    positions = {}       # {ts_code: {"shares": N, "buy_price": P}}
    cash = INIT_CAPITAL * top_n
    daily_vals = []      # 每日总市值

    # 年度收益记录
    year_results = []    # [{year, strategy_ret, benchmark_ret, stocks}]

    # 获取每个调仓日的前一个交易日（用于选股）
    def get_prev_trading_day(td):
        conn = sqlite3.connect(DB_PATH)
        row = pd.read_sql_query(
            "SELECT MAX(trade_date) FROM daily WHERE trade_date < ?",
            conn, params=(td,),
        )
        conn.close()
        return str(row.iloc[0, 0]) if row.iloc[0, 0] else td

    # 持仓代码列表
    def current_codes():
        return set(positions.keys())

    # 遍历每天
    trading_year = None
    years_done = set()

    for i, td in enumerate(trade_dates):
        year = td[:4]

        # 判断是否是今年的第一个交易日（调仓日）
        is_rebalance_day = (year in yearly_first and td == yearly_first[year])

        # 获取当日所有持仓股票的收盘价
        total_value = cash
        for code, pos in positions.items():
            close = get_hfq_price(code, td, "close")
            if close:
                total_value += pos["shares"] * close
        daily_vals.append({"date": td, "value": total_value})

        # === 调仓日执行 ===
        if is_rebalance_day:
            # 用前一个交易日数据选股
            prev_td = get_prev_trading_day(td)
            print(f"\n── {year}年调仓日: {td} (选股基准: {prev_td}) ──")

            # 如果这是第一年，记录年初的基准指数值
            if year not in years_done:
                idx_start = get_index_close(benchmark_idx, td)
                if idx_start is None:
                    idx_start = get_index_close(benchmark_idx, prev_td)
                years_done.add(year)

            # 选股
            selected = selector.select_stocks(date=prev_td)
            if selected is None or len(selected) == 0:
                print(f"  [WARN] {year}年选股失败，保持现有持仓")
                continue

            new_codes = selected["ts_code"].tolist()
            new_code_set = set(new_codes)
            old_codes = current_codes()

            # 卖出不在新池中的旧股票
            codes_to_sell = old_codes - new_code_set
            if codes_to_sell:
                print(f"  卖出 {len(codes_to_sell)} 只: {', '.join(codes_to_sell)[:60]}...")
            for code in codes_to_sell:
                pos = positions.pop(code, None)
                if pos:
                    open_price = get_hfq_price(code, td, "open")
                    if open_price:
                        revenue = pos["shares"] * open_price * 0.99955  # 扣手续费
                        cash += revenue

            # 等权重买入新股
            old_in_new = old_codes & new_code_set
            new_to_buy = [c for c in new_codes if c not in old_codes]

            # 等分资金（预留手续费空间）
            remaining_slots = top_n - len(old_in_new)
            cash_per_stock = cash * 0.98 / remaining_slots if remaining_slots > 0 else 0

            bought = 0
            skipped = []
            for code in new_to_buy:
                open_price = get_hfq_price(code, td, "open")
                if open_price is None or open_price <= 0:
                    skipped.append(code)
                    continue
                max_shares = int(cash_per_stock / open_price / 100) * 100
                if max_shares <= 0:
                    skipped.append(code)
                    continue
                cost = max_shares * open_price * 1.0002  # +手续费
                if cost > cash:
                    skipped.append(code)
                    continue
                positions[code] = {"shares": max_shares, "buy_price": open_price}
                cash -= cost
                bought += 1

            if skipped:
                print(f"  [跳过] {len(skipped)} 只: {', '.join(skipped)[:60]}...")

            print(f"  买入 {bought} 只, 持有 {len(old_in_new)} 只, 当前持仓 {len(positions)} 只")
            print(f"  现金: {cash:.2f}")

    # ──── 回测结束：强制平仓 ────
    print(f"\n  {'─'*60}")
    if positions:
        total_cashout = 0
        last_holdings = list(positions.keys())  # 平仓前捕获末年持仓，供汇总展示
        for code, pos in list(positions.items()):
            close = get_hfq_price(code, trade_dates[-1], "close")
            if close:
                # 卖出：扣佣金+印花税，同 sell() 逻辑
                revenue = pos["shares"] * close
                fee = max(revenue * 0.0002, 5.0)
                tax = revenue * 0.001
                net_revenue = revenue - fee - tax
                cash += net_revenue
                total_cashout += net_revenue
                print(f"  平仓 {code}({pos['shares']}股) @ {close:.2f}, 净回收 {net_revenue:.2f}")
        positions.clear()
        # 用平仓后的现金更新最后一天的 daily_value
        if daily_vals:
            daily_vals[-1]["value"] = cash
        print(f"  平仓完成，最终现金: {cash:.2f}")

    # ──── 计算年度收益 ────
    print(f"\n\n{'='*70}")
    print(f"  📊 年度收益对比")
    print(f"{'='*70}")
    print(f"  [口径] 策略 = 后复权价(含分红再投, 投资者真实总回报)")
    print(f"  [口径] 基准 = {benchmark_idx} 价格指数(不含分红)")
    print(f"  [提示] 带 * 年度为区间未结束的半年度，收益按实际持有期计")
    print(f"  {'年份':<9} {'策略收益':>10} {'基准收益':>10} {'超额收益':>10} {'持仓股票'}")
    print(f"  {'─'*60}")

    # 按年计算收益
    year_groups = {}
    for d in daily_vals:
        y = d["date"][:4]
        if y not in year_groups:
            year_groups[y] = {"first": d["value"], "first_date": d["date"], "n_days": 0}
        year_groups[y]["last"] = d["value"]
        year_groups[y]["last_date"] = d["date"]
        year_groups[y]["n_days"] += 1

    total_strategy_return = 0
    for idx, year in enumerate(sorted(year_groups.keys())):
        yg = year_groups[year]
        if idx == 0:
            # 第一年：从年初到年底
            year_start = INIT_CAPITAL * top_n
        else:
            year_start = year_groups[year]["first"]

        year_end = yg["last"]
        if year_start > 0:
            strategy_ret = (year_end / year_start - 1) * 100
        else:
            strategy_ret = 0

        # 基准指数年度收益（价格指数，不含分红）
        benchmark_ret = 0
        b_start_idx = get_index_close(benchmark_idx, yg["first_date"])
        b_end_idx = get_index_close(benchmark_idx, yg["last_date"])
        if b_start_idx and b_end_idx and b_start_idx > 0:
            benchmark_ret = (b_end_idx / b_start_idx - 1) * 100

        excess = strategy_ret - benchmark_ret

        # 区间未结束的年度（如回测截止在年中）标注 *
        is_partial = yg.get("n_days", 0) < 200
        year_label = f"{year}*" if is_partial else year

        # 获取该年持仓股票简要信息（末年用平仓前捕获的持仓）
        if idx == len(year_groups) - 1:
            year_stocks = ", ".join(last_holdings[:5])
        else:
            year_stocks = "-"

        print(f"  {year_label:<9} {strategy_ret:>+9.2f}% {benchmark_ret:>+9.2f}% {excess:>+9.2f}%  {year_stocks[:36]}")
        total_strategy_return = strategy_ret

    # 总收益（回测全程）
    final_value = daily_vals[-1]["value"]
    total_return = (final_value / (INIT_CAPITAL * top_n) - 1) * 100

    # 基准总收益
    first_date = daily_vals[0]["date"]
    last_date = daily_vals[-1]["date"]
    b_total_start = get_index_close(benchmark_idx, first_date)
    b_total_end = get_index_close(benchmark_idx, last_date)
    b_total_ret = (b_total_end / b_total_start - 1) * 100 if b_total_start and b_total_end else 0

    total_excess = total_return - b_total_ret

    print(f"  {'─'*60}")
    print(f"  {'全程':<8} {total_return:>+9.2f}% {b_total_ret:>+9.2f}% {total_excess:>+9.2f}%")

    # 年化收益
    days = len(trade_dates)
    years_span = days / 252
    annual_return = ((final_value / (INIT_CAPITAL * top_n)) ** (1 / years_span) - 1) * 100 if years_span > 0 else 0

    # 风控指标
    vals = np.array([d["value"] for d in daily_vals])
    cummax = np.maximum.accumulate(vals)
    drawdowns = (vals - cummax) / cummax
    max_dd = float(np.min(drawdowns)) * 100
    rets = np.diff(vals) / vals[:-1]
    sharpe = (np.mean(rets) * 252 - 0.025) / (np.std(rets) * np.sqrt(252)) if len(rets) > 0 else 0

    print(f"\n{'='*70}")
    print(f"  📈 最终汇总")
    print(f"{'='*70}")
    profit_amount = final_value - INIT_CAPITAL * top_n
    print(f"  初始资金: {INIT_CAPITAL * top_n:>10,.2f}")
    print(f"  最终资产: {final_value:>10,.2f}")
    print(f"  总盈亏: {profit_amount:>+10,.2f} 元")
    print(f"  总收益率: {total_return:>+9.2f}%")
    print(f"  年化收益率: {annual_return:>+9.2f}%")
    print(f"  基准收益: {b_total_ret:>+9.2f}%")
    print(f"  超额收益: {total_excess:>+9.2f}%")
    print(f"  最大回撤: {max_dd:>+9.2f}%")
    print(f"  夏普比率: {sharpe:>9.4f}")
    print(f"  调仓次数: {len(years)} 次")
    print(f"{'='*70}")
    print(f"  [口径说明] 策略=后复权(含分红再投); 基准={benchmark_idx}价格指数(不含分红)。")
    print(f"  [口径说明] 二者口径不同，超额收益含策略分红优势，非纯粹选股超额。")
    print(f"{'='*70}\n")

    # 保存CSV
    csv_dir = "data/results/dogs_annual"
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = f"{csv_dir}/backtest_{start_date}_{end_date}.csv"
    pd.DataFrame(daily_vals).to_csv(csv_path, index=False)
    print(f"  日净值已保存 → {csv_path}\n")

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "trades": len(years),
        "daily_values": daily_vals,
        "year_results": year_results,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="狗股策略年度调仓回测")
    parser.add_argument("start_date", nargs="?", default="20200102")
    parser.add_argument("end_date", nargs="?", default="20261231")
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--capital", type=int, default=None,
                        help="每只股票初始资金（默认跟随 config 的 per_stock_capital）")
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()

    run_backtest(args.start_date, args.end_date, top_n=args.top_n,
                 select_only=args.select_only, capital=args.capital)
