"""
价值投资策略回测脚本 - 按月调仓版本
策略规则：
1. 每月第五个交易日调仓
2. 每次选5只股票，每只投入1万元
3. 等权重配置，始终保持满仓
4. 如果某只股票不够买1手（100股），则跳过选下一只
5. 买入后持有到下次调仓日
"""

import pandas as pd
import sqlite3
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_fetcher import DataFetcher
from value_stock_selector import ValueStockSelector

class ValueStrategyBacktest:
    """价值投资策略回测"""
    
    def __init__(self, initial_cash=50000, db_path="D:/tu-shareData/astock_daily.db",
                 freq="monthly", size_neutral=False, value_pct=None, top_n=5,
                 stock_pool="zz800", value_mode="pobreak", downside_filter=False):
        """
        初始化回测

        Args:
            initial_cash: 初始资金（默认5万，每只股票1万）
            db_path: 数据库路径
            freq: 调仓频率 monthly=每月第五个交易日 / yearly=每年首个交易日(持有>=1年，验证Fama-French长期有效)
            size_neutral: 市值中性化（控制市值因子，纯价值得分）
            value_pct: BM 分位筛选（如0.3=全市场BM前30%）
            top_n: 每期选股数量
            stock_pool: 股票池 hs300/zz500/zz800/zz1000
            value_mode: 选股模式 pobreak(破净价值) / pure_bm(放宽破净·BM分位门槛)
        """
        self.initial_cash = initial_cash
        self.db_path = db_path
        self.freq = freq
        self.size_neutral = size_neutral
        self.value_pct = value_pct
        self.top_n = top_n
        self.stock_pool = stock_pool
        self.value_mode = value_mode
        self.downside_filter = downside_filter
        self.data_fetcher = DataFetcher()

        # 读取配置
        import config
        self.config = config.VALUE_STRATEGY
        self.config["top_n"] = top_n
        self.config["stock_pool"] = stock_pool
        self.config["value_mode"] = value_mode
        self.selector = ValueStockSelector(self.config, self.data_fetcher)
        # 覆盖中性化/分位/模式开关（命令行优先）
        self.selector.value_size_neutral = size_neutral
        self.selector.value_pct = value_pct
        self.selector.value_mode = value_mode
        
    def get_trading_days(self, start_date, end_date):
        """
        获取指定日期范围内的所有交易日
        
        Args:
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            
        Returns:
            list: 交易日列表 (YYYYMMDD格式)
        """
        conn = sqlite3.connect(self.db_path)
        query = f"""
            SELECT DISTINCT trade_date 
            FROM daily 
            WHERE trade_date >= '{start_date}' 
              AND trade_date <= '{end_date}'
            ORDER BY trade_date
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df['trade_date'].tolist()
    
    def get_rebalance_days(self, trading_days, freq=None):
        """
        获取调仓日列表

        Args:
            trading_days: 所有交易日列表
            freq: monthly=每月第五个交易日 / yearly=每年首个交易日(持有>=1年)

        Returns:
            list: 调仓日列表
        """
        freq = freq or self.freq
        df = pd.DataFrame({'trade_date': trading_days})
        df['year_month'] = df['trade_date'].str[:6]  # YYYYMM
        df['year'] = df['trade_date'].str[:4]
        df['day'] = df['trade_date'].str[6:8]

        rebalance_days = []
        if freq == "yearly":
            # 每年首个交易日调仓，单只持仓跨越>=1年，验证 Fama-French "长期有效"
            for y in sorted(df['year'].unique()):
                y_days = df[df['year'] == y]['trade_date'].tolist()
                if len(y_days) >= 1:
                    rebalance_days.append(y_days[0])
        else:
            # 每月第五个交易日
            for ym in df['year_month'].unique():
                month_days = df[df['year_month'] == ym]['trade_date'].tolist()
                if len(month_days) >= 5:
                    rebalance_days.append(month_days[4])  # 索引4 = 第5个交易日

        return rebalance_days

    # 兼容旧名
    def get_monthly_rebalance_days(self, trading_days):
        return self.get_rebalance_days(trading_days, "monthly")
    
    def get_stock_price(self, ts_code, trade_date):
        """
        获取某股票在某交易日的收盘价
        
        Args:
            ts_code: 股票代码 (000001.SZ)
            trade_date: 交易日期 (YYYYMMDD)
            
        Returns:
            float: 收盘价，如果不存在返回None
        """
        conn = sqlite3.connect(self.db_path)
        query = f"""
            SELECT close 
            FROM daily 
            WHERE ts_code = '{ts_code}' 
              AND trade_date = '{trade_date}'
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(df) > 0:
            return df['close'].iloc[0]
        return None
    
    def select_stocks_for_month(self, rebalance_date):
        """
        为某月调仓日选股
        
        Args:
            rebalance_date: 调仓日期 (YYYYMMDD)
            
        Returns:
            list: 选出的股票列表 (最多5只)
        """
        # 使用value_stock_selector选股
        # 需要设置report_date为调仓日期前的最新财报日期
        report_date = self._get_latest_report_date(rebalance_date)
        
        print(f"\n  📅 调仓日: {rebalance_date}, 使用财报日期: {report_date}")
        
        # 临时修改selector的date和report_date
        original_date = self.selector.date
        original_report_date = self.selector.report_date
        self.selector.date = rebalance_date
        self.selector.report_date = report_date
        
        # 执行选股（下跌通道风控筛经 self.downside_filter 透传，默认关）
        selected_df = self.selector.select_stocks(downside_filter=self.downside_filter)
        
        # 恢复原始日期
        self.selector.date = original_date
        self.selector.report_date = original_report_date
        
        if selected_df is None or len(selected_df) == 0:
            print(f"  ⚠️ 没有选出股票")
            return []
        
        # 按综合评分排序，取前10只（备用）
        if 'score' in selected_df.columns:
            selected_df = selected_df.sort_values('score', ascending=False)
        
        print(f"  ✓ 选出 {len(selected_df)} 只股票，准备选取前{self.top_n}只")

        # 选取前 top_n 只（检查是否够买1手）
        selected_stocks = []
        for _, row in selected_df.iterrows():
            if len(selected_stocks) >= self.top_n:
                break
            
            ts_code = row['ts_code']
            price = self.get_stock_price(ts_code, rebalance_date)
            
            if price is None:
                print(f"    ⚠️ {ts_code} 在 {rebalance_date} 无数据，跳过")
                continue
            
            # 检查是否够买1手（100股）
            if 10000 >= price * 100:  # 1万元 >= 1手金额
                selected_stocks.append({
                    'ts_code': ts_code,
                    'name': row.get('name', ts_code),
                    'price': price,
                    'shares': int(10000 / (price * 100)) * 100  # 向下取整到100股
                })
                print(f"    ✓ {ts_code} ({row.get('name', '')}) 价格{price:.2f}，可买{int(10000 / (price * 100)) * 100}股")
            else:
                print(f"    ⚠️ {ts_code} ({row.get('name', '')}) 价格{price:.2f}，1万元不够买1手，跳过")
        
        return selected_stocks
    
    def _get_latest_report_date(self, trade_date):
        """
        获取交易日期前的最新财报日期
        
        Args:
            trade_date: 交易日期 (YYYYMMDD)
            
        Returns:
            str: 财报日期 (YYYYMMDD)
        """
        year = int(trade_date[:4])
        month = int(trade_date[4:6])
        
        # 财报发布时间：Q1=4月底, Q2=8月底, Q3=10月底, Q4=次年4月底
        if month <= 4:  # 1-4月，使用去年Q3财报
            return f"{year-1}0930"
        elif month <= 8:  # 5-8月，使用今年Q1财报
            return f"{year}0331"
        elif month <= 10:  # 9-10月，使用今年Q2财报
            return f"{year}0630"
        else:  # 11-12月，使用今年Q3财报
            return f"{year}0930"
    
    def run_backtest(self, start_date, end_date):
        """
        运行回测
        
        Args:
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            
        Returns:
            dict: 回测结果
        """
        print("=" * 80)
        print("价值投资策略回测")
        print("=" * 80)
        print(f"回测周期: {start_date} 至 {end_date}")
        print(f"初始资金: {self.initial_cash:,.0f} 元")
        print(f"策略规则: {'每年首个交易日' if self.freq=='yearly' else '每月第五个交易日'}调仓，"
              f"每次选{self.top_n}只股票，每只投入1万元")
        print(f"增强开关: 市值中性化={'开' if self.size_neutral else '关'} | "
              f"BM分位筛选={('前%.0f%%'%(self.value_pct*100)) if self.value_pct else '关'} | "
              f"模式={self.value_mode} | 池={self.stock_pool} | "
              f"下跌通道风控筛={'开' if self.downside_filter else '关'}")
        print("=" * 80)

        # 获取所有交易日
        trading_days = self.get_trading_days(start_date, end_date)
        print(f"\n📅 总交易日数: {len(trading_days)}")

        # 获取调仓日
        rebalance_days = self.get_rebalance_days(trading_days, self.freq)
        print(f"📅 调仓次数: {len(rebalance_days)} 次")
        print(f"   调仓日期: {rebalance_days[:5]}...")  # 显示前5个
        
        # 初始化持仓和资金
        cash = self.initial_cash
        positions = []  # [{ts_code, name, shares, cost_price, cost_amount}]
        portfolio_values = []  # 记录每日组合价值
        
        # 逐日模拟
        print(f"\n开始模拟交易...")
        
        for i, trade_date in enumerate(trading_days):
            # 检查是否需要调仓
            if trade_date in rebalance_days:
                print(f"\n{'='*60}")
                print(f"🔄 调仓日: {trade_date}")
                
                # 卖出当前持仓
                if len(positions) > 0:
                    print(f"  卖出当前持仓:")
                    for pos in positions:
                        price = self.get_stock_price(pos['ts_code'], trade_date)
                        if price:
                            sell_amount = pos['shares'] * price
                            cash += sell_amount
                            print(f"    ✓ {pos['ts_code']} ({pos['name']}) 卖出 {pos['shares']}股 @ {price:.2f} = {sell_amount:,.0f}元")
                        else:
                            print(f"    ⚠️ {pos['ts_code']} 无数据，无法卖出")
                
                # 清空持仓
                positions = []
                
                # 选股
                selected_stocks = self.select_stocks_for_month(trade_date)
                
                # 买入新股票
                if len(selected_stocks) > 0:
                    cash_per_stock = cash / len(selected_stocks)
                    print(f"\n  买入新持仓 (每只 {cash_per_stock:,.0f} 元):")
                    
                    for stock in selected_stocks:
                        # 实际买入金额
                        max_shares = int(cash_per_stock / (stock['price'] * 100)) * 100
                        if max_shares > 0:
                            actual_shares = min(stock['shares'], max_shares)
                            cost = actual_shares * stock['price']
                            cash -= cost
                            
                            positions.append({
                                'ts_code': stock['ts_code'],
                                'name': stock['name'],
                                'shares': actual_shares,
                                'cost_price': stock['price'],
                                'cost_amount': cost
                            })
                            
                            print(f"    ✓ {stock['ts_code']} ({stock['name']}) 买入 {actual_shares}股 @ {stock['price']:.2f} = {cost:,.0f}元")
            
            # 计算当日组合价值
            portfolio_value = cash
            for pos in positions:
                price = self.get_stock_price(pos['ts_code'], trade_date)
                if price:
                    portfolio_value += pos['shares'] * price
            
            portfolio_values.append({
                'trade_date': trade_date,
                'portfolio_value': portfolio_value,
                'cash': cash
            })
        
        # 回测结束，计算收益
        final_value = portfolio_values[-1]['portfolio_value']
        total_return = (final_value - self.initial_cash) / self.initial_cash
        
        print("\n" + "=" * 80)
        print("回测结果")
        print("=" * 80)
        print(f"初始资金: {self.initial_cash:,.0f} 元")
        print(f"最终资产: {final_value:,.0f} 元")
        print(f"总收益: {final_value - self.initial_cash:,.0f} 元")
        print(f"收益率: {total_return:.2%}")
        print(f"年化收益: {total_return / 2 * 100:.2f}% (2年)")
        print("=" * 80)
        
        # 保存结果
        result_df = pd.DataFrame(portfolio_values)
        result_df['return'] = (result_df['portfolio_value'] / self.initial_cash - 1)
        
        output_file = f"data/results/value_strategy/backtest_result_{start_date}_{end_date}.csv"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"\n✓ 结果已保存: {output_file}")
        
        return {
            'initial_cash': self.initial_cash,
            'final_value': final_value,
            'total_return': total_return,
            'result_df': result_df
        }

def main():
    """主函数（支持命令行参数）"""
    import argparse
    ap = argparse.ArgumentParser(description="价值投资策略回测（破净/纯BM + 可选市值中性化/BM分位/年度调仓）")
    ap.add_argument("--start", default="20220102")
    ap.add_argument("--end", default="20231229")
    ap.add_argument("--initial-cash", type=float, default=50000)
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--pool", default="zz800",
                    choices=["hs300", "zz500", "zz800", "zz1000"])
    ap.add_argument("--freq", default="monthly", choices=["monthly", "yearly"],
                    help="monthly=每月调仓 / yearly=每年调仓(持有>=1年，验证Fama-French长期有效)")
    ap.add_argument("--size-neutral", action="store_true",
                    help="市值中性化：对BM回归掉市值取残差作纯价值得分")
    ap.add_argument("--value-pct", type=float, default=None,
                    help="BM分位筛选，如0.3=全市场BM前30%（Fama-French前20-30%口径）")
    ap.add_argument("--mode", default="pobreak", choices=["pobreak", "pure_bm"],
                    help="pobreak=破净价值(PB<1+ROE质量) | pure_bm=放宽破净·全市场BM前N%门槛")
    ap.add_argument("--downside-filter", action="store_true",
                    help="下跌通道风控筛：剔除仍处下跌通道(贴lows/在MA下/量未缩)的候选，降波动不增收益")
    ap.add_argument("--portfolio-layer", default=None,
                    help="组合层分散: 权重 equity,bond,gold 逗号分隔(如 0.7,0.15,0.15), "
                         "把本策略日频NAV与国债511260+黄金518880做月度再平衡。默认None=不开启")
    ap.add_argument("--layer-scheme", default="static", choices=["static", "invvol"],
                    help="组合层权重方案: static=固定权重 | invvol=逆波动月度")
    args = ap.parse_args()

    backtest = ValueStrategyBacktest(
        initial_cash=args.initial_cash,
        db_path="D:/tu-shareData/astock_daily.db",
        freq=args.freq,
        size_neutral=args.size_neutral,
        value_pct=args.value_pct,
        top_n=args.top_n,
        stock_pool=args.pool,
        value_mode=args.mode,
        downside_filter=args.downside_filter,
    )

    result = backtest.run_backtest(start_date=args.start, end_date=args.end)

    # 组合层分散 (opt-in, 默认不开启; 不动现有选股/持仓逻辑)
    if args.portfolio_layer:
        from portfolio_layer import PortfolioLayer
        wk = [float(x) for x in args.portfolio_layer.split(",")]
        lowcorr = ("bond", "gold")
        names = ["equity"] + list(lowcorr)
        weights = {n: wk[i] for i, n in enumerate(names)}
        layer = PortfolioLayer(
            equity_csv=f"data/results/value_strategy/backtest_result_{args.start}_{args.end}.csv",
            weights=weights, lowcorr=lowcorr, scheme=args.layer_scheme,
            equity_col="portfolio_value")
        layer.run().report()

    print("\n✓ 回测完成！")

if __name__ == "__main__":
    main()
