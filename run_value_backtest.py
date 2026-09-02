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
                 stock_pool="zz800", value_mode="pobreak", downside_filter=False,
                 price_mode="hfq", pb_gate="off", pb_gate_lo=30.0, pb_gate_hi=70.0,
                 pb_gate_lag=1, pb_gate_universe="all"):
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
            price_mode: NAV 计价口径 raw(原始价，不含分红) / hfq(后复权，逐持仓 buy_factor 归一化，
                        含分红+送转；买价与整手判定恒为 raw)
            pb_gate: 破净率 overlay 开关 off=不加(默认,现有行为) / pct=滚动分位 / abs=绝对破净率
            pb_gate_lo / pb_gate_hi: 阈值。pct 模式为分位(0-100)，abs 模式为百分比(如 10.0 = 10%%)
            pb_gate_lag: 信号滞后交易日数（默认1=用调仓日前一交易日的破净率，避免"当日收盘价已知"质疑）
            pb_gate_universe: 破净率口径 all=全A / nobj=剔北交所 / clean=再剔ST
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
        self.price_mode = price_mode
        # ---- 破净率 overlay（opt-in，默认 off = 完全保持现有行为）
        self.pb_gate = pb_gate
        self.pb_gate_lo = pb_gate_lo
        self.pb_gate_hi = pb_gate_hi
        self.pb_gate_lag = int(pb_gate_lag)
        self.pb_gate_universe = pb_gate_universe
        self._nb_series = None      # {trade_date: rate}，懒加载
        self._nb_pct = None         # {trade_date: 750日滚动分位}
        self._px_cache = {}         # (ts_code, trade_date) -> close，减少重复查询
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
        # overlay 双账模式下同一 (code, date) 会被查两遍，加缓存避免翻倍
        key = (ts_code, trade_date)
        if key in self._px_cache:
            return self._px_cache[key]

        conn = sqlite3.connect(self.db_path)
        query = f"""
            SELECT close
            FROM daily
            WHERE ts_code = '{ts_code}'
              AND trade_date = '{trade_date}'
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        price = df['close'].iloc[0] if len(df) > 0 else None
        self._px_cache[key] = price
        return price

    def _adj_factor(self, ts_code, trade_date):
        """查某股票截至某交易日的**最近一个** adj_factor（自带 as-of 语义，天然 ffill）。

        缺行日（全市场同缺 132 天那类）自动继承前一交易日的因子；
        该代码在 trade_date 之前无任何因子记录时返回 None（调用方按 1.0 处理并如实降级）。
        """
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            "SELECT adj_factor FROM adj_factor WHERE ts_code=? AND trade_date<=? "
            "ORDER BY trade_date DESC LIMIT 1",
            conn, params=(ts_code, str(trade_date)))
        conn.close()
        if len(df) > 0 and df['adj_factor'].iloc[0]:
            return float(df['adj_factor'].iloc[0])
        return None

    # ------------------------------------------------- 破净率 overlay（opt-in）
    def _load_net_break(self):
        """懒加载全市场破净率日序列。

        优先读 `net_break_rate.py` 的缓存 CSV；没有则现场构建。
        🔴 按项目约定，数据缺失**不静默降级**，直接 raise 让调用方补全。
        """
        if self._nb_series is not None:
            return
        import os
        import bisect
        cache = os.path.join("data", "results", "net_break", "market_net_break.csv")
        if os.path.exists(cache):
            df = pd.read_csv(cache, dtype={"trade_date": str})
        else:
            import net_break_rate
            df = net_break_rate.build_series()
        col = {"all": "rate_all", "nobj": "rate_nobj",
               "clean": "rate_clean"}[self.pb_gate_universe]
        df = df[["trade_date", col]].dropna().sort_values("trade_date").reset_index(drop=True)
        df["pct"] = df[col].rolling(750, min_periods=250).apply(
            lambda w: (w[-1] >= w[:-1]).mean() * 100, raw=True)
        self._nb_series = dict(zip(df["trade_date"], df[col]))
        self._nb_pct = dict(zip(df["trade_date"], df["pct"]))
        self._nb_keys = sorted(self._nb_series.keys())
        self._bisect = bisect
        print(f"[破净率] 载入 {len(self._nb_series)} 个交易日，口径={col}，lag={self.pb_gate_lag}")

    def _nb_value(self, trade_date):
        """取（调仓日 − lag）的破净率及其分位，as-of 语义天然 ffill。"""
        if self._nb_series is None:
            self._load_net_break()
        keys = self._nb_keys
        i = self._bisect.bisect_right(keys, str(trade_date)) - 1 - self.pb_gate_lag
        if i < 0:
            return None, None
        d = keys[i]
        return self._nb_series[d], self._nb_pct[d]

    def _gate_exposure(self, trade_date):
        """破净率 → 目标仓位（1.0 满仓 / 0.5 半仓 / 0.0 空仓）。

        方向：破净率**高** = 市场便宜 = 该满仓（与"PB 分位低→满仓"是同一逻辑的镜像）。
        信号缺失（数据不足或窗口不够）→ **回退满仓**，绝不擅自降仓。
        """
        if self.pb_gate == "off":
            return 1.0, None
        rate, pct = self._nb_value(trade_date)
        if rate is None:
            return 1.0, None
        if self.pb_gate == "abs":
            v = rate * 100.0
        else:
            if pct != pct:
                return 1.0, None
            v = pct
        if v >= self.pb_gate_hi:
            return 1.0, v
        if v <= self.pb_gate_lo:
            return 0.0, v
        return 0.5, v

    # --- 以下三个是 overlay 双账模式的共用原子操作 ---
    # 设计要点：选股**只做一次**，两本账买同一个篮子，唯一差别是仓位缩放。
    # 这样 A/B 对照把"选股"这个最慢也最容易引入差异的环节完全共享，
    # 隔离出的变量只有 exposure 本身（同 --ic-mode 那样单变量对照的思路）。
    def _sell_all(self, positions, trade_date, cash, tag=""):
        t = f"[{tag}] " if tag else ""
        for pos in positions:
            price = self.get_stock_price(pos['ts_code'], trade_date)
            if price:
                sell_px = price * self._hfq_ratio(pos, trade_date)
                sell_amount = pos['shares'] * sell_px
                cash += sell_amount
                print(f"    ✓ {t}{pos['ts_code']} ({pos['name']}) 卖出 "
                      f"{pos['shares']}股 @ {sell_px:.2f} = {sell_amount:,.0f}元")
            else:
                print(f"    ⚠️ {t}{pos['ts_code']} 无数据，无法卖出")
        return cash

    def _buy_all(self, cash, selected_stocks, trade_date, exposure, tag=""):
        deploy = cash * exposure
        cps = deploy / len(selected_stocks) if selected_stocks else 0.0
        t = f"[{tag}] " if tag else ""
        print(f"\n  买入新持仓 {t}(可部署 {deploy:,.0f} 元 / 每只 {cps:,.0f} 元):")
        positions = []
        if cps <= 0 or not selected_stocks:
            return cash, positions
        for stock in selected_stocks:
            max_shares = int(cps / (stock['price'] * 100)) * 100
            if max_shares > 0:
                actual_shares = min(stock['shares'], max_shares)
                cost = actual_shares * stock['price']
                cash -= cost
                positions.append({
                    'ts_code': stock['ts_code'],
                    'name': stock['name'],
                    'shares': actual_shares,
                    'cost_price': stock['price'],
                    'cost_amount': cost,
                    # hfq 归一化基准因子（买入日 as-of）；raw 模式下不用
                    'buy_factor': (self._adj_factor(stock['ts_code'], trade_date)
                                   if self.price_mode == "hfq" else 1.0),
                })
                print(f"    ✓ {t}{stock['ts_code']} ({stock['name']}) 买入 "
                      f"{actual_shares}股 @ {stock['price']:.2f} = {cost:,.0f}元")
        return cash, positions

    def _mark(self, cash, positions, trade_date):
        v = cash
        for pos in positions:
            price = self.get_stock_price(pos['ts_code'], trade_date)
            if price:
                v += pos['shares'] * price * self._hfq_ratio(pos, trade_date)
        return v

    def _hfq_ratio(self, pos, trade_date):
        """hfq 估值比率 = f(今) / f(买入)。raw 模式恒为 1.0。

        - adj_factor 单调不减 → 比率 ≥ 1（含分红+送转的累计贡献）；
        - 买入日因子缺失（fb=None）时**永久锁 1.0**——绝不拿 f(今)/1.0，
          否则绝对因子值（~7.8×）会整段虚增（同 §12.13 跨空间教训）。
        """
        if self.price_mode != "hfq":
            return 1.0
        fb = pos.get('buy_factor')
        if not fb:
            return 1.0
        ft = self._adj_factor(pos['ts_code'], trade_date)
        if not ft:
            return 1.0
        return float(ft) / float(fb)
    
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
        if self.price_mode == "hfq":
            print("【NAV 计价口径】hfq 后复权（逐持仓 buy_factor 归一化：估值/卖出=f(今)/f(买入)×raw，"
                  "含分红+送转；买入价与整手判定恒为 raw）。无全收益基准可比——本脚本不打印超额。")
        else:
            print("【NAV 计价口径】raw 原始价（不含分红；送转股会导致持仓市值凭空蒸发，见审计报告 §10）")
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

        # overlay 双账：账 G 受 gate 控制，账 F 恒满仓做对照
        dual = self.pb_gate != "off"
        cash_f, positions_f, exp = self.initial_cash, [], 1.0

        # 逐日模拟
        print(f"\n开始模拟交易...")

        for i, trade_date in enumerate(trading_days):
            # 检查是否需要调仓
            if trade_date in rebalance_days:
                print(f"\n{'='*60}")
                print(f"🔄 调仓日: {trade_date}")

                if dual:
                    exp, sig_v = self._gate_exposure(trade_date)
                    desc = {1.0: "满仓", 0.5: "半仓", 0.0: "空仓"}[exp]
                    sv = "无(回退满仓)" if sig_v is None else f"{sig_v:.2f}"
                    print(f"  🚦 破净率 overlay ({self.pb_gate}): 信号={sv} → "
                          f"仓位 {exp*100:.0f}%（{desc}）")

                # 卖出当前持仓
                if len(positions) > 0:
                    print(f"  卖出当前持仓:")
                    cash = self._sell_all(positions, trade_date, cash,
                                          tag="gate" if dual else "")
                if dual and len(positions_f) > 0:
                    cash_f = self._sell_all(positions_f, trade_date, cash_f, tag="对照")

                # 清空持仓
                positions = []
                positions_f = []

                # 选股（双账共享同一篮子）
                selected_stocks = self.select_stocks_for_month(trade_date)

                # 买入新股票
                if len(selected_stocks) > 0:
                    cash, positions = self._buy_all(cash, selected_stocks, trade_date,
                                                    exp, tag="gate" if dual else "")
                    if dual:
                        cash_f, positions_f = self._buy_all(
                            cash_f, selected_stocks, trade_date, 1.0, tag="对照")

            # 计算当日组合价值
            portfolio_value = self._mark(cash, positions, trade_date)

            row = {
                'trade_date': trade_date,
                'portfolio_value': portfolio_value,
                'cash': cash
            }
            if dual:
                row['portfolio_value_full'] = self._mark(cash_f, positions_f, trade_date)
                row['exposure'] = exp
            portfolio_values.append(row)
        
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

        # overlay 双账对照（同选股、同路径，唯一变量 = 仓位）
        if dual:
            dfv = pd.DataFrame(portfolio_values)
            stat = {}
            for col, lab in [("portfolio_value", "gate(破净率择时)"),
                             ("portfolio_value_full", "对照(恒满仓)")]:
                s = dfv[col].astype(float)
                yrs = len(s) / 244.0
                tot = s.iloc[-1] / s.iloc[0] - 1
                ann = (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else np.nan
                mdd = (s / s.cummax() - 1).min()
                r = s.pct_change().dropna()
                shp = r.mean() / r.std() * np.sqrt(244) if r.std() > 0 else np.nan
                stat[lab] = {"总收益%": round(tot * 100, 2), "年化%": round(ann * 100, 2),
                             "最大回撤%": round(mdd * 100, 2), "夏普": round(shp, 2)}
            cmp_df = pd.DataFrame(stat).T
            print("\n" + "=" * 80)
            print(f"破净率 overlay 对照（{self.pb_gate} lo={self.pb_gate_lo} "
                  f"hi={self.pb_gate_hi} lag={self.pb_gate_lag}）")
            print("=" * 80)
            print(cmp_df.to_string())
            d_ann = stat["gate(破净率择时)"]["年化%"] - stat["对照(恒满仓)"]["年化%"]
            d_mdd = stat["gate(破净率择时)"]["最大回撤%"] - stat["对照(恒满仓)"]["最大回撤%"]
            print(f"\n年化差(gate − 满仓) = {d_ann:+.2f}pp   回撤差 = {d_mdd:+.2f}pp")
            print(f"平均仓位 = {dfv['exposure'].mean()*100:.1f}%   "
                  f"空仓天数占比 = {(dfv['exposure']==0).mean()*100:.1f}%   "
                  f"满仓天数占比 = {(dfv['exposure']==1).mean()*100:.1f}%")

            # 逐年
            dfv["year"] = dfv["trade_date"].str[:4]
            yr = []
            for y, s in dfv.groupby("year"):
                a = s["portfolio_value"].iloc[-1] / s["portfolio_value"].iloc[0] - 1
                b = s["portfolio_value_full"].iloc[-1] / s["portfolio_value_full"].iloc[0] - 1
                yr.append({"year": y, "gate%": round(a * 100, 2),
                           "满仓%": round(b * 100, 2), "差%": round((a - b) * 100, 2)})
            ydf = pd.DataFrame(yr)
            print("\n--- 逐年 ---")
            print(ydf.to_string(index=False))
            win = (ydf["差%"] > 0).sum()
            print(f"\ngate 跑赢满仓年份 {win}/{len(ydf)}")
        
        # 保存结果
        result_df = pd.DataFrame(portfolio_values)
        result_df['return'] = (result_df['portfolio_value'] / self.initial_cash - 1)
        
        pm_tag = "_hfq" if self.price_mode == "hfq" else ""
        output_file = f"data/results/value_strategy/backtest_result{pm_tag}_{start_date}_{end_date}.csv"
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
                    help="BM分位筛选，如0.3=全市场BM前30%%（Fama-French前20-30%%口径）")
    ap.add_argument("--mode", default="pobreak", choices=["pobreak", "pure_bm"],
                    help="pobreak=破净价值(PB<1+ROE质量) | pure_bm=放宽破净·全市场BM前N%%门槛")
    ap.add_argument("--downside-filter", action="store_true",
                    help="下跌通道风控筛：剔除仍处下跌通道(贴lows/在MA下/量未缩)的候选，降波动不增收益")
    ap.add_argument("--price-mode", default="hfq", choices=["raw", "hfq"],
                    help="NAV 计价口径: hfq=后复权(默认,逐持仓 buy_factor 归一化,含分红+送转,真实总回报) | "
                         "raw=原始价(旧口径,不含分红;送转致持仓市值凭空蒸发,见审计报告 §10). raw 仅供复现历史结论")
    # ---- 破净率 overlay（opt-in；默认 off = 完全保持现有行为，被 import 复用也不会变）
    ap.add_argument("--pb-gate", default="off", choices=["off", "pct", "abs"],
                    help="全市场破净率择时 overlay: off=不加(默认) | pct=滚动分位 | abs=绝对破净率")
    ap.add_argument("--pb-gate-lo", type=float, default=30.0,
                    help="下阈值: pct 模式为分位(默认30) = 低于此值空仓; abs 模式为百分比(如 5.0 = 5%%)")
    ap.add_argument("--pb-gate-hi", type=float, default=70.0,
                    help="上阈值: pct 模式为分位(默认70) = 高于此值满仓; abs 模式为百分比(如 10.0 = 10%%)")
    ap.add_argument("--pb-gate-lag", type=int, default=1,
                    help="信号滞后交易日数(默认1): 用调仓日前一交易日的破净率, 避免当日收盘价已知才可得的质疑")
    ap.add_argument("--pb-gate-universe", default="all", choices=["all", "nobj", "clean"],
                    help="破净率口径: all=全A | nobj=剔北交所 | clean=再剔ST")
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
        price_mode=args.price_mode,
        pb_gate=args.pb_gate,
        pb_gate_lo=args.pb_gate_lo,
        pb_gate_hi=args.pb_gate_hi,
        pb_gate_lag=args.pb_gate_lag,
        pb_gate_universe=args.pb_gate_universe,
    )

    result = backtest.run_backtest(start_date=args.start, end_date=args.end)

    # 组合层分散 (opt-in, 默认不开启; 不动现有选股/持仓逻辑)
    if args.portfolio_layer:
        from portfolio_layer import PortfolioLayer
        wk = [float(x) for x in args.portfolio_layer.split(",")]
        lowcorr = ("bond", "gold")
        names = ["equity"] + list(lowcorr)
        weights = {n: wk[i] for i, n in enumerate(names)}
        pm_tag = "_hfq" if args.price_mode == "hfq" else ""
        layer = PortfolioLayer(
            equity_csv=f"data/results/value_strategy/backtest_result{pm_tag}_{args.start}_{args.end}.csv",
            weights=weights, lowcorr=lowcorr, scheme=args.layer_scheme,
            equity_col="portfolio_value")
        layer.run().report()

    print("\n✓ 回测完成！")

if __name__ == "__main__":
    main()
