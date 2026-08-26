"""
portfolio_layer.py — 组合层低相关分散 (opt-in 模块, 验证 gated_alpha_report.md §5)

用途
----
把任一 equity 策略的日频 NAV(来自回测落盘 CSV) 与低相关 sleeve(国债ETF/黄金ETF)
做月度再平衡组合, 看是否比"满仓股票"更优(同等降尾下少牺牲收益 / 更高夏普)。

与 run_diversified_portfolio.py 的关系
--------------------------------
run_diversified_portfolio.py 用预计算的 alpha_nav.csv(静态) 演示组合层效果;
本模块是"可复用落地层": 任何策略跑完得到日频 NAV CSV 后, 直接喂进来混债券/黄金,
不依赖 alpha_nav.csv。用于把组合层从"演示"变成"真实引擎可调用"。

纪律
----
- 无未来函数: 月度权重用 <= 上月末 的滚动波动(仅 invvol 方案用; static 用固定权重)
- 后复权 NAV = close * adj_factor(累计), 防除权假跳
- 成本: 默认不计(组合层对比); 真实落地时应加月度换手成本(见 README 注)
- 本模块只计算、不写库、不改任何现有策略逻辑

用法
----
    from portfolio_layer import PortfolioLayer
    layer = PortfolioLayer(equity_csv="data/results/value_strategy/backtest_result_20180101_20251231.csv",
                            weights={"equity":0.7,"bond":0.15,"gold":0.15})
    layer.run().report()
"""
import sqlite3
import numpy as np
import pandas as pd

DB = "D:/tu-shareData/astock_daily.db"

# 低相关 sleeve 定义 (etf_daily + etf_adj_factor)
LOWCORR = {
    "bond": "511260.SH",   # 国债10Y ETF
    "gold": "518880.SH",   # 黄金 ETF
    "cash": "511990.SH",   # 货币 ETF (现金替代)
}


def load_etf_adj_nav(con, code):
    """后复权 NAV(Series 起点=1.0), 索引=trade_date(str)。"""
    df = pd.read_sql(
        f"SELECT trade_date, close FROM etf_daily WHERE ts_code='{code}' ORDER BY trade_date",
        con)
    af = pd.read_sql(
        f"SELECT trade_date, adj_factor FROM etf_adj_factor WHERE ts_code='{code}' ORDER BY trade_date",
        con)
    if df.empty:
        return None
    df["trade_date"] = df["trade_date"].astype(str)
    if not af.empty:
        af["trade_date"] = af["trade_date"].astype(str)
        m = df.merge(af, on="trade_date", how="left").sort_values("trade_date")
        m["adj_factor"] = m["adj_factor"].ffill().bfill()
        m["adj_close"] = m["close"] * (m["adj_factor"] / m["adj_factor"].iloc[0])
    else:
        m = df.copy()
        m["adj_close"] = m["close"]
    s = (m["adj_close"] / m["adj_close"].iloc[0]).rename(code)
    s.index = m["trade_date"].astype(str)
    return s


def load_cash_rf(con):
    """货币ETF日收益作无风险利率(年化)。"""
    nav = load_etf_adj_nav(con, LOWCORR["cash"])
    if nav is None:
        return 0.0
    r = nav.pct_change().dropna()
    return r.mean() * 252


class PortfolioLayer:
    def __init__(self, equity_csv=None, equity_series=None,
                 weights=None, lowcorr=("bond", "gold"),
                 scheme="static", equity_col="portfolio_value",
                 rf_from_cash=True):
        """
        equity_csv: 策略回测落盘的日频 NAV CSV(需含 trade_date + equity_col)
        equity_series: 或直传 Series(索引=trade_date(str), 值=组合价值)
        weights: dict, 如 {"equity":0.7,"bond":0.15,"gold":0.15}
                 若 scheme="invvol" 则忽略 weights, 用逆波动加权
        lowcorr: tuple, 选用哪几个低相关 sleeve
        scheme: static(用 weights) | invvol(逆波动, 月度)
        """
        if equity_csv is not None:
            df = pd.read_csv(equity_csv)
            df["trade_date"] = df["trade_date"].astype(str)
            if equity_col not in df.columns:
                # 兼容 alpha_nav.csv 风格(列名=策略名)
                equity_col = [c for c in df.columns if c != "trade_date"][0]
            s = df.set_index("trade_date")[equity_col]
            self.equity = (s / s.iloc[0]).rename("equity")
        elif equity_series is not None:
            self.equity = (equity_series / equity_series.iloc[0]).rename("equity")
        else:
            raise ValueError("需提供 equity_csv 或 equity_series")
        self.weights = weights or {"equity": 0.7, "bond": 0.15, "gold": 0.15}
        self.lowcorr = [k for k in lowcorr if k in LOWCORR]
        self.scheme = scheme
        self.rf_from_cash = rf_from_cash
        self._con = sqlite3.connect(DB)

    def _load_sleeves(self):
        navs = {"equity": self.equity}
        for k in self.lowcorr:
            s = load_etf_adj_nav(self._con, LOWCORR[k])
            if s is not None:
                navs[k] = s
        # 对齐到 equity 日历(缺失日=持有 ffill/bfill)
        idx = sorted(self.equity.index)
        out = {}
        for name, s in navs.items():
            if name == "equity":
                out[name] = s.reindex(idx)
            else:
                out[name] = s.reindex(idx).ffill().bfill()
        out["equity"] = out["equity"].dropna()
        returns = pd.DataFrame({n: s.pct_change().dropna()
                                for n, s in out.items()}).dropna(how="any")
        return returns

    def _monthly_weights(self, returns):
        names = list(returns.columns)
        idx = returns.index
        month = pd.Series(idx, index=idx).str[:6]
        is_start = month != month.shift(1)
        w = pd.DataFrame(0.0, index=idx, columns=names)
        cur = None
        if self.scheme == "invvol":
            vol = returns.rolling(60).std() * np.sqrt(252)
        for i, d in enumerate(idx):
            if is_start.iloc[i] or cur is None:
                if self.scheme == "static":
                    rest = (1 - self.weights.get("equity", 0.7)) / (len(names) - 1)
                    cur = {n: (self.weights.get("equity", 0.7) if n == "equity"
                              else rest) for n in names}
                else:  # invvol
                    v = vol.iloc[i - 1] if i > 0 else vol.iloc[0]
                    v = v.replace(0, np.nan)
                    inv = 1.0 / v
                    inv = inv.fillna(inv.mean())
                    tot = inv.sum()
                    cur = {n: inv[n] / tot for n in names}
            for n in names:
                w.at[d, n] = cur[n]
        return w

    @staticmethod
    def _metrics(nav, daily, rf_annual):
        n = len(daily)
        years = n / 252.0
        total = nav.iloc[-1] / nav.iloc[0] - 1
        cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1
        vol = daily.std() * np.sqrt(252)
        rf_d = rf_annual / 252.0
        sharpe = ((daily.mean() - rf_d) / daily.std() * np.sqrt(252)
                  if daily.std() > 0 else 0.0)
        peak = nav.cummax()
        mdd = (nav / peak - 1).min()
        q = daily.quantile(0.01)
        cvar = daily[daily <= q].mean()
        return {"总收益": total, "年化": cagr, "年化波动": vol,
                "夏普": sharpe, "最大回撤": mdd, "CVaR99(日)": cvar,
                "年数": years, "rf": rf_annual}

    def run(self):
        returns = self._load_sleeves()
        self.d0, self.d1 = returns.index[0], returns.index[-1]
        self.rf = load_cash_rf(self._con) if self.rf_from_cash else 0.0
        self._con.close()
        # 纯 equity 基线
        eq_nav = (1 + returns["equity"]).cumprod()
        self.base = self._metrics(eq_nav, returns["equity"], self.rf)
        # blended
        w = self._monthly_weights(returns)
        w = w.reindex(returns.index).ffill()
        pr = (returns * w).sum(axis=1)
        nav = (1 + pr).cumprod()
        self.blended = self._metrics(nav, pr, self.rf)
        self.weights_mean = w.mean().to_dict()
        self.corr = returns.corr()
        return self

    def report(self):
        print(f"\n=== 组合层 {self.scheme} (窗口 {self.d0}~{self.d1}) ===")
        print(f"无风险利率(年化, 货基): {self.rf:.4f}")
        print(f"平均权重: {self.weights_mean}")
        df = pd.DataFrame([self.base, self.blended],
                          index=["纯equity", "blended"]).T
        print(df.to_string(float_format=lambda x: f"{x:.4f}"))
        print("\n=== 日收益相关性 ===")
        print(self.corr.to_string(float_format=lambda x: f"{x:.3f}"))
        return self


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--equity-csv", required=True)
    ap.add_argument("--weights", default="0.7,0.15,0.15",
                    help="equity,bond,gold 权重, 逗号分隔")
    ap.add_argument("--lowcorr", default="bond,gold")
    ap.add_argument("--scheme", default="static", choices=["static", "invvol"])
    ap.add_argument("--equity-col", default="portfolio_value")
    args = ap.parse_args()
    wk = [float(x) for x in args.weights.split(",")]
    names = ["equity"] + args.lowcorr.split(",")
    weights = {n: wk[i] for i, n in enumerate(names)}
    PortfolioLayer(equity_csv=args.equity_csv, weights=weights,
                   lowcorr=args.lowcorr.split(","), scheme=args.scheme,
                   equity_col=args.equity_col).run().report()
