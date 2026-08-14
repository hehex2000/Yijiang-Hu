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

from config import DATA, SELECTION, GLOBAL, DOGS_OF_MARKET, BACKTEST, VALUE_STRATEGY
from run_monthly_rebalance import compute_reality_discounts, stamp_duty_rate

DB_PATH = DATA.get("local_db_path", "D:/tu-shareData/astock_daily.db")
# 初始【总】资金：默认跟随 config 的 total_capital（与「选股+回测」一致），
# 也可通过 run_backtest(capital=...) 或 CLI --capital 覆盖。
# 注意：此处 capital 表示「总资金」，每只 = 总资金 ÷ 选股数。
TOTAL_CAPITAL = BACKTEST.get("total_capital", 500000)


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
    """获取单只股票在某日的「原始(未复权)」价格。

    必须与 get_hfq_price 保持相同的「前向填充」口径(取 trade_date <= 当日
    最近一条)，否则在 adj_factor/daily 存在缺口的交易日会取不到价 → 回退到
    last_raw。若 last_raw 又被误种成 hfq 价，raw 轨道会被 hfq 价污染，
    年度 raw 收益出现 -50%~+149% 的荒谬跳变。这里统一前向填充，根治该 bug。
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        f"SELECT {price_type} AS p FROM daily WHERE ts_code=? AND trade_date<=? "
        f"ORDER BY trade_date DESC LIMIT 1",
        conn, params=(ts_code, trade_date),
    )
    conn.close()
    if len(df) == 0 or df.iloc[0]["p"] is None:
        return None
    return float(df.iloc[0]["p"])


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
    # 2) 复权因子：优先取 trade_date <= 当日 最近一条（前向填充）；
    #    若该日之前无任何因子数据（本库 adj_factor 起点=20150105，晚于回测起点），
    #    则向后取「首个可用因子」作为该日前置锚点（后向填充）。
    #    【关键 bug 修复】旧逻辑在因子缺失时退化为原始价(因子=1)，导致 2014 年
    #    买入按因子=1 记账、2015-01-05 起按真实因子(raw×11.4)估值 → 单日凭空 +1038%
    #    的台阶，全程收益被放大到 +4210%（真实应约 +280~+450%）。后向填充使 hfq
    #    连续，买入/估值沿用同一因子口径，虚假台阶消失。
    fac = pd.read_sql_query(
        "SELECT adj_factor FROM adj_factor WHERE ts_code = ? AND trade_date <= ? "
        "ORDER BY trade_date DESC LIMIT 1",
        conn, params=(ts_code, trade_date),
    )
    f = None
    if len(fac) > 0 and fac.iloc[0]["adj_factor"] is not None:
        f = float(fac.iloc[0]["adj_factor"])
    else:
        fac2 = pd.read_sql_query(
            "SELECT adj_factor FROM adj_factor WHERE ts_code = ? AND trade_date > ? "
            "ORDER BY trade_date ASC LIMIT 1",
            conn, params=(ts_code, trade_date),
        )
        if len(fac2) > 0 and fac2.iloc[0]["adj_factor"] is not None:
            f = float(fac2.iloc[0]["adj_factor"])
    conn.close()
    if len(px) == 0 or px.iloc[0]["p"] is None:
        return None
    p = float(px.iloc[0]["p"])
    if f is None or f == 0:
        return p  # 全周期都无因子数据 → 退化为原始价
    return p * f


def get_adj_factor(ts_code, trade_date):
    """取单只股票在 trade_date 的复权因子(前向填充，缺则后向填充首个可用因子)。

    与 get_hfq_price 的因子口径完全一致，专供「逐持仓买入日因子归一化」记录
    buy_factor 使用。返回 float 或 None(全周期都无因子)。
    """
    conn = sqlite3.connect(DB_PATH)
    fac = pd.read_sql_query(
        "SELECT adj_factor FROM adj_factor WHERE ts_code = ? AND trade_date <= ? "
        "ORDER BY trade_date DESC LIMIT 1",
        conn, params=(ts_code, trade_date),
    )
    f = None
    if len(fac) > 0 and fac.iloc[0]["adj_factor"] is not None:
        f = float(fac.iloc[0]["adj_factor"])
    else:
        fac2 = pd.read_sql_query(
            "SELECT adj_factor FROM adj_factor WHERE ts_code = ? AND trade_date > ? "
            "ORDER BY trade_date ASC LIMIT 1",
            conn, params=(ts_code, trade_date),
        )
        if len(fac2) > 0 and fac2.iloc[0]["adj_factor"] is not None:
            f = float(fac2.iloc[0]["adj_factor"])
    conn.close()
    return f


def _load_dividends():
    """一次性加载分红明细：{ts_code: [(ex_date, cash_div_per_share), ...]}。

    数据来自 dividend_detail 表（ex_date=除权日, cash_div=每股现金分红）。
    供「现金趴账」模型使用：持有期内每到除权日，把每股分红计入闲置现金，
    而非像后复权价那样假设分红当日即再投。
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT ts_code, ex_date, cash_div FROM dividend_detail "
        "WHERE ex_date IS NOT NULL AND cash_div IS NOT NULL AND cash_div > 0",
        conn)
    conn.close()
    d = {}
    for _, r in df.iterrows():
        code = str(r["ts_code"])
        d.setdefault(code, []).append((str(r["ex_date"]), float(r["cash_div"])))
    # 每个股票的除权日按时间排序，便于按 buy_date 之后过滤
    for code in d:
        d[code].sort(key=lambda x: x[0])
    return d


_DIV_CACHE = None
def get_dividends():
    """返回分红明细缓存（进程内只加载一次）"""
    global _DIV_CACHE
    if _DIV_CACHE is None:
        _DIV_CACHE = _load_dividends()
    return _DIV_CACHE


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


# 股票名称缓存（避免每年重复查库）
_NAME_CACHE = {}
def get_stock_name(ts_code):
    """从 stock_basic 表查股票名称（带缓存）"""
    if ts_code in _NAME_CACHE:
        return _NAME_CACHE[ts_code]
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT name FROM stock_basic WHERE ts_code=?", conn, params=(ts_code,),
    )
    conn.close()
    nm = df.iloc[0]["name"] if (len(df) > 0 and df.iloc[0]["name"]) else ts_code
    _NAME_CACHE[ts_code] = nm
    return nm


# ============================================================
#  EP 行业中性选股（年度调仓版，同源 run_ep_neutral.py）
#  仅作为 EP 因子的「多头组合」落地：行业内 5 分组取最便宜 G5，
#  按行业等比例配额取 top_n（保持行业中性）。obv_filter 开启时
#  再做 OBV 吸筹过滤（便宜且有人在买）。年度化：每年初 T-1 数据选股。
# ============================================================
# 行业剔除（基于 stock_basic.industry）
_EP_FINANCIAL = {"银行", "证券", "保险", "多元金融"}
_EP_UTILITY   = {"火力发电", "水力发电", "新型电力", "供气供热", "水务"}
_EP_IPO_MIN_DAYS = 60

def _ep_get_conn():
    return sqlite3.connect(DB_PATH)

def _ep_load_basic():
    """缓存 stock_basic: ts_code -> {industry, name, list_date, excluded}"""
    conn = _ep_get_conn()
    df = pd.read_sql_query(
        "SELECT ts_code, name, industry, list_date FROM stock_basic", conn)
    conn.close()
    m = {}
    for _, r in df.iterrows():
        code = str(r["ts_code"])
        name = str(r["name"]) if pd.notna(r["name"]) else ""
        ind = str(r["industry"]) if pd.notna(r["industry"]) else None
        ld = str(r["list_date"]) if pd.notna(r["list_date"]) else ""
        excluded = (code.endswith(".BJ")
                    or "ST" in name.upper() or name.startswith("*"))
        m[code] = {"name": name, "industry": ind, "list_date": ld, "excluded": excluded}
    return m

_EP_BASIC = None
def _ep_basic():
    global _EP_BASIC
    if _EP_BASIC is None:
        _EP_BASIC = _ep_load_basic()
    return _EP_BASIC

def _ep_pool_constituents(stock_pool, asof_date):
    """返回时点成分股集合；all/None -> None（全A股不过滤）"""
    idx = {"hs300": "000300.SH", "zz500": "000905.SH", "zz800": "000906.SH",
           "zz1000": "000852.SH", "zz2000": "932000.SH", "all": None}.get(stock_pool)
    if idx is None:
        return None
    conn = _ep_get_conn()
    try:
        snap = conn.execute(
            "SELECT MAX(CAST(trade_date AS INTEGER)) FROM index_constituent "
            "WHERE index_code=? AND CAST(trade_date AS INTEGER) <= CAST(? AS INTEGER)",
            (idx, asof_date)).fetchone()
        if not snap or snap[0] is None:
            return set()
        rows = conn.execute(
            "SELECT ts_code FROM index_constituent WHERE index_code=? "
            "AND CAST(trade_date AS INTEGER)=CAST(? AS INTEGER)",
            (idx, snap[0])).fetchall()
    finally:
        conn.close()
    return set(str(r[0]) for r in rows)

def _ep_obv_filter(codes, end_date, lookback=20):
    """OBV 净流量>0（吸筹）过滤；数据不足原样返回。"""
    if not codes:
        return list(codes)
    conn = _ep_get_conn()
    dates = pd.read_sql_query(
        "SELECT DISTINCT trade_date FROM daily WHERE trade_date <= ? "
        "ORDER BY trade_date DESC LIMIT ?",
        conn, params=(end_date, lookback + 1))["trade_date"].tolist()
    if len(dates) < 2:
        conn.close()
        return list(codes)
    dates = sorted(dates)
    ph_c = ",".join("?" * len(codes))
    ph_d = ",".join("?" * len(dates))
    df = pd.read_sql_query(
        f"SELECT ts_code, trade_date, close, vol FROM daily "
        f"WHERE ts_code IN ({ph_c}) AND trade_date IN ({ph_d}) ORDER BY ts_code, trade_date ASC",
        conn, params=list(codes) + dates)
    conn.close()
    keep = []
    for code, g in df.groupby("ts_code"):
        g = g.sort_values("trade_date")
        closes = g["close"].astype(float).to_numpy()
        vols = g["vol"].astype(float).to_numpy()
        if len(closes) < 2:
            keep.append(code)
            continue
        contrib = np.sign(np.diff(closes)) * vols[1:]
        if float(contrib.sum()) > 0:
            keep.append(code)
    return keep

def _ep_top_n_industry_neutral(g5, top_n):
    """G5 内按行业等比例配额取最便宜 top_n（保持行业中性）。

    采用「最大余数法(Hamilton)」分配：先按占比向下取整，再把剩余名额
    依次分给小数部分最大的行业，且不超该行业 G5 实际数量。该算法必然
    在 top_n 只内收敛，规避旧实现在「sum(quotas)>top_n 且各行业配额
    均=1」时 while diff<0 找不到可减项而陷入死循环的 bug（年度化 + OBV
    过滤后 G5 变少、行业分散时极易触发）。
    """
    ind_counts = g5['industry'].value_counts()
    total = len(g5)
    top_n = int(min(top_n, total))
    if top_n <= 0:
        return g5.iloc[0:0]
    # 比例配额（浮点）
    raw = ind_counts / total * top_n
    quotas = np.floor(raw).astype(int)          # 向下取整基础配额
    remainder = top_n - int(quotas.sum())        # 待分配余量
    # 按小数部分从大到小分配剩余名额（最大余数法），且不超行业可用数
    frac_order = (raw - quotas).sort_values(ascending=False).index.tolist()
    for ind in frac_order:
        if remainder <= 0:
            break
        cap = int(ind_counts[ind])               # 该行业 G5 实际可入选上限
        if quotas[ind] < cap:
            quotas[ind] += 1
            remainder -= 1
    picks = []
    for ind, q in quotas.items():
        if int(q) <= 0:
            continue
        sub = g5[g5['industry'] == ind].sort_values('ep', ascending=False).head(int(q))
        picks.append(sub)
    if not picks:
        return g5.sort_values('ep', ascending=False).head(top_n)
    return pd.concat(picks)

def select_ep_neutral_annual(prev_td, top_n=None, obv_filter=None, stock_pool=None):
    """年度调仓 EP 行业中性选股（同源 run_ep_neutral.py，年度化）。

    返回 DataFrame[ts_code, ep, pe_ttm, industry]。point-in-time: 用 prev_td 的 pe_ttm。
      · EP = 1/PE_TTM，全局 1%/99% 缩尾
      · 在每个申万细分行业内按 EP 降序 5 分组，取最便宜五分位 G5
      · top_n 给定时按行业等比例配额（保持行业中性），默认持有全部 G5
      · obv_filter 给定天数时再做 OBV 吸筹过滤（便宜且有人在买）
    """
    if stock_pool is None:
        stock_pool = SELECTION.get("stock_pool", "all")
    basic = _ep_basic()
    conn = _ep_get_conn()
    pool_set = _ep_pool_constituents(stock_pool, prev_td)
    if pool_set is not None and len(pool_set) == 0:
        pool_set = None  # 无时点快照则退回全A股

    rows = pd.read_sql_query(
        "SELECT DISTINCT ts_code FROM daily WHERE trade_date = ?",
        conn, params=(prev_td,))
    trading = set(str(c) for c in rows["ts_code"].tolist())

    eligible = set()
    d_rb = datetime.strptime(prev_td, "%Y%m%d")
    for c in trading:
        if pool_set is not None and c not in pool_set:
            continue
        info = basic.get(c)
        if info is None:
            if c.endswith(".BJ"):
                continue
            eligible.add(c)
            continue
        if info["excluded"]:
            continue
        ind = info["industry"]
        if ind in _EP_FINANCIAL or ind in _EP_UTILITY:
            continue
        ld = info["list_date"]
        if ld:
            try:
                if (d_rb - datetime.strptime(ld, "%Y%m%d")).days < _EP_IPO_MIN_DAYS:
                    continue
            except Exception:
                pass
        eligible.add(c)

    pe = pd.read_sql_query(
        "SELECT ts_code, pe_ttm FROM daily_basic WHERE trade_date = ? AND pe_ttm > 0",
        conn, params=(prev_td,))
    conn.close()
    if pe.empty:
        return pd.DataFrame(columns=["ts_code"])
    pe["ts_code"] = pe["ts_code"].astype(str)
    pe = pe[pe["ts_code"].isin(eligible)].copy()
    if pe.empty:
        return pd.DataFrame(columns=["ts_code"])
    pe["industry"] = pe["ts_code"].map(lambda c: (basic.get(c) or {}).get("industry"))
    pe = pe[pe["industry"].notna()].copy()
    if pe.empty:
        return pd.DataFrame(columns=["ts_code"])

    pe["ep"] = 1.0 / pe["pe_ttm"].astype(float)
    lo, hi = pe["ep"].quantile([0.01, 0.99])
    pe["epw"] = pe["ep"].clip(lo, hi)

    pe["g"] = np.nan
    for ind, g in pe.groupby("industry"):
        if len(g) < 5:
            continue
        pe.loc[g.index, "g"] = pd.qcut(g["epw"].rank(method="first"), 5, labels=False) + 1
    pe = pe[pe["g"] == 5].copy()
    if pe.empty:
        return pd.DataFrame(columns=["ts_code"])

    if obv_filter:
        keep = set(_ep_obv_filter(pe["ts_code"].tolist(), prev_td, lookback=obv_filter))
        pe = pe[pe["ts_code"].isin(keep)].copy()
        if pe.empty:
            return pd.DataFrame(columns=["ts_code"])

    if top_n is not None and top_n > 0 and top_n < len(pe):
        pe = _ep_top_n_industry_neutral(pe, top_n)
    pe = pe.sort_values("ep", ascending=False).reset_index(drop=True)
    return pe[["ts_code", "ep", "pe_ttm", "industry"]]


def run_backtest(start_date="20200102", end_date="20261231", top_n=None, select_only=False, capital=None, strategy="dogs", value_mode="pobreak", price_mode="dual", industry_cap=0, interrupt_start=None, interrupt_months=0, interrupt_pct=0.0, trail_stop=0.0, dd_de_risk=0.0):
    """执行狗股/价值 年度调仓回测

    Args:
        capital: 初始【总】资金（元）。每只 = 总资金 ÷ 选股数，等权买入。
        strategy: 子策略 — "dogs"(狗股: 高股息+低PB+均值回归) 或 "value"(价值选股: 破净+ROE质量+自由现金流)
        value_mode: 价值选股模式 pobreak(破净) / pure_bm(放宽破净·BM分位门槛)
    """
    if top_n is None:
        top_n = SELECTION.get("top_n", 5)

    # 初始【总】资金：优先用传入 capital（命令行 --capital / bat 菜单设置），
    # 否则跟随 config 的 total_capital（与「选股+回测」一致）。
    # 此处 capital 表示「总资金」，每只 = 总资金 ÷ 选股数。
    if capital is None:
        capital = BACKTEST.get("total_capital", 500000)
    global TOTAL_CAPITAL
    TOTAL_CAPITAL = capital

    # 神奇公式提示：30只等权需每只≥3万（100万总），否则单价偏高股买不起→空仓拖累
    if strategy == "magic" and top_n > 0 and TOTAL_CAPITAL / top_n < 30000:
        print(f"\n  [提示] 神奇公式建议 N=30 + 总资金≥100万（每只≥3万）。")
        print(f"        当前每只约 {TOTAL_CAPITAL // top_n:,} 元，单价偏高股票将因"
              f"买不起被跳过，可能空仓拖累收益。")
        print(f"        可在主菜单 [3] 设置选股数量/资金，或改用 standalone: "
              f"run_magic_formula.py（5月调仓版）。\n")

    # 同步选股数量 / 股票池到两个子策略配置
    DOGS_OF_MARKET["top_n"] = top_n
    DOGS_OF_MARKET["stock_pool"] = SELECTION["stock_pool"]
    VALUE_STRATEGY["top_n"] = top_n
    VALUE_STRATEGY["value_mode"] = value_mode
    VALUE_STRATEGY["stock_pool"] = SELECTION["stock_pool"]

    # 子策略：狗股 / 价值选股 / 神奇公式(Magic Formula)
    strategy = (strategy or "dogs").lower()
    if strategy not in ("dogs", "value", "magic", "ep", "ep_obv"):
        strategy = "dogs"
    if strategy == "magic":
        strategy_name = "神奇公式(Magic Formula·ROC+EY双排名)"
    elif strategy == "value":
        strategy_name = "价值选股(破净+ROE+现金流)"
    elif strategy == "ep":
        strategy_name = "EP行业中性(行业中性G5·年度)"
    elif strategy == "ep_obv":
        strategy_name = "EP+OBV吸筹(行业中性G5·年度)"
    else:
        strategy_name = "狗股策略(高股息+低PB)"

    # 获取交易日
    trade_dates = get_trade_dates(start_date, end_date)
    if not trade_dates:
        print("[ERROR] 无交易日数据")
        return

    # 每年第一个交易日
    yearly_first = get_first_trading_days(trade_dates)
    years = sorted(yearly_first.keys())
    print(f"\n{'='*70}")
    print(f"  {strategy_name} · 年度调仓回测")
    print(f"  区间: {start_date} ~ {end_date}  |  选股: {top_n} 只")
    print(f"  总资金: {TOTAL_CAPITAL:,} 元  →  每支约 {TOTAL_CAPITAL // top_n:,} 元")
    print(f"  调仓频率: 每年初 (共 {len(years)} 年)")
    print(f"  股票池: {SELECTION.get('stock_pool', 'zz800')}")
    _rc = []
    if trail_stop > 0:
        _rc.append(f"移动止损 -{trail_stop:g}%")
    if dd_de_risk > 0:
        _rc.append(f"回撤减仓 -{dd_de_risk:g}%")
    print(f"  盘中风控: {' + '.join(_rc) if _rc else '无'}")
    print(f"{'='*70}\n")

    benchmark_idx = get_stock_pool_index()
    print(f"  基准指数: {benchmark_idx}")
    print()

    from src.dogs_of_market_selector import DogsOfMarketSelector
    from src.value_stock_selector import ValueStockSelector
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
    if strategy == "value":
        selector = ValueStockSelector(VALUE_STRATEGY, fetcher)
    else:
        selector = DogsOfMarketSelector(DOGS_OF_MARKET, fetcher)

    # 选股执行（按子策略分派；价值选股按选股年份推导报告期并多层回退，
    # 避免用到「未来」财务数据，同时放宽候选池确保能选出足够股票）
    def do_select(prev_td):
        if strategy == "magic":
            # 神奇公式：复用 run_magic_formula 的选股逻辑（point-in-time 财务+
            # 市值，剔除 ST / BJ/金融/公用）。T-1 日(prev_td)数据选股、T 日开盘执行。
            from run_magic_formula import select_magic_formula
            return select_magic_formula(prev_td, top_n=top_n, prev_date=prev_td,
                                       verbose=True, stock_pool=SELECTION.get("stock_pool", "all"))
        if strategy == "value":
            # 价值选股条件较严，先放宽候选池(×2)，最后再截断到 top_n。
            # 选股实际走模块级 select_value_stocks(prev_td)，内部按 T-1 日做 point-in-time
            # 财务过滤（end_date<=prev_td AND ann_date<=prev_td）；原 do_select 里反复设置
            # VALUE_STRATEGY["report_date"] 是死代码（selector 已实例化、且 select_stocks
            # 根本不读取 self.report_date），已移除。
            VALUE_STRATEGY["top_n"] = max(top_n * 2, top_n + 10)
            sel = selector.select_stocks(date=prev_td, top_n=VALUE_STRATEGY["top_n"])
            if sel is not None and len(sel) > 0:
                return sel.head(top_n).reset_index(drop=True)
            return None
        if strategy in ("ep", "ep_obv"):
            # EP 行业中性（年度化）：T-1 日 pe_ttm 选股、T 日开盘执行
            obv = 20 if strategy == "ep_obv" else None
            return select_ep_neutral_annual(
                prev_td, top_n=top_n, obv_filter=obv,
                stock_pool=SELECTION.get("stock_pool", "all"))
        else:
            if industry_cap and industry_cap > 0:
                # ── 单行业上限约束（A/B 实验开关，默认关闭）──
                # 扩大候选池后按 score 顺序贪心选取：同一 stock_basic.industry
                # 最多 industry_cap 只，超出则跳过取下一名。防止 2021 式单行业(证券80%)集中。
                orig_top = selector.top_n
                selector.top_n = max(top_n * 5, top_n + 20)
                try:
                    cand = selector.select_stocks(date=prev_td)
                finally:
                    selector.top_n = orig_top
                if cand is None or len(cand) == 0:
                    return cand
                basic = _ep_basic()
                picked, counts, skipped = [], {}, []
                for _, r in cand.iterrows():
                    code = str(r["ts_code"])
                    ind = (basic.get(code) or {}).get("industry") or "未知"
                    if counts.get(ind, 0) >= industry_cap:
                        skipped.append(f"{code}({ind})")
                        continue
                    counts[ind] = counts.get(ind, 0) + 1
                    picked.append(r)
                    if len(picked) >= top_n:
                        break
                out = pd.DataFrame(picked).reset_index(drop=True)
                if skipped:
                    print(f"  [行业上限≤{industry_cap}] 跳过 {len(skipped)} 只超限股: {', '.join(skipped[:6])}")
                inds = [((basic.get(str(c)) or {}).get("industry") or "未知") for c in out["ts_code"]]
                print(f"  [行业上限≤{industry_cap}] 最终行业分布: {dict(pd.Series(inds).value_counts())}")
                return out
            return selector.select_stocks(date=prev_td)

    # 回测状态
    positions = {}       # {ts_code: {"shares": N, "buy_price": P}}
    cash = TOTAL_CAPITAL           # hfq 记账现金(后复权口径)
    cash_raw = TOTAL_CAPITAL       # raw 记账现金(原始价口径，独立账本！)
    cash_idle = TOTAL_CAPITAL      # 真实趴账模型现金：分红作闲置现金，至下次调仓日才再投

    # 年内风控状态（移动止损 / 组合回撤减仓；trail_stop=0 且 dd_de_risk=0 时完全不触发，零开销）
    peak_nav = TOTAL_CAPITAL        # 组合峰值净值(raw 口径)，用于回撤减仓判定
    de_risked = False               # 是否已触发「回撤减仓」清仓至现金
    n_stop = 0                      # 移动止损触发次数（统计用）
    n_dd = 0                        # 回撤减仓触发次数（统计用）
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

    # 组合当日 NAV(raw 口径)：现金 + 各持仓市值(last_raw 即今日收盘原始价)
    def port_nav_raw():
        return cash_raw + sum(p["shares_raw"] * p["last_raw"]
                              for p in positions.values() if p.get("last_raw") is not None)

    # 以今日收盘价清仓单只持仓（与调仓卖出同口径：扣 0.045% 费用，三账本同步）
    # 仅在年内风控触发时调用，不影响正常调仓逻辑。
    def sell_at_close(code, td):
        nonlocal cash, cash_raw, cash_idle
        pos = positions.pop(code, None)
        if not pos:
            return
        rev_hfq = pos.get("last_hfq")          # 今日归一化后复权收盘(估值循环已写入)
        cr = pos.get("last_raw")               # 今日原始价收盘
        if rev_hfq is not None:
            cash += pos["shares"] * rev_hfq * 0.99955
        if cr is not None:
            cash_raw += pos["shares_raw"] * cr * 0.99955
            cash_idle += pos["shares_real"] * cr * 0.99955

    # 遍历每天
    trading_year = None
    years_done = set()
    rebalance_records = []   # 每次调仓后的持仓快照（暴露诊断/归因模块输入，不影响交易逻辑）

    # 加载分红明细（现金趴账模型用）
    divs = get_dividends()

    for i, td in enumerate(trade_dates):
        year = td[:4]

        # 判断是否是今年的第一个交易日（调仓日）
        is_rebalance_day = (year in yearly_first and td == yearly_first[year])

        # 获取当日所有持仓股票的收盘价（双轨：hfq 后复权含分红 / raw 原始价不含分红）
        # 双轨各自满仓：hfq 轨道用 shares(hfq 价定股)估值，raw 轨道用 shares_raw(raw 价定股)估值。
        # 两账本独立、各自充分投资 → 口径严格可比，无「闲置现金幻觉」。
        total_value = cash
        total_value_raw = cash_raw
        total_value_real = cash_idle     # 真实趴账轨道：持仓按原始价估值 + 闲置现金(含趴账分红)
        for code, pos in positions.items():
            # 后复权价（含分红再投）—— 对照口径1（对年度调仓高估）
            # 【逐持仓买入日因子归一化】hfq 净值 = raw × f(今)/f(买入)，
            # 使每年严格 hfq ≥ raw（分红只增不减），且消除 pre-2015→2015 的
            # 虚假台阶（买入用 f=1、估值用 f=27 造成的 28× 跳变）。
            ch = get_hfq_price(code, td, "close")
            if ch is None:
                ch = pos.get("last_hfq")   # 退市/数据缺口兜底：用最后已知价(已归一化)，避免 NAV 静默归零
            else:
                bf = pos.get("buy_factor") or 1.0
                ch = ch / bf                # 全局后复权价 ÷ 买入日因子 = 归一到买入日的后复权价
            if ch is not None:
                pos["last_hfq"] = ch
                total_value += pos["shares"] * ch
            # 原始价（不含分红）—— 对照口径2（漏计股息，偏低）
            cr = get_stock_price(code, td, "close")
            if cr is None:
                cr = pos.get("last_raw")
            if cr is not None:
                pos["last_raw"] = cr
                total_value_raw += pos["shares_raw"] * cr
                total_value_real += pos["shares_real"] * cr
            # 分红趴账模型：在除权日把每股分红计入闲置现金（不当日再投，留至下次调仓）。
            # 仅计入买入日之后发生的除权（避免把建仓前已宣布的分红算作收益）。
            # 这是年度调仓策略的真实分红处理：现金在账上闲置约 0~11 个月，年化收益≈0。
            for exd, dv in divs.get(code, []):
                if exd == td and exd > pos.get("buy_date", ""):
                    cash_idle += pos["shares_real"] * dv
        daily_vals.append({"date": td, "value": total_value,
                           "value_raw": total_value_raw, "value_real": total_value_real})

        # === 年内风控：移动止损 + 组合回撤减仓（用今日收盘价触发，不影响调仓逻辑）===
        if (trail_stop > 0 or dd_de_risk > 0) and positions:
            # 更新每个持仓峰值(以原始价=投资者看到的市价)
            for code, pos in positions.items():
                cr = pos.get("last_raw")
                if cr is not None and (pos.get("peak_raw") is None or cr > pos["peak_raw"]):
                    pos["peak_raw"] = cr
            # 1) 移动止损：单持仓从峰值回撤超 trail_stop% → 清仓
            if trail_stop > 0:
                for code in list(positions.keys()):
                    pos = positions[code]
                    cr, peak = pos.get("last_raw"), pos.get("peak_raw")
                    if cr is not None and peak is not None and cr <= peak * (1 - trail_stop / 100.0):
                        sell_at_close(code, td)
                        n_stop += 1
                        print(f"  [移动止损] {code} 市价 {cr:.2f} ≤ 峰值 {peak:.2f}×(1-{trail_stop:.0f}%)，清仓")
            # 2) 组合回撤减仓：组合从峰值回撤超 dd_de_risk% → 全部清仓至现金
            if dd_de_risk > 0 and not de_risked:
                nav_now = port_nav_raw()
                if nav_now <= peak_nav * (1 - dd_de_risk / 100.0):
                    for code in list(positions.keys()):
                        sell_at_close(code, td)
                    de_risked = True
                    n_dd += 1
                    print(f"  [回撤减仓] 组合 {nav_now:.0f} ≤ 峰值 {peak_nav:.0f}×(1-{dd_de_risk:.0f}%)，全部清仓至现金")
            # 更新组合峰值(用当日最新 NAV)
            peak_nav = max(peak_nav, port_nav_raw())

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
            selected = do_select(prev_td)
            if selected is None or len(selected) == 0:
                print(f"  [WARN] {year}年选股失败，保持现有持仓")
                continue

            new_codes = selected["ts_code"].tolist()
            # 打印本年入选清单（带名称），便于与另一子策略逐年对照
            sel_names = "、".join(f"{c}({get_stock_name(c)})" for c in new_codes)
            print(f"  本年入选 {len(new_codes)}/{top_n} 只: {sel_names}")
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
                    # 归一化卖出价 = 全局后复权开盘价 ÷ 买入日因子（与每日估值同一口径）
                    if open_price is None:
                        rev_hfq = pos.get("last_hfq")  # 退市/缺口兜底(已归一化)
                    else:
                        bf = pos.get("buy_factor") or 1.0
                        rev_hfq = open_price / bf
                    # raw 轨道独立现金：用原始价记账(不含分红)
                    open_price_raw = get_stock_price(code, td, "open")
                    if open_price_raw is None:
                        open_price_raw = open_price if open_price else pos.get("last_raw")
                    if rev_hfq:
                        revenue = pos["shares"] * rev_hfq * 0.99955  # 扣手续费
                        cash += revenue
                        if open_price_raw:
                            revenue_raw = pos["shares_raw"] * open_price_raw * 0.99955
                            cash_raw += revenue_raw
                            revenue_real = pos["shares_real"] * open_price_raw * 0.99955
                            cash_idle += revenue_real    # 趴账账本：卖出回收现金(含趴账分红)

            # 等权重买入新股（双轨各自满仓：hfq 轨道按 hfq 价定股数，
            # raw 轨道按 raw 价定股数，两账本独立满仓 → 口径可比，无闲置现金幻觉）
            old_in_new = old_codes & new_code_set
            new_to_buy = [c for c in new_codes if c not in old_codes]

            # 等分资金（预留手续费空间）；三账本各自按比例
            remaining_slots = top_n - len(old_in_new)
            cash_per_stock = cash * 0.98 / remaining_slots if remaining_slots > 0 else 0
            cash_per_stock_raw = cash_raw * 0.98 / remaining_slots if remaining_slots > 0 else 0
            # 真实趴账账本：用「含趴账分红的闲置现金」定股(分红在调仓日再投)
            cash_per_real = cash_idle * 0.98 / remaining_slots if remaining_slots > 0 else 0

            bought = 0
            skipped = []
            for code in new_to_buy:
                open_price = get_hfq_price(code, td, "open")   # 全局后复权开盘价 = raw_open × f(买入)
                if open_price is None or open_price <= 0:
                    skipped.append(code)
                    continue
                # 买入日复权因子（前向填充，缺则后向填充首个可用因子）
                bf = get_adj_factor(code, td)
                if bf is None or bf == 0:
                    bf = 1.0
                # 归一化买入价 = 全局后复权价 ÷ 买入日因子 = raw_open（与每日估值同一口径）
                norm_open = open_price / bf
                # raw 开盘价(原始价)用于给 last_raw 兜底种值，必须与 hfq 区分，
                # 否则 raw 轨道会拿后复权价当原始价估值 → 年度 raw 收益错乱。
                open_raw = get_stock_price(code, td, "open")
                if open_raw is None or open_raw <= 0:
                    open_raw = norm_open  # 极端兜底：raw 取不到则用归一化价(罕见)
                # hfq 轨道股数（按归一化买入价满仓 → 与现金投入严格一致）
                max_shares = int(cash_per_stock / norm_open / 100) * 100
                # raw 轨道股数（按 raw 价满仓，两账本独立，各自充分投资）
                max_shares_raw = int(cash_per_stock_raw / open_raw / 100) * 100
                # 真实趴账轨道股数（按 raw 价满仓，用含分红的闲置现金 cash_idle 定股）
                max_shares_real = int(cash_per_real / open_raw / 100) * 100
                if max_shares <= 0 or max_shares_raw <= 0 or max_shares_real <= 0:
                    skipped.append(code)
                    continue
                cost = max_shares * norm_open * 1.0002  # +手续费(hfq 账本, 归一化口径)
                cost_raw = max_shares_raw * open_raw * 1.0002  # +手续费(raw 账本)
                cost_real = max_shares_real * open_raw * 1.0002  # +手续费(真实趴账账本)
                if cost > cash or cost_raw > cash_raw or cost_real > cash_idle:
                    skipped.append(code)
                    continue
                positions[code] = {"shares": max_shares, "shares_raw": max_shares_raw,
                                    "shares_real": max_shares_real,
                                    "buy_price": norm_open,
                                    "last_hfq": norm_open, "last_raw": open_raw,
                                    "peak_raw": open_raw,
                                    "buy_factor": bf, "buy_date": td}
                cash -= cost
                cash_raw -= cost_raw
                cash_idle -= cost_real      # 趴账账本：用含分红闲置现金买入(分红在调仓日再投)
                bought += 1

            if skipped:
                print(f"  [跳过] {len(skipped)} 只: {', '.join(skipped)[:60]}...")

            print(f"  买入 {bought} 只, 持有 {len(old_in_new)} 只, 当前持仓 {len(positions)} 只")
            print(f"  现金: {cash:.2f}")

            # 风控状态复位：年度调仓后开启新一轮峰值跟踪与回撤减仓窗口
            peak_nav = port_nav_raw()
            de_risked = False

            # 记录本次调仓后的持仓快照（暴露诊断 / 归因模块输入；纯记录，不影响交易）
            for _c in sorted(positions.keys()):
                rebalance_records.append({"rebalance_date": td, "ts_code": _c})

    # 还原 VALUE_STRATEGY.top_n（do_select 内被临时放大过）
    VALUE_STRATEGY["top_n"] = top_n

    # ──── 回测结束：强制平仓 ────
    print(f"\n  {'─'*60}")
    if positions:
        total_cashout = 0
        last_holdings = list(positions.keys())  # 平仓前捕获末年持仓，供汇总展示
        for code, pos in list(positions.items()):
            close = get_hfq_price(code, trade_dates[-1], "close")
            if close is None:
                close = pos.get("last_hfq")  # 退市/数据缺口兜底(已归一化)，确保资本可回收
                if close is not None:
                    print(f"  [WARN] {code} 末日取不到后复权价，按最后已知价 {close:.2f} 平仓（疑似退市/数据缺口）")
            else:
                # 归一化平仓价 = 全局后复权收盘价 ÷ 买入日因子（与每日估值/卖出同一口径）
                bf = pos.get("buy_factor") or 1.0
                close = close / bf
            # raw 轨道独立平仓价(原始价)
            close_raw = get_stock_price(code, trade_dates[-1], "close")
            if close_raw is None:
                close_raw = pos.get("last_raw")
            if close:
                # 卖出：扣佣金+印花税，同 sell() 逻辑
                revenue = pos["shares"] * close
                fee = max(revenue * 0.0002, 5.0)
                tax = revenue * stamp_duty_rate(trade_dates[-1])
                net_revenue = revenue - fee - tax
                cash += net_revenue
                total_cashout += net_revenue
                print(f"  平仓 {code}({pos['shares']}股) @ {close:.2f}, 净回收 {net_revenue:.2f}")
                if close_raw:
                    revenue_raw = pos["shares_raw"] * close_raw
                    fee_raw = max(revenue_raw * 0.0002, 5.0)
                    tax_raw = revenue_raw * stamp_duty_rate(trade_dates[-1])
                    net_revenue_raw = revenue_raw - fee_raw - tax_raw
                    cash_raw += net_revenue_raw
                    # 真实趴账账本：末年平仓回收(用真实轨道股数)
                    revenue_real = pos["shares_real"] * close_raw
                    fee_real = max(revenue_real * 0.0002, 5.0)
                    tax_real = revenue_real * stamp_duty_rate(trade_dates[-1])
                    net_revenue_real = revenue_real - fee_real - tax_real
                    cash_idle += net_revenue_real
        positions.clear()
        # 用平仓后的现金更新最后一天的 daily_value（hfq 与 raw 双轨独立更新）
        if daily_vals:
            daily_vals[-1]["value"] = cash
            daily_vals[-1]["value_raw"] = cash_raw
            daily_vals[-1]["value_real"] = cash_idle
        print(f"  平仓完成，最终现金: {cash:.2f}")

    # ──── 计算年度收益 ────
    print(f"\n\n{'='*70}")
    print(f"  📊 年度收益对比")
    print(f"{'='*70}")
    print(f"  [口径] 策略(真实) = 分红趴账模型：持仓按原始价估值、分红作现金闲置至下次调仓日再投(年度调仓真实总回报)")
    print(f"  [口径] 策略(hfq) = 后复权价(假设分红当日即再投，对年度调仓【高估】)")
    print(f"  [口径] 策略(raw) = 原始价(不含分红，漏计股息，【偏低】)")
    print(f"  [口径] 基准 = {benchmark_idx} 价格指数(不含分红)")
    print(f"  [提示] 带 * 年度为区间未结束的半年度，收益按实际持有期计")
    print(f"  {'年份':<9} {'策略(真实)':>10} {'策略(hfq)':>10} {'策略(raw)':>10} {'基准':>10} {'超额(真实)':>11} {'超额(hfq)':>10} {'持仓'}")
    print(f"  {'─'*80}")

    # 按年计算收益
    year_groups = {}
    for d in daily_vals:
        y = d["date"][:4]
        if y not in year_groups:
            year_groups[y] = {"first": d["value"], "first_raw": d["value_raw"],
                              "first_real": d["value_real"],
                              "first_date": d["date"], "n_days": 0}
        year_groups[y]["last"] = d["value"]
        year_groups[y]["last_raw"] = d["value_raw"]
        year_groups[y]["last_real"] = d["value_real"]
        year_groups[y]["last_date"] = d["date"]
        year_groups[y]["n_days"] += 1

    total_strategy_return = 0
    total_strategy_return_raw = 0
    total_strategy_return_real = 0
    for idx, year in enumerate(sorted(year_groups.keys())):
        yg = year_groups[year]
        if idx == 0:
            # 第一年：从年初到年底
            year_start = TOTAL_CAPITAL
            year_start_raw = TOTAL_CAPITAL
            year_start_real = TOTAL_CAPITAL
        else:
            year_start = year_groups[year]["first"]
            year_start_raw = year_groups[year]["first_raw"]
            year_start_real = year_groups[year]["first_real"]

        year_end = yg["last"]
        year_end_raw = yg["last_raw"]
        year_end_real = yg["last_real"]
        if year_start > 0:
            strategy_ret = (year_end / year_start - 1) * 100
        else:
            strategy_ret = 0
        if year_start_raw > 0:
            strategy_ret_raw = (year_end_raw / year_start_raw - 1) * 100
        else:
            strategy_ret_raw = 0
        if year_start_real > 0:
            strategy_ret_real = (year_end_real / year_start_real - 1) * 100
        else:
            strategy_ret_real = 0

        # 基准指数年度收益（价格指数，不含分红）
        benchmark_ret = 0
        b_start_idx = get_index_close(benchmark_idx, yg["first_date"])
        b_end_idx = get_index_close(benchmark_idx, yg["last_date"])
        if b_start_idx and b_end_idx and b_start_idx > 0:
            benchmark_ret = (b_end_idx / b_start_idx - 1) * 100

        excess = strategy_ret - benchmark_ret
        excess_raw = strategy_ret_raw - benchmark_ret
        excess_real = strategy_ret_real - benchmark_ret

        # 区间未结束的年度（如回测截止在年中）标注 *
        is_partial = yg.get("n_days", 0) < 200
        year_label = f"{year}*" if is_partial else year

        # 获取该年持仓股票简要信息（末年用平仓前捕获的持仓）
        if idx == len(year_groups) - 1:
            year_stocks = ", ".join(last_holdings[:5])
        else:
            year_stocks = "-"

        print(f"  {year_label:<9} {strategy_ret_real:>+9.2f}% {strategy_ret:>+9.2f}% {strategy_ret_raw:>+9.2f}% {benchmark_ret:>+9.2f}% {excess_real:>+10.2f}% {excess:>+9.2f}%  {year_stocks[:20]}")
        total_strategy_return = strategy_ret
        total_strategy_return_raw = strategy_ret_raw
        total_strategy_return_real = strategy_ret_real

    # 总收益（回测全程）
    final_value = daily_vals[-1]["value"]
    final_value_raw = daily_vals[-1]["value_raw"]
    final_value_real = daily_vals[-1]["value_real"]
    total_return = (final_value / TOTAL_CAPITAL - 1) * 100
    total_return_raw = (final_value_raw / TOTAL_CAPITAL - 1) * 100
    total_return_real = (final_value_real / TOTAL_CAPITAL - 1) * 100

    # 基准总收益
    first_date = daily_vals[0]["date"]
    last_date = daily_vals[-1]["date"]
    b_total_start = get_index_close(benchmark_idx, first_date)
    b_total_end = get_index_close(benchmark_idx, last_date)
    b_total_ret = (b_total_end / b_total_start - 1) * 100 if b_total_start and b_total_end else 0

    total_excess = total_return - b_total_ret
    total_excess_raw = total_return_raw - b_total_ret
    total_excess_real = total_return_real - b_total_ret

    print(f"  {'─'*80}")
    print(f"  {'全程':<8} {total_return_real:>+9.2f}% {total_return:>+9.2f}% {total_return_raw:>+9.2f}% {b_total_ret:>+9.2f}% {total_excess_real:>+10.2f}% {total_excess:>+9.2f}%")

    # 年化收益（真实趴账轨道为主口径）
    days = len(trade_dates)
    years_span = days / 252
    annual_return = ((final_value_real / TOTAL_CAPITAL) ** (1 / years_span) - 1) * 100 if years_span > 0 else 0

    # 风控指标：三轨道净值
    vals = np.array([d["value"] for d in daily_vals])
    vals_raw = np.array([d["value_raw"] for d in daily_vals])
    vals_real = np.array([d["value_real"] for d in daily_vals])

    # 最大回撤：raw(原始价·真实价格波动) 为主口径（后复权除息日不跌、曲线被分红填平，回撤偏低失真）；
    # hfq(后复权) 与 真实趴账(real) 各给一档作为参考。
    cum_raw = np.maximum.accumulate(vals_raw)
    dd_raw = (vals_raw - cum_raw) / cum_raw
    max_dd = float(np.min(dd_raw)) * 100            # 主口径：真实价格波动（raw）
    cum_hfq = np.maximum.accumulate(vals)
    dd_hfq = (vals - cum_hfq) / cum_hfq
    max_dd_hfq = float(np.min(dd_hfq)) * 100         # 参考：后复权（偏低）
    cum_real = np.maximum.accumulate(vals_real)
    dd_real = (vals_real - cum_real) / cum_real
    max_dd_real = float(np.min(dd_real)) * 100       # 参考：真实趴账轨道回撤

    # 夏普：真实趴账轨道为主
    rets_real = np.diff(vals_real) / vals_real[:-1]
    sharpe = (np.mean(rets_real) * 252 - 0.025) / (np.std(rets_real) * np.sqrt(252)) if len(rets_real) > 0 else 0

    print(f"\n{'='*70}")
    print(f"  📈 最终汇总")
    print(f"{'='*70}")
    profit_amount = final_value_real - TOTAL_CAPITAL
    print(f"  初始资金: {TOTAL_CAPITAL:>10,.2f}")
    print(f"  最终资产(真实趴账): {final_value_real:>10,.2f}")
    print(f"  总盈亏: {profit_amount:>+10,.2f} 元")
    print(f"  {'─'*66}")
    print(f"  ★ 总收益率(真实·分红趴账再投): {total_return_real:>+9.2f}%   ← 年度调仓真实总回报(主口径)")
    print(f"  总收益率(hfq·含分红当日再投·高估): {total_return:>+9.2f}%")
    print(f"  总收益率(raw·原始价·漏分红):     {total_return_raw:>+9.2f}%")
    print(f"  年化收益率(真实): {annual_return:>+9.2f}%")
    print(f"  {'─'*66}")
    print(f"  基准收益({benchmark_idx}价格指数·不含分红): {b_total_ret:>+9.2f}%")
    print(f"  超额收益(真实·分红趴账总回报-基准): {total_excess_real:>+9.2f}%")
    print(f"  超额收益(hfq·高估): {total_excess:>+9.2f}%")
    print(f"  超额收益(raw·纯选股α·漏分红): {total_excess_raw:>+9.2f}%")
    print(f"  分红趴账 vs 后复权溢价(hfq-真实=立即再投复利增益): {total_return - total_return_real:>+9.2f}%")
    print(f"  {'─'*66}")
    print(f"  最大回撤(raw·原始价·真实价格波动): {max_dd:>+9.2f}%   ← 主口径(后复权除息不跌致回撤偏低)")
    print(f"  最大回撤(真实·分红趴账): {max_dd_real:>+9.2f}%")
    print(f"  最大回撤(hfq·后复权·平滑偏低): {max_dd_hfq:>+9.2f}%")
    print(f"  夏普比率(真实): {sharpe:>9.4f}")
    print(f"  调仓次数: {len(years)} 次")

    # ── 现实折扣三件套（扣通胀 / 定投拖累 / 中断模拟）──
    disc = compute_reality_discounts(
        daily_vals, TOTAL_CAPITAL,
        interrupt_start=interrupt_start,
        interrupt_months=interrupt_months,
        interrupt_pct=interrupt_pct,
    )
    if "real_total_return" in disc:
        print(f"  ── 现实折扣（预期管理，不改收益计算）──")
        print(f"  扣通胀真实总收益：{disc['real_total_return']:+.2f}% ｜ 真实年化：{disc['real_annual_return']:+.2f}%")
    if "dca_drag_pct" in disc:
        print(f"  定投对比(DCA)：一次性建仓较分12月定投 {disc['dca_drag_pct']:+.2f}%"
              f"（正=一次性占优·负=定投占优）｜ 终值 一次性 {disc['dca_lump_final']:,.0f} / 定投 {disc['dca_dca_final']:,.0f}")
    if "interrupt_loss_pct" in disc:
        print(f"  中断模拟：{interrupt_start}起撤{interrupt_pct*100:.0f}%持有{interrupt_months}月，"
              f"终值损失 {disc['interrupt_loss_pct']:+.2f}%（终值 {disc['interrupt_final']:,.0f}）")

    print(f"{'='*70}")
    print(f"  [口径说明] 策略(真实)=分红趴账模型(持仓按原始价估值+分红作闲置现金至下次调仓日再投)，为年度调仓真实总回报；")
    print(f"  [口径说明] 策略(hfq)=后复权(假设分红当日即再投)，对年度调仓【高估】；策略(raw)=原始价(不含分红)【偏低】；基准={benchmark_idx}价格指数(不含分红)。")
    print(f"  [口径说明] 最大回撤主口径用 raw(原始价·真实价格波动)；hfq 因除息日不跌、曲线被分红填平，回撤系统性偏低，仅作参考。")
    print(f"{'='*70}\n")

    # 保存CSV
    csv_dir = "data/results/dogs_annual"
    os.makedirs(csv_dir, exist_ok=True)
    _cap_tag = f"_indcap{industry_cap}" if industry_cap else ""
    _rc_tag = ""
    if trail_stop > 0:
        _rc_tag += f"_trail{int(round(trail_stop))}"
    if dd_de_risk > 0:
        _rc_tag += f"_dd{int(round(dd_de_risk))}"
    if trail_stop > 0 or dd_de_risk > 0:
        print(f"  [年内风控] 移动止损触发 {n_stop} 次, 组合回撤减仓触发 {n_dd} 次"
              f" (trail={trail_stop:.0f}%, dd={dd_de_risk:.0f}%)")
    csv_path = f"{csv_dir}/backtest_{start_date}_{end_date}{_cap_tag}{_rc_tag}.csv"
    pd.DataFrame(daily_vals).to_csv(csv_path, index=False)
    print(f"  日净值已保存 → {csv_path}\n")

    # 持仓快照（调仓后）：暴露诊断 / 归因模块输入
    if rebalance_records:
        hold_csv = f"{csv_dir}/holdings_{start_date}_{end_date}{_cap_tag}{_rc_tag}.csv"
        pd.DataFrame(rebalance_records).to_csv(hold_csv, index=False)
        n_reb = len(set(r["rebalance_date"] for r in rebalance_records))
        print(f"  持仓快照已保存 → {hold_csv} ({n_reb} 次调仓, {len(rebalance_records)} 条记录)\n")

    return {
        "total_return": total_return_real,
        "total_return_hfq": total_return,
        "total_return_raw": total_return_raw,
        "annual_return": annual_return,
        "max_drawdown": max_dd,
        "max_drawdown_hfq": max_dd_hfq,
        "max_drawdown_real": max_dd_real,
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
    parser.add_argument("--strategy", type=str, default="dogs",
                        help="子策略: dogs(狗股) / value(价值选股) / magic(神奇公式) / ep(EP行业中性) / ep_obv(EP+OBV吸筹过滤)")
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--industry-cap", type=int, default=0,
                        help="单行业最多持有N只(0=不限制)。A/B实验用，如 --industry-cap 2")
    parser.add_argument("--interrupt-start", type=str, default=None,
                        help="现实折扣-中断模拟：从 YYYYMM 起撤出部分资金（配合 --interrupt-months/--interrupt-pct）")
    parser.add_argument("--interrupt-months", type=int, default=0,
                        help="中断模拟持续月数（默认0=关闭）")
    parser.add_argument("--interrupt-pct", type=float, default=0.0,
                        help="中断模拟撤出比例(0~1，如 0.5=撤一半)，默认0")
    parser.add_argument("--stock-pool", type=str, default=None,
                        help="股票池 hs300/zz500/zz800/zz1000/all，显式指定可避免被 config.GLOBAL 漂移影响")
    parser.add_argument("--trail-stop", type=float, default=0.0,
                        help="移动止损%%：个股自持有期最高价回撤超过该比例即按当日收盘清仓(0=关闭)")
    parser.add_argument("--dd-de-risk", type=float, default=0.0,
                        help="回撤减仓%%：组合净值自峰值回撤超过该比例即全部清仓转现金，持有至下次年度调仓(0=关闭)")
    args = parser.parse_args()

    # ── 股票池显式覆盖（防 config.GLOBAL["stock_pool"] 被别的任务改掉导致静默漂移）──
    if args.stock_pool:
        SELECTION["stock_pool"] = args.stock_pool
        DOGS_OF_MARKET["stock_pool"] = args.stock_pool
        VALUE_STRATEGY["stock_pool"] = args.stock_pool

    run_backtest(args.start_date, args.end_date, top_n=args.top_n,
                 select_only=args.select_only, capital=args.capital,
                 strategy=args.strategy, industry_cap=args.industry_cap,
                 interrupt_start=args.interrupt_start,
                 interrupt_months=args.interrupt_months,
                 interrupt_pct=args.interrupt_pct,
                 trail_stop=args.trail_stop,
                 dd_de_risk=args.dd_de_risk)
