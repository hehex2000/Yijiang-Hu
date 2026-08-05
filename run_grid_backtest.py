"""
百分比网格交易回测 —— 不猜方向，靠波段收割差价

原理：
  价格每涨  grid_pct  →  卖出一份（锁定利润）
  价格每跌  grid_pct  →  买入一份（降低成本）
  市场波段本身就是利润来源，不需要预测涨跌。

设计立场（参考《网格交易策略详解与回测》BV1Yf4y177cA 提炼，不盲从）：
  · 网格中枢（核心价值）：以「滚动移动平均(MA)」为中枢，买/卖线每日挂在 中枢±gap 上，
    跟随市场下沉/回升而移动——单边下跌后反弹时卖线随中枢抬升被价格触及(可卖)，
    而非钉死在周期顶永远够不到；买线随中枢下移、把弹药沿整个下跌路径铺开而非山顶打光。
    仍保留「穿越式·永不冻结」特性。center_mode="fixed" 时退化为旧版(锚定首日收盘价)供对照。
  · 保留「宽基/风格指数 ETF 标的 + 50% 底仓起步 + 非对称浮亏不割」——契合文档"选宽基、留子弹"思想。
  · 提取文档两大风控缺口补强：① 总止损线（硬熔断，防深套虚无）；② 波动率关网（波动突升暂停、恢复自动重启）。
  · 不照搬：零仓起步（我们需底仓可收割）、强制清仓止损（宽基长期向上宜持有）、1% 过密档（摩擦成本高，仅作可选预设）。

参考：Rundle et al. (2019) MDPI Applied Sciences；B站 BV1Yf4y177cA 网格交易详解
"""

import sys
import os
import sqlite3
import datetime as _dtmod
import numpy as np
import pandas as pd

# ── 复用现有模块 ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_monthly_rebalance import (
    get_conn, get_price, get_open_price,
    INIT_CAPITAL, INDEX_DISPLAY_NAME,
    COMMISSION_RATE, COMMISSION_MIN, SLIPPAGE_RATE, compute_reality_discounts,
)
from run_backtest import calc_win_rate_from_trades
from run_etf_rotation import PremiumGate

# ETF显示名称（专供网格交易使用）
ETF_DISPLAY_NAME = {
    "510300.SH": "沪深300ETF",
    "510500.SH": "中证500ETF",
    "512100.SH": "中证1000ETF",
    "515800.SH": "中证800ETF",
    "512910.SH": "中证800ETF",
}

# ETF → 对应基准指数（用于网格回测基准对比展示，避免把指数价格回报误当ETF回报）
ETF_TO_INDEX = {
    "510300.SH": "000300.SH",
    "510500.SH": "000905.SH",
    "515800.SH": "000906.SH",
    "512910.SH": "000906.SH",
    "512100.SH": "000852.SH",
}


def _is_index_code(ts_code):
    """判断 ts_code 是否走指数行情（index_daily）。

    指数代码特征：
      - 上交所/中证系列：000xxx.SH / 931xxx.SH / 932xxx.SH（如 000001 上证指数、
        000688 科创50、000698 科创100、932000 中证2000）
      - 深交所系列：399xxx.SZ（如 399001 深证成指、399006 创业板指、399673 创业板50）
    ETF/个股代码（5xxxxx.SH、588xxx.SH、15xxxxx.SZ、6xxxxx.SH 等）均不匹配。
    """
    if ts_code.endswith(".SZ"):
        return ts_code[:3] == "399"
    if ts_code.endswith(".SH"):
        return ts_code[:3] in ("000", "931", "932")
    return False


def _load_etf_adjusted(conn, ts_code, start_date, end_date):
    """读取 ETF 日线并做前复权（以最新有复权因子的交易日为基准）。

    返回 (df, note)：
      - df 含 trade_date/open/high/low/close（前复权后，最新价=真实可交易价）
      - note 为打印提示；若 etf_adj_factor 无数据则退化为未复权价。

    修复点：etf_daily 存的是未复权价，遇到份额拆分会出现单日暴涨/暴跌的
    断点（如 512100 在 2022-09-05 的 +176%），直接当连续价差会制造虚假收益。
    用 Tushare fund_adj 的复权因子前复权即可消除断点。
    """
    df = pd.read_sql_query(
        "SELECT trade_date, open, high, low, close FROM etf_daily "
        "WHERE ts_code = ? AND trade_date >= ? AND trade_date <= ? ORDER BY trade_date",
        conn, params=(ts_code, start_date, end_date),
    )
    if df.empty:
        return df, ""

    adj = pd.read_sql_query(
        "SELECT trade_date, adj_factor FROM etf_adj_factor "
        "WHERE ts_code = ? AND trade_date >= ? AND trade_date <= ? ORDER BY trade_date",
        conn, params=(ts_code, start_date, end_date),
    )
    if adj is None or adj.empty:
        return df, "（无复权因子，使用未复权价）"

    adj["trade_date"] = adj["trade_date"].astype("int64")
    df["trade_date"] = df["trade_date"].astype("int64")
    df = df.merge(adj, on="trade_date", how="left")
    # 缺失交易日：先向后填（用首个有效因子填早期），再向前填（用末个有效因子填晚期）
    df["adj_factor"] = df["adj_factor"].bfill().ffill()
    # 基准：用回测区间内（已填充）最后一个交易日的因子，
    # 保证区间末价 = 真实可交易价（而非全表最新日，避免 end_date<因子最新日时被错误缩放）
    base_factor = float(df["adj_factor"].iloc[-1])
    if base_factor and base_factor > 0:
        for col in ("open", "high", "low", "close"):
            df[col] = df[col] * df["adj_factor"] / base_factor
    df = df.drop(columns=["adj_factor"])
    return df, "（已前复权：etf_adj_factor）"


def _top10_concentration(conn, index_code, end):
    """取该指数在 end(含)之前最近一个快照日前十大成分股权重之和(%)，量化"分散悄悄集中"风险。

    与 show_index_etf_changes.py 同一口径：weight 字段本身为百分比（5.008 => 5.008%）。
    返回百分比数值（如 62.3 表示前十大占 62.3%）；无 index_weight 数据返回 None。
    """
    row = conn.execute(
        "SELECT MAX(trade_date) FROM index_weight "
        "WHERE index_code = ? AND CAST(trade_date AS TEXT) <= ?",
        (index_code, end),
    ).fetchone()
    if not row or row[0] is None:
        return None
    td = row[0]
    weights = conn.execute(
        "SELECT weight FROM index_weight WHERE index_code = ? AND trade_date = ? "
        "ORDER BY weight DESC LIMIT 10",
        (index_code, td),
    ).fetchall()
    if not weights:
        return None
    return sum(float(r[0]) for r in weights)


# ── 网格专属费用模型 ──
# 网格策略的标的只可能是 ETF 或指数（见下方 ETFS 菜单），二者均免收印花税：
#   · ETF（场内交易型开放式指数基金）卖出不收印花税；
#   · 指数本身不可直接交易（回测以指数行情代理），同样无印花税。
# 仅 A 股个股卖出才收印花税（现千分之0.5，历史上曾为千1）。故此处固定 stamp=0，
# 与 run_etf_rotation.py 的「ETF 免印花税」口径保持一致，避免对 ETF 多扣费用。
def calc_fee_grid(buy_or_sell, price, shares):
    """网格专用交易费用：佣金 + 滑点，免印花税（ETF/指数）。"""
    amount = price * shares
    commission = max(amount * COMMISSION_RATE, COMMISSION_MIN)
    slippage = amount * SLIPPAGE_RATE
    return commission + slippage


# ── 默认参数 ──
GRID_PCT = 0.02          # 每格百分比 (2%)
PER_GRID_CASH = 5000     # 每格交易金额
CENTER_MODE = "fixed"    # 网格中枢模式：fixed=旧版固定价位网格(通用稳健·默认)；ma=滚动MA中枢(震荡/深跌市更优·需主动开启)
CENTER_MA_WINDOW = 60    # 中枢MA窗口(日)；仅 center_mode=ma 生效
MA_GRID_LEVELS = 60      # MA模式每侧生成的格线层数(覆盖 ±60 档，足够宽)
INIT_POSITION_PCT = 0.5  # 初始持仓比例（50%建仓）
POS_MIN_FRAC = 0.0       # 持仓下限（占初始底仓比例）：设为0=移除硬下限。
                           # 原因：原 0.3 下限会在大涨市中把仓位 draining 到下限后，
                           # 因 `units>pos_min` 守卫永久失效而导致网格冻结（2025-03 后一年多零成交）。
                           # 移除后纯网格在顶部卖到空仓是正常行为；强牛市请用 --trend-filter 保仓。
POS_MAX_FRAC = 4.0       # 持仓上限（占初始底仓比例）：保护现金，防止单边下跌全仓被套。
                          # 原为 2.0，但 2022-24 筑底回升中网格低位持续买入会填满 2×上限(28,600份)，
                          # 封顶后 `units<pos_max` 守卫永久阻断买入；叠加趋势过滤(牛市价在MA上不卖) →
                          # 双锁冻结、只能抱满仓(2023底后零成交)。提到 4× 让网格在牛市回调中仍能补仓、
                          # 不被份额上限过早封死。注意：仍是份额上限，价涨后是缩水的金额上限，属已知局限。


def generate_grid_levels(base_price, grid_pct, num_levels_up=20, num_levels_down=20):
    """
    生成百分比网格线

    从 base_price 向上下展开：
      向上: base × (1+pct)^1, base × (1+pct)^2, ...
      向下: base × (1-pct)^1, base × (1-pct)^2, ...

    Returns:
        levels_down: 买单价格线（从高到低）
        levels_up:   卖单价格线（从低到高）
    """
    levels_down = []
    for i in range(1, num_levels_down + 1):
        levels_down.append(base_price * (1 - grid_pct) ** i)

    levels_up = []
    for i in range(1, num_levels_up + 1):
        levels_up.append(base_price * (1 + grid_pct) ** i)

    return levels_down, levels_up


def run_grid_backtest(ts_code="000300.SH", start_date="20200102", end_date="20251231",
                      grid_pct=GRID_PCT, per_grid_cash=PER_GRID_CASH,
                      init_position_pct=INIT_POSITION_PCT, initial_capital=INIT_CAPITAL,
                      mode="symmetric", sell_pct=None,
                      trend_filter=False, ma_window=250,
                      stop_loss=None, vol_filter=False, vol_window=20, vol_mult=2.5,
                      compare_buyhold=False, caveat=False,
                      center_mode=CENTER_MODE, center_ma_window=CENTER_MA_WINDOW,
                      interrupt_start=None, interrupt_months=0, interrupt_pct=0.0,
                      premium_filter="qdii"):
    """
    百分比网格交易回测

    Args:
        ts_code:           标的代码（ETF或指数）
        start_date:        回测开始日期 YYYYMMDD
        end_date:          回测结束日期 YYYYMMDD
        grid_pct:          每格百分比 (0.02 = 2%)
        per_grid_cash:     每格交易金额（元）
        init_position_pct: 初始持仓比例
        initial_capital:   初始资金

    Returns:
        dict: 绩效指标
    """
    # 判断标的类型（指数走 index_daily，其余 ETF/个股走 etf_daily+复权）
    is_index = _is_index_code(ts_code)
    table = "index_daily" if is_index else "daily"
    lot_size = 1 if is_index else 100  # 指数最少1份，ETF/股票最少1手100股

    display = INDEX_DISPLAY_NAME.get(ts_code, ETF_DISPLAY_NAME.get(ts_code, ts_code))

    # ── 折溢价闸门（默认 qdii：跨境ETF≥8%溢价拦截；指数/国内ETF不拦）──
    gate = PremiumGate([{"code": ts_code, "name": display}], mode=premium_filter)
    if gate.enabled:
        print(f"  折溢价过滤：开启 [{premium_filter}]（跨境≥8%溢价拦截，指数/国内放行）")

    print("=" * 70)
    print(f"百分比网格交易回测")
    print("=" * 70)
    print(f"  标的：{ts_code} ({display})")
    print(f"  网格：每涨跌 {grid_pct*100:.0f}% 触发买卖")
    print(f"  每格金额：{per_grid_cash:,.0f} 元")
    print(f"  初始仓位：{init_position_pct*100:.0f}%")
    # 防止起始/结束日期填反导致「时间倒流」查不到数据
    if start_date > end_date:
        start_date, end_date = end_date, start_date
        print(f"  [提示] 起始日期晚于结束日期，已自动交换为 {start_date} ~ {end_date}")
    # 若起始日早于标的数据起点（如科创50/科创100/中证2000 上市较晚），
    # 自动顺延到数据起点，避免"数据不足"误报（bat/CLI 直接传 early start 时也生效）。
    try:
        _c = get_conn()
        _tbl = "index_daily" if is_index else "etf_daily"
        _mn = pd.read_sql_query(
            f"SELECT MIN(trade_date) m FROM {_tbl} WHERE ts_code=?", _c, params=(ts_code,)
        ).iloc[0, 0]
        _c.close()
        if _mn and str(_mn) > start_date:
            print(f"  [提示] 起始日 {start_date} 早于 {display} 数据起点 {_mn}，已顺延")
            start_date = str(_mn)
    except Exception:
        pass
    print(f"  回测区间：{start_date} ~ {end_date}")
    print(f"  佣金：万2.5（最低5元）| 印花税：ETF/指数免收 | 滑点：0.1%")
    if mode == "asymmetric":
        sp = sell_pct if sell_pct is not None else grid_pct * 2.5
        print(f"  模式：非对称（买 {grid_pct*100:.0f}% / 卖 {sp*100:.1f}%，锚定成本线·浮盈才卖）")
    else:
        print(f"  模式：对称（买=卖 {grid_pct*100:.0f}%，锚定前收）")
    if trend_filter:
        print(f"  趋势过滤：开启（站上 {ma_window} 日均线只持有不卖，抑制趋势踏空）")
    if stop_loss is not None:
        print(f"  总止损线：开启（净值峰回撤 ≤ {stop_loss*100:.0f}% 硬熔断·停止新建网格·仅持有）")
    if vol_filter:
        print(f"  波动率关网：开启（{vol_window}日年化波动 > 基准×{vol_mult:.1f} 暂停网格）")
    print()

    # ===== 1. 获取历史数据 =====
    conn = get_conn()
    if is_index:
        df = pd.read_sql_query(
            "SELECT trade_date, open, high, low, close FROM index_daily "
            "WHERE ts_code = ? AND trade_date >= ? AND trade_date <= ? ORDER BY trade_date",
            conn, params=(ts_code, start_date, end_date),
        )
        adj_note = ""
    else:
        df, adj_note = _load_etf_adjusted(conn, ts_code, start_date, end_date)

    # 价格清洗：部分指数/ETF 的首行（上市基准日）只有 close 有值、
    # open/high/low 为 NaN（如 科创50 20191231 基准行）。网格以 open 下单、
    # close 估值，NaN 会让整段净值变 NaN。用 close 回填缺失的 open/high/low，
    # 仍缺失关键价的行直接丢弃，保证后续计算干净。
    for _col in ("open", "high", "low"):
        df[_col] = df[_col].fillna(df["close"])
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

    # 基准明细（供第6节输出）：ETF裸价回报 + 对应指数价格回报
    bench_detail = {}
    if not is_index:
        try:
            _raw = pd.read_sql_query(
                "SELECT trade_date, close FROM etf_daily "
                "WHERE ts_code = ? AND trade_date >= ? AND trade_date <= ? ORDER BY trade_date",
                conn, params=(ts_code, start_date, end_date))
            if len(_raw) >= 2:
                _p0, _p1 = float(_raw.iloc[0]["close"]), float(_raw.iloc[-1]["close"])
                bench_detail["etf_price_ret"] = (_p1 / _p0 - 1) * 100.0
            _idx_code = ETF_TO_INDEX.get(ts_code)
            if _idx_code:
                _idx = pd.read_sql_query(
                    "SELECT trade_date, close FROM index_daily "
                    "WHERE ts_code = ? AND trade_date >= ? AND trade_date <= ? ORDER BY trade_date",
                    conn, params=(_idx_code, start_date, end_date))
                if len(_idx) >= 2:
                    _i0, _i1 = float(_idx.iloc[0]["close"]), float(_idx.iloc[-1]["close"])
                    bench_detail["idx_code"] = _idx_code
                    bench_detail["idx_price_ret"] = (_i1 / _i0 - 1) * 100.0
        except Exception:
            bench_detail = {}

    # 前十大权重集中度（"分散悄悄集中"风险量化，供 --caveat 硬提醒）。
    # ETF 标的需先映射回基准指数（ETF本身无 index_weight），无映射/无数据则为 None。
    conc_top10 = None
    try:
        _idx_code = ts_code if is_index else ETF_TO_INDEX.get(ts_code)
        if _idx_code:
            conc_top10 = _top10_concentration(conn, _idx_code, end_date)
    except Exception:
        conc_top10 = None
    conn.close()

    # ===== 1.5 趋势过滤用的 MA 序列（向前多取 ma_window 交易日作窗口）=====
    # 价格站上 MA(ma_window) 上方 → 只持有不卖（抑制上涨趋势中的踏空）。
    # 需向前多取历史才能在第 1 天就得到有效 MA；窗口未填满前 MA 为 NaN → 过滤不生效。
    ma_series = {}
    if trend_filter and ma_window and ma_window > 1:
        try:
            import datetime as _dt
            _sd = _dt.datetime.strptime(start_date, "%Y%m%d")
            _ext_start = (_sd - _dt.timedelta(days=int(ma_window * 2))).strftime("%Y%m%d")
            _c = get_conn()
            if is_index:
                _ext = pd.read_sql_query(
                    "SELECT trade_date, close FROM index_daily "
                    "WHERE ts_code = ? AND trade_date >= ? AND trade_date <= ? ORDER BY trade_date",
                    _c, params=(ts_code, _ext_start, end_date),
                )
            else:
                _ext, _ = _load_etf_adjusted(_c, ts_code, _ext_start, end_date)
            _c.close()
            if not _ext.empty:
                _ext["ma"] = _ext["close"].rolling(int(ma_window)).mean()
                ma_series = {int(d): float(m) for d, m in zip(_ext["trade_date"], _ext["ma"]) if pd.notna(m)}
        except Exception as e:
            print(f"  [WARN] 趋势过滤 MA 计算失败，已关闭过滤：{e}")
            ma_series = {}

    # ===== 1.6 波动率关网用的短期波动率序列 =====
    # 文档核心思想：用波动率作网格开关——波动平稳→开网，波动突升→关网(趋势将来临)。
    # 这里算 vol_window 日年化实现波动率，以其全样本中位数为基准，
    # 超过 vol_mult 倍视为"突升"→当日暂停网格（恢复后自动重启）。
    vol_high = {}
    if vol_filter:
        try:
            _ret = df["close"].pct_change()
            _vol = _ret.rolling(int(vol_window)).std() * np.sqrt(252)
            _base = _vol.median()
            if pd.notna(_base) and _base > 0:
                for d, v in zip(df["trade_date"], _vol):
                    di = int(d.item()) if hasattr(d, "item") else int(d)
                    vol_high[di] = (pd.notna(v) and v > _base * vol_mult)
            else:
                print(f"  [WARN] 波动率基准无效，已关闭波动率关网")
                vol_filter = False
        except Exception as e:
            print(f"  [WARN] 波动率关网计算失败，已关闭：{e}")
            vol_high = {}

    # ===== 1.55 中枢网格用的滚动中枢序列（MA）=====
    # 中枢 = 近期收盘价的滚动 MA；价格围绕中枢上下波动时，买/卖线挂在 中枢±gap。
    # 向前多取历史窗口使回测首日即可得到有效中枢；窗口未填满前中枢退化为 base_price(=固定模式)。
    center_series = {}
    if center_mode == "ma" and center_ma_window and center_ma_window > 1:
        try:
            _sd2 = _dtmod.datetime.strptime(start_date, "%Y%m%d")
            _ext_start2 = (_sd2 - _dtmod.timedelta(days=int(center_ma_window * 3))).strftime("%Y%m%d")
            _c2 = get_conn()
            if is_index:
                _ext2 = pd.read_sql_query(
                    "SELECT trade_date, close FROM index_daily "
                    "WHERE ts_code = ? AND trade_date >= ? AND trade_date <= ? ORDER BY trade_date",
                    _c2, params=(ts_code, _ext_start2, end_date))
            else:
                _ext2, _ = _load_etf_adjusted(_c2, ts_code, _ext_start2, end_date)
            _c2.close()
            if not _ext2.empty:
                _ext2["ma"] = _ext2["close"].rolling(int(center_ma_window)).mean()
                center_series = {int(d): float(m) for d, m in zip(_ext2["trade_date"], _ext2["ma"]) if pd.notna(m)}
        except Exception as e:
            print(f"  [WARN] 中枢MA计算失败，退化为固定价位网格：{e}")
            center_series = {}
            center_mode = "fixed"

    if len(df) < 2:
        print(f"⚠️ 数据不足（{len(df)}条），无法回测")
        return None

    print(f"  交易日数：{len(df)}")
    print(f"  价格范围：{df['close'].min():.2f} ~ {df['close'].max():.2f}")
    if adj_note:
        print(f"  {adj_note}")

    # ===== 2. 网格中枢（升级：以滚动 MA 为中枢的中枢网格）=====
    # 相较旧版「固定价位·无限网格」(线钉死在首日收盘价、永不移动)：
    #   中枢网格把买/卖线挂在一个【滚动移动平均】上——中枢随市场下沉/回升而移动，
    #   格线跟随中枢走。好处：单边下跌后反弹时，卖线会随中枢抬升而被价格触及(可卖出)，
    #   而不是被钉死在周期顶永远够不到；买线也随中枢下移、把弹药沿整个下跌路径铺开，
    #   而非在山顶区一次性打光。仍保留「穿越式·永不冻结」特性。
    #   center_mode="fixed" 时退化为旧版(锚定首日收盘价)，用于对照。
    base_price = float(df.iloc[0]['close'])

    if mode == "asymmetric":
        sp = sell_pct if sell_pct is not None else grid_pct * 2.5
        buy_gap = grid_pct
        sell_gap = sp
    else:
        buy_gap = sell_gap = grid_pct

    if center_mode == "ma":
        print(f"  网格模式：滚动MA中枢网格（中枢=MA{center_ma_window}·穿越式·永不冻结）")
        print(f"  网格间距：{buy_gap*100:.0f}%（买线随中枢下潜 / 卖线随中枢上浮）")
        print(f"  初始中枢：MA{center_ma_window}(首段)≈{base_price:.2f} ｜ 卖线随中枢抬升、买线随中枢下移")
    elif mode == "asymmetric":
        print(f"  网格间距：非对称 买{buy_gap*100:.0f}% / 卖{sell_gap*100:.1f}%（固定价位·无限网格）")
    else:
        print(f"  网格间距：{grid_pct*100:.0f}%（固定价位·无限网格）")
    if center_mode != "ma":
        print(f"  初始挂单：买 {base_price*(1-buy_gap):.2f} / 卖 {base_price*(1+sell_gap):.2f}")
    print()

    # ===== 3. 初始化仓位（按金额，股票取整手）=====
    cash = initial_capital * (1 - init_position_pct)
    position_amount = initial_capital * init_position_pct
    first_open = float(df.iloc[0]['open'])
    raw_units = position_amount / first_open if first_open > 0 else 0
    if lot_size > 1:
        units = int(raw_units / lot_size) * lot_size  # 股票取整手
    else:
        units = raw_units  # 指数取实际份数
    fee = calc_fee_grid('buy', first_open, units) if units > 0 else 0
    cash = initial_capital - units * first_open - fee
    print(f"  建仓：{units:,.0f}份 @ {first_open:.2f}，投入 {units*first_open:,.0f}，现金 {cash:,.0f}")

    # 持仓上下限：永不清仓（下限>0 保证上涨时总有货可卖），也不无限加仓（上限护现金）
    pos_min = units * POS_MIN_FRAC
    pos_max = units * POS_MAX_FRAC

    # 生成网格线
    # 固定模式：锚定首日收盘价的绝对价位线（永不移动，预计算一次）。
    # MA中枢模式：线在循环内按「当日中枢」实时生成（中枢随MA移动 → 线随中枢走）。
    if center_mode == "fixed":
        _N = 400
        buy_lines  = sorted([base_price * (1 - buy_gap) ** k for k in range(1, _N + 1)], reverse=True)  # 由近及远向下
        sell_lines = sorted([base_price * (1 + sell_gap) ** k for k in range(1, _N + 1)])               # 由近及远向上
        _precomp_lines = (buy_lines, sell_lines)
    else:
        _precomp_lines = None

    daily_vals = []
    trades = []
    prev_close = float(df.iloc[0]['close'])
    buy_count = 0
    sell_count = 0
    halted = False        # 总止损线是否已触发（触发后永久关网·仅持有）
    peak_value = None     # 净值峰值（用于计算回撤）

    # 记录初始建仓（用于胜率计算）
    trades.append({
        "date": int(df.iloc[0]['trade_date']), "action": "BUY",
        "price": first_open, "shares": units, "reason": "initial_position"
    })

    for _, row in df.iterrows():
        td = int(row['trade_date']) if hasattr(row['trade_date'], 'item') else int(row['trade_date'])
        op = row['open']
        hi = row['high']
        lo = row['low']
        cl = row['close']

        # ── 中枢网格：固定模式用预计算线；MA模式按当日中枢实时生成格线 ──
        if center_mode == "fixed":
            cur_buy_lines, cur_sell_lines = _precomp_lines
            cur_center = base_price
        else:
            cur_center = center_series.get(td, base_price)
            cur_buy_lines  = sorted([cur_center * (1 - buy_gap) ** k for k in range(1, MA_GRID_LEVELS + 1)], reverse=True)
            cur_sell_lines = sorted([cur_center * (1 + sell_gap) ** k for k in range(1, MA_GRID_LEVELS + 1)])

        # ── 趋势过滤：价格站上 MA(ma_window) 上方 → 只持有不卖（抑制趋势踏空）──
        allow_sell_trend = True
        if trend_filter and ma_series:
            _ma = ma_series.get(td)
            if _ma is not None and cl > _ma:
                allow_sell_trend = False

        # 非对称：仅当价格高于中枢（已处浮盈区）才允许收割，浮亏不割肉
        allow_sell = True
        if mode == "asymmetric":
            _ref = cur_center if center_mode == "ma" else base_price
            allow_sell = prev_close > _ref

        # ── 风控闸门：总止损线(硬熔断) + 波动率关网(日级) ──
        # 任一触发则当日仅持有、不新建网格交易；其余逻辑不变。
        grid_enabled = True
        if halted:
            grid_enabled = False
        elif vol_filter and vol_high.get(td, False):
            grid_enabled = False

        # ── 固定价位·无限网格核心：检测当日穿越的格线并交易 ──
        # 价格从 prev_close 出发，当日区间 [lo, hi] 可能穿越多条固定格线：
        #   下跌日(close<=prev_close)：价格跌破 [lo, prev_close) 内的每条买入线 → 逐格买入
        #   上涨日(close>prev_close) ：价格涨破 (prev_close, hi] 内的每条卖出线 → 逐格卖出
        # 按当日方向只处理一侧，避免单日 whipsaw 同时买卖刷单；每条线可被反复穿越。
        if grid_enabled and cl <= prev_close:
            for line in cur_buy_lines:
                if lo <= line < prev_close and units < pos_max:
                    buy_units = per_grid_cash / line if line > 0 else 0
                    if lot_size > 1:
                        buy_units = int(buy_units / lot_size) * lot_size
                    if buy_units > pos_max - units:
                        buy_units = int((pos_max - units) / lot_size) * lot_size if lot_size > 1 else (pos_max - units)
                    if buy_units > 0:
                        buy_cost = buy_units * line
                        buy_fee = calc_fee_grid('buy', line, buy_units)
                        if buy_cost + buy_fee > cash or (lot_size > 1 and buy_units < lot_size):
                            if lot_size > 1:
                                min_units = int(cash / (line * lot_size + calc_fee_grid('buy', line, lot_size))) * lot_size
                            else:
                                min_units = 0
                            if min_units >= lot_size:
                                buy_units = min_units
                                buy_cost = buy_units * line
                                buy_fee = calc_fee_grid('buy', line, buy_units)
                            else:
                                buy_units = 0
                    if buy_units > 0 and buy_cost + buy_fee <= cash:
                        # ── 折溢价闸门（默认 qdii）：高溢价时跳过本次网格买入 ──
                        do_buy = True
                        if gate.enabled:
                            allow, prem = gate.check(ts_code, td)
                            if not allow:
                                do_buy = False
                                if verbose:
                                    print(f"  🚫 折溢价拦截 {td} 格线{line:.2f}：{display} 溢价 {prem:+.2%} ≥ 阈值，跳过买入")
                        if do_buy:
                            cash -= buy_cost + buy_fee
                            units += buy_units
                            buy_count += 1
                            print(f"  📥 买入 {td} 格线{line:.2f}：{buy_units:,.0f}份 @ {line:.2f}，现金{cash:,.0f}")
                            trades.append({
                                "date": td, "action": "BUY", "price": line,
                                "shares": buy_units, "reason": f"grid_{line:.2f}"
                            })
        elif grid_enabled:
            if allow_sell and allow_sell_trend and units > pos_min:
                for line in cur_sell_lines:
                    if prev_close < line <= hi and units > pos_min:
                        sell_units = per_grid_cash / line
                        if lot_size > 1:
                            sell_units = int(sell_units / lot_size) * lot_size
                        if sell_units > units - pos_min:
                            sell_units = int((units - pos_min) / lot_size) * lot_size if lot_size > 1 else (units - pos_min)
                        if sell_units > 0:
                            proceeds = sell_units * line
                            fee = calc_fee_grid('sell', line, sell_units)
                            cash += proceeds - fee
                            units -= sell_units
                            sell_count += 1
                            print(f"  📤 卖出 {td} 格线{line:.2f}：{sell_units:,.0f}份 @ {line:.2f}，现金{cash:,.0f}")
                            trades.append({
                                "date": td, "action": "SELL", "price": line,
                                "shares": sell_units, "reason": f"grid_{line:.2f}"
                            })

        # ── 每日净值记录 ──
        tv = cash + units * cl
        daily_vals.append({"date": td, "value": tv, "units": units, "cash": cash})

        # ── 风控：总止损线判定（基于当日净值，影响后续交易日）──
        # 文档"解决方案B：人为设置总止损线"。我们落地为【硬熔断·停止新建网格、仅持有】，
        # 而非文档说的"强制清仓"——宽基长期向上，底部强平反而最坏；止的是"继续加码的暴露"。
        if stop_loss is not None:
            if peak_value is None or tv > peak_value:
                peak_value = tv
            if not halted:
                dd = (tv - peak_value) / peak_value if peak_value > 0 else 0.0
                if dd <= stop_loss:
                    halted = True
                    print(f"  ⛔ 触发总止损线({stop_loss*100:.0f}%) {td}：净值峰回撤 {dd*100:.1f}%，停止新建网格（仅持有）")

        prev_close = cl

    # ===== 4. 平仓结算 =====
    if units > 0:
        last_close = float(df.iloc[-1]['close'])
        proceeds = units * last_close
        fee = calc_fee_grid('sell', last_close, units)
        cash += proceeds - fee
        print(f"\n  平仓：{units:,.0f}份 @ {last_close:.2f}，现金 {cash:,.0f}")
        trades.append({
            "date": df.iloc[-1]['trade_date'], "action": "SELL",
            "price": last_close, "shares": units, "reason": "backtest_end"
        })

    # ===== 5. 绩效计算 =====
    final_value = cash
    total_return = (final_value / initial_capital - 1) * 100
    days = len(df)
    years = days / 252
    annual_return = ((final_value / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0

    # 胜率只统计「真实网格交易」：剔除初始建仓与回测结束强制平仓两笔边界交易，
    # 否则会把"底仓被动持有到到期"算成必胜，虚高胜率。
    grid_trades = [t for t in trades if t.get("reason") not in ("initial_position", "backtest_end")]
    win_rate, win_cnt, total_closed = calc_win_rate_from_trades(grid_trades)

    vals = np.array([d["value"] for d in daily_vals])
    cummax = np.maximum.accumulate(vals)
    safe_cummax = np.where(cummax == 0, 1, np.array(cummax, dtype=float))
    drawdowns = (vals - cummax) / safe_cummax
    max_dd = float(np.min(drawdowns)) * 100

    rets = np.diff(vals) / np.where(vals[:-1] == 0, 1, vals[:-1])
    if len(rets) > 1 and np.std(rets) > 0:
        sharpe = (np.mean(rets) * 252 - 0.025) / (np.std(rets) * np.sqrt(252))
    else:
        sharpe = 0.0

    # ===== 6. 基准对比 =====
    idx_return = (float(df.iloc[-1]['close']) / float(df.iloc[0]['close']) - 1) * 100

    # 网格 vs 买入持有 对照（文章《长期持有ETF的真相》方法论量化）：
    # 同一笔钱、首日满仓买入并躺平到结束（不交易、不风控），与主动网格对照。
    # 用与网格完全相同的价格口径（指数=裸价/ETF=前复权总回报），保证苹果对苹果。
    bh_return = None
    bh_regime = ""
    if compare_buyhold:
        _bh_first = float(df.iloc[0]['open'])
        _bh_last = float(df.iloc[-1]['close'])
        _bh_units = int(initial_capital / (_bh_first * lot_size)) * lot_size if lot_size > 0 else 0
        _bh_left = initial_capital - _bh_units * _bh_first
        _bh_final = _bh_units * _bh_last + _bh_left
        bh_return = (_bh_final / initial_capital - 1) * 100 if initial_capital > 0 else 0.0
        # 修正：0 笔卖出时不能声称"网格收割占优"——差异来自初始仓位暴露，非波段收割
        if total_return > bh_return:
            if sell_count == 0:
                bh_regime = "网格未触发卖出（无波段收割）—— 差异来自仓位暴露，非网格策略贡献"
            else:
                bh_regime = "震荡/波段市 —— 网格收割占优"
        else:
            bh_regime = "单边趋势市 —— 买入持有占优"

    # ===== 7. 输出 =====
    print(f"\n{'=' * 70}")
    print(f"  网格交易回测结果")
    print(f"{'=' * 70}")
    profit_amount = final_value - initial_capital
    print(f"  初始资金：{initial_capital:,.0f}")
    print(f"  最终资产：{final_value:,.0f}")
    print(f"  总盈亏：{profit_amount:+,.0f} 元")
    print(f"  总收益率：{total_return:+.2f}%")
    print(f"  年化收益率：{annual_return:+.2f}%")
    print(f"  最大回撤：{max_dd:.2f}%")
    print(f"  夏普比率：{sharpe:.2f}")
    print(f"  网格交易次数：{buy_count + sell_count}（买{buy_count}次 / 卖{sell_count}次）")
    print(f"  胜率：{win_rate:.1f}%（{win_cnt}胜 / {total_closed}笔平仓）")
    print(f"  ── 基准对比 ──")
    if is_index:
        print(f"  {display}（指数价格回报）：{idx_return:+.2f}%")
        print(f"  超额收益（策略 vs 指数）：{total_return - idx_return:+.2f}%")
    else:
        print(f"  {display} 买入持有（总回报/前复权）：{idx_return:+.2f}%")
        if bench_detail:
            _pr = bench_detail.get("etf_price_ret", 0.0)
            print(f"      ├─ 价格变动（裸价）：{_pr:+.2f}%")
            print(f"      └─ 股息贡献：{idx_return - _pr:+.2f}%")
            _ic = bench_detail.get("idx_code")
            _ir = bench_detail.get("idx_price_ret")
            if _ic and _ir is not None:
                _ic_name = INDEX_DISPLAY_NAME.get(_ic, _ic)
                print(f"  基准指数 {_ic_name}({_ic}) 价格口径：{_ir:+.2f}%")
        print(f"  超额收益（策略 vs ETF总回报）：{total_return - idx_return:+.2f}%")

    # ── 现实折扣三件套（扣通胀 / 定投拖累 / 中断模拟）──
    disc = compute_reality_discounts(
        daily_vals, initial_capital,
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

    # ===== 7b. 网格 vs 买入持有 对照（可选开关）=====
    if compare_buyhold and bh_return is not None:
        print(f"  ── 网格 vs 买入持有（文章方法论对照）──")
        print(f"  买入持有（满仓躺平·{display}）：{bh_return:+.2f}%")
        print(f"  网格策略：{total_return:+.2f}%")
        print(f"  差额（网格 − 持有）：{total_return - bh_return:+.2f}%")
        print(f"  市场形态判定：{bh_regime}")
        if sell_count == 0:
            print(f"    · 注意：本段网格 0 笔卖出，所谓优于持有来自初始仓位较低(少亏)，非波段收割贡献。")
        else:
            print(f"    · 长横盘/震荡市网格靠收割占优；单边牛市持有占优（文章：日经30年/纳指13年横盘最伤买入持有）")

    # ===== 7c. 情景提示（可选开关）=====
    if caveat:
        print(f"\n  ── 情景提示（长期持有ETF文章的边界提醒）──")
        print(f"  · 市场级幸存者偏差：本回测仅覆盖 A 股单市场，未含其他市场崩盘样本（文章：美国百年10%≠全球）")
        print(f"  · 未来收益下行：历史高收益依赖过去市场环境，机构预测未来权益年化或降至 3-6%，复利差异巨大")
        # 集中度：用前十大权重占比数据支撑，>40% 升级为硬提醒，无数据则保留定性提示
        if conc_top10 is not None:
            if conc_top10 > 40:
                print(f"  · 集中度预警（数据支撑）：该指数前十大占比 {conc_top10:.1f}%，分散度偏低（被动'分散'悄悄集中，少数巨头决定走势）")
            else:
                print(f"  · 集中度（数据支撑）：该指数前十大占比 {conc_top10:.1f}%，分散度尚可（宽基效应仍在）")
        else:
            print(f"  · 集中度认知：若持有 科创50/上证50 等，权重高度集中于少数巨头，分散≠真分散（本标的无 index_weight 数据，无法量化）")

    # 保存
    csv_dir = "data/results/grid_backtest"
    os.makedirs(csv_dir, exist_ok=True)
    if mode == "asymmetric":
        sp = sell_pct if sell_pct is not None else grid_pct * 2.5
        mode_tag = f"asym_b{grid_pct*100:.0f}_s{sp*100:.0f}"
    else:
        mode_tag = f"sym_{grid_pct*100:.0f}"
    if center_mode == "ma":
        mode_tag += f"_ma{center_ma_window}"
    elif center_mode == "fixed":
        mode_tag += "_fixed"
    if trend_filter:
        mode_tag += f"_tf{ma_window}"
    csv_path = f"{csv_dir}/grid_{ts_code.replace('.','_')}_{mode_tag}_{start_date}_{end_date}.csv"
    pd.DataFrame(daily_vals).to_csv(csv_path, index=False)
    print(f"\n  结果已保存：{csv_path}")

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "trades": buy_count + sell_count,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "idx_return": idx_return,
        "buyhold_return": bh_return,
        "regime": bh_regime,
        "daily_values": daily_vals,
    }


# ══════════════════════════════════════════
#  入口
# ══════════════════════════════════════════

def _interactive_menu():
    """交互式网格策略菜单：选标的 + 选策略。

    取代原「1%/2%/3% 间距」菜单，固定提供 3 个策略预设：
      [1] 2% 对称网格      （保留原标准）
      [2] 4% 对称网格      （默认·推荐：宽间距减少趋势踏空）
      [3] 非对称 2/8%      （锚定成本线·浮亏不割·浮盈宽卖）
    """
    ETFS = {
        "1": ("510300.SH", "沪深300 ETF"),
        "2": ("510500.SH", "中证500 ETF"),
        "3": ("515800.SH", "中证800 ETF"),
        # ── 以下为宽基/风格指数（走 index_daily 干净行情，无需复权）──
        "4":  ("000001.SH", "上证指数"),
        "5":  ("399001.SZ", "深证成指"),
        "6":  ("000016.SH", "上证50"),
        "7":  ("000852.SH", "中证1000"),
        "8":  ("000688.SH", "科创50"),
        "9":  ("000698.SH", "科创100"),
        "10": ("399006.SZ", "创业板指"),
        "11": ("399673.SZ", "创业板50"),
        "12": ("932000.SH", "中证2000"),
    }
    # 3 个网格策略预设（取代原 1%/2%/3% 间距菜单）
    STRATEGIES = {
        "1": {"name": "2% 对称网格", "grid_pct": 0.02, "mode": "symmetric", "sell_pct": None,
              "desc": "窄间距·高频收割（标准）"},
        "2": {"name": "4% 对称网格", "grid_pct": 0.04, "mode": "symmetric", "sell_pct": None,
              "desc": "宽间距·低频·减少趋势踏空（推荐）"},
        "3": {"name": "1% 密网格", "grid_pct": 0.01, "mode": "symmetric", "sell_pct": None,
              "desc": "极密·高频·吃满A股波动（文档:档越密收益越高;摩擦成本也高）"},
        "4": {"name": "3% 中网格", "grid_pct": 0.03, "mode": "symmetric", "sell_pct": None,
              "desc": "中频·平衡收割与踏空"},
        "5": {"name": "非对称 2/8%", "grid_pct": 0.02, "mode": "asymmetric", "sell_pct": 0.08,
              "desc": "锚定成本线·浮亏不割·浮盈宽卖"},
    }
    DEFAULT_STRAT = "2"  # 默认 = 4% 对称
    DEFAULT_START, DEFAULT_END = "20180102", "20260703"  # 历史全区间

    print("\n" + "=" * 60)
    print("  网格交易策略（百分比网格·波段收割）")
    print("=" * 60)

    while True:
        print("\n  请选择标的:")
        print("  ----------------")
        for k, (code, name) in ETFS.items():
            print(f"  [{k}] {name} ({code})")
        print("  [0] 退出")
        choice = input("\n请选择 (1-12, 0退出): ").strip()
        if choice == "0":
            print("已退出。")
            return
        if choice not in ETFS:
            print("⚠ 无效选择，请重试。")
            continue
        ts_code, ts_name = ETFS[choice]

        print(f"\n  请设置网格策略（{ts_name}）:")
        print("  ----------------")
        for k, s in STRATEGIES.items():
            tag = " ◀默认" if k == DEFAULT_STRAT else ""
            print(f"  [{k}] {s['name']} —— {s['desc']}{tag}")
        print("  [0] 返回")
        sc = input(f"\n请选择 (1-5, 0返回, 回车={DEFAULT_STRAT}默认): ").strip()
        if sc == "0":
            continue
        if sc == "":
            sc = DEFAULT_STRAT
        if sc not in STRATEGIES:
            print("⚠ 无效选择，请重试。")
            continue
        strat = STRATEGIES[sc]

        # 趋势过滤开关（站上 250 日均线只持有不卖，缩小与买入持有的差距）
        print(f"\n  趋势过滤（站上 250 日均线只持有不卖）:")
        print("  [1] 关闭（纯网格）")
        print("  [2] 开启")
        tf = input("\n请选择 (1-2, 回车=关闭): ").strip()
        trend_on = (tf == "2")
        ma_win = 250
        if trend_on:
            mw = input("  均线周期（默认250，回车=250）: ").strip()
            if mw:
                try:
                    ma_win = int(mw)
                except ValueError:
                    ma_win = 250

        # 总止损线（硬熔断：净值峰回撤超阈值→停止新建网格·仅持有）
        print(f"\n  总止损线（净值峰回撤超阈值即停止新建网格·仅持有）:")
        print("  [1] 关闭")
        print("  [2] 开启（-30%）")
        sl = input("\n请选择 (1-2, 回车=关闭): ").strip()
        stop_on = (sl == "2")
        stop_loss = -0.30 if stop_on else None

        # 波动率关网（短期波动突升→暂停网格·恢复自动重启）
        print(f"\n  波动率关网（短期波动突升暂停网格·恢复自动重启）:")
        print("  [1] 关闭")
        print("  [2] 开启（20日年化波动>基准×2.5）")
        vf = input("\n请选择 (1-2, 回车=关闭): ").strip()
        vol_on = (vf == "2")

        # 文章《长期持有ETF的真相》借鉴开关（默认关）
        bh = input("\n网格 vs 买入持有 对照 (1-2, 回车=关闭): ").strip()
        bh_on = (bh == "2")
        cv = input("\n情景提示[幸存者偏差/未来收益下行/集中度] (1-2, 回车=关闭): ").strip()
        cv_on = (cv == "2")

        tag = f"{strat['name']}" + (f" + 趋势过滤(MA{ma_win})" if trend_on else "")
        if stop_on:
            tag += " + 止损线"
        if vol_on:
            tag += " + 波动率关网"
        if bh_on:
            tag += " + 对照持有"
        if cv_on:
            tag += " + 提示"

        # 起始日自动校正：若默认起始早于标的数据起点（如科创50/科创100/中证2000
        # 上市较晚），用数据起点作为回测开始，避免"数据不足"误报。
        eff_start = DEFAULT_START
        try:
            _c = get_conn()
            _idx = _is_index_code(ts_code)
            _tbl = "index_daily" if _idx else "etf_daily"
            _mn = pd.read_sql_query(
                f"SELECT MIN(trade_date) m FROM {_tbl} WHERE ts_code=?",
                _c, params=(ts_code,),
            ).iloc[0, 0]
            _c.close()
            if _mn and str(_mn) > DEFAULT_START:
                eff_start = str(_mn)
        except Exception:
            pass
        _rng = eff_start if eff_start == DEFAULT_START else f"{eff_start}(数据起点, 默认{DEFAULT_START}过早)"
        print(f"\n>>> 运行：{ts_name} / {tag} / 区间 {_rng}~{DEFAULT_END}\n")
        run_grid_backtest(
            ts_code=ts_code,
            start_date=eff_start,
            end_date=DEFAULT_END,
            grid_pct=strat["grid_pct"],
            mode=strat["mode"],
            sell_pct=strat["sell_pct"],
            trend_filter=trend_on,
            ma_window=ma_win,
            stop_loss=stop_loss,
            vol_filter=vol_on,
            vol_window=20,
            vol_mult=2.5,
            compare_buyhold=bh_on,
            caveat=cv_on,
        )
        # 跑完返回选标的界面，可继续换标的/策略


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="百分比网格交易回测")
    parser.add_argument("ts_code", nargs="?", default="000300.SH", help="标的代码（省略则进入交互菜单）")
    parser.add_argument("start_date", nargs="?", default="20200102", help="开始日期 YYYYMMDD")
    parser.add_argument("end_date", nargs="?", default="20251231", help="结束日期 YYYYMMDD")
    parser.add_argument("--grid-pct", type=float, default=GRID_PCT, help="每格百分比（默认0.02=2%%）")
    parser.add_argument("--per-grid", type=int, default=PER_GRID_CASH, help="每格交易金额（默认5000）")
    parser.add_argument("--init-pos", type=float, default=INIT_POSITION_PCT, help="初始持仓比例（默认0.5）")
    parser.add_argument("--capital", type=int, default=INIT_CAPITAL, help="初始资金（默认100000）")
    parser.add_argument("--mode", choices=["symmetric", "asymmetric"], default="symmetric",
                        help="网格模式：symmetric=对称（买卖同距）；asymmetric=非对称（锚定成本线·卖距更宽·浮盈才卖）")
    parser.add_argument("--sell-pct", type=float, default=None,
                        help="非对称模式下的卖出间距，不填则默认=买距×2.5")
    parser.add_argument("--trend-filter", action="store_true",
                        help="开启趋势过滤：价格站上 250 日均线只持有不卖（抑制趋势踏空）")
    parser.add_argument("--ma-window", type=int, default=250,
                        help="趋势过滤的均线周期（默认250）")
    parser.add_argument("--stop-loss", type=float, default=None,
                        help="总止损线（小数，如 -0.30）：净值从峰值回撤超此阈值即硬熔断、停止新建网格（仅持有）。默认关闭")
    parser.add_argument("--vol-filter", action="store_true",
                        help="波动率关网：短期年化波动突升时暂停网格交易，恢复后自动重启")
    parser.add_argument("--vol-window", type=int, default=20,
                        help="波动率关网的回看窗口（默认20日）")
    parser.add_argument("--vol-mult", type=float, default=2.5,
                        help="波动率关网阈值倍数（默认2.5倍基准波动）")
    parser.add_argument("--compare-buyhold", action="store_true",
                        help="网格 vs 买入持有 对照（量化《长期持有ETF的真相》'满仓躺平'方法论）")
    parser.add_argument("--center-mode", choices=["fixed", "ma"], default=CENTER_MODE,
                        help="网格中枢模式：ma=滚动MA中枢网格(升级,默认)；fixed=旧版固定价位网格(锚定首日收盘价)")
    parser.add_argument("--center-ma-window", type=int, default=CENTER_MA_WINDOW,
                        help="中枢MA窗口(默认60日)；仅 center-mode=ma 生效")
    parser.add_argument("--premium-filter", type=str, default="qdii",
                        choices=["off", "uniform", "strict", "qdii", "rolling"],
                        help="折溢价闸门(默认qdii)：跨境ETF≥8%%溢价拦截；off=关闭。依据BV1YP326jE7S(用NAV非IOPV)")
    parser.add_argument("--caveat", action="store_true",
                        help="打印情景提示：市场级幸存者偏差 / 未来收益下行 / 集中度（前十大占比>40%%时升级为数据支撑的硬提醒）")
    parser.add_argument("--menu", action="store_true", help="强制进入交互式菜单（选标的+选策略）")
    parser.add_argument("--interrupt-start", type=str, default=None,
                        help="现实折扣-中断模拟：从 YYYYMM 起撤出部分资金（配合 --interrupt-months/--interrupt-pct）")
    parser.add_argument("--interrupt-months", type=int, default=0,
                        help="中断模拟持续月数（默认0=关闭）")
    parser.add_argument("--interrupt-pct", type=float, default=0.0,
                        help="中断模拟撤出比例(0~1，如 0.5=撤一半)，默认0")
    args = parser.parse_args()

    # 无参数 / 显式 --menu → 进入交互菜单
    if args.menu or len(sys.argv) == 1:
        _interactive_menu()
    else:
        run_grid_backtest(
            ts_code=args.ts_code,
            start_date=args.start_date,
            end_date=args.end_date,
            grid_pct=args.grid_pct,
            per_grid_cash=args.per_grid,
            init_position_pct=args.init_pos,
            initial_capital=args.capital,
            mode=args.mode,
            sell_pct=args.sell_pct,
            premium_filter=args.premium_filter,
            trend_filter=args.trend_filter,
            ma_window=args.ma_window,
            stop_loss=args.stop_loss,
            vol_filter=args.vol_filter,
            vol_window=args.vol_window,
            vol_mult=args.vol_mult,
            compare_buyhold=args.compare_buyhold,
            center_mode=args.center_mode,
            center_ma_window=args.center_ma_window,
            caveat=args.caveat,
            interrupt_start=args.interrupt_start,
            interrupt_months=args.interrupt_months,
            interrupt_pct=args.interrupt_pct,
        )
