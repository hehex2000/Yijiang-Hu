# -*- coding: utf-8 -*-
"""
指数择时覆盖复现实验（证伪向 · 真实成本版）
============================================
目标：在视频声称的"超大盘指数"=中证超大盘(000043.SH) 上，用标准参数实现 8 个技术因子
      + 第9类【达瓦斯箱体】(白名单UP主Jim视频方法，箱体宽度参数化 5/10/15%)
      作为【单指数择时覆盖】(timing overlay)，看是否有任何一组能复现视频声称的
      "超买超卖 +9%超额 / 总收益+30% / 最大回撤仅10%"。

重要假设（全部文档化，非忠实复现，是"标准实现能否复现"的证伪实验）：
  1. 语义：因子 = 在指数上做择时（信号多→持指数，信号空→空仓现金）。视频"降回撤到10%"
     是择时语义，不是成分股选股。
  2. 空仓现金收益：默认 货基年化 2%（也跑 0% 作对照）。
  3. 交易成本：真实分科目（A股零售），见下方 COST_* 常量：
       佣金 万2.5（单笔最低 ¥5）、印花税 卖出万5、过户费 万0.1（双边）、滑点 10bp（双边损失向）。
       滑点 SLIP_BP 在文件顶部，可改（本次按用户要求先跑 10bp）。
       组合以 ¥ 计（INIT_CAPITAL=¥100万），¥5 最低佣金在此规模下不绑定。
  4. 信号在 T 日收盘算出，应用于 T→T+1 收益（无前视）。
  5. 因子默认参数（如视频未给，按业界标准默认）：
       - RSI(14) 超买超卖：RSI>70 离场，RSI<30 回补，中间维持状态
       - KDJ(9,3,3)：K 上穿 D 进场，K 下穿 D 离场
       - CCI(14)：CCI<-100 进场(超卖)，CCI>+100 离场(超买)
       - OBV 斜率：OBV>其 MA(20) 持仓，否则空仓
       - 缺口：当日开盘相对前收低开>2% → 空仓，否则持仓（避跳空风险）
       - 下跌天数：连续下跌>=3日 → 进场(博反弹)，连续上涨>=5日 → 离场，否则维持
       - 趋势：收盘价>MA(60) 持仓，否则空仓
       - 风险收益比：滚动60日 收益/波动(类Sharpe) >0 持仓，否则空仓

运行：venv_ml/Scripts/python.exe factor_timing_experiment.py
"""
import sqlite3
import numpy as np
import os

DB = r"D:\tu-shareData\astock_daily.db"
START, END = "20160101", "20260620"
CASH_ANNUAL = 0.02  # 货基年化

# ── 真实分科目交易成本假设（A股零售）──
INIT_CAPITAL = 1_000_000.0   # 账户规模 ¥100万；¥5 最低佣金在此规模下不绑定
COMMISSION_RATE = 0.00025    # 佣金 万2.5
COMMISSION_MIN = 5.0         # 单笔最低佣金 ¥5
STAMP_RATE = 0.0005          # 印花税 万5（仅卖出方）
TRANSFER_RATE = 0.00001      # 过户费 万0.1（双边）
SLIP_BP = 10.0               # 滑点 10bp，双边损失向（本次默认；之后可调）


def get_series(code, start=START, end=END):
    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT trade_date,open,high,low,close,vol,pre_close FROM index_daily "
        "WHERE ts_code=? AND trade_date>=? AND trade_date<=? ORDER BY trade_date",
        (code, start, end),
    ).fetchall()
    con.close()
    return rows


# ─── 指标计算 ────────────────────────────────────────────────────────────
def rsi(closes, period=14):
    n = len(closes)
    r = np.full(n, np.nan)
    gains = np.where(np.diff(closes) > 0, np.diff(closes), 0.0)
    losses = np.where(np.diff(closes) < 0, -np.diff(closes), 0.0)
    ag = gains[:period].mean()
    al = losses[:period].mean()
    r[period] = 100 - 100 / (1 + (ag / al if al > 0 else 999))
    for i in range(period + 1, n):
        ag = (ag * (period - 1) + gains[i - 1]) / period
        al = (al * (period - 1) + losses[i - 1]) / period
        rs = ag / al if al > 0 else 999
        r[i] = 100 - 100 / (1 + rs)
    return r


def kdj(high, low, close, n=9):
    nn = len(close)
    rsv = np.zeros(nn)
    for i in range(n - 1, nn):
        hh = max(high[i - n + 1:i + 1])
        ll = min(low[i - n + 1:i + 1])
        rsv[i] = (close[i] - ll) / (hh - ll) * 100 if hh > ll else 0
    K = np.full(nn, 50.0)
    D = np.full(nn, 50.0)
    for i in range(n - 1, nn):
        K[i] = (2 / 3) * K[i - 1] + (1 / 3) * rsv[i]
        D[i] = (2 / 3) * D[i - 1] + (1 / 3) * K[i]
    J = 3 * K - 2 * D
    return K, D, J


def cci(high, low, close, period=14):
    nn = len(close)
    tp = (high + low + close) / 3.0
    c = np.zeros(nn)
    for i in range(period - 1, nn):
        seg = tp[i - period + 1:i + 1]
        sma = seg.mean()
        md = np.mean(np.abs(seg - sma))
        c[i] = (tp[i] - sma) / (0.015 * md) if md > 0 else 0
    return c


def obv(close, vol):
    o = np.zeros(len(close))
    for i in range(1, len(close)):
        if close[i] > close[i - 1]:
            o[i] = o[i - 1] + vol[i]
        elif close[i] < close[i - 1]:
            o[i] = o[i - 1] - vol[i]
        else:
            o[i] = o[i - 1]
    return o


def sma(arr, n):
    out = np.full(len(arr), np.nan)
    for i in range(n - 1, len(arr)):
        out[i] = arr[i - n + 1:i + 1].mean()
    return out


# ─── 信号生成（返回 long 布尔数组，long[t]=True 表示 T 日收盘后决定持有 T→T+1）───
def sig_rsi(closes):
    r = rsi(closes, 14)
    long = np.ones(len(closes), dtype=bool)
    pos = True
    for i in range(len(closes)):
        if np.isnan(r[i]):
            long[i] = pos
            continue
        if r[i] > 70:
            pos = False
        elif r[i] < 30:
            pos = True
        long[i] = pos
    return long


def sig_kdj(high, low, close):
    K, D, _ = kdj(high, low, close, 9)
    long = np.zeros(len(close), dtype=bool)
    pos = True
    for i in range(1, len(close)):
        if K[i - 1] <= D[i - 1] and K[i] > D[i]:   # 金叉
            pos = True
        elif K[i - 1] >= D[i - 1] and K[i] < D[i]:  # 死叉
            pos = False
        long[i] = pos
    long[0] = True
    return long


def sig_cci(high, low, close):
    c = cci(high, low, close, 14)
    long = np.ones(len(close), dtype=bool)
    pos = True
    for i in range(len(close)):
        if np.isnan(c[i]) or c[i] == 0:
            long[i] = pos
            continue
        if c[i] < -100:
            pos = True
        elif c[i] > 100:
            pos = False
        long[i] = pos
    return long


def sig_obv(close, vol):
    o = obv(close, vol)
    ma = sma(o, 20)
    long = np.ones(len(close), dtype=bool)
    for i in range(len(close)):
        if np.isnan(ma[i]):
            long[i] = True
        else:
            long[i] = o[i] >= ma[i]
    return long


def sig_gap(open_, pre_close, thr=0.02):
    long = np.ones(len(open_), dtype=bool)
    for i in range(len(open_)):
        if pre_close[i] and open_[i] / pre_close[i] - 1 < -thr:
            long[i] = False  # 低开超阈值→空仓
        else:
            long[i] = True
    return long


def sig_downdays(close):
    long = np.ones(len(close), dtype=bool)
    pos = True
    cons_down = 0
    cons_up = 0
    for i in range(len(close)):
        if i == 0:
            long[i] = True
            continue
        if close[i] < close[i - 1]:
            cons_down += 1
            cons_up = 0
        elif close[i] > close[i - 1]:
            cons_up += 1
            cons_down = 0
        else:
            pass
        if cons_down >= 3:
            pos = True
        elif cons_up >= 5:
            pos = False
        long[i] = pos
    return long


def sig_trend(close, n=60):
    ma = sma(close, n)
    long = np.ones(len(close), dtype=bool)
    for i in range(len(close)):
        if np.isnan(ma[i]):
            long[i] = True
        else:
            long[i] = close[i] >= ma[i]
    return long


def sig_riskreturn(close, n=60):
    long = np.ones(len(close), dtype=bool)
    rets = np.diff(np.log(close))
    for i in range(len(close)):
        if i < n:
            long[i] = True
            continue
        seg = rets[i - n:i]
        cum = close[i] / close[i - n] - 1
        vol = seg.std() * np.sqrt(252)
        ratio = cum / vol if vol > 0 else 0
        long[i] = ratio > 0
    return long


def sig_darvas(high, low, close, box_pct=0.10, win=20):
    """达瓦斯箱体（参数化 · 严格避免"网络改良版"固定比例）。

    核心规则（Jim 视频强调不可删的四条）：
      1) 选已走出上升趋势的标的  2) 等箱体成形  3) 只在向上突破箱顶时行动
      4) 跌破箱底(移动止损)即认错离场。
    实现要点：
      - 箱体宽度 box_pct 参数化（默认10%，本次跑 5/10/15 三档）。
        原书明确：不同股票箱体宽度不同（窄到极窄、宽到10%+），绝不可硬编码。
      - 空仓期：用 trailing 窗口识别"紧凑箱体"(箱顶=窗口最高高, 箱底=窗口最低低,
        且 箱顶/箱底-1<=box_pct)，收盘突破箱顶=买入信号。
      - 持仓期：箱顶随新高上移；当"新高至当前最低低"仍在 box_pct 内，视为更高箱体，
        把止损上移到该箱底（移动止损）。盘中 low<=止损 即认错离场。
      - 信号无前视：用当日收盘/低价决定当日 long[i]，作用于 i→i+1。
    """
    n = len(close)
    long = np.zeros(n, dtype=bool)  # 起始空仓，等箱体
    in_mkt = False
    stop = 0.0
    box_top = 0.0
    last_high_i = -1
    for i in range(1, n):
        h, l, c = high[i], low[i], close[i]
        if not in_mkt:
            lo = max(1, i - win + 1)
            # 箱顶/箱底用"前一日之前"的窗口，避免当日最高已含箱顶导致突破永不触发
            hwin = max(high[lo:i]) if i > lo else high[i]
            lwin = min(low[lo:i]) if i > lo else low[i]
            if hwin > 0 and (hwin / lwin - 1) <= box_pct:
                box_top = hwin
                box_bottom = lwin
                if c > box_top:  # 收盘突破前高箱顶→行动
                    in_mkt = True
                    long[i] = True
                    stop = box_bottom
                    last_high_i = i
                    continue
            long[i] = False
        else:
            long[i] = True
            if h > box_top:
                box_top = h
                last_high_i = i
            # 当前箱体下沿 = 自箱顶出现以来的最低低
            seg_lo = last_high_i + 1 if last_high_i < i else i
            new_bottom = min(low[seg_lo:i + 1]) if seg_lo <= i else l
            if box_top > 0 and (box_top / new_bottom - 1) <= box_pct:
                stop = max(stop, new_bottom)  # 更高箱体→上移止损
            if l <= stop:  # 跌破箱底→认错离场（止损指令触发）
                in_mkt = False
                long[i] = False
                stop = 0.0
                box_top = 0.0
    return long


# ─── 回测（真实分科目成本，组合以 ¥ 计）──────────────────────────────────────
def backtest(closes, long_sig, cash_annual=CASH_ANNUAL, slip_bp=SLIP_BP, init_capital=INIT_CAPITAL):
    """真实分科目成本回测（组合以 ¥ 计）。

    多仓：全仓持有指数（units×price）；空仓：持现金（按 cash_annual 增长）。
    状态切换日发生一次买或卖，按真实费用扣：
        买：佣金 max(¥5, 买额×万2.5) + 过户(买额×万0.1) + 滑点(买额×slip)
        卖：佣金 + 过户 + 印花税(卖额×万5) + 滑点(卖额×slip)
    滑点损失向：买价=close×(1+slip)，卖价=close×(1-slip)。
    信号无前视：用 long_sig[i-1] 决定当日持仓（T日收盘信号→T→T+1）。
    建仓（day0）不收费用，与原始"ret=1.0 起算"一致；买入持有因全程不切换而仅基准成本。
    """
    n = len(closes)
    slip = slip_bp / 10000.0
    cash_daily = 1 + cash_annual / 252.0

    # 初始（无建仓成本）
    if long_sig[0]:
        units = init_capital / closes[0]
        cash = 0.0
    else:
        units = 0.0
        cash = init_capital

    values = [units * closes[0] + cash]
    peak = values[0]
    mdd = 0.0
    days_in = 0
    switches = 0

    for i in range(1, n):
        prev_long = long_sig[i - 1]
        cur_long = long_sig[i]
        switched = prev_long != cur_long
        if prev_long:
            if switched:  # 卖出
                px = closes[i] * (1 - slip)
                notional = units * px
                fee = max(COMMISSION_MIN, notional * COMMISSION_RATE) + notional * TRANSFER_RATE + notional * STAMP_RATE
                cash = notional - fee
                units = 0.0
                switches += 1
            days_in += 1
        else:
            cash *= cash_daily
            if switched:  # 买入
                px = closes[i] * (1 + slip)
                notional = cash
                fee = max(COMMISSION_MIN, notional * COMMISSION_RATE) + notional * TRANSFER_RATE
                units = (notional - fee) / px
                cash = 0.0
                switches += 1
        value = units * closes[i] + cash
        values.append(value)
        if value > peak:
            peak = value
        dd = value / peak - 1
        if dd < mdd:
            mdd = dd

    total = values[-1] / init_capital - 1
    return total, mdd, days_in / (n - 1), switches


def _ann(tot, n_days):
    years = n_days / 252.0
    return (1 + tot) ** (1 / years) - 1 if years > 0 else 0.0


def main():
    rows = get_series("000043.SH")
    if not rows:
        print("无 000043 数据")
        return
    dates = [r[0] for r in rows]
    open_ = np.array([float(r[1]) for r in rows])
    high = np.array([float(r[2]) for r in rows])
    low = np.array([float(r[3]) for r in rows])
    close = np.array([float(r[4]) for r in rows])
    vol = np.array([float(r[5]) for r in rows])
    pre_close = np.array([float(r[6]) if r[6] else np.nan for r in rows])
    # pre_close 缺失时用前一日 close 近似
    for i in range(1, len(pre_close)):
        if np.isnan(pre_close[i]):
            pre_close[i] = close[i - 1]

    factors = {
        "买入持有(基准)": np.ones(len(close), dtype=bool),
        "RSI(14)超买超卖": sig_rsi(close),
        "KDJ(9,3,3)": sig_kdj(high, low, close),
        "CCI(14)±100": sig_cci(high, low, close),
        "OBV斜率(MA20)": sig_obv(close, vol),
        "缺口(低开>2%空仓)": sig_gap(open_, pre_close),
        "下跌天数(连跌3买/连涨5卖)": sig_downdays(close),
        "趋势(MA60)": sig_trend(close),
        "风险收益比(60d>0)": sig_riskreturn(close),
        # ── 方向A：达瓦斯箱体（参数化宽度 5/10/15%，避免硬编码固定比例）──
        "达瓦斯箱体(5%)": sig_darvas(high, low, close, box_pct=0.05),
        "达瓦斯箱体(10%)": sig_darvas(high, low, close, box_pct=0.10),
        "达瓦斯箱体(15%)": sig_darvas(high, low, close, box_pct=0.15),
    }

    base_ret, base_mdd, _, _ = backtest(close, factors["买入持有(基准)"])

    print(f"中证超大盘 000043.SH  区间 {dates[0]}~{dates[-1]}  ({len(close)}交易日)")
    print(f"视频声称基准: 总收益19% / 回撤46%  |  实测基准(买入持有,含真实成本): 总收益{base_ret*100:.1f}% / 回撤{base_mdd*100:.1f}%")
    print(f"视频冠军因子声称: 总收益+30% / 回撤仅10%")
    print(f"成本假设: 佣金万2.5(最低¥5) / 印花税卖出万5 / 过户万0.1 / 滑点{SLIP_BP:.0f}bp双边损失向 / 账户¥{INIT_CAPITAL/1e4:.0f}万")
    print()
    print(f"{'因子':<26}{'总收益%':>9}{'回撤%':>9}{'超额pp':>9}{'年化%':>9}{'持有时长%':>10}{'换仓':>7}")
    print("-" * 80)

    results = []
    for name, sig in factors.items():
        tot, mdd, din, sw = backtest(close, sig, CASH_ANNUAL)
        ann = _ann(tot, len(close))
        excess = (tot - base_ret) * 100
        print(f"{name:<26}{tot*100:>8.1f}{mdd*100:>8.1f}{excess:>8.1f}{ann*100:>8.1f}{din*100:>9.1f}{sw:>7d}")
        results.append((name, CASH_ANNUAL, tot, mdd, excess, ann, din, sw))

    # 对照：空仓现金 0% 年化
    print()
    print("对照（空仓现金年化 0% 时，其余成本相同）：")
    print(f"{'因子':<26}{'总收益%':>9}{'回撤%':>9}{'换仓':>7}")
    print("-" * 53)
    for name, sig in factors.items():
        tot, mdd, _, sw = backtest(close, sig, 0.0)
        print(f"{name:<26}{tot*100:>8.1f}{mdd*100:>8.1f}{sw:>7d}")

    # 结论判定
    print()
    print("=== 复现判定（视频: +30%总收益 & 10%回撤）===")
    reproduced = []
    for name, sig in factors.items():
        if name == "买入持有(基准)":
            continue
        tot, mdd, _, _ = backtest(close, sig, CASH_ANNUAL)
        ok = (tot >= 0.25) and (mdd >= -0.15)
        flag = "✅接近" if ok else "❌达不到"
        print(f"  {name:<24} 总收益{tot*100:>6.1f}%  回撤{mdd*100:>6.1f}%  {flag}")
        if ok:
            reproduced.append(name)
    if not reproduced:
        print("  >>> 真实成本下，标准实现全面达不到视频声称的 +30%/10%DD 组合")
        print("  >>> 结论倾向：视频结果疑似过拟合 / 选择性展示（40因子只展10，冠军参数未披露）")
    else:
        print(f"  >>> 可复现因子: {reproduced}")

    # 保存 CSV
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "factor_timing_000043.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("factor,cash_annual,slip_bp,total_return,max_drawdown,excess_pp,annualized,time_in_market,switches\n")
        for name, cash, tot, mdd, excess, ann, din, sw in results:
            f.write(f"{name},{cash},{SLIP_BP},{tot:.4f},{mdd:.4f},{excess:.2f},{ann:.4f},{din:.4f},{sw}\n")
    print(f"\nCSV 已保存: {out_csv}")

    # ─── RSI(14) 滑点敏感度（验证真实成本下是否仍远超视频 +30%）───
    print()
    print("=" * 82)
    print(f"RSI(14) 超买超卖 · 滑点敏感度（佣金万2.5/印花税万5/过户万0.1 固定，仅滑点扫描）")
    print("=" * 82)
    rsi_sig = factors["RSI(14)超买超卖"]
    slip_grid = [2.0, 5.0, 10.0, 20.0]
    print(f"{'现金':<8}{'滑点bp':>8}{'总收益%':>10}{'回撤%':>9}{'换仓':>7}{'超额pp':>10}")
    print("-" * 56)
    rsi_rows = []
    for cash in (0.0, CASH_ANNUAL):
        for sb in slip_grid:
            tot, mdd, din, sw = backtest(close, rsi_sig, cash, slip_bp=sb)
            excess = (tot - base_ret) * 100
            tag = "货基2%" if cash == CASH_ANNUAL else "现金0%"
            print(f"{tag:<8}{sb:>6.0f}{tot*100:>10.1f}{mdd*100:>8.1f}{sw:>7d}{excess:>9.1f}")
            rsi_rows.append((cash, sb, tot, mdd, sw, excess))
    print()
    print(f"  基准(买入持有): 总收益{base_ret*100:.1f}%  回撤{base_mdd*100:.1f}%")
    print(f"  视频冠军声称: 总收益+30% / 回撤10%")
    # 关键判定：最严情景（滑点20bp + 现金0%）仍是否≥30%
    worst = min(r for r in rsi_rows if r[0] == 0.0 and r[1] == 20.0)
    _, _, wtot, _, _, _ = worst
    print(f"  => 最严情景(滑点20bp, 现金0%): RSI 总收益{wtot*100:.1f}% "
          f"{'仍≥视频+30%（标准RSI 即可超越视频冠军）' if wtot >= 0.30 else '已低于视频+30%'}")

    rsi_csv = os.path.join(out_dir, "factor_timing_rsi_cost.csv")
    with open(rsi_csv, "w", encoding="utf-8") as f:
        f.write("cash_annual,slip_bp,total_return,max_drawdown,switches,excess_pp\n")
        for cash, sb, tot, mdd, sw, excess in rsi_rows:
            f.write(f"{cash},{sb},{tot:.4f},{mdd:.4f},{sw},{excess:.2f}\n")
    print(f"RSI滑点敏感度 CSV 已保存: {rsi_csv}")

    # ─── 全因子在滑点10bp下对照（看成本对排序的影响）───
    print()
    print("─" * 82)
    print(f"全因子对照（真实成本 / 滑点{SLIP_BP:.0f}bp / 货基2%现金）：")
    print(f"{'因子':<26}{'总收益%':>9}{'回撤%':>9}{'超额pp':>9}{'换仓':>7}")
    print("-" * 60)
    for name, sig in factors.items():
        if name == "买入持有(基准)":
            continue
        tot, mdd, din, sw = backtest(close, sig, CASH_ANNUAL)
        excess = (tot - base_ret) * 100
        print(f"{name:<26}{tot*100:>8.1f}{mdd*100:>8.1f}{excess:>8.1f}{sw:>7d}")
    print("─" * 82)


if __name__ == "__main__":
    main()
