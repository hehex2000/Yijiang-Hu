# -*- coding: utf-8 -*-
"""
仓位管理 P1 · 组合级 shared-pool sizing 引擎 + 6 档 A/B
=====================================================

背景（P0 已定，见 docs/position_sizing_experiment_plan.md §3.1.1）：
  Bootstrap 蒙特卡洛证明「破产概率 0%」「尾部回撤随 f 单调」「视频 66.7% 处方劣于平台 20% 默认」。
  但 P0 无日历时间轴 → 算不了 Sharpe，且终值被「并发上限 floor(1/f) ↔ 成交笔数」耦合污染。

P1 补上 P0 缺的三件事：
  1. 真实日历引擎 → 可算 Sharpe / CAGR / 暴露度（平均在市资金占比）
  2. 组合级共享资金池（初始 100 万）→ 信号之间真的在抢钱，仓位算法才有意义
  3. 部署率自查 → 每次建仓实际部署额 / 目标额，目标 ≈100%

★ 单变量隔离的关键设计（与布林带系列同纪律）：
  信号只抽取一次——用 1e8 超大本金 + use_kelly=False 跑插件拿到纯信号流（date/code/action/price），
  6 档共用同一条信号流，只有「每笔投入多少权益」不同。
  → use_kelly 只影响股数不影响信号触发（_enter_long 要求 position==0 才进，room 恒 >0），
     故关掉它不会改变信号集，反而杜绝「股数被 cap 成 0 → success=False → 状态机走偏」的隐患。

档位（只改仓位算法，信号/出场/成本/股票池全共用）：
  FULL            f = 100%          满仓，实际同时只持 1 只
  FIXED20         f = 20%           平台当前默认口径（kelly_cap=0.20 的等效固定值）
  KELLY_FULL      f = (b·p−q)/b     滚动估计 p/b（只用已平仓交易，防前视）
  KELLY_HALF      f = 凯利/2
  RISK_2PCT_NOM   f = 2% / 3%       视频隐含处方（名义止损）→ 66.7%
  RISK_2PCT_REAL  f = 2% / 滚动实测亏损距离   修正版风险预算
"""
import sys
import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from run_backtest import load_stock_prices, _get_index_constituents_from_db  # noqa: E402
from backtest.mean_reversion_plugin import MeanReversionStrategyPlugin  # noqa: E402
from backtest.base_strategy import BaseStrategy  # noqa: E402
import config  # noqa: E402

START = "20190101"
END = "20260731"
TOP_N = 40
BENCH = "000300.SH"
INIT_CAPITAL = 1_000_000.0   # 组合级共享资金池
SIG_CAPITAL = 1e8            # 抽信号用的超大本金（保证现金不约束信号）

RISK_BUDGET = 0.02           # 2% 单笔风险预算
NOMINAL_STOP = 0.03          # 名义止损距离 3%
PRIOR_P, PRIOR_B = 0.5, 1.5  # 凯利保守先验（样本不足时）
PRIOR_DIST = 0.06            # 风险预算保守先验距离 = 名义3%×2（实测亏损是名义的2~6倍）
MIN_SAMPLE = 20              # 滚动估计最小样本

TIERS = ["FULL", "FIXED20", "KELLY_FULL", "KELLY_HALF",
         "RISK_2PCT_NOM", "RISK_2PCT_REAL"]
OUT_DIR = HERE / "data" / "results" / "position_sizing"

# 复用 base_strategy 的真实分科目成本公式（opt-in 主口径）
_FEE = BaseStrategy("fee", 0.0, {})


def buy_fee(amount: float, date) -> float:
    return _FEE._real_fee_buy(amount, date)


def sell_fee(amount: float, date):
    return _FEE._real_fee_sell(amount, date)   # -> (fee, stamp)


# ─────────────────────────────────────────────────────────────
# 仓位算法：f = 单笔投入占当前权益的比例
# ─────────────────────────────────────────────────────────────
def compute_f(tier: str, closed: list) -> float:
    """closed: 已实现 round trip 的净收益百分比列表（只含 t 之前平仓的，防前视）。"""
    if tier == "FULL":
        return 1.0
    if tier == "FIXED20":
        return 0.20
    if tier == "RISK_2PCT_NOM":
        return RISK_BUDGET / NOMINAL_STOP

    n = len(closed)
    if n < MIN_SAMPLE:
        p, b = PRIOR_P, PRIOR_B
    else:
        arr = np.asarray(closed, dtype=float)
        w, l = arr[arr > 0], arr[arr <= 0]
        p = len(w) / len(arr)
        aw = float(np.mean(w)) if len(w) else 0.0
        al = abs(float(np.mean(l))) if len(l) else 0.0
        b = aw / al if al > 0 else PRIOR_B

    if tier in ("KELLY_FULL", "KELLY_HALF"):
        f = (b * p - (1 - p)) / b if b > 0 else 0.0
        if tier == "KELLY_HALF":
            f /= 2.0
    elif tier == "RISK_2PCT_REAL":
        if n < MIN_SAMPLE:
            d = PRIOR_DIST
        else:
            arr = np.asarray(closed, dtype=float)
            l = arr[arr < 0]
            d = float(np.mean(-l)) / 100.0 if len(l) else PRIOR_DIST
        f = RISK_BUDGET / d if d > 0 else 0.0
    else:
        f = 0.20
    return float(min(max(f, 0.0), 1.0))


# ─────────────────────────────────────────────────────────────
# 信号抽取（只做一次，6 档共用）
# ─────────────────────────────────────────────────────────────
def extract_signals(codes: list):
    conn = sqlite3.connect(config.DATA["local_db_path"])
    events, px_series = [], {}
    for i, code in enumerate(codes, 1):
        try:
            df = load_stock_prices(code, START, END, conn, lookback_days=250)
        except Exception as e:  # noqa: BLE001
            print(f"  [ERR] load {code}: {e}")
            continue
        if df is None or len(df) < 30:
            continue
        df = df.reset_index(drop=True)
        try:
            start_idx = int(df[df["trade_date"] >= START].index.min())
        except Exception:  # noqa: BLE001
            continue

        cfg = dict(config.STRATEGIES["mean_reversion"])
        cfg["use_kelly"] = False    # 只影响股数不影响信号 → 关掉以抽纯信号
        cfg["real_cost"] = False
        strat = MeanReversionStrategyPlugin(SIG_CAPITAL, cfg)
        res = strat.run(df, start_idx)

        for t in res.get("trades", []):
            if t.get("reason") == "回测结束平仓":
                continue           # 期末强平由组合引擎统一处理
            act = t.get("action", "")
            if not (act.startswith("BUY") or act.startswith("SELL")):
                continue
            events.append({"date": t["date"], "code": code,
                           "action": "BUY" if act.startswith("BUY") else "SELL",
                           "price": float(t["price"])})

        s = df[["trade_date", "adj_close"]].copy()
        s = s[s["trade_date"] >= START]
        px_series[code] = s.set_index("trade_date")["adj_close"].astype(float)
        if i % 10 == 0 or i == 1:
            print(f"  ...[{i:>2}/{len(codes)}] {code} 信号抽取完成，累计事件 {len(events)}")
    conn.close()

    events.sort(key=lambda e: (e["date"], e["code"]))
    px = pd.DataFrame(px_series).sort_index().ffill().fillna(0.0)
    return events, px


# ─────────────────────────────────────────────────────────────
# 组合级模拟
# ─────────────────────────────────────────────────────────────
def simulate(tier: str, events: list, px: pd.DataFrame, init_capital: float = INIT_CAPITAL,
             fixed_f: float = None, fixed_cap: int = None):
    codes = list(px.columns)
    ci = {c: i for i, c in enumerate(codes)}
    px_mat = px.values.astype(float)
    calendar = list(px.index)
    ndays = len(calendar)

    buys, sells = {}, {}
    for e in events:
        if e["code"] not in ci:
            continue
        (buys if e["action"] == "BUY" else sells).setdefault(e["date"], []).append(e)

    cash = init_capital
    shares_vec = np.zeros(len(codes))
    cost_vec = np.zeros(len(codes))
    closed = []
    nav = np.zeros(ndays)
    inv = np.zeros(ndays)
    last_equity = init_capital

    n_sig_buy = sum(len(v) for v in buys.values())
    n_taken, skip_cap, skip_cash, skip_lot = 0, 0, 0, 0
    deploys, f_used = [], []

    for t, date in enumerate(calendar):
        # 先卖后买：释放现金，避免同日现金流错序造成的虚假资金约束
        for e in sells.get(date, []):
            j = ci[e["code"]]
            if shares_vec[j] <= 0:
                continue
            amount = shares_vec[j] * e["price"]
            fee, tax = sell_fee(amount, date)
            net = amount - fee - tax
            cash += net
            if cost_vec[j] > 0:
                closed.append((net - cost_vec[j]) / cost_vec[j] * 100.0)
            shares_vec[j] = 0.0
            cost_vec[j] = 0.0

        for e in buys.get(date, []):
            j = ci[e["code"]]
            if shares_vec[j] > 0:
                continue
            f = fixed_f if fixed_f is not None else compute_f(tier, closed)
            f_used.append(f)
            # cap 与 f 解耦：给了 fixed_cap 就不再用 floor(1/f)，
            # 使「同 cap ⇒ 成交信号集合完全相同」，从而干净分离「每笔投多少」
            cap = fixed_cap if fixed_cap else (max(1, int(np.floor(1.0 / f))) if f > 0 else 1)
            if int((shares_vec > 0).sum()) >= cap:
                skip_cap += 1
                continue
            target = f * last_equity
            price = e["price"]
            sh = int(target / price / 100) * 100
            if sh <= 0:
                skip_lot += 1
                continue
            amount = sh * price
            total = amount + buy_fee(amount, date)
            if total > cash:
                sh = int(cash * 0.999 / price / 100) * 100
                if sh <= 0:
                    skip_cash += 1
                    continue
                amount = sh * price
                total = amount + buy_fee(amount, date)
                if total > cash:
                    skip_cash += 1
                    continue
            cash -= total
            shares_vec[j] = float(sh)
            cost_vec[j] = total
            deploys.append(amount / target if target > 0 else 0.0)
            n_taken += 1

        prices_t = px_mat[t]
        invested = float(np.dot(shares_vec, prices_t))
        equity = cash + invested
        nav[t], inv[t] = equity, invested
        last_equity = equity

    # 期末强平（与插件「回测结束平仓」同口径，保证各档可比）
    last_date, prices_end = calendar[-1], px_mat[-1]
    for c, j in ci.items():
        if shares_vec[j] > 0:
            amount = shares_vec[j] * prices_end[j]
            fee, tax = sell_fee(amount, last_date)
            net = amount - fee - tax
            cash += net
            if cost_vec[j] > 0:
                closed.append((net - cost_vec[j]) / cost_vec[j] * 100.0)
            shares_vec[j] = 0.0
    terminal = cash

    years = (pd.Timestamp(END) - pd.Timestamp(START)).days / 365.25
    total_ret = terminal / init_capital - 1.0
    cagr = (terminal / init_capital) ** (1 / years) - 1.0 if terminal > 0 and years > 0 else -1.0
    rets = np.diff(nav) / np.where(nav[:-1] > 0, nav[:-1], np.nan)
    rets = rets[np.isfinite(rets)]
    sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(252)) if len(rets) > 1 and np.std(rets) > 0 else 0.0
    peak = np.maximum.accumulate(nav)
    mdd = float((nav / np.where(peak > 0, peak, 1.0) - 1.0).min())
    exposure = float(np.mean(inv / np.where(nav > 0, nav, 1.0)))
    arr = np.asarray(closed, dtype=float) if closed else np.array([0.0])

    cap_used = fixed_cap if fixed_cap else (max(1, int(np.floor(1.0 / np.mean(f_used)))) if f_used else 1)
    return {
        "tier": tier,
        "cap": cap_used,
        "f_mean": float(np.mean(f_used)) if f_used else 0.0,
        "f_min": float(np.min(f_used)) if f_used else 0.0,
        "f_max": float(np.max(f_used)) if f_used else 0.0,
        "n_signal_buy": n_sig_buy,
        "n_taken": n_taken,
        "take_rate": n_taken / n_sig_buy if n_sig_buy else 0.0,
        "skip_cap": skip_cap,
        "skip_cash": skip_cash,
        "skip_lot": skip_lot,
        "deploy": float(np.mean(deploys)) if deploys else 0.0,
        "terminal": terminal,
        "total_ret_pct": total_ret * 100,
        "cagr_pct": cagr * 100,
        "sharpe": sharpe,
        "mdd_pct": mdd * 100,
        "exposure": exposure,
        "ret_per_exposure": (cagr * 100 / exposure) if exposure > 0 else 0.0,
        "n_closed": len(closed),
        "win_rate_pct": float((arr > 0).mean() * 100),
        "avg_rt_pct": float(np.mean(arr)),
    }, pd.Series(nav, index=calendar, name=tier)


P2_F_LIST = [1.0, 0.667, 0.393, 0.30, 0.20, 0.107, 0.05]
P2_CAP_LIST = [1, 2, 3, 5, 8]


def run_grid(events: list, px: pd.DataFrame):
    """P2① · f × cap 解耦网格。

    P1 的混淆：cap 恒等于 floor(1/f) ⇒ f 同时改变「每笔投多少」和「能成交哪些信号」。
    本网格把 cap 钉死：只要 f×cap ≤ 1（不触现金约束），同一 cap 下**成交的信号集合完全相同**，
    档间差异便纯粹来自每笔投入比例 f —— 这才是凯利/风险预算真正该回答的问题。

    自证：同 cap 各 f 的 n_taken 必须完全一致且 skip_cash=0；否则说明现金约束介入，该组作废。
    """
    rows = []
    print("\n[P2①] f × cap 解耦网格（同 cap ⇒ 成交信号集合相同，只改每笔投入比例）")
    print(f"  {'cap':>4}{'f':>7}{'成交':>7}{'skip_cash':>10}{'CAGR%':>8}{'Sharpe':>8}"
          f"{'MDD%':>9}{'暴露度':>8}{'CAGR/暴露':>10}{'均笔%':>8}")
    print("  " + "-" * 86)
    for cap in P2_CAP_LIST:
        for f in P2_F_LIST:
            if f * cap > 1.0 + 1e-9:
                continue                     # 需杠杆，跳过
            m, _ = simulate(f"f={f:.3f}", events, px, fixed_f=f, fixed_cap=cap)
            rows.append(m)
            print(f"  {cap:>4}{f:>7.3f}{m['n_taken']:>7}{m['skip_cash']:>10}"
                  f"{m['cagr_pct']:>8.2f}{m['sharpe']:>8.2f}{m['mdd_pct']:>9.2f}"
                  f"{m['exposure']*100:>7.1f}%{m['ret_per_exposure']:>10.2f}{m['avg_rt_pct']:>8.3f}")
        grp = [r for r in rows if r["cap"] == cap]
        nts = {r["n_taken"] for r in grp}
        scs = sum(r["skip_cash"] for r in grp)
        ok = len(nts) == 1 and scs == 0
        print(f"      └ cap={cap} 自证: {'✅ 成交信号集合一致' if ok else f'⚠️ 不一致 {sorted(nts)} (skip_cash={scs})'}\n")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="store_true",
                    help="P2① f×cap 解耦网格（默认跑 P1 的 6 档 A/B）")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 104)
    print("P1 · 组合级 shared-pool sizing 引擎 + 6 档 A/B")
    print("=" * 104)
    print(f"  区间 {START}~{END}｜股票池 沪深300 as-of {START} 前 {TOP_N} 只｜初始资金 {INIT_CAPITAL:,.0f}")
    print(f"  成本 真实分科目（佣金万2.5+滑点0.1%+过户费+日期感知印花税）｜凯利/风险预算 滚动估计(最小样本{MIN_SAMPLE})")

    dfu = _get_index_constituents_from_db(BENCH, as_of_date=START)
    codes = sorted(dfu["code"].tolist())[:TOP_N] if dfu is not None and not dfu.empty else []
    print(f"[股票池] 沪深300 as-of {START} → {len(codes)} 只\n")

    print("[1/2] 抽取信号流（6 档共用，超大本金 + use_kelly=False）")
    events, px = extract_signals(codes)
    nb = sum(1 for e in events if e["action"] == "BUY")
    ns = sum(1 for e in events if e["action"] == "SELL")
    print(f"  信号事件：BUY {nb} / SELL {ns}｜价格矩阵 {px.shape[0]} 日 × {px.shape[1]} 只\n")

    if args.grid:
        df = run_grid(events, px)
        df.to_csv(OUT_DIR / "sizing_grid.csv", index=False, encoding="utf-8-sig")
        print(f"[已保存] {OUT_DIR}/sizing_grid.csv")
        return

    print("[2/2] 6 档组合级回测")
    rows, navs = [], []
    for tier in TIERS:
        m, nav = simulate(tier, events, px)
        rows.append(m)
        navs.append(nav)
        print(f"  {tier:<16} 终值 {m['terminal']:>14,.0f}  CAGR {m['cagr_pct']:>6.2f}%  "
              f"Sharpe {m['sharpe']:>5.2f}  MDD {m['mdd_pct']:>7.2f}%  成交 {m['n_taken']:>4}/{m['n_signal_buy']}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "sizing_ablation.csv", index=False, encoding="utf-8-sig")
    pd.concat(navs, axis=1).to_csv(OUT_DIR / "sizing_nav.csv", encoding="utf-8-sig")

    print("\n" + "=" * 104)
    print("P1 · 6 档对比（组合级共享资金池，真实成本，信号流完全相同）")
    print("=" * 104)
    hdr = (f"{'档位':<16}{'f均值':>7}{'成交/信号':>11}{'部署率':>8}"
           f"{'CAGR%':>8}{'Sharpe':>8}{'MDD%':>9}{'暴露度':>8}{'CAGR/暴露':>10}{'均笔%':>8}{'胜率%':>8}")
    print(hdr)
    print("-" * 104)
    for r in rows:
        print(f"{r['tier']:<16}{r['f_mean']:>7.3f}{r['n_taken']:>6}/{r['n_signal_buy']:<4}"
              f"{r['deploy']*100:>7.1f}%{r['cagr_pct']:>8.2f}{r['sharpe']:>8.2f}"
              f"{r['mdd_pct']:>9.2f}{r['exposure']*100:>7.1f}%{r['ret_per_exposure']:>10.2f}"
              f"{r['avg_rt_pct']:>8.3f}{r['win_rate_pct']:>8.1f}")

    print("\n[部署率自查] 目标 ≈100%（每次建仓实际部署额/目标额）；<100% 说明存在现金/整手约束被静默吸收")
    bad = [r for r in rows if r["deploy"] < 0.98]
    if bad:
        for r in bad:
            print(f"  ⚠️ {r['tier']}: 部署率 {r['deploy']*100:.1f}%  "
                  f"(skip_cap={r['skip_cap']} skip_cash={r['skip_cash']} skip_lot={r['skip_lot']})")
    else:
        print("  ✅ 全部档位部署率 ≥98%")

    print("\n[容量代价] 信号因并发上限/资金不足被跳过（这是仓位算法的真实成本，不是 bug）")
    for r in rows:
        print(f"  {r['tier']:<16} 成交率 {r['take_rate']*100:>5.1f}%  "
              f"skip_cap={r['skip_cap']:<5} skip_cash={r['skip_cash']:<5} skip_lot={r['skip_lot']}")

    # P0 对照：固定 f 档按 f 精确匹配；动态档仅列 P1 自身
    p0_path = OUT_DIR / "sizing_bootstrap.csv"
    if p0_path.exists():
        try:
            p0 = pd.read_csv(p0_path)
            if "method" in p0.columns:
                p0 = p0[p0["method"] == "block"]
            print("\n[P0 对照] P0=25 年 1775 次抽样的分布，P1=7.5 年单一实现路径；"
                  "量级不可直接比，只看方向/排序")
            print("  ⚠️ 口径修正：P0 的 mdd 存小数(已×100)；mdd_p95 是【浅尾】(95% 路径好于此)，"
                  "\n     中位看 mdd_p50，真正的深尾需 p5（P0 未存，待补）")
            print(f"  {'档位':<16}{'f':>7}{'P0中位MDD%':>13}{'P0浅尾P95%':>13}{'P1实现MDD%':>13}")
            for r in rows:
                hit = p0[np.isclose(p0["f"].astype(float), r["f_mean"], atol=0.002)]
                if not hit.empty:
                    h = hit.iloc[0]
                    print(f"  {r['tier']:<16}{r['f_mean']:>7.3f}{h['mdd_p50']*100:>13.2f}"
                          f"{h['mdd_p95']*100:>13.2f}{r['mdd_pct']:>13.2f}")
                else:
                    print(f"  {r['tier']:<16}{r['f_mean']:>7.3f}{'—':>13}{'—':>13}"
                          f"{r['mdd_pct']:>13.2f}   (动态 {r['f_min']:.3f}~{r['f_max']:.3f})")
        except Exception as e:  # noqa: BLE001
            print(f"  [P0 对照跳过] {e}")

    print(f"\n[已保存] {OUT_DIR}/sizing_ablation.csv  +  sizing_nav.csv")


if __name__ == "__main__":
    main()
