# -*- coding: utf-8 -*-
"""
组合级共享资金池引擎（opt-in，默认关闭）
================================================

★ 为什么需要这个模块
-----------------------------------------------------------------
平台主流程 `run_backtest()` 是**逐票独立**架构：

    capital = 总资金 / N
    for code in stock_data:
        strategy = PluginClass(capital, scfg)     # 每票一个独立实例
        result = strategy.run(df, start_idx)

每只股票拿到 1/N 的资金、彼此不共享现金，于是：

    真实单票仓位 = position_pct(0.50) / N(40) = 1.25%
    组合总暴露   ≈ 8.5%        ← 91.5% 的资金在闲置
    CAGR ≈ 3.40%（详见 docs/position_sizing_experiment_plan.md §3.6）

而 `kelly_max_position=0.20` 的封顶基准是**组合总资金**（0.20×100万=20万），
单票现金只有 2.5 万 —— 夹子恒不触发，是一段从未生效的死代码。

本模块提供**组合级共享资金池**口径：一个资金池，所有信号按同一时间轴抢钱，
逐日盯市（浮亏可见），单笔投入 = 当前权益 × f。

★ 设计纪律（与全项目一致）
-----------------------------------------------------------------
1. **opt-in**：只有 config 里显式 `portfolio_shared_pool=True` 才走这条路，
   默认 False ⇒ 主流程行为零变更。
2. **信号只抽一次**：用 1e8 超大本金 + `use_kelly=False` 跑插件拿纯信号流，
   与仓位解耦（use_kelly 只改股数不改信号触发）。
3. **强制暴露度自查**：必须报 暴露度 / 均并发 / skip_cash / skip_lot / 部署率，
   防止"收益高"其实是"多投钱"的假象。

来源：run_position_sizing_ablation.py（P1/P2 实验引擎，已验证）。
本模块是其**平台化精简版**，口径保持一致，去掉实验用的凯利/风险预算/随机丢弃分支。
"""
import numpy as np
import pandas as pd

from backtest.base_strategy import BaseStrategy

SIG_CAPITAL = 1e8          # 抽信号用的超大本金：保证现金不约束信号触发

# 复用 base_strategy 的真实分科目成本公式（与 P1 引擎同口径）
_FEE = BaseStrategy("fee", 0.0, {})


def buy_fee(amount: float, date) -> float:
    return _FEE._real_fee_buy(amount, date)


def sell_fee(amount: float, date):
    return _FEE._real_fee_sell(amount, date)   # -> (fee, stamp)


# ─────────────────────────────────────────────────────────────
# 1. 信号抽取
# ─────────────────────────────────────────────────────────────
def extract_signals(stock_data: dict, plugin_class, cfg: dict, start: str):
    """从平台**已加载**的 stock_data 抽信号（不重复读库，保证与逐票口径同源）。

    stock_data: {code: (name, df, start_idx)}  ← run_backtest 里已 load 好
    返回: (events, px)
      events: [{"date","code","action","price"}, ...] 按 (date, code) 排序
      px:     DataFrame，index=trade_date，列=code，值=adj_close（用于逐日盯市）
    """
    events, px_series = [], {}
    for code, (name, df, start_idx) in stock_data.items():
        try:
            d = df.reset_index(drop=True)
            if "trade_date" not in d.columns:
                continue
            si = int(d[d["trade_date"] >= start].index.min())
        except Exception:  # noqa: BLE001
            continue

        c = dict(cfg)
        c["use_kelly"] = False     # 只影响股数不影响信号 → 关掉以抽纯信号
        c["real_cost"] = False     # 成本在组合引擎里统一算，避免重复扣费
        try:
            res = plugin_class(SIG_CAPITAL, c).run(d, si)
        except Exception:  # noqa: BLE001
            continue

        for t in res.get("trades", []):
            if t.get("reason") == "回测结束平仓":
                continue                      # 期末强平由组合引擎统一处理
            act = str(t.get("action", ""))
            if not (act.startswith("BUY") or act.startswith("SELL")):
                continue
            events.append({
                "date": t["date"], "code": code,
                "action": "BUY" if act.startswith("BUY") else "SELL",
                "price": float(t["price"]),
            })

        s = d[["trade_date", "adj_close"]].copy()
        s = s[s["trade_date"] >= start]
        px_series[code] = s.set_index("trade_date")["adj_close"].astype(float)

    events.sort(key=lambda e: (e["date"], e["code"]))
    px = pd.DataFrame(px_series).sort_index().ffill().fillna(0.0)
    return events, px


# ─────────────────────────────────────────────────────────────
# 2. 组合级模拟
# ─────────────────────────────────────────────────────────────
def simulate(events: list, px: pd.DataFrame, init_capital: float,
             f: float = 0.10, cap: int = 0, label: str = "PORTFOLIO"):
    """组合级共享资金池回测。

    f   : 单笔投入占【当前权益】的比例（0.10 = 10%）
    cap : 并发持仓上限，0 表示不限（靠现金自然饱和）—— 平台真实口径
    返回: (metrics dict, nav Series)
    """
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
    pos_cnt = np.zeros(ndays)
    last_equity = init_capital

    n_sig_buy = sum(len(v) for v in buys.values())
    n_taken, skip_cap, skip_cash, skip_lot = 0, 0, 0, 0
    deploys = []

    for t, date in enumerate(calendar):
        # 先卖后买：释放现金，避免同日现金流错序造成虚假资金约束
        for e in sells.get(date, []):
            j = ci[e["code"]]
            if shares_vec[j] <= 0:
                continue
            amount = shares_vec[j] * e["price"]
            fee, tax = sell_fee(amount, date)
            cash += amount - fee - tax
            if cost_vec[j] > 0:
                closed.append((amount - fee - tax - cost_vec[j]) / cost_vec[j] * 100.0)
            shares_vec[j] = 0.0
            cost_vec[j] = 0.0

        n_open = int((shares_vec > 0).sum())
        for e in buys.get(date, []):
            j = ci[e["code"]]
            if shares_vec[j] > 0:
                continue
            if cap and n_open >= cap:
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
            n_open += 1
            n_taken += 1

        prices_t = px_mat[t]
        invested = float(np.dot(shares_vec, prices_t))
        equity = cash + invested
        nav[t], inv[t] = equity, invested
        pos_cnt[t] = float((shares_vec > 0).sum())
        last_equity = equity

    # 期末强平（与插件「回测结束平仓」同口径）
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

    years = max((pd.Timestamp(calendar[-1]) - pd.Timestamp(calendar[0])).days / 365.25, 1e-9)
    total_ret = terminal / init_capital - 1.0
    cagr = (terminal / init_capital) ** (1 / years) - 1.0 if terminal > 0 else -1.0

    rets = np.diff(nav) / np.where(nav[:-1] > 0, nav[:-1], np.nan)
    rets = rets[np.isfinite(rets)]
    sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(252)) if len(rets) > 1 and np.std(rets) > 0 else 0.0

    peak_arr = np.maximum.accumulate(nav)
    dd_arr = nav / np.where(peak_arr > 0, peak_arr, 1.0) - 1.0
    mdd = float(dd_arr.min())
    i_tr = int(np.argmin(dd_arr))
    i_pk = int(np.argmax(nav[:i_tr + 1])) if i_tr > 0 else 0
    dd_period = (calendar[i_pk], calendar[i_tr])

    exposure = float(np.mean(inv / np.where(nav > 0, nav, 1.0)))
    arr = np.asarray(closed, dtype=float) if closed else np.array([0.0])

    return {
        "label": label,
        "f": f,
        "cap": cap,
        "n_stocks": len(codes),
        "n_signal_buy": n_sig_buy,
        "n_taken": n_taken,
        "take_rate": n_taken / n_sig_buy if n_sig_buy else 0.0,
        "avg_conc": float(np.mean(pos_cnt)),
        "p90_conc": float(np.percentile(pos_cnt, 90)),
        "max_conc": float(np.max(pos_cnt)),
        "skip_cap": skip_cap,
        "skip_cash": skip_cash,
        "skip_lot": skip_lot,
        "deploy": float(np.mean(deploys)) if deploys else 0.0,
        "terminal": terminal,
        "total_ret_pct": total_ret * 100,
        "cagr_pct": cagr * 100,
        "sharpe": sharpe,
        "mdd_pct": mdd * 100,
        "dd_period": dd_period,
        "exposure": exposure,
        "ret_per_exposure": (cagr * 100 / exposure) if exposure > 0 else 0.0,
        "n_closed": len(closed),
        "win_rate_pct": float((arr > 0).mean() * 100),
        "avg_rt_pct": float(np.mean(arr)),
    }, pd.Series(nav, index=calendar, name=label)


# ─────────────────────────────────────────────────────────────
# 3. 平台入口：给 run_backtest 用
# ─────────────────────────────────────────────────────────────
def run_portfolio_mode(stock_data: dict, plugin_class, scfg: dict,
                       total_capital: float, start: str, end: str,
                       idx_ret: float = 0.0, bh_mean: float = 0.0,
                       save_nav: bool = True):
    """跑一次组合级回测，返回一条**兼容 run_backtest 汇总结构**的记录。

    返回的 dict 字段与逐票 results 元素一致（ret/ann_ret/exc/...），
    因此 all_summaries 的 mean/median/best 等聚合对它退化为自身值，无需改汇总代码。
    """
    f = float(scfg.get("portfolio_f", 0.10))
    cap = int(scfg.get("portfolio_cap", 0) or 0)

    events, px = extract_signals(stock_data, plugin_class, scfg, start)
    if px.empty or not events:
        return None

    m, nav = simulate(events, px, float(total_capital), f=f, cap=cap)

    if save_nav:
        try:
            from pathlib import Path
            out = Path("data/results/position_sizing")
            out.mkdir(parents=True, exist_ok=True)
            nav.to_csv(out / "portfolio_mode_nav.csv", encoding="utf-8-sig")
        except Exception:  # noqa: BLE001
            pass

    ret = m["total_ret_pct"]
    return {
        "code": "组合",
        "name": f"池{len(stock_data)}只",
        "initial": float(total_capital),
        "final_val": m["terminal"],
        "profit": m["terminal"] - float(total_capital),
        "ret": ret,
        "ann_ret": m["cagr_pct"],
        "exc": ret - idx_ret,
        "vs_bh": ret - bh_mean,
        "trades": m["n_taken"],
        "beat": ret > idx_ret,
        "max_dd": m["mdd_pct"],
        "dd_period": m["dd_period"],
        "win_rate": m["win_rate_pct"],
        "metrics": m,
    }
