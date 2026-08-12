"""
position_sizing.py — 可复用仓位管理模块
========================================

把「金字塔 / 倒金字塔 / 马丁格尔」加仓思想，落到平台 run_monthly_rebalance 的
「月度调仓 + 目标权重」范式下，成为**组合层面按持仓盈亏(P&L)调权重**的方案。

与事件驱动式金字塔(在单一仓位内逐步加码)不同，本模块是组合权重的倾斜——
更契合月调仓结构，且天然受「总仓位=1」「单票上限」约束，可防马丁格尔爆仓。

四种方案 (scheme)：
  equal     (SINGLE)   : 等权重。基线，无倾斜。
  pyramid   (正金字塔) : 盈利仓位加权（赢家温和加注）。
                          raw = 1 + alpha·relu(pnl)；alpha 默认 0.5。
                          对应原定义「底部下大注、上涨少加」，保守。
  inverted  (倒金字塔) : 盈利仓位加权且激进（越涨越加、买在高位）。
                          raw = 1 + gamma·relu(pnl)；gamma 默认 1.0。最危险。
                          对应原定义「越涨越加」10-20-30-40。
  martingale(马丁格尔) : 亏损仓位加权（越亏越补 / 摊平成本）。
                          raw = 1 + beta·relu(−pnl)；带单票权重上限防爆仓。
                          beta 默认 0.5，max_w_ratio 默认 2.0（单票权重封顶 = 2×等权）。

所有方案返回 weights: dict(code->float)，已归一化 sum≈1。
martingale 对单票权重做 max_w 截断后重归一化，避免全仓压在一只下跌票上爆仓。

历史教训（见 data/results/pyramid/ 系列报告）：
  ① 入场信号质量 >> 仓位管理。烂信号上任何加仓都是优化一笔不该下的注。
  ② 马丁格尔尾部风险巨大（万笔版 10.11% 交易亏>20%，是其他的 37 倍），
     故本模块强制 max_w 上限，把"有限资金爆仓"显式化而非隐藏。
"""

# 方案元数据（供 CLI / 报告引用）
SCHEMES = {
    "equal":     {"label": "一次性/等权(SINGLE)", "desc": "等权重基线，无盈亏倾斜"},
    "pyramid":   {"label": "正金字塔(PYRAMID)",   "desc": "盈利仓位温和加注(赢家加注)"},
    "inverted":  {"label": "倒金字塔(INVERTED)",  "desc": "盈利仓位激进加注(越涨越加)"},
    "martingale": {"label": "马丁格尔(MARTINGALE)", "desc": "亏损仓位加注(越亏越补)，带单票上限"},
}

# 默认参数
ALPHA = 0.5        # pyramid 赢家倾斜强度
GAMMA = 1.0        # inverted 赢家倾斜强度（更激进）
BETA = 0.5         # martingale 输家倾斜强度
MAX_W_RATIO = 2.0  # martingale 单票权重上限 = MAX_W_RATIO × 等权（自适应防爆仓）


def _relu(x):
    return x if x > 0 else 0.0


def compute_target_weights(scheme, codes, pnl=None,
                           alpha=ALPHA, gamma=GAMMA, beta=BETA,
                           max_w_ratio=MAX_W_RATIO):
    """计算目标权重 dict(code->float, 归一化 sum≈1)。

    Args:
        scheme: equal/pyramid/inverted/martingale
        codes : 本期目标持仓列表（已选出的股票）
        pnl   : dict(code->float) 持仓自买入以来的收益率(小数, 如 0.12/-0.12)。
                不在 pnl 中的 code（新买入）视为 0（基线权重）。
        alpha/gamma/beta: 各方案倾斜强度
        max_w_ratio: martingale 单票权重上限 = max_w_ratio × 等权(1/n)，自适应防爆仓
    Returns:
        dict(code->weight)
    """
    codes = list(codes)
    n = len(codes)
    if n == 0:
        return {}
    pnl = pnl or {}
    raw = {}
    for c in codes:
        p = float(pnl.get(c, 0.0) or 0.0)
        if scheme == "pyramid":
            raw[c] = 1.0 + alpha * _relu(p)
        elif scheme == "inverted":
            raw[c] = 1.0 + gamma * _relu(p)
        elif scheme == "martingale":
            raw[c] = 1.0 + beta * _relu(-p)
        else:  # equal 或未知 -> 等权
            raw[c] = 1.0
    tot = sum(raw.values())
    if tot <= 0:
        return {c: 1.0 / n for c in codes}
    w = {c: v / tot for c, v in raw.items()}

    # 马丁格尔：单票权重封顶(= max_w_ratio × 等权)后重归一化（防爆仓）
    if scheme == "martingale" and max_w_ratio and max_w_ratio > 0:
        cap = max_w_ratio / n
        w = {c: min(v, cap) for c, v in w.items()}
        tot2 = sum(w.values())
        if tot2 > 0:
            w = {c: v / tot2 for c, v in w.items()}
    return w


def rebalance_to_targets(positions, cash, target_weights, td,
                         get_open_price, calc_fee, lot=100, buy_idx_default=0):
    """按目标权重对本期持仓做再平衡（买新 + 对已有仓位加减至目标权重）。

    调用前应已卖出「不在目标持仓内」的旧仓位。

    Args:
        positions     : dict(code-> {shares,buy_price,buy_idx,highest_close,stop_triggered})
        cash          : 当前现金(float)
        target_weights: dict(code->weight)，仅含本期目标持仓
        td            : 调仓日(YYYYMMDD)，成交价用当日开盘价
        get_open_price: 引擎的价格函数
        calc_fee      : 引擎的费用函数
        lot           : 交易手数(默认100)
    Returns:
        (positions, cash, trades)
        trades: list of dict(date,action,code,name,price,shares,reason)
    """
    if not target_weights:
        return positions, cash, []

    # 调仓日组合总市值（用开盘价估值）
    total_assets = float(cash)
    for c, pos in positions.items():
        op = get_open_price(c, td)
        if op is not None:
            total_assets += pos["shares"] * op

    trades = []
    name_cache = {}

    def _name(code):
        if code not in name_cache:
            from run_monthly_rebalance import get_stock_name
            name_cache[code] = get_stock_name(code)
        return name_cache[code]

    for c, w in target_weights.items():
        op = get_open_price(c, td)
        if op is None or op <= 0:
            continue
        target_val = total_assets * w
        if c in positions:
            cur_val = positions[c]["shares"] * op
            delta = target_val - cur_val
        else:
            cur_val = 0.0
            delta = target_val

        if delta > 0:  # 买入（新仓 或 加仓至目标）
            max_shares = int(delta / op / lot) * lot
            if max_shares < lot:
                continue
            cost = max_shares * op
            fee = calc_fee('buy', op, max_shares)
            if cost + fee <= cash:
                cash -= cost + fee
                existing = positions.get(c)
                if existing:
                    positions[c] = {
                        "shares": existing["shares"] + max_shares,
                        "buy_price": existing.get("buy_price", op),
                        "buy_idx": existing.get("buy_idx", buy_idx_default),
                        "highest_close": max(existing.get("highest_close", op), op),
                        "stop_triggered": existing.get("stop_triggered", False),
                    }
                else:
                    positions[c] = {
                        "shares": max_shares,
                        "buy_price": op, "buy_idx": buy_idx_default,
                        "highest_close": op, "stop_triggered": False,
                    }
                trades.append({"date": td, "action": "BUY", "code": c,
                               "name": _name(c), "price": op,
                               "shares": max_shares, "reason": "sizing_rebalance"})
        elif delta < 0:  # 卖出多余仓位（减仓至目标）
            sell_shares = int((-delta) / op / lot) * lot
            if c not in positions:
                continue
            sell_shares = min(sell_shares, positions[c]["shares"])
            sell_shares = int(sell_shares / lot) * lot
            if sell_shares >= lot:
                proceeds = sell_shares * op
                fee = calc_fee('sell', op, sell_shares)
                cash += proceeds - fee
                positions[c]["shares"] -= sell_shares
                if positions[c]["shares"] <= 0:
                    del positions[c]
                trades.append({"date": td, "action": "SELL", "code": c,
                               "name": _name(c), "price": op,
                               "shares": sell_shares, "reason": "sizing_rebalance"})

    return positions, cash, trades


if __name__ == "__main__":
    # 自测：演示四种方案在同一组持仓上的权重差异
    held = {"A": {"shares": 100, "buy_price": 10.0},
            "B": {"shares": 100, "buy_price": 10.0},
            "C": {"shares": 100, "buy_price": 10.0}}
    open_px = {"A": 12.0, "B": 10.0, "C": 8.5}  # A 盈+20% / B 平 / C 亏-15%
    codes = ["A", "B", "C", "D"]                  # D 为新入选
    pnl = {c: (open_px[c] - held[c]["buy_price"]) / held[c]["buy_price"]
           for c in ["A", "B", "C"]}
    pnl["D"] = 0.0
    for sch in ["equal", "pyramid", "inverted", "martingale"]:
        w = compute_target_weights(sch, codes, pnl=pnl)
        s = "  ".join(f"{c}={w[c]*100:.1f}%" for c in codes)
        print(f"{sch:10s} | {s}  (sum={sum(w.values())*100:.1f}%)")
    print("\n说明: pyramid 给盈利的 A 更高权重；inverted 给 A 更高(更激进)；"
          "martingale 给亏损的 C 更高权重(且受 max_w 封顶)。")
