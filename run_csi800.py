# -*- coding: utf-8 -*-
"""
run_csi800.py —— 波段策略 × 下行开关：中证800股票池批量回测
=============================================================================
目的：把已定死参数的波段策略(N_MOM=20/MIN_SWING=5%/D_CONFIRM=3/K_EXIT=3)
      和下行开关(RULE_A 完整确认回补 / RULE_B 快速回补)放到中证800全部成分股上
      做截面验证——不是挑几只票看个案，而是看【分布】：
        1) 多少比例的股票：开关显著降回撤(>3pp)且年化代价可接受(<3pp) → PASS
        2) 多少比例的股票：年化差<-3pp且回撤无改善 → FAIL(保费白交)
        3) 多少比例的股票：年化差>0 → 反例(需警惕)
      这是对"下行保险"结论的最大样本检验。

不新增任何策略参数；回测口径与 run_swing_trend.py 完全一致：
  t日收盘信号、t+1开盘执行(含滑点)、复权价、涨跌停/停牌顺延、夏普减rf。

数据源（自动探测，两种schema都支持）：
  A) tushare导出库：表 daily(ts_code,trade_date,...) + adj_factor(ts_code,trade_date,adj_factor)
     → 复权价 = 原始价 × adj_factor（与 run_swing_trend.load_symbol 同口径）
  B) 项目库：表 daily_kline(symbol,date,...,adjust='qfq')（已是前复权价）

数据库路径解析顺序：
  1) --db 命令行参数
  2) 本地 config.py 的 DATA["local_db_path"]（与 run_swing_trend.main 相同）
  3) 默认 data/market.db

中证800成分股获取顺序：
  1) --symbols-file CSV文件（每行一个代码，可带名称列）
  2) 库中 index_weight 表（index_code='000906.SH'，取最新快照）
     !!! 幸存者偏差警示：最新快照只含当前成分股，已被调出的股票（往往表现差）
         不在样本内，会高估策略效果。--members union 可用历史并集缓解。
  3) 都没有 → 用 daily 表中全部股票（打印警告）

涨跌停阈值：按代码前缀自动区分（主板9.5%/创业板·科创板19.5%），
  通过设置 run_swing_trend.LIMIT_PCT 实现，不改动任何已有文件。
  ST股(5%)无法从现有表可靠识别，未过滤——原样呈现。

用法：
  py run_csi800.py                        # 全量800只
  py run_csi800.py --max-n 20             # 先跑前20只试管线
  py run_csi800.py --db "D:\\tu-sharedata\\xxx.db"
  py run_csi800.py --members union        # 用历史成分并集(缓解幸存者偏差)
  py run_csi800.py --symbols-file csi800.csv
输出：csi800_results.csv（逐股明细） + 控制台截面汇总
"""
import sys
import os
import sqlite3
import argparse

import numpy as np
import pandas as pd

import run_swing_trend as st
from run_swing_trend import (
    N_MOM, RF_ANN, COMMISSION, SLIPPAGE, INITIAL, START_LOOK,
    compute_conditions, perf, label_regimes,
)
from run_swing_trend import run_swing_backtest, classify_exits
from run_downside_switch import run_switch

# Windows重定向输出时防止编码崩溃（不影响正常打印）
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

INDEX_CODE  = "000906.SH"   # 中证800
MIN_ROWS    = 300           # 预热后最少交易日
WARM        = max(N_MOM, 25)
HIGH_LIMIT_PREFIX = ("300", "301", "688", "689")   # 创业板/科创板 → 19.5%


# ==================== 数据库与schema探测 ====================
def get_db_path(cli_path: str) -> str:
    if cli_path:
        return cli_path
    try:
        import config   # 本地全局配置（与 run_swing_trend.main 同源）
        return config.DATA["local_db_path"]
    except Exception:
        return os.path.join("data", "market.db")


def detect_schema(con) -> str:
    """返回 'tushare'（daily+adj_factor）或 'kline'（daily_kline, 已qfq）。"""
    cur = con.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {r[0] for r in cur.fetchall()}
    if "daily" in tables:
        if "adj_factor" in tables:
            return "tushare"
        raise RuntimeError("有 daily 表但缺 adj_factor，无法复权，请检查数据库")
    if "daily_kline" in tables:
        return "kline"
    raise RuntimeError("数据库中既无 daily 也无 daily_kline 表")


def suffix_of(code: str) -> str:
    """裸代码 → 交易所后缀（tushare 口径）。"""
    c = code.split(".")[0]
    if c.startswith(("6", "9")):
        return ".SH"
    if c.startswith(("0", "2", "3")):
        return ".SZ"
    return ".BJ"


def norm_symbol(code: str, mode: str) -> str:
    """统一代码格式：tushare模式带后缀，kline模式裸代码。"""
    c = code.strip()
    if mode == "tushare":
        return c if "." in c else c + suffix_of(c)
    return c.split(".")[0]


def load_symbol_auto(con, code: str, mode: str) -> pd.DataFrame:
    """自动按schema加载复权OHLC；index统一为YYYYMMDD字符串（perf()依赖此格式）。"""
    if mode == "tushare":
        q = (f"SELECT trade_date, open, high, low, close FROM daily "
             f"WHERE ts_code='{code}' AND trade_date>='{START_LOOK}'")
        px = pd.read_sql(q, con)
        adj = pd.read_sql(
            f"SELECT trade_date, adj_factor FROM adj_factor WHERE ts_code='{code}'", con)
        adj["trade_date"] = adj["trade_date"].astype(str)
        px = px.merge(adj, on="trade_date", how="left")
        px["adj_factor"] = px["adj_factor"].ffill().fillna(1.0)
        for col in ["open", "high", "low", "close"]:
            px[col] = px[col] * px["adj_factor"]
    else:  # kline：已是前复权
        q = (f"SELECT date, open, high, low, close FROM daily_kline "
             f"WHERE symbol='{code}'")
        px = pd.read_sql(q, con).rename(columns={"date": "trade_date"})
    px["trade_date"] = px["trade_date"].astype(str).str.replace("-", "", regex=False)
    px = px.dropna(subset=["close"]).sort_values("trade_date").set_index("trade_date")
    return px[["open", "high", "low", "close"]]


# ==================== 成分股获取 ====================
def get_members(con, mode: str, symbols_file: str, members_mode: str):
    """返回 (代码列表, 说明)。"""
    if symbols_file:
        s = pd.read_csv(symbols_file, dtype=str)
        codes = s.iloc[:, 0].dropna().tolist()
        return codes, f"文件 {symbols_file} ({len(codes)}只)"
    tables = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    if "index_weight" in tables:
        try:
            if members_mode == "union":
                q = (f"SELECT DISTINCT ts_code FROM index_weight "
                     f"WHERE index_code='{INDEX_CODE}'")
                codes = [r[0] for r in con.execute(q).fetchall()]
                return codes, f"index_weight 历史并集 ({len(codes)}只, 缓解幸存者偏差)"
            q = (f"SELECT ts_code FROM index_weight WHERE index_code='{INDEX_CODE}' "
                 f"AND trade_date=(SELECT MAX(trade_date) FROM index_weight "
                 f"WHERE index_code='{INDEX_CODE}')")
            codes = [r[0] for r in con.execute(q).fetchall()]
            if codes:
                return codes, (f"index_weight 最新快照 ({len(codes)}只, "
                               f"!!! 存在幸存者偏差，已调出股票不在样本内)")
        except Exception as e:
            print(f"[warn] index_weight 读取失败({e})，回退全表股票")
    # 回退：全部股票
    if mode == "tushare":
        codes = [r[0] for r in con.execute(
            "SELECT DISTINCT ts_code FROM daily").fetchall()]
    else:
        codes = [r[0] for r in con.execute(
            "SELECT DISTINCT symbol FROM daily_kline").fetchall()]
    return codes, f"!!! 未找到中证800成分表，回退为库中全部股票 ({len(codes)}只)"


# ==================== 单只股票回测 ====================
def set_limit_pct(code: str):
    """按板块设置涨跌停近似阈值（改模块全局，不改文件）。"""
    c = code.split(".")[0]
    st.LIMIT_PCT = 0.195 if c.startswith(HIGH_LIMIT_PREFIX) else 0.095


def run_one(con, code_raw: str, mode: str):
    """跑一只票：持有基准 / 波段策略 / 开关A / 开关B。返回(行dict或None, 原因)。"""
    code = norm_symbol(code_raw, mode)
    try:
        df = load_symbol_auto(con, code, mode)
    except Exception as e:
        return None, f"读取失败:{e}"
    if len(df) < MIN_ROWS + WARM:
        return None, "数据不足"

    set_limit_pct(code)
    cond = compute_conditions(df).iloc[WARM:]
    dates = cond.index
    o = cond["open"].values.astype(float)
    c = cond["close"].values.astype(float)
    if not (o[0] == o[0] and o[0] > 0):
        return None, "首日无行情"

    # 持有基准（净口径，与 run_downside_switch 同）
    sh_h = INITIAL / (o[0] * (1 + SLIPPAGE) * (1 + COMMISSION))
    hold_nav = pd.Series(sh_h * c, index=dates)
    mh = perf(hold_nav)

    # 波段策略（独立择时版，参照）
    swing = run_swing_backtest(cond)
    swing["trades"] = classify_exits(cond, swing["trades"])
    ms = perf(swing["nav_net"])

    # 下行开关两版
    sw = {}
    for rule in ("A", "B"):
        sw[rule] = run_switch(cond, rule)

    row = dict(
        code=code, rows=len(dates), start=dates[0], end=dates[-1],
        hold_cagr=mh["cagr"], hold_mdd=mh["mdd"], hold_sharpe=mh["sharpe"],
        swing_cagr=ms["cagr"], swing_mdd=ms["mdd"], swing_sharpe=ms["sharpe"],
        swing_trades=len(swing["trades"]),
        swing_win=float(np.mean([t["ret_n"] > 0 for t in swing["trades"]]))
        if swing["trades"] else np.nan,
    )
    for rule in ("A", "B"):
        res = sw[rule]
        m = perf(res["nav"])
        row[f"sw{rule}_cagr"] = m["cagr"]
        row[f"sw{rule}_mdd"] = m["mdd"]
        row[f"sw{rule}_sharpe"] = m["sharpe"]
        row[f"sw{rule}_cagr_diff"] = m["cagr"] - mh["cagr"]
        row[f"sw{rule}_mdd_diff"] = m["mdd"] - mh["mdd"]      # 正=回撤更低
        row[f"sw{rule}_tim"] = res["time_in_mkt"]
        hg = res["hedges"]
        row[f"sw{rule}_hedges"] = len(hg)
        ok = [h for h in hg if h["avoided"] < 0]
        row[f"sw{rule}_hedge_ok"] = len(ok)
        row[f"sw{rule}_hedge_ok_ratio"] = len(ok) / len(hg) if hg else np.nan
        # 截面分类（与 run_validation_extension 同标准，互斥）
        dd, ca = row[f"sw{rule}_mdd_diff"], row[f"sw{rule}_cagr_diff"]
        if dd > 0.03 and ca > -0.03:
            row[f"sw{rule}_cls"] = "PASS"
        elif ca < -0.03 and dd <= 0.03:
            row[f"sw{rule}_cls"] = "FAIL"
        elif ca > 0:
            row[f"sw{rule}_cls"] = "反例"
        else:
            row[f"sw{rule}_cls"] = "中性"

    # 跨阶段超额（向量化，供汇总池化）
    regimes = label_regimes(cond)
    r_h = hold_nav.pct_change()
    regime_excess = {}
    for name, nav in [("swing", swing["nav_net"]),
                      ("A", sw["A"]["nav"]), ("B", sw["B"]["nav"])]:
        diff = (nav.pct_change() - r_h).dropna()
        lab = regimes.reindex(diff.index)
        g = diff.groupby(lab)
        regime_excess[name] = (g.sum(), g.count())

    navs = {"hold": hold_nav, "swing": swing["nav_net"],
            "A": sw["A"]["nav"], "B": sw["B"]["nav"]}
    return (row, swing["trades"], regime_excess, navs), None


# ==================== 主流程 ====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None, help="数据库路径(默认读本地config.py)")
    ap.add_argument("--symbols-file", default=None, help="成分股CSV(第一列为代码)")
    ap.add_argument("--members", choices=["latest", "union"], default="latest",
                    help="index_weight取最新快照还是历史并集")
    ap.add_argument("--max-n", type=int, default=0, help="只跑前N只(调试用)")
    args = ap.parse_args()

    db_path = get_db_path(args.db)
    print(f"[db] {db_path}")
    con = sqlite3.connect(db_path)
    mode = detect_schema(con)
    print(f"[schema] {mode}" + ("（daily+adj_factor, 复权计算）"
                                if mode == "tushare" else "（daily_kline, 已qfq）"))

    codes, src_note = get_members(con, mode, args.symbols_file, args.members)
    print(f"[members] {src_note}")
    if args.max_n:
        codes = codes[:args.max_n]
        print(f"[members] --max-n={args.max_n} 截取前 {len(codes)} 只调试")

    rows, all_trades, skip_reasons = [], [], {}
    regime_sum = {k: None for k in ("swing", "A", "B")}
    regime_cnt = {k: None for k in ("swing", "A", "B")}
    nav_rets = {k: [] for k in ("hold", "swing", "A", "B")}   # 每股逐日收益，供组合池化

    for k, code in enumerate(codes, 1):
        out, why = run_one(con, code, mode)
        if out is None:
            skip_reasons[why] = skip_reasons.get(why, 0) + 1
        else:
            row, trades, rex, navs = out
            rows.append(row)
            all_trades.extend(
                dict(code=row["code"], **{kk: t.get(kk) for kk in
                     ("ret_n", "lag", "chase", "exit_cls")}) for t in trades)
            for name in ("swing", "A", "B"):
                s, c_ = rex[name]
                regime_sum[name] = s if regime_sum[name] is None else regime_sum[name].add(s, fill_value=0)
                regime_cnt[name] = c_ if regime_cnt[name] is None else regime_cnt[name].add(c_, fill_value=0)
            for name, nav in navs.items():
                nav_rets[name].append(nav.pct_change().dropna())
        if k % 50 == 0 or k == len(codes):
            print(f"[prog] {k}/{len(codes)}  ok={len(rows)}  skip={k-len(rows)}")

    con.close()
    if not rows:
        print("!!! 无任何股票完成回测，请检查数据库/成分股来源")
        return

    df = pd.DataFrame(rows)
    df.to_csv("csi800_results.csv", index=False, encoding="utf-8-sig")

    # ---------- 截面汇总 ----------
    n = len(df)
    print("\n" + "=" * 72)
    print(f"中证800批量回测截面汇总（{n} 只完成，参数全部沿用已定死值）")
    print("=" * 72)
    if skip_reasons:
        print(f"跳过 {len(codes)-n} 只：", {k: v for k, v in skip_reasons.items()})

    print("\n--- 持有基准截面（截面中位数）---")
    print(f"  年化 {df.hold_cagr.median()*100:6.2f}%   "
          f"回撤 {df.hold_mdd.median()*100:7.2f}%   夏普 {df.hold_sharpe.median():5.2f}")

    print("\n--- 波段策略(独立择时版) vs 持有 ---")
    print(f"  年化差中位数 {(df.swing_cagr-df.hold_cagr).median()*100:+6.2f}pp   "
          f"回撤差中位数 {(df.swing_mdd-df.hold_mdd).median()*100:+6.2f}pp   "
          f"跑赢持有比例 {(df.swing_cagr>df.hold_cagr).mean()*100:5.1f}%")
    if all_trades:
        tr = pd.DataFrame(all_trades)
        print(f"  交易池 {len(tr)} 笔   胜率 {(tr.ret_n>0).mean()*100:.1f}%   "
              f"平均净收益 {tr.ret_n.mean()*100:+.2f}%   "
              f"平均确认滞后 {tr.lag.mean()*100:.1f}%   平均追高 {tr.chase.mean()*100:.1f}%")

    for rule in ("A", "B"):
        print(f"\n--- 下行开关 RULE_{rule}"
              f"（{'up_cond确认3日回补' if rule=='A' else '动量>0且站上20日线回补'}）vs 持有 ---")
        print(f"  年化差中位数 {df[f'sw{rule}_cagr_diff'].median()*100:+6.2f}pp   "
              f"回撤差中位数 {df[f'sw{rule}_mdd_diff'].median()*100:+6.2f}pp(正=更好)")
        print(f"  回撤显著改善(>3pp)比例 {(df[f'sw{rule}_mdd_diff']>0.03).mean()*100:5.1f}%   "
              f"在场比例中位数 {df[f'sw{rule}_tim'].median()*100:5.1f}%   "
              f"避险成功比例中位数 {df[f'sw{rule}_hedge_ok_ratio'].median()*100:5.1f}%")
        vc = df[f"sw{rule}_cls"].value_counts()
        print(f"  判定: " + "  ".join(
            f"{k} {v}/{n}({v/n*100:.0f}%)" for k, v in vc.items()))

    # 跨阶段池化超额
    print("\n--- 跨阶段池化年化超额(pp, 相对一直持有) ---")
    print(f"{'阶段':>6s}{'天数':>10s}{'波段':>10s}{'开关A':>10s}{'开关B':>10s}")
    for st_name in ["上升段", "下降段", "震荡段", "转向段", "混合"]:
        line = f"{st_name:>6s}"
        tot_cnt = 0
        vals = []
        for name in ("swing", "A", "B"):
            if regime_sum[name] is not None and st_name in regime_sum[name].index:
                s = regime_sum[name][st_name]
                c_ = regime_cnt[name][st_name]
                vals.append(f"{s/c_*252*100:+9.2f}pp")
                tot_cnt = max(tot_cnt, int(c_))
            else:
                vals.append(f"{'--':>10s}")
        if tot_cnt:
            print(line + f"{tot_cnt:>10d}" + "".join(vals))

    # ---------- 组合层面：逐日收益按当日有数据的股票等权平均，每日再平衡 ----------
    print("\n--- 组合层面（等权每日再平衡，逐日池化重算）---")
    port = {}
    n_stock = None
    for name in ("hold", "swing", "A", "B"):
        rets = pd.concat(nav_rets[name], axis=1).sort_index()   # 日期 × 个股，NaN=当日无数据/未入场（必须排序，否则cumprod错乱）
        port_ret = rets.mean(axis=1)                       # 等权、只对当日有数据的股票求均值
        nav = (1 + port_ret).cumprod()
        port[name] = perf(nav)
        if n_stock is None:
            n_stock = rets.notna().sum(axis=1)
    cov = n_stock[n_stock > 0]
    print(f"  池化区间 {cov.index.min()}~{cov.index.max()}   "
          f"日均在池股票数 中位 {int(cov.median())} / 最少 {int(cov.min())} / 最多 {int(cov.max())}")
    labels = {"hold": "持有基准", "swing": "波段策略", "A": "开关A", "B": "开关B"}
    print(f"  {'':6s}{'年化':>10s}{'回撤':>10s}{'夏普':>8s}{'年化差':>10s}{'回撤差':>10s}")
    for name in ("hold", "swing", "A", "B"):
        m = port[name]
        d_ca = (m["cagr"] - port["hold"]["cagr"]) * 100
        d_dd = (m["mdd"] - port["hold"]["mdd"]) * 100      # 正=回撤更浅
        print(f"  {labels[name]:6s}{m['cagr']*100:>9.2f}%{m['mdd']*100:>9.2f}%"
              f"{m['sharpe']:>8.2f}{d_ca:>+9.2f}pp{d_dd:>+9.2f}pp(正=更好)")

    print(f"\n[save] csi800_results.csv（逐股明细 {n} 行）")
    print("\n!!! 已知口径限制（原样声明，不藏）：")
    print("  1) 成分股若取最新快照 → 幸存者偏差（已调出股票不在样本，结果偏乐观）")
    print("  2) ST股未过滤（5%涨跌停近似误差）")
    print("  3) 各股区间起点=各自预热完成日，年化为各自区间口径")


if __name__ == "__main__":
    main()
