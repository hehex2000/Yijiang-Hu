# -*- coding: utf-8 -*-
"""
指数 / ETF 涨跌一览（回测周期表现）
====================================

统计主要宽基 / 风格指数在回测区间的涨跌情况（长期买入 ETF 视角）：
    - 以对应 ETF 为展示标的（现价 = 当前买入价）；
    - 若有对应指数，用指数干净收益路径校正 ETF 长期涨跌（ETF 现价 × 指数区间涨跌），
      消除 ETF 份额拆分与 etf_daily 数据偏差；
    - 无对应指数时，回退到 ETF 自身 pct_chg 复权序列；
    - 该指数完全无 ETF 时，兜底用指数本身。

用法:
    python show_index_etf_changes.py START END [--no-html]
    python show_index_etf_changes.py            # 使用 config.py 的回测区间

示例:
    python show_index_etf_changes.py 20260201 20260703
"""

import os
import sys
import sqlite3
import argparse

# ── 数据库路径（与 config.py / download_etf_data.py 保持一致）────────────
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config  # noqa
    DB_PATH = config.DATA.get("local_db_path", r"D:/tu-shareData/astock_daily.db")
except Exception:
    DB_PATH = r"D:/tu-shareData/astock_daily.db"

# ── 主要指数 → ETF 映射 ───────────────────────────────────────────────
# 顺序即展示顺序。etf_code 为 None 时强制使用指数本身。
# 修改此列表即可增删指数 / 调整 ETF 对应关系。
INDEX_MAP = [
    # 名称,           指数代码,      ETF代码,        ETF名称,          分类
    ("沪深300",      "000300.SH",  "510300.SH",  "沪深300ETF",     "宽基·大盘"),
    ("上证50",       "000016.SH",  "510050.SH",  "上证50ETF",      "宽基·超大盘"),
    ("中证800",      "000906.SH",  "515800.SH",  "中证800ETF",     "宽基·大中盘"),
    ("中证500",      "000905.SH",  "510500.SH",  "中证500ETF",     "宽基·中盘"),
    ("中证1000",     "000852.SH",  "512100.SH",  "中证1000ETF",    "宽基·小盘"),
    ("中证2000",     "932000.SH",  "563300.SH",  "中证2000ETF",    "宽基·微盘"),
    ("上证指数",     "000001.SH",  "510210.SH",  "上证指数ETF",    "综合"),
    ("深证成指",     "399001.SZ",  "159903.SZ",  "深成ETF",        "综合"),
    ("中证全指",     "000985.SH",  None,         "（无ETF·用指数）","全市场"),
    ("创业板指",     "399006.SZ",  "159915.SZ",  "创业板ETF",      "成长·创业板"),
    ("创业板50",     "399673.SZ",  "159949.SZ",  "创业板50ETF",    "成长·创业板龙头"),
    ("科创50",       "000688.SH",  "588000.SH",  "科创50ETF",      "成长·科创板"),
    ("科创板指(科创100)", "000698.SH", "588190.SH", "科创100ETF",  "成长·科创板中盘"),
]

TRADING_DAYS = 252


# ── 查询辅助 ──────────────────────────────────────────────────────────

def _has_code(conn, table, code):
    row = conn.execute(
        f"SELECT 1 FROM {table} WHERE ts_code = ? LIMIT 1", (code,)
    ).fetchone()
    return row is not None


def _query_index_series(conn, code, start, end):
    """返回指数区间逐日 [date, close, high, low]（升序）。"""
    sql = """
        SELECT CAST(trade_date AS TEXT), close, high, low
        FROM index_daily
        WHERE ts_code = ?
          AND CAST(trade_date AS TEXT) >= ?
          AND CAST(trade_date AS TEXT) <= ?
        ORDER BY trade_date ASC
    """
    out = []
    for r in conn.execute(sql, (code, start, end)).fetchall():
        d, close, high, low = r
        if close is None:
            continue
        out.append((d, close, high, low))
    return out


def _query_etf_series(conn, code, start, end):
    """返回 ETF 区间逐日 [date, close, high, low, ret]（升序）。

    ret 取 pct_chg（已复权日收益率），可消除份额拆分造成的断点。
    """
    sql = """
        SELECT CAST(trade_date AS TEXT), close, high, low, pre_close, pct_chg
        FROM etf_daily
        WHERE ts_code = ?
          AND CAST(trade_date AS TEXT) >= ?
          AND CAST(trade_date AS TEXT) <= ?
        ORDER BY trade_date ASC
    """
    out = []
    for r in conn.execute(sql, (code, start, end)).fetchall():
        d, close, high, low, pre, pct = r
        if close is None:
            continue
        if pct is not None:
            ret = pct / 100.0
        elif pre:
            ret = (close / pre - 1.0)
        else:
            ret = 0.0
        out.append({"date": d, "close": close, "high": high,
                    "low": low, "ret": ret})
    return out


def _latest_etf_close(conn, code, end):
    """取 ETF 在 end（含）之前最近一个交易日的真实收盘价（即当前买入价）。"""
    row = conn.execute(
        """SELECT close FROM etf_daily
           WHERE ts_code = ? AND CAST(trade_date AS TEXT) <= ?
           ORDER BY trade_date DESC LIMIT 1""",
        (code, end),
    ).fetchone()
    return row[0] if row else None


def _pack(name, code, kind, ref, start_dt, start_close, end_dt, end_close,
          pct, amplitude, max_dd, ann, n_days):
    return {
        "name": name,
        "code": code,
        "kind": kind,
        "ref": ref,
        "start_dt": start_dt,
        "start_close": start_close,
        "end_dt": end_dt,
        "end_close": end_close,
        "pct": pct,
        "amplitude": amplitude,
        "max_dd": max_dd,
        "ann": ann,
        "n_days": n_days,
    }


def _max_drawdown(seq):
    run_max = -1e18
    max_dd = 0.0
    for c in seq:
        run_max = max(run_max, c)
        dd = c / run_max - 1.0
        max_dd = min(max_dd, dd)
    return max_dd * 100.0


def compute_one(conn, name, index_code, etf_code, start, end):
    """计算单个指数/ETF 在 [start, end] 的表现。

    设计（适配“长期买入指数 ETF 做投资”场景）：
      - 优先以 ETF 作为展示标的（国内通过买 ETF 实现被动指数投资）；
      - 若对应指数存在，用「指数干净收益率路径」校正 ETF 长期涨跌
        （ETF 现价 × 指数区间涨跌），消除 ETF 份额拆分与 etf_daily 数据偏差；
      - 若指数不存在，则回退到 ETF 自身 pct_chg 复权序列；
      - 若该指数完全无 ETF，则兜底用指数本身（纯指数）。
    """
    etf_avail = bool(etf_code) and _has_code(conn, "etf_daily", etf_code)
    idx_avail = _has_code(conn, "index_daily", index_code)

    # ── 情形 A：有 ETF 且有指数 → ETF 现价 × 指数收益路径校正 ──
    if etf_avail and idx_avail:
        etf_end = _latest_etf_close(conn, etf_code, end)
        if etf_end is None:
            return None
        idx_rows = _query_index_series(conn, index_code, start, end)
        if len(idx_rows) < 2:
            return None
        idx_closes = [r[1] for r in idx_rows]
        dates = [r[0] for r in idx_rows]
        ratio = [c / idx_closes[0] for c in idx_closes]   # 指数累积收益
        end_ratio = ratio[-1]
        if end_ratio <= 0:
            return None
        # 以 ETF 现价为锚，重建 ETF 后复权价格路径（= 指数收益 × 现价）
        scale = etf_end / end_ratio
        adj = [scale * x for x in ratio]

        start_dt, start_close = dates[0], adj[0]
        end_dt, end_close = dates[-1], adj[-1]            # == etf_end
        pct = (end_ratio - 1.0) * 100.0

        hi = max(r[2] for r in idx_rows if r[2] is not None)
        lo = min(r[3] for r in idx_rows if r[3] is not None)
        amplitude = (hi / lo - 1.0) * 100.0 if lo else 0.0

        years = len(adj) / TRADING_DAYS
        ann = ((end_ratio) ** (1.0 / years) - 1.0) * 100.0 if years > 0 else 0.0

        return _pack(name, etf_code, "ETF(指数校正)", index_code,
                     start_dt, start_close, end_dt, end_close,
                     pct, amplitude, _max_drawdown(adj), ann, len(adj))

    # ── 情形 B：有 ETF 但无对应指数 → ETF 自身 pct_chg 复权序列 ──
    if etf_avail:
        rows = _query_etf_series(conn, etf_code, start, end)
        if len(rows) < 2:
            return None
        n = len(rows)
        adj = [0.0] * n
        adj_h = [0.0] * n
        adj_l = [0.0] * n
        adj[-1] = rows[-1]["close"]
        for i in range(n - 2, -1, -1):
            factor = 1.0 + rows[i + 1]["ret"]
            if factor <= 0:
                factor = 1.0
            adj[i] = adj[i + 1] / factor
        for i in range(n):
            m = (adj[i] / rows[i]["close"]) if rows[i]["close"] else 1.0
            adj_h[i] = (rows[i]["high"] * m) if rows[i]["high"] is not None else adj[i]
            adj_l[i] = (rows[i]["low"] * m) if rows[i]["low"] is not None else adj[i]

        start_dt, start_close = rows[0]["date"], adj[0]
        end_dt, end_close = rows[-1]["date"], adj[-1]
        pct = (end_close / start_close - 1.0) * 100.0
        hi = max(adj_h); lo = min(adj_l)
        amplitude = (hi / lo - 1.0) * 100.0 if lo else 0.0
        years = n / TRADING_DAYS
        ann = ((end_close / start_close) ** (1.0 / years) - 1.0) * 100.0 if years > 0 else 0.0
        return _pack(name, etf_code, "ETF", etf_code,
                     start_dt, start_close, end_dt, end_close,
                     pct, amplitude, _max_drawdown(adj), ann, n)

    # ── 情形 C：无 ETF（etf_code 为空）→ 纯指数兜底 ──
    if idx_avail:
        idx_rows = _query_index_series(conn, index_code, start, end)
        if len(idx_rows) < 2:
            return None
        closes = [r[1] for r in idx_rows]
        dates = [r[0] for r in idx_rows]
        start_dt, start_close = dates[0], closes[0]
        end_dt, end_close = dates[-1], closes[-1]
        pct = (end_close / start_close - 1.0) * 100.0
        hi = max(r[2] for r in idx_rows if r[2] is not None)
        lo = min(r[3] for r in idx_rows if r[3] is not None)
        amplitude = (hi / lo - 1.0) * 100.0 if lo else 0.0
        years = len(closes) / TRADING_DAYS
        ann = ((end_close / start_close) ** (1.0 / years) - 1.0) * 100.0 if years > 0 else 0.0
        return _pack(name, index_code, "指数", index_code,
                     start_dt, start_close, end_dt, end_close,
                     pct, amplitude, _max_drawdown(closes), ann, len(closes))

    return None


# ── 控制台表格 ──────────────────────────────────────────────────────────

def _color(text, pct, tty):
    if not tty:
        return text
    code = "31" if pct >= 0 else "32"  # 红涨绿跌（A股习惯）
    return f"\033[{code}m{text}\033[0m"


def print_console(results, start, end):
    tty = sys.stdout.isatty()
    ups = [r for r in results if r["pct"] >= 0]
    downs = [r for r in results if r["pct"] < 0]

    print()
    print("=" * 82)
    print(f"  指数 / ETF 涨跌一览（长期买入 ETF 视角）   区间: {start} ~ {end}")
    print("  以 ETF 为标的(现价=当前买入价)；长期涨跌用对应指数收益路径校正")
    print("  （ETF现价 × 指数区间涨跌，消除 ETF 份额拆分与数据偏差）；无指数时回退 ETF 复权")
    print("=" * 82)
    print(f"  {'指数':<16}{'标的':<11}{'类型':<14}"
          f"{'起(复权)→现价':<22}{'区间涨跌':>9}{'振幅':>8}{'最大回撤':>9}{'年化':>8}")
    print("-" * 82)
    for i, r in enumerate(results, 1):
        arrow = "▲" if r["pct"] >= 0 else "▼"
        arrow_colored = _color(arrow, r["pct"], tty)
        pct_str = _color(f"{r['pct']:+7.2f}%", r["pct"], tty)
        kind_disp = (f"ETF↔{r['ref']}" if r["kind"] == "ETF(指数校正)"
                     else r["kind"])
        line = (f"  {i:>2}.{r['name']:<13}{r['code']:<11}{kind_disp:<14}"
                f"{r['start_close']:>9.2f}→{r['end_close']:>9.2f} "
                f"{arrow_colored}{pct_str:>7}{r['amplitude']:>7.1f}%"
                f"{r['max_dd']:>8.1f}%{r['ann']:>7.1f}%")
        print(line)
    print("-" * 78)
    if ups or downs:
        best = max(results, key=lambda x: x["pct"])
        worst = min(results, key=lambda x: x["pct"])
        print(f"  上涨 {len(ups)} / 下跌 {len(downs)} 只")
        print(f"  最强: {best['name']} ({best['pct']:+.2f}%)   "
              f"最弱: {worst['name']} ({worst['pct']:+.2f}%)")
    print("=" * 78)
    print()


# ── HTML 报告 ───────────────────────────────────────────────────────────

def _bar_html(pct):
    width = min(abs(pct) * 1.2, 45)  # 缩放
    color = "#e23b3b" if pct >= 0 else "#1aa260"  # 红涨绿跌
    if pct >= 0:
        return (f'<div class="bar-pos" style="width:{width:.1f}px;'
                f'background:{color}"></div>')
    return (f'<div style="width:{45-width:.1f}px;display:inline-block"></div>'
            f'<div class="bar-neg" style="width:{width:.1f}px;'
            f'background:{color}"></div>')


def write_html(results, start, end, out_path):
    rows = []
    for r in results:
        sign = "+" if r["pct"] >= 0 else ""
        if r["kind"] == "ETF(指数校正)":
            tag = f'<span class="tag" title="基准指数 {r["ref"]}">ETF·校正</span>'
        else:
            tag = f'<span class="tag">{r["kind"]}</span>'
        rows.append(f"""
        <tr>
          <td class="name">{r['name']}</td>
          <td>{r['code']}</td>
          <td>{tag}</td>
          <td>{r['start_dt']}</td>
          <td class="num">{r['start_close']:.2f}</td>
          <td>{r['end_dt']}</td>
          <td class="num">{r['end_close']:.2f}</td>
          <td class="num pct">{sign}{r['pct']:.2f}%</td>
          <td class="chart">{_bar_html(r['pct'])}<span class="chv">{sign}{r['pct']:.2f}%</span></td>
          <td class="num">{r['amplitude']:.1f}%</td>
          <td class="num">{r['max_dd']:.1f}%</td>
          <td class="num">{r['ann']:.1f}%</td>
        </tr>""")

    ups = sum(1 for r in results if r["pct"] >= 0)
    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>指数/ETF涨跌一览 {start}~{end}</title>
<style>
  body {{ font-family: -apple-system, "Microsoft YaHei", sans-serif;
         background:#f5f6f8; color:#222; margin:0; padding:24px; }}
  h1 {{ font-size:20px; margin:0 0 4px; }}
  .meta {{ color:#666; font-size:13px; margin-bottom:16px; }}
  .summary {{ display:flex; gap:12px; margin-bottom:18px; }}
  .card {{ background:#fff; border-radius:10px; padding:12px 16px;
          box-shadow:0 1px 3px rgba(0,0,0,.08); flex:1; }}
  .card .k {{ font-size:12px; color:#888; }}
  .card .v {{ font-size:22px; font-weight:700; margin-top:4px; }}
  .up {{ color:#e23b3b; }} .down {{ color:#1aa260; }}
  table {{ border-collapse:collapse; width:100%; background:#fff;
          border-radius:10px; overflow:hidden; box-shadow:0 1px 3px rgba(0,0,0,.08); }}
  th,td {{ padding:9px 12px; text-align:left; font-size:13px;
          border-bottom:1px solid #eef0f2; }}
  th {{ background:#fafbfc; color:#555; font-weight:600; }}
  td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  td.name {{ font-weight:600; }}
  .pct {{ font-weight:700; }}
  .tag {{ display:inline-block; padding:1px 7px; border-radius:6px;
         font-size:11px; background:#eef2ff; color:#3a5; }}
  .chart {{ width:220px; }}
  .bar-pos,.bar-neg {{ height:14px; border-radius:3px; display:inline-block;
           vertical-align:middle; }}
  .chv {{ font-size:12px; margin-left:6px; color:#555; }}
  tr:hover {{ background:#fafcff; }}
</style></head>
<body>
  <h1>指数 / ETF 涨跌一览</h1>
  <div class="meta">回测区间 {start} ~ {end} ｜ 以 ETF 为标的（现价=当前买入价），长期涨跌用对应指数收益路径校正（ETF 现价 × 指数区间涨跌，消除 ETF 份额拆分与数据偏差）；无对应指数时回退 ETF 复权序列 ｜ 红涨绿跌</div>
  <div class="summary">
    <div class="card"><div class="k">覆盖指数</div><div class="v">{len(results)}</div></div>
    <div class="card"><div class="k">上涨 / 下跌</div>
        <div class="v"><span class="up">{ups}</span> / <span class="down">{len(results)-ups}</span></div></div>
    <div class="card"><div class="k">最强</div><div class="v up">{max(results,key=lambda x:x['pct'])['name']} {max(results,key=lambda x:x['pct'])['pct']:+.2f}%</div></div>
    <div class="card"><div class="k">最弱</div><div class="v down">{min(results,key=lambda x:x['pct'])['name']} {min(results,key=lambda x:x['pct'])['pct']:+.2f}%</div></div>
  </div>
  <table>
    <thead><tr>
      <th>指数</th><th>标的(ETF)</th><th>类型</th><th>起始日</th><th>起始(复权)</th>
      <th>结束日</th><th>现价(ETF)</th><th>区间涨跌</th><th>涨跌分布</th>
      <th>振幅</th><th>最大回撤</th><th>年化</th>
    </tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body></html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


# ── 主流程 ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="指数/ETF涨跌一览")
    ap.add_argument("start", nargs="?", default=None, help="开始日期 YYYYMMDD")
    ap.add_argument("end", nargs="?", default=None, help="结束日期 YYYYMMDD")
    ap.add_argument("--no-html", action="store_true", help="不生成HTML报告")
    args = ap.parse_args()

    if not args.start or not args.end:
        # 回退：尝试读取 config.py 的回测区间
        try:
            import config  # noqa
            args.start = args.start or config.GLOBAL["backtest_start"]
            args.end = args.end or config.GLOBAL["backtest_end"]
        except Exception:
            args.start = args.start or "20240101"
            args.end = args.end or "20260703"

    start, end = args.start, args.end
    print(f"[*] 区间 {start} ~ {end}，数据库: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    results = []
    missing = []
    for name, idx_code, etf_code, etf_name, _cat in INDEX_MAP:
        try:
            r = compute_one(conn, name, idx_code, etf_code, start, end)
        except Exception as e:
            r = None
            print(f"    [WARN] {name}: {e}")
        if r is None:
            missing.append(name)
            continue
        results.append(r)
    conn.close()

    if not results:
        print("[!] 区间内没有可用数据，请检查日期或数据库。")
        return

    # 按区间涨跌幅从高到低排序（最强在前）
    results.sort(key=lambda x: x["pct"], reverse=True)

    print_console(results, start, end)

    if missing:
        print(f"  [提示] 以下指数在区间内无数据，已跳过: {', '.join(missing)}")

    if not args.no_html:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"index_etf_changes_{start}_{end}.html")
        write_html(results, start, end, out_path)
        print(f"  [HTML报告] {out_path}")


if __name__ == "__main__":
    main()
