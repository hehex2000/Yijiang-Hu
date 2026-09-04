#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
补全【红利类指数】时点成分权重 —— 用 .CSI 后缀下载, 映射回库里 .SH 代码
============================================================================
根因(2026-08-31 发现):
  红利类指数在 Tushare index_weight 接口的**代码后缀是 .CSI, 不是 .SH**:
      000922.SH → 0 行   (此前据此误判"Tushare 无红利指数成分")
      000922.CSI → 100 行 ✓
      930955.SH → 0 行
      930955.CSI → 100 行 ✓
  → 此前"红利指数无成分数据、无法自建全收益"的结论是后缀错误造成的假象。

用途:
  补全后可支撑: 红利类股票池(universe)的时点构建、红利指数成分分析。
  注: 全收益基准已不需要它 —— 官方全收益/净收益指数已由 download_tr_index.py 下载。

用法:
  ./venv_ml/Scripts/python.exe backfill_dividend_constituents.py
  ./venv_ml/Scripts/python.exe backfill_dividend_constituents.py --start 201001
依赖: tushare, config.DATA.tushare_token / local_db_path
"""
import sys, os, time, sqlite3, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA

DB = DATA.get("local_db_path", r"D:\tu-shareData\astock_daily.db")
TOKEN = DATA.get("tushare_token", "")

# (库内代码, Tushare下载代码, 名称)
MAP = [
    ("000922.SH", "000922.CSI", "中证红利"),
    ("930955.SH", "930955.CSI", "红利低波100"),
    ("000015.SH", "000015.SH", "上证红利"),
    ("399324.SZ", "399324.SZ", "深证红利"),
    # 2026-09-01 新增：中证全指同样是 .CSI 才有数据（000985.SH → 0 行）。
    #   此前 `backfill_index_constituent.py` 里"000985 已知无 index_weight 数据"的注释
    #   是**后缀错误造成的假象**（与 000922/930955 同一个坑）。
    #   补全它才能自建 000985 全收益 → 让"全A池"策略的 hfq 超额有正确基准。
    ("000985.SH", "000985.CSI", "中证全指"),
]


def get_pro():
    import tushare as ts
    if not TOKEN:
        print("[错误] config.DATA.tushare_token 为空")
        sys.exit(1)
    ts.set_token(TOKEN)
    return ts.pro_api()


def fetch_month(pro, code, ms, me, attempt=0):
    try:
        df = pro.index_weight(index_code=code, start_date=ms, end_date=me)
    except Exception as e:
        msg = str(e)
        if ("频率" in msg or "每分钟" in msg or "rate" in msg.lower()) and attempt < 4:
            print(f"    [限速] {code} {ms}: 退避 60s 重试({attempt+1})")
            time.sleep(60)
            return fetch_month(pro, code, ms, me, attempt + 1)
        print(f"    [跳过] {code} {ms}: {msg[:70]}")
        return []
    if df is None or df.empty:
        return []
    out = []
    for _, r in df.iterrows():
        try:
            out.append((str(r["con_code"]), str(r["trade_date"]), float(r["weight"])))
        except Exception:
            continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="201001", help="起始 YYYYMM")
    ap.add_argument("--end", default="", help="结束 YYYYMM, 默认当月")
    ap.add_argument("--only", default="", help="只补某个库内代码，如 000985.SH")
    args = ap.parse_args()

    end_ym = args.end if args.end else time.strftime("%Y%m")
    start_ym = args.start if len(args.start) == 6 else args.start + "01"
    end_ym = end_ym if len(end_ym) == 6 else end_ym + "12"

    conn = sqlite3.connect(DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS index_constituent (
        index_code TEXT, ts_code TEXT, trade_date TEXT,
        weight REAL, index_name TEXT,
        UNIQUE(ts_code, index_code, trade_date))""")
    conn.commit()
    pro = get_pro()

    print("=" * 78)
    print(f"补全红利类指数时点成分  {start_ym}~{end_ym}")
    print("  (Tushare 侧用 .CSI, 落库映射为库内代码)")
    print("=" * 78)

    for local, remote, name in MAP:
        if args.only and local != args.only:
            continue
        sy, sm = int(start_ym[:4]), int(start_ym[4:6])
        ey, em = int(end_ym[:4]), int(end_ym[4:6])
        total, snaps, empty = 0, set(), 0
        print(f"\n=== {local} {name}  (下载代码 {remote}) ===")
        for y in range(sy, ey + 1):
            for m in range(1, 13):
                if (y == sy and m < sm) or (y == ey and m > em):
                    continue
                ms = f"{y}{m:02d}01"
                me = f"{y+1}0101" if m == 12 else f"{y}{m+1:02d}01"
                rows = fetch_month(pro, remote, ms, me)
                if not rows:
                    empty += 1
                    time.sleep(0.2)
                    continue
                conn.executemany(
                    "INSERT OR REPLACE INTO index_constituent "
                    "(ts_code, index_code, index_name, trade_date, weight) VALUES (?,?,?,?,?)",
                    [(c, local, name, d, w) for c, d, w in rows])
                conn.commit()
                total += len(rows)
                snaps.update(d for _, d, _ in rows)
                time.sleep(0.25)
        nd = conn.execute("SELECT COUNT(DISTINCT trade_date) FROM index_constituent "
                          "WHERE index_code=?", (local,)).fetchone()[0]
        rng = conn.execute("SELECT MIN(trade_date),MAX(trade_date) FROM index_constituent "
                           "WHERE index_code=?", (local,)).fetchone()
        print(f"  写入 {total} 行 / {len(snaps)} 个新快照 (空月 {empty})")
        print(f"  库内快照总数 {nd}  区间 {rng[0]}~{rng[1]}")

    print("\n" + "=" * 78)
    print("各指数成分覆盖")
    print("=" * 78)
    for r in conn.execute(
            "SELECT index_code, COUNT(DISTINCT trade_date) nd, MIN(trade_date), MAX(trade_date) "
            "FROM index_constituent GROUP BY index_code ORDER BY nd DESC"):
        print(f"  {r[0]:12s} 快照 {r[1]:4d}   {r[2]}~{r[3]}")
    conn.close()


if __name__ == "__main__":
    main()
