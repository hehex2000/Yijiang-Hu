#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 Tushare 下载【官方全收益/净收益指数】日线, 并与自建版本交叉验证
====================================================================
重大发现(2026-08-31):
  中证指数在 Tushare 的**全收益**系列代码与价格指数不同, 直接用价格代码查会返回空,
  造成"Tushare 无数据"的假象。已实测确认:

    000300.SH 沪深300(价格)  → H00300.CSI  300收益     (沪深300全收益)
    000906.SH 中证800(价格)  → H00906.CSI  800收益     (中证800全收益)
    000922.SH 中证红利(价格) → H00922.CSI  中红收益    (中证红利全收益)
    930955.SH 红利低波100    → H20955.CSI  红利低波100全收益
    000922.SH               → 000922CNY020.CSI 中证红利净收益 (已扣红利税!)

  另: 红利类指数在 index_weight 接口必须用 **.CSI 后缀** (000922.SH 返回 0 行,
      000922.CSI 返回 100 行) —— 这是此前"红利指数无成分数据"的根因。

用途:
  1) 官方全收益指数可作为基准的**权威口径**, 校验/替代 build_tr_index.py 自建版本
  2) 净收益指数(扣红利税)可补上"个人投资者实际到手"口径, 解决视频未扣税的问题

用法:
  ./venv_ml/Scripts/python.exe download_tr_index.py
  ./venv_ml/Scripts/python.exe download_tr_index.py --start 20100101 --end 20260828
依赖: tushare, config.DATA.tushare_token / local_db_path
"""
import sys, os, time, sqlite3, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA

DB = DATA.get("local_db_path", r"D:\tu-shareData\astock_daily.db")
TOKEN = DATA.get("tushare_token", "")

# (本地价格指数代码, Tushare全收益/净收益代码, 名称)
DOWNLOADS = [
    ("000300.SH", "H00300.CSI", "沪深300全收益"),
    ("000906.SH", "H00906.CSI", "中证800全收益"),
    ("000922.SH", "H00922.CSI", "中证红利全收益"),
    ("000922.SH", "000922CNY020.CSI", "中证红利净收益(扣税)"),
    ("930955.SH", "H20955.CSI", "红利低波100全收益"),
]


def get_pro():
    import tushare as ts
    if not TOKEN:
        print("[错误] config.DATA.tushare_token 为空")
        sys.exit(1)
    ts.set_token(TOKEN)
    return ts.pro_api()


def ensure_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS index_tr_official (
            local_code TEXT, tr_code TEXT, idx_name TEXT,
            trade_date TEXT, close REAL,
            PRIMARY KEY (tr_code, trade_date)
        )""")
    conn.commit()


def download(pro, conn, local_code, tr_code, name, start, end):
    total = 0
    y0, y1 = int(start[:4]), int(end[:4])
    for y in range(y0, y1 + 1):
        s = f"{y}0101" if y == y0 else f"{y}0101"
        e = f"{y}1231" if y < y1 else end
        try:
            df = pro.index_daily(ts_code=tr_code, start_date=s, end_date=e)
        except Exception as ex:
            msg = str(ex)
            if ("频率" in msg or "每分钟" in msg) :
                print(f"    [限速] {tr_code} {y}: 退避 60s")
                time.sleep(60)
                try:
                    df = pro.index_daily(ts_code=tr_code, start_date=s, end_date=e)
                except Exception as ex2:
                    print(f"    [跳过] {tr_code} {y}: {str(ex2)[:60]}")
                    continue
            else:
                print(f"    [跳过] {tr_code} {y}: {msg[:60]}")
                continue
        if df is None or df.empty:
            time.sleep(0.25)
            continue
        rows = []
        for _, r in df.iterrows():
            try:
                rows.append((local_code, tr_code, name, str(r["trade_date"]),
                             float(r["close"])))
            except Exception:
                continue
        conn.executemany(
            "INSERT OR REPLACE INTO index_tr_official "
            "(local_code, tr_code, idx_name, trade_date, close) VALUES (?,?,?,?,?)", rows)
        conn.commit()
        total += len(rows)
        time.sleep(0.28)
    n = conn.execute("SELECT COUNT(*) FROM index_tr_official WHERE tr_code=?",
                     (tr_code,)).fetchone()[0]
    rng = conn.execute("SELECT MIN(trade_date),MAX(trade_date) FROM index_tr_official "
                       "WHERE tr_code=?", (tr_code,)).fetchone()
    print(f"  [完成] {tr_code:20s} {name:22s} 累计 {n:5d} 行  {rng[0]}~{rng[1]}")
    return n


def cross_validate(conn):
    """官方全收益 vs 自建全收益 —— 独立交叉验证"""
    print("\n" + "=" * 86)
    print("交叉验证: 官方全收益指数(下载)  vs  自建全收益指数(build_tr_index.py)")
    print("=" * 86)
    for local, tr_code, name in DOWNLOADS:
        if "净收益" in name:
            continue
        q = """SELECT o.trade_date, o.close AS off_tr, b.idx_tr AS self_tr
               FROM index_tr_official o
               JOIN index_total_return b
                 ON b.trade_date=o.trade_date AND b.index_code=o.local_code
               WHERE o.tr_code=? ORDER BY o.trade_date"""
        df = pd.read_sql_query(q, conn, params=(tr_code,))
        if len(df) < 60:
            print(f"  {local} {name}: 重叠样本仅 {len(df)} 天, 跳过")
            continue
        df = df.dropna()
        yrs = len(df) / 252.0
        a_off = (df["off_tr"].iloc[-1] / df["off_tr"].iloc[0]) ** (1 / yrs) - 1
        a_self = (df["self_tr"].iloc[-1] / df["self_tr"].iloc[0]) ** (1 / yrs) - 1
        corr = df["off_tr"].pct_change().corr(df["self_tr"].pct_change())
        print(f"\n  {local}  {name}   ({df['trade_date'].iloc[0]}~{df['trade_date'].iloc[-1]}, {yrs:.1f}年)")
        print(f"    官方全收益年化 : {a_off*100:+.2f}%")
        print(f"    自建全收益年化 : {a_self*100:+.2f}%")
        print(f"    年化差异       : {(a_self-a_off)*100:+.3f}pp    日收益相关性 {corr:.4f}")
        if abs(a_self - a_off) < 0.008:
            print("    ✓ 两者吻合(差异<0.8pp): 自建方法可靠")
        else:
            print("    ⚠️ 差异偏大, 需排查自建口径")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20100101")
    ap.add_argument("--end", default="20260828")
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args()

    import pandas as pd
    globals()["pd"] = pd

    conn = sqlite3.connect(DB)
    ensure_table(conn)
    pro = get_pro()
    print("=" * 86)
    print(f"下载 Tushare 官方全收益/净收益指数  {args.start}~{args.end}")
    print("=" * 86)
    for local, tr, name in DOWNLOADS:
        download(pro, conn, local, tr, name, args.start, args.end)
    if not args.no_validate:
        cross_validate(conn)
    conn.close()


if __name__ == "__main__":
    main()
