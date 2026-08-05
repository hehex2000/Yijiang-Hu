"""
下载每股股利历史 → dividend_detail 表
========================================
为"红利质量复合"提供分红增长因子所需的真实每股股利数据。

Tushare dividend 接口每个 ts_code 返回多条（预案/股东大会通过/实施），
本脚本只保留 div_proc=='实施' 的现金分红记录（cash_div=每股股利）。

用法：
    python download_dividend.py                 # 默认下载 hs300+zz500+zz800 成分股
    python download_dividend.py --pool hs300    # 只下沪深300
    python download_dividend.py --pool all      # 下全A（较慢，约数千次调用）
    python download_dividend.py --force         # 清空旧表重新下载

依赖：tushare（venv_ml）、config_tushare.TUSHARE_TOKEN
"""
import argparse
import sqlite3
import time
import sys

import tushare as ts
from config_tushare import TUSHARE_TOKEN

DB_PATH = "D:/tu-shareData/astock_daily.db"
POOL_CODES = {
    "hs300": "000300.SH",
    "zz500": "000905.SH",
    "zz800": "000906.SH",
    "zz1000": "000852.SH",
}


def get_conn():
    return sqlite3.connect(DB_PATH)


def ensure_table(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dividend_detail (
            ts_code  TEXT,
            end_date TEXT,
            ex_date  TEXT,
            cash_div REAL,
            div_proc TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_dd_code ON dividend_detail(ts_code)")
    conn.commit()


def get_pool_stocks(conn, pools):
    """返回需要下载的 ts_code 集合（合并多池成分股，剔除 688/.BJ）"""
    codes = set()
    for p in pools:
        if p == "all":
            df = conn.execute(
                "SELECT ts_code FROM stock_basic WHERE ts_code NOT LIKE '688%' AND ts_code NOT LIKE '%.BJ'"
            ).fetchall()
            return {r[0] for r in df}
        idx = POOL_CODES.get(p)
        if not idx:
            continue
        df = conn.execute(
            "SELECT ts_code FROM index_constituent WHERE index_code = ? "
            "AND ts_code NOT LIKE '688%' AND ts_code NOT LIKE '%.BJ'",
            (idx,),
        ).fetchall()
        codes |= {r[0] for r in df}
    return codes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", nargs="+", default=["hs300", "zz500", "zz800"],
                    help="成分股池：hs300/zz500/zz800/zz1000/all")
    ap.add_argument("--force", action="store_true", help="清空旧表重新下载")
    args = ap.parse_args()

    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()

    conn = get_conn()
    ensure_table(conn)
    if args.force:
        conn.execute("DELETE FROM dividend_detail")
        conn.commit()

    stocks = get_pool_stocks(conn, args.pool)
    print(f"[dividend] 待下载 {len(stocks)} 只（池={args.pool}）")

    done = set(r[0] for r in conn.execute("SELECT DISTINCT ts_code FROM dividend_detail").fetchall())
    todo = sorted(stocks - done)
    print(f"[dividend] 已有 {len(done)} 只，本次新增 {len(todo)} 只")

    ok = 0
    for i, code in enumerate(todo, 1):
        try:
            df = pro.dividend(ts_code=code,
                              fields="ts_code,end_date,ex_date,cash_div,div_proc")
        except Exception as e:
            print(f"  ! {code} 接口异常: {e}", file=sys.stderr)
            time.sleep(1.0)
            continue
        if df is None or len(df) == 0:
            continue
        rows = []
        for _, r in df.iterrows():
            if str(r.get("div_proc", "")) != "实施":
                continue
            cd = r.get("cash_div")
            if cd is None or (isinstance(cd, float) and cd != cd):  # NaN
                continue
            rows.append((code, str(r["end_date"]), str(r.get("ex_date") or ""),
                         float(cd), "实施"))
        if rows:
            conn.executemany(
                "INSERT INTO dividend_detail(ts_code,end_date,ex_date,cash_div,div_proc) "
                "VALUES (?,?,?,?,?)", rows
            )
            conn.commit()
            ok += 1
        if i % 50 == 0:
            print(f"  进度 {i}/{len(todo)}，已入库 {ok} 只")
        time.sleep(0.25)  # 限速，避免触发 Tushare 积分限频

    print(f"[dividend] 完成。本次新增入库 {ok} 只，表内共 "
          f"{conn.execute('SELECT COUNT(DISTINCT ts_code) FROM dividend_detail').fetchone()[0]} 只")
    conn.close()


if __name__ == "__main__":
    main()
