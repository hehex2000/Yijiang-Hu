"""
下载宏观月度三件套 → cn_cpi / cn_ppi / cn_money_supply 表
========================================================
目的：解锁 BV1Lghg6NEA 22 条规律里的
  ④ M1 增速拐点领先大顶 3-6 月（cn_money_supply 的 m1_yoy）
  ⑲ PPI 回升周期股先涨 / CPI 回升消费股后涨（cn_ppi / cn_cpi）

Tushare 接口（月度，积分要求 2000）：
  - cn_cpi          居民消费价格（month, nt_yoy 同比 / nt_mom 环比 / nt_acc_yoy 累计同比）
  - cn_ppi          工业生产者出厂价格（month, ppi_yoy / ppi_mom / ppi_acc_yoy）
  - cn_money_supply 货币供应量（month, m0/m1/m2 及各自同比）

用法：
    python download_macro_cn.py            # 增量下载（每次全量拉，表很小）
    python download_macro_cn.py --force    # 清空旧表重新下载

依赖：tushare（venv_ml）、config_tushare.TUSHARE_TOKEN
"""
import argparse
import sqlite3
import sys
import time

import tushare as ts

from config_tushare import TUSHARE_TOKEN

DB_PATH = "D:/tu-shareData/astock_daily.db"

# 接口名 → 表名（字段原样落库，month 是唯一主键）
INTERFACES = ["cn_cpi", "cn_ppi", "cn_m"]


def get_conn():
    return sqlite3.connect(DB_PATH)


def ensure_table(conn, table, fields):
    cols = ",\n            ".join(f"{f} TEXT" for f in fields)
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            {cols},
            PRIMARY KEY (month)
        )
        """
    )
    conn.commit()


def download_one(pro, conn, name):
    df = pro.query(name)
    if df is None or df.empty:
        print(f"  [{name}] ❌ 返回空（可能是积分权限不足或接口名有变）")
        return 0
    fields = list(df.columns)
    ensure_table(conn, name, fields)
    placeholders = ",".join("?" for _ in fields)
    rows = [tuple(None if v is None else str(v) for v in r) for r in df[fields].values]
    conn.executemany(
        f"INSERT OR REPLACE INTO {name} ({','.join(fields)}) VALUES ({placeholders})",
        rows,
    )
    conn.commit()
    months = df["month"].astype(str)
    print(f"  [{name}] ✅ {len(rows):4d} 行  字段={fields}")
    print(f"           month 覆盖 {months.min()} ~ {months.max()}")
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="清空旧表重新下载")
    args = ap.parse_args()

    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()
    conn = get_conn()

    if args.force:
        for t in INTERFACES:
            conn.execute(f"DROP TABLE IF EXISTS {t}")
        conn.commit()
        print("已清空旧表，重新下载。")

    print("开始下载宏观月度三件套（cn_cpi / cn_ppi / cn_money_supply）...")
    total = {}
    for i, name in enumerate(INTERFACES):
        if i:
            time.sleep(0.3)  # 接口限频间隔
        try:
            total[name] = download_one(pro, conn, name)
        except Exception as e:
            print(f"  [{name}] ❌ 异常：{type(e).__name__}: {e}")
            total[name] = 0

    conn.close()

    print("\n=== 汇总 ===")
    ok = sum(1 for v in total.values() if v > 0)
    for k, v in total.items():
        print(f"  {k:20s} {v:5d} 行")
    if ok < len(INTERFACES):
        print("⚠️ 有接口未下到：先查积分（cn_* 系列要求 2000 分）与接口名，不要绕过降级。")
        sys.exit(1)
    print("三表齐备，④（M1 拐点）与 ⑲（PPI/CPI 轮动）数据缺口已解锁。")


if __name__ == "__main__":
    main()
