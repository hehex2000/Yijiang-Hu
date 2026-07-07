# -*- coding: utf-8 -*-
"""
回补 index_constituent 历史时点快照
====================================
狗股策略(及其它指数成分策略)做长期回测时，必须用「调仓日当时」的真实成分股，
否则会引入 生存偏差(只含幸存者) + 前视偏差(用未来才知道的成分股)，导致收益虚高。

本脚本用 Tushare 的 index_weight 接口，把各主要指数的历史成分股权重
(仅在每次调仓日发布的快照) 回补进本地库 index_constituent 表，
使 src/dogs_of_market_selector._get_constituents 的「时点查询」真正生效。

用法:
    python download_index_constituents.py                # 回补配置股票池对应的指数
    python download_index_constituents.py --all          # 回补全部主要指数
    python download_index_constituents.py --start 20140101 --end 20260706

依赖: tushare (已装 1.4.29), config.DATA.tushare_token
注意: index_weight 每次调用约 1 积分, 返回该指数全部历史调仓快照, 成本很低。
"""
import sys, os, argparse, sqlite3
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

try:
    import tushare as ts
except ImportError:
    print("[错误] 未安装 tushare，请先 pip install tushare")
    sys.exit(1)

from config import DATA

DB_PATH = DATA.get("local_db_path", r"D:\tu-shareData\astock_daily.db")
TOKEN = DATA.get("tushare_token", "")

# 主要指数映射: stock_pool -> Tushare index_code
POOL_TO_INDEX = {
    "hs300": "000300.SH",
    "zz500": "000905.SH",
    "zz800": "000906.SH",
    "zz1000": "000852.SH",
    "sz50":  "000016.SH",
}
# --all 时回补的全部指数
ALL_INDICES = ["000300.SH", "000905.SH", "000906.SH", "000852.SH", "000016.SH"]


def get_pro():
    if not TOKEN:
        print("[错误] config.DATA.tushare_token 为空，无法调用 Tushare")
        sys.exit(1)
    ts.set_token(TOKEN)
    return ts.pro_api()


def fetch_snapshots(pro, index_code, start, end):
    """用 index_weight 拉取该指数在 [start, end] 的全部历史成分股快照。

    返回 list of (index_code, con_code, trade_date, weight)
    index_weight 仅在每次调仓日有记录，天然就是「时点快照」。

    注意: Tushare index_weight 的入参是 index_code(指数代码), 返回列是
    con_code(成分股代码) 而非 ts_code。为避免单次超出行数上限，按年分块拉取。
    """
    import time
    rows = []
    sy, ey = int(start[:4]), int(end[:4])
    for y in range(sy, ey + 1):
        s = f"{y}0101"
        e = f"{y}1231"
        if e > end:
            e = end
        try:
            df = pro.index_weight(index_code=index_code, start_date=s, end_date=e)
        except Exception as ex:
            print(f"    [年块跳过] {index_code} {y}: {ex}")
            continue
        if df is None or df.empty:
            continue
        for _, r in df.iterrows():
            td = str(r["trade_date"])
            code = str(r["con_code"])
            w = float(r["weight"]) if pd.notna(r.get("weight")) else None
            rows.append((index_code, code, td, w))
        time.sleep(0.3)  # 避免触发分钟级调用频率限制
    return rows


def upsert(conn, rows):
    """先清空该批 index_code 的旧记录，再整体写入（幂等）。"""
    if not rows:
        return 0
    index_codes = sorted({r[0] for r in rows})
    cur = conn.cursor()
    for ic in index_codes:
        cur.execute("DELETE FROM index_constituent WHERE index_code = ?", (ic,))
    cur.executemany(
        "INSERT INTO index_constituent (index_code, ts_code, trade_date, weight) VALUES (?,?,?,?)",
        rows,
    )
    conn.commit()
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description="回补 index_constituent 历史时点快照")
    ap.add_argument("--all", action="store_true", help="回补全部主要指数")
    ap.add_argument("--start", default="20140101")
    ap.add_argument("--end", default="20260706")
    args = ap.parse_args()

    # 决定要回补哪些指数
    if args.all:
        targets = ALL_INDICES
    else:
        from config import SELECTION
        pool = SELECTION.get("stock_pool", "zz800")
        ic = POOL_TO_INDEX.get(pool)
        if not ic:
            print(f"[警告] stock_pool={pool} 无对应指数映射，回补全部主要指数")
            targets = ALL_INDICES
        else:
            targets = [ic]

    print(f"目标指数: {targets}")
    print(f"区间: {args.start} ~ {args.end}")
    print(f"数据库: {DB_PATH}\n")

    pro = get_pro()
    conn = sqlite3.connect(DB_PATH)

    # 现有快照数(回补前)
    before = {ic: conn.execute(
        "SELECT COUNT(DISTINCT trade_date) FROM index_constituent WHERE index_code=?", (ic,)
    ).fetchone()[0] for ic in targets}
    print("回补前各指数时点快照数:", before, "\n")

    total = 0
    for ic in targets:
        try:
            rows = fetch_snapshots(pro, ic, args.start, args.end)
        except Exception as e:
            print(f"  [跳过] {ic} 拉取失败: {e}")
            continue
        if not rows:
            print(f"  [空]   {ic} 无数据(可能不支持 index_weight 或区间无记录)")
            continue
        n = upsert(conn, rows)
        nsnap = conn.execute(
            "SELECT COUNT(DISTINCT trade_date) FROM index_constituent WHERE index_code=?", (ic,)
        ).fetchone()[0]
        print(f"  [完成] {ic}: 写入 {n} 行, 时点快照数 -> {nsnap}")
        total += n

    conn.close()
    print(f"\n全部完成, 共写入 {total} 行。")
    print("现在重新运行 run_dogs_annual.py，成分股将按各调仓日真实时点选取，")


if __name__ == "__main__":
    main()
