# -*- coding: utf-8 -*-
"""
ETF 日线数据下载工具（tushare → astock_daily.db）
================================================

下载策略轮动所需的主要ETF日线数据，存入 etf_daily 表。
支持增量更新（已下载日期会跳过）。

用法:
    python download_etf_data.py              # 下载全部标的
    python download_etf_data.py --list        # 仅打印待下载标的
    python download_etf_data.py --recent 60   # 仅下载最近60个交易日
"""

import sys, os, time, sqlite3, argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── 配置 ──────────────────────────────────────────────────

# 需要下载的 ETF 列表（核心轮动池）
ETF_TARGETS = [
    # ══ 宽基（风格轮动）══
    # 代码           名称              跟踪指数
    ("510300.SH",  "沪深300ETF",      "000300.SH"),   # 大盘价值
    ("510050.SH",  "上证50ETF",       "000016.SH"),   # 超大盘蓝筹
    ("510500.SH",  "中证500ETF",      "000905.SH"),   # 中盘成长
    ("512100.SH",  "中证1000ETF",     "000852.SH"),   # 小盘
    ("515800.SH",  "中证800ETF",      "000906.SH"),   # 大中盘（中证800指数）
    ("159915.SZ",  "创业板ETF",       "399006.SZ"),   # 科技成长
    ("159949.SZ",  "创业板50ETF",     "399006.SZ"),   # 创业板龙头
    ("588000.SH",  "科创50ETF",       None),          # 科创板硬科技

    # ══ 行业/主题 ══
    ("512880.SH",  "证券ETF",         None),          # 券商（高beta，牛市旗手）
    ("512010.SH",  "医药ETF",         None),          # 医药（防御+成长）
    ("159928.SZ",  "消费ETF",         None),          # 消费龙头
    ("515050.SH",  "5G通信ETF",       None),          # 科技主题
    ("515030.SH",  "新能源车ETF",     None),          # 新能源赛道
    ("159766.SZ",  "旅游ETF",         None),          # 消费复苏

    # ══ 交易型/避险 ══
    ("518880.SH",  "黄金ETF",         None),          # 商品避险
    ("510880.SH",  "红利ETF",         None),          # 高股息防守
    ("511990.SH",  "华宝添益(货币)",  None),          # 货币基金
]

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "data", "tu-sharedata", "astock_daily.db")

# tushare 请求间隔（秒），避免触发频率限制
TUSHARE_INTERVAL = 0.5


# ── 数据库操作 ────────────────────────────────────────────

def get_conn():
    """获取数据库连接"""
    conn = sqlite3.connect(DB_PATH)
    return conn


def ensure_table(conn):
    """确保 etf_daily 表存在"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS etf_daily (
            ts_code     TEXT NOT NULL,
            trade_date  TEXT NOT NULL,
            open        REAL,
            high        REAL,
            low         REAL,
            close       REAL,
            pre_close   REAL,
            change      REAL,
            pct_chg     REAL,
            vol         REAL,
            amount      REAL,
            PRIMARY KEY (ts_code, trade_date)
        )
    """)
    conn.commit()


def get_existing_dates(conn, ts_code):
    """获取已下载的日期集合"""
    rows = conn.execute(
        "SELECT trade_date FROM etf_daily WHERE ts_code = ?",
        (ts_code,)
    ).fetchall()
    return {r[0] for r in rows}


# ── tushare 下载 ──────────────────────────────────────────

def download_fund_daily(ts_code, start_date, end_date):
    """通过 tushare fund_daily 接口下载日线数据

    返回: DataFrame 或 None
    """
    import tushare as ts
    pro = ts.pro_api()

    try:
        df = pro.fund_daily(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            fields="ts_code,trade_date,open,high,low,close,"
                   "pre_close,change,pct_chg,vol,amount"
        )
        if df is not None and len(df) > 0:
            # tushare 返回的数据按日期降序
            df = df.sort_values("trade_date").reset_index(drop=True)
            return df
        return None
    except Exception as e:
        print(f"    [ERR] {ts_code}: {e}")
        return None


def get_earliest_date(ts_code):
    """估算需要从哪天开始下载：查询 stock_basic 中的上市日期"""
    try:
        conn = get_conn()
        # 先查 stock_basic（虽然stock_basic可能没有ETF，但试一下）
        row = conn.execute(
            "SELECT list_date FROM stock_basic WHERE ts_code = ?",
            (ts_code,)
        ).fetchone()
        conn.close()
        if row and row[0]:
            return row[0].replace("-", "")
    except:
        pass
    # 默认从上市时间往回多取一些
    return "20100101"


def get_all_etf_codes():
    """从 tushare 获取全市场未退市 ETF 列表"""
    import tushare as ts
    pro = ts.pro_api()
    print("  正在从 tushare 获取全市场ETF列表...")
    df = pro.fund_basic(market='E', fields='ts_code,name,list_date,delist_date')
    # 未退市的
    df_active = df[df['delist_date'].isna() | (df['delist_date'] == '')]
    print(f"  共找到 {len(df_active)} 只未退市ETF")
    result = [(row['ts_code'], row['name'], None) for _, row in df_active.iterrows()]
    return result


def download_etf(targets=None, recent_days=None, all_etfs=False):
    """批量下载 ETF 日线数据

    Args:
        targets: 要下载的ETF列表，默认全部
        recent_days: 只下载最近 N 个交易日，None=全量
    """
    import tushare as ts
    pro = ts.pro_api()
    token = ts.get_token()
    if not token:
        print("[ERR] tushare token 未设置！请先设置 token。")
        return

    if targets is None:
        if all_etfs:
            targets = get_all_etf_codes()
        else:
            targets = ETF_TARGETS

    conn = get_conn()
    ensure_table(conn)

    print(f"\n{'=' * 60}")
    print(f"  ETF日线数据下载")
    print(f"{'=' * 60}")
    print(f"  数据库: {DB_PATH}")
    print(f"  标的数: {len(targets)}")
    print(f"{'=' * 60}")
    print()

    # 如果指定 recent_days，计算开始日期
    if recent_days:
        from datetime import datetime, timedelta
        ref_date = datetime.now()
        start_date = (ref_date - timedelta(days=recent_days * 1.5)).strftime("%Y%m%d")
    else:
        start_date = "20100101"

    # 获取所有交易日（用于确定最近交易日）
    all_trade_dates = set()
    idx_rows = conn.execute(
        "SELECT DISTINCT trade_date FROM index_daily ORDER BY trade_date DESC"
    ).fetchall()
    all_trade_dates = {r[0] for r in idx_rows}

    total_inserted = 0
    total_skipped = 0
    total_errors = 0

    for ts_code, name, tracked_index in targets:
        print(f"  [{ts_code}] {name}...")

        # 查看已下载数据
        existing = get_existing_dates(conn, ts_code)
        print(f"    已存在: {len(existing)} 条")

        # 确定下载区间
        if recent_days and all_trade_dates:
            sorted_dates = sorted(all_trade_dates, reverse=True)
            cutoff = sorted_dates[0]
            for d in sorted_dates:
                if len([x for x in sorted_dates if x >= d]) >= recent_days:
                    cutoff = d
                    break
            start = str(cutoff)
        else:
            start = start_date

        end = "20991231"

        # 尝试下载
        df = download_fund_daily(ts_code, start, end)

        if df is None or len(df) == 0:
            print(f"    [WARN] 未获取到数据，跳过")
            total_errors += 1
            time.sleep(TUSHARE_INTERVAL)
            continue

        # 增量插入（跳过已存在的日期）
        inserted = 0
        skipped = 0
        for _, row in df.iterrows():
            date_str = str(row["trade_date"])
            if date_str in existing:
                skipped += 1
                continue
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO etf_daily
                    (ts_code, trade_date, open, high, low, close,
                     pre_close, change, pct_chg, vol, amount)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(row["ts_code"]), date_str,
                    float(row["open"]) if pd.notna(row["open"]) else None,
                    float(row["high"]) if pd.notna(row["high"]) else None,
                    float(row["low"]) if pd.notna(row["low"]) else None,
                    float(row["close"]) if pd.notna(row["close"]) else None,
                    float(row["pre_close"]) if pd.notna(row["pre_close"]) else None,
                    float(row["change"]) if pd.notna(row["change"]) else None,
                    float(row["pct_chg"]) if pd.notna(row["pct_chg"]) else None,
                    float(row["vol"]) if pd.notna(row["vol"]) else None,
                    float(row["amount"]) if pd.notna(row["amount"]) else None,
                ))
                inserted += 1
            except Exception as e:
                print(f"    [ERR] 插入失败 {ts_code} {date_str}: {e}")

        conn.commit()
        total_inserted += inserted
        total_skipped += skipped

        print(f"    新增: {inserted} 条 | 跳过: {skipped} 条 | "
              f"总计: {len(existing) + inserted} 条")

        # tushare 频率限制
        time.sleep(TUSHARE_INTERVAL)

    conn.close()

    # ── 汇总 ──
    print(f"\n{'=' * 60}")
    print(f"  下载完成")
    print(f"{'=' * 60}")
    print(f"  新增: {total_inserted} 条")
    print(f"  跳过: {total_skipped} 条")
    print(f"  错误: {total_errors} 个标的")
    print(f"{'=' * 60}")

    # 打印最终统计
    conn2 = get_conn()
    print(f"\n  etf_daily 表最终统计:")
    rows = conn2.execute(
        "SELECT ts_code, COUNT(*) FROM etf_daily GROUP BY ts_code ORDER BY ts_code"
    ).fetchall()
    for code, cnt in rows:
        name_lookup = {c: n for c, n, _ in ETF_TARGETS}
        display_name = name_lookup.get(code, code)
        print(f"    {code} ({display_name}): {cnt} 条")
    conn2.close()


def list_targets():
    """打印待下载标的"""
    print(f"\n{'=' * 50}")
    print(f"  ETF轮动策略 - 待下载标的")
    print(f"{'=' * 50}")
    for ts_code, name, tracked_index in ETF_TARGETS:
        idx_str = f" → {tracked_index}" if tracked_index else " (独立)"
        print(f"  {ts_code}  {name}{idx_str}")
    print(f"{'=' * 50}")


# ── CLI 入口 ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETF日线数据下载")
    parser.add_argument("--list", action="store_true", help="仅打印待下载标的列表")
    parser.add_argument("--all", action="store_true", help="下载全市场所有未退市ETF（约2200只，需较长时间）")
    parser.add_argument("--recent", type=int, default=None,
                        help="只下载最近 N 个交易日（默认全量）")
    parser.add_argument("--codes", nargs="+", default=None,
                        help="指定下载哪些ETF代码，空格分隔（默认全部）")
    args = parser.parse_args()

    if args.list:
        list_targets()
        sys.exit(0)

    if args.all:
        download_etf(all_etfs=True, recent_days=args.recent)
    else:
        targets = ETF_TARGETS
        if args.codes:
            found = [t for t in ETF_TARGETS if t[0] in args.codes]
            not_found = set(args.codes) - {t[0] for t in ETF_TARGETS}
            for code in not_found:
                # 动态创建临时条目（代码即名称，无跟踪指数）
                found.append((code, code, None))
                print(f"  [+] 临时添加 {code}（不在预定义列表中，将直接下载）")
            targets = found
        download_etf(targets=targets, recent_days=args.recent)
