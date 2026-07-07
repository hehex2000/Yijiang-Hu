# -*- coding: utf-8 -*-
"""
批量下载 balancesheet（资产负债表）和 cashflow（现金流量表）
====================================================
按 end_date（报告期）批量下载，和 income 的下載方式一致。

用法:
    python download_bs_cf.py                     # 下载全部缺失报告期
    python download_bs_cf.py --recent 8           # 仅下载最近8个季度
    python download_bs_cf.py --table balancesheet # 只下载资产负债表
    python download_bs_cf.py --table cashflow     # 只下载现金流量表
"""

import sys, os, time, sqlite3, argparse
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── 配置 ──────────────────────────────────────────────────
try:
    from config import DATA
    token = DATA.get("tushare_token", "")
except (ImportError, KeyError, AttributeError):
    token = ""

if not token:
    print("[ERR] 无法读取 tushare_token，请检查 config.py")
    sys.exit(1)

# 数据库路径
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "data", "tu-sharedata", "astock_daily.db")

# API 间隔（秒）
TUSHARE_INTERVAL = 0.6

# ── 下载配置 ────────────────────────────────────────────────
BALANCESHEET_FIELDS = [
    "ts_code", "ann_date", "f_ann_date", "end_date", "report_type",
    "total_assets", "total_liab", "total_equity", "money_cap",
    "accounts_receiv", "inventories", "fix_assets", "intang_assets", "goodwill",
]
CASHFLOW_FIELDS = [
    "ts_code", "ann_date", "f_ann_date", "end_date", "report_type",
    "net_cashflow_oper", "net_cashflow_invest", "net_cashflow_finance",
]

# ── 生成待下载报告期 ────────────────────────────────────────
def generate_quarter_end_dates(start_year=2012):
    """生成 YYYYMMDD 格式的季度末日期列表（从start_year至今）"""
    now = datetime.now()
    dates = []
    for year in range(start_year, now.year + 1):
        for month in [3, 6, 9, 12]:
            if year == now.year and datetime(year, month, 1) + timedelta(days=31) > now:
                break
            if month == 12:
                d = f"{year}1231"
            else:
                d = f"{year}{month:02d}30" if month in (6, 9) else f"{year}{month:02d}31"
            dates.append(d)
    return dates


def get_missing_periods(table_name, log_conn, all_periods):
    """从 financial_download_log 查出已下载并成功的报告期"""
    done = set()
    rows = log_conn.execute(
        "SELECT period FROM financial_download_log WHERE data_type = ? AND status = 'done'",
        (table_name,)
    ).fetchall()
    done = {r[0] for r in rows}
    missing = [p for p in all_periods if p not in done]
    return missing


# ── 下载核心 ────────────────────────────────────────────────
def download_table(ts_pro, table_name, end_date, db_path=DB_PATH):
    """
    下载指定报告期的财务数据，写入数据库

    Args:
        ts_pro: Tushare Pro API
        table_name: "balancesheet" 或 "cashflow"
        end_date: YYYYMMDD 报告期
        db_path: 数据库路径

    Returns:
        int: 下载条数，失败返回 -1
    """
    if table_name == "balancesheet":
        fields = BALANCESHEET_FIELDS
        api_func = ts_pro.balancesheet
    elif table_name == "cashflow":
        fields = CASHFLOW_FIELDS
        api_func = ts_pro.cashflow
    else:
        print(f"  [ERR] 未知表名: {table_name}")
        return -1

    try:
        df = api_func(end_date=end_date, fields=",".join(fields))
    except Exception as e:
        print(f"  [ERR] API 调用失败: {e}")
        return -1

    if df is None or len(df) == 0:
        print(f"  [INFO] {end_date} 无数据")
        return 0

    # ── 写入数据库 ──
    conn = sqlite3.connect(db_path)
    try:
        # 用 replace 或 insert ignore 避免重复
        existing = conn.execute(
            f"SELECT COUNT(*) FROM {table_name} WHERE end_date = ? AND ts_code IN "
            f"(SELECT ts_code FROM {table_name} WHERE end_date = ?)",
            (end_date, end_date)
        ).fetchone()[0]
    except:
        existing = 0

    if existing > 0:
        # 已有部分数据，使用 INSERT OR REPLACE
        df.to_sql(table_name, conn, if_exists="replace", 
                   index=False, method="multi")
    else:
        df.to_sql(table_name, conn, if_exists="append",
                   index=False, method="multi")

    conn.commit()
    count = len(df)
    conn.close()
    return count


# ── 日志更新 ────────────────────────────────────────────────
def update_log(log_conn, table_name, period, status, retry_count=0):
    """更新 financial_download_log"""
    log_conn.execute(
        """INSERT OR REPLACE INTO financial_download_log
           (data_type, period, status, retry_count, update_time)
           VALUES (?, ?, ?, ?, ?)""",
        (table_name, period, status, retry_count, datetime.now().isoformat())
    )
    log_conn.commit()


# ── 入口 ────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载资产负债表/现金流量表")
    parser.add_argument("--table", choices=["balancesheet", "cashflow", "all"],
                        default="all", help="要下载的表（默认 all=两个都下）")
    parser.add_argument("--recent", type=int, default=0,
                        help="仅下载最近N个季度（默认0=全部）")
    parser.add_argument("--start-year", type=int, default=2012,
                        help="起始年份（默认2012）")
    args = parser.parse_args()

    # 初始化 Tushare
    import tushare as ts
    ts.set_token(token)
    ts_pro = ts.pro_api()

    # 生成所有报告期
    all_periods = generate_quarter_end_dates(args.start_year)
    if args.recent > 0:
        all_periods = all_periods[-args.recent:]
    print(f"待检查报告期: {len(all_periods)} 个")

    # 确定要下载的表
    tables_to_download = []
    if args.table in ("balancesheet", "all"):
        tables_to_download.append("balancesheet")
    if args.table in ("cashflow", "all"):
        tables_to_download.append("cashflow")

    log_conn = sqlite3.connect(DB_PATH)

    for table_name in tables_to_download:
        print(f"\n{'=' * 60}")
        print(f"  开始下载 {table_name}")
        print(f"{'=' * 60}")

        missing = get_missing_periods(table_name, log_conn, all_periods)
        if not missing:
            print(f"  所有报告期已下载完毕，无需补充")
            continue

        print(f"  待下载报告期: {len(missing)} 个")

        total_rows = 0
        fail_count = 0

        for i, period in enumerate(missing):
            retry = 0
            max_retry = 3
            success = False

            while retry < max_retry and not success:
                if i > 0 and retry == 0:
                    time.sleep(TUSHARE_INTERVAL)

                rows = download_table(ts_pro, table_name, period, DB_PATH)
                if rows >= 0:
                    total_rows += max(rows, 0)
                    print(f"  [{i+1}/{len(missing)}] {period} → {rows} 条")
                    update_log(log_conn, table_name, period, 
                               "failed" if rows == 0 else "done", retry)
                    success = True
                else:
                    retry += 1
                    if retry < max_retry:
                        print(f"  [{i+1}/{len(missing)}] {period} 重试 {retry}/{max_retry}...")
                        time.sleep(3)
                    else:
                        print(f"  [{i+1}/{len(missing)}] {period} ❌ 下载失败（已重试{max_retry}次）")
                        update_log(log_conn, table_name, period, "failed", retry)
                        fail_count += 1

        print(f"\n  {table_name} 下载完成：{total_rows} 条成功，{fail_count} 个失败")

    log_conn.close()
    print(f"\n✅ 全部完成！")
