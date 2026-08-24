# -*- coding: utf-8 -*-
"""
补全 fina_indicator 的 Piotroski F-score 必需六列：roa / gross_margin / asset_turn / ocfps / eps / current_ratio。

背景：
  计划 plan_piotroski.md 的 P0 第一优先级数据补全项。本地 fina_indicator 表已被
  backfill_fina_2013_2014.py 等 ALTER 加过 ocfps/debt_to_assets/ocf_to_debt/ar_turn 等列
  （value_stock_selector 在用），但 F-score 必需的 roa/gross_margin/asset_turn 仍缺；
  且 ocfps/eps/current_ratio 此前也仅 ~77% 覆盖（大蓝筹尤甚），导致 item 2/4/6 得 0，
  故本轮一并补六列。

关键正确性约束（血泪教训）：
  绝不能用 INSERT OR REPLACE 整行写入——那会把已存在、由其他 backfill 写入的列
  （ocfps/debt_to_assets/ocf_to_debt/ar_turn）一并清空成 NULL。本脚本只 UPDATE 这六列，
  仅当 (ts_code,end_date) 原本不在表内时才 INSERT OR IGNORE 一条基础行。

复用范式：
  token 读取 / 逐只拉取 / 限速重试 参照 backfill_fina_2013_2014.py。

用法：
  python backfill_fina_fscore.py            # 全量补全（支持断点续传）
  python backfill_fina_fscore.py --check    # 只校验填充率，不拉数据
"""

import os
import sys
import time
import sqlite3

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# ── 数据库路径（优先 config，fallback 硬编码）─────────────────
try:
    from config import DATA
    DB_PATH = DATA.get("local_db_path", "D:/tu-shareData/astock_daily.db")
except Exception:
    DB_PATH = "D:/tu-shareData/astock_daily.db"

NEW_COLS = ["roa", "gross_margin", "asset_turn", "ocfps", "eps", "current_ratio"]  # 本库列名
# tushare fina_indicator 真实字段名映射：资产周转率在 tushare 叫 assets_turn（带 s）。
# ocfps/eps/current_ratio 是 F-score item 2/4/6 的核心输入，此前漏补（仅~77%覆盖），
# 本轮一并并入回填，使 9 项全覆盖。
TS_FIELD_MAP = {"roa": "roa", "gross_margin": "gross_margin",
                "asset_turn": "assets_turn", "ocfps": "ocfps",
                "eps": "eps", "current_ratio": "current_ratio"}
FIELDS = "ts_code,end_date," + ",".join(TS_FIELD_MAP.values())
START_DATE = "20090101"
END_DATE = "20261231"
SLEEP = 0.06  # 调用间隔（秒）；付费 token 限速宽松
PROGRESS_FILE = os.path.join(HERE, "data", "results", "_fscore_backfill_progress.txt")


# ── 读取 token（同 backfill_fina_2013_2014.py 范式）─────────────
def get_token():
    try:
        import config_tushare as ct
        t = ct.TUSHARE_TOKEN
        if t:
            return t
    except Exception:
        pass
    try:
        from config import DATA
        t = DATA.get("tushare_token", "")
        if t:
            return t
    except Exception:
        pass
    return ""


def ensure_cols(db_path):
    conn = sqlite3.connect(db_path)
    existing = {r[1] for r in conn.execute("PRAGMA table_info(fina_indicator)").fetchall()}
    for c in NEW_COLS:
        if c not in existing:
            conn.execute(f"ALTER TABLE fina_indicator ADD COLUMN {c} REAL")
            print(f"  + 新增列 {c}")
        else:
            print(f"  = 列已存在（跳过）: {c}")
    conn.commit()
    conn.close()


def get_stock_list(db_path):
    import pandas as pd
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT ts_code FROM stock_basic ORDER BY ts_code", conn)
        if len(df) > 0:
            conn.close()
            return df["ts_code"].tolist()
    except Exception:
        pass
    df = pd.read_sql_query(
        "SELECT DISTINCT ts_code FROM daily WHERE trade_date >= '20120101'", conn)
    conn.close()
    return df["ts_code"].tolist()


def load_progress(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return set(l.strip() for l in f if l.strip())
    return set()


def save_progress(path, done):
    with open(path, "w", encoding="utf-8") as f:
        for c in sorted(done):
            f.write(c + "\n")


def write_rows(db_path, rows):
    """只 UPDATE 六列；若该 (ts_code,end_date) 原本不在表内才 INSERT OR IGNORE 基础行。"""
    cols = NEW_COLS
    set_sql = ",".join(f"{c}=?" for c in cols)
    ins_cols = ["ts_code", "end_date"] + cols
    ins_sql = (f"INSERT OR IGNORE INTO fina_indicator ({','.join(ins_cols)}) "
               f"VALUES ({','.join(['?'] * len(ins_cols))})")
    conn = sqlite3.connect(db_path)
    for r in rows:
        # r = (ts_code, end_date, roa, gross_margin, asset_turn, ocfps, eps, current_ratio)
        vals = tuple(r[2:2 + len(cols)]) + (r[0], r[1])
        cur = conn.execute(
            f"UPDATE fina_indicator SET {set_sql} WHERE ts_code=? AND end_date=?", vals)
        if cur.rowcount == 0:
            conn.execute(ins_sql, r)
    conn.commit()
    conn.close()


def check_fill(db_path):
    conn = sqlite3.connect(db_path)
    total = conn.execute("SELECT COUNT(*) FROM fina_indicator").fetchone()[0]
    existing = {r[1] for r in conn.execute("PRAGMA table_info(fina_indicator)").fetchall()}
    print("=" * 60)
    print(f"填充率校验（总 {total} 行）：")
    for col in NEW_COLS:
        if col not in existing:
            print(f"  {col}: 列不存在（需先跑补全）")
            continue
        n = conn.execute(
            f"SELECT COUNT(*) FROM fina_indicator WHERE {col} IS NOT NULL").fetchone()[0]
        print(f"  {col}: {n}/{total} 非空 ({100.0 * n / max(total, 1):.1f}%)")
    conn.close()


def main():
    if "--check" in sys.argv:
        check_fill(DB_PATH)
        return

    import pandas as pd
    import tushare as ts

    token = get_token()
    if not token:
        print("[ERR] 无法读取 tushare_token，请检查 config_tushare.py / config.py")
        sys.exit(1)
    ts.set_token(token)
    pro = ts.pro_api()

    ensure_cols(DB_PATH)

    # 只补缺口：六列任一为 NULL 的 ts_code（roa/gross_margin/asset_turn 已~97%，
    # ocfps/eps/current_ratio 此前漏补仅~77%，驱动本轮主要缺口）。
    conn = sqlite3.connect(DB_PATH)
    where = " OR ".join(f"{c} IS NULL" for c in NEW_COLS)
    gap = conn.execute(
        f"SELECT DISTINCT ts_code FROM fina_indicator WHERE {where}").fetchall()
    conn.close()
    gap_stocks = [r[0] for r in gap]

    done = load_progress(PROGRESS_FILE)
    remaining = [s for s in gap_stocks if s not in done]
    print("=" * 70)
    print(f"补全 fina_indicator 六列 roa/gross_margin/asset_turn/ocfps/eps/current_ratio（Piotroski F-score 必需）")
    print(f"  缺口股票 {len(gap_stocks)} | 已完成 {len(done)} | 本轮待处理 {len(remaining)}")
    print(f"  区间 {START_DATE}~{END_DATE} | 断点续传文件 {PROGRESS_FILE}")
    print("=" * 70)

    ok = 0
    empty = 0
    buf = []

    def flush():
        nonlocal ok, buf
        if buf:
            write_rows(DB_PATH, buf)
            ok += len(buf)
            buf = []

    for i, tc in enumerate(remaining):
        if i % 100 == 0:
            print(f"  进度 {i}/{len(remaining)} | 累计写入 {ok} 行")
            flush()
            save_progress(PROGRESS_FILE, done)
        df = None
        try:
            df = pro.fina_indicator(
                ts_code=tc, start_date=START_DATE, end_date=END_DATE, fields=FIELDS)
        except Exception as e:
            msg = str(e)
            if "超限" in msg or "rate" in msg.lower():
                print(f"  [限速] {tc} 等待60s...")
                time.sleep(60)
                try:
                    df = pro.fina_indicator(
                        ts_code=tc, start_date=START_DATE, end_date=END_DATE, fields=FIELDS)
                except Exception:
                    df = None
            else:
                df = None
        if df is not None and len(df) > 0:
            for _, row in df.iterrows():
                buf.append((row["ts_code"], row["end_date"],
                            row.get("roa"), row.get("gross_margin"),
                            row.get("assets_turn"), row.get("ocfps"),
                            row.get("eps"), row.get("current_ratio")))
        else:
            empty += 1
        done.add(tc)
        time.sleep(SLEEP)

    flush()
    save_progress(PROGRESS_FILE, done)
    print(f"\n拉取完成：写入 {ok} 行，空响应 {empty} 只")
    check_fill(DB_PATH)
    print("完成。")


if __name__ == "__main__":
    main()
