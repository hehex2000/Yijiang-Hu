# -*- coding: utf-8 -*-
"""补跑唯一缺失象限：raw + 止损OFF（关 loguru 噪音，结果写文件）"""
import sys
import os
import io
import contextlib
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("LOGURU_LEVEL", "ERROR")
try:
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
except Exception:
    pass

import run_monthly_rebalance as m

m.PRICE_MODE = "raw"
m._ADJ_CACHE.clear()
m._ADJ_REF.clear()
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    res = m.run_backtest(start_date="20200102", end_date="20260723", top_n=20,
                         selection_method="value", stop_loss_pct=0.0)

out = {
    "quadrant": "raw_stopOFF",
    "total_return": res["total_return"],
    "annual_return": res["annual_return"],
    "max_drawdown": res["max_drawdown"],
    "sharpe": res["sharpe"],
    "trades": res["trades"],
}
print("RESULT_JSON " + json.dumps(out, ensure_ascii=False))
with open("_diag_raw_nostop.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
