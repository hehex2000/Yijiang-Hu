"""
为 run_backtest.py 添加胜率计算功能
"""
import re

def add_win_rate_calculation():
    """
    为 run_backtest.py 添加胜率计算
    1. 在文件顶部添加 calc_win_rate() 函数
    2. 修改每个策略函数以跟踪交易并计算胜率
    3. 更新输出格式以包含胜率
    """
    
    with open(r"C:\Users\99395\workbuddy\multi_factor_selection\run_backtest.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # 1. 添加 calc_win_rate() 函数（在 imports 之后）
    calc_win_rate_func = '''
# ---- 胜率计算函数 ----
def calc_win_rate_from_trades(trade_records):
    """
    从交易记录计算胜率
    trade_records: list of dicts with keys: action, price, shares
    Returns: (win_rate_pct, win_count, total_closed)
    """
    if not trade_records:
        return 0.0, 0, 0
    
    # FIFO 匹配买卖对
    pending_buys = []  # [{"price": p, "shares": s}]
    win = 0
    total = 0
    
    for t in trade_records:
        action = t.get("action", "")
        shares = t.get("shares", 0)
        price = t.get("price", 0.0)
        
        if action.startswith("BUY"):
            pending_buys.append({"price": price, "shares": shares})
        elif action.startswith("SELL"):
            remaining = shares
            while remaining > 0 and pending_buys:
                first = pending_buys[0]
                match_shares = min(first["shares"], remaining)
                pnl = (price - first["price"]) * match_shares
                total += 1
                if pnl > 0:
                    win += 1
                
                first["shares"] -= match_shares
                remaining -= match_shares
                
                if first["shares"] <= 0:
                    pending_buys.pop(0)
    
    wr = (win / total * 100) if total > 0 else 0.0
    return wr, win, total

'''
    
    # 在第一个函数定义之前插入 calc_win_rate_func
    # 找到第一个 "def " 的位置
    first_def_pos = content.find("\ndef ")
    if first_def_pos == -1:
        print("[ERR] 找不到函数定义")
        return
    
    # 在第一个函数定义之前插入
    content = content[:first_def_pos] + calc_win_rate_func + content[first_def_pos:]
    
    # 2. 保存修改后的文件
    with open(r"C:\Users\99395\workbuddy\multi_factor_selection\run_backtest.py", "w", encoding="utf-8") as f:
        f.write(content)
    
    print("[OK] 已添加 calc_win_rate_from_trades() 函数")

if __name__ == "__main__":
    add_win_rate_calculation()
