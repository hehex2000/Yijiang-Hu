"""
双均线 + Jim 框架状态机增强版（与基础版 dual_ma 做 A/B）

基础版 DualMAStrategyPlugin：金叉买 50% 仓 / 死叉清仓（经典"金叉买死叉卖"）。
本子类强制开启 state_machine，把信号逻辑升级为 Jim 视频里的可重复框架：
  - 方向状态机 UP / UNCLEAR / DOWN（价格相对双均线位置 + 双均线斜率方向）
  - 收盘确认（日频天然收盘确认）+ 看不清状态（不出手）+ 失效≠反转（不裸空）
  - 信号漏斗：原始金叉 → 方向确认 → 实际入场；震荡市停在 UNCLEAR 过滤 whipsaw
其余参数（仓位/止盈止损/ATR/凯利/均线周期）与基础版完全一致，仅信号逻辑不同。
"""

from backtest.dual_ma_plugin import DualMAStrategyPlugin


class DualMAJimPlugin(DualMAStrategyPlugin):
    """双均线策略 + Jim 框架状态机增强版"""

    def __init__(self, capital: float, cfg: dict):
        cfg = dict(cfg)
        cfg["state_machine"] = True  # 强制开启状态机（与基础版唯一差异）
        super().__init__(capital, cfg)
        self.name = "DualMAJimPlugin"
