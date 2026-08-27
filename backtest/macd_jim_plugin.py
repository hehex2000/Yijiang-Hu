"""MACD 趋势跟随择时策略 - Jim 状态机增强版（继承 MacdTimingPlugin）
============================================================
强制开启方向状态机（state_machine=True），与 macd_timing 同参数 A/B。

自动发现：MacdJimPlugin → 配置键 macd_jim → 在 run_backtest 逐股择时框架
（即「选股与择时」）中与 macd_timing 并排；此处以 macd_jim 替代朴素
macd_timing 作为默认 MACD 插件（config 中 macd_timing 已 disabled）。

与 macd_timing 唯一差异：信号逻辑（方向状态机 + 收盘确认 + 看不清 +
失效≠反转 + 信号漏斗），详见 macd_timing_plugin._run_state_machine。
"""
from backtest.macd_timing_plugin import MacdTimingPlugin


class MacdJimPlugin(MacdTimingPlugin):
    def __init__(self, capital, cfg):
        cfg = dict(cfg)
        cfg["state_machine"] = True
        super().__init__(capital, cfg)
        self.name = "MacdJimPlugin"
