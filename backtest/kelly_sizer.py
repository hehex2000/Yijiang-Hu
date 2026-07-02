# -*- coding: utf-8 -*-
"""
凯利公式仓位计算器（Kelly Position Sizer）

基于约翰·凯利(John Kelly)1956年推导的公式，用于计算每次交易的最优投入比例。

公式: f = (b * p - q) / b
- b: 盈亏比（赚时赚多少 ÷ 亏时亏多少）
- p: 胜率（交易盈利的概率）
- q: 败率（q = 1 - p）
- f: 全凯利最优仓位比例

实操要点（来自UP主Jim《凯利公式》视频）：
1. 全凯利太激进 → 使用半凯利（f/2）降低波动和回撤
2. 输入参数是估计值非真值 → 在结果上再打折（safety_discount）
3. 凯利公式的真正价值不是精确到小数点，而是给你一条警戒线

参考: 《凯利公式——一个教我们每次买票买多少的数学公式》
"""


class KellySizer:
    """
    凯利公式仓位计算器

    使用示例:
        kelly = KellySizer(
            estimated_win_rate=0.55,
            estimated_win_loss_ratio=1.5,
            kelly_fraction=0.5,          # 半凯利
            safety_discount=0.8,         # 参数不确定性打折
            max_position_pct=0.25,
            min_position_pct=0.05,
        )
        position_pct = kelly.get_position_pct()  # 返回 0.0 ~ 0.25
    """

    def __init__(
        self,
        estimated_win_rate: float = 0.55,
        estimated_win_loss_ratio: float = 1.5,
        kelly_fraction: float = 0.5,
        max_position_pct: float = 0.25,
        min_position_pct: float = 0.05,
        safety_discount: float = 0.8,
    ):
        """
        Args:
            estimated_win_rate: 估计胜率（默认0.55，即55%的胜率）
                - 视频提到典型策略胜率在55-65%之间
                - 如果实际胜率未知，宁可低估不可高估
            estimated_win_loss_ratio: 估计盈亏比（默认1.5）
                - 即赚1.5元亏1元
                - 视频举例b=1（赚亏一样多）
            kelly_fraction: 凯利折扣系数（默认0.5 = 半凯利）
                - 1.0 = 全凯利（理论最优，但波动极大）
                - 0.5 = 半凯利（视频推荐，保留大部分增长率）
                - 0.25 = 四分之一凯利（极度保守）
            max_position_pct: 最大仓位上限（默认0.25 = 25%）
                - 防止凯利公式在极端参数下给出过激仓位
            min_position_pct: 最小仓位下限（默认0.05 = 5%）
                - 低于此值不交易（没有任何实际意义）
            safety_discount: 参数不确定性折扣（默认0.8）
                - 视频强调：输入的胜率和盈亏比是估计值
                - 额外再打8折是对估计误差的自我保护
        """
        self.estimated_win_rate = estimated_win_rate
        self.estimated_win_loss_ratio = estimated_win_loss_ratio
        self.kelly_fraction = kelly_fraction
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct
        self.safety_discount = safety_discount

    def get_position_pct(self) -> float:
        """
        计算本次应投入资金的比例

        计算步骤:
        1. 验证参数有效性（b > 0, p in [0, 1]）
        2. 计算全凯利 f* = (b×p - q) / b
        3. f* ≤ 0 → 期望值为负或零 → 不交易（返回0）
        4. 应用半凯利折扣: f = f* × kelly_fraction
        5. 应用不确定性保护: f = f × safety_discount
        6. 夹在 [min_position_pct, max_position_pct] 之间

        Returns:
            float: 应投入资金的比例（0.0 ~ max_position_pct）
        """
        p = self.estimated_win_rate
        b = self.estimated_win_loss_ratio

        # 边界检查
        if b <= 0:
            # 盈亏比为0或负 → 没有交易价值
            return 0.0
        if p <= 0:
            # 胜率为0 → 绝不交易
            return 0.0
        if p >= 1.0:
            # 胜率100% → 应该投入最大仓位（但现实中不存在）
            return self.max_position_pct

        q = 1.0 - p

        # 计算全凯利
        # f* = (b×p - q) / b
        # 例如: b=1, p=0.6, q=0.4 → f* = (1×0.6 - 0.4) / 1 = 0.2
        f_star = (b * p - q) / b

        # 无正向期望 → 不交易
        if f_star <= 0:
            return 0.0

        # 应用折扣
        f = f_star * self.kelly_fraction * self.safety_discount

        # 夹在安全边界内
        f = max(self.min_position_pct, min(f, self.max_position_pct))

        return f

    def get_position_amount(self, available_cash: float) -> float:
        """
        根据可用资金计算具体投入金额

        Args:
            available_cash: 可用资金

        Returns:
            float: 应投入的金额
        """
        pct = self.get_position_pct()
        return available_cash * pct

    def get_info(self) -> dict:
        """
        返回当前凯利计算结果的详细信息（用于日志/调试）

        Returns:
            dict: {f_star, kelly_fraction, safety_discount, f_final, position_pct}
        """
        p = self.estimated_win_rate
        b = self.estimated_win_loss_ratio
        q = 1.0 - p

        if b <= 0:
            return {"error": "盈亏比无效", "position_pct": 0.0}

        f_star = (b * p - q) / b
        if f_star <= 0:
            return {"f_star": f_star, "note": "期望值为负，不交易", "position_pct": 0.0}

        f_final = f_star * self.kelly_fraction * self.safety_discount
        position_pct = max(self.min_position_pct, min(f_final, self.max_position_pct))

        return {
            "f_star": round(f_star, 4),
            "kelly_fraction": self.kelly_fraction,
            "safety_discount": self.safety_discount,
            "f_final": round(f_final, 4),
            "position_pct": round(position_pct, 4),
        }

    def __repr__(self) -> str:
        info = self.get_info()
        if "error" in info:
            return f"KellySizer(不交易: {info['error']})"
        return (
            f"KellySizer(p={self.estimated_win_rate}, b={self.estimated_win_loss_ratio}, "
            f"f*={info['f_star']:.2%}, 半凯利={info['kelly_fraction']}, "
            f"仓位={info['position_pct']:.2%})"
        )


# ============================================================
# 预设配置（可直接使用的常见参数组合）
# ============================================================

# 保守配置：适合布林带/均值回归等低胜率策略
CONSERVATIVE_KELLY = {
    "estimated_win_rate": 0.50,
    "estimated_win_loss_ratio": 1.8,
    "kelly_fraction": 0.5,
    "max_position_pct": 0.20,
    "min_position_pct": 0.05,
    "safety_discount": 0.8,
}

# 平衡配置：适合RSI/MACD等中等胜率策略
BALANCED_KELLY = {
    "estimated_win_rate": 0.55,
    "estimated_win_loss_ratio": 1.5,
    "kelly_fraction": 0.5,
    "max_position_pct": 0.25,
    "min_position_pct": 0.05,
    "safety_discount": 0.8,
}

# 积极配置：适合海龟趋势等高盈亏比策略（但胜率低）
AGGRESSIVE_KELLY = {
    "estimated_win_rate": 0.40,
    "estimated_win_loss_ratio": 3.0,
    "kelly_fraction": 0.5,
    "max_position_pct": 0.25,
    "min_position_pct": 0.05,
    "safety_discount": 0.8,
}
