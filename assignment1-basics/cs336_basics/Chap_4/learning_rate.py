import math
from typing import Callable, Any

# ==========================================
# 1. 核心调度函数实现
# ==========================================
def cosine_learning_rate_schedule(
    t: int, 
    alpha_max: float, 
    alpha_min: float, 
    Tw: int, 
    Tc: int
) -> float:
    """
    计算带有预热期的余弦退火学习率。
    """
    # 阶段 1: Warm-up (热身期)
    if t < Tw:
        return (t / Tw) * alpha_max
        
    # 阶段 2: Cosine annealing (余弦退火期)
    elif Tw <= t <= Tc:
        # 计算余弦内部的角度 (比例从 0 渐变到 1，乘以 pi 变成 0 到 pi)
        progress = (t - Tw) / (Tc - Tw)
        cosine_decay = 0.5 * (1 + math.cos(progress * math.pi))
        return alpha_min + cosine_decay * (alpha_max - alpha_min)
        
    # 阶段 3: Post-annealing (平稳收尾期)
    else:
        return alpha_min
