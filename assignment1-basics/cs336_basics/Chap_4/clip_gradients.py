import torch
import math
from typing import Iterable, List

# ==========================================
# 1. 核心梯度裁剪函数
# ==========================================
def clip_gradients(parameters: Iterable[torch.Tensor], max_norm: float) -> None:
    """
    计算所有参数的全局梯度 L2 范数，如果超出最大值，则在原地(in-place)对梯度进行等比例缩放。
    """
    eps = 1e-6
    
    # 1. 过滤掉没有梯度的参数 (比如还没参与前向传播、或者 requires_grad=False 的层)
    params_with_grads = [p for p in parameters if p.grad is not None]
    if not params_with_grads:
        return

    # 2. 计算全局 L2 范数 ||g||_2
    # 我们先算出每个参数梯度的平方和，然后全部加起来，最后开一次根号
    # 使用 .detach() 确保这些计算不会被计入计算图
    sq_norms = [p.grad.detach().pow(2).sum() for p in params_with_grads]
    total_norm = torch.sqrt(sum(sq_norms))
    
    # 3. 判断并执行裁剪
    if total_norm > max_norm:
        # 计算缩放因子 (Scale Factor)
        scale = max_norm / (total_norm + eps)
        
        # 4. 原地修改每个参数的梯度
        for p in params_with_grads:
            # .mul_() 是 PyTorch 的原地乘法操作 (In-place multiplication)
            # 这意味着它直接修改了显存中的值，而不是创建一个新的 Tensor
            p.grad.detach().mul_(scale)
