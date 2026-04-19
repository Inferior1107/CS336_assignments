import torch
import math
from typing import Optional, Callable

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        # 1. 健壮性检查
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameters: {betas}")
        
        # 2. 将超参数打包到 defaults 字典中，交给父类管理
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        # 遍历参数组 (比如有的层学习率不同，这里会有多个 group)
        for group in self.param_groups:
            # 提取当前组的超参数
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            # 遍历当前组内的每一个参数张量 p
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p] # 获取参数 p 专属的状态字典

                # ==========================================
                # 步骤 1: 状态初始化 (init)
                # ==========================================
                if len(state) == 0:
                    state["t"] = 0
                    # 创建与 p 形状完全相同的全 0 张量
                    state["m"] = torch.zeros_like(p.data) 
                    state["v"] = torch.zeros_like(p.data)

                # 获取历史状态
                m = state["m"]
                v = state["v"]
                
                # 步数加 1
                state["t"] += 1
                t = state["t"]

                # ==========================================
                # 步骤 2: 更新一阶和二阶矩 (Update moments)
                # ==========================================
                # m <- beta1 * m + (1 - beta1) * g
                m = beta1 * m + (1 - beta1) * grad
                
                # v <- beta2 * v + (1 - beta2) * g^2
                v = beta2 * v + (1 - beta2) * (grad ** 2)

                # [重点] 将更新后的 m 和 v 存回 state 字典，供下一次迭代使用！
                state["m"] = m
                state["v"] = v

                # ==========================================
                # 步骤 3: 计算偏差校正后的学习率 (Bias correction)
                # ==========================================
                # alpha_t <- alpha * sqrt(1 - beta2^t) / (1 - beta1^t)
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                alpha_t = lr * math.sqrt(bias_correction2) / bias_correction1

                # ==========================================
                # 步骤 4: 更新参数 (Update parameters)
                # ==========================================
                # theta <- theta - alpha_t * m / (sqrt(v) + eps)
                p.data -= alpha_t * m / (torch.sqrt(v) + eps)

                # ==========================================
                # 步骤 5: 权重衰减 (Apply weight decay)
                # ==========================================
                # theta <- theta - alpha * lambda * theta
                if weight_decay != 0:
                    p.data -= lr * weight_decay * p.data

        return loss