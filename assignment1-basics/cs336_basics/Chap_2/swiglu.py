import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from jaxtyping import Float

from .linear import Linear 

class SwiGLU(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        d_ff: int | None = None,
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        
        # 讲义要求：计算 8/3 * d_model，并向 64 向上取整以对齐硬件
        if d_ff is None:
            raw_d_ff = int(d_model * 8 / 3)
            # 这是一个非常经典的向上取整到 64 倍数的算法 (LLaMA 官方实现也是如此)
            d_ff = 64 * ((raw_d_ff + 63) // 64)

        # 实例化三个权重矩阵 (注意输入输出方向)
        # w1: 负责 Gate (门控)
        self.w1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        # w3: 负责 Value (信息)
        self.w3 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        # w2: 负责 Output (投影回原维度)
        self.w2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)

    def forward(
        self, 
        x: Float[Tensor, "... d_model"]
    ) -> Float[Tensor, "... d_model"]:
        
        # 公式：FFN(x) = W2( SiLU(W1(x)) * W3(x) )
        
        # 1. 门控分支：通过 W1 并使用 SiLU 激活
        # (讲义提到可以用 x * torch.sigmoid(x)，但 F.silu 是底层优化过的等价实现)
        gate = F.silu(self.w1(x))
        
        # 2. 信息分支：通过 W3 (不加激活函数)
        value = self.w3(x)
        
        # 3. 逐元素相乘并投影回原维度
        return self.w2(gate * value)