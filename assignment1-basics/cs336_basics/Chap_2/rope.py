import torch
import torch.nn as nn
import einx
from torch import Tensor
from jaxtyping import Float, Int

class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self, 
        theta: float, 
        d_k: int, 
        max_seq_len: int, 
        device: torch.device | None = None
    ):
        super().__init__()
        self.d_k = d_k

        # 1. 计算频率指针 (inv_freq): theta^(-2k/d)
        # 用 arange 步长为 2 取出偶数索引 0, 2, 4... 对应公式里的 2k
        exponents = torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k
        inv_freq = 1.0 / (theta ** exponents)

        # 2. 生成绝对位置序列: 0, 1, 2, ..., max_seq_len - 1
        t = torch.arange(max_seq_len, device=device, dtype=torch.float32)

        # 3. 计算每个位置在每个频率下的旋转角度 (m * theta_k)
        # torch.outer 做外积，生成一个形状为 (max_seq_len, d_k // 2) 的矩阵
        freqs = torch.outer(t, inv_freq)

        # 4. 预计算所有的 sin 和 cos，并注册为不保存的 buffer (persistent=False)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_k"],
        token_positions: Int[Tensor, "... seq_len"]
    ) -> Float[Tensor, "... seq_len d_k"]:
        
        # 1. 查表：根据传入的 token_positions 获取对应的 cos 和 sin
        # 利用 PyTorch 的高级索引，拿到的形状是 (... seq_len, d_k // 2)
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        # 2. 将 x 的最后一维拆分成成对的坐标
        # 例如 d_k=64，拆成 32 组，每组 2 个元素。
        # 这里 einx 发挥了巨大威力，直接帮你把内存重塑得清清楚楚
        x_reshaped = einx.rearrange("... (d_half two) -> ... d_half two", x, two=2)
        
        # 提取实部(x0)和虚部(x1)
        x0 = x_reshaped[..., 0]  # (... seq_len, d_k // 2)
        x1 = x_reshaped[..., 1]  # (... seq_len, d_k // 2)

        # 3. 执行纯数学的二维旋转矩阵乘法
        # 按照讲义公式 (8) 里的矩阵展开：
        # [ cos  -sin ]   [ x0 ]   [ x0*cos - x1*sin ]
        # [ sin   cos ] * [ x1 ] = [ x0*sin + x1*cos ]
        x0_out = x0 * cos - x1 * sin
        x1_out = x0 * sin + x1 * cos

        # 4. 把旋转后的 x0 和 x1 重新拼成对，并展平回 d_k
        x_out = torch.stack([x0_out, x1_out], dim=-1)
        return einx.rearrange("... d_half two -> ... (d_half two)", x_out)