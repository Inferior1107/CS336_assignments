import torch
import torch.nn as nn
import math
import einx
from torch import Tensor
from jaxtyping import Float

from .linear import Linear

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        # 计算每个头的维度 (d_k = d_v = d_model / h)
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除！"
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        # 实例化 4 个线性投影层
        # 巧妙之处：我们不为每个头单独建 Linear，而是用一个巨大的 (d_model, d_model) 一次性算完！
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(
        self, 
        x: Float[Tensor, "... seq_len d_model"]
    ) -> Float[Tensor, "... seq_len d_model"]:
        
        seq_len = x.size(-2)

        # 1. 投影：一次性算出所有头的 Q, K, V
        # 形状变化: (... seq_len, d_model) -> (... seq_len, d_model)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 2. 切头 (Split Heads)：einx 的高光时刻！
        # 把 d_model 拆成 (num_heads, d_k)，并把 num_heads 移动到前面去作为并行的 Batch
        # 形状变化: (... seq_len, h * d) -> (... h, seq_len, d)
        Q = einx.rearrange("... s (h d) -> ... h s d", Q, h=self.num_heads)
        K = einx.rearrange("... s (h d) -> ... h s d", K, h=self.num_heads)
        V = einx.rearrange("... s (h d) -> ... h s d", V, h=self.num_heads)

        # 3. 缩放点积打分 (Scaled Dot-Product)
        # @ 运算符会自动处理前面所有的 batch 和 head 维度，只在最后两维做矩阵乘法
        # 形状变化: Q(... h, seq_len, d) @ K^T(... h, d, seq_len) -> (... h, seq_len, seq_len)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 4. 因果掩码 (Causal Mask)
        # 生成下三角矩阵：左下角是 True (允许看)，右上角是 False (禁止偷看未来)
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        scores = scores.masked_fill(~mask, float('-inf'))

        # 5. Softmax 归一化与提取 Value
        attn_weights = torch.softmax(scores, dim=-1)
        out = attn_weights @ V  # 形状: (... h, seq_len, d_v)

        # 6. 缝合 (Concatenate Heads)
        # 把并行的头重新拼回 d_model 维度
        # 形状变化: (... h, seq_len, d_v) -> (... seq_len, h * d_v)
        out = einx.rearrange("... h s d -> ... s (h d)", out)

        # 7. 最终输出投影
        return self.o_proj(out)