import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int

from .rmsnorm import RMSNorm
from .attention import MultiHeadAttention
from .swiglu import SwiGLU

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        
        # 子层 1: Pre-Norm + 多头自注意力机制
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadAttention(
            d_model=d_model, 
            num_heads=num_heads, 
            max_seq_len=max_seq_len, 
            theta=theta,             
            device=device, 
            dtype=dtype
        )

        # 子层 2: Pre-Norm + SwiGLU 前馈网络
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(
            d_model=d_model, 
            d_ff=d_ff, 
            device=device, 
            dtype=dtype
        )

    def forward(
        self, 
        x: Float[Tensor, "... seq_len d_model"],
        token_positions: Int[Tensor, "... seq_len"] | None = None
    ) -> Float[Tensor, "... seq_len d_model"]:
        
        # 残差流 1：注意力层
        x = x + self.attn(self.ln1(x), token_positions=token_positions)
        
        # 残差流 2：前馈网络层
        x = x + self.ffn(self.ln2(x))
        
        return x