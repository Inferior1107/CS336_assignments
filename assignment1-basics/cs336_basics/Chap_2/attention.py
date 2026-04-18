import torch
import torch.nn as nn
import math
import einx
from torch import Tensor
from jaxtyping import Float, Int

from .linear import Linear
from .rope import RotaryPositionalEmbedding

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,  # 接收 RoPE 最大长度
        theta: float,      # 接收 RoPE 旋转角度常数
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除！"
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        # 实例化 4 个线性投影层
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        # 实例化 RoPE (必须在 __init__ 里，以保证 buffer 能够移动到 GPU)
        self.rope = RotaryPositionalEmbedding(
            theta=theta, 
            d_k=self.d_k, 
            max_seq_len=max_seq_len, 
            device=device
        )

    def forward(
        self, 
        x: Float[Tensor, "... seq_len d_model"],
        token_positions: Int[Tensor, "... seq_len"] | None = None  # 接收位置信息
    ) -> Float[Tensor, "... seq_len d_model"]:
        
        seq_len = x.size(-2)

        # 1. 投影：一次性算出所有头的 Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 2. 切头 (Split Heads)
        Q = einx.rearrange("... s (h d) -> ... h s d", Q, h=self.num_heads)
        K = einx.rearrange("... s (h d) -> ... h s d", K, h=self.num_heads)
        V = einx.rearrange("... s (h d) -> ... h s d", V, h=self.num_heads)

        # 3. 注入 RoPE
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)
            
        pos_for_rope = token_positions.unsqueeze(-2)  # 扩展 head 维度以触发广播
        Q = self.rope(Q, pos_for_rope)
        K = self.rope(K, pos_for_rope)

        # 4. 缩放点积打分 (einx 显式指定维度)
        logits = einx.dot('... h s_q d, ... h s_k d -> ... h s_q s_k', Q, K) / math.sqrt(self.d_k)

        # 5. 因果掩码与 Softmax 归一化
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        attn_weights = torch.softmax(torch.where(mask, logits, float('-inf')), dim=-1)

        # 6. 提取 Value 并加权求和
        out = einx.dot('... h s_q s_k, ... h s_k d_v -> ... h s_q d_v', attn_weights, V)

        # 7. 缝合 (Concatenate Heads) 并输出投影
        out = einx.rearrange("... h s d -> ... s (h d)", out)
        return self.o_proj(out)