import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int

# 导入你的组件
from .transformer import TransformerBlock
from .rmsnorm import RMSNorm

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float = 10000.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        
        # 1. 词嵌入层 (Token Embedding)
        # 将输入的 token id (比如 [102, 54, ...]) 变成稠密的 d_model 向量
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=d_model, 
            device=device, 
            dtype=dtype
        )
        
        # 2. 核心 Transformer 堆叠层 (The Blocks)
        # 像汉堡一样堆叠 num_layers 层的 TransformerBlock
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=theta,
                device=device,
                dtype=dtype
            ) for _ in range(num_layers)
        ])
        
        # 3. 最终的归一化层 (Final Layer Norm)
        # 在输出预测之前，按照 LLaMA 的架构，再过一次 RMSNorm 稳住数据
        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)
        
        # 4. 预测输出头 (Language Modeling Head)
        # 把 d_model 维度的隐变量，映射回 vocab_size，算出生词本上每个词的概率打分
        # 注意：大模型的 LM Head 通常是不带偏置 (bias) 的！
        self.lm_head = nn.Linear(
            in_features=d_model, 
            out_features=vocab_size, 
            bias=False, 
            device=device, 
            dtype=dtype
        )

    def forward(
        self, 
        x: Int[Tensor, "batch seq_len"]
    ) -> Float[Tensor, "batch seq_len vocab_size"]:
        
        # 获取当前序列的长度，并生成位置索引给 RoPE 用
        seq_len = x.size(1)
        token_positions = torch.arange(seq_len, device=x.device)
        
        # 1. 过 Embedding 层
        h = self.token_embedding(x)
        
        # 2. 依次穿过每一层 Transformer Block
        for layer in self.layers:
            h = layer(h, token_positions=token_positions)
            
        # 3. 过最终的 RMSNorm
        h = self.final_norm(h)
        
        # 4. 映射到词表大小，得到 Logits (未归一化的概率打分)
        logits = self.lm_head(h)
        
        return logits