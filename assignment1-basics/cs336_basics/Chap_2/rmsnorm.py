import torch
import torch.nn as nn
import einx
from torch import Tensor
from jaxtyping import Float

class RMSNorm(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        eps: float = 1e-5, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        
        self.eps = eps
        
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(
        self, 
        x: Float[Tensor, "... d_model"]
    ) -> Float[Tensor, "... d_model"]:
        
        in_dtype = x.dtype
        x_f32 = x.to(torch.float32)

        variance = einx.mean("... [d_model] -> ... 1", x_f32 ** 2)
        rms = torch.sqrt(variance + self.eps)

        x_normed = (x_f32 / rms).to(in_dtype)

        return einx.multiply("... d_model, d_model -> ... d_model", x_normed, self.weight)