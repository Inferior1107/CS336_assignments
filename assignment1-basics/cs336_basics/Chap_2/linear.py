import torch
import torch.nn as nn
import math
import einx
from torch import Tensor
from jaxtyping import Float

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )

        self.init_weights()
    
    def init_weights(self) -> None:
        sigma = math.sqrt(2.0 / (self.in_features + self.out_features))

        nn.init.trunc_normal_(
            self.W,
            mean=0.0,
            std=sigma,
            a=-3.0 * sigma,
            b=3.0 * sigma,
        )

    def forward(
        self, 
        x: Float[Tensor, "... d_in"]
    ) -> Float[Tensor, "... d_out"]:
        """
        Applies a linear transformation to the incoming data.
        """
        return einx.dot('... d_in, d_out d_in -> ... d_out', x, self.W)