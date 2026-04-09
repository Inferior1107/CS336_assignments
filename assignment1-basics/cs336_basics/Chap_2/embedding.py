import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int

class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )

        self.init_weights()

    def init_weights(self) -> None:
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0
        )

    def forward(
        self, 
        token_ids: Int[Tensor, "... sequence_length"]
    ) -> Float[Tensor, "... sequence_length embedding_dim"]:
        """
        Lookup the embedding vectors for the given token IDs.
        """

        return self.weight[token_ids]