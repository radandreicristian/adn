import torch.nn as nn
from performer_pytorch import SelfAttention
import torch
import torch.nn.functional as f


class FavorPlusAttention(nn.Module):
    def __init__(
        self, d_hidden: int, n_heads: int, p_dropout: float, use_mask: bool = False
    ) -> None:
        super(FavorPlusAttention, self).__init__()
        self.d_hidden = d_hidden

        assert self.d_hidden % n_heads == 0, (
            "Hidden size not divisible by number of " "heads."
        )

        self.n_heads = n_heads
        self.linear_self_attention = SelfAttention(
            dim=self.d_hidden,
            heads=self.n_heads,
            dim_head=self.d_hidden // n_heads,
            local_window_size=self.d_hidden,
            causal=False,
        )

    def forward(self, x: torch.Tensor):
        # features (batch, seq, n_nodes, d_hidden_feat+d_hidden_pos)
        return self.linear_self_attention(x)

    def forward_single(self, x: torch.Tensor, **kwargs):
        return self.forward(x=x)
