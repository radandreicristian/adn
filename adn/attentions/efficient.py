import torch.nn as nn
import torch
from einops import rearrange


class EfficientSelfAttention(nn.Module):
    def __init__(self,
                 d_hidden: int,
                 n_heads: int,
                 p_dropout: float,
                 use_mask: bool = False
                 ):
        super(EfficientSelfAttention, self).__init__()
        assert d_hidden % n_heads == 0, "Hidden dimension must be divisible by n_heads."
        self.n_heads = n_heads
        self.d_head = d_hidden // n_heads

        self.to_qkv = nn.Linear(in_features=3*d_hidden, out_features=3*d_hidden)

        self.scale = self.d_head**0.5
        if n_heads == 1:
            self.to_out = nn.Identity()
        else:
            self.to_out = nn.Sequential(
                nn.Linear(in_features=d_hidden, out_features=d_hidden),
                nn.Dropout(p=p_dropout),
            )

    def forward(self, q, k, v):
        x = torch.cat((q, k, v), dim=-1)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.n_heads), qkv
        )
        q = q.softmax(dim=-1) * self.scale**-0.25
        k = q.softmax(dim=-2) * self.scale**-0.25

        context_vectors = k.transpose(-1, -2) @ v
        attention = q @ context_vectors

        attention = rearrange(attention, "b h n d -> b n (h d)")

        return self.to_out(attention)

    def forward_single(self, x: torch.Tensor, **kwargs):
        return self.forward(q=x, k=x, v=x)
