import torch.nn as nn
import torch
from einops import rearrange
from torch.nn.init import xavier_uniform_


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

        self.to_q = nn.Linear(in_features=d_hidden, out_features=d_hidden, bias=True)
        self.to_k = nn.Linear(in_features=d_hidden, out_features=d_hidden, bias=True)
        self.to_v = nn.Linear(in_features=d_hidden, out_features=d_hidden, bias=True)

        self.scale = self.d_head**0.5
        if n_heads == 1:
            self.to_out = nn.Dropout(p=p_dropout)
        else:
            self.to_out = nn.Sequential(
                nn.Linear(in_features=d_hidden, out_features=d_hidden, bias=False),
                nn.Dropout(p=p_dropout),
            )
        self.use_mask = use_mask
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.to_q.weight)
        xavier_uniform_(self.to_k.weight)
        xavier_uniform_(self.to_v.weight)

    def forward(self, q, k, v):
        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.n_heads), (q, k, v)
        )

        q = q.softmax(dim=-1) * self.scale**-0.25

        context_vectors = (k.transpose(-1, -2) @ v).softmax(dim=-2) * self.scale**-0.25
        attention = q @ context_vectors

        attention = rearrange(attention, "b h n d -> b n (h d)")

        return self.to_out(attention)

    def forward_single(self, x: torch.Tensor, **kwargs):
        return self.forward(q=x, k=x, v=x)
