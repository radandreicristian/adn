import torch.nn as nn
import torch
from einops import rearrange
import torch.nn.functional as f
from torch.nn.init import xavier_uniform_


class LinearAttention(nn.Module):
    def __init__(self,
                 d_hidden: int,
                 n_heads: int,
                 p_dropout: float,
                 use_mask: bool = False,
                 **kwargs
                 ):
        super(LinearAttention, self).__init__()
        self.eps = kwargs.get("eps")
        self.n_heads = n_heads

        self.to_q = nn.Linear(in_features=d_hidden, out_features=d_hidden, bias=True)
        self.to_k = nn.Linear(in_features=d_hidden, out_features=d_hidden, bias=True)
        self.to_v = nn.Linear(in_features=d_hidden, out_features=d_hidden, bias=True)

        if n_heads == 1:
            self.to_out = nn.Dropout(p=p_dropout)
        else:
            self.to_out = nn.Sequential(
                nn.Linear(in_features=d_hidden, out_features=d_hidden),
                nn.Dropout(p=p_dropout),
            )
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.to_q.weight)
        xavier_uniform_(self.to_k.weight)
        xavier_uniform_(self.to_v.weight)

    @staticmethod
    def elu_feature_kernel(x):
        return f.elu(x) + 1

    def forward(self, q, k, v):
        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b n h d", h=self.n_heads), (q, k, v)
        )

        q = self.elu_feature_kernel(q)
        k = self.elu_feature_kernel(k)

        # kv (batch, heads, d, d)
        kv = torch.einsum("nshd,nshm->nhmd", k, v)
        z = 1/(torch.einsum("nlhd,nhd->nlh", q, k.sum(dim=1))+self.eps)

        # v (batch, len, heads, d)
        v = torch.einsum("nlhd, nhmd, nlh-> nlhm", q, kv, z)

        v = rearrange(v, 'b l h d -> b l (h d)')
        return self.to_out(v)

    def forward_single(self, x: torch.Tensor, **kwargs):
        return self.forward(q=x, k=x, v=x)
