import torch.nn as nn
import torch
from einops import rearrange
import torch.nn.functional as f


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

    @staticmethod
    def elu_feature_kernel(x):
        return f.elu(x) + 1

    def forward(self, q, k, v):
        q, k, v = map(lambda x: rearrange(x, 'b l (h d) -> b l h d', h=self.n_heads),
                      (q, k, v))
        q = self.elu_feature_kernel(q)
        k = self.elu_feature_kernel(k)

        # kv (batch, heads, d, d)
        kv = torch.einsum("nshd,nshm->nhmd", k, v)
        z = 1/(torch.einsum("nlhd,nhd->nlh", q, k.sum(dim=1))+self.eps)

        # v (batch, len, heads, d)
        v = torch.einsum("nlhd,nhmd,nlh->nlhm", q, kv, z)

        v = rearrange(v, 'b l h d -> b l (h d)')
        return v.contiguous()

    def forward_single(self, x: torch.Tensor, **kwargs):
        return self.forward(q=x, k=x, v=x)
