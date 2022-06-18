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

        self.to_qkv = nn.Linear(in_features=3 * d_hidden, out_features=3 * d_hidden,
                                bias=False)
        if n_heads == 1:
            self.to_out = nn.Identity()
        else:
            self.to_out = nn.Sequential(
                nn.Linear(in_features=d_hidden, out_features=d_hidden),
                nn.Dropout(p=p_dropout),
            )

    @staticmethod
    def elu_feature_kernel(x):
        return f.elu(x) + 1

    def forward(self, q, k, v):
        x = torch.cat((q, k, v), dim=-1)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b l (h d) -> b l h d', h=self.n_heads),
                      qkv)
        q = self.elu_feature_kernel(q)
        k = self.elu_feature_kernel(k)

        # kv (batch, heads, d, d)
        kv = torch.einsum("nshd,nshm->nhmd", k, v)
        z = 1/(torch.einsum("nlhd,nhd->nlh", q, k.sum(dim=1))+self.eps)

        # v (batch, len, heads, d)
        v = torch.einsum("nlhd,nhmd,nlh->nlhm", q, kv, z)

        v = rearrange(v, 'b l h d -> b l (h d)')
        return self.to_out(v)

    def forward_single(self, x: torch.Tensor, **kwargs):
        return self.forward(q=x, k=x, v=x)
