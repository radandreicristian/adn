import torch.nn as nn
import torch
from reformer_pytorch import LSHSelfAttention


class LshSelfAttention(nn.Module):
    def __init__(self,
                 d_hidden: int,
                 n_heads: int,
                 p_dropout: float,
                 use_mask: bool = False,
                 **kwargs
                 ):
        super(LshSelfAttention, self).__init__()
        assert d_hidden % n_heads == 0, "Hidden dimension must be divisible by n_heads."

        self.bucket_size = kwargs.get("bucket_size")
        n_hashses = kwargs.get("n_hashes")
        self.lsh_attention = LSHSelfAttention(dim=d_hidden,
                                              heads=n_heads,
                                              bucket_size=self.bucket_size,
                                              n_hashes=n_hashses,
                                              causal=False)

    def forward(self, x: torch.Tensor):
        # x (b, l, d)
        b, l, d = x.shape

        # Seq len must be divisble with 2*bucket_size. Pad to match lens
        padding = 2 * self.bucket_size - l % (2 * self.bucket_size)

        if padding != 0:
            pad_tensor = torch.zeros((b, padding, d)).to(x.device)
            x = torch.cat([x, pad_tensor], dim=-2)
        x = self.lsh_attention(x)
        return x[:, :-padding, :]

    def forward_single(self, x: torch.Tensor, **kwargs):
        return self.forward(x=x)
