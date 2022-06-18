import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import Tensor
import torch.nn.functional as f
from torch.nn.init import xavier_uniform_


class MultiHeadAttention(nn.Module):
    """A custom multi-head attention module."""

    def __init__(
        self, d_hidden: int, n_heads: int, p_dropout: float, use_mask: bool = False
    ) -> None:
        """
        Initialize the module.

        :param d_hidden: The feature dimension.
        :param n_heads: The number of heads in the multi-head attention.
        :param p_dropout: The dropout probability.
        :param use_mask: Whether to use masking (for temporal attention - prevents
        attending to events in the future).
        """
        super(MultiHeadAttention, self).__init__()
        assert d_hidden % n_heads == 0, "Hidden dimension must be divisible by n_heads."
        self.n_heads = n_heads
        self.d_head = d_hidden // n_heads

        self.to_q = nn.Linear(in_features=d_hidden, out_features=d_hidden, bias=True)
        self.to_k = nn.Linear(in_features=d_hidden, out_features=d_hidden, bias=True)
        self.to_v = nn.Linear(in_features=d_hidden, out_features=d_hidden, bias=True)

        self.scale = self.d_head**-0.5
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

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """
        Forward three tensors of shape (BT, N, D) or (BN, T, D).

        :param q: The "queries" to compose the attention map with.
        :param k: The "keys" to compute the attention map with.
        :param v: The "values" to apply the attention map to.
        :return: A tensor of shape (BT, N, D) or (BN, T, D).
        """
        # q, k, v (BT, N, D) or (BN, T, D)
        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.n_heads), (q, k, v)
        )

        attention_map = q @ k.transpose(-1, -2) * self.scale
        if self.use_mask:
            # Assume masking is only done after a spatial split.
            batch_size, _, seq_len, _ = q.shape
            mask = torch.tril(torch.ones(seq_len, seq_len).to(q.device))
            mask = repeat(mask, "m n -> b h m n", b=batch_size, h=self.n_heads).to(
                torch.bool
            )
            condition = torch.tensor([-(2**15) + 1], dtype=torch.float32).to(q.device)
            attention_map = torch.where(mask, attention_map, condition)
        attention_map = f.softmax(attention_map, dim=-1)
        x = attention_map @ v
        x = rearrange(x, "b h n d -> b n (h d)")
        return self.to_out(x)

    def forward_single(self, x: torch.Tensor, **kwargs):
        return self.forward(q=x, k=x, v=x)
