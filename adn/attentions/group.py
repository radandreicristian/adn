import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import Tensor
import torch.nn.functional as f


class GroupAttention(nn.Module):
    """A custom group attention module."""

    def __init__(
            self,
            d_hidden: int,
            n_heads: int,
            p_dropout: float,
            use_mask: bool = False,
            **kwargs
    ) -> None:
        """
        Initialize the module.

        :param d_hidden: The feature dimension.
        :param n_heads: The number of heads in the multi-head attention.
        :param p_dropout: The dropout probability.
        :param use_mask: Whether to use masking (for temporal attention - prevents
        attending to events in the future).
        """
        super(GroupAttention, self).__init__()
        assert d_hidden % n_heads == 0, "Hidden dimension must be divisible by n_heads."
        self.n_heads = n_heads
        self.d_head = d_hidden // n_heads

        self.to_qkv = nn.Linear(in_features=3 * d_hidden, out_features=3 * d_hidden)
        self.scale = self.d_head**0.5
        if n_heads == 1:
            self.to_out = nn.Identity()
        else:
            self.to_out = nn.Sequential(
                nn.Linear(in_features=d_hidden, out_features=d_hidden),
                nn.Dropout(p=p_dropout),
            )
        self.use_mask = use_mask

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """
        Forward three tensors of shape (BT, N, D) or (BN, T, D).

        :param q: The "queries" to compose the attention map with.
        :param k: The "keys" to compute the attention map with.
        :param v: The "values" to apply the attention map to.
        :return: A tensor of shape (BT, N, D) or (BN, T, D).
        """
        # q, k, v (BT, N, D) or (BN, T, D)
        x = torch.cat((q, k, v), dim=-1)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.n_heads), qkv
        )

        attention_map = q @ k.transpose(-1, -2) * self.scale
        if self.use_mask:
            # Assume masking is only done after a spatial split.
            batch_size, _, seq_len, _ = q.shape
            mask = torch.tril(torch.ones(seq_len, seq_len).to(x.device))
            mask = repeat(mask, "m n -> b h m n", b=batch_size, h=self.n_heads).to(
                torch.bool
            )
            condition = torch.tensor([-(2**15) + 1], dtype=torch.float32).to(x.device)
            attention_map = torch.where(mask, attention_map, condition)
        attention_map = f.softmax(attention_map, dim=-1)
        x = attention_map @ v
        x = rearrange(x, "b h n d -> b n (h d)")
        return self.to_out(x)

    def forward_single(self, x: torch.Tensor, **kwargs):
        is_testing = kwargs.get("is_testing", False)

        # Group attentions only during training/evaluation
        if not is_testing:
            b, _, d = x.shape
            partitions = kwargs.get("partitions")
            chunk_size = len(partitions[0])
            odd_chunk_size = len(partitions[-1])

            result = []

            for partition in partitions:
                x_ = x[:, partition, :]
                group_size = len(partition)

                # pad the smaller chunk if there is one
                if group_size == odd_chunk_size:
                    padding = chunk_size - odd_chunk_size
                    pad_tensor = torch.zeros((b, padding, d)).to(x.device)
                    x_ = torch.cat([x_, pad_tensor], dim=1)
                group_attention = self.forward(q=x_, k=x_, v=x_)
                result.append(group_attention)

            result = torch.cat(result, dim=1)
            r = torch.cat(partitions)

            # Revert the shuffling
            return result[:, torch.argsort(r), :]
        else:
            return self.forward(q=x, k=x, v=x)