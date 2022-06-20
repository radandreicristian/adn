import torch.nn as nn
from linformer import LinformerSelfAttention
import torch


class LinformerAttention(nn.Module):
    def __init__(
        self, d_hidden: int, n_heads: int, p_dropout: float, use_mask: bool = False,
            **kwargs
    ) -> None:
        super(LinformerAttention, self).__init__()
        self.d_hidden = d_hidden

        assert self.d_hidden % n_heads == 0, (
            "Hidden size not divisible by number of " "heads."
        )
        k = kwargs.get("k")
        seq_len = kwargs.get("n_nodes")

        self.n_heads = n_heads
        self.linear_self_attention = LinformerSelfAttention(
            dim=self.d_hidden,
            heads=n_heads,
            seq_len=seq_len,
            k=k,
            one_kv_head=True
        )

    def forward(self, x: torch.Tensor):
        # features (batch, seq, n_nodes, d_hidden_feat+d_hidden_pos)
        return self.linear_self_attention(x)

    def forward_single(self, x: torch.Tensor, **kwargs):
        return self.forward(x=x)
