import torch.nn as nn
from linformer import LinformerSelfAttention
import torch
import torch.nn.functional as f


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
        seq_len = kwargs.get("seq_len")

        self.n_heads = n_heads
        self.linear_self_attention = LinformerSelfAttention(
            dim=self.d_hidden,
            heads=self.n_heads,
            seq_len=seq_len,
            k=k,
            one_kv_head=True
        )

        if n_heads == 1:
            self.to_out = nn.Identity()
        else:
            self.to_out = nn.Sequential(
                nn.Linear(in_features=d_hidden, out_features=d_hidden),
                nn.Dropout(p=p_dropout),
            )
        self.fc_out = nn.Linear(in_features=self.d_hidden, out_features=d_hidden)

    def forward(self, x: torch.Tensor):
        # features (batch, seq, n_nodes, d_hidden_feat+d_hidden_pos)
        h = self.linear_self_attention(x)
        return f.relu(self.fc_out(h))

    def forward_single(self, x: torch.Tensor, **kwargs):
        return self.forward(x=x)
