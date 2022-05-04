from typing import Type

import einops
import torch
import torch.nn as nn
import torch.nn.functional as f
from einops import rearrange, repeat
from torch import Tensor


class ResidualNormFeedforward(nn.Module):
    """A module consisting of a 2-layer MLP, a layer normalization and a residual."""

    def __init__(
        self,
        d_hidden: int,
        d_feedforward: int,
        p_dropout: float,
        activation: Type[nn.Module],
    ) -> None:
        """
        Initialize the module.

        :param d_hidden: The input and output dimension of the MLP.
        :param d_feedforward: The hidden dimension of the MLP.
        :param p_dropout: The dropout probability.
        :param activation: The activation function between the MLP layers.
        """
        super(ResidualNormFeedforward, self).__init__()
        self.dropout = nn.Dropout(p_dropout)
        self.activation = activation()
        self.fc_in = nn.Linear(in_features=d_hidden, out_features=d_feedforward)
        self.fc_out = nn.Linear(in_features=d_feedforward, out_features=d_hidden)

        self.layer_norm = nn.LayerNorm(d_hidden)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward a tensor of shape (..., d_hidden) through the module.

        :param x: A tensor of shape (..., d_hidden).
        :return: A tensor of shape (..., d_hidden).
        """
        h = self.activation(self.fc_in(x))
        h = self.fc_out(self.dropout(h))
        return self.layer_norm(x + h)


class SpatialSplit(nn.Module):
    """A module for spatial splitting."""

    def __init__(self):
        """
        Initialize the module.

        This operation is complementary to the spatial merge.
        """
        super(SpatialSplit, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward a tensor of shape (B, N, T, D).

        The result is a tensor with the first and second dimensions merged.
        :param x: A tensor of shape (B, N, T, D).
        :return: A tensor of shape (BN, T, D).
        """
        # x (batch, nodes, time, dim)
        return einops.rearrange(x, "b n t d -> (b n) t d")


class SpatialMerge(nn.Module):
    """A module for spatial merging."""

    def __init__(self, batch_size: int) -> None:
        """
        Initialize the module.

        This operation is complementary to the spatial split.
        :param batch_size: The batch size of the module.
        """
        super(SpatialMerge, self).__init__()
        self.batch_size = batch_size

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward a tensor of shape (BN, T, D).

        The result is a tensor with the first dimensions split in two.
        :param x: A tensor of shape (BN, T, D).
        :return: A tensor of shape (B, N, T, D).
        """
        return einops.rearrange(x, "(b n) t d -> b n t d", b=self.batch_size)


class TemporalSplit(nn.Module):
    """A module for temporal splitting."""

    def __init__(self):
        """
        Initialize the module.

        This operation is complementary to the temporal merge.
        """
        super(TemporalSplit, self).__init__()

    def forward(self, x):
        """
        Forward a tensor of shape (B, N, T, D).

        The result is a tensor with the first and third dimensions merged.
        :param x: A tensor of shape (B, N, T, D).
        :return:  A tensor of shape (BT, N, D).
        """
        return einops.rearrange(x, "b n t d -> (b t) n d")


class TemporalMerge(nn.Module):
    """A module for temporal merging."""

    def __init__(self, batch_size: int) -> None:
        """
        Initialize the module.

        This operation is complementary to the temporal split.
        :param batch_size: The batch size of the module.
        """
        super(TemporalMerge, self).__init__()
        self.batch_size = batch_size

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward a tensor of shape (BT, N, D).

        The result is a tensor with the first dimensions split in two.
        :param x: A tensor of shape (BT, N, D).
        :return: A tensor of shape (B, N, T, D).
        """
        return einops.rearrange(x, "(b t) n d -> b n t d", b=self.batch_size)


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
            # Assume masking is only done after a spatial split, so data is (BN, T, D)
            batch_size, _, seq_len, _ = q.shape
            mask = torch.tril(torch.ones(seq_len, seq_len).to(x.device))
            mask = repeat(mask, "a b -> j n a b", j=batch_size, n=self.n_heads).to(
                torch.bool
            )
            condition = torch.tensor([-(2**15) + 1], dtype=torch.float32).to(x.device)
            attention_map = torch.where(mask, attention_map, condition)
        attention_map = f.softmax(attention_map, dim=-1)
        x = attention_map @ v
        x = rearrange(x, "b h n d -> b n (h d)")
        return self.to_out(x)


class SelfAttentionBlock(nn.Module):
    """A wrapper over the multi-head attention in case K, Q and V are the same."""

    def __init__(
        self,
        d_hidden: int,
        n_heads: int,
        p_dropout: float,
        use_mask: bool = False,
    ) -> None:
        """
        Initialize the module.

        :param d_hidden: The feature dimension.
        :param n_heads: The number of heads in the multi-head attention.
        :param p_dropout: The dropout probability.
        :param use_mask: Whether to use masking (for temporal attention - prevents
        attending to events in the future).
        """
        super(SelfAttentionBlock, self).__init__()
        self.attention_block = MultiHeadAttention(
            d_hidden=d_hidden, n_heads=n_heads, p_dropout=p_dropout, use_mask=use_mask
        )

        self.layer_norm_attention = nn.LayerNorm(d_hidden)

    def forward(self, x) -> Tensor:
        """
        Forward a tensor of shape (BT, N, D) or (BN, T, D).

        :param x: A tensor of shape (BT, N, D) or (BN, T, D) which is fed through
        three different fully-connected layers and used as Q, K and V in the attention.
        :return: A tensor of shape (BT, N, D) or (BT, N, D).
        """
        h = self.layer_norm_attention(x + self.attention_block(q=x, k=x, v=x))
        return h


class CrossAttentionBlock(nn.Module):
    """A wrapper over the multi-head attention in case K, Q, and V are different."""

    def __init__(
        self,
        d_hidden: int,
        n_heads: int,
        p_dropout: float,
        use_mask: bool = False,
    ):
        """
        Initialize the module.

        :param d_hidden: The feature dimension.
        :param n_heads: The number of heads in the multi-head attention.
        :param p_dropout: The dropout probability.
        :param use_mask: Whether to use masking (for temporal attention - prevents
        attending to events in the future).
        """
        super(CrossAttentionBlock, self).__init__()
        self.attention_block = MultiHeadAttention(
            d_hidden=d_hidden, n_heads=n_heads, p_dropout=p_dropout, use_mask=use_mask
        )

        self.layer_norm_attention = nn.LayerNorm(d_hidden)

    def forward(self, q, k, v) -> Tensor:
        """
        Forward three tensors of shape (BT, N, D) or (BN, T, D).

        :param q: The "queries" to compose the attention map with.
        :param k: The "keys" to compute the attention map with.
        :param v: The "values" to apply the attention map to.
        :return: A tensor of shape (BT, N, D) or (BN, T, D).
        """
        h = self.layer_norm_attention(v + self.attention_block(q=q, k=k, v=v))
        return h


class Encoder(nn.Module):
    """
    The encoder module.

    Consists of a series of spatio-temporal index manipulations and two attentions
    applied sequentially.
    """

    def __init__(
        self,
        d_hidden: int,
        d_feedforward: int,
        n_heads: int,
        p_dropout: float,
        batch_size: int,
    ) -> None:
        """
        Initialize the module.

        :param d_hidden: The feature dimension.
        :param d_feedforward: The hidden dimension of the MLP.
        :param n_heads: The number of heads in the multi-head attention.
        :param p_dropout: The dropout probability.
        :param batch_size: The batch size.
        """
        super(Encoder, self).__init__()
        self.spatial_split = SpatialSplit()

        self.temporal_attention = SelfAttentionBlock(
            d_hidden=d_hidden, n_heads=n_heads, p_dropout=p_dropout
        )

        self.temporal_feedforward = ResidualNormFeedforward(
            d_hidden=d_hidden,
            d_feedforward=d_feedforward,
            p_dropout=p_dropout,
            activation=nn.ReLU,
        )
        self.spatial_merge = SpatialMerge(batch_size=batch_size)

        self.temporal_split = TemporalSplit()

        self.spatial_attention = SelfAttentionBlock(
            d_hidden=d_hidden, n_heads=n_heads, p_dropout=p_dropout
        )

        self.spatial_feedforward = ResidualNormFeedforward(
            d_hidden=d_hidden,
            d_feedforward=d_feedforward,
            p_dropout=p_dropout,
            activation=nn.ReLU,
        )

        self.temporal_merge = TemporalMerge(batch_size=batch_size)

    def forward(self, source_features: Tensor) -> Tensor:
        """
        Forward a tensor of shape (B, N, T, D).

        :param source_features: A tensor of shape (B, N, T, D).
        :return: A tensor of shape (B, N, T, D), representing the encoded features.
        """
        # h (BN, T, D)
        hidden = self.spatial_split(source_features)

        # h (BN, T, D)
        hidden = self.temporal_attention(hidden)

        # h (BN, T, D)
        hidden = self.temporal_feedforward(hidden)

        # h (B, N, T, D)
        hidden = self.spatial_merge(hidden)

        # h (BT, N, D)
        hidden = self.temporal_split(hidden)

        # h (BT, N, D)
        hidden = self.spatial_attention(hidden)

        # h (BT, N, D)
        hidden = self.spatial_feedforward(hidden)

        # h (B, N, T, D)
        hidden = self.temporal_merge(hidden)

        return hidden


class Decoder(nn.Module):
    """
    The decoder module.

    Consists of a series of spatio-temporal index manipulations and three attentions
    applied sequentially.
    """

    def __init__(
        self,
        d_hidden: int,
        d_feedforward: int,
        n_heads: int,
        p_dropout: float,
        batch_size: int,
    ) -> None:
        """
        Initialize the module.

        :param d_hidden: The feature dimension.
        :param d_feedforward: The hidden dimension of the MLP.
        :param n_heads: The number of heads in the multi-head attention.
        :param p_dropout: The dropout probability.
        :param batch_size: The batch size.
        """
        super(Decoder, self).__init__()
        self.spatial_split = SpatialSplit()

        self.temporal_self_attention = SelfAttentionBlock(
            d_hidden=d_hidden, n_heads=n_heads, p_dropout=p_dropout, use_mask=True
        )

        self.temporal_cross_attention = CrossAttentionBlock(
            d_hidden=d_hidden, n_heads=n_heads, p_dropout=p_dropout
        )

        self.temporal_feedforward = ResidualNormFeedforward(
            d_hidden=d_hidden,
            d_feedforward=d_feedforward,
            p_dropout=p_dropout,
            activation=nn.ReLU,
        )

        self.spatial_merge = SpatialMerge(batch_size=batch_size)

        self.temporal_split = TemporalSplit()

        self.spatial_attention = SelfAttentionBlock(
            d_hidden=d_hidden, n_heads=n_heads, p_dropout=p_dropout
        )

        self.spatial_feedforward = ResidualNormFeedforward(
            d_hidden=d_hidden,
            d_feedforward=d_feedforward,
            p_dropout=p_dropout,
            activation=nn.ReLU,
        )

        self.temporal_merge = TemporalMerge(batch_size=batch_size)

    def forward(self, source_features: Tensor, target_features: Tensor) -> Tensor:
        """
        Forward a tensor of shape (B, N, T, D).

        :param source_features: A tensor of shape (B, N, T, D).
        :param target_features: A tensor of shape (B, N, T, D).
        :return: A tensor of shape (B, N, T, D), representing the encoded features.
        """
        # h (BN, T, D)
        source_features = self.spatial_split(source_features)

        # h (BN, T', D)
        target_features = self.spatial_split(target_features)

        # h (BN, T', D)
        target_features = self.temporal_self_attention(target_features)

        # todo - There is a dimension mismatch here...
        # h (BN, T, D)
        hidden = self.temporal_cross_attention(
            q=target_features, k=source_features, v=source_features
        )

        # h (BN, T, D)
        hidden = self.temporal_feedforward(hidden)

        # h (B, N, T, D)
        hidden = self.spatial_merge(hidden)

        # h (BT, N, D)
        hidden = self.temporal_split(hidden)

        # h (BT, N, D)
        hidden = self.spatial_attention(hidden)

        # h (BT, N, D)
        hidden = self.spatial_feedforward(hidden)

        # h (B, N, T, D)
        hidden = self.temporal_merge(hidden)

        return hidden


class PositionalEncoding(nn.Module):
    """
    A module for generating positional embeddings, as in the Transformer paper.

    https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py
    """

    def __init__(self, d_hidden: int, n_position: int = 500) -> None:
        """
        Initialize the module.

        :param d_hidden: The hidden dimension.
        :param n_position: The number of indices (positions) for which to generate.
        """
        super(PositionalEncoding, self).__init__()
        # Not a parameter
        self.register_buffer(
            "pos_table", self._get_sinusoid_encoding_table(d_hidden, n_position)
        )

    def _get_sinusoid_encoding_table(self, d_hidden, n_position) -> Tensor:
        """
        Generate positional embedding of a specific dimension for a number of positions.

        :param d_hidden: The hidden dimension.
        :param n_position: The number of positions (indices).
        :return: A tensor of shape (1, n_position, 1, d_hidden).
        """

        def get_position_angle_vec(position):
            base = torch.tensor([10000])
            exponent = torch.tensor(
                [2 * (h // 2) / d_hidden for h in range(d_hidden)], dtype=torch.float32
            )
            array = position / torch.pow(base, exponent)
            return array

        sinusoid_table = torch.stack(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
        )
        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        # positional_embeddings (1, pos, 1, d_hidden)
        positional_embeddings = sinusoid_table.unsqueeze(0).unsqueeze(-2)
        return positional_embeddings

    def forward(self, x):
        """
        Forward a tensor of shape (B, N, T, D).

        Adds the positional encodings to the tensor and return it.
        :param x: A tensor of shape (B, N, T, D).
        :return: A tensor of shape (B, N, T, D).
        """
        return x + self.pos_table[:, : x.size(1), :, :].clone().detach()


class ADN(nn.Module):
    """The Attention-Diffusion Network module."""

    def __init__(
        self,
        d_features: int,
        d_hidden: int,
        d_feedforward: int,
        n_heads: int,
        p_dropout: int,
        batch_size: int,
        n_blocks: int,
        n_nodes: int,
    ) -> None:
        """
        Initialize the module.

        :param d_features: The dimension of the input features (usually 1 or 2 i.e.
        speed / traffic flow reading values).
        :param d_hidden: The hidden dimension.
        :param d_feedforward: The hidden dimension of the MLP.
        :param n_heads: The number of heads in the multi-head attention.
        :param p_dropout: The dropout probability.
        :param batch_size: The batch size.
        :param n_blocks: The number of stacked encoder and decoder blocks.
        :param n_nodes: Number of nodes (spatial locations).
        """
        super(ADN, self).__init__()

        # Todo - Implement padding.
        self.positional_embedder = PositionalEncoding(d_hidden=d_hidden, n_position=500)

        self.feature_linear_in = nn.Linear(
            in_features=d_features, out_features=d_hidden
        )

        self.encoders = nn.ModuleList(
            [
                Encoder(
                    d_hidden=d_hidden,
                    d_feedforward=d_feedforward,
                    n_heads=n_heads,
                    p_dropout=p_dropout,
                    batch_size=batch_size,
                )
                for _ in range(n_blocks)
            ]
        )

        self.decoders = nn.ModuleList(
            [
                Decoder(
                    d_hidden=d_hidden,
                    d_feedforward=d_feedforward,
                    n_heads=n_heads,
                    p_dropout=p_dropout,
                    batch_size=batch_size,
                )
                for _ in range(n_blocks)
            ]
        )

        self.minute_interval_embedding = nn.Embedding(
            num_embeddings=288, embedding_dim=d_hidden
        )

        self.day_embedding = nn.Embedding(num_embeddings=7, embedding_dim=d_hidden)

        self.spatial_embedding = nn.Embedding(
            num_embeddings=n_nodes, embedding_dim=d_hidden
        )

        self.feature_linear_out = nn.Linear(
            in_features=d_hidden, out_features=d_features
        )

    def init(
        self, x: Tensor, temporal_descriptor: Tensor, spatial_descriptor: Tensor
    ) -> Tensor:
        """
        Initialize the features for the encoder and decoder.

        Equivalent of the ENC-INIT and DEC-INIT blocks from the paper.
        :param x: The features tensor of shape (B, N, T, D_in)
        :param temporal_descriptor: A tensor of shape (B, T, 2) which contains one-hot
        encoded information about the interval of the day and day of the week.
        :param spatial_descriptor: A tensor of shape (B, N, 1) which contains at
        least a consistent index of spatial locations in a sequence).
        :return: A tensor of shape (B, N, T, D) which is a result of adding the
        features from the descriptors, the positional encodings and the features.
        """
        b, t, _ = temporal_descriptor.shape
        _, n, _ = spatial_descriptor.shape

        # temporal_descriptor (B, T, 2)
        minute_index = temporal_descriptor[..., 0]
        day_index = temporal_descriptor[..., 1]

        minute_embedding = self.minute_interval_embedding(minute_index)

        # minute_embedding (B, N, T, D)
        minute_embedding = repeat(minute_embedding, "b t d -> b n t d", n=n)

        day_embedding = self.day_embedding(day_index)

        # day_embedding (B, N, T, D)
        day_embedding = repeat(day_embedding, "b t d -> b n t d", n=n)

        # spatial_descriptor (B, N, 1)
        spatial_embedding = self.spatial_embedding(torch.squeeze(spatial_descriptor))

        # spatial_embedding (B, N, T, D)
        spatial_embedding = repeat(spatial_embedding, "b n d -> b n t d", t=t)

        spatio_temporal_embedding = minute_embedding + day_embedding + spatial_embedding

        # embedding (B, N, T, D)
        embedding = self.positional_embedder(spatio_temporal_embedding)

        # feature (B, N, T, D)
        feature = self.feature_linear_in(x)

        return embedding + feature

    def forward(
        self,
        source_features: Tensor,
        source_temporal_descriptor: Tensor,
        source_spatial_descriptor: Tensor,
        target_features: Tensor,
        target_temporal_descriptor: Tensor,
        target_spatial_descriptor: Tensor,
    ) -> Tensor:
        """
        Forward the input tensors through the model.

        :param source_features: Features of the source sequence of shape (B, N, T, D).
        :param source_temporal_descriptor: Temporal descriptor of the source sequence
        of shape (B, T, 2).
        :param source_spatial_descriptor: Spatial descriptor of the source sequence
        of shape (B, N, 1).
        :param target_features: Features of the source sequence of shape (B, N, T, D).
        :param target_temporal_descriptor: Temporal descriptor of the source sequence
        of shape (B, T, 2).
        :param target_spatial_descriptor: Spatial descriptor of the source sequence
        of shape (B, N, 1).
        :return: A tensor of shape (B, N, T, D) corresponding to the prediction.
        """
        source_features = self.init(
            x=source_features,
            temporal_descriptor=source_temporal_descriptor,
            spatial_descriptor=source_spatial_descriptor,
        )

        target_features = self.init(
            x=target_features,
            temporal_descriptor=target_temporal_descriptor,
            spatial_descriptor=target_spatial_descriptor,
        )

        for encoder in self.encoders:
            source_features = encoder(source_features)

        for decoder in self.decoders:
            target_features = decoder(
                source_features=source_features, target_features=target_features
            )

        return self.feature_linear_out(target_features)
