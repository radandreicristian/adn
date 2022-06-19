from typing import Type

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import Tensor
from torch.nn.init import xavier_uniform_

from adn.attentions import AttentionFactory, FULL


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
        self.dropout_hidden = nn.Dropout(p_dropout)
        self.dropout_output = nn.Dropout(p_dropout)
        self.activation = activation()
        self.fc_in = nn.Linear(in_features=d_hidden, out_features=d_feedforward)
        self.fc_out = nn.Linear(in_features=d_feedforward, out_features=d_hidden)

        self.layer_norm = nn.LayerNorm(d_hidden)
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.fc_in.weight)
        xavier_uniform_(self.fc_out.weight)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward a tensor of shape (..., d_hidden) through the module.

        :param x: A tensor of shape (..., d_hidden).
        :return: A tensor of shape (..., d_hidden).
        """
        h = self.activation(self.fc_in(x))
        h = self.fc_out(self.dropout_hidden(h))
        return self.layer_norm(self.dropout_output(h) + x)


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
        return rearrange(x, "b n t d -> (b n) t d")


class SpatialMerge(nn.Module):
    """A module for spatial merging."""

    def __init__(self, spatial_seq_len: int) -> None:
        """
        Initialize the module.

        This operation is complementary to the spatial split.
        :param spatial_seq_len: The length of the spatial dimension.
        """
        super(SpatialMerge, self).__init__()
        self.spatial_seq_len = spatial_seq_len

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward a tensor of shape (BN, T, D).

        The result is a tensor with the first dimensions split in two.
        :param x: A tensor of shape (BN, T, D).
        :return: A tensor of shape (B, N, T, D).
        """
        return rearrange(x, "(b n) t d -> b n t d", n=self.spatial_seq_len)


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
        return rearrange(x, "b n t d -> (b t) n d")


class TemporalMerge(nn.Module):
    """A module for temporal merging."""

    def __init__(self, temporal_seq_len: int) -> None:
        """
        Initialize the module.

        This operation is complementary to the temporal split.
        :param temporal_seq_len: The length of the temporal dimension.
        """
        super(TemporalMerge, self).__init__()
        self.temporal_seq_len = temporal_seq_len

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward a tensor of shape (BT, N, D).

        The result is a tensor with the first dimensions split in two.
        :param x: A tensor of shape (BT, N, D).
        :return: A tensor of shape (B, N, T, D).
        """
        return rearrange(x, "(b t) n d -> b n t d", t=self.temporal_seq_len)


class SelfAttentionBlock(nn.Module):
    """
    A wrapper over the multi-head attention.

    Use this when Q, K and V should be the transformations of the same source tensor.
    """

    def __init__(
            self,
            d_hidden: int,
            n_heads: int,
            p_dropout: float,
            attention_type: str,
            use_mask: bool = False,
            **attention_kwargs,
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
        self.attention_block = AttentionFactory.build_attention(
            attention_type=attention_type,
            d_hidden=d_hidden,
            n_heads=n_heads,
            p_dropout=p_dropout,
            use_mask=use_mask,
            **attention_kwargs,
        )

        self.layer_norm_attention = nn.LayerNorm(d_hidden)

    def forward(self, x, **kwargs) -> Tensor:
        """
        Forward a tensor of shape (BT, N, D) or (BN, T, D).

        :param x: A tensor of shape (BT, N, D) or (BN, T, D) which is fed through
        three different fully-connected layers and used as Q, K and V in the attention.
        :return: A tensor of shape (BT, N, D) or (BT, N, D).
        """
        h = self.layer_norm_attention(x + self.attention_block.forward_single(x=x,
                                                                              **kwargs))
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

        # The cross-attention is only used on the temporal dimension. Here we always
        # use full-attention.
        self.attention_block = AttentionFactory.build_attention(
            attention_type=FULL,
            d_hidden=d_hidden,
            n_heads=n_heads,
            p_dropout=p_dropout,
            use_mask=use_mask,
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
        x = self.layer_norm_attention(q + self.attention_block(q=q, k=k, v=v))
        return x


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
            spatial_seq_len: int,
            temporal_seq_len: int,
            spatial_attention_type: str,
            **spatial_attention_kwargs,
    ) -> None:
        """
        Initialize the module.

        :param d_hidden: The feature dimension.
        :param d_feedforward: The hidden dimension of the MLP.
        :param n_heads: The number of heads in the multi-head attention.
        :param p_dropout: The dropout probability.
        :param spatial_seq_len: The length of the spatial dimension.
        :param temporal_seq_len: The length of the temporal dimension.
        """
        super(Encoder, self).__init__()
        self.spatial_split = SpatialSplit()

        self.temporal_attention = SelfAttentionBlock(
            d_hidden=d_hidden, n_heads=n_heads, p_dropout=p_dropout,
            attention_type=FULL, use_mask=True
        )

        self.spatial_merge = SpatialMerge(spatial_seq_len=spatial_seq_len)

        self.temporal_split = TemporalSplit()

        self.spatial_attention = SelfAttentionBlock(
            d_hidden=d_hidden, n_heads=n_heads, p_dropout=p_dropout,
            attention_type=spatial_attention_type, **spatial_attention_kwargs
        )

        self.feedforward_temporal = ResidualNormFeedforward(
            d_hidden=d_hidden,
            d_feedforward=d_feedforward,
            p_dropout=p_dropout,
            activation=nn.ReLU,
        )

        self.feedforward_spatial = ResidualNormFeedforward(
            d_hidden=d_hidden,
            d_feedforward=d_feedforward,
            p_dropout=p_dropout,
            activation=nn.ReLU,
        )

        self.temporal_merge = TemporalMerge(temporal_seq_len=temporal_seq_len)

    def forward(self, source_features: Tensor, **kwargs) -> Tensor:
        """
        Forward a tensor of shape (B, N, T, D).

        :param source_features: A tensor of shape (B, N, T, D).
        :return: A tensor of shape (B, N, T, D), representing the encoded features.
        """
        # h (BN, T, D)
        hidden = self.spatial_split(source_features)

        # h (BN, T, D)
        hidden = self.temporal_attention(hidden)

        hidden = self.feedforward_temporal(hidden)

        # h (B, N, T, D)
        hidden = self.spatial_merge(hidden)

        # h (BT, N, D)
        hidden = self.temporal_split(hidden)

        # h (BT, N, D)
        hidden = self.spatial_attention(hidden, **kwargs)

        hidden = self.feedforward_spatial(hidden)

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
            spatial_seq_len: int,
            temporal_seq_len: int,
            spatial_attention_type: str,
            **spatial_attention_kwargs,
    ) -> None:
        """
        Initialize the module.

        :param d_hidden: The feature dimension.
        :param d_feedforward: The hidden dimension of the MLP.
        :param n_heads: The number of heads in the multi-head attention.
        :param p_dropout: The dropout probability.
        :param spatial_seq_len: The length of the spatial dimension.
        :param temporal_seq_len: The length of the temporal dimension.
        """
        super(Decoder, self).__init__()
        self.spatial_split = SpatialSplit()

        self.temporal_self_attention = SelfAttentionBlock(
            d_hidden=d_hidden, n_heads=n_heads, p_dropout=p_dropout, use_mask=True,
            attention_type=FULL
        )

        self.temporal_cross_attention = CrossAttentionBlock(
            d_hidden=d_hidden, n_heads=n_heads, p_dropout=p_dropout, use_mask=True
        )

        self.spatial_merge = SpatialMerge(spatial_seq_len=spatial_seq_len)

        self.temporal_split = TemporalSplit()

        self.spatial_attention = SelfAttentionBlock(
            d_hidden=d_hidden, n_heads=n_heads, p_dropout=p_dropout,
            attention_type=spatial_attention_type, **spatial_attention_kwargs
        )

        self.feedforward_self_temporal = ResidualNormFeedforward(
            d_hidden=d_hidden,
            d_feedforward=d_feedforward,
            p_dropout=p_dropout,
            activation=nn.ReLU,
        )

        self.feedforward_cross_temporal = ResidualNormFeedforward(
            d_hidden=d_hidden,
            d_feedforward=d_feedforward,
            p_dropout=p_dropout,
            activation=nn.ReLU,
        )

        self.feedforward_spatial = ResidualNormFeedforward(
            d_hidden=d_hidden,
            d_feedforward=d_feedforward,
            p_dropout=p_dropout,
            activation=nn.ReLU,
        )

        self.temporal_merge = TemporalMerge(temporal_seq_len=temporal_seq_len)

    def forward(self, source_features: Tensor, target_features: Tensor, **kwargs
                ) -> Tensor:
        """
        Forward a tensor of shape (B, N, T, D).

        :param source_features: A tensor of shape (B, N, T, D).
        :param target_features: A tensor of shape (B, N, T, D).
        :return: A tensor of shape (B, N, T, D), representing the encoded features.
        """
        # source_features (BN, T, D)
        source_features = self.spatial_split(source_features)

        # target_features (BN, T', D)
        target_features = self.spatial_split(target_features)

        # target_features (BN, T', D)
        target_features = self.temporal_self_attention(target_features)

        # target_features (BN, T', D)
        target_features = self.feedforward_self_temporal(target_features)

        # hidden (BN, T, D)
        hidden = self.temporal_cross_attention(
            q=target_features, k=source_features, v=source_features
        )

        # hidden (BN, T, D)
        hidden = self.feedforward_cross_temporal(hidden)

        # hidden (B, N, T, D)
        hidden = self.spatial_merge(hidden)

        # hidden (BT, N, D)
        hidden = self.temporal_split(hidden)

        # hidden (BT, N, D)
        hidden = self.spatial_attention(hidden, **kwargs)

        # hidden (BT, N, D)
        hidden = self.feedforward_spatial(hidden)

        # hidden (B, N, T, D)
        hidden = self.temporal_merge(hidden)

        return hidden


class ADN(nn.Module):
    """The Attention-Diffusion Network module."""

    def __init__(
            self,
            d_features: int,
            d_hidden: int,
            d_feedforward: int,
            n_heads: int,
            p_dropout: int,
            n_blocks: int,
            spatial_seq_len: int,
            temporal_seq_len: int,
            spatial_attention_type: str,
            **spatial_attention_kwargs
    ) -> None:
        """
        Initialize the module.

        :param d_features: The dimension of the input features (usually 1 or 2 i.e.
        speed / traffic flow reading values).
        :param d_hidden: The hidden dimension.
        :param d_feedforward: The hidden dimension of the MLP.
        :param n_heads: The number of heads in the multi-head attention.
        :param p_dropout: The dropout probability.
        :param n_blocks: The number of stacked encoder and decoder blocks.
        :param spatial_seq_len: The length of the spatial dimension.
        :param temporal_seq_len: The length of the temporal dimension.
        """
        super(ADN, self).__init__()

        # Todo - Implement padding.

        self.partitions = None
        self.spatial_seq_len = spatial_seq_len

        self.dropout_embedding = nn.Dropout(p=p_dropout)
        self.layer_norm_embedding = nn.LayerNorm(d_hidden)

        self.feature_linear_in = nn.Linear(
            in_features=d_features, out_features=d_hidden, bias=False
        )

        self.encoders = nn.ModuleList(
            [
                Encoder(
                    d_hidden=d_hidden,
                    d_feedforward=d_feedforward,
                    n_heads=n_heads,
                    p_dropout=p_dropout,
                    spatial_seq_len=spatial_seq_len,
                    temporal_seq_len=temporal_seq_len,
                    spatial_attention_type=spatial_attention_type,
                    **spatial_attention_kwargs
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
                    spatial_seq_len=spatial_seq_len,
                    temporal_seq_len=temporal_seq_len,
                    spatial_attention_type=spatial_attention_type,
                    **spatial_attention_kwargs
                )
                for _ in range(n_blocks)
            ]
        )

        self.minute_interval_embedding = nn.Embedding(
            num_embeddings=288, embedding_dim=d_hidden
        )

        self.day_embedding = nn.Embedding(num_embeddings=7, embedding_dim=d_hidden)

        self.spatial_embedding = nn.Embedding(
            num_embeddings=spatial_seq_len, embedding_dim=d_hidden
        )

        self.feature_linear_out = nn.Linear(
            in_features=d_hidden, out_features=d_features, bias=False
        )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.day_embedding.weight)
        nn.init.xavier_uniform_(self.spatial_embedding.weight)
        nn.init.xavier_uniform_(self.minute_interval_embedding.weight)
        nn.init.xavier_uniform_(self.feature_linear_out.weight)
        nn.init.xavier_uniform_(self.feature_linear_in.weight)

    def set_partitions(self, n_partitions: int = 16):
        indices = torch.randperm(self.spatial_seq_len)
        self.partitions = torch.chunk(indices, chunks=n_partitions)

    def get_partitions(self):
        return self.partitions

    def init(
            self,
            x: Tensor,
            temporal_descriptor_interval_of_day: Tensor,
            temporal_descriptor_day_of_week: Tensor,
            spatial_descriptor: Tensor,
    ) -> Tensor:
        """
        Initialize the features for the encoder and decoder.

        Equivalent of the ENC-INIT and DEC-INIT blocks from the paper.
        :param x: The features tensor of shape (B, N, T, D_in)
        :param temporal_descriptor_interval_of_day: A tensor of shape (B, T) which
        contains one-hot encoded information about the interval of the day.
        :param temporal_descriptor_day_of_week: A tensor of shape (B, T) which
        contains one-hot encoded information about the day of the week.
        :param spatial_descriptor: A tensor of shape (B, N, 1) which contains at
        least a consistent index of spatial locations in a sequence).
        :return: A tensor of shape (B, N, T, D) which is a result of adding the
        features from the descriptors, the positional encodings and the features.
        """
        b, t, _ = temporal_descriptor_interval_of_day.shape
        n, _ = spatial_descriptor.shape

        # minute_embedding (B, N, T, D)
        minute_embedding = self.minute_interval_embedding(
            temporal_descriptor_interval_of_day
        )

        # day_embedding (B, N, T, D)
        day_embedding = self.day_embedding(temporal_descriptor_day_of_week)

        # spatial_descriptor (B, N, 1)
        spatial_descriptor = repeat(spatial_descriptor, "n t-> b n t", b=b)

        spatial_embedding = self.spatial_embedding(
            torch.squeeze(spatial_descriptor, dim=-1)
        )

        # spatial_embedding (B, N, T, D)
        spatio_temporal_embedding = minute_embedding + day_embedding + spatial_embedding

        # feature (B, N, T, D)
        feature = self.feature_linear_in(x)

        return self.layer_norm_embedding(self.dropout_embedding(
            spatio_temporal_embedding + feature))

    def forward(
            self,
            src_features: Tensor,
            src_interval_of_day: Tensor,
            src_day_of_week: Tensor,
            src_spatial_descriptor,
            tgt_features: Tensor,
            tgt_interval_of_day: Tensor,
            tgt_day_of_week: Tensor,
            tgt_spatial_descriptor: Tensor,
            **kwargs
    ) -> Tensor:
        """
        Forward the input tensors through the model.

        :param src_features: Features of the source sequence of shape (B, N, T, D).
        :param src_interval_of_day: Interval-of-day temporal
        descriptor of the source sequence of shape (B, T).
        :param src_day_of_week: Day-of-the-week temporal
        descriptor of the source sequence of shape (B, T).
        :param src_spatial_descriptor: Spatial descriptor of the source
        sequence of shape (B, N, 1)
        :param tgt_features: Features of the source sequence of shape (B, N, T, D).
        :param tgt_interval_of_day: Interval-of-day temporal
        descriptor of the source sequence of shape (B, T).
        :param tgt_day_of_week: Day-of-the-week temporal
        descriptor of the source sequence of shape (B, T).
        :param tgt_spatial_descriptor: Spatial descriptor of the source
        sequence of shape (B, N, 1)
        :return: A tensor of shape (B, N, T, D) corresponding to the prediction.
        """

        # source_features (B, N, T, D)
        source_features = self.init(
            x=src_features,
            temporal_descriptor_interval_of_day=src_interval_of_day,
            temporal_descriptor_day_of_week=src_day_of_week,
            spatial_descriptor=src_spatial_descriptor,
        )

        # target_features (B, N, T, D)
        target_features = self.init(
            x=tgt_features,
            temporal_descriptor_interval_of_day=tgt_interval_of_day,
            temporal_descriptor_day_of_week=tgt_day_of_week,
            spatial_descriptor=tgt_spatial_descriptor,
        )

        # source_features (B, N, T, D)
        for encoder in self.encoders:
            source_features = encoder(source_features, **kwargs)

        # target_features (B, N, T, D)
        for decoder in self.decoders:
            target_features = decoder(
                source_features=source_features, target_features=target_features,
                **kwargs
            )

        # (B, N, T, 1)
        return self.feature_linear_out(target_features)
