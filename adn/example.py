import time

import torch
from einops import repeat

from adn import ADN

if __name__ == "__main__":

    d_features = 1
    d_hidden = 32
    d_feedforward = 256
    n_heads = 2
    p_dropout = 0.3
    batch_size = 8
    n_blocks = 3
    spatial_seq_len = 315
    temporal_seq_len = 12

    config = {
        "d_features": d_features,
        "d_hidden": d_hidden,
        "d_feedforward": d_feedforward,
        "n_heads": n_heads,
        "p_dropout": p_dropout,
        "spatial_seq_len": spatial_seq_len,
        "temporal_seq_len": temporal_seq_len,
        "n_blocks": n_blocks,
        "spatial_attention_type": "group"
    }

    model = ADN(**config)

    # spatial_descriptors - Each spatial location is an index 0...N
    spatial_range = torch.arange(start=0, end=spatial_seq_len)

    src_spatial_descriptor = repeat(spatial_range, "n -> n t", t=temporal_seq_len)

    tgt_spatial_descriptor = repeat(spatial_range, "n -> n t", t=temporal_seq_len)

    src_temporal_descriptor_interval_of_day = torch.randint(
        low=0,
        high=287,
        size=(batch_size, spatial_seq_len, temporal_seq_len),
    )
    src_temporal_descriptor_day_of_week = torch.randint(
        low=0, high=6, size=(batch_size, spatial_seq_len, temporal_seq_len)
    )

    tgt_temporal_descriptor_interval_of_day = torch.randint(
        low=0,
        high=287,
        size=(batch_size, spatial_seq_len, temporal_seq_len),
    )
    tgt_temporal_descriptor_day_of_week = torch.randint(
        low=0, high=6, size=(batch_size, spatial_seq_len, temporal_seq_len)
    )

    # features (B, N, T, d_features)
    src_features = torch.randn(batch_size, spatial_seq_len, temporal_seq_len, d_features)
    tgt_features = torch.randn(batch_size, spatial_seq_len, temporal_seq_len, d_features)

    start = time.time()

    forward_kwargs = {"partitions": model.make_partitions()}

    predict = model(
        src_features=src_features,
        src_interval_of_day=src_temporal_descriptor_interval_of_day,
        src_day_of_week=src_temporal_descriptor_day_of_week,
        src_spatial_descriptor=src_spatial_descriptor,
        tgt_features=tgt_features,
        tgt_interval_of_day=tgt_temporal_descriptor_interval_of_day,
        tgt_day_of_week=tgt_temporal_descriptor_day_of_week,
        tgt_spatial_descriptor=tgt_spatial_descriptor,
        **forward_kwargs
    )

    finish = time.time()
    print(finish - start)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(total_params)
    print(predict.shape)
