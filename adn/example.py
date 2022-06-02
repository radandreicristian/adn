import torch
from einops import repeat

from adn import ADN

if __name__ == "__main__":

    d_features = 1
    d_hidden = 4
    d_feedforward = 4
    n_heads = 2
    p_dropout = 0
    batch_size = 2
    n_blocks = 1
    n_nodes = 315
    time_steps = 12

    config = {
        "d_features": d_features,
        "d_hidden": d_hidden,
        "d_feedforward": d_feedforward,
        "n_heads": n_heads,
        "p_dropout": p_dropout,
        "batch_size": batch_size,
        "n_blocks": n_blocks,
        "n_nodes": n_nodes,
    }

    model = ADN(**config)

    # spatial_descriptors - Each spatial location is an index 0...N
    spatial_range = torch.arange(start=0, end=n_nodes)

    src_spatial_descriptor = repeat(spatial_range, "n -> b n h", b=batch_size, h=1)

    tgt_spatial_descriptor = repeat(spatial_range, "n -> b n h", b=batch_size, h=1)

    # temporal_descriptor - At index [..., 0] - one-hot-encoding of 5-mins intervals
    # in a day. At index [..., 1] - one-hot-encoding of the day of the week.

    src_temporal_descriptor_interval_of_day = torch.randint(
        low=0,
        high=287,
        size=(batch_size, time_steps),
    )
    src_temporal_descriptor_day_of_week = torch.randint(
        low=0, high=6, size=(batch_size, time_steps)
    )

    tgt_temporal_descriptor_interval_of_day = torch.randint(
        low=0,
        high=287,
        size=(batch_size, time_steps),
    )
    tgt_temporal_descriptor_day_of_week = torch.randint(
        low=0, high=6, size=(batch_size, time_steps)
    )

    # features (B, N, T, d_features)
    src_features = torch.randn(batch_size, n_nodes, time_steps, d_features)
    tgt_features = torch.randn(batch_size, n_nodes, time_steps, d_features)

    predict = model(
        src_features=src_features,
        src_temporal_descriptor_interval_of_day=src_temporal_descriptor_interval_of_day,
        src_temporal_descriptor_day_of_week=src_temporal_descriptor_day_of_week,
        src_spatial_descriptor=src_spatial_descriptor,
        tgt_features=tgt_features,
        tgt_temporal_descriptor_interval_of_day=tgt_temporal_descriptor_interval_of_day,
        tgt_temporal_descriptor_day_of_week=tgt_temporal_descriptor_day_of_week,
        tgt_spatial_descriptor=tgt_spatial_descriptor,
    )

    print(predict.shape)
