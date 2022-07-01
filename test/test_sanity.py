import time
import unittest
import torch
from adn import ADN
from einops import repeat


class ShapeSanityTestCase(unittest.TestCase):
    d_features = 1
    d_hidden = 32
    d_feedforward = 256
    n_heads = 2
    p_dropout = 0.3
    batch_size = 8
    n_blocks = 3
    spatial_seq_len = 325
    temporal_seq_len = 12

    def test_full(self):

        attention_type = "full"

        model_config = {
            "d_features": self.d_features,
            "d_hidden": self.d_hidden,
            "d_feedforward": self.d_feedforward,
            "n_heads": self.n_heads,
            "p_dropout": self.p_dropout,
            "spatial_seq_len": self.spatial_seq_len,
            "temporal_seq_len": self.temporal_seq_len,
            "n_blocks": self.n_blocks,
            "spatial_attention_type": attention_type
        }

        attention_config = {

        }

        model = ADN(**model_config, **attention_config)

        # spatial_descriptors - Each spatial location is an index 0...N
        spatial_range = torch.arange(start=0, end=self.spatial_seq_len)

        src_spatial_descriptor = repeat(spatial_range, "n -> n t",
                                        t=self.temporal_seq_len)

        tgt_spatial_descriptor = repeat(spatial_range, "n -> n t",
                                        t=self.temporal_seq_len)

        src_temporal_descriptor_interval_of_day = torch.randint(
            low=0,
            high=287,
            size=(self.batch_size, self.spatial_seq_len, self.temporal_seq_len),
        )
        src_temporal_descriptor_day_of_week = torch.randint(
            low=0, high=6, size=(self.batch_size, self.spatial_seq_len,
                                 self.temporal_seq_len)
        )

        tgt_temporal_descriptor_interval_of_day = torch.randint(
            low=0,
            high=287,
            size=(self.batch_size, self.spatial_seq_len, self.temporal_seq_len),
        )
        tgt_temporal_descriptor_day_of_week = torch.randint(
            low=0, high=6, size=(self.batch_size, self.spatial_seq_len,
                                 self.temporal_seq_len)
        )

        # features (B, N, T, d_features)
        src_features = torch.randn(self.batch_size, self.spatial_seq_len,
                                   self.temporal_seq_len,self.d_features)
        tgt_features = torch.randn(self.batch_size, self.spatial_seq_len,
                                   self.temporal_seq_len,self.d_features)

        start = time.time()

        forward_kwargs = {}

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
        print(f"Runtime {attention_type}: {(finish - start):.2f}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.numel())
        """
        print(f"Total Parameters {attention_type}: {total_params}")

        expected_shape = (self.batch_size, self.spatial_seq_len,
                          self.temporal_seq_len, self.d_features)
        assert(predict.shape == expected_shape)

    def test_group(self):

        attention_type = "group"

        model_config = {
            "d_features": self.d_features,
            "d_hidden": self.d_hidden,
            "d_feedforward": self.d_feedforward,
            "n_heads": self.n_heads,
            "p_dropout": self.p_dropout,
            "spatial_seq_len": self.spatial_seq_len,
            "temporal_seq_len": self.temporal_seq_len,
            "n_blocks": self.n_blocks,
            "spatial_attention_type": attention_type
        }

        attention_config = {

        }

        model = ADN(**model_config, **attention_config)

        # spatial_descriptors - Each spatial location is an index 0...N
        spatial_range = torch.arange(start=0, end=self.spatial_seq_len)

        src_spatial_descriptor = repeat(spatial_range, "n -> n t",
                                        t=self.temporal_seq_len)

        tgt_spatial_descriptor = repeat(spatial_range, "n -> n t",
                                        t=self.temporal_seq_len)

        src_temporal_descriptor_interval_of_day = torch.randint(
            low=0,
            high=287,
            size=(self.batch_size, self.spatial_seq_len, self.temporal_seq_len),
        )
        src_temporal_descriptor_day_of_week = torch.randint(
            low=0, high=6, size=(self.batch_size, self.spatial_seq_len,
                                 self.temporal_seq_len)
        )

        tgt_temporal_descriptor_interval_of_day = torch.randint(
            low=0,
            high=287,
            size=(self.batch_size, self.spatial_seq_len, self.temporal_seq_len),
        )
        tgt_temporal_descriptor_day_of_week = torch.randint(
            low=0, high=6, size=(self.batch_size, self.spatial_seq_len,
                                 self.temporal_seq_len)
        )

        # features (B, N, T, d_features)
        src_features = torch.randn(self.batch_size, self.spatial_seq_len,
                                   self.temporal_seq_len,self.d_features)
        tgt_features = torch.randn(self.batch_size, self.spatial_seq_len,
                                   self.temporal_seq_len,self.d_features)

        start = time.time()

        model.set_partitions()

        forward_kwargs = {"partitions": model.get_partitions()}

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
        print(f"Runtime {attention_type}: {(finish - start):.2f}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Parameters {attention_type}: {total_params}")

        expected_shape = (self.batch_size, self.spatial_seq_len,
                          self.temporal_seq_len, self.d_features)
        assert(predict.shape == expected_shape)

    def test_lsh(self):
        attention_type = "lsh"

        model_config = {
            "d_features": self.d_features,
            "d_hidden": self.d_hidden,
            "d_feedforward": self.d_feedforward,
            "n_heads": self.n_heads,
            "p_dropout": self.p_dropout,
            "spatial_seq_len": self.spatial_seq_len,
            "temporal_seq_len": self.temporal_seq_len,
            "n_blocks": self.n_blocks,
            "spatial_attention_type": attention_type
        }

        attention_config = {"bucket_size": 8, "n_hashes": 8}

        model = ADN(**model_config, **attention_config)

        # spatial_descriptors - Each spatial location is an index 0...N
        spatial_range = torch.arange(start=0, end=self.spatial_seq_len)

        src_spatial_descriptor = repeat(spatial_range, "n -> n t",
                                        t=self.temporal_seq_len)

        tgt_spatial_descriptor = repeat(spatial_range, "n -> n t",
                                        t=self.temporal_seq_len)

        src_temporal_descriptor_interval_of_day = torch.randint(
            low=0,
            high=287,
            size=(self.batch_size, self.spatial_seq_len, self.temporal_seq_len),
        )
        src_temporal_descriptor_day_of_week = torch.randint(
            low=0, high=6, size=(self.batch_size, self.spatial_seq_len,
                                 self.temporal_seq_len)
        )

        tgt_temporal_descriptor_interval_of_day = torch.randint(
            low=0,
            high=287,
            size=(self.batch_size, self.spatial_seq_len, self.temporal_seq_len),
        )
        tgt_temporal_descriptor_day_of_week = torch.randint(
            low=0, high=6, size=(self.batch_size, self.spatial_seq_len,
                                 self.temporal_seq_len)
        )

        # features (B, N, T, d_features)
        src_features = torch.randn(self.batch_size, self.spatial_seq_len,
                                   self.temporal_seq_len, self.d_features)
        tgt_features = torch.randn(self.batch_size, self.spatial_seq_len,
                                   self.temporal_seq_len, self.d_features)

        start = time.time()

        forward_kwargs = {}

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
        print(f"Runtime {attention_type}: {(finish - start):.2f}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Parameters {attention_type}: {total_params}")

        expected_shape = (self.batch_size, self.spatial_seq_len,
                          self.temporal_seq_len, self.d_features)
        assert (predict.shape == expected_shape)

    def test_linear(self):
        attention_type = "linear"

        model_config = {
            "d_features": self.d_features,
            "d_hidden": self.d_hidden,
            "d_feedforward": self.d_feedforward,
            "n_heads": self.n_heads,
            "p_dropout": self.p_dropout,
            "spatial_seq_len": self.spatial_seq_len,
            "temporal_seq_len": self.temporal_seq_len,
            "n_blocks": self.n_blocks,
            "spatial_attention_type": attention_type
        }

        attention_config = {"eps": 0.1}

        model = ADN(**model_config, **attention_config)

        # spatial_descriptors - Each spatial location is an index 0...N
        spatial_range = torch.arange(start=0, end=self.spatial_seq_len)

        src_spatial_descriptor = repeat(spatial_range, "n -> n t",
                                        t=self.temporal_seq_len)

        tgt_spatial_descriptor = repeat(spatial_range, "n -> n t",
                                        t=self.temporal_seq_len)

        src_temporal_descriptor_interval_of_day = torch.randint(
            low=0,
            high=287,
            size=(self.batch_size, self.spatial_seq_len, self.temporal_seq_len),
        )
        src_temporal_descriptor_day_of_week = torch.randint(
            low=0, high=6, size=(self.batch_size, self.spatial_seq_len,
                                 self.temporal_seq_len)
        )

        tgt_temporal_descriptor_interval_of_day = torch.randint(
            low=0,
            high=287,
            size=(self.batch_size, self.spatial_seq_len, self.temporal_seq_len),
        )
        tgt_temporal_descriptor_day_of_week = torch.randint(
            low=0, high=6, size=(self.batch_size, self.spatial_seq_len,
                                 self.temporal_seq_len)
        )

        # features (B, N, T, d_features)
        src_features = torch.randn(self.batch_size, self.spatial_seq_len,
                                   self.temporal_seq_len, self.d_features)
        tgt_features = torch.randn(self.batch_size, self.spatial_seq_len,
                                   self.temporal_seq_len, self.d_features)

        start = time.time()

        forward_kwargs = {}

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
        print(f"Runtime {attention_type}: {(finish - start):.2f}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Parameters {attention_type}: {total_params}")

        expected_shape = (self.batch_size, self.spatial_seq_len,
                          self.temporal_seq_len, self.d_features)
        assert (predict.shape == expected_shape)

    def test_efficient(self):
        attention_type = "efficient"

        model_config = {
            "d_features": self.d_features,
            "d_hidden": self.d_hidden,
            "d_feedforward": self.d_feedforward,
            "n_heads": self.n_heads,
            "p_dropout": self.p_dropout,
            "spatial_seq_len": self.spatial_seq_len,
            "temporal_seq_len": self.temporal_seq_len,
            "n_blocks": self.n_blocks,
            "spatial_attention_type": attention_type
        }

        attention_config = {}

        model = ADN(**model_config, **attention_config)

        # spatial_descriptors - Each spatial location is an index 0...N
        spatial_range = torch.arange(start=0, end=self.spatial_seq_len)

        src_spatial_descriptor = repeat(spatial_range, "n -> n t",
                                        t=self.temporal_seq_len)

        tgt_spatial_descriptor = repeat(spatial_range, "n -> n t",
                                        t=self.temporal_seq_len)

        src_temporal_descriptor_interval_of_day = torch.randint(
            low=0,
            high=287,
            size=(self.batch_size, self.spatial_seq_len, self.temporal_seq_len),
        )
        src_temporal_descriptor_day_of_week = torch.randint(
            low=0, high=6, size=(self.batch_size, self.spatial_seq_len,
                                 self.temporal_seq_len)
        )

        tgt_temporal_descriptor_interval_of_day = torch.randint(
            low=0,
            high=287,
            size=(self.batch_size, self.spatial_seq_len, self.temporal_seq_len),
        )
        tgt_temporal_descriptor_day_of_week = torch.randint(
            low=0, high=6, size=(self.batch_size, self.spatial_seq_len,
                                 self.temporal_seq_len)
        )

        # features (B, N, T, d_features)
        src_features = torch.randn(self.batch_size, self.spatial_seq_len,
                                   self.temporal_seq_len, self.d_features)
        tgt_features = torch.randn(self.batch_size, self.spatial_seq_len,
                                   self.temporal_seq_len, self.d_features)

        start = time.time()

        forward_kwargs = {}

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
        print(f"Runtime {attention_type}: {(finish - start):.2f}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Parameters {attention_type}: {total_params}")

        expected_shape = (self.batch_size, self.spatial_seq_len,
                          self.temporal_seq_len, self.d_features)
        assert (predict.shape == expected_shape)

    def test_favorplus(self):
        attention_type = "favorplus"

        model_config = {
            "d_features": self.d_features,
            "d_hidden": self.d_hidden,
            "d_feedforward": self.d_feedforward,
            "n_heads": self.n_heads,
            "p_dropout": self.p_dropout,
            "spatial_seq_len": self.spatial_seq_len,
            "temporal_seq_len": self.temporal_seq_len,
            "n_blocks": self.n_blocks,
            "spatial_attention_type": attention_type
        }

        attention_config = {}

        model = ADN(**model_config, **attention_config)

        # spatial_descriptors - Each spatial location is an index 0...N
        spatial_range = torch.arange(start=0, end=self.spatial_seq_len)

        src_spatial_descriptor = repeat(spatial_range, "n -> n t",
                                        t=self.temporal_seq_len)

        tgt_spatial_descriptor = repeat(spatial_range, "n -> n t",
                                        t=self.temporal_seq_len)

        src_temporal_descriptor_interval_of_day = torch.randint(
            low=0,
            high=287,
            size=(self.batch_size, self.spatial_seq_len, self.temporal_seq_len),
        )
        src_temporal_descriptor_day_of_week = torch.randint(
            low=0, high=6, size=(self.batch_size, self.spatial_seq_len,
                                 self.temporal_seq_len)
        )

        tgt_temporal_descriptor_interval_of_day = torch.randint(
            low=0,
            high=287,
            size=(self.batch_size, self.spatial_seq_len, self.temporal_seq_len),
        )
        tgt_temporal_descriptor_day_of_week = torch.randint(
            low=0, high=6, size=(self.batch_size, self.spatial_seq_len,
                                 self.temporal_seq_len)
        )

        # features (B, N, T, d_features)
        src_features = torch.randn(self.batch_size, self.spatial_seq_len,
                                   self.temporal_seq_len, self.d_features)
        tgt_features = torch.randn(self.batch_size, self.spatial_seq_len,
                                   self.temporal_seq_len, self.d_features)

        start = time.time()

        forward_kwargs = {}

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
        print(f"Runtime {attention_type}: {(finish - start):.2f}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Parameters {attention_type}: {total_params}")

        expected_shape = (self.batch_size, self.spatial_seq_len,
                          self.temporal_seq_len, self.d_features)
        assert (predict.shape == expected_shape)

    def test_linformer(self):
        attention_type = "linformer"

        model_config = {
            "d_features": self.d_features,
            "d_hidden": self.d_hidden,
            "d_feedforward": self.d_feedforward,
            "n_heads": self.n_heads,
            "p_dropout": self.p_dropout,
            "spatial_seq_len": self.spatial_seq_len,
            "temporal_seq_len": self.temporal_seq_len,
            "n_blocks": self.n_blocks,
            "spatial_attention_type": attention_type
        }

        attention_config = {
            "k": 16,
            "seq_len": self.spatial_seq_len
        }

        model = ADN(**model_config, **attention_config)

        # spatial_descriptors - Each spatial location is an index 0...N
        spatial_range = torch.arange(start=0, end=self.spatial_seq_len)

        src_spatial_descriptor = repeat(spatial_range, "n -> n t",
                                        t=self.temporal_seq_len)

        tgt_spatial_descriptor = repeat(spatial_range, "n -> n t",
                                        t=self.temporal_seq_len)

        src_temporal_descriptor_interval_of_day = torch.randint(
            low=0,
            high=287,
            size=(self.batch_size, self.spatial_seq_len, self.temporal_seq_len),
        )
        src_temporal_descriptor_day_of_week = torch.randint(
            low=0, high=6, size=(self.batch_size, self.spatial_seq_len,
                                 self.temporal_seq_len)
        )

        tgt_temporal_descriptor_interval_of_day = torch.randint(
            low=0,
            high=287,
            size=(self.batch_size, self.spatial_seq_len, self.temporal_seq_len),
        )
        tgt_temporal_descriptor_day_of_week = torch.randint(
            low=0, high=6, size=(self.batch_size, self.spatial_seq_len,
                                 self.temporal_seq_len)
        )

        # features (B, N, T, d_features)
        src_features = torch.randn(self.batch_size, self.spatial_seq_len,
                                   self.temporal_seq_len, self.d_features)
        tgt_features = torch.randn(self.batch_size, self.spatial_seq_len,
                                   self.temporal_seq_len, self.d_features)

        start = time.time()

        forward_kwargs = {}

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
        print(f"Runtime {attention_type}: {(finish - start):.2f}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Parameters {attention_type}: {total_params}")

        expected_shape = (self.batch_size, self.spatial_seq_len,
                          self.temporal_seq_len, self.d_features)
        assert (predict.shape == expected_shape)

if __name__ == '__main__':
    unittest.main()
