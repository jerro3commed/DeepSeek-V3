"""Unit tests for inference.cache."""

from __future__ import annotations

import pytest
import torch

from inference.cache import CacheConfig, KVCache


CPU_DTYPE = torch.float32


def make_config(**kwargs) -> CacheConfig:
    defaults = dict(
        max_batch_size=2,
        max_seq_len=16,
        num_layers=2,
        num_heads=4,
        head_dim=8,
        dtype=CPU_DTYPE,
        device="cpu",
    )
    defaults.update(kwargs)
    return CacheConfig(**defaults)


class TestCacheConfig:
    def test_defaults(self):
        cfg = CacheConfig(device="cpu")
        assert cfg.max_batch_size == 1
        assert cfg.max_seq_len == 2048

    def test_invalid_batch_size_raises(self):
        with pytest.raises(ValueError, match="max_batch_size"):
            CacheConfig(max_batch_size=0, device="cpu")

    def test_invalid_seq_len_raises(self):
        with pytest.raises(ValueError, match="max_seq_len"):
            CacheConfig(max_seq_len=-1, device="cpu")

    def test_invalid_head_dim_raises(self):
        with pytest.raises(ValueError, match="head_dim"):
            CacheConfig(head_dim=0, device="cpu")


class TestKVCache:
    def test_initial_seq_len(self):
        cache = KVCache(make_config())
        assert cache.seq_len(0) == 0
        assert cache.seq_len(1) == 0

    def test_update_returns_correct_slice(self):
        cache = KVCache(make_config())
        keys = torch.ones(3, 4, 8)
        vals = torch.ones(3, 4, 8) * 2
        k_out, v_out = cache.update(0, 0, keys, vals)
        assert k_out.shape == (3, 4, 8)
        assert v_out.shape == (3, 4, 8)
        assert cache.seq_len(0) == 3

    def test_update_accumulates(self):
        cache = KVCache(make_config())
        keys1 = torch.ones(2, 4, 8)
        vals1 = torch.ones(2, 4, 8)
        cache.update(0, 0, keys1, vals1)
        keys2 = torch.ones(3, 4, 8) * 2
        vals2 = torch.ones(3, 4, 8) * 2
        k_out, v_out = cache.update(0, 0, keys2, vals2)
        assert k_out.shape == (5, 4, 8)
        assert cache.seq_len(0) == 5

    def test_overflow_raises(self):
        cache = KVCache(make_config(max_seq_len=4))
        keys = torch.ones(5, 4, 8)
        vals = torch.ones(5, 4, 8)
        with pytest.raises(RuntimeError, match="Cache overflow"):
            cache.update(0, 0, keys, vals)

    def test_reset_single(self):
        cache = KVCache(make_config())
        cache.update(0, 0, torch.ones(3, 4, 8), torch.ones(3, 4, 8))
        cache.reset(batch_idx=0)
        assert cache.seq_len(0) == 0

    def test_reset_all(self):
        cache = KVCache(make_config())
        cache.update(0, 0, torch.ones(2, 4, 8), torch.ones(2, 4, 8))
        cache.update(0, 1, torch.ones(3, 4, 8), torch.ones(3, 4, 8))
        cache.reset()
        assert cache.seq_len(0) == 0
        assert cache.seq_len(1) == 0
