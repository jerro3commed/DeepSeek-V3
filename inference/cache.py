"""KV-cache management for incremental decoding."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class CacheConfig:
    max_batch_size: int = 1
    max_seq_len: int = 4096  # increased from 2048 to support longer contexts
    num_layers: int = 64
    num_heads: int = 128
    head_dim: int = 128
    dtype: torch.dtype = torch.bfloat16
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if self.max_seq_len < 1:
            raise ValueError("max_seq_len must be >= 1")
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if self.head_dim < 1:
            raise ValueError("head_dim must be >= 1")


class KVCache:
    """Pre-allocated key/value cache for a single generation request batch."""

    def __init__(self, config: CacheConfig) -> None:
        self.config = config
        self._seq_lens: List[int] = [0] * config.max_batch_size
        shape = (
            config.num_layers,
            config.max_batch_size,
            config.max_seq_len,
            config.num_heads,
            config.head_dim,
        )
        self.k_cache: torch.Tensor = torch.zeros(shape, dtype=config.dtype, device=config.device)
        self.v_cache: torch.Tensor = torch.zeros(shape, dtype=config.dtype, device=config.device)

    def update(
        self,
        layer_idx: int,
        batch_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append new keys/values and return the full cached slice."""
        seq_len = self._seq_lens[batch_idx]
        new_len = keys.shape[0]
        if seq_len + new_len > self.config.max_seq_len:
            raise RuntimeError(
                f"Cache overflow: seq_len {seq_len + new_len} > max_seq_len {self.config.max_seq_len}"
            )
        self.k_cache[layer_idx, batch_idx, seq_len : seq_len + new_len] = keys
        self.v_cache[layer_idx, batch_idx, seq_len : seq_len + new_len] = values
        self._seq_lens[batch_idx] += new_len
        total = self._seq_lens[batch_idx]
        return (
            self.k_cache[layer_idx, batch_idx, :total],
            self.v_cache[layer_idx, batch_idx, :total],
        )

    def seq_len(self, batch_idx: int) -> int:
        return self._seq_lens[batch_idx]

    def reset(self, batch_idx: Optional[int] = None) -> None:
        """Clear cache for one or all batch entries."""
        if batch_idx is None:
            self._seq_lens = [0] * self.config.max_batch_size
            self.k_cache.zero_()
            self.v_cache.zero_()
        else:
            self._seq_lens[batch_idx] = 0
            self.k_cache[:, batch_idx].zero_()
            self.v_cache[:, batch_idx].zero_()
