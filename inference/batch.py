"""Batch management for inference requests.

Provides utilities for grouping multiple generation requests into
efficient batches, respecting hardware constraints and priority.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

from .sampling import SamplingParams
from .stopping import StoppingCriteria


@dataclass
class GenerationRequest:
    """A single generation request with its associated parameters."""

    request_id: str
    prompt_tokens: List[int]
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    stopping_criteria: StoppingCriteria = field(default_factory=StoppingCriteria)
    arrival_time: float = field(default_factory=time.monotonic)
    priority: int = 0  # Higher value = higher priority

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be a non-empty string")
        if not self.prompt_tokens:
            raise ValueError("prompt_tokens must be a non-empty list")
        if not isinstance(self.priority, int):
            raise TypeError("priority must be an integer")

    @property
    def prompt_length(self) -> int:
        """Number of tokens in the prompt."""
        return len(self.prompt_tokens)


@dataclass
class BatchConfig:
    """Configuration controlling how requests are batched together.

    Personal note: lowered max_batch_size to 4 and max_tokens_per_batch to 2048
    to better fit my local GPU (8 GB VRAM). Also bumped max_waiting_time slightly
    to 0.1 s so partial batches aren't flushed too eagerly on slower hardware.
    """

    max_batch_size: int = 4          # reduced from 8 for local GPU
    max_tokens_per_batch: int = 2048  # reduced from 4096 for local GPU
    max_waiting_time: float = 0.1    # increased from 0.05 s

    def __post_init__(self) -> None:
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if self.max_tokens_per_batch < 1:
            raise ValueError("max_tokens_per_batch must be >= 1")
        if self.max_waiting_time < 0.0:
            raise ValueError("max_waiting_time must be >= 0")


class RequestBatch:
    """An immutable snapshot of requests grouped for a single forward pass."""

    def __init__(self, requests: List[GenerationRequest]) -> None:
        if not requests:
            raise ValueError("A batch must contain at least one request")
        self._requests: List[GenerationRequest] = list(requests)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of requests in the batch."""
        return len(self._requests)

    @property
    def requests(self) -> List[GenerationRequest]:
        """Read-only view of the requests."""
        return list(self._requests)

    @property
    def total_prompt_tokens(self) -> int:
        """Sum of prompt token counts across all requests."""
        return sum(r.prompt_length for r in self._requests)

    def request_ids(self) -> List[str]:
        """Return the IDs of all requests in this batch."""
        return [r.request_id for r in self._requests]
