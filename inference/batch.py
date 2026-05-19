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
    """Configuration controlling how requests are batched together."""

    max_batch_size: int = 8
    max_tokens_per_batch: int = 4096
    max_waiting_time: float = 0.05  # seconds before flushing a partial batch

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


class BatchScheduler:
    """Greedy scheduler that groups pending requests into batches.

    Requests are sorted by (priority DESC, arrival_time ASC) before
    being packed into batches that respect ``BatchConfig`` limits.
    """

    def __init__(self, config: Optional[BatchConfig] = None) -> None:
        self.config = config or BatchConfig()
        self._queue: List[GenerationRequest] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, request: GenerationRequest) -> None:
        """Enqueue a request for scheduling."""
        self._queue.append(request)

    def pending(self) -> int:
        """Number of requests currently waiting to be batched."""
        return len(self._queue)

    def next_batch(self) -> Optional[RequestBatch]:
        """Build and return the next batch, or ``None`` if the queue is empty.

        Requests are selected greedily in priority/arrival order until either
        ``max_batch_size`` or ``max_tokens_per_batch`` would be exceeded.
        """
        if not self._queue:
            return None

        # Sort: higher priority first, then earlier arrival first.
        self._queue.sort(key=lambda r: (-r.priority, r.arrival_time))

        selected: List[GenerationRequest] = []
        token_budget = self.config.max_tokens_per_batch

        for request in self._queue:
            if len(selected) >= self.config.max_batch_size:
                break
            if request.prompt_length > token_budget:
                # Skip requests that would overflow the token budget.
                continue
            selected.append(request)
            token_budget -= request.prompt_length

        if not selected:
            return None

        # Remove selected requests from the queue.
        selected_ids = {r.request_id for r in selected}
        self._queue = [r for r in self._queue if r.request_id not in selected_ids]

        return RequestBatch(selected)
