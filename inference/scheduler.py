"""Batch scheduling logic for managing generation requests.

This module provides a simple priority-based scheduler that groups
pending GenerationRequests into batches for efficient inference.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

from .batch import BatchConfig, BatchManager, GenerationRequest


@dataclass
class SchedulerConfig:
    """Configuration for the request scheduler."""

    max_batch_size: int = 8
    """Maximum number of requests to group into a single batch."""

    max_waiting_time_ms: float = 50.0
    """Maximum time (ms) a request may wait before it is force-scheduled."""

    max_queue_size: int = 256
    """Hard cap on the number of requests that may sit in the queue."""

    def __post_init__(self) -> None:
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if self.max_waiting_time_ms < 0:
            raise ValueError("max_waiting_time_ms must be non-negative")
        if self.max_queue_size < self.max_batch_size:
            raise ValueError(
                "max_queue_size must be >= max_batch_size"
            )


@dataclass
class _QueueEntry:
    """Internal wrapper that tracks when a request entered the queue."""

    request: GenerationRequest
    enqueue_time: float = field(default_factory=time.monotonic)


class Scheduler:
    """FIFO scheduler with configurable batching and wait-time limits.

    Example::

        cfg = SchedulerConfig(max_batch_size=4)
        scheduler = Scheduler(cfg)
        scheduler.add(request)
        if scheduler.has_batch():
            batch_manager = scheduler.next_batch()
    """

    def __init__(self, config: Optional[SchedulerConfig] = None) -> None:
        self._config = config or SchedulerConfig()
        self._queue: Deque[_QueueEntry] = deque()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def config(self) -> SchedulerConfig:
        return self._config

    @property
    def queue_size(self) -> int:
        """Number of requests currently waiting in the queue."""
        return len(self._queue)

    def add(self, request: GenerationRequest) -> None:
        """Enqueue a single generation request.

        Raises:
            OverflowError: If the queue has reached ``max_queue_size``.
        """
        if len(self._queue) >= self._config.max_queue_size:
            raise OverflowError(
                f"Scheduler queue is full ({self._config.max_queue_size} requests)"
            )
        self._queue.append(_QueueEntry(request=request))

    def has_batch(self) -> bool:
        """Return True when at least one batch is ready to be dispatched.

        A batch is considered ready when *either*:
        - The queue holds ``max_batch_size`` or more requests, **or**
        - The oldest queued request has waited longer than
          ``max_waiting_time_ms``.
        """
        if not self._queue:
            return False
        if len(self._queue) >= self._config.max_batch_size:
            return True
        oldest = self._queue[0]
        elapsed_ms = (time.monotonic() - oldest.enqueue_time) * 1_000
        return elapsed_ms >= self._config.max_waiting_time_ms

    def next_batch(self) -> BatchManager:
        """Dequeue up to ``max_batch_size`` requests and return a BatchManager.

        Raises:
            RuntimeError: If the queue is empty.
        """
        if not self._queue:
            raise RuntimeError("No requests in the scheduler queue")

        entries: List[_QueueEntry] = []
        while self._queue and len(entries) < self._config.max_batch_size:
            entries.append(self._queue.popleft())

        requests = [e.request for e in entries]
        batch_cfg = BatchConfig(max_batch_size=len(requests))
        manager = BatchManager(batch_cfg)
        for req in requests:
            manager.add_request(req)
        return manager

    def clear(self) -> None:
        """Remove all pending requests from the queue."""
        self._queue.clear()
