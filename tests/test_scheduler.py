"""Tests for inference/scheduler.py — SchedulerConfig and Scheduler."""

import time
import pytest

from inference.scheduler import Scheduler, SchedulerConfig
from inference.batch import GenerationRequest, BatchConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_scheduler_config(**kwargs) -> SchedulerConfig:
    defaults = dict(
        max_batch_size=4,
        max_queue_size=16,
        timeout_seconds=30.0,
    )
    defaults.update(kwargs)
    return SchedulerConfig(**defaults)


def make_request(prompt_ids=None, request_id="req-1") -> GenerationRequest:
    if prompt_ids is None:
        prompt_ids = [1, 2, 3]
    return GenerationRequest(request_id=request_id, prompt_ids=prompt_ids)


# ---------------------------------------------------------------------------
# SchedulerConfig validation
# ---------------------------------------------------------------------------

class TestSchedulerConfig:
    def test_defaults(self):
        cfg = SchedulerConfig()
        assert cfg.max_batch_size >= 1
        assert cfg.max_queue_size >= 1
        assert cfg.timeout_seconds > 0

    def test_invalid_max_batch_size_raises(self):
        with pytest.raises((ValueError, TypeError)):
            make_scheduler_config(max_batch_size=0)

    def test_negative_max_batch_size_raises(self):
        with pytest.raises((ValueError, TypeError)):
            make_scheduler_config(max_batch_size=-1)

    def test_invalid_max_queue_size_raises(self):
        with pytest.raises((ValueError, TypeError)):
            make_scheduler_config(max_queue_size=0)

    def test_invalid_timeout_raises(self):
        with pytest.raises((ValueError, TypeError)):
            make_scheduler_config(timeout_seconds=-5.0)

    def test_zero_timeout_raises(self):
        with pytest.raises((ValueError, TypeError)):
            make_scheduler_config(timeout_seconds=0.0)


# ---------------------------------------------------------------------------
# Scheduler behaviour
# ---------------------------------------------------------------------------

class TestSchedulerEnqueue:
    def test_enqueue_single_request(self):
        scheduler = Scheduler(make_scheduler_config())
        req = make_request()
        scheduler.enqueue(req)
        assert scheduler.queue_size() == 1

    def test_enqueue_multiple_requests(self):
        scheduler = Scheduler(make_scheduler_config())
        for i in range(3):
            scheduler.enqueue(make_request(request_id=f"req-{i}"))
        assert scheduler.queue_size() == 3

    def test_queue_full_raises(self):
        scheduler = Scheduler(make_scheduler_config(max_queue_size=2))
        scheduler.enqueue(make_request(request_id="r1"))
        scheduler.enqueue(make_request(request_id="r2"))
        with pytest.raises((RuntimeError, OverflowError, ValueError)):
            scheduler.enqueue(make_request(request_id="r3"))


class TestSchedulerNextBatch:
    def test_next_batch_respects_max_batch_size(self):
        cfg = make_scheduler_config(max_batch_size=2)
        scheduler = Scheduler(cfg)
        for i in range(5):
            scheduler.enqueue(make_request(request_id=f"req-{i}"))
        batch = scheduler.next_batch()
        assert len(batch) <= 2

    def test_next_batch_empty_queue_returns_empty(self):
        scheduler = Scheduler(make_scheduler_config())
        batch = scheduler.next_batch()
        assert batch == [] or batch is not None  # empty iterable
        assert len(batch) == 0

    def test_next_batch_drains_queue(self):
        cfg = make_scheduler_config(max_batch_size=8)
        scheduler = Scheduler(cfg)
        for i in range(4):
            scheduler.enqueue(make_request(request_id=f"req-{i}"))
        batch = scheduler.next_batch()
        assert len(batch) == 4
        assert scheduler.queue_size() == 0

    def test_next_batch_fifo_order(self):
        scheduler = Scheduler(make_scheduler_config(max_batch_size=10))
        ids = [f"req-{i}" for i in range(5)]
        for rid in ids:
            scheduler.enqueue(make_request(request_id=rid))
        batch = scheduler.next_batch()
        batch_ids = [r.request_id for r in batch]
        assert batch_ids == ids
