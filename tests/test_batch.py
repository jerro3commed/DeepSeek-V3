"""Tests for inference/batch.py — GenerationRequest, BatchConfig, and Batch."""

import pytest
import torch

from inference.batch import (
    BatchConfig,
    GenerationRequest,
    Batch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_request(prompt_ids=None, request_id="req-1", max_new_tokens=64):
    if prompt_ids is None:
        prompt_ids = [1, 2, 3, 4]
    return GenerationRequest(
        request_id=request_id,
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
    )


def make_config(**kwargs):
    defaults = dict(max_batch_size=4, max_seq_len=512)
    defaults.update(kwargs)
    return BatchConfig(**defaults)


# ---------------------------------------------------------------------------
# GenerationRequest
# ---------------------------------------------------------------------------

class TestGenerationRequest:
    def test_prompt_length(self):
        req = make_request(prompt_ids=[10, 20, 30])
        assert req.prompt_length == 3

    def test_empty_prompt_ids_raises(self):
        with pytest.raises(ValueError, match="prompt_ids"):
            make_request(prompt_ids=[])

    def test_invalid_max_new_tokens_raises(self):
        with pytest.raises(ValueError, match="max_new_tokens"):
            make_request(max_new_tokens=0)

    def test_negative_max_new_tokens_raises(self):
        with pytest.raises(ValueError, match="max_new_tokens"):
            make_request(max_new_tokens=-5)

    def test_empty_request_id_raises(self):
        with pytest.raises(ValueError, match="request_id"):
            make_request(request_id="")

    def test_generated_ids_starts_empty(self):
        req = make_request()
        assert req.generated_ids == []

    def test_is_finished_false_initially(self):
        req = make_request(max_new_tokens=2)
        assert not req.is_finished

    def test_is_finished_true_when_limit_reached(self):
        req = make_request(max_new_tokens=2)
        req.generated_ids = [5, 6]
        assert req.is_finished


# ---------------------------------------------------------------------------
# BatchConfig
# ---------------------------------------------------------------------------

class TestBatchConfig:
    def test_defaults(self):
        cfg = make_config()
        assert cfg.max_batch_size == 4
        assert cfg.max_seq_len == 512

    def test_invalid_max_batch_size_raises(self):
        with pytest.raises(ValueError, match="max_batch_size"):
            make_config(max_batch_size=0)

    def test_invalid_max_seq_len_raises(self):
        with pytest.raises(ValueError, match="max_seq_len"):
            make_config(max_seq_len=-1)

    def test_to_dict_round_trip(self):
        cfg = make_config(max_batch_size=8, max_seq_len=1024)
        restored = BatchConfig(**cfg.to_dict())
        assert restored.max_batch_size == cfg.max_batch_size
        assert restored.max_seq_len == cfg.max_seq_len


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

class TestBatch:
    def test_add_and_size(self):
        cfg = make_config(max_batch_size=4)
        batch = Batch(config=cfg)
        batch.add(make_request(request_id="r1"))
        batch.add(make_request(request_id="r2"))
        assert batch.size == 2

    def test_add_beyond_capacity_raises(self):
        cfg = make_config(max_batch_size=2)
        batch = Batch(config=cfg)
        batch.add(make_request(request_id="r1"))
        batch.add(make_request(request_id="r2"))
        with pytest.raises(RuntimeError, match="capacity"):
            batch.add(make_request(request_id="r3"))

    def test_is_full(self):
        cfg = make_config(max_batch_size=2)
        batch = Batch(config=cfg)
        assert not batch.is_full
        batch.add(make_request(request_id="r1"))
        batch.add(make_request(request_id="r2"))
        assert batch.is_full

    def test_remove_finished(self):
        cfg = make_config(max_batch_size=4)
        batch = Batch(config=cfg)
        req = make_request(request_id="r1", max_new_tokens=1)
        req.generated_ids = [99]  # mark finished
        batch.add(req)
        batch.add(make_request(request_id="r2"))
        finished = batch.remove_finished()
        assert len(finished) == 1
        assert finished[0].request_id == "r1"
        assert batch.size == 1

    def test_input_ids_tensor_shape(self):
        cfg = make_config(max_batch_size=4, max_seq_len=16)
        batch = Batch(config=cfg)
        batch.add(make_request(prompt_ids=[1, 2, 3], request_id="r1"))
        batch.add(make_request(prompt_ids=[4, 5], request_id="r2"))
        tensor = batch.get_input_ids(pad_id=0)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (2, 3)  # batch=2, max prompt len in batch=3
