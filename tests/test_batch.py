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
    # Using larger defaults here to better reflect realistic DeepSeek-V3 workloads
    defaults = dict(max_batch_size=8, max_seq_len=2048)
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
        # make_config uses max_batch_size=8, max_seq_len=2048 as personal defaults
        cfg = make_config()
        assert cfg.max_batch_size == 8
        assert cfg.max_seq_len == 2048

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
