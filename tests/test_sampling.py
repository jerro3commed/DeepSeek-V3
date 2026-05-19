"""Unit tests for inference/sampling.py."""

import pytest
import torch

from inference.sampling import (
    SamplingParams,
    apply_repetition_penalty,
    top_k_top_p_filter,
    sample_next_token,
)


class TestSamplingParams:
    def test_defaults(self):
        p = SamplingParams()
        assert p.temperature == 1.0
        assert p.top_p == 1.0
        assert p.top_k == 0
        assert p.repetition_penalty == 1.0
        assert p.max_new_tokens == 512

    def test_invalid_temperature_raises(self):
        with pytest.raises(ValueError, match="temperature"):
            SamplingParams(temperature=0.0)

    def test_temperature_too_high_raises(self):
        with pytest.raises(ValueError, match="temperature"):
            SamplingParams(temperature=2.5)

    def test_invalid_top_p_raises(self):
        with pytest.raises(ValueError, match="top_p"):
            SamplingParams(top_p=0.0)

    def test_invalid_top_k_raises(self):
        with pytest.raises(ValueError, match="top_k"):
            SamplingParams(top_k=-1)

    def test_invalid_repetition_penalty_raises(self):
        with pytest.raises(ValueError, match="repetition_penalty"):
            SamplingParams(repetition_penalty=0.0)

    def test_invalid_max_new_tokens_raises(self):
        with pytest.raises(ValueError, match="max_new_tokens"):
            SamplingParams(max_new_tokens=0)


class TestRepetitionPenalty:
    def test_no_penalty_identity(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        input_ids = torch.tensor([[0, 1]])
        result = apply_repetition_penalty(logits.clone(), input_ids, penalty=1.0)
        assert torch.allclose(result, logits)

    def test_penalty_reduces_positive_logits(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        input_ids = torch.tensor([[2]])  # token 2 has logit 3.0
        result = apply_repetition_penalty(logits.clone(), input_ids, penalty=2.0)
        assert result[0, 2] < 3.0


class TestTopKTopP:
    def test_top_k_limits_candidates(self):
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        filtered = top_k_top_p_filter(logits.clone(), top_k=2)
        finite_count = torch.isfinite(filtered).sum().item()
        assert finite_count == 2

    def test_top_p_1_no_change(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = top_k_top_p_filter(logits.clone(), top_p=1.0)
        assert torch.allclose(result, logits)


class TestSampleNextToken:
    def test_returns_valid_token_id(self):
        vocab_size = 100
        logits = torch.randn(1, vocab_size)
        params = SamplingParams(temperature=1.0)
        token = sample_next_token(logits, params)
        assert token.shape == (1, 1)
        assert 0 <= token.item() < vocab_size

    def test_greedy_like_low_temperature(self):
        logits = torch.zeros(1, 50)
        logits[0, 42] = 100.0  # overwhelmingly likely token
        params = SamplingParams(temperature=0.01)
        token = sample_next_token(logits, params)
        assert token.item() == 42
