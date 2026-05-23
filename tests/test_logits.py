"""Tests for inference/logits.py — LogitsProcessorConfig and LogitsProcessor."""

import math
import pytest
import torch

from inference.logits import LogitsProcessor, LogitsProcessorConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(**kwargs) -> LogitsProcessorConfig:
    defaults = dict(
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.0,
        min_tokens_to_keep=1,
    )
    defaults.update(kwargs)
    return LogitsProcessorConfig(**defaults)


def uniform_logits(vocab_size: int = 16) -> torch.Tensor:
    """Return a 1-D logits tensor with equal values."""
    return torch.zeros(vocab_size)


# ---------------------------------------------------------------------------
# LogitsProcessorConfig validation
# ---------------------------------------------------------------------------

class TestLogitsProcessorConfig:
    def test_defaults(self):
        cfg = make_config()
        assert cfg.temperature == 1.0
        assert cfg.top_k == 0
        assert cfg.top_p == 1.0
        assert cfg.repetition_penalty == 1.0
        assert cfg.min_tokens_to_keep == 1

    def test_invalid_temperature_raises(self):
        with pytest.raises((ValueError, TypeError)):
            make_config(temperature="hot")

    def test_zero_temperature_raises(self):
        with pytest.raises(ValueError):
            make_config(temperature=0.0)

    def test_negative_temperature_raises(self):
        with pytest.raises(ValueError):
            make_config(temperature=-0.5)

    def test_negative_top_k_raises(self):
        with pytest.raises(ValueError):
            make_config(top_k=-1)

    def test_top_p_out_of_range_raises(self):
        with pytest.raises(ValueError):
            make_config(top_p=1.5)
        with pytest.raises(ValueError):
            make_config(top_p=0.0)

    def test_repetition_penalty_below_one_raises(self):
        with pytest.raises(ValueError):
            make_config(repetition_penalty=0.5)

    def test_min_tokens_to_keep_zero_raises(self):
        with pytest.raises(ValueError):
            make_config(min_tokens_to_keep=0)


# ---------------------------------------------------------------------------
# LogitsProcessor behaviour
# ---------------------------------------------------------------------------

class TestLogitsProcessor:
    def test_temperature_scaling(self):
        """Higher temperature flattens the distribution."""
        logits = torch.tensor([2.0, 1.0, 0.5, 0.0])
        proc_low = LogitsProcessor(make_config(temperature=0.1))
        proc_high = LogitsProcessor(make_config(temperature=2.0))

        out_low = proc_low(logits.clone(), input_ids=[])
        out_high = proc_high(logits.clone(), input_ids=[])

        # Low temperature should produce a sharper (higher-variance) distribution
        assert out_low.std().item() > out_high.std().item()

    def test_top_k_filters_tokens(self):
        """Only top-k tokens should remain finite after processing."""
        vocab_size = 16
        logits = torch.arange(vocab_size, dtype=torch.float)
        proc = LogitsProcessor(make_config(top_k=4))
        out = proc(logits.clone(), input_ids=[])

        finite_count = torch.isfinite(out).sum().item()
        assert finite_count == 4

    def test_top_p_nucleus_filtering(self):
        """Top-p should remove low-probability tokens."""
        # Skewed distribution: token 0 dominates
        logits = torch.tensor([10.0] + [-10.0] * 15)
        proc = LogitsProcessor(make_config(top_p=0.9))
        out = proc(logits.clone(), input_ids=[])

        # Token 0 must survive; most others should be filtered
        assert torch.isfinite(out[0])
        filtered = (~torch.isfinite(out)).sum().item()
        assert filtered > 0

    def test_repetition_penalty_reduces_repeated_token(self):
        """Tokens already in input_ids should have reduced logits."""
        logits = torch.ones(16)
        proc = LogitsProcessor(make_config(repetition_penalty=1.5))
        out = proc(logits.clone(), input_ids=[3, 7])

        # Repeated tokens should have lower logit than non-repeated
        assert out[3].item() < out[0].item()
        assert out[7].item() < out[0].item()

    def test_no_penalty_with_empty_input_ids(self):
        """No repetition penalty when input_ids is empty."""
        logits = torch.ones(8)
        proc = LogitsProcessor(make_config(repetition_penalty=2.0))
        out = proc(logits.clone(), input_ids=[])
        # All logits should be equal (only temperature applied, temp=1.0)
        assert torch.allclose(out, out[0].expand_as(out))

    def test_greedy_returns_highest_logit(self):
        """With top_k=1 the processor should keep only the argmax token."""
        logits = torch.tensor([0.1, 0.5, 5.0, 0.2])
        proc = LogitsProcessor(make_config(top_k=1))
        out = proc(logits.clone(), input_ids=[])
        best = int(logits.argmax())
        assert torch.isfinite(out[best])
        for i in range(len(logits)):
            if i != best:
                assert not torch.isfinite(out[i])
