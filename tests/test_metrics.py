"""Tests for inference.metrics.GenerationMetrics."""

import time
import pytest
from inference.metrics import GenerationMetrics


class TestGenerationMetrics:
    def test_defaults(self):
        m = GenerationMetrics(prompt_tokens=10)
        assert m.prompt_tokens == 10
        assert m.generated_tokens == 0
        assert m.end_time is None

    def test_negative_prompt_tokens_raises(self):
        with pytest.raises(ValueError, match="prompt_tokens"):
            GenerationMetrics(prompt_tokens=-1)

    def test_negative_generated_tokens_raises(self):
        with pytest.raises(ValueError, match="generated_tokens"):
            GenerationMetrics(prompt_tokens=0, generated_tokens=-1)

    def test_record_token_increments_count(self):
        m = GenerationMetrics(prompt_tokens=5)
        m.record_token()
        m.record_token()
        assert m.generated_tokens == 2

    def test_total_tokens(self):
        m = GenerationMetrics(prompt_tokens=8)
        m.record_token()
        m.record_token()
        assert m.total_tokens == 10

    def test_finish_sets_end_time(self):
        m = GenerationMetrics(prompt_tokens=4)
        assert m.end_time is None
        m.finish()
        assert m.end_time is not None

    def test_finish_is_idempotent(self):
        # Calling finish() twice should not overwrite the first end_time.
        m = GenerationMetrics(prompt_tokens=4)
        m.finish()
        first_end_time = m.end_time
        time.sleep(0.01)
        m.finish()
        assert m.end_time == first_end_time

    def test_elapsed_seconds_positive(self):
        m = GenerationMetrics(prompt_tokens=4)
        time.sleep(0.01)
        m.finish()
        assert m.elapsed_seconds > 0

    def test_tokens_per_second_nonzero(self):
        m = GenerationMetrics(prompt_tokens=4)
        for _ in range(5):
            m.record_token()
        time.sleep(0.05)
        m.finish()
        assert m.tokens_per_second > 0

    def test_tokens_per_second_zero_when_no_tokens(self):
        # tokens_per_second should be 0 (not an error) when no tokens were generated
        m = GenerationMetrics(prompt_tokens=4)
        time.sleep(0.01)
        m.finish()
        assert m.tokens_per_second == 0

    def test_time_to_first_token_none_before_any_token(self):
        m = GenerationMetrics(prompt_tokens=4)
        assert m.time_to_first_token is None

    def test_time_to_first_token_after_record(self):
        m = GenerationMetrics(prompt_tokens=4)
        time.sleep(0.01)
        m.record_token()
        ttft = m.time_to_first_token
        assert ttft is not None
        assert ttft >= 0.005

    def test_to_dict_keys(self):
        m = GenerationMetrics(prompt_tokens=3)
        m.record_token()
        m.finish()
        d = m.to_dict()
        expected_keys = {
            "prompt_tokens",
            "generated_tokens",
            "total_tokens",
            "elapsed_seconds",
            "tokens_per_second",
            "time_to_first_token",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values(self):
        m = GenerationMetrics(prompt_tokens=6)
        m.record_token()
        m.record_token()
        m.finish()
        d = m.to_dict()
        assert d["prompt_tokens"] == 6
        assert d["generated_tokens"] == 2
        assert d["total_tokens"] == 8
