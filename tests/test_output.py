"""Tests for inference/output.py."""

import pytest
from inference.output import GenerationOutput
from inference.metrics import GenerationMetrics


class TestGenerationOutputDefaults:
    def test_defaults(self):
        out = GenerationOutput(text="hello")
        assert out.text == "hello"
        assert out.token_ids == []
        assert out.finish_reason == "unknown"
        assert out.metrics is None

    def test_invalid_finish_reason_raises(self):
        with pytest.raises(ValueError, match="finish_reason"):
            GenerationOutput(text="hi", finish_reason="bad_reason")

    def test_valid_finish_reasons(self):
        for reason in ("eos", "stop_sequence", "length", "unknown"):
            out = GenerationOutput(text="", finish_reason=reason)
            assert out.finish_reason == reason


class TestNumTokens:
    def test_empty_token_ids(self):
        out = GenerationOutput(text="")
        assert out.num_tokens == 0

    def test_non_empty_token_ids(self):
        out = GenerationOutput(text="hi", token_ids=[1, 2, 3])
        assert out.num_tokens == 3


class TestIsComplete:
    def test_eos_is_complete(self):
        assert GenerationOutput(text="", finish_reason="eos").is_complete() is True

    def test_stop_sequence_is_complete(self):
        assert GenerationOutput(text="", finish_reason="stop_sequence").is_complete() is True

    def test_length_is_not_complete(self):
        assert GenerationOutput(text="", finish_reason="length").is_complete() is False

    def test_unknown_is_not_complete(self):
        assert GenerationOutput(text="", finish_reason="unknown").is_complete() is False


class TestSerialisation:
    def test_to_dict_no_metrics(self):
        out = GenerationOutput(text="world", token_ids=[5, 6], finish_reason="eos")
        d = out.to_dict()
        assert d["text"] == "world"
        assert d["token_ids"] == [5, 6]
        assert d["finish_reason"] == "eos"
        assert d["num_tokens"] == 2
        assert "metrics" not in d

    def test_to_dict_with_metrics(self):
        import time
        m = GenerationMetrics(prompt_tokens=10, generated_tokens=0)
        m.start_time = time.monotonic() - 1.0
        m.record_token()
        m.finish()
        out = GenerationOutput(text="hi", finish_reason="eos", metrics=m)
        d = out.to_dict()
        assert "metrics" in d
        assert d["metrics"]["prompt_tokens"] == 10

    def test_round_trip(self):
        out = GenerationOutput(text="test", token_ids=[1, 2], finish_reason="length")
        restored = GenerationOutput.from_dict(out.to_dict())
        assert restored.text == "test"
        assert restored.token_ids == [1, 2]
        assert restored.finish_reason == "length"

    def test_from_dict_defaults(self):
        out = GenerationOutput.from_dict({"text": "hi"})
        assert out.finish_reason == "unknown"
        assert out.token_ids == []
