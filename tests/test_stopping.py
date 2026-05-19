"""Tests for inference/stopping.py."""

import pytest
from inference.stopping import StoppingCriteria


class TestStoppingCriteriaDefaults:
    def test_defaults(self):
        sc = StoppingCriteria()
        assert sc.max_new_tokens == 512
        assert sc.stop_sequences == []
        assert sc.stop_token_ids == []
        assert sc.eos_token_id is None

    def test_invalid_max_new_tokens_raises(self):
        with pytest.raises(ValueError, match="max_new_tokens"):
            StoppingCriteria(max_new_tokens=0)

    def test_negative_max_new_tokens_raises(self):
        with pytest.raises(ValueError):
            StoppingCriteria(max_new_tokens=-5)

    def test_empty_stop_sequence_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            StoppingCriteria(stop_sequences=[""])

    def test_invalid_stop_sequences_type_raises(self):
        with pytest.raises(TypeError):
            StoppingCriteria(stop_sequences="stop")


class TestShouldStopOnToken:
    def test_stops_at_max_new_tokens(self):
        sc = StoppingCriteria(max_new_tokens=3)
        assert sc.should_stop_on_token(token_id=99, generated_count=3) is True

    def test_does_not_stop_before_max(self):
        sc = StoppingCriteria(max_new_tokens=3)
        assert sc.should_stop_on_token(token_id=99, generated_count=2) is False

    def test_stops_on_eos_token(self):
        sc = StoppingCriteria(eos_token_id=2)
        assert sc.should_stop_on_token(token_id=2, generated_count=1) is True

    def test_does_not_stop_on_other_token(self):
        sc = StoppingCriteria(eos_token_id=2)
        assert sc.should_stop_on_token(token_id=5, generated_count=1) is False

    def test_stops_on_custom_stop_token_id(self):
        sc = StoppingCriteria(stop_token_ids=[100, 200])
        assert sc.should_stop_on_token(token_id=100, generated_count=1) is True
        assert sc.should_stop_on_token(token_id=200, generated_count=1) is True
        assert sc.should_stop_on_token(token_id=50, generated_count=1) is False


class TestShouldStopOnText:
    def test_stops_on_stop_sequence(self):
        sc = StoppingCriteria(stop_sequences=["<|end|>", "STOP"])
        assert sc.should_stop_on_text("Hello<|end|>", generated_count=5) is True

    def test_does_not_stop_without_sequence(self):
        sc = StoppingCriteria(stop_sequences=["<|end|>"])
        assert sc.should_stop_on_text("Hello world", generated_count=5) is False

    def test_stops_at_max_new_tokens_via_text(self):
        sc = StoppingCriteria(max_new_tokens=2)
        assert sc.should_stop_on_text("Hi", generated_count=2) is True


class TestSerialisation:
    def test_round_trip(self):
        sc = StoppingCriteria(
            max_new_tokens=128,
            stop_sequences=["</s>"],
            stop_token_ids=[1, 2],
            eos_token_id=3,
        )
        restored = StoppingCriteria.from_dict(sc.to_dict())
        assert restored.max_new_tokens == 128
        assert restored.stop_sequences == ["</s>"]
        assert restored.stop_token_ids == [1, 2]
        assert restored.eos_token_id == 3

    def test_from_dict_defaults(self):
        sc = StoppingCriteria.from_dict({})
        assert sc.max_new_tokens == 512
        assert sc.eos_token_id is None
