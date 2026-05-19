"""Unit tests for the inference config and engine."""
import pytest
from unittest.mock import MagicMock, patch

from inference.config import InferenceConfig
from inference.engine import InferenceEngine


# ---------------------------------------------------------------------------
# InferenceConfig tests
# ---------------------------------------------------------------------------

class TestInferenceConfig:
    def test_defaults(self):
        cfg = InferenceConfig()
        assert cfg.dtype == "bfloat16"
        assert cfg.max_new_tokens == 2048
        assert cfg.tokenizer_path == cfg.model_path

    def test_tokenizer_path_fallback(self):
        cfg = InferenceConfig(model_path="/my/model", tokenizer_path=None)
        assert cfg.tokenizer_path == "/my/model"

    def test_invalid_dtype_raises(self):
        with pytest.raises(AssertionError):
            InferenceConfig(dtype="int8")

    def test_invalid_temperature_raises(self):
        with pytest.raises(AssertionError):
            InferenceConfig(temperature=0.0)

    def test_to_dict_roundtrip(self):
        cfg = InferenceConfig(temperature=1.0, top_p=0.95)
        restored = InferenceConfig.from_dict(cfg.to_dict())
        assert restored.temperature == cfg.temperature
        assert restored.top_p == cfg.top_p


# ---------------------------------------------------------------------------
# InferenceEngine tests  (model/tokenizer are mocked)
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_engine():
    """Return an engine with mocked model and tokenizer."""
    engine = InferenceEngine(InferenceConfig(device="cpu", use_flash_attention=False))

    fake_tokenizer = MagicMock()
    fake_tokenizer.return_value = {"input_ids": MagicMock(shape=[1, 5])}
    fake_tokenizer.batch_decode.return_value = ["Hello world"]

    fake_model = MagicMock()
    import torch
    fake_model.generate.return_value = torch.zeros((1, 10), dtype=torch.long)

    engine.tokenizer = fake_tokenizer
    engine.model = fake_model
    engine._loaded = True
    return engine


class TestInferenceEngine:
    def test_generate_returns_list(self, mock_engine):
        results = mock_engine.generate("Hello")
        assert isinstance(results, list)
        assert len(results) == 1

    def test_generate_batch(self, mock_engine):
        mock_engine.tokenizer.batch_decode.return_value = ["A", "B"]
        results = mock_engine.generate(["prompt1", "prompt2"])
        assert len(results) == 2

    def test_generate_without_load_raises(self):
        engine = InferenceEngine()
        with pytest.raises(RuntimeError, match="load()"):
            engine.generate("test")

    @patch("inference.engine.AutoModelForCausalLM.from_pretrained")
    @patch("inference.engine.AutoTokenizer.from_pretrained")
    def test_load_sets_loaded_flag(self, mock_tok, mock_model):
        mock_model.return_value = MagicMock()
        mock_tok.return_value = MagicMock()
        engine = InferenceEngine(InferenceConfig(device="cpu", use_flash_attention=False))
        engine.load()
        assert engine._loaded is True
