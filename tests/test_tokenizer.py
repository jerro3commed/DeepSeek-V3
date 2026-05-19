"""Unit tests for inference.tokenizer."""

import pytest

from inference.tokenizer import Tokenizer, TokenizerConfig


class TestTokenizerConfig:
    def test_defaults(self):
        cfg = TokenizerConfig(tokenizer_path="/tmp/tok")
        assert cfg.max_length == 4096
        assert cfg.add_bos_token is True
        assert cfg.padding_side == "left"

    def test_invalid_max_length_raises(self):
        with pytest.raises(ValueError, match="max_length must be positive"):
            TokenizerConfig(tokenizer_path="/tmp/tok", max_length=0)

    def test_negative_max_length_raises(self):
        with pytest.raises(ValueError, match="max_length must be positive"):
            TokenizerConfig(tokenizer_path="/tmp/tok", max_length=-1)

    def test_invalid_padding_side_raises(self):
        with pytest.raises(ValueError, match="padding_side"):
            TokenizerConfig(tokenizer_path="/tmp/tok", padding_side="center")

    def test_valid_padding_sides(self):
        for side in ("left", "right"):
            cfg = TokenizerConfig(tokenizer_path="/tmp/tok", padding_side=side)
            assert cfg.padding_side == side


class TestTokenizer:
    def _make(self, path: str = "/tmp/tok") -> Tokenizer:
        return Tokenizer(TokenizerConfig(tokenizer_path=path))

    def test_not_loaded_initially(self):
        tok = self._make()
        assert not tok._loaded
        assert tok.vocab_size is None

    def test_encode_before_load_raises(self):
        tok = self._make()
        with pytest.raises(RuntimeError, match="not loaded"):
            tok.encode("hello")

    def test_decode_before_load_raises(self):
        tok = self._make()
        with pytest.raises(RuntimeError, match="not loaded"):
            tok.decode([1, 2, 3])

    def test_load_missing_path_raises(self):
        tok = self._make(path="/nonexistent/path/to/tokenizer")
        with pytest.raises(FileNotFoundError):
            tok.load()

    def test_load_empty_path_raises(self):
        tok = self._make(path="")
        with pytest.raises(ValueError, match="tokenizer_path must be set"):
            tok.load()

    def test_unload_clears_state(self):
        tok = self._make()
        # Manually set loaded state to simulate a loaded tokenizer.
        tok._loaded = True
        tok._tokenizer = object()
        tok.unload()
        assert not tok._loaded
        assert tok._tokenizer is None
        assert tok.vocab_size is None
