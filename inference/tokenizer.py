"""Tokenizer wrapper for DeepSeek-V3 inference."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class TokenizerConfig:
    """Configuration for the tokenizer."""

    tokenizer_path: str = ""
    max_length: int = 4096
    add_bos_token: bool = True
    add_eos_token: bool = False
    padding_side: str = "left"
    special_tokens: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_length <= 0:
            raise ValueError(f"max_length must be positive, got {self.max_length}")
        if self.padding_side not in ("left", "right"):
            raise ValueError(
                f"padding_side must be 'left' or 'right', got '{self.padding_side}'"
            )


class Tokenizer:
    """Lightweight tokenizer wrapper used during inference."""

    def __init__(self, config: TokenizerConfig) -> None:
        self.config = config
        self._tokenizer = None
        self._loaded = False

    def load(self) -> None:
        """Load the underlying tokenizer from *tokenizer_path*."""
        if not self.config.tokenizer_path:
            raise ValueError("tokenizer_path must be set before calling load()")
        if not os.path.exists(self.config.tokenizer_path):
            raise FileNotFoundError(
                f"Tokenizer path not found: {self.config.tokenizer_path}"
            )
        # Lazy import so the rest of the module is importable without transformers.
        from transformers import AutoTokenizer  # type: ignore

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_path,
            padding_side=self.config.padding_side,
        )
        if self.config.special_tokens:
            self._tokenizer.add_special_tokens(self.config.special_tokens)
        self._loaded = True

    def encode(
        self,
        text: Union[str, List[str]],
        *,
        truncation: bool = True,
    ) -> List[List[int]]:
        """Encode *text* and return a list of token-id lists."""
        if not self._loaded:
            raise RuntimeError("Tokenizer is not loaded. Call load() first.")
        inputs = self._tokenizer(
            text if isinstance(text, list) else [text],
            truncation=truncation,
            max_length=self.config.max_length,
            padding=False,
        )
        return inputs["input_ids"]

    def decode(self, token_ids: List[int], *, skip_special_tokens: bool = True) -> str:
        """Decode a list of token ids back to a string."""
        if not self._loaded:
            raise RuntimeError("Tokenizer is not loaded. Call load() first.")
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def unload(self) -> None:
        """Release tokenizer resources."""
        self._tokenizer = None
        self._loaded = False

    @property
    def vocab_size(self) -> Optional[int]:
        """Return the vocabulary size, or None if the tokenizer is not loaded."""
        if self._tokenizer is None:
            return None
        return self._tokenizer.vocab_size
