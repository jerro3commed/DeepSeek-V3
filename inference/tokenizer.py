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
        add_special_tokens: bool = True,
    ) -> List[List[int]]:
        """Encode *text* and return a list of token-id lists.

        Args:
            text: A single string or a list of strings to encode.
            truncation: Whether to truncate sequences to *max_length*.
            add_special_tokens: Whether to add BOS/EOS and other special tokens.
                Defaults to True; set to False when you want raw token ids.
        """
        if not self._loaded:
            raise RuntimeError("Tokenizer is not loaded. Call load() first.")
        inputs = self._tokenizer(
            text if isinstance(text, list) else [text],
            truncation=truncation,
            max_length=self.config.max_length,
            padding=False,
            add_special_tokens=add_special_tokens,
        )
        return inputs["input_ids"]

    def decode(self, token_ids: List[int], *, skip_special_tokens: bool = True) -> str:
        """Decode a list of token ids back to a string.

        Args:
            token_ids: List of integer token ids to decode.
            skip_special_tokens: Whether to remove special tokens (e.g. BOS/EOS)
                from the decoded output. Defaults to True.
        """
        if not self._loaded:
            raise RuntimeError("Tokenizer is not loaded. Call load() first.")
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
