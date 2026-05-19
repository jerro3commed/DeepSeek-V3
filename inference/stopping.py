"""Stopping criteria for text generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence


@dataclass
class StoppingCriteria:
    """Defines conditions under which generation should stop."""

    max_new_tokens: int = 512
    stop_sequences: List[str] = field(default_factory=list)
    stop_token_ids: List[int] = field(default_factory=list)
    eos_token_id: Optional[int] = None

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError(
                f"max_new_tokens must be >= 1, got {self.max_new_tokens}"
            )
        if not isinstance(self.stop_sequences, list):
            raise TypeError("stop_sequences must be a list of strings")
        for s in self.stop_sequences:
            if not isinstance(s, str) or len(s) == 0:
                raise ValueError("Each stop sequence must be a non-empty string")

    def should_stop_on_token(self, token_id: int, generated_count: int) -> bool:
        """Return True if generation should stop based on a newly produced token."""
        if generated_count >= self.max_new_tokens:
            return True
        if self.eos_token_id is not None and token_id == self.eos_token_id:
            return True
        if token_id in self.stop_token_ids:
            return True
        return False

    def should_stop_on_text(self, text: str, generated_count: int) -> bool:
        """Return True if the decoded text contains a stop sequence."""
        if generated_count >= self.max_new_tokens:
            return True
        for seq in self.stop_sequences:
            if seq in text:
                return True
        return False

    def to_dict(self) -> dict:
        return {
            "max_new_tokens": self.max_new_tokens,
            "stop_sequences": list(self.stop_sequences),
            "stop_token_ids": list(self.stop_token_ids),
            "eos_token_id": self.eos_token_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StoppingCriteria":
        return cls(
            max_new_tokens=data.get("max_new_tokens", 512),
            stop_sequences=data.get("stop_sequences", []),
            stop_token_ids=data.get("stop_token_ids", []),
            eos_token_id=data.get("eos_token_id"),
        )
