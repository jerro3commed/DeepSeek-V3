"""Data structures for generation output."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from inference.metrics import GenerationMetrics


@dataclass
class GenerationOutput:
    """Holds the result of a single generation request."""

    text: str
    token_ids: List[int] = field(default_factory=list)
    finish_reason: str = "unknown"  # "eos", "stop_sequence", "length", "unknown"
    metrics: Optional[GenerationMetrics] = None

    _VALID_FINISH_REASONS = frozenset({"eos", "stop_sequence", "length", "unknown"})

    def __post_init__(self) -> None:
        if self.finish_reason not in self._VALID_FINISH_REASONS:
            raise ValueError(
                f"finish_reason must be one of {sorted(self._VALID_FINISH_REASONS)}, "
                f"got '{self.finish_reason}'"
            )

    @property
    def num_tokens(self) -> int:
        """Number of generated tokens."""
        return len(self.token_ids)

    def is_complete(self) -> bool:
        """Return True when generation ended naturally (eos or stop sequence)."""
        return self.finish_reason in {"eos", "stop_sequence"}

    def is_truncated(self) -> bool:
        """Return True when generation was cut off due to a token length limit."""
        return self.finish_reason == "length"

    def to_dict(self) -> dict:
        result = {
            "text": self.text,
            "token_ids": list(self.token_ids),
            "finish_reason": self.finish_reason,
            "num_tokens": self.num_tokens,
        }
        if self.metrics is not None:
            result["metrics"] = {
                "prompt_tokens": self.metrics.prompt_tokens,
                "generated_tokens": self.metrics.generated_tokens,
                "elapsed_seconds": self.metrics.elapsed_seconds,
            }
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "GenerationOutput":
        return cls(
            text=data["text"],
            token_ids=data.get("token_ids", []),
            finish_reason=data.get("finish_reason", "unknown"),
        )
