"""Generation metrics tracking for inference runs."""

from dataclasses import dataclass, field
from typing import List, Optional
import time


@dataclass
class GenerationMetrics:
    """Tracks timing and token statistics for a single generation."""

    prompt_tokens: int = 0
    generated_tokens: int = 0
    start_time: float = field(default_factory=time.monotonic)
    end_time: Optional[float] = None
    _token_timestamps: List[float] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if self.prompt_tokens < 0:
            raise ValueError("prompt_tokens must be >= 0")
        if self.generated_tokens < 0:
            raise ValueError("generated_tokens must be >= 0")

    def record_token(self) -> None:
        """Record the timestamp of a newly generated token."""
        self._token_timestamps.append(time.monotonic())
        self.generated_tokens += 1

    def finish(self) -> None:
        """Mark generation as complete."""
        self.end_time = time.monotonic()

    @property
    def elapsed_seconds(self) -> float:
        """Total wall-clock time from start to finish (or now)."""
        end = self.end_time if self.end_time is not None else time.monotonic()
        return max(end - self.start_time, 1e-9)

    @property
    def tokens_per_second(self) -> float:
        """Throughput in generated tokens per second."""
        return self.generated_tokens / self.elapsed_seconds

    @property
    def total_tokens(self) -> int:
        """Sum of prompt and generated tokens."""
        return self.prompt_tokens + self.generated_tokens

    @property
    def time_to_first_token(self) -> Optional[float]:
        """Latency from start until the first token was produced."""
        if not self._token_timestamps:
            return None
        return self._token_timestamps[0] - self.start_time

    @property
    def inter_token_latency(self) -> Optional[float]:
        """Average time between consecutive tokens (excludes time-to-first-token)."""
        if len(self._token_timestamps) < 2:
            return None
        gaps = [
            self._token_timestamps[i] - self._token_timestamps[i - 1]
            for i in range(1, len(self._token_timestamps))
        ]
        return sum(gaps) / len(gaps)

    def to_dict(self) -> dict:
        """Serialize metrics to a plain dictionary."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "total_tokens": self.total_tokens,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "time_to_first_token": (
                round(self.time_to_first_token, 4)
                if self.time_to_first_token is not None
                else None
            ),
            "inter_token_latency": (
                round(self.inter_token_latency, 4)
                if self.inter_token_latency is not None
                else None
            ),
        }
