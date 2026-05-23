"""Sampling strategies for DeepSeek-V3 inference."""

from dataclasses import dataclass, field
from typing import Optional
import torch
import torch.nn.functional as F


@dataclass
class SamplingParams:
    """Parameters controlling token sampling during generation."""

    temperature: float = 1.0
    top_p: float = 0.95  # changed from 1.0 -- nucleus sampling on by default feels more useful
    top_k: int = 0
    repetition_penalty: float = 1.0
    min_new_tokens: int = 0
    max_new_tokens: int = 512
    stop_token_ids: list = field(default_factory=list)

    def __post_init__(self):
        if not 0.0 < self.temperature <= 2.0:
            raise ValueError(f"temperature must be in (0, 2], got {self.temperature}")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")
        if self.repetition_penalty <= 0.0:
            raise ValueError(f"repetition_penalty must be > 0, got {self.repetition_penalty}")
        if self.max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be > 0, got {self.max_new_tokens}")


def apply_repetition_penalty(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """Penalise tokens that have already appeared in the context."""
    if penalty == 1.0:
        return logits
    score = torch.gather(logits, 1, input_ids)
    score = torch.where(score < 0, score * penalty, score / penalty)
    return logits.scatter_(1, input_ids, score)


def top_k_top_p_filter(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Apply top-k and/or nucleus (top-p) filtering to logits."""
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, k).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # shift cumulative probs right so the token that pushes us over top_p is kept
        sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits[sorted_indices_to_remove] = float("-inf")
        logits = torch.zeros_like(logits).scatter_(-1, sorted_indices, sorted_logits)

    return logits


def sample_next_token(
    logits: torch.Tensor,
    params: SamplingParams,
    input_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sample the next token id given a logits tensor (batch, vocab).

    Args:
        logits: Raw logits from the model, shape (batch, vocab).
        params: Sampling parameters (temperature, top_p, top_k, etc.).
        input_ids: Previously generated token ids used for repetition penalty.

    Returns:
        Sampled token ids, shape (batch, 1).
    """
    if input_ids is not None:
        logits = apply_repetition_penalty(logits, input_ids, params.repetition_penalty)

    logits = logits / params.temperature
    logits = top_k_top_p_filter(logits, top_k=params.top_k, top_p=params.top_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
