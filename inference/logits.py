"""Logits processing utilities for token generation.

Provides a composable pipeline for transforming raw model logits before
sampling, including temperature scaling, bias application, and masking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class LogitsProcessorConfig:
    """Configuration for the logits processing pipeline."""

    # Tokens whose logits should be set to -inf (hard block)
    banned_token_ids: List[int] = field(default_factory=list)

    # Per-token additive bias applied before sampling
    token_bias: Dict[int, float] = field(default_factory=dict)

    # If True, logits are normalised to log-probabilities after processing
    return_log_probs: bool = False

    def __post_init__(self) -> None:
        for tid in self.banned_token_ids:
            if tid < 0:
                raise ValueError(
                    f"banned_token_ids must be non-negative, got {tid}"
                )
        for tid, bias in self.token_bias.items():
            if tid < 0:
                raise ValueError(
                    f"token_bias keys must be non-negative, got {tid}"
                )
            if not math.isfinite(bias):
                raise ValueError(
                    f"token_bias values must be finite, got {bias} for token {tid}"
                )


class LogitsProcessor:
    """Applies a sequence of transformations to raw logits tensors.

    All operations are performed in-place where possible to avoid
    unnecessary allocations during hot generation loops.
    """

    def __init__(self, config: LogitsProcessorConfig) -> None:
        self.config = config

        # Pre-convert to tensors once so we avoid Python loops per step
        self._banned: Optional[torch.Tensor] = (
            torch.tensor(config.banned_token_ids, dtype=torch.long)
            if config.banned_token_ids
            else None
        )

        self._bias_indices: Optional[torch.Tensor] = None
        self._bias_values: Optional[torch.Tensor] = None
        if config.token_bias:
            indices = list(config.token_bias.keys())
            values = list(config.token_bias.values())
            self._bias_indices = torch.tensor(indices, dtype=torch.long)
            self._bias_values = torch.tensor(values, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Process *logits* and return the transformed tensor.

        Args:
            logits: Float tensor of shape ``(vocab_size,)`` or
                ``(batch, vocab_size)``.
            temperature: Positive scaling factor; values < 1 sharpen the
                distribution, values > 1 flatten it.

        Returns:
            Processed logits (or log-probabilities when
            ``config.return_log_probs`` is ``True``).
        """
        logits = logits.clone()

        self._apply_bans(logits)
        self._apply_bias(logits)
        self._apply_temperature(logits, temperature)

        if self.config.return_log_probs:
            return torch.log_softmax(logits, dim=-1)
        return logits

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_bans(self, logits: torch.Tensor) -> None:
        """Set logits for banned tokens to negative infinity."""
        if self._banned is None:
            return
        device_banned = self._banned.to(logits.device)
        logits[..., device_banned] = float("-inf")

    def _apply_bias(self, logits: torch.Tensor) -> None:
        """Add per-token biases to logits."""
        if self._bias_indices is None:
            return
        idx = self._bias_indices.to(logits.device)
        val = self._bias_values.to(logits.device, dtype=logits.dtype)
        logits[..., idx] += val

    @staticmethod
    def _apply_temperature(logits: torch.Tensor, temperature: float) -> None:
        """Divide logits by *temperature* (skipped when temperature == 1.0)."""
        if temperature != 1.0:
            if temperature <= 0.0:
                raise ValueError(
                    f"temperature must be positive, got {temperature}"
                )
            logits /= temperature
