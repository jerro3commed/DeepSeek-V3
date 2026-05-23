"""Model loading and forward-pass utilities for DeepSeek-V3 inference."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Supported dtypes for model weights
_DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


@dataclass
class ModelConfig:
    """Configuration for loading a DeepSeek-V3 model checkpoint."""

    model_path: str
    """Path to the directory containing model weights and config."""

    dtype: str = "bfloat16"
    """Weight dtype: 'float16', 'bfloat16', or 'float32'."""

    device: str = "cuda"
    """Target device string, e.g. 'cuda', 'cuda:0', or 'cpu'."""

    max_batch_size: int = 1
    """Maximum batch size the model will be called with."""

    trust_remote_code: bool = False
    """Whether to allow execution of remote model code."""

    extra_kwargs: dict = field(default_factory=dict)
    """Additional keyword arguments forwarded to the underlying loader."""

    def __post_init__(self) -> None:
        if self.dtype not in _DTYPE_MAP:
            raise ValueError(
                f"Unsupported dtype '{self.dtype}'. "
                f"Choose from: {sorted(_DTYPE_MAP)}"
            )
        if self.max_batch_size < 1:
            raise ValueError(
                f"max_batch_size must be >= 1, got {self.max_batch_size}"
            )
        model_dir = Path(self.model_path)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"model_path does not exist: {self.model_path}"
            )

    @property
    def torch_dtype(self) -> torch.dtype:
        """Resolve the configured dtype string to a :class:`torch.dtype`."""
        return _DTYPE_MAP[self.dtype]


class ModelWrapper:
    """Thin wrapper around a loaded transformer model.

    Handles device placement, dtype casting, and exposes a uniform
    :meth:`forward` interface used by :class:`~inference.engine.InferenceEngine`.
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._model: Optional[torch.nn.Module] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load model weights from :attr:`config.model_path` onto the target device."""
        try:
            from transformers import AutoModelForCausalLM  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "'transformers' is required to load model weights. "
                "Install it with: pip install transformers"
            ) from exc

        logger.info(
            "Loading model from '%s' [dtype=%s, device=%s]",
            self.config.model_path,
            self.config.dtype,
            self.config.device,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=self.config.torch_dtype,
            trust_remote_code=self.config.trust_remote_code,
            **self.config.extra_kwargs,
        ).to(self.config.device)
        self._model.eval()
        logger.info("Model loaded successfully.")

    def unload(self) -> None:
        """Release model weights and free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Model unloaded and memory released.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        use_cache: bool = True,
    ) -> tuple[torch.Tensor, object]:
        """Run a single forward pass and return ``(logits, past_key_values)``.

        Parameters
        ----------
        input_ids:
            Token ID tensor of shape ``(batch, seq_len)``.
        attention_mask:
            Optional boolean mask of the same shape as *input_ids*.
        past_key_values:
            Cached key/value states from previous steps, or ``None``.
        use_cache:
            Whether to return updated KV cache states.

        Returns
        -------
        tuple[torch.Tensor, object]
            ``logits`` of shape ``(batch, seq_len, vocab_size)`` and the
            updated ``past_key_values`` (or ``None`` when *use_cache* is
            ``False``).
        """
        if self._model is None:
            raise RuntimeError(
                "Model is not loaded. Call ModelWrapper.load() first."
            )

        with torch.inference_mode():
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
        return outputs.logits, outputs.past_key_values

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """Return ``True`` if model weights are currently in memory."""
        return self._model is not None

    def __repr__(self) -> str:  # pragma: no cover
        status = "loaded" if self.is_loaded else "unloaded"
        return (
            f"ModelWrapper(path={self.config.model_path!r}, "
            f"dtype={self.config.dtype!r}, status={status})"
        )
