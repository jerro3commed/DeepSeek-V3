# Configuration for DeepSeek-V3 inference
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InferenceConfig:
    """Configuration class for DeepSeek-V3 inference settings."""

    # Model settings
    model_path: str = "deepseek-ai/DeepSeek-V3"
    dtype: str = "bfloat16"  # Options: float16, bfloat16, float32
    device: str = "cuda"     # Options: cuda, cpu

    # Generation settings
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0  # Changed from 1.1 -- repetition penalty was causing
                                     # slightly degraded outputs on my test prompts
    do_sample: bool = True

    # Memory / performance settings
    use_flash_attention: bool = True
    tensor_parallel_size: int = 1
    max_batch_size: int = 4
    max_seq_len: int = 4096

    # Tokenizer settings
    tokenizer_path: Optional[str] = None  # Defaults to model_path if None
    padding_side: str = "left"

    # Logging
    verbose: bool = False

    def __post_init__(self):
        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_path
        assert self.dtype in ("float16", "bfloat16", "float32"), (
            f"Unsupported dtype: {self.dtype}"
        )
        assert 0.0 < self.temperature <= 2.0, "temperature must be in (0, 2]"
        assert 0.0 < self.top_p <= 1.0, "top_p must be in (0, 1]"

    def to_dict(self) -> dict:
        """Serialize config to a plain dict."""
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, data: dict) -> "InferenceConfig":
        """Deserialize config from a plain dict."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
