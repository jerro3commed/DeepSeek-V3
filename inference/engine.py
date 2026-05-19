# Inference engine for DeepSeek-V3
import logging
from typing import List, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from inference.config import InferenceConfig

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Wraps the DeepSeek-V3 model for convenient text generation."""

    def __init__(self, config: Optional[InferenceConfig] = None):
        self.config = config or InferenceConfig()
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self) -> None:
        """Load model and tokenizer from disk / hub."""
        cfg = self.config
        if cfg.verbose:
            logger.setLevel(logging.DEBUG)

        logger.info("Loading tokenizer from %s", cfg.tokenizer_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.tokenizer_path,
            padding_side=cfg.padding_side,
            trust_remote_code=True,
        )

        torch_dtype = getattr(torch, cfg.dtype)
        attn_impl = "flash_attention_2" if cfg.use_flash_attention else "eager"

        logger.info("Loading model from %s (dtype=%s)", cfg.model_path, cfg.dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            torch_dtype=torch_dtype,
            device_map=cfg.device,
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )
        self.model.eval()
        self._loaded = True
        logger.info("Model loaded successfully.")

    def generate(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: Optional[int] = None,
    ) -> List[str]:
        """Generate text for one or more prompts."""
        if not self._loaded:
            raise RuntimeError("Call engine.load() before generate().")

        if isinstance(prompts, str):
            prompts = [prompts]

        cfg = self.config
        max_new_tokens = max_new_tokens or cfg.max_new_tokens

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_seq_len,
        ).to(cfg.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                repetition_penalty=cfg.repetition_penalty,
                do_sample=cfg.do_sample,
            )

        # Decode only newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        results = self.tokenizer.batch_decode(
            output_ids[:, input_len:], skip_special_tokens=True
        )
        return results

    def unload(self) -> None:
        """Free GPU memory."""
        del self.model
        self.model = None
        self._loaded = False
        torch.cuda.empty_cache()
        logger.info("Model unloaded and GPU cache cleared.")
