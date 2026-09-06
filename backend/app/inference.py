"""Model loading and text generation.

Two modes, chosen by settings.use_real_model:
  - real:  loads Qwen2.5-1.5B-Instruct in 4-bit (bitsandbytes NF4) via `transformers` +
           the trained LoRA adapter via `peft.PeftModel`, and runs real `.generate()` calls.
           Requires a CUDA GPU (this project targets an RTX 4050, 6GB VRAM).
  - mock:  used in CI and in tests where no GPU is available / no adapter has been trained yet.
           Returns a deterministic templated response so the API, caching, and DB plumbing can
           be exercised without a GPU. Never used to fabricate the evaluation numbers reported
           in the README - those come from scripts/evaluate.py run against the real model.
"""

import hashlib
import os
import time
from functools import lru_cache

from app.config import Settings, get_settings


class GenerationResult:
    def __init__(self, text: str, tokens_used: int, inference_time_ms: int):
        self.text = text
        self.tokens_used = tokens_used
        self.inference_time_ms = inference_time_ms


def input_hash(prompt: str, max_tokens: int, temperature: float) -> str:
    key = f"{prompt}|{max_tokens}|{temperature}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


class MockGenerator:
    """Deterministic, GPU-free stand-in used by CI and unit tests."""

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> GenerationResult:
        start = time.time()
        text = (
            "Thanks for reaching out! Based on your message, here is what I'd suggest: "
            f"please check your account settings and try again. (mock response for: {prompt[:60]!r})"
        )
        elapsed_ms = int((time.time() - start) * 1000) + 5  # never exactly 0, mirrors real latency floor
        return GenerationResult(text=text, tokens_used=len(text.split()), inference_time_ms=elapsed_ms)


class RealGenerator:
    """Loads Qwen2.5-1.5B-Instruct + trained LoRA adapter in 4-bit and runs real generation."""

    def __init__(self, settings: Settings):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError(
                "use_real_model=True but no CUDA GPU is available. Set USE_REAL_MODEL=false "
                "to run in mock mode, or run on a machine with an NVIDIA GPU."
            )

        adapter_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", settings.adapter_dir)
        )
        if not os.path.isdir(adapter_dir):
            raise RuntimeError(
                f"No trained adapter found at {adapter_dir}. Run scripts/train.py first, or set "
                "USE_REAL_MODEL=false to run in mock mode."
            )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(settings.base_model_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            settings.base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        self.model = PeftModel.from_pretrained(base_model, adapter_dir)
        self.model.eval()
        self.torch = torch

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> GenerationResult:
        messages = [
            {"role": "system", "content": "You are a helpful customer support assistant."},
            {"role": "user", "content": prompt},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        start = time.time()
        with self.torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                do_sample=temperature > 0,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        elapsed_ms = int((time.time() - start) * 1000)

        generated = output_ids[0][input_ids.shape[-1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return GenerationResult(text=text.strip(), tokens_used=len(generated), inference_time_ms=elapsed_ms)


@lru_cache
def get_generator():
    settings = get_settings()
    if settings.use_real_model:
        return RealGenerator(settings)
    return MockGenerator()
