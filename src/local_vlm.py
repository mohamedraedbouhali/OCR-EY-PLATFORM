"""Local VLM engine served by Ollama — fully offline, runs on this machine.

Default engine of the pipeline: no API key, no data leaves the computer.
Requires the Ollama server (https://ollama.com) with a vision-capable model
pulled, e.g. `ollama pull qwen3-vl:4b` (fits a 6 GB laptop GPU).
"""

from __future__ import annotations

import ollama

from .config import MAX_TOKENS, OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_NUM_CTX
from .vlm_base import TRUNCATION_WARNING, VLMEngine, VLMResult


def is_available(model: str = OLLAMA_MODEL, host: str = OLLAMA_HOST) -> bool:
    """True when the Ollama server responds and the configured model is pulled."""
    try:
        client = ollama.Client(host=host, timeout=3)
        names = [m.model or "" for m in client.list().models]
    except Exception:
        return False
    base = model.split(":")[0]
    return any(n == model or n.split(":")[0] == base for n in names)


class OllamaOCR(VLMEngine):
    def __init__(self, model: str = OLLAMA_MODEL, host: str = OLLAMA_HOST):
        self.model = model
        self.client = ollama.Client(host=host)

    def _generate(self, system: str, text: str,
                  image: tuple[bytes, str] | None = None) -> VLMResult:
        user: dict = {"role": "user", "content": text}
        if image is not None:
            user["images"] = [image[0]]  # raw bytes; the client base64-encodes them

        response = self.client.chat(
            model=self.model,
            messages=[{"role": "system", "content": system}, user],
            options={"num_predict": MAX_TOKENS, "temperature": 0,
                     "num_ctx": OLLAMA_NUM_CTX},
        )
        out = (response.message.content or "").strip()
        if not out and getattr(response.message, "thinking", None):
            raise RuntimeError(
                f"model '{self.model}' spent its whole token budget in 'thinking' "
                f"mode and returned no transcription — use an *-instruct tag "
                f"instead (e.g. qwen3-vl:4b-instruct)")
        if response.done_reason == "length":
            out += TRUNCATION_WARNING
        return VLMResult(
            text=out,
            model=response.model or self.model,
            input_tokens=response.prompt_eval_count or 0,
            output_tokens=response.eval_count or 0,
            stop_reason=response.done_reason,
        )
