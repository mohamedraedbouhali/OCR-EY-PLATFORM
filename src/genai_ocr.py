"""Cloud VLM engine built on the Groq API (Llama 4 vision models).

Optional — the pipeline defaults to the local Ollama engine (src/local_vlm.py);
this one is kept as a selectable cloud alternative for benchmarking.

Groq's vision models accept images only (no native PDF input), so PDF handling
is done by the shared base class, which rasterizes pages locally with PyMuPDF.
"""

from __future__ import annotations

import base64

from groq import Groq

from .config import MAX_TOKENS, MODEL
from .vlm_base import TRUNCATION_WARNING, VLMEngine, VLMResult

GenAIResult = VLMResult  # backward-compat alias


class GroqOCR(VLMEngine):
    def __init__(self, api_key: str | None = None):
        # Falls back to GROQ_API_KEY from the environment
        self.client = Groq(api_key=api_key) if api_key else Groq()

    def _generate(self, system: str, text: str,
                  image: tuple[bytes, str] | None = None) -> VLMResult:
        if image is not None:
            data = base64.standard_b64encode(image[0]).decode()
            content: list | str = [
                {"type": "image_url",
                 "image_url": {"url": f"data:{image[1]};base64,{data}"}},
                {"type": "text", "text": text},
            ]
        else:
            content = text

        completion = self.client.chat.completions.create(
            model=MODEL,
            max_completion_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
        )
        choice = completion.choices[0]
        out = (choice.message.content or "").strip()
        if choice.finish_reason == "length":
            out += TRUNCATION_WARNING
        usage = completion.usage
        return VLMResult(
            text=out,
            model=completion.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            stop_reason=choice.finish_reason,
        )
