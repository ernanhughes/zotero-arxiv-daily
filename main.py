# llm.py
from __future__ import annotations

import os
from typing import List, Dict, Optional
from llama_cpp import Llama
from loguru import logger


# ============================================================
# Configuration
# ============================================================

DEFAULT_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "model.gguf")
DEFAULT_CONTEXT = int(os.environ.get("LLM_CONTEXT", 4096))
DEFAULT_MAX_TOKENS = 512
SAFE_INPUT_CHARS = 6000  # prevents context overflow


# ============================================================
# Local LLM Wrapper
# ============================================================

class LocalLLM:
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        n_ctx: int = DEFAULT_CONTEXT,
        temperature: float = 0.3,
    ):
        logger.info(f"Loading local model from: {model_path}")
        logger.info(f"Using context window: {n_ctx}")

        self.n_ctx = n_ctx
        self.temperature = temperature

        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_batch=512,
            verbose=False,
        )

    # --------------------------------------------------------
    # Safety: Truncate overly long prompts
    # --------------------------------------------------------

    def _safe_truncate(self, text: str) -> str:
        if len(text) > SAFE_INPUT_CHARS:
            logger.warning("Prompt truncated to prevent context overflow.")
            return text[:SAFE_INPUT_CHARS]
        return text

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        try:
            # Convert messages to a single prompt
            prompt = self._format_messages(messages)

            prompt = self._safe_truncate(prompt)

            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": "You are a helpful academic assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=max_tokens,
            )

            return response["choices"][0]["message"]["content"].strip()

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return "Summary unavailable due to model limitation."

    # --------------------------------------------------------
    # Message Formatting
    # --------------------------------------------------------

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        formatted = ""
        for m in messages:
            formatted += f"{m['role'].upper()}:\n{m['content']}\n\n"
        return formatted


# ============================================================
# Global LLM Interface
# ============================================================

_GLOBAL_LLM: Optional[LocalLLM] = None


def set_global_llm(
    model_path: str = DEFAULT_MODEL_PATH,
    lang: str = "English",
):
    global _GLOBAL_LLM

    _GLOBAL_LLM = LocalLLM(model_path=model_path)

    logger.info(f"Local LLM initialized (Language={lang})")


def get_global_llm() -> LocalLLM:
    if _GLOBAL_LLM is None:
        raise RuntimeError("Global LLM not initialized.")
    return _GLOBAL_LLM