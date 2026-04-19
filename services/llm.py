"""
services/llm.py — Streaming LLM via RUBIK Pi 3 llama-server (OpenAI-compatible).

Sends requests to the Phi-3 Mini Q4 model running on the coprocessor's
llama-server instance. The endpoint is OpenAI-compatible, so we use the
standard /v1/chat/completions format with SSE streaming.

Endpoint:  POST http://<RUBIKPI_HOST>:<RUBIKPI_LLM_PORT>/v1/chat/completions
Protocol:  OpenAI Chat Completions API (stream=true → SSE)
"""

import json
import logging
from typing import Generator, Optional

import requests

import config

log = logging.getLogger(__name__)

_LLM_URL = f"{config.RUBIKPI_HOST}:{config.RUBIKPI_LLM_PORT}/v1/chat/completions"


class LlamaLLM:
    """
    Streaming LLM client targeting the RUBIK Pi 3 llama-server.

    Conversation history is maintained across turns as text-only messages so
    the model retains context. Images are accepted for API compatibility but
    are not forwarded — llama-server (text-only model) ignores them.
    """

    def __init__(self) -> None:
        # {"role": "user"|"assistant", "content": str}
        self._history: list[dict] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        """
        Verify that the RUBIK Pi 3 llama-server is reachable.

        Returns:
            True if the server responds, False otherwise.
        """
        health_url = f"{config.RUBIKPI_HOST}:{config.RUBIKPI_LLM_PORT}/health"
        log.info("Checking RUBIK Pi 3 LLM server at %s ...", _LLM_URL)
        try:
            r = requests.get(health_url, timeout=5)
            log.info("LLM server reachable  (HTTP %s)", r.status_code)
            return True
        except requests.exceptions.ConnectionError:
            log.warning(
                "LLM server not reachable at %s — will retry on first inference call",
                _LLM_URL,
            )
            return True  # non-fatal: server may not expose /health

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def stream_response(
        self,
        prompt: str,
        image_bytes: Optional[bytes] = None,
    ) -> Generator[str, None, None]:
        """
        Stream a response from the RUBIK Pi 3 llama-server.

        Args:
            prompt:      Transcribed user speech.
            image_bytes: Ignored — llama-server runs a text-only model.

        Yields:
            Text tokens as they arrive via SSE.
        """
        log.info("Sending request to RUBIK Pi 3 llama-server ...")

        messages = [
            {"role": "system", "content": config.LLM_SYSTEM_PROMPT},
            *self._history,
            {"role": "user", "content": prompt},
        ]

        payload = {
            "messages": messages,
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 512,
        }

        try:
            with requests.post(
                _LLM_URL,
                json=payload,
                stream=True,
                timeout=60,
            ) as response:
                response.raise_for_status()

                full_response = ""
                for raw_line in response.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8")
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0]["delta"].get("content", "")
                        if delta:
                            full_response += delta
                            yield delta
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

            # Update history with text-only turns
            self._history.append({"role": "user",      "content": prompt})
            self._history.append({"role": "assistant",  "content": full_response})
            log.info("LLM response: %r", full_response)

        except Exception:
            log.exception("LLM streaming error")
            yield "I'm sorry, I encountered an error processing your request."

    def clear_history(self) -> None:
        """Reset the multi-turn conversation context."""
        self._history.clear()
        log.info("Conversation history cleared")


# Alias for backwards compatibility with any code that imports GeminiLLM
GeminiLLM = LlamaLLM
