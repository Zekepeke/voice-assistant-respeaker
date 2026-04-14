"""
services/llm.py — Streaming multimodal LLM via Google Gemini.

Multimodal request structure
─────────────────────────────
When image_bytes is provided, the current turn's Content is built as:

    Content(role="user", parts=[
        Part(inline_data=Blob(mime_type="image/jpeg", data=<bytes>)),
        Part(text=<transcript>),
    ])

The image comes FIRST — Gemini attends to leading parts more reliably.

History management
──────────────────
Conversation history is maintained across turns as text-only Content
objects so the model retains context. Images are intentionally NOT stored
in history: re-sending large JPEGs on every turn would make requests
expensive and slow. The model already has the description from its
previous answer if the user asks a follow-up about what it saw.
"""

import logging
from typing import Generator, Optional

import config

log = logging.getLogger(__name__)


class GeminiLLM:
    """
    Streaming Google Gemini client with optional vision input.

    Keep one instance alive for the program lifetime — the client
    connection and conversation history are maintained across turns.
    """

    def __init__(
        self,
        api_key: str = config.GEMINI_API_KEY,
        model:   str = config.GEMINI_MODEL,
    ) -> None:
        self.api_key = api_key
        self.model   = model
        self._client = None  # genai.Client — created in initialize()

        # Text-only turn history: list of {"role": str, "parts": [...]}
        self._history: list[dict] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        """
        Create the google-genai client.

        Returns:
            True on success, False on import or auth error.
        """
        try:
            from google import genai
        except ImportError:
            log.error(
                "google-genai not installed. Run: pip install google-genai"
            )
            return False

        try:
            self._client = genai.Client(api_key=self.api_key)
            log.info("Gemini client ready  model=%s", self.model)
            return True

        except Exception:
            log.exception("Failed to initialise Gemini client")
            return False

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def stream_response(
        self,
        prompt:      str,
        image_bytes: Optional[bytes] = None,
    ) -> Generator[str, None, None]:
        """
        Stream a response from Gemini, with an optional camera image.

        Args:
            prompt:      Transcribed user speech.
            image_bytes: In-memory JPEG bytes (from CameraCapture).
                         Pass None for a text-only request.

        Yields:
            Text tokens as they arrive from the API.
        """
        if self._client is None:
            log.error("GeminiLLM not initialised — call initialize() first")
            return

        # Late import to avoid module-level dependency issues
        from google.genai import types as genai_types

        has_image = bool(image_bytes)
        log.info(
            "Sending request to Gemini  [%s]",
            "vision + text" if has_image else "text-only",
        )

        try:
            # --- Build current-turn Content --------------------------------
            # Image first (if present), then the text question.
            parts: list = []

            if has_image:
                parts.append(
                    genai_types.Part(
                        inline_data=genai_types.Blob(
                            mime_type="image/jpeg",
                            data=image_bytes,
                        )
                    )
                )

            parts.append(genai_types.Part(text=prompt))

            current_content = genai_types.Content(role="user", parts=parts)

            # --- Assemble full contents list --------------------------------
            # Prepend text-only history so Gemini has conversational context.
            contents = list(self._history) + [current_content]

            # --- Stream from Gemini -----------------------------------------
            response = self._client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    system_instruction=config.SYSTEM_PROMPT,
                    temperature=config.GEMINI_TEMPERATURE,
                    max_output_tokens=config.GEMINI_MAX_TOKENS,
                ),
            )

            full_response = ""
            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
                    yield chunk.text

            # --- Update history (text-only) --------------------------------
            # Store the user's text (not the image) so history stays compact.
            self._history.append(
                {"role": "user",  "parts": [{"text": prompt}]}
            )
            self._history.append(
                {"role": "model", "parts": [{"text": full_response}]}
            )

            log.info("Gemini response: %r", full_response)

        except Exception:
            log.exception("Gemini streaming error")
            yield "I'm sorry, I encountered an error processing your request."

    def clear_history(self) -> None:
        """Reset the multi-turn conversation context."""
        self._history.clear()
        log.info("Conversation history cleared")
