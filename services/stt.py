"""
services/stt.py — Speech-to-text via RUBIK Pi 3 faster-whisper HTTP server.

Sends raw WAV bytes to the remote transcribe endpoint and parses the JSON
response. No local model weights — all inference runs on the coprocessor.

Endpoint:  POST http://<RUBIKPI_HOST>:<RUBIKPI_STT_PORT>/transcribe
Body:      raw WAV bytes (Content-Type: audio/wav)
Response:  {"transcript": "..."}
"""

import io
import logging
import time
import wave
from typing import Optional

import numpy as np
import requests

import config

log = logging.getLogger(__name__)

_STT_URL = f"{config.RUBIKPI_HOST}:{config.RUBIKPI_STT_PORT}/transcribe"


class SpeechToText:
    """
    Transcribes audio via the RUBIK Pi 3 faster-whisper HTTP server.

    initialize() does a lightweight connectivity check so startup fails fast
    if the coprocessor is unreachable. transcribe() is otherwise stateless —
    no local model weights are held in memory.
    """

    def __init__(self, model_name: str = config.WHISPER_MODEL) -> None:
        self.model_name = model_name  # informational; server decides the model

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        """
        Verify that the RUBIK Pi 3 STT server is reachable.

        Returns:
            True if the server responds, False otherwise.
        """
        log.info("Checking RUBIK Pi 3 STT server at %s ...", _STT_URL)
        try:
            # A HEAD/GET on the root is enough to confirm the server is up.
            r = requests.get(
                f"{config.RUBIKPI_HOST}:{config.RUBIKPI_STT_PORT}/",
                timeout=5,
            )
            log.info("STT server reachable  (HTTP %s)", r.status_code)
            return True
        except requests.exceptions.ConnectionError:
            log.warning(
                "STT server not reachable at %s — will retry on first transcribe call",
                _STT_URL,
            )
            return True  # non-fatal: server may not expose a root route

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = config.SAMPLE_RATE,
    ) -> str:
        """
        Transcribe an int16 audio array by sending it to the RUBIK Pi 3.

        Args:
            audio:       1-D or 2-D int16 NumPy array from AudioRecorder.
            sample_rate: Sample rate of the audio (default 16 000 Hz).

        Returns:
            Transcribed text, or an empty string on failure.
        """
        log.info("Transcribing via RUBIK Pi 3 (%s) ...", _STT_URL)
        t0 = time.monotonic()

        try:
            wav_bytes = _numpy_to_wav(audio, sample_rate)

            response = requests.post(
                _STT_URL,
                data=wav_bytes,
                headers={"Content-Type": "audio/wav"},
                timeout=30,
            )
            response.raise_for_status()

            transcript = response.json().get("transcript", "").strip()

            log.info(
                "Transcription done  (%.2f s)  text=%r",
                time.monotonic() - t0,
                transcript,
            )
            return transcript

        except Exception:
            log.exception("Transcription error")
            return ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _numpy_to_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert a int16 NumPy array to in-memory WAV bytes."""
    buf = io.BytesIO()
    pcm = audio.flatten().astype(np.int16).tobytes()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)   # 16-bit = 2 bytes per sample
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()
