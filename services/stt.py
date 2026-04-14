"""
services/stt.py — Local speech-to-text via faster-whisper.

Runs entirely on-device using int8-quantised Whisper on the Pi 5's
ARM Cortex-A76 CPU. No API key, no network dependency.

Model size guide (speed vs. accuracy on Pi 5):
  tiny.en   ~  40 MB   ~0.4 s RTF  — good for simple commands
  base.en   ~ 140 MB   ~0.8 s RTF  — recommended default
  small.en  ~ 460 MB   ~2.5 s RTF  — highest accuracy for on-device
"""

import logging
import time
from typing import Optional

import numpy as np

import config

log = logging.getLogger(__name__)


class SpeechToText:
    """
    Wraps faster-whisper for on-device int8 transcription.

    Keep one instance alive for the lifetime of the program — model weights
    stay in memory so subsequent calls are fast (no reload penalty).
    """

    def __init__(self, model_name: str = config.WHISPER_MODEL) -> None:
        self.model_name = model_name
        self._model     = None  # WhisperModel — loaded in initialize()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        """
        Load the Whisper model weights into memory.

        The first run may download the model; subsequent runs load from cache.

        Returns:
            True on success, False on import or load error.
        """
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            log.error(
                "faster-whisper not installed. Run: pip install faster-whisper"
            )
            return False

        log.info("Loading Whisper model '%s' ...", self.model_name)
        log.info("  (First run downloads model weights to ~/.cache/huggingface/)")

        try:
            self._model = WhisperModel(
                self.model_name,
                device=config.WHISPER_DEVICE,
                compute_type=config.WHISPER_COMPUTE_TYPE,
            )
            log.info("Whisper '%s' loaded", self.model_name)
            return True

        except Exception:
            log.exception("Failed to load Whisper model")
            return False

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = config.SAMPLE_RATE,
    ) -> str:
        """
        Transcribe an int16 audio array to text.

        Args:
            audio:       1-D or 2-D int16 NumPy array from AudioRecorder.
            sample_rate: Sample rate of the audio (default 16 000 Hz).

        Returns:
            Transcribed text, or an empty string on failure.
        """
        if self._model is None:
            log.error("SpeechToText not initialised — call initialize() first")
            return ""

        log.info("Transcribing ...")
        t0 = time.monotonic()

        try:
            # faster-whisper expects float32 in [-1, 1], flat 1-D array
            audio_f32 = audio.flatten().astype(np.float32) / 32_768.0

            segments, _info = self._model.transcribe(
                audio_f32,
                language="en",
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )

            text = " ".join(seg.text for seg in segments).strip()

            log.info(
                "Transcription done  (%.2f s)  text=%r",
                time.monotonic() - t0,
                text,
            )
            return text

        except Exception:
            log.exception("Transcription error")
            return ""
