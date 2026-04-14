"""
hardware/audio.py — ReSpeaker mic array I/O.

Provides:
  WakeWordDetector  — Continuous openwakeword listener (fully local, no API key)
  AudioRecorder     — VAD-gated microphone recorder

The key latency fix in this module is the `mic_ready` threading.Event parameter
on AudioRecorder.record(). The orchestrator (main.py) passes this event to the
recorder, which sets it the instant the sounddevice InputStream is open. The
camera capture thread waits on this event before firing, guaranteeing that mic
capture has started before the shutter trips — so the user's first syllable is
never lost to a race condition.
"""

import logging
import threading
from typing import Optional

import numpy as np
import sounddevice as sd

import config

log = logging.getLogger(__name__)


# =============================================================================
# HELPERS
# =============================================================================

def list_audio_devices() -> None:
    """Log all available audio devices — useful for debugging device indices."""
    log.info("Available audio devices:")
    for i, device in enumerate(sd.query_devices()):
        log.info(
            "  [%d] %s  (in=%d, out=%d)",
            i,
            device["name"],
            device["max_input_channels"],
            device["max_output_channels"],
        )
    log.info("  Default input:  %s", sd.default.device[0])
    log.info("  Default output: %s", sd.default.device[1])


def rms_amplitude(audio_chunk: np.ndarray) -> float:
    """Return the root-mean-square amplitude of a 1-D int16 chunk."""
    return float(np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2)))


def is_silence(
    audio_chunk: np.ndarray,
    threshold: int = config.SILENCE_THRESHOLD,
) -> bool:
    """Return True when the chunk's RMS amplitude is below threshold."""
    return rms_amplitude(audio_chunk) < threshold


# =============================================================================
# WAKE WORD DETECTOR
# =============================================================================

class WakeWordDetector:
    """
    Continuously feeds 80 ms audio chunks to openwakeword until the configured
    wake word is detected.

    Fully local — no API key, no cloud dependency.
    """

    def __init__(
        self,
        model_name:   str   = config.WAKE_WORD_MODEL,
        threshold:    float = config.WAKE_WORD_THRESHOLD,
        display_name: str   = config.WAKE_WORD_DISPLAY,
    ) -> None:
        self.model_name   = model_name
        self.threshold    = threshold
        self.display_name = display_name
        self._model       = None   # OWWModel — loaded in initialize()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        """
        Download (if needed) and load the openwakeword model.

        Returns:
            True on success, False on any import / download error.
        """
        try:
            import openwakeword
            from openwakeword.model import Model as OWWModel
        except ImportError:
            log.error(
                "openwakeword not installed. Run: pip install openwakeword"
            )
            return False

        log.info("Loading wake word model '%s' ...", self.model_name)
        log.info("  (Models are downloaded automatically on first use)")

        try:
            openwakeword.utils.download_models()
            self._model = OWWModel(
                wakeword_models=[self.model_name],
                inference_framework="onnx",
            )
            log.info(
                "Wake word detector ready  trigger='%s'  threshold=%.2f",
                self.display_name,
                self.threshold,
            )
            return True

        except Exception:
            log.exception("Failed to initialise openwakeword")
            return False

    def cleanup(self) -> None:
        """Release the openwakeword model reference."""
        self._model = None
        log.debug("WakeWordDetector released")

    # ------------------------------------------------------------------
    # Detection loop
    # ------------------------------------------------------------------

    def listen_for_wake_word(self) -> bool:
        """
        Block until the wake word is detected on the default input device.

        Feeds 80 ms (1 280 sample) chunks to openwakeword in a tight loop.

        Returns:
            True on detection, False on stream error.
        """
        if self._model is None:
            log.error("WakeWordDetector not initialised — call initialize() first")
            return False

        log.info("Listening for '%s' ...", self.display_name)

        try:
            with sd.InputStream(
                samplerate=config.SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=config.WAKE_WORD_CHUNK_SIZE,
            ) as stream:
                while True:
                    audio_chunk, overflowed = stream.read(
                        config.WAKE_WORD_CHUNK_SIZE
                    )

                    if overflowed:
                        log.warning("Wake word audio buffer overflow")

                    audio_flat  = audio_chunk.flatten().astype(np.int16)
                    predictions: dict[str, float] = self._model.predict(audio_flat)

                    for _name, score in predictions.items():
                        if score >= self.threshold:
                            log.info(
                                "Wake word '%s' detected  score=%.2f",
                                self.display_name,
                                score,
                            )
                            # Reset internal state to prevent repeated triggers
                            self._model.reset()
                            return True

        except Exception:
            log.exception("Error in wake word detection loop")
            return False


# =============================================================================
# AUDIO RECORDER (VAD-GATED)
# =============================================================================

class AudioRecorder:
    """
    Records from the microphone and stops automatically when VAD detects
    a sustained period of silence.

    The optional `mic_ready` threading.Event in record() is the fix for the
    parallel-capture race condition:

        Audio thread sets mic_ready  →  Camera thread wakes up and fires
        ─────────────────────────────────────────────────────────────────
        This guarantees the mic InputStream is open and actively buffering
        before the camera shutter trips, so the user's first phoneme is
        captured even if the camera is initialising.
    """

    def __init__(
        self,
        sample_rate:       int   = config.SAMPLE_RATE,
        silence_threshold: int   = config.SILENCE_THRESHOLD,
        silence_duration:  float = config.SILENCE_DURATION,
        max_duration:      float = config.MAX_RECORDING_DURATION,
        min_duration:      float = config.MIN_RECORDING_DURATION,
        chunk_ms:          int   = config.RECORDING_CHUNK_MS,
    ) -> None:
        self.sample_rate       = sample_rate
        self.silence_threshold = silence_threshold
        self.silence_duration  = silence_duration
        self.max_duration      = max_duration
        self.min_duration      = min_duration
        # Number of samples per chunk (100 ms gives responsive VAD loop)
        self.chunk_samples: int = int(sample_rate * chunk_ms / 1_000)

    def record(
        self,
        mic_ready: Optional[threading.Event] = None,
    ) -> Optional[np.ndarray]:
        """
        Open the microphone and record until VAD silence is detected.

        Args:
            mic_ready: If provided, set() is called the instant the
                       InputStream is open so a waiting camera thread can
                       fire without delay.

        Returns:
            int16 NumPy array of the full recording, or None on error.
        """
        import time

        log.info("Recording ...  speak now, pause when done")

        audio_chunks: list[np.ndarray] = []
        start_time    = time.monotonic()
        last_speech_t = start_time

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=config.CHANNELS,
                dtype=config.DTYPE,
                blocksize=self.chunk_samples,
            ) as stream:

                # --- KEY FIX: signal camera thread that mic is open --------
                # The camera capture thread waits on this event. By setting it
                # here — inside the `with sd.InputStream(...)` block — we
                # guarantee the mic is actively buffering audio before the
                # camera fires. Without this, a scheduler quirk could let the
                # camera thread run first, meaning a slow camera init would
                # delay audio start and clip the user's first words.
                if mic_ready is not None:
                    mic_ready.set()
                    log.debug("mic_ready event set — signalling camera thread")

                while True:
                    chunk, _overflow = stream.read(self.chunk_samples)
                    audio_chunks.append(chunk.copy())

                    now     = time.monotonic()
                    elapsed = now - start_time

                    # Hard ceiling
                    if elapsed >= self.max_duration:
                        log.info(
                            "Max recording duration (%.0fs) reached", self.max_duration
                        )
                        break

                    # VAD — only active after minimum duration
                    if elapsed >= self.min_duration:
                        if not is_silence(chunk, self.silence_threshold):
                            last_speech_t = now
                        elif (now - last_speech_t) >= self.silence_duration:
                            log.info(
                                "Silence detected (%.1fs) — stopping", self.silence_duration
                            )
                            break

                    # Simple level meter on a single log line
                    level = int(np.abs(chunk).mean() / 100)
                    print(
                        f"\r  [{elapsed:.1f}s] {'|' * min(level, 40):<40}",
                        end="",
                        flush=True,
                    )

            # Merge chunks
            recording = np.concatenate(audio_chunks)
            duration  = len(recording) / self.sample_rate
            print()  # newline after the level meter
            log.info("Recorded %.1f s of audio", duration)
            return recording

        except Exception:
            print()
            log.exception("Recording error")
            return None
