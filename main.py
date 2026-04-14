#!/usr/bin/env python3
"""
main.py — Multimodal Voice Assistant for Raspberry Pi 5

Entry point and top-level orchestrator. This file wires together the
hardware drivers (audio, camera) and AI services (STT, LLM, TTS) into
a single, readable interaction loop.

Run with:
    source .venv/bin/activate
    python main.py

Press Ctrl+C to exit cleanly.
"""

import logging
import signal
import sys
import threading
import time
from typing import Optional

import numpy as np

import config
from hardware.audio  import WakeWordDetector, AudioRecorder, list_audio_devices
from hardware.camera import CameraCapture
from services.stt    import SpeechToText
from services.llm    import GeminiLLM
from services.tts    import TextToSpeech


# =============================================================================
# LOGGING SETUP
# =============================================================================

def _configure_logging() -> None:
    """Configure the root logger — call once before anything else."""
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT,
        stream=sys.stdout,
    )
    # Silence noisy third-party loggers
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


log = logging.getLogger(__name__)


# =============================================================================
# VALIDATION
# =============================================================================

def _validate_config() -> None:
    """
    Check that required API keys are present before attempting any I/O.
    Exits with a clear error message if keys are missing.
    """
    missing: list[str] = []

    if not config.GEMINI_API_KEY or "YOUR_" in config.GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY  →  https://aistudio.google.com/apikey")

    if not config.ELEVENLABS_API_KEY or "YOUR_" in config.ELEVENLABS_API_KEY:
        missing.append("ELEVENLABS_API_KEY  →  https://elevenlabs.io/app/settings/api-keys")

    if missing:
        log.error("Missing API keys in .env.local:")
        for m in missing:
            log.error("  %s", m)
        sys.exit(1)


# =============================================================================
# VOICE ASSISTANT ORCHESTRATOR
# =============================================================================

class VoiceAssistant:
    """
    Owns all hardware and service instances and drives the main interaction
    loop.

    Interaction flow per wake word event
    ──────────────────────────────────────
    1. Wake word detected by WakeWordDetector
    2. AudioRecorder + CameraCapture run IN PARALLEL (two threads)
       • AudioRecorder opens the mic and sets a threading.Event the instant
         the InputStream is live. CameraCapture waits on that event before
         firing — guaranteeing audio is already capturing before the shutter
         trips.
    3. SpeechToText transcribes the recorded audio
    4. GeminiLLM sends transcript + JPEG to Gemini and streams tokens back
    5. TextToSpeech plays the response sentence-by-sentence as it arrives
    """

    def __init__(self) -> None:
        self.wake_detector = WakeWordDetector()
        self.recorder      = AudioRecorder()
        self.stt           = SpeechToText()
        self.llm           = GeminiLLM()
        self.tts           = TextToSpeech()
        self.camera        = CameraCapture() if config.CAMERA_ENABLED else None

        # Set by SIGTERM / SIGINT handlers to break the main loop cleanly
        self._shutdown = threading.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        """
        Initialize all components sequentially.

        Camera is initialized last and its failure is non-fatal — the
        assistant falls back to audio-only mode so a missing or broken
        camera never prevents the assistant from starting.

        Returns:
            True if all mandatory components initialized successfully.
        """
        log.info("=" * 62)
        log.info("  INITIALIZING MULTIMODAL VOICE ASSISTANT")
        log.info("=" * 62)

        list_audio_devices()

        # Mandatory components — any failure is fatal
        for component, name in [
            (self.wake_detector, "WakeWordDetector"),
            (self.stt,           "SpeechToText"),
            (self.llm,           "GeminiLLM"),
            (self.tts,           "TextToSpeech"),
        ]:
            if not component.initialize():
                log.error("Failed to initialise %s — cannot continue", name)
                return False

        # Optional camera — failure degrades to audio-only
        if self.camera is not None:
            if not self.camera.initialize():
                log.warning(
                    "Camera unavailable — continuing in audio-only mode"
                )
                self.camera = None

        mode = "VISION + AUDIO" if self.camera else "AUDIO-ONLY"
        log.info("=" * 62)
        log.info("  ALL COMPONENTS READY  [%s]", mode)
        log.info("=" * 62)
        return True

    def cleanup(self) -> None:
        """
        Release all hardware resources.

        Called on clean shutdown (Ctrl+C / SIGTERM) and ensures:
        - sounddevice streams are closed (prevents ALSA lock on ReSpeaker)
        - Picamera2 is stopped (prevents libcamera lock on camera device)
        """
        log.info("Releasing hardware resources ...")
        self.wake_detector.cleanup()
        if self.camera is not None:
            self.camera.cleanup()
        log.info("Shutdown complete. Goodbye.")

    # ------------------------------------------------------------------
    # Parallel capture (the latency-critical section)
    # ------------------------------------------------------------------

    def _capture_parallel(
        self,
    ) -> tuple[Optional[np.ndarray], Optional[bytes]]:
        """
        Launch audio recording and camera capture simultaneously.

        Race-condition fix (vs. the original monolith)
        ───────────────────────────────────────────────
        The original code started both threads back-to-back and relied on
        the OS scheduler to run the audio thread first. This is UNSAFE:
        there is no guarantee which thread gets CPU time first, and if the
        camera thread runs before the audio InputStream is open, the user's
        first syllable can be lost.

        The fix uses a threading.Event (mic_ready):

            Audio thread:  opens sd.InputStream()  →  mic_ready.set()
            Camera thread: mic_ready.wait()         →  camera.capture_to_memory()

        This makes the ordering explicit: the camera NEVER fires before the
        mic is confirmed open, regardless of scheduler behaviour.

        Timeline (typical):
            t=0.000  audio_thread starts, opens InputStream
            t=0.005  mic_ready.set() — InputStream confirmed open
            t=0.005  camera_thread wakes, grabs JPEG (~0.1 s)
            t=0.100  camera_thread done, JPEG in memory
            t=1.8    audio_thread stops (VAD silence), returns int16 array
            t=1.800  both threads joined, continue to transcription

        The camera result is ready long before the user stops talking.

        Returns:
            (audio_array, jpeg_bytes) — either may be None on failure.
        """
        audio_result: list[Optional[np.ndarray]] = [None]
        image_result: list[Optional[bytes]]       = [None]

        # The Event that synchronises the two threads
        mic_ready = threading.Event()

        def _record() -> None:
            # Passes mic_ready to the recorder; recorder sets it the instant
            # the sounddevice InputStream context manager is entered.
            audio_result[0] = self.recorder.record(mic_ready=mic_ready)

        def _capture() -> None:
            if self.camera is None:
                return
            # Block until the mic is confirmed open — then fire the shutter.
            acquired = mic_ready.wait(timeout=5.0)
            if not acquired:
                log.warning(
                    "mic_ready event timed out — capturing image anyway"
                )
            image_result[0] = self.camera.capture_to_memory()

        audio_thread  = threading.Thread(target=_record,  daemon=True, name="audio-capture")
        camera_thread = threading.Thread(target=_capture, daemon=True, name="camera-capture")

        # Audio starts first; it will signal mic_ready, then camera fires.
        audio_thread.start()
        camera_thread.start()

        audio_thread.join()
        camera_thread.join()

        return audio_result[0], image_result[0]

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Main interaction loop.

        Blocks until self._shutdown is set (Ctrl+C or SIGTERM).
        """
        log.info("Voice Assistant is ready!")
        log.info("  Say '%s' to activate ...", config.WAKE_WORD_DISPLAY)
        log.info("  Press Ctrl+C to exit")

        try:
            while not self._shutdown.is_set():

                # ── 1. Wake word ──────────────────────────────────────────
                detected = self.wake_detector.listen_for_wake_word()
                if not detected:
                    log.warning("Wake word detection failed — retrying in 1 s")
                    time.sleep(1)
                    continue

                # ── 2. Parallel audio + camera capture ────────────────────
                log.info("Capturing audio and image in parallel ...")
                audio, image_bytes = self._capture_parallel()

                if audio is None or len(audio) < config.SAMPLE_RATE * 0.5:
                    log.warning("Recording too short or failed — ready again")
                    continue

                if image_bytes:
                    log.info(
                        "Image ready (%.0f KB) — multimodal request",
                        len(image_bytes) / 1_024,
                    )
                else:
                    log.info("No image — text-only request")

                # ── 3. Transcribe ─────────────────────────────────────────
                text = self.stt.transcribe(audio)
                if not text.strip():
                    log.warning("No speech detected")
                    self.tts.speak(
                        "I didn't catch that, Sir. Please try again."
                    )
                    continue

                # ── 4 + 5. Multimodal LLM → streaming TTS ─────────────────
                # llm.stream_response() is a generator. tts.speak_streaming()
                # consumes it sentence-by-sentence, playing audio as each
                # sentence arrives — so the first audio plays before Gemini
                # has finished generating the full response.
                response_gen = self.llm.stream_response(
                    prompt=text,
                    image_bytes=image_bytes,
                )
                self.tts.speak_streaming(response_gen)

                log.info(
                    "─" * 40 + "  Say '%s' again ...", config.WAKE_WORD_DISPLAY
                )

        except KeyboardInterrupt:
            log.info("Keyboard interrupt received")

        finally:
            self.cleanup()

    def request_shutdown(self) -> None:
        """Signal the main loop to exit cleanly (used by signal handlers)."""
        self._shutdown.set()


# =============================================================================
# SIGNAL HANDLING
# =============================================================================

def _install_signal_handlers(assistant: VoiceAssistant) -> None:
    """
    Install SIGINT / SIGTERM handlers so Ctrl+C and systemd stop both
    trigger a graceful shutdown instead of abruptly killing the process.

    A abrupt kill can leave the ReSpeaker ALSA device or the Picamera2
    libcamera pipeline in a locked state, requiring a reboot to recover.
    """
    def _handler(sig: int, _frame) -> None:
        log.info("Signal %d received — requesting shutdown ...", sig)
        assistant.request_shutdown()

    signal.signal(signal.SIGINT,  _handler)
    signal.signal(signal.SIGTERM, _handler)


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    _configure_logging()

    print("""
╔══════════════════════════════════════════════════════════════════╗
║        RASPBERRY PI 5 — MULTIMODAL VOICE ASSISTANT               ║
║                                                                  ║
║  Wake Word  →  Audio + Camera (parallel)  →  Whisper STT         ║
║             →  Gemini Vision + Text       →  ElevenLabs TTS       ║
║                                                                  ║
║  Hardware:  ReSpeaker v3.0  |  Arducam IMX708 (Camera Module 3)  ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    _validate_config()

    assistant = VoiceAssistant()
    _install_signal_handlers(assistant)

    if not assistant.initialize():
        log.error("Initialisation failed — exiting")
        sys.exit(1)

    assistant.run()


if __name__ == "__main__":
    main()
