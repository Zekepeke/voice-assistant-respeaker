"""
services/tts.py — Streaming text-to-speech via ElevenLabs.

Pipeline
────────
Gemini streams tokens  →  sentence buffer  →  ElevenLabs Turbo v2.5
                                                      ↓
                                           PCM 16 kHz  →  sounddevice

The sentence buffer lets audio start playing as soon as the FIRST complete
sentence arrives — the user hears the answer before Gemini has finished
generating the rest of the response.

Worker-thread architecture
──────────────────────────
A background TTS worker thread dequeues complete sentences and serialises
ElevenLabs requests + sounddevice playback. The main thread keeps feeding
sentences without ever blocking on audio I/O.
"""

import logging
import queue
import re
import threading
from typing import Generator

import numpy as np
import sounddevice as sd

import config

log = logging.getLogger(__name__)

# Regex that splits on whitespace after a sentence-ending punctuation mark.
# "Hello world. How are you?" → ["Hello world.", "How are you?"]
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(buffer: str) -> tuple[list[str], str]:
    """
    Extract complete sentences from the accumulation buffer.

    Returns:
        (complete_sentences, remainder)  — remainder has no trailing sentence end.
    """
    parts = _SENTENCE_END.split(buffer)
    if len(parts) <= 1:
        return [], buffer
    return parts[:-1], parts[-1]


class TextToSpeech:
    """
    ElevenLabs streaming TTS with sentence-level buffering.

    Keep one instance alive for the program lifetime.
    """

    def __init__(
        self,
        api_key:  str = config.ELEVENLABS_API_KEY,
        voice_id: str = config.ELEVENLABS_VOICE_ID,
    ) -> None:
        self.api_key  = api_key
        self.voice_id = voice_id
        self._client  = None  # ElevenLabs — created in initialize()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        """
        Create the ElevenLabs API client.

        Returns:
            True on success, False on import or auth error.
        """
        try:
            from elevenlabs import ElevenLabs
        except ImportError:
            log.error(
                "elevenlabs not installed. Run: pip install elevenlabs"
            )
            return False

        try:
            self._client = ElevenLabs(api_key=self.api_key)
            log.info("ElevenLabs client ready  voice=%s", self.voice_id)
            return True

        except Exception:
            log.exception("Failed to initialise ElevenLabs client")
            return False

    # ------------------------------------------------------------------
    # Playback helpers
    # ------------------------------------------------------------------

    def _play_pcm_stream(self, audio_stream) -> None:
        """
        Collect PCM chunks from an ElevenLabs audio iterator and play them
        immediately via sounddevice.

        Using sd.play() + sd.wait() on the full buffer (rather than
        streaming chunk-by-chunk into sounddevice) gives cleaner playback
        on Pi OS because PortAudio on ARM handles large pre-filled buffers
        more reliably than tiny streaming writes.
        """
        audio_bytes = b"".join(audio_stream)
        if not audio_bytes:
            return
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        sd.play(audio_array, samplerate=config.SAMPLE_RATE)
        sd.wait()

    def speak(self, text: str) -> None:
        """
        Non-streaming TTS — useful for short status phrases like
        "I didn't catch that" where latency is not critical.
        """
        if self._client is None:
            log.error("TextToSpeech not initialised")
            return

        try:
            audio_iter = self._client.text_to_speech.convert(
                voice_id=self.voice_id,
                text=text,
                model_id=config.ELEVENLABS_MODEL_ID,
                output_format=config.ELEVENLABS_OUTPUT_FORMAT,
            )
            self._play_pcm_stream(audio_iter)
        except Exception:
            log.exception("TTS error")

    # ------------------------------------------------------------------
    # Streaming pipeline
    # ------------------------------------------------------------------

    def speak_streaming(self, text_generator: Generator[str, None, None]) -> None:
        """
        Consume a streaming text generator and play audio as sentences complete.

        Thread layout
        ─────────────
        Caller thread  — iterates text_generator, accumulates sentences,
                         enqueues them into sentence_queue.
        Worker thread  — dequeues sentences, calls ElevenLabs, plays audio.

        The worker runs slightly behind the caller, which means:
          • Sentence 1 audio starts playing while Gemini generates sentence 2.
          • No audio gap between sentences (worker queues them up).
          • The caller never blocks on audio I/O.
        """
        if self._client is None:
            log.error("TextToSpeech not initialised")
            return

        log.info("Speaking ...")

        sentence_queue:     queue.Queue[str | None] = queue.Queue()
        streaming_complete: threading.Event         = threading.Event()

        def _worker() -> None:
            """Background thread: sentence → ElevenLabs → speaker."""
            while True:
                try:
                    sentence = sentence_queue.get(timeout=0.1)
                except queue.Empty:
                    # Check if the producer is done and the queue is drained
                    if streaming_complete.is_set() and sentence_queue.empty():
                        break
                    continue

                if sentence is None:
                    # Explicit poison pill — exit cleanly
                    break

                try:
                    audio_stream = self._client.text_to_speech.stream(
                        voice_id=self.voice_id,
                        text=sentence,
                        model_id=config.ELEVENLABS_MODEL_ID,
                        output_format=config.ELEVENLABS_OUTPUT_FORMAT,
                    )
                    self._play_pcm_stream(audio_stream)
                except Exception:
                    log.exception("TTS worker error on sentence %r", sentence)
                finally:
                    sentence_queue.task_done()

        worker = threading.Thread(target=_worker, daemon=True, name="tts-worker")
        worker.start()

        buffer = ""
        try:
            for token in text_generator:
                buffer += token
                sentences, buffer = _split_sentences(buffer)
                for sentence in sentences:
                    if sentence.strip():
                        sentence_queue.put(sentence.strip())

            # Flush any remaining text that did not end with punctuation
            if buffer.strip():
                sentence_queue.put(buffer.strip())

        except Exception:
            log.exception("Error consuming text generator")

        finally:
            streaming_complete.set()
            # Wait for all audio to finish playing (up to 60 s)
            worker.join(timeout=60)
            if worker.is_alive():
                log.warning("TTS worker did not finish within timeout")
