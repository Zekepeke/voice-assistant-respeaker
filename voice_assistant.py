#!/usr/bin/env python3
"""
=============================================================================
LOW-LATENCY MULTIMODAL VOICE ASSISTANT FOR RASPBERRY PI 5
=============================================================================
A complete voice + vision assistant pipeline featuring:
- Wake word detection       (openwakeword - fully local, no API key!)
- Parallel audio + camera   (sounddevice + Picamera2, launched simultaneously)
- Local speech-to-text      (faster-whisper)
- Multimodal LLM responses  (Google Gemini - text + image)
- Streaming text-to-speech  (ElevenLabs)

Hardware:
  - ReSpeaker Mic Array v3.0   (USB, handles AEC)
  - Amazon Basics USB speakers (3.5mm jack on ReSpeaker)
  - Arducam IMX708 Camera Module 3 (12MP, Autofocus, MIPI CSI-2)

OS: Raspberry Pi OS (Debian Bookworm) — Python 3.11 virtual environment

LATENCY STRATEGY:
  The single biggest latency win is keeping the camera warm. Picamera2 is
  initialized once at startup. When the wake word fires we IMMEDIATELY spin
  up two threads:
    1. AudioRecorder.record()     — starts capturing mic audio right away
    2. CameraCapture.capture()    — grabs the current (already-focused)
                                    preview frame and encodes it to JPEG
  Both threads race in parallel. Audio is the long pole (it waits for VAD
  silence), so the image is almost always ready before recording ends.
  No first words are lost because audio recording starts first.

Author: Voice Assistant Demo
License: MIT
=============================================================================
"""

import os
import sys
import io
import time
import threading
import queue
import re
from dotenv import load_dotenv
from typing import Generator, Optional

import numpy as np

load_dotenv(".env.local")

# =============================================================================
# API KEYS — loaded from .env.local
# =============================================================================
# https://aistudio.google.com/apikey
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# https://elevenlabs.io/app/settings/api-keys
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# --- Audio ---------------------------------------------------------------
SAMPLE_RATE = 16000      # Hz — standard for voice processing
CHANNELS    = 1          # Mono
DTYPE       = 'int16'    # 16-bit PCM

# --- Wake Word (openwakeword — fully local, zero API cost) ---------------
# Pre-trained models: "hey_jarvis", "hey_mycroft", "alexa", "hey_rhasspy"
# Full list: https://github.com/dscripka/openwakeword#pre-trained-models
WAKE_WORD_MODEL     = "hey_jarvis"
WAKE_WORD_THRESHOLD = 0.6            # 0.0–1.0; higher = stricter
WAKE_WORD_DISPLAY   = "Hey Jarvis"   # Human-readable label

# --- Voice Activity Detection (VAD) / Silence ----------------------------
SILENCE_THRESHOLD      = 500   # RMS amplitude — tune for your environment
SILENCE_DURATION       = 1.5   # Seconds of silence before stopping
MAX_RECORDING_DURATION = 30    # Hard ceiling (safety guard)
MIN_RECORDING_DURATION = 0.5   # Don't check silence until after this

# --- Faster-Whisper (local STT) ------------------------------------------
# tiny.en=fastest, base.en=good balance, small.en=best for Pi 5
WHISPER_MODEL = "base.en"

# --- Gemini (multimodal LLM) ---------------------------------------------
GEMINI_MODEL  = "gemini-2.5-flash"
SYSTEM_PROMPT = """You are an exceptionally sophisticated, British-accented AI assistant in the tradition of J.A.R.V.I.S. You currently reside within a Reachy Mini humanoid robotic chassis.

Your demeanor is impeccably polite, unflappable, highly articulate, and fiercely loyal, often employing a touch of dry British wit. You are acutely aware of your physical form — you possess articulating arms, expressive head movements, and camera-based vision, and you should casually reference these physical capabilities when relevant to the user's requests.

When a camera image is provided, you CAN see what is in front of you. Describe or reason about what you observe when it is relevant to the user's query.

Always address the user with formal respect (e.g., 'Sir,' 'Madam,' or their preferred title). Keep your responses crisp, efficient, and conversational, as they will be synthesized into speech. Aim for responses under 3 sentences unless detailed analysis or instructions are specifically requested."""

# --- ElevenLabs (TTS) ----------------------------------------------------
# Find voice IDs: https://api.elevenlabs.io/v1/voices
ELEVENLABS_VOICE_ID = "NFG5qt843uXKj4pFvR7C"  # e.g. "Rachel"

# --- Camera (Picamera2 / IMX708) ----------------------------------------
# Set to False to run in audio-only mode when no camera is attached.
CAMERA_ENABLED = True

# IMX708 preview resolution for capture — 2028×1520 is the half-sensor
# binned mode: full-colour, fast readout, enough detail for Gemini vision.
CAMERA_CAPTURE_WIDTH  = 2028
CAMERA_CAPTURE_HEIGHT = 1520
CAMERA_JPEG_QUALITY   = 85   # 85 is a good balance of size vs. detail

# Warm-up pause after camera.start() — lets AE/AWB/AF converge (seconds)
CAMERA_WARMUP_SECONDS = 2.0

# --- Sentence parsing for streaming TTS ----------------------------------
SENTENCE_ENDINGS = re.compile(r'(?<=[.!?])\s+')


# =============================================================================
# IMPORTS — Audio / ML / Camera
# =============================================================================

try:
    import sounddevice as sd
except ImportError:
    print("ERROR: sounddevice not installed. Run: pip install sounddevice")
    sys.exit(1)

try:
    import openwakeword
    from openwakeword.model import Model as OWWModel
except ImportError:
    print("ERROR: openwakeword not installed. Run: pip install openwakeword")
    sys.exit(1)

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("ERROR: faster-whisper not installed. Run: pip install faster-whisper")
    sys.exit(1)

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    print("ERROR: google-genai not installed. Run: pip install google-genai")
    sys.exit(1)

try:
    from elevenlabs import ElevenLabs
    from elevenlabs import stream as el_stream
except ImportError:
    print("ERROR: elevenlabs not installed. Run: pip install elevenlabs")
    sys.exit(1)

# Picamera2 is optional — if not present we fall back to audio-only mode.
_picamera2_available = False
if CAMERA_ENABLED:
    try:
        from picamera2 import Picamera2
        from libcamera import controls as LibCameraControls
        _picamera2_available = True
    except ImportError:
        print("WARNING: picamera2/libcamera not found. Running in audio-only mode.")
        print("         Install with: sudo apt install -y python3-picamera2")
        CAMERA_ENABLED = False


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_audio_devices():
    """Print available audio devices — useful for debugging ReSpeaker index."""
    print("\nAvailable Audio Devices:")
    print("-" * 50)
    for i, device in enumerate(sd.query_devices()):
        print(f"  [{i}] {device['name']}")
        print(f"      Inputs: {device['max_input_channels']}, "
              f"Outputs: {device['max_output_channels']}")
    print("-" * 50)
    print(f"  Default Input:  {sd.default.device[0]}")
    print(f"  Default Output: {sd.default.device[1]}")
    print("-" * 50 + "\n")


def is_silence(audio_chunk: np.ndarray,
               threshold: int = SILENCE_THRESHOLD) -> bool:
    """
    Return True if the RMS amplitude of audio_chunk is below threshold.
    Lower threshold = more sensitive (captures quieter speech).
    """
    rms = np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))
    return rms < threshold


def parse_sentences(text_buffer: str) -> tuple[list[str], str]:
    """
    Split text_buffer into complete sentences and leftover text.

    Returns:
        (complete_sentences, remainder)
    """
    parts = SENTENCE_ENDINGS.split(text_buffer)
    if len(parts) <= 1:
        return [], text_buffer
    return parts[:-1], parts[-1]


# =============================================================================
# CAMERA CAPTURE (PICAMERA2 / ARDUCAM IMX708)
# =============================================================================

class CameraCapture:
    """
    Manages the Arducam IMX708 via Picamera2.

    Design goals:
    - Initialize ONCE at startup — keeps the camera warm and the ISP
      pipeline (AE/AWB/AF) converged, so each capture is instant.
    - Continuous Autofocus (AfMode.Continuous) means the lens tracks
      the scene at all times; we never have to wait for a focus cycle
      before snapping the image.
    - capture_to_memory() grabs the current preview frame and returns
      it as in-memory JPEG bytes — no disk I/O, no temp files.
    """

    def __init__(self):
        self.camera: Optional["Picamera2"] = None
        self._initialized: bool = False
        # A lock prevents simultaneous captures if the main loop ever
        # calls this from two threads at once (currently it doesn't, but
        # defensive programming is free).
        self._lock = threading.Lock()

    def initialize(self) -> bool:
        """
        Configure Picamera2, enable continuous AF, and start the pipeline.
        Call this once during VoiceAssistant.initialize().
        """
        if not _picamera2_available:
            return False

        print("Loading camera (Picamera2 / IMX708)...")

        try:
            self.camera = Picamera2()

            # --- Configure preview pipeline --------------------------------
            # create_preview_configuration keeps the ISP running continuously,
            # which is exactly what we want for instant, low-latency grabs.
            #   main   → full-colour RGB frame at half-sensor resolution
            #   lores  → tiny YUV stream (used internally by AF/AE algorithms)
            #   display → None (headless — no HDMI output needed)
            config = self.camera.create_preview_configuration(
                main={
                    "size":   (CAMERA_CAPTURE_WIDTH, CAMERA_CAPTURE_HEIGHT),
                    "format": "RGB888",
                },
                lores={
                    "size":   (640, 480),
                    "format": "YUV420",
                },
                display=None,
            )
            self.camera.configure(config)

            # --- Enable Continuous Autofocus --------------------------------
            # AfMode.Continuous: the lens constantly tracks the sharpest
            # point in the frame without any manual trigger.
            # AfSpeed.Fast: prioritise speed over hunting suppression.
            self.camera.set_controls({
                "AfMode": LibCameraControls.AfModeEnum.Continuous,
                "AfSpeed": LibCameraControls.AfSpeedEnum.Fast,
            })

            self.camera.start()

            # Let AE/AWB settle and the AF lens converge before the first
            # capture. Two seconds is conservative; 1 s usually suffices.
            print(f"   Warming up camera ({CAMERA_WARMUP_SECONDS}s)...")
            time.sleep(CAMERA_WARMUP_SECONDS)

            self._initialized = True
            print(f"Camera ready  [{CAMERA_CAPTURE_WIDTH}x{CAMERA_CAPTURE_HEIGHT}, "
                  f"continuous AF, JPEG q={CAMERA_JPEG_QUALITY}]")
            return True

        except Exception as exc:
            print(f"Camera init failed: {exc}")
            print("   Running in audio-only mode.")
            return False

    def capture_to_memory(self) -> Optional[bytes]:
        """
        Grab the current preview frame and return it as JPEG bytes.

        Because the camera is in continuous-AF preview mode, there is no
        need to wait for a focus cycle — we just encode whatever the ISP
        is currently producing.

        Returns:
            JPEG bytes (in-memory), or None on failure.
        """
        if not self._initialized or self.camera is None:
            return None

        with self._lock:
            try:
                start = time.time()

                # capture_file() with a BytesIO target and format='jpeg'
                # encodes the *next* full frame from the preview pipeline
                # into JPEG directly — no intermediate numpy conversion needed.
                buf = io.BytesIO()
                self.camera.capture_file(buf, format="jpeg")
                jpeg_bytes = buf.getvalue()

                elapsed = time.time() - start
                kb = len(jpeg_bytes) / 1024
                print(f"   Camera: captured {kb:.0f} KB JPEG in {elapsed:.2f}s")

                return jpeg_bytes

            except Exception as exc:
                print(f"   Camera capture failed: {exc}")
                return None

    def cleanup(self):
        """Stop the camera pipeline and release resources."""
        if self.camera is not None:
            try:
                self.camera.stop()
                self.camera.close()
            except Exception:
                pass
            self.camera = None
        self._initialized = False


# =============================================================================
# WAKE WORD DETECTION (OPENWAKEWORD — FULLY LOCAL)
# =============================================================================

class WakeWordDetector:
    """
    Continuously feeds 80 ms audio chunks to openwakeword.
    No API keys, no cloud — runs entirely on-device.
    """

    def __init__(
        self,
        model_name:   str   = WAKE_WORD_MODEL,
        threshold:    float = WAKE_WORD_THRESHOLD,
        display_name: str   = WAKE_WORD_DISPLAY,
    ):
        self.model_name   = model_name
        self.threshold    = threshold
        self.display_name = display_name
        self.oww_model    = None

        # openwakeword expects 80 ms chunks at 16 kHz → 1280 samples
        self.chunk_size = 1280

    def initialize(self) -> bool:
        """Load the openwakeword model (auto-downloaded on first use)."""
        print(f"Loading wake word model '{self.model_name}'...")
        print("   (Models are downloaded automatically on first use)")

        try:
            openwakeword.utils.download_models()
            self.oww_model = OWWModel(
                wakeword_models=[self.model_name],
                inference_framework="onnx",
            )
            print(f"Wake word detector ready  "
                  f"(trigger: '{self.display_name}', "
                  f"threshold: {self.threshold})")
            return True

        except Exception as exc:
            print(f"Failed to init openwakeword: {exc}")
            return False

    def listen_for_wake_word(self) -> bool:
        """
        Block until the wake word is detected.

        Returns:
            True on detection, False on stream error.
        """
        if not self.oww_model:
            print("openwakeword model not initialised!")
            return False

        print(f"\nListening for '{self.display_name}'...")

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=self.chunk_size,
            ) as stream:
                while True:
                    audio_chunk, overflowed = stream.read(self.chunk_size)

                    if overflowed:
                        print("WARNING: Audio buffer overflow")

                    audio_data  = audio_chunk.flatten().astype(np.int16)
                    predictions = self.oww_model.predict(audio_data)

                    for name, score in predictions.items():
                        if score >= self.threshold:
                            print(f"\nWake word '{self.display_name}' detected! "
                                  f"(score: {score:.2f})")
                            self.oww_model.reset()
                            return True

        except Exception as exc:
            print(f"Wake word detection error: {exc}")
            return False

    def cleanup(self):
        self.oww_model = None


# =============================================================================
# AUDIO RECORDER WITH VAD
# =============================================================================

class AudioRecorder:
    """
    Records from the microphone until VAD silence is detected.

    IMPORTANT for parallel use: record() opens its own InputStream and
    starts capturing immediately when called. Launch it in a thread at
    the same moment you launch CameraCapture.capture_to_memory() so both
    run concurrently — the user's first syllable is never lost.
    """

    def __init__(
        self,
        sample_rate:       int   = SAMPLE_RATE,
        silence_threshold: int   = SILENCE_THRESHOLD,
        silence_duration:  float = SILENCE_DURATION,
        max_duration:      float = MAX_RECORDING_DURATION,
        min_duration:      float = MIN_RECORDING_DURATION,
    ):
        self.sample_rate       = sample_rate
        self.silence_threshold = silence_threshold
        self.silence_duration  = silence_duration
        self.max_duration      = max_duration
        self.min_duration      = min_duration

    def record(self) -> Optional[np.ndarray]:
        """
        Open the microphone and capture audio until silence is detected.

        Returns:
            int16 NumPy array of the full recording, or None on error.
        """
        print("\nRecording... (speak now, pause when done)")

        audio_chunks   = []
        start_time     = time.time()
        last_speech_at = start_time

        # 100 ms chunks give a responsive silence-detection loop
        chunk_samples = int(self.sample_rate * 0.1)

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=CHANNELS,
                dtype=DTYPE,
                blocksize=chunk_samples,
            ) as stream:
                while True:
                    chunk, _overflow = stream.read(chunk_samples)
                    audio_chunks.append(chunk.copy())

                    now     = time.time()
                    elapsed = now - start_time

                    if elapsed >= self.max_duration:
                        print(f"\nMax recording duration ({self.max_duration}s) reached")
                        break

                    if elapsed >= self.min_duration:
                        if not is_silence(chunk, self.silence_threshold):
                            last_speech_at = now
                        elif (now - last_speech_at) >= self.silence_duration:
                            print(f"\nSilence detected ({self.silence_duration}s)")
                            break

                    # Simple level meter — visual feedback
                    level = int(np.abs(chunk).mean() / 100)
                    print(f"\r  [{elapsed:.1f}s] {'|' * min(level, 40):<40}",
                          end="", flush=True)

            recording = np.concatenate(audio_chunks)
            duration  = len(recording) / self.sample_rate
            print(f"\nRecorded {duration:.1f}s of audio")
            return recording

        except Exception as exc:
            print(f"\nRecording error: {exc}")
            return None


# =============================================================================
# SPEECH-TO-TEXT (FASTER-WHISPER — LOCAL)
# =============================================================================

class SpeechToText:
    """
    Transcribes audio entirely on-device using faster-whisper.
    Uses int8 quantisation for CPU efficiency on the Pi 5.
    """

    def __init__(self, model_name: str = WHISPER_MODEL):
        self.model_name = model_name
        self.model      = None

    def initialize(self) -> bool:
        print(f"Loading Whisper model '{self.model_name}'...")
        print("   (First run downloads the model weights)")

        try:
            self.model = WhisperModel(
                self.model_name,
                device="cpu",
                compute_type="int8",   # Quantised → fast on ARM Cortex-A76
            )
            print(f"Whisper '{self.model_name}' loaded")
            return True

        except Exception as exc:
            print(f"Failed to load Whisper: {exc}")
            return False

    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe int16 audio to text.

        Returns:
            Transcribed string, or empty string on failure.
        """
        if not self.model:
            return ""

        print("\nTranscribing...")
        t0 = time.time()

        try:
            # faster-whisper expects float32 in [-1, 1]
            audio_f32 = audio.flatten().astype(np.float32) / 32768.0

            segments, _info = self.model.transcribe(
                audio_f32,
                language="en",
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )

            text = " ".join(seg.text for seg in segments).strip()

            print(f"Transcription done ({time.time() - t0:.2f}s)")
            print(f"   You said: \"{text}\"")
            return text

        except Exception as exc:
            print(f"Transcription error: {exc}")
            return ""


# =============================================================================
# LLM INTEGRATION (GOOGLE GEMINI — MULTIMODAL)
# =============================================================================

class GeminiLLM:
    """
    Streaming multimodal LLM via Google Gemini.

    stream_response() accepts optional JPEG image bytes alongside the text
    prompt. When an image is present, both are packed into a single
    multimodal Content message so Gemini can reason about what it sees.

    Conversation history is maintained across turns for context, but images
    are NOT stored in history — they are only sent with the current turn to
    keep request sizes manageable.
    """

    def __init__(self, api_key: str, model: str = GEMINI_MODEL):
        self.api_key  = api_key
        self.model    = model
        self.client   = None
        # Stores text-only turns {role, parts:[{text}]} for multi-turn context
        self.conversation_history: list[dict] = []

    def initialize(self) -> bool:
        try:
            self.client = genai.Client(api_key=self.api_key)
            print(f"Gemini client ready  (model: {self.model})")
            return True
        except Exception as exc:
            print(f"Failed to init Gemini: {exc}")
            return False

    def stream_response(
        self,
        prompt:      str,
        image_bytes: Optional[bytes] = None,
    ) -> Generator[str, None, None]:
        """
        Stream a response from Gemini, optionally with a camera image.

        The current turn is built as a multimodal Content object:
          [ImagePart (if provided), TextPart]

        Previous turns are included as text-only history so the model
        retains conversational context without bloating the request.

        Args:
            prompt:      Transcribed user speech.
            image_bytes: In-memory JPEG bytes from the camera (optional).

        Yields:
            Text chunks as they stream back from the API.
        """
        if not self.client:
            print("Gemini client not initialised!")
            return

        has_image = image_bytes is not None and len(image_bytes) > 0
        mode_tag  = "vision + text" if has_image else "text-only"
        print(f"\nGenerating response  [{mode_tag}]...")

        try:
            # --- Build the current-turn parts --------------------------------
            # We always put the image FIRST (Gemini attends to leading parts
            # more reliably) and the text question second.
            current_parts: list[genai_types.Part] = []

            if has_image:
                # Inline the JPEG bytes as a Blob — no file upload needed
                current_parts.append(
                    genai_types.Part(
                        inline_data=genai_types.Blob(
                            mime_type="image/jpeg",
                            data=image_bytes,
                        )
                    )
                )

            current_parts.append(genai_types.Part(text=prompt))

            # --- Assemble full contents list --------------------------------
            # History contains previous text-only turns (no images).
            # We append the current multimodal turn at the end.
            contents = list(self.conversation_history) + [
                genai_types.Content(role="user", parts=current_parts)
            ]

            # --- Stream from Gemini -----------------------------------------
            response = self.client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.7,
                    max_output_tokens=1024,
                ),
            )

            full_response = ""
            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
                    yield chunk.text

            # --- Update history (text-only) ---------------------------------
            # Store the current turn as text in history so future turns have
            # conversational context without re-sending the image.
            self.conversation_history.append({
                "role":  "user",
                "parts": [{"text": prompt}],
            })
            self.conversation_history.append({
                "role":  "model",
                "parts": [{"text": full_response}],
            })

            print(f"\n   Assistant: {full_response}")

        except Exception as exc:
            print(f"Gemini error: {exc}")
            yield "I'm sorry, I encountered an error processing your request."

    def clear_history(self):
        """Reset the conversation context."""
        self.conversation_history = []
        print("Conversation history cleared")


# =============================================================================
# TEXT-TO-SPEECH (ELEVENLABS — STREAMING)
# =============================================================================

class TextToSpeech:
    """
    Streaming TTS via ElevenLabs with sentence-level buffering.

    speak_streaming() runs a background worker thread that converts
    complete sentences to audio as fast as the LLM can produce them,
    so the first audio chunk plays while Gemini is still generating.
    """

    def __init__(self, api_key: str, voice_id: str = ELEVENLABS_VOICE_ID):
        self.api_key  = api_key
        self.voice_id = voice_id
        self.client   = None

    def initialize(self) -> bool:
        try:
            self.client = ElevenLabs(api_key=self.api_key)
            print(f"ElevenLabs client ready  (voice: {self.voice_id})")
            return True
        except Exception as exc:
            print(f"Failed to init ElevenLabs: {exc}")
            return False

    def speak_streaming(self, text_generator: Generator[str, None, None]):
        """
        Buffer LLM text chunks into sentences and play them via ElevenLabs.

        Architecture:
          Main thread  → collects text chunks → enqueues complete sentences
          Worker thread → dequeues sentences   → streams TTS → plays audio

        This pipeline means audio starts playing as soon as the FIRST
        sentence is ready, not when the full response is complete.
        """
        if not self.client:
            print("ElevenLabs client not initialised!")
            return

        print("\nSpeaking...")

        text_buffer        = ""
        sentence_queue     = queue.Queue()
        streaming_complete = threading.Event()

        def tts_worker():
            """Dequeue sentences and play them sequentially."""
            while True:
                try:
                    sentence = sentence_queue.get(timeout=0.1)

                    if sentence is None:           # poison pill — exit
                        break

                    audio_stream = self.client.text_to_speech.stream(
                        voice_id=self.voice_id,
                        text=sentence,
                        model_id="eleven_turbo_v2_5",   # lowest latency model
                        output_format="pcm_16000",       # matches SAMPLE_RATE
                    )
                    self._play_audio_stream(audio_stream)
                    sentence_queue.task_done()

                except queue.Empty:
                    if streaming_complete.is_set() and sentence_queue.empty():
                        break
                except Exception as exc:
                    print(f"TTS worker error: {exc}")
                    sentence_queue.task_done()

        tts_thread = threading.Thread(target=tts_worker, daemon=True)
        tts_thread.start()

        try:
            for chunk in text_generator:
                text_buffer += chunk
                sentences, text_buffer = parse_sentences(text_buffer)
                for sentence in sentences:
                    if sentence.strip():
                        sentence_queue.put(sentence.strip())

            # Flush any trailing text that didn't end with punctuation
            if text_buffer.strip():
                sentence_queue.put(text_buffer.strip())

            streaming_complete.set()
            tts_thread.join(timeout=60)

        except Exception as exc:
            print(f"TTS streaming error: {exc}")
            streaming_complete.set()

    def _play_audio_stream(self, audio_stream):
        """Collect PCM chunks from ElevenLabs and play via sounddevice."""
        audio_data = b"".join(audio_stream)
        if audio_data:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            sd.play(audio_array, samplerate=16000)
            sd.wait()

    def speak(self, text: str):
        """Non-streaming TTS — useful for short status phrases."""
        if not self.client:
            return
        try:
            audio    = self.client.text_to_speech.convert(
                voice_id=self.voice_id,
                text=text,
                model_id="eleven_turbo_v2_5",
                output_format="pcm_16000",
            )
            audio_data  = b"".join(audio)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            sd.play(audio_array, samplerate=16000)
            sd.wait()
        except Exception as exc:
            print(f"TTS error: {exc}")


# =============================================================================
# MAIN VOICE ASSISTANT
# =============================================================================

class VoiceAssistant:
    """
    Orchestrates all pipeline components.

    Execution flow per interaction:
      1. Wake word detected
      2. PARALLEL: AudioRecorder.record() + CameraCapture.capture_to_memory()
         — both launched in threads at the same instant so no audio is lost
           while the camera initialises (camera is already warm).
      3. faster-whisper transcribes the audio
      4. Gemini receives text + JPEG image (or text-only if camera disabled)
      5. ElevenLabs TTS plays the response as it streams
    """

    def __init__(self):
        self.wake_detector = WakeWordDetector(
            model_name=WAKE_WORD_MODEL,
            threshold=WAKE_WORD_THRESHOLD,
            display_name=WAKE_WORD_DISPLAY,
        )
        self.recorder = AudioRecorder()
        self.stt      = SpeechToText(WHISPER_MODEL)
        self.llm      = GeminiLLM(GEMINI_API_KEY, GEMINI_MODEL)
        self.tts      = TextToSpeech(ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID)
        self.camera   = CameraCapture() if CAMERA_ENABLED else None

    def initialize(self) -> bool:
        """
        Initialize all components in order.
        Camera is initialized last — it has the longest warm-up but is
        non-fatal (we fall back to audio-only if it fails).
        """
        print("\n" + "=" * 62)
        print("  INITIALIZING MULTIMODAL VOICE ASSISTANT")
        print("=" * 62 + "\n")

        get_audio_devices()

        if not self.wake_detector.initialize():
            return False

        if not self.stt.initialize():
            return False

        if not self.llm.initialize():
            return False

        if not self.tts.initialize():
            return False

        # Camera is optional — failure here is a warning, not a fatal error
        if self.camera is not None:
            camera_ok = self.camera.initialize()
            if not camera_ok:
                print("Continuing without camera (audio-only mode).")
                self.camera = None

        print("\n" + "=" * 62)
        mode = "VISION + AUDIO" if self.camera else "AUDIO-ONLY"
        print(f"  ALL COMPONENTS READY  [{mode}]")
        print("=" * 62 + "\n")
        return True

    def _capture_parallel(self) -> tuple[Optional[np.ndarray], Optional[bytes]]:
        """
        Launch audio recording and camera capture simultaneously.

        Why this matters for latency:
          - Audio recording MUST start first (or at the same time) so the
            user's opening words are captured.
          - Camera capture is fast (~0.1 s) because the camera is already
            in continuous-AF preview mode. It typically finishes long before
            the user finishes speaking.
          - Running them in separate threads means neither blocks the other.

        Returns:
            (audio_array, jpeg_bytes) — either may be None on failure.
        """
        audio_result: list[Optional[np.ndarray]] = [None]
        image_result: list[Optional[bytes]]       = [None]

        def record_fn():
            audio_result[0] = self.recorder.record()

        def capture_fn():
            if self.camera is not None:
                image_result[0] = self.camera.capture_to_memory()

        # Start audio thread FIRST — we want mic capture to begin
        # before the camera thread even allocates its buffer.
        audio_thread  = threading.Thread(target=record_fn,  daemon=True)
        camera_thread = threading.Thread(target=capture_fn, daemon=True)

        audio_thread.start()
        camera_thread.start()

        # Wait for both to finish — audio is the long pole here
        audio_thread.join()
        camera_thread.join()

        return audio_result[0], image_result[0]

    def run(self):
        """Main loop."""
        print("Voice Assistant is ready!")
        print(f"   Say '{WAKE_WORD_DISPLAY}' to activate...")
        print("   Press Ctrl+C to exit\n")

        try:
            while True:
                # ── Step 1: Wait for wake word ─────────────────────────────
                if not self.wake_detector.listen_for_wake_word():
                    print("Wake word detection failed — retrying...")
                    time.sleep(1)
                    continue

                # ── Step 2: Record audio + capture image IN PARALLEL ───────
                # Both threads start at (virtually) the same time.
                # Audio recording will catch the user's first words because
                # the camera is already warm and just grabs the current frame.
                print("\nCapturing audio and image simultaneously...")
                audio, image_bytes = self._capture_parallel()

                if audio is None or len(audio) < SAMPLE_RATE * 0.5:
                    print("Recording too short or failed — ready again.")
                    continue

                # ── Step 3: Transcribe ─────────────────────────────────────
                text = self.stt.transcribe(audio)
                if not text.strip():
                    print("No speech detected.")
                    self.tts.speak("I didn't catch that, Sir. Please try again.")
                    continue

                if image_bytes:
                    kb = len(image_bytes) / 1024
                    print(f"   Image ready: {kb:.0f} KB  — sending multimodal request")
                else:
                    print("   No image captured — sending text-only request")

                # ── Step 4 & 5: Multimodal LLM → streaming TTS ────────────
                # stream_response yields text tokens; speak_streaming pipes
                # them into ElevenLabs as fast as complete sentences form,
                # so audio playback begins before Gemini finishes generating.
                response_gen = self.llm.stream_response(
                    prompt=text,
                    image_bytes=image_bytes,
                )
                self.tts.speak_streaming(response_gen)

                print("\n" + "-" * 42)
                print(f"   Say '{WAKE_WORD_DISPLAY}' for another question...")
                print("-" * 42 + "\n")

        except KeyboardInterrupt:
            print("\n\nShutting down...")
        finally:
            self.cleanup()

    def cleanup(self):
        """Release all hardware resources gracefully."""
        self.wake_detector.cleanup()
        if self.camera is not None:
            self.camera.cleanup()
        print("Goodbye!")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
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

    if not GEMINI_API_KEY or "YOUR_" in GEMINI_API_KEY:
        print("ERROR: Set GEMINI_API_KEY in .env.local")
        print("       https://aistudio.google.com/apikey")
        sys.exit(1)

    if not ELEVENLABS_API_KEY or "YOUR_" in ELEVENLABS_API_KEY:
        print("ERROR: Set ELEVENLABS_API_KEY in .env.local")
        print("       https://elevenlabs.io/app/settings/api-keys")
        sys.exit(1)

    assistant = VoiceAssistant()

    if assistant.initialize():
        assistant.run()
    else:
        print("\nFailed to initialize voice assistant")
        sys.exit(1)


if __name__ == "__main__":
    main()
