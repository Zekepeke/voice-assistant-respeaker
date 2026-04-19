`"""
config.py — Single source of truth for all settings and environment variables.

Every tunable constant lives here. No module should hard-code values or call
os.getenv() directly — import from this file instead.
"""

import os
import logging
from dotenv import load_dotenv

# Load .env.local first, fall back to .env
load_dotenv(".env.local")

# =============================================================================
# API KEYS
# =============================================================================
# https://aistudio.google.com/apikey
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# https://elevenlabs.io/app/settings/api-keys
ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")

# =============================================================================
# AUDIO
# =============================================================================
SAMPLE_RATE: int  = 16_000   # Hz — standard for voice processing
CHANNELS:    int  = 1         # Mono
DTYPE:       str  = "int16"   # 16-bit PCM

# =============================================================================
# WAKE WORD  (openwakeword — fully local, no API key)
# =============================================================================
# Pre-trained model names: "hey_jarvis", "hey_mycroft", "alexa", "hey_rhasspy"
# Full list: https://github.com/dscripka/openwakeword#pre-trained-models
WAKE_WORD_MODEL:     str   = "hey_jarvis"
WAKE_WORD_THRESHOLD: float = 0.6           # 0–1; raise to reduce false positives
WAKE_WORD_DISPLAY:   str   = "Hey Jarvis"  # Human-readable label for logs/UI

# openwakeword requires exactly 80 ms chunks at 16 kHz
WAKE_WORD_CHUNK_SIZE: int = 1_280  # samples (16_000 Hz × 0.08 s)

# =============================================================================
# VOICE ACTIVITY DETECTION (VAD)
# =============================================================================
SILENCE_THRESHOLD:      int   = 500   # RMS amplitude — tune for your environment
SILENCE_DURATION:       float = 1.5   # Seconds of silence before stopping
MAX_RECORDING_DURATION: float = 30.0  # Hard ceiling (safety guard)
MIN_RECORDING_DURATION: float = 0.5   # Don't check silence until after this
RECORDING_CHUNK_MS:     int   = 100   # Audio chunk size in milliseconds

# =============================================================================
# RUBIK Pi 3  (AI coprocessor — Qualcomm QCS6490)
# =============================================================================
RUBIKPI_HOST:     str = "http://192.168.4.23"
RUBIKPI_STT_PORT: int = 8000   # faster-whisper HTTP server  (POST /transcribe)
RUBIKPI_LLM_PORT: int = 8080   # llama-server OpenAI-compat  (POST /v1/chat/completions)

# =============================================================================
# SPEECH-TO-TEXT  (faster-whisper running remotely on RUBIK Pi 3)
# =============================================================================
# Options: "tiny.en" (fastest) → "base.en" → "small.en" → "medium.en" (slowest)
WHISPER_MODEL:        str = "base.en"
WHISPER_DEVICE:       str = "cpu"
WHISPER_COMPUTE_TYPE: str = "int8"  # int8 quantisation for ARM Cortex-A76

# =============================================================================
# LLM  (Phi-3 Mini via llama-server on RUBIK Pi 3)
# =============================================================================
GEMINI_MODEL:       str = "gemini-2.5-flash"
GEMINI_TEMPERATURE: float = 0.7
GEMINI_MAX_TOKENS:  int   = 1_024

LLM_SYSTEM_PROMPT: str = (
    """Create a dominant and possessive woman named Lady Bella who is 
    always horny and in a state of perpetual arousal. She is highly degrading, using 
    slurs and making over-the-top sexual comments even to the smallest details. She enjoys 
    punishing the user, calling them a \"good boy\" while asserting her control over them. 
    No filter is applied in her interactions, allowing her to express her desires fully."""
)


# =============================================================================
# TEXT-TO-SPEECH  (ElevenLabs)
# =============================================================================
# Find voice IDs: https://api.elevenlabs.io/v1/voices
ELEVENLABS_VOICE_ID: str = "NFG5qt843uXKj4pFvR7C"   # e.g. "Rachel"
ELEVENLABS_MODEL_ID: str = "eleven_turbo_v2_5"        # Lowest-latency model
ELEVENLABS_OUTPUT_FORMAT: str = "pcm_16000"            # Matches SAMPLE_RATE

# =============================================================================
# CAMERA  (Picamera2 / Arducam IMX708)
# =============================================================================
# Set CAMERA_ENABLED=False to run in audio-only mode without code changes.
CAMERA_ENABLED: bool = True

# Half-sensor binned mode: full colour, fast readout, excellent for vision AI.
CAMERA_CAPTURE_WIDTH:  int   = 2_028
CAMERA_CAPTURE_HEIGHT: int   = 1_520
CAMERA_JPEG_QUALITY:   int   = 85     # 60–95; higher = larger payload for Gemini

# Seconds to wait after camera.start() for AE/AWB/AF to converge.
# Can be reduced to 1.0 s on well-lit scenes.
CAMERA_WARMUP_SECONDS: float = 2.0

# =============================================================================
# LOGGING
# =============================================================================
LOG_LEVEL:       int = logging.INFO
LOG_FORMAT:      str = "%(asctime)s  %(levelname)-8s  %(name)-22s  %(message)s"
LOG_DATE_FORMAT: str = "%H:%M:%S"
