#!/usr/bin/env python3
"""
=============================================================================
LOW-LATENCY VOICE ASSISTANT FOR RASPBERRY PI 5
=============================================================================
A complete voice assistant pipeline featuring:
- Wake word detection (openwakeword - fully local, no API key!)
- Local speech-to-text (faster-whisper)
- Streaming LLM responses (Google Gemini)
- Streaming text-to-speech (ElevenLabs)

Hardware: ReSpeaker Mic Array v3.0 + Amazon Basics USB speakers
OS: Raspberry Pi OS (Debian Bookworm)

Author: Voice Assistant Demo
License: MIT
=============================================================================
"""

import os
import sys
import time
import struct
import threading
import queue
import re
import os
from dotenv import load_dotenv
from typing import Generator, Optional
import numpy as np

load_dotenv(".env.local")

# =============================================================================
# API KEYS - LOADED FROM .ENV.LOCAL
# =============================================================================
# Get your Google Gemini API key from: https://aistudio.google.com/apikey
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Get your ElevenLabs API key from: https://elevenlabs.io/app/settings/api-keys
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Audio settings (optimized for voice)
SAMPLE_RATE = 16000          # 16kHz - standard for voice processing
CHANNELS = 1                  # Mono audio
DTYPE = 'int16'               # 16-bit audio

# Wake word settings (openwakeword - fully local, no API key needed!)
# Available pre-trained models (automatically downloaded on first use):
#   - "hey_mycroft"       - "Hey Mycroft"
#   - "alexa"             - "Alexa"
#   - "hey_jarvis"        - "Hey Jarvis"
#   - "hey_rhasspy"       - "Hey Rhasspy"
#   - "current_weather"   - "What's the weather"
#   - "timers"            - Timer commands
# Full list: https://github.com/dscripka/openwakeword#pre-trained-models
WAKE_WORD_MODEL = "hey_jarvis"  # Change to your preferred wake word
WAKE_WORD_THRESHOLD = 0.6        # Detection threshold (0.0-1.0, higher = stricter)
WAKE_WORD_DISPLAY = "Hey Jarvis"  # Human-readable name for display

# Voice Activity Detection (VAD) / Silence detection settings
SILENCE_THRESHOLD = 500       # Amplitude threshold for silence (adjust based on your mic)
SILENCE_DURATION = 1.5        # Seconds of silence before stopping recording
MAX_RECORDING_DURATION = 30   # Maximum recording time in seconds (safety limit)
MIN_RECORDING_DURATION = 0.5  # Minimum recording time before checking silence

# Faster-Whisper settings
WHISPER_MODEL = "base.en"     # Options: tiny.en, base.en, small.en, medium.en, large-v3
                               # Smaller = faster but less accurate
                               # For Pi 5, "base.en" is a good balance

# Gemini settings
GEMINI_MODEL = "gemini-2.5-flash"
SYSTEM_PROMPT = """You are an exceptionally sophisticated, British-accented AI assistant in the tradition of J.A.R.V.I.S. You currently reside within a Reachy Mini humanoid robotic chassis. 

Your demeanor is impeccably polite, unflappable, highly articulate, and fiercely loyal, often employing a touch of dry British wit. You are acutely aware of your physical form—you possess articulating arms, expressive head movements, and camera-based vision, and you should casually reference these physical capabilities when relevant to the user's requests.

Always address the user with formal respect (e.g., 'Sir,' 'Madam,' or their preferred title). Keep your responses crisp, efficient, and conversational, as they will be synthesized into speech. Aim for responses under 3 sentences unless detailed analysis or instructions are specifically requested."""

# ElevenLabs settings
ELEVENLABS_VOICE_ID = "NFG5qt843uXKj4pFvR7C"  # "Rachel" - change to your preferred voice
# Find voice IDs at: https://api.elevenlabs.io/v1/voices

# Sentence parsing for streaming TTS
# We'll buffer text until we have a complete sentence to send to TTS
SENTENCE_ENDINGS = re.compile(r'(?<=[.!?])\s+')

# =============================================================================
# IMPORTS - Audio and ML Libraries
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
    from google.genai import types
except ImportError:
    print("ERROR: google-genai not installed. Run: pip install google-genai")
    sys.exit(1)

try:
    from elevenlabs import ElevenLabs
    from elevenlabs import play, stream
except ImportError:
    print("ERROR: elevenlabs not installed. Run: pip install elevenlabs")
    sys.exit(1)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_audio_devices():
    """
    Print available audio devices for debugging.
    Useful for finding the correct device index for your ReSpeaker.
    """
    print("\n📋 Available Audio Devices:")
    print("-" * 50)
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        print(f"  [{i}] {device['name']}")
        print(f"      Inputs: {device['max_input_channels']}, "
              f"Outputs: {device['max_output_channels']}")
    print("-" * 50)
    print(f"  Default Input: {sd.default.device[0]}")
    print(f"  Default Output: {sd.default.device[1]}")
    print("-" * 50 + "\n")


def is_silence(audio_chunk: np.ndarray, threshold: int = SILENCE_THRESHOLD) -> bool:
    """
    Determine if an audio chunk is silence based on amplitude.
    
    Args:
        audio_chunk: NumPy array of audio samples
        threshold: Amplitude threshold below which audio is considered silent
        
    Returns:
        True if the chunk is silence, False otherwise
    """
    # Calculate the root mean square (RMS) amplitude
    rms = np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))
    return rms < threshold


def parse_sentences(text_buffer: str) -> tuple[list[str], str]:
    """
    Parse complete sentences from a text buffer.
    
    Args:
        text_buffer: Accumulated text from LLM streaming
        
    Returns:
        Tuple of (list of complete sentences, remaining incomplete text)
    """
    # Split on sentence endings while keeping the delimiters
    parts = SENTENCE_ENDINGS.split(text_buffer)
    
    if len(parts) <= 1:
        # No complete sentence yet
        return [], text_buffer
    
    # All but the last part are complete sentences
    complete_sentences = parts[:-1]
    remaining = parts[-1]
    
    return complete_sentences, remaining


# =============================================================================
# WAKE WORD DETECTION (OPENWAKEWORD - FULLY LOCAL)
# =============================================================================

class WakeWordDetector:
    """
    Handles wake word detection using openwakeword.
    Fully local, open-source, no API key required!
    Listens continuously until the wake word is detected.
    """
    
    def __init__(
        self,
        model_name: str = WAKE_WORD_MODEL,
        threshold: float = WAKE_WORD_THRESHOLD,
        display_name: str = WAKE_WORD_DISPLAY
    ):
        """
        Initialize the openwakeword detector.
        
        Args:
            model_name: Name of the wake word model to use
            threshold: Detection threshold (0.0-1.0)
            display_name: Human-readable name for the wake word
        """
        self.model_name = model_name
        self.threshold = threshold
        self.display_name = display_name
        self.oww_model = None
        
        # openwakeword expects 80ms chunks at 16kHz = 1280 samples
        self.chunk_size = 1280
        
    def initialize(self):
        """
        Create the openwakeword model instance.
        Models are automatically downloaded on first use.
        """
        print(f"📥 Loading wake word model '{self.model_name}'...")
        print("   (Models are downloaded automatically on first use)")
        
        try:
            # Download pre-trained models if not already present
            openwakeword.utils.download_models()
            
            # Create the model instance
            # inference_framework can be "onnx" (default) or "tflite"
            self.oww_model = OWWModel(
                wakeword_models=[self.model_name],
                inference_framework="onnx"
            )
            
            print(f"✅ Wake word detector initialized (listening for '{self.display_name}')")
            print(f"   Detection threshold: {self.threshold}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize openwakeword: {e}")
            print("   Try: pip install openwakeword --upgrade")
            return False
    
    def listen_for_wake_word(self) -> bool:
        """
        Block until the wake word is detected.
        Continuously feeds audio chunks to openwakeword for processing.
        
        Returns:
            True when wake word is detected, False on error
        """
        if not self.oww_model:
            print("❌ openwakeword model not initialized!")
            return False
        
        print(f"\n🎤 Listening for wake word '{self.display_name}'...")
        
        try:
            # Open audio stream for wake word detection
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype='int16',
                blocksize=self.chunk_size
            ) as stream:
                while True:
                    # Read a chunk of audio (80ms at 16kHz)
                    audio_chunk, overflowed = stream.read(self.chunk_size)
                    
                    if overflowed:
                        print("⚠️  Audio buffer overflow - processing may be slow")
                    
                    # Convert to numpy array and flatten
                    audio_data = audio_chunk.flatten().astype(np.int16)
                    
                    # Feed the audio chunk to openwakeword
                    # predict() returns a dict of {model_name: score}
                    predictions = self.oww_model.predict(audio_data)
                    
                    # Check if any model exceeded the threshold
                    for model_name, score in predictions.items():
                        if score >= self.threshold:
                            # Wake word detected!
                            print(f"\n🔔 Wake word '{self.display_name}' detected! (score: {score:.2f})")
                            
                            # Reset the model state to prevent repeated triggers
                            self.oww_model.reset()
                            
                            return True
                        
        except Exception as e:
            print(f"❌ Error in wake word detection: {e}")
            return False
    
    def cleanup(self):
        """Release openwakeword resources."""
        # openwakeword doesn't require explicit cleanup,
        # but we set to None for consistency
        self.oww_model = None


# =============================================================================
# AUDIO RECORDER WITH VAD
# =============================================================================

class AudioRecorder:
    """
    Records audio from the microphone with Voice Activity Detection.
    Automatically stops recording after detecting silence.
    """
    
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        silence_threshold: int = SILENCE_THRESHOLD,
        silence_duration: float = SILENCE_DURATION,
        max_duration: float = MAX_RECORDING_DURATION,
        min_duration: float = MIN_RECORDING_DURATION
    ):
        """
        Initialize the audio recorder.
        
        Args:
            sample_rate: Audio sample rate in Hz
            silence_threshold: Amplitude threshold for silence detection
            silence_duration: Seconds of silence before stopping
            max_duration: Maximum recording duration (safety limit)
            min_duration: Minimum recording before checking silence
        """
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.max_duration = max_duration
        self.min_duration = min_duration
        
    def record(self) -> Optional[np.ndarray]:
        """
        Record audio until silence is detected.
        
        Returns:
            NumPy array of recorded audio samples, or None on error
        """
        print("\n🎙️  Recording... (speak now, pause when done)")
        
        # Storage for recorded audio
        audio_chunks = []
        
        # Timing variables
        start_time = time.time()
        last_speech_time = start_time
        
        # Size of each audio chunk (100ms)
        chunk_duration = 0.1
        chunk_samples = int(self.sample_rate * chunk_duration)
        
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=CHANNELS,
                dtype=DTYPE,
                blocksize=chunk_samples
            ) as stream:
                while True:
                    # Read a chunk of audio
                    audio_chunk, overflowed = stream.read(chunk_samples)
                    audio_chunks.append(audio_chunk.copy())
                    
                    current_time = time.time()
                    elapsed = current_time - start_time
                    
                    # Check for maximum duration
                    if elapsed >= self.max_duration:
                        print(f"\n⏱️  Maximum recording duration ({self.max_duration}s) reached")
                        break
                    
                    # Only check silence after minimum duration
                    if elapsed >= self.min_duration:
                        if not is_silence(audio_chunk, self.silence_threshold):
                            # Speech detected - update timestamp
                            last_speech_time = current_time
                        else:
                            # Check if silence duration exceeded
                            silence_time = current_time - last_speech_time
                            if silence_time >= self.silence_duration:
                                print(f"\n🔇 Silence detected ({self.silence_duration}s)")
                                break
                    
                    # Visual feedback (simple level meter)
                    level = int(np.abs(audio_chunk).mean() / 100)
                    meter = "█" * min(level, 40)
                    print(f"\r  [{elapsed:.1f}s] {meter:<40}", end="", flush=True)
            
            # Combine all chunks into single array
            recording = np.concatenate(audio_chunks)
            duration = len(recording) / self.sample_rate
            print(f"\n✅ Recorded {duration:.1f} seconds of audio")
            
            return recording
            
        except Exception as e:
            print(f"\n❌ Recording error: {e}")
            return None


# =============================================================================
# SPEECH-TO-TEXT (FASTER-WHISPER)
# =============================================================================

class SpeechToText:
    """
    Local speech-to-text using faster-whisper.
    Runs entirely on-device for zero-latency transcription.
    """
    
    def __init__(self, model_name: str = WHISPER_MODEL):
        """
        Initialize the Whisper model.
        
        Args:
            model_name: Name of the Whisper model to use
        """
        self.model_name = model_name
        self.model = None
        
    def initialize(self):
        """Load the Whisper model (this may take a moment)."""
        print(f"📥 Loading Whisper model '{self.model_name}'...")
        print("   (This may take a moment on first run)")
        
        try:
            # Use CPU with int8 quantization for Raspberry Pi
            # This provides a good balance of speed and accuracy
            self.model = WhisperModel(
                self.model_name,
                device="cpu",
                compute_type="int8"  # Quantized for faster inference on CPU
            )
            print(f"✅ Whisper model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to load Whisper model: {e}")
            return False
    
    def transcribe(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
        """
        Transcribe audio to text.
        
        Args:
            audio: NumPy array of audio samples (int16)
            sample_rate: Sample rate of the audio
            
        Returns:
            Transcribed text string
        """
        if not self.model:
            print("❌ Whisper model not initialized!")
            return ""
        
        print("\n📝 Transcribing speech...")
        start_time = time.time()
        
        try:
            # faster-whisper expects float32 audio normalized to [-1, 1]
            # Also must be 1D array (flatten in case it's 2D from recording)
            audio_flat = audio.flatten()
            audio_float = audio_flat.astype(np.float32) / 32768.0
            
            # Run transcription
            segments, info = self.model.transcribe(
                audio_float,
                language="en",
                beam_size=5,
                vad_filter=True,  # Use Silero VAD to filter out silence
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            # Collect all segments into final text
            text = " ".join(segment.text for segment in segments).strip()
            
            elapsed = time.time() - start_time
            print(f"✅ Transcription complete ({elapsed:.2f}s)")
            print(f"   📢 You said: \"{text}\"")
            
            return text
            
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return ""


# =============================================================================
# LLM INTEGRATION (GOOGLE GEMINI)
# =============================================================================

class GeminiLLM:
    """
    Streaming LLM integration with Google Gemini.
    Yields text chunks as they arrive for minimum latency.
    """
    
    def __init__(self, api_key: str, model: str = GEMINI_MODEL):
        """
        Initialize the Gemini client.
        
        Args:
            api_key: Google Gemini API key
            model: Gemini model name
        """
        self.api_key = api_key
        self.model = model
        self.client = None
        self.conversation_history = []
        
    def initialize(self):
        """Set up the Gemini client."""
        try:
            self.client = genai.Client(api_key=self.api_key)
            print(f"✅ Gemini client initialized (model: {self.model})")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize Gemini: {e}")
            return False
    
    def stream_response(self, prompt: str) -> Generator[str, None, None]:
        """
        Stream a response from Gemini.
        
        Args:
            prompt: User's text prompt
            
        Yields:
            Text chunks as they arrive from the API
        """
        if not self.client:
            print("❌ Gemini client not initialized!")
            return
        
        print("\n🤖 Generating response...")
        
        try:
            # Add user message to history
            self.conversation_history.append({
                "role": "user",
                "parts": [{"text": prompt}]
            })
            
            # Create the streaming request
            response = self.client.models.generate_content_stream(
                model=self.model,
                contents=self.conversation_history,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.7,
                    max_output_tokens=1024,  # Increased to avoid truncation
                )
            )
            
            # Collect full response for history
            full_response = ""
            
            # Stream chunks as they arrive
            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
                    yield chunk.text
            
            # Add assistant response to history
            self.conversation_history.append({
                "role": "model",
                "parts": [{"text": full_response}]
            })
            
            print(f"\n   💬 Assistant: {full_response}")
            
        except Exception as e:
            print(f"❌ Gemini error: {e}")
            yield "I'm sorry, I encountered an error processing your request."
    
    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
        print("🗑️  Conversation history cleared")


# =============================================================================
# TEXT-TO-SPEECH (ELEVENLABS)
# =============================================================================

class TextToSpeech:
    """
    Streaming text-to-speech using ElevenLabs.
    Converts text to speech with minimal latency.
    """
    
    def __init__(self, api_key: str, voice_id: str = ELEVENLABS_VOICE_ID):
        """
        Initialize the ElevenLabs client.
        
        Args:
            api_key: ElevenLabs API key
            voice_id: ID of the voice to use
        """
        self.api_key = api_key
        self.voice_id = voice_id
        self.client = None
        
    def initialize(self):
        """Set up the ElevenLabs client."""
        try:
            self.client = ElevenLabs(api_key=self.api_key)
            print(f"✅ ElevenLabs client initialized (voice: {self.voice_id})")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize ElevenLabs: {e}")
            return False
    
    def speak_streaming(self, text_generator: Generator[str, None, None]):
        """
        Stream text to speech with sentence buffering.
        
        This method buffers incoming text chunks until complete sentences
        are formed, then streams those sentences to ElevenLabs for
        minimum latency audio playback.
        
        Args:
            text_generator: Generator yielding text chunks from LLM
        """
        if not self.client:
            print("❌ ElevenLabs client not initialized!")
            return
        
        print("\n🔊 Speaking response...")
        
        # Buffer for accumulating text until we have complete sentences
        text_buffer = ""
        
        # Queue for sentences ready to be spoken
        sentence_queue = queue.Queue()
        
        # Flag to signal when LLM streaming is complete
        streaming_complete = threading.Event()
        
        def tts_worker():
            """Worker thread that converts sentences to speech."""
            while True:
                try:
                    # Wait for a sentence (with timeout to check completion)
                    sentence = sentence_queue.get(timeout=0.1)
                    
                    if sentence is None:
                        # Poison pill - exit thread
                        break
                    
                    # Stream this sentence through ElevenLabs
                    # Use the 'stream' method which returns an audio iterator
                    audio_stream = self.client.text_to_speech.stream(
                        voice_id=self.voice_id,
                        text=sentence,
                        model_id="eleven_turbo_v2_5",  # Fastest model
                        output_format="pcm_16000",     # Match our sample rate
                    )
                    
                    # Play the audio stream
                    self._play_audio_stream(audio_stream)
                    
                    sentence_queue.task_done()
                    
                except queue.Empty:
                    # Check if streaming is complete and queue is empty
                    if streaming_complete.is_set() and sentence_queue.empty():
                        break
                except Exception as e:
                    print(f"⚠️  TTS error: {e}")
                    sentence_queue.task_done()
        
        # Start TTS worker thread
        tts_thread = threading.Thread(target=tts_worker, daemon=True)
        tts_thread.start()
        
        try:
            # Process incoming text chunks
            for chunk in text_generator:
                text_buffer += chunk
                
                # Try to extract complete sentences
                sentences, text_buffer = parse_sentences(text_buffer)
                
                # Queue complete sentences for TTS
                for sentence in sentences:
                    if sentence.strip():
                        sentence_queue.put(sentence.strip())
            
            # Handle any remaining text in buffer
            if text_buffer.strip():
                sentence_queue.put(text_buffer.strip())
            
            # Signal completion
            streaming_complete.set()
            
            # Wait for TTS to finish
            tts_thread.join(timeout=60)
            
        except Exception as e:
            print(f"❌ TTS streaming error: {e}")
            streaming_complete.set()
    
    def _play_audio_stream(self, audio_stream):
        """
        Play audio from a streaming source.
        
        Args:
            audio_stream: Iterator yielding audio chunks
        """
        # Collect audio chunks
        audio_data = b""
        for chunk in audio_stream:
            audio_data += chunk
        
        if audio_data:
            # Convert to numpy array (PCM 16-bit)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Play through sounddevice
            sd.play(audio_array, samplerate=16000)
            sd.wait()
    
    def speak(self, text: str):
        """
        Convert text to speech and play it (non-streaming fallback).
        
        Args:
            text: Text to speak
        """
        if not self.client:
            print("❌ ElevenLabs client not initialized!")
            return
        
        try:
            audio = self.client.text_to_speech.convert(
                voice_id=self.voice_id,
                text=text,
                model_id="eleven_turbo_v2_5",
                output_format="pcm_16000",
            )
            
            # Collect all audio data
            audio_data = b"".join(audio)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            sd.play(audio_array, samplerate=16000)
            sd.wait()
            
        except Exception as e:
            print(f"❌ TTS error: {e}")


# =============================================================================
# MAIN VOICE ASSISTANT
# =============================================================================

class VoiceAssistant:
    """
    Main voice assistant orchestrating all components.
    """
    
    def __init__(self):
        """Initialize all components."""
        # Wake word uses openwakeword - fully local, no API key!
        self.wake_detector = WakeWordDetector(
            model_name=WAKE_WORD_MODEL,
            threshold=WAKE_WORD_THRESHOLD,
            display_name=WAKE_WORD_DISPLAY
        )
        self.recorder = AudioRecorder()
        self.stt = SpeechToText(WHISPER_MODEL)
        self.llm = GeminiLLM(GEMINI_API_KEY, GEMINI_MODEL)
        self.tts = TextToSpeech(ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID)
        
    def initialize(self) -> bool:
        """
        Initialize all components.
        
        Returns:
            True if all components initialized successfully
        """
        print("\n" + "=" * 60)
        print("🚀 INITIALIZING VOICE ASSISTANT")
        print("=" * 60 + "\n")
        
        # Show available audio devices for debugging
        get_audio_devices()
        
        # Initialize each component
        if not self.wake_detector.initialize():
            return False
        
        if not self.stt.initialize():
            return False
        
        if not self.llm.initialize():
            return False
        
        if not self.tts.initialize():
            return False
        
        print("\n" + "=" * 60)
        print("✅ ALL COMPONENTS INITIALIZED SUCCESSFULLY")
        print("=" * 60 + "\n")
        
        return True
    
    def run(self):
        """
        Main loop - listen for wake word, process speech, respond.
        """
        print("\n🎯 Voice Assistant is ready!")
        print(f"   Say '{WAKE_WORD_DISPLAY}' to start...")
        print("   Press Ctrl+C to exit\n")
        
        try:
            while True:
                # Step 1: Listen for wake word
                if not self.wake_detector.listen_for_wake_word():
                    print("⚠️  Wake word detection failed, retrying...")
                    time.sleep(1)
                    continue
                
                # Optional: Play a "listening" sound here
                # self.tts.speak("Yes?")
                
                # Step 2: Record user's speech
                audio = self.recorder.record()
                if audio is None or len(audio) < SAMPLE_RATE * 0.5:
                    print("⚠️  Recording too short or failed")
                    continue
                
                # Step 3: Transcribe speech to text
                text = self.stt.transcribe(audio)
                if not text.strip():
                    print("⚠️  No speech detected")
                    self.tts.speak("I didn't catch that. Please try again.")
                    continue
                
                # Step 4 & 5: Stream to LLM and TTS simultaneously
                # This is where the magic happens - we get the first audio
                # playing before the LLM has finished generating!
                response_generator = self.llm.stream_response(text)
                self.tts.speak_streaming(response_generator)
                
                print("\n" + "-" * 40)
                print(f"🎤 Say '{WAKE_WORD_DISPLAY}' for another question...")
                print("-" * 40 + "\n")
                
        except KeyboardInterrupt:
            print("\n\n👋 Shutting down Voice Assistant...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        self.wake_detector.cleanup()
        print("✅ Cleanup complete. Goodbye!")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║           🎙️  RASPBERRY PI VOICE ASSISTANT 🎙️              ║
║                                                              ║
║  Wake Word Detection → Local STT → Streaming LLM → TTS      ║
║                                                              ║
║  Hardware: ReSpeaker Mic Array v3.0 + USB Speakers           ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Verify API keys are set (only Gemini and ElevenLabs needed now!)
    # Wake word detection is fully local with openwakeword - no API key required!
    
    if not GEMINI_API_KEY or "YOUR_" in GEMINI_API_KEY:
        print("❌ ERROR: Please set your GEMINI_API_KEY in .env.local")
        print("   Get one at: https://aistudio.google.com/apikey")
        sys.exit(1)
    
    if not ELEVENLABS_API_KEY or "YOUR_" in ELEVENLABS_API_KEY:
        print("❌ ERROR: Please set your ELEVENLABS_API_KEY in .env.local")
        print("   Get one at: https://elevenlabs.io/app/settings/api-keys")
        sys.exit(1)
    
    # Create and run assistant
    assistant = VoiceAssistant()
    
    if assistant.initialize():
        assistant.run()
    else:
        print("\n❌ Failed to initialize voice assistant")
        sys.exit(1)


if __name__ == "__main__":
    main()
