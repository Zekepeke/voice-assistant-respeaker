# Low-Latency Voice Assistant for Raspberry Pi 5

A complete voice assistant pipeline featuring:
- Wake word detection (openwakeword - fully local, no API key!)
- Local speech-to-text (faster-whisper)
- Streaming LLM responses (Google Gemini)
- Streaming text-to-speech (ElevenLabs)

## Architecture

### Hardware Context

*   **Audio Input/Output**: ReSpeaker Mic Array v3.0 connected via USB. The ReSpeaker handles Acoustic Echo Cancellation (AEC).
*   **Speaker**: Amazon Basics USB-powered speakers plugged directly into the ReSpeaker's 3.5mm audio jack.
*   **OS**: Raspberry Pi OS (Debian Bookworm) running inside a Python virtual environment.

### Tech Stack & Libraries

*   **Wake Word**: `openwakeword` for completely local, open-source wake word detection (no API keys required).
*   **Audio Capture/Playback**: `sounddevice`
*   **Speech-to-Text (STT)**: `faster-whisper` running entirely locally for zero-latency transcription.
*   **LLM**: `google-genai` using the `gemini-2.5-flash` model.
*   **Text-to-Speech (TTS)**: `elevenlabs` official Python SDK.

### Execution Flow (The Pipeline)

1.  **Listen**: The script runs continuously, feeding audio chunks from `sounddevice` into `openwakeword` until the wake word is detected.
2.  **Record**: Once triggered, record the user's voice until a period of silence is detected (VAD).
3.  **Transcribe**: Pass the recorded audio buffer to local `faster-whisper` to get the text prompt.
4.  **Stream to LLM**: Send the transcribed text to the Gemini API with streaming enabled.
5.  **Stream to TTS**: As chunks of text stream back from Gemini, they are parsed into sentences and immediately piped into the ElevenLabs streaming TTS function to ensure the absolute minimum latency before the audio starts playing.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd respeaker-demo
    ```

2.  **Create a Python virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up API Keys:**
    Create a file named `.env.local` by copying the example file:
    ```bash
    cp .env.example .env.local
    ```
    Now, edit `.env.local` and insert your API keys:
    ```
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    ELEVENLABS_API_KEY="YOUR_ELEVENLABS_API_KEY"
    ```

5.  **Run the assistant:**
    ```bash
    python voice_assistant.py
    ```
