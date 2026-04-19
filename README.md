# Multimodal Voice Assistant — Raspberry Pi 5 + RUBIK Pi 3

A low-latency voice + vision assistant pipeline running on Raspberry Pi 5 with a RUBIK Pi 3 (Qualcomm QCS6490) as an AI coprocessor.  
Speaks, listens, and **sees** — STT and LLM inference offloaded to the RUBIK Pi 3, TTS via ElevenLabs, vision via a 12 MP autofocus camera.

---

## Architecture

### Hardware

| Component | Model | Interface |
|---|---|---|
| Single-board computer | Raspberry Pi 5 (4 GB / 8 GB) | — |
| AI Coprocessor | RUBIK Pi 3 (Qualcomm QCS6490) | Ethernet/Wi-Fi |
| Microphone array | ReSpeaker Mic Array v3.0 | USB |
| Speaker | Amazon Basics USB-powered speaker | 3.5 mm (ReSpeaker jack) |
| Camera | Arducam IMX708 (Camera Module 3) — 12 MP, Autofocus | 15–22 pin FFC MIPI CSI-2 |

> The ReSpeaker performs on-device Acoustic Echo Cancellation (AEC), so the
> microphone does not pick up the assistant's own TTS output.

### Software Stack

| Layer | Library / Service | Notes |
|---|---|---|
| Wake word | `openwakeword` | Fully local on Pi 5 — no API key, no cloud |
| Audio I/O | `sounddevice` | Direct ALSA access via libportaudio |
| Camera | `picamera2` | Official libcamera Python wrapper for Pi 5 |
| Speech-to-text | `faster-whisper` on RUBIK Pi 3 | Remote HTTP — int8 Whisper on QCS6490 NPU |
| LLM | Phi-3 Mini (Dolphin) via `llama-server` on RUBIK Pi 3 | Remote HTTP — OpenAI-compatible endpoint |
| Text-to-speech | `elevenlabs` | Streaming PCM → sounddevice playback |

### Execution Flow

```
┌─────────────────────────────────────────────────────────────┐
│  1. PASSIVE LOOP  (Raspberry Pi 5)                          │
│     openwakeword feeds 80 ms audio chunks until             │
│     "Hey Jarvis" is detected (score ≥ threshold)            │
└────────────────────┬────────────────────────────────────────┘
                     │ wake word fired
                     ▼
┌──────────────────────────────────────────────────────────────┐
│  2. PARALLEL CAPTURE  (Raspberry Pi 5)                       │
│     Two threads, started simultaneously                      │
│                                                              │
│   Thread A — AudioRecorder                                   │
│     sounddevice InputStream → VAD silence detection →        │
│     int16 NumPy array (user's speech)                        │
│                                                              │
│   Thread B — CameraCapture                                   │
│     Picamera2 preview (already warm, continuous AF) →        │
│     capture_file() → in-memory JPEG bytes                    │
│                                                              │
│   Both threads join(); audio is the long pole.               │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  3. TRANSCRIBE  (RUBIK Pi 3 — Qualcomm QCS6490)             │
│     Pi 5 → POST /transcribe (raw WAV bytes)                 │
│     faster-whisper on RUBIK Pi 3 → JSON {"transcript":"…"}  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  4. LLM REQUEST  (RUBIK Pi 3 — Qualcomm QCS6490)            │
│     Pi 5 → POST /v1/chat/completions (OpenAI-compat SSE)    │
│     Phi-3 Mini Q4 via llama-server                          │
│       messages: [system, history…, user transcript]         │
│     → streaming token generator (SSE)                       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  5. STREAMING TTS  (Raspberry Pi 5 + ElevenLabs cloud)      │
│     Tokens → sentence buffer → ElevenLabs Turbo v2.5 →      │
│     PCM 16 kHz → sounddevice playback                       │
│     (first audio plays before LLM finishes generating)      │
└─────────────────────────────────────────────────────────────┘
```

#### Why latency stays low

- **Camera always warm.** Picamera2 is initialized once at startup. With
  `AfMode.Continuous`, the lens tracks the scene at all times. When the wake
  word fires, `capture_file()` grabs the current frame in ~0.1 s — no cold
  libcamera init, no autofocus wait.
- **Parallel capture.** The microphone starts recording at exactly the same
  moment the camera grabs its frame. Neither blocks the other.
- **No disk I/O.** The captured image is held in a `BytesIO` buffer and sent
  directly to Gemini as inline base64 — zero filesystem overhead.
- **Streaming LLM + TTS.** ElevenLabs begins synthesising the first sentence
  while Gemini is still generating the rest.

---

## Setup

### 1. System dependencies

Run these on the Pi **before** creating the Python environment.

```bash
# Update package lists
sudo apt update

# libcamera runtime + tools (required by picamera2)
sudo apt install -y libcamera-apps libcamera-dev

# Picamera2 Python library and its system-level dependencies
sudo apt install -y python3-picamera2

# PortAudio (required by sounddevice)
sudo apt install -y libportaudio2 portaudio19-dev

# Optional: verify the camera is detected by libcamera
libcamera-hello --list-cameras
```

> **Virtual environment note:** `python3-picamera2` installs into the system
> Python. To use it inside a venv, either:
>
> - Create the venv with `--system-site-packages`:
>   ```bash
>   python3 -m venv --system-site-packages .venv
>   ```
> - Or install the PyPI package instead (no system access needed):
>   ```bash
>   pip install picamera2
>   ```
>   The PyPI package pulls in `libcamera` bindings automatically on Pi OS Bookworm.

### 2. Clone and enter the repo

```bash
git clone <repository-url>
cd respeaker-demo
```

### 3. Create the Python virtual environment

```bash
# Use --system-site-packages if you installed picamera2 via apt above
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
```

### 4. Install Python dependencies

```bash
pip install \
  sounddevice \
  numpy \
  openwakeword \
  requests \
  elevenlabs \
  python-dotenv \
  picamera2        # skip if installed via apt + system-site-packages
```

Or use a `requirements.txt` if present:

```bash
pip install -r requirements.txt
```

### 5. Configure API keys

```bash
cp .env.example .env.local
```

Edit `.env.local`:

```dotenv
GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
ELEVENLABS_API_KEY="YOUR_ELEVENLABS_API_KEY"
```

- Gemini key: <https://aistudio.google.com/apikey>
- ElevenLabs key: <https://elevenlabs.io/app/settings/api-keys>

### 6. (Optional) Verify camera

```bash
# Quick preview test — should show a live feed for 5 seconds
libcamera-hello -t 5000

# Or from Python
python3 -c "from picamera2 import Picamera2; c = Picamera2(); print(c.camera_properties)"
```

### 7. Run

```bash
source .venv/bin/activate
python voice_assistant.py
```

Say **"Hey Jarvis"** to activate.

---

## RUBIK Pi 3 Setup

The RUBIK Pi 3 (Qualcomm QCS6490) acts as the AI coprocessor, running both the
STT and LLM inference servers. The Pi 5 reaches it over the local network.

**Assumed IP address:** `192.168.4.28`  
To change this, update `RUBIKPI_HOST` in [config.py](config.py).

### STT server — faster-whisper HTTP

Runs on port **8000**. Accepts raw WAV bytes, returns a JSON transcript.

```bash
# On the RUBIK Pi 3
pip install faster-whisper flask

python3 - <<'EOF'
from flask import Flask, request, jsonify
import tempfile, os
from faster_whisper import WhisperModel

app = Flask(__name__)
model = WhisperModel("base.en", device="cpu", compute_type="int8")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(request.data)
        tmp = f.name
    try:
        segments, _ = model.transcribe(tmp, language="en")
        text = " ".join(s.text for s in segments).strip()
    finally:
        os.unlink(tmp)
    return jsonify({"transcript": text})

app.run(host="0.0.0.0", port=8000)
EOF
```

**Endpoint summary:**

| Method | Path | Body | Response |
|---|---|---|---|
| POST | `/transcribe` | raw WAV bytes (`Content-Type: audio/wav`) | `{"transcript": "..."}` |

### LLM server — llama-server (Phi-3 Mini Q4)

Runs on port **8080** with an OpenAI-compatible API.

```bash
# On the RUBIK Pi 3 — download llama.cpp and a Phi-3 Mini GGUF model first
./llama-server \
  --model ./phi-3-mini-4k-instruct-q4.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx-size 4096
```

**Endpoint summary:**

| Method | Path | Body | Response |
|---|---|---|---|
| POST | `/v1/chat/completions` | OpenAI Chat Completions JSON (`"stream": true`) | SSE token stream |
| GET  | `/health` | — | `{"status": "ok"}` |

---

## Configuration

All tunable constants live at the top of `voice_assistant.py`.

| Constant | Default | Description |
|---|---|---|
| `WAKE_WORD_MODEL` | `"hey_jarvis"` | openwakeword model name |
| `WAKE_WORD_THRESHOLD` | `0.6` | Detection sensitivity (0–1) |
| `SILENCE_THRESHOLD` | `500` | RMS amplitude below which audio is silence |
| `SILENCE_DURATION` | `1.5` | Seconds of silence before recording stops |
| `WHISPER_MODEL` | `"base.en"` | `tiny.en` / `base.en` / `small.en` |
| `GEMINI_MODEL` | `"gemini-2.5-flash"` | Gemini model ID |
| `CAMERA_ENABLED` | `True` | Set `False` to run audio-only |
| `CAMERA_CAPTURE_WIDTH/HEIGHT` | `2028 × 1520` | Half-sensor binned mode (fast, high quality) |
| `CAMERA_JPEG_QUALITY` | `85` | JPEG compression (60–95) |
| `CAMERA_WARMUP_SECONDS` | `2.0` | Seconds to wait for AE/AWB/AF to converge |
| `ELEVENLABS_VOICE_ID` | `"NFG5qt843uXKj4pFvR7C"` | Voice ID from ElevenLabs dashboard |

---

## Troubleshooting

**Camera not detected**
```bash
libcamera-hello --list-cameras
# Should show: "Available cameras: 1"
# If empty, check the FFC ribbon cable is seated correctly (Pi 5 uses a
# 15-pin connector; the Arducam adapter bridges to 22-pin).
```

**`ImportError: No module named 'picamera2'` inside venv**  
Re-create the venv with `--system-site-packages`, or run `pip install picamera2`.

**Audio device index wrong**  
The script prints all available devices on startup. Set `sd.default.device` at
the top of `voice_assistant.py` to pin a specific input/output index.

**Wake word fires too easily / too rarely**  
Adjust `WAKE_WORD_THRESHOLD` (raise to reduce false positives, lower to
increase sensitivity).

**Whisper transcription is slow**  
Switch to `WHISPER_MODEL = "tiny.en"` for the fastest option, or run
`faster-whisper` with `compute_type="float16"` if you attach a compatible
accelerator.
