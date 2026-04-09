#!/bin/bash
# =============================================================================
# VOICE ASSISTANT SETUP SCRIPT FOR RASPBERRY PI 5
# =============================================================================
# Run this script to install all system dependencies and Python packages
# 
# Usage: 
#   chmod +x setup.sh
#   ./setup.sh
# =============================================================================

set -e  # Exit on any error

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         🚀 Voice Assistant Setup Script                      ║"
echo "║              Raspberry Pi 5 + ReSpeaker                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# -----------------------------------------------------------------------------
# STEP 1: Update system packages
# -----------------------------------------------------------------------------
echo "📦 Step 1/5: Updating system packages..."
sudo apt-get update

# -----------------------------------------------------------------------------
# STEP 2: Install system dependencies
# -----------------------------------------------------------------------------
echo ""
echo "📦 Step 2/5: Installing system dependencies..."

# Audio libraries required by sounddevice and portaudio
sudo apt-get install -y \
    portaudio19-dev \
    libportaudio2 \
    libportaudiocpp0 \
    python3-pyaudio \
    libasound2-dev \
    libsndfile1

# FFmpeg for audio processing (used by whisper)
sudo apt-get install -y ffmpeg

# Build tools (may be needed for some pip packages)
sudo apt-get install -y \
    build-essential \
    python3-dev \
    python3-pip \
    python3-venv

echo "✅ System dependencies installed"

# -----------------------------------------------------------------------------
# STEP 3: Set up Python virtual environment
# -----------------------------------------------------------------------------
echo ""
echo "🐍 Step 3/5: Setting up Python virtual environment..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/.venv"

if [ ! -d "$VENV_PATH" ]; then
    echo "   Creating virtual environment at $VENV_PATH..."
    python3 -m venv "$VENV_PATH"
else
    echo "   Virtual environment already exists at $VENV_PATH"
fi

# Activate virtual environment
source "$VENV_PATH/bin/activate"

echo "✅ Virtual environment activated"

# -----------------------------------------------------------------------------
# STEP 4: Upgrade pip and install Python packages
# -----------------------------------------------------------------------------
echo ""
echo "📦 Step 4/5: Installing Python packages..."

# Upgrade pip first
pip install --upgrade pip wheel setuptools

# Install requirements
pip install -r "$SCRIPT_DIR/requirements.txt"

echo "✅ Python packages installed"

# -----------------------------------------------------------------------------
# STEP 5: Configure audio devices
# -----------------------------------------------------------------------------
echo ""
echo "🔊 Step 5/5: Checking audio configuration..."

echo ""
echo "📋 Available audio devices:"
python3 -c "
import sounddevice as sd
print()
for i, d in enumerate(sd.query_devices()):
    in_ch = d['max_input_channels']
    out_ch = d['max_output_channels']
    if in_ch > 0 or out_ch > 0:
        status = []
        if in_ch > 0: status.append(f'{in_ch} in')
        if out_ch > 0: status.append(f'{out_ch} out')
        print(f'  [{i}] {d[\"name\"]} ({", \".join(status)})')
print()
print(f'  Default Input:  {sd.default.device[0]}')
print(f'  Default Output: {sd.default.device[1]}')
print()
"

# -----------------------------------------------------------------------------
# COMPLETE
# -----------------------------------------------------------------------------
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    ✅ SETUP COMPLETE!                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "📝 NEXT STEPS:"
echo ""
echo "   1. Edit voice_assistant.py and add your API keys:"
echo "      - GEMINI_API_KEY (https://aistudio.google.com/apikey)"
echo "      - ELEVENLABS_API_KEY (https://elevenlabs.io/app/settings/api-keys)"
echo ""
echo "   NOTE: Wake word detection is now fully local with openwakeword!"
echo "         No API key required - just say 'Hey Mycroft' to activate."
echo ""
echo "   2. Activate the virtual environment:"
echo "      source .venv/bin/activate"
echo ""
echo "   3. Run the voice assistant:"
echo "      python voice_assistant.py"
echo ""
echo "   4. Say 'Hey Mycroft' (default wake word) to start!"
echo ""
echo "💡 TIP: If ReSpeaker isn't detected, try unplugging and replugging it."
echo "💡 TIP: Change WAKE_WORD_MODEL in voice_assistant.py for different wake words:"
echo "        'alexa', 'hey_jarvis', 'hey_rhasspy', etc."
echo ""
