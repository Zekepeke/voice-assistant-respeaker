#!/usr/bin/env python3
"""
=============================================================================
WAKE WORD TEST SCRIPT
=============================================================================
Tests openwakeword detection with real-time audio visualization.
Shows detection scores so you can verify it's working and tune the threshold.

Usage:
    python test_wakeword.py

Say the wake word and watch the scores - when it exceeds the threshold,
detection triggers!
=============================================================================
"""

import numpy as np
import time

# =============================================================================
# CONFIGURATION - CHANGE THESE TO TEST DIFFERENT WAKE WORDS
# =============================================================================

# Available pre-trained models in openwakeword:
#   "alexa"           - responds to "Alexa"
#   "hey_mycroft"     - responds to "Hey Mycroft" 
#   "hey_jarvis"      - responds to "Hey Jarvis"
#   "hey_rhasspy"     - responds to "Hey Rhasspy"
#
# NOTE: There is NO "hey_computer" model! Use one of the above.

WAKE_WORD_MODEL = "hey_mycroft"  # Try: "alexa", "hey_jarvis", "hey_rhasspy"
THRESHOLD = 0.5                   # Detection threshold (lower = more sensitive)

# =============================================================================

print("Loading libraries...")

try:
    import sounddevice as sd
except ImportError:
    print("ERROR: pip install sounddevice")
    exit(1)

try:
    import openwakeword
    from openwakeword.model import Model as OWWModel
except ImportError:
    print("ERROR: pip install openwakeword")
    exit(1)

# Audio settings
SAMPLE_RATE = 16000
CHUNK_SIZE = 1280  # 80ms chunks required by openwakeword

def main():
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║              🎤 WAKE WORD TEST SCRIPT 🎤                     ║
╠══════════════════════════════════════════════════════════════╣
║  Model: {WAKE_WORD_MODEL:<50} ║
║  Threshold: {THRESHOLD:<47} ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    # Download models if needed
    print("📥 Downloading/loading models (first run may take a moment)...")
    openwakeword.utils.download_models()
    
    # List available models
    print("\n📋 Available pre-trained wake word models:")
    print("   - alexa         → say 'Alexa'")
    print("   - hey_mycroft   → say 'Hey Mycroft'")
    print("   - hey_jarvis    → say 'Hey Jarvis'")
    print("   - hey_rhasspy   → say 'Hey Rhasspy'")
    print(f"\n   Currently testing: '{WAKE_WORD_MODEL}'")
    
    # Create model
    print(f"\n🔧 Loading model '{WAKE_WORD_MODEL}'...")
    try:
        model = OWWModel(
            wakeword_models=[WAKE_WORD_MODEL],
            inference_framework="onnx"
        )
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("\nTry one of the available models listed above.")
        return
    
    print("✅ Model loaded!")
    print("\n" + "=" * 60)
    print(f"🎤 LISTENING - Say the wake word for '{WAKE_WORD_MODEL}'")
    print("=" * 60)
    print("\nScore visualization (threshold shown with |):")
    print(f"  0.0 {'─' * 20}|{'─' * 29} 1.0")
    print("                      ↑")
    print(f"              threshold={THRESHOLD}")
    print("\nPress Ctrl+C to stop\n")
    
    detection_count = 0
    
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='int16',
            blocksize=CHUNK_SIZE
        ) as stream:
            while True:
                # Read audio chunk
                audio_chunk, overflowed = stream.read(CHUNK_SIZE)
                audio_data = audio_chunk.flatten().astype(np.int16)
                
                # Get prediction scores
                predictions = model.predict(audio_data)
                
                # Visualize the score
                for model_name, score in predictions.items():
                    # Create a visual bar
                    bar_length = 50
                    threshold_pos = int(THRESHOLD * bar_length)
                    score_pos = int(min(score, 1.0) * bar_length)
                    
                    bar = ""
                    for i in range(bar_length):
                        if i == score_pos:
                            bar += "█"
                        elif i == threshold_pos:
                            bar += "|"
                        elif i < score_pos:
                            bar += "▓"
                        else:
                            bar += "░"
                    
                    # Check for detection
                    if score >= THRESHOLD:
                        detection_count += 1
                        print(f"\r  [{bar}] {score:.3f} 🔔 DETECTED! (#{detection_count})     ")
                        model.reset()  # Reset to prevent repeated triggers
                        time.sleep(0.5)  # Brief pause after detection
                    else:
                        print(f"\r  [{bar}] {score:.3f}          ", end="", flush=True)
                        
    except KeyboardInterrupt:
        print(f"\n\n👋 Test complete! Detected {detection_count} times.")
        print("\n💡 Tips:")
        print("   - If score stays near 0: speak louder or closer to mic")
        print("   - If score spikes but doesn't reach threshold: lower THRESHOLD")
        print("   - If false triggers: raise THRESHOLD")
        print(f"   - Make sure you're saying the correct phrase for '{WAKE_WORD_MODEL}'")

if __name__ == "__main__":
    main()
