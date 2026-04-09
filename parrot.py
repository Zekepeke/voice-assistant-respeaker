import sounddevice as sd

# Configuration
duration = 5      # How long the parrot listens (in seconds)
sample_rate = 16000  # 16kHz is the standard, optimized rate for voice recording

print("🦜 Say something! Recording for 5 seconds...")

# Start recording from the default input (ReSpeaker)
recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
sd.wait()  # Tell Python to wait until the 5 seconds are fully up

print("🔊 Parroting back...")

# Play the recording back through the default output (ReSpeaker -> PC Speakers)
sd.play(recording, sample_rate)
sd.wait()  # Tell Python to wait until the audio finishes playing

print("Demo complete!")
