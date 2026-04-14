"""hardware — Physical I/O drivers for the ReSpeaker mic array and IMX708 camera."""

from hardware.audio import WakeWordDetector, AudioRecorder
from hardware.camera import CameraCapture

__all__ = ["WakeWordDetector", "AudioRecorder", "CameraCapture"]
