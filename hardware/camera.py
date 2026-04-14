"""
hardware/camera.py — Arducam IMX708 (Camera Module 3) via Picamera2.

Design goals
────────────
1. Warm camera — Picamera2 is initialised ONCE at startup. Continuous AF
   (AfMode.Continuous) keeps the lens always focused on the scene, so
   capture_to_memory() grabs the current preview frame with no AF wait.

2. Zero disk I/O — frames are encoded to JPEG inside a BytesIO buffer and
   returned as raw bytes. The calling code passes those bytes straight to
   Gemini as an inline Blob.

3. Graceful degradation — if picamera2/libcamera is not installed the class
   initialises cleanly and every capture returns None, allowing the rest of
   the pipeline to run in audio-only mode.

Install notes
─────────────
  sudo apt install -y python3-picamera2          # system Python
  # OR, inside a venv:
  pip install picamera2                          # pulls libcamera bindings
"""

import io
import logging
import threading
from typing import Optional

import config

log = logging.getLogger(__name__)

# Attempt to import Picamera2 once at module load time so the import error
# is surfaced early (at startup) rather than on first capture.
try:
    from picamera2 import Picamera2
    from libcamera import controls as LibCameraControls
    _PICAMERA2_AVAILABLE = True
except ImportError:
    _PICAMERA2_AVAILABLE = False
    log.warning(
        "picamera2 / libcamera not found — camera will be disabled.\n"
        "  Install: sudo apt install -y python3-picamera2\n"
        "      OR:  pip install picamera2"
    )


class CameraCapture:
    """
    Manages the Arducam IMX708 in continuous-AF preview mode.

    Typical usage
    ─────────────
        cam = CameraCapture()
        cam.initialize()            # called once at startup
        jpeg = cam.capture_to_memory()   # called per interaction
        cam.cleanup()               # called on shutdown
    """

    def __init__(self) -> None:
        self._camera: Optional["Picamera2"] = None
        self._initialised: bool = False
        # Prevent concurrent captures from two threads (defensive)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        """
        Configure Picamera2 for continuous-AF preview and start the pipeline.

        Configuration choices
        ─────────────────────
        • create_preview_configuration — keeps the ISP running continuously
          (AE/AWB/AF always active) so capture_to_memory() is near-instant.
        • main stream 2028×1520 RGB888 — IMX708 half-sensor binned mode:
          full colour, fast 30 fps readout, enough detail for Gemini vision.
        • lores stream 640×480 YUV420 — small auxiliary stream used by the
          AF/AE algorithms internally; not used in application code.
        • display=None — headless; no HDMI output needed.
        • AfMode.Continuous + AfSpeed.Fast — lens constantly tracks the
          sharpest point; zero manual AF trigger needed per capture.

        Returns:
            True on success, False on any hardware / library error.
        """
        if not _PICAMERA2_AVAILABLE or not config.CAMERA_ENABLED:
            log.info("Camera disabled or unavailable — skipping initialisation")
            return False

        log.info(
            "Initialising camera  %dx%d  JPEG q=%d",
            config.CAMERA_CAPTURE_WIDTH,
            config.CAMERA_CAPTURE_HEIGHT,
            config.CAMERA_JPEG_QUALITY,
        )

        try:
            self._camera = Picamera2()

            cam_config = self._camera.create_preview_configuration(
                main={
                    "size":   (config.CAMERA_CAPTURE_WIDTH,
                               config.CAMERA_CAPTURE_HEIGHT),
                    "format": "RGB888",
                },
                lores={
                    "size":   (640, 480),
                    "format": "YUV420",
                },
                display=None,  # Headless — no display output
            )
            self._camera.configure(cam_config)

            # Enable continuous autofocus with maximum speed
            self._camera.set_controls({
                "AfMode": LibCameraControls.AfModeEnum.Continuous,
                "AfSpeed": LibCameraControls.AfSpeedEnum.Fast,
            })

            self._camera.start()

            # Allow AE/AWB/AF to converge before the first capture.
            import time
            log.info("Camera warm-up %.1f s ...", config.CAMERA_WARMUP_SECONDS)
            time.sleep(config.CAMERA_WARMUP_SECONDS)

            self._initialised = True
            log.info("Camera ready")
            return True

        except Exception:
            log.exception("Camera initialisation failed — running without camera")
            self._camera = None
            return False

    def cleanup(self) -> None:
        """Stop the Picamera2 pipeline and release the camera resource."""
        if self._camera is not None:
            try:
                self._camera.stop()
                self._camera.close()
                log.info("Camera released")
            except Exception:
                log.warning("Error releasing camera", exc_info=True)
            finally:
                self._camera = None
        self._initialised = False

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def capture_to_memory(self) -> Optional[bytes]:
        """
        Grab the current preview frame and return it as in-memory JPEG bytes.

        Because AfMode.Continuous is active, the lens is always tracking the
        scene — there is no per-capture focus cycle to wait for.

        capture_file(buf, format='jpeg') encodes the *next* complete frame
        from the preview pipeline directly into a BytesIO buffer. No numpy
        conversion, no PIL dependency, no disk write.

        Returns:
            Raw JPEG bytes, or None if the camera is unavailable.
        """
        if not self._initialised or self._camera is None:
            return None

        with self._lock:
            import time

            try:
                t0  = time.monotonic()
                buf = io.BytesIO()
                self._camera.capture_file(buf, format="jpeg")
                jpeg_bytes = buf.getvalue()

                elapsed = time.monotonic() - t0
                log.info(
                    "Camera captured  %.0f KB  in %.2f s",
                    len(jpeg_bytes) / 1_024,
                    elapsed,
                )
                return jpeg_bytes

            except Exception:
                log.exception("Camera capture failed")
                return None
