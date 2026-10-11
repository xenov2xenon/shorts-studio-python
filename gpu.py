"""
gpu.py
------
Probes the local FFmpeg build for hardware encoders and returns the best
available option. Falls back to CPU (libx264) automatically if nothing
matches - same "graceful degradation" idea as the original app, just done
by checking `ffmpeg -encoders` instead of assuming a vendor from the OS.
"""

from __future__ import annotations

import subprocess


def detect_encoder() -> str | None:
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        ).stdout
    except FileNotFoundError:
        return None

    if "h264_nvenc" in out:
        return "nvenc"
    if "h264_amf" in out:
        return "amf"
    if "h264_qsv" in out:
        return "qsv"
    return None
