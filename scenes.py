"""
scenes.py
---------
Detects visual scene changes / hard cuts using FFmpeg's own `select` filter
with the `scene` score, instead of decoding and diffing every frame in
Python. FFmpeg does this in C, so it's fast and needs no extra GPU/CV libs.

Each detected scene-cut timestamp becomes a "visual excitement" spike: a lot
of cuts close together (fast edit, action, montage) reads as more visually
energetic than one static shot for two minutes straight.
"""

from __future__ import annotations

import re
import subprocess
import numpy as np


def detect_scene_cuts(path: str, threshold: float = 0.35) -> list[float]:
    """
    Returns a sorted list of timestamps (seconds) where FFmpeg thinks a
    scene/shot change happened. `threshold` is 0..1, lower = more sensitive.
    """
    cmd = [
        "ffmpeg", "-i", path,
        "-filter:v", f"select='gt(scene,{threshold})',showinfo",
        "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True)
    times = []
    for line in proc.stderr.splitlines():
        m = re.search(r"pts_time:(\d+\.?\d*)", line)
        if m:
            times.append(float(m.group(1)))
    return sorted(times)


def scene_density_curve(cut_times: list[float], times: np.ndarray, window: float = 3.0) -> np.ndarray:
    """
    Converts a sparse list of cut timestamps into a smooth 0..1 curve aligned
    to the same `times` axis used by the audio analysis, so it can be added
    into the combined highlight score. `window` is how many seconds around
    each sample point we count cuts within.
    """
    cuts = np.array(cut_times)
    density = np.zeros_like(times)
    if len(cuts) == 0:
        return density
    for i, t in enumerate(times):
        density[i] = np.sum((cuts >= t - window / 2) & (cuts <= t + window / 2))
    hi = density.max()
    return density / hi if hi > 0 else density
