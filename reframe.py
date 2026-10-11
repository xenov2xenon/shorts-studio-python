"""
reframe.py
----------
When cropping a widescreen source down to 9:16, a plain center-crop often
chops off whoever is talking if they're not dead-center. This does a cheap
motion-weighted reframe: sample a handful of frames from the clip, find
where the *motion* (frame-to-frame pixel difference) is concentrated
horizontally, and crop around that instead of the geometric center.

This uses OpenCV for frame sampling/diffing only - no ML models, no GPU
needed, so it works the same on any machine.
"""

from __future__ import annotations

import cv2
import numpy as np


def _sample_motion_center_x(path: str, start: float, end: float, samples: int = 12) -> float:
    """
    Returns a 0..1 fraction across the frame width representing where most
    of the motion happened during [start, end]. 0.5 = center (fallback).
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0.5

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    timestamps = np.linspace(start, end, samples)
    prev_gray = None
    weighted_sum = 0.0
    weight_total = 0.0

    for t in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (160, 90))  # cheap: downsample before diffing

        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            col_energy = diff.sum(axis=0).astype(np.float64)  # motion per column
            if col_energy.sum() > 0:
                xs = np.linspace(0, 1, len(col_energy))
                center = float(np.average(xs, weights=col_energy))
                weight = float(col_energy.sum())
                weighted_sum += center * weight
                weight_total += weight

        prev_gray = gray

    cap.release()
    if weight_total == 0:
        return 0.5
    return weighted_sum / weight_total


def compute_crop_center(path: str, start: float, end: float) -> float:
    """
    Public entry point. Returns clamped 0..1 horizontal crop center,
    keeping enough margin that a 9:16 crop out of a 16:9 source never
    runs off either edge.
    """
    center = _sample_motion_center_x(path, start, end)
    # Clamp so a 9:16 crop (roughly 56% of a 16:9 frame's width) stays in bounds
    margin = 0.28
    return float(np.clip(center, margin, 1 - margin))
