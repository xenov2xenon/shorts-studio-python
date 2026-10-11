"""
audio.py
--------
Turns the raw waveform into a per-second "how interesting is this moment"
score. This is the single biggest upgrade over plain RMS-loudness scoring:

  1. RMS energy       -> how LOUD a moment is (shouting, music swells, hype)
  2. Onset strength    -> how SUDDEN a moment is (laughs, punchlines, stingers,
                           a beat drop, a gasp). A loud-but-flat moment (e.g.
                           someone talking steadily) scores lower here than a
                           sudden spike, which is usually the actually
                           "clippable" bit.
  3. Silence map        -> used later to snap clip boundaries so we never
                           cut off a word or a laugh mid-way.

Combining (1) and (2) instead of just (1) is why this finds punchlines and
reactions, not just "the loud part."
"""

from __future__ import annotations

import subprocess
import tempfile
import os

import numpy as np

try:
    import librosa
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "librosa is required for audio analysis. Install with:\n"
        "    pip install librosa soundfile"
    ) from e


HOP_SECONDS = 0.5  # resolution of the analysis timeline


def _extract_wav(path: str, sr: int) -> str:
    """
    librosa/soundfile can't read most video containers (mp4, mkv, ...)
    directly. FFmpeg can read literally anything, so we always decode the
    audio track to a temp mono WAV first, then hand that to librosa.
    """
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", path,
        "-vn", "-ac", "1", "-ar", str(sr),
        wav_path,
    ]
    subprocess.run(cmd, check=True)
    return wav_path


def _normalize(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def analyze_audio(path: str, sr: int = 22050) -> dict:
    """
    Load audio and return a dict of aligned per-hop arrays:
        times        (n,)  seconds, hop-aligned
        rms          (n,)  normalized 0..1 loudness
        onset        (n,)  normalized 0..1 "suddenness" / excitement
        excitement   (n,)  weighted combination of the two
        silence_mask (n,)  True where the audio is near-silent (for boundary snapping)
    """
    wav_path = _extract_wav(path, sr)
    try:
        y, sr = librosa.load(wav_path, sr=sr, mono=True)
    finally:
        os.remove(wav_path)
    hop_length = int(HOP_SECONDS * sr)

    # 1. Loudness envelope
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

    # 2. Onset strength envelope (spectral flux) - catches sudden events
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    # Align lengths (librosa can differ by 1 frame between features)
    n = min(len(rms), len(onset_env))
    rms, onset_env = rms[:n], onset_env[:n]

    times = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=hop_length)

    rms_n = _normalize(rms)
    onset_n = _normalize(onset_env)

    # Weighted blend: onset (excitement/spikes) counts for more than
    # raw loudness, because a steady loud passage isn't as "clippable"
    # as a sudden reaction.
    excitement = 0.45 * rms_n + 0.55 * onset_n

    # Silence mask, for later boundary snapping (avoid cutting mid-word)
    db = librosa.amplitude_to_db(rms + 1e-9, ref=np.max)
    silence_mask = db < -35.0  # tune threshold if your source is very quiet

    return {
        "times": times,
        "rms": rms_n,
        "onset": onset_n,
        "excitement": excitement,
        "silence_mask": silence_mask,
        "hop_seconds": HOP_SECONDS,
    }


def nearest_silence_time(analysis: dict, target_time: float, search_window: float = 1.5) -> float:
    """
    Find the nearest silent moment to `target_time` within +/- search_window
    seconds. Used to snap a clip's start/end onto a natural pause instead of
    slicing through a word.
    """
    times = analysis["times"]
    mask = analysis["silence_mask"]
    lo = target_time - search_window
    hi = target_time + search_window
    idx = np.where((times >= lo) & (times <= hi) & mask)[0]
    if len(idx) == 0:
        return target_time
    best = idx[np.argmin(np.abs(times[idx] - target_time))]
    return float(times[best])
