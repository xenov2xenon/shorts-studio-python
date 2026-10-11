"""
scorer.py
---------
Combines audio excitement + scene-cut density into one score per timestamp,
then picks the best non-overlapping windows for clips.

Two upgrades over "just take the top-N loudest windows":

1. Anticipation offset: a clip that STARTS exactly on the peak often clips
   off the setup and you just hear the punchline with no context. We shift
   the window start slightly earlier so there's a beat of lead-in.

2. Boundary snapping: clip start/end are nudged onto nearby silence (see
   audio.nearest_silence_time) so cuts land in natural pauses instead of
   slicing a word or a laugh in half.
"""

from __future__ import annotations

import numpy as np

from . import audio as audio_mod


def combined_score(audio_analysis: dict, scene_density: np.ndarray, scene_weight: float = 0.35) -> np.ndarray:
    exc = audio_analysis["excitement"]
    scene_density = scene_density[: len(exc)]
    score = (1 - scene_weight) * exc + scene_weight * scene_density
    return score


def pick_clips(
    times: np.ndarray,
    score: np.ndarray,
    audio_analysis: dict,
    clip_duration: float,
    num_clips: int,
    lead_in: float = 1.5,
    min_gap: float = 2.0,
    search_range: tuple[float, float] | None = None,
) -> list[tuple[float, float, float]]:
    """
    Returns up to `num_clips` (start, end, score) tuples, non-overlapping
    (with `min_gap` seconds between them), highest score first, sorted back
    into chronological order for a natural playback order in the output
    folder.
    """
    hop = audio_analysis["hop_seconds"]
    total_duration = float(times[-1] + hop)

    lo, hi = (search_range if search_range else (0.0, total_duration))
    valid = (times >= lo) & (times <= hi)

    candidates = list(zip(times[valid], score[valid]))
    candidates.sort(key=lambda c: c[1], reverse=True)

    chosen: list[tuple[float, float, float]] = []
    for peak_time, peak_score in candidates:
        if len(chosen) >= num_clips:
            break

        raw_start = max(lo, peak_time - lead_in)
        raw_end = min(hi, raw_start + clip_duration)
        raw_start = max(lo, raw_end - clip_duration)  # keep full duration near the edges

        # snap to natural pauses so we don't cut mid-word/mid-laugh
        start = audio_mod.nearest_silence_time(audio_analysis, raw_start)
        end = audio_mod.nearest_silence_time(audio_analysis, raw_end)
        if end - start < clip_duration * 0.6:
            # snapping collapsed the window too much, fall back to raw cut
            start, end = raw_start, raw_end

        overlaps = any(not (end <= s - min_gap or start >= e + min_gap) for s, e, _ in chosen)
        if overlaps:
            continue

        chosen.append((start, end, float(peak_score)))

    chosen.sort(key=lambda c: c[0])
    return chosen
