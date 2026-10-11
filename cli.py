"""
cli.py
------
End-to-end pipeline: audio analysis -> scene detection -> combined scoring
-> clip selection -> motion-weighted reframe -> ffmpeg export.

Run `python -m shortsmaker.cli --help` for all options.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import audio as audio_mod
from . import scenes as scenes_mod
from . import scorer as scorer_mod
from . import reframe as reframe_mod
from . import export as export_mod
from . import gpu as gpu_mod


def main(argv=None):
    p = argparse.ArgumentParser(description="Find highlight moments in a long video and export vertical shorts.")
    p.add_argument("input", help="Path to the source video (mp4/mov/mkv/webm)")
    p.add_argument("-o", "--out-dir", default="shorts_out", help="Output folder (default: shorts_out)")
    p.add_argument("-n", "--num-clips", type=int, default=5, help="How many shorts to generate")
    p.add_argument("-d", "--duration", type=float, default=30.0, help="Each short's duration in seconds")
    p.add_argument("--start", type=float, default=None, help="Only analyze from this time (s)")
    p.add_argument("--end", type=float, default=None, help="Only analyze until this time (s)")
    p.add_argument("--no-waveform", action="store_true", help="Disable the burned-in audio waveform overlay")
    p.add_argument("--zoom", action="store_true",
                   help="Enable a slow Ken-Burns zoom (OFF by default: FFmpeg's zoompan "
                        "filter is very slow / can hang on some builds - test on a short "
                        "clip first before relying on it)")
    p.add_argument("--width", type=int, default=1080)
    p.add_argument("--height", type=int, default=1920)
    p.add_argument("--scene-weight", type=float, default=0.35, help="0..1, how much visual cuts count vs audio")
    args = p.parse_args(argv)

    src = args.input
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Analyzing audio energy + excitement in {src} ...")
    a = audio_mod.analyze_audio(src)

    print("[2/5] Detecting scene cuts ...")
    cuts = scenes_mod.detect_scene_cuts(src)
    density = scenes_mod.scene_density_curve(cuts, a["times"])

    print("[3/5] Scoring and selecting best non-overlapping clips ...")
    score = scorer_mod.combined_score(a, density, scene_weight=args.scene_weight)
    search_range = (args.start or 0.0, args.end) if (args.start or args.end) else None
    clips = scorer_mod.pick_clips(
        a["times"], score, a,
        clip_duration=args.duration,
        num_clips=args.num_clips,
        search_range=search_range,
    )

    if not clips:
        print("No clips found - try a lower --scene-weight or a shorter --duration.", file=sys.stderr)
        return 1

    encoder = gpu_mod.detect_encoder()
    print(f"[4/5] Encoder: {encoder or 'CPU (libx264)'}")

    print(f"[5/5] Exporting {len(clips)} clip(s) to {out_dir}/ ...")
    for i, (start, end, sc) in enumerate(clips, start=1):
        crop_center = reframe_mod.compute_crop_center(src, start, end)
        out_path = out_dir / f"short_{i:02d}_{int(start)}s.mp4"
        print(f"  -> clip {i}: {start:.1f}s - {end:.1f}s (score {sc:.2f}, crop_x={crop_center:.2f})")
        export_mod.build_and_run(
            src, str(out_path), start, end, crop_center,
            out_w=args.width, out_h=args.height,
            waveform=not args.no_waveform,
            zoom=args.zoom,
            gpu=encoder,
        )
        export_mod.make_thumbnail(str(out_path), str(out_path.with_suffix(".jpg")))

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
