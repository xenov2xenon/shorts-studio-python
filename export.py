"""
export.py
---------
Builds and runs the FFmpeg command that turns one (start, end) window from
the source video into a finished vertical short. Everything here is one
filter_complex graph so FFmpeg does it in a single pass.

Engagement features baked in here:
  - loudnorm            : normalizes loudness so quiet clips don't feel
                           weak and loud ones don't clip/distort
  - motion-weighted crop : from reframe.py, keeps the subject in frame
  - showwaves overlay    : burns a live audio-waveform bar onto the video
                           (the visual "podcast clip" look that performs
                           well on shorts feeds)
  - fade in/out          : video + audio, so clips don't start/end abruptly
  - subtle zoom (OFF by default) : a slow creeping zoom via FFmpeg's
                           `zoompan` filter. This is disabled by default
                           because `zoompan` is notoriously slow / can hang
                           on some FFmpeg builds (verified while testing
                           this tool - a 2-second clip never finished).
                           Enable with --zoom only after confirming it
                           completes quickly on your machine.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def build_and_run(
    source_path: str,
    out_path: str,
    start: float,
    end: float,
    crop_center_x: float,
    out_w: int = 1080,
    out_h: int = 1920,
    fade_seconds: float = 0.35,
    waveform: bool = True,
    waveform_color: str = "0x39FF88",
    zoom: bool = False,
    gpu: str | None = None,  # "nvenc" | "amf" | "qsv" | None (CPU)
) -> None:
    duration = end - start
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # --- crop math -----------------------------------------------------
    # Crop a 9:16 slice out of the source at the motion-weighted center,
    # then scale up to the target resolution.
    filters = []
    filters.append(
        f"[0:v]crop=ih*9/16:ih:"
        f"(iw-ih*9/16)*{crop_center_x}:0,"
        f"scale={out_w}:{out_h}"
    )

    if zoom:
        # gentle 0% -> 6% creeping zoom over the clip (Ken-Burns style)
        frames = max(1, int(duration * 30))
        filters[-1] += (
            f",zoompan=z='min(zoom+0.0005,1.06)':d={frames}:"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={out_w}x{out_h}"
        )

    filters[-1] += ",fps=30[base]"

    video_label = "[base]"
    if waveform:
        filters.append(
            f"[0:a]showwaves=s={out_w}x220:mode=cline:colors={waveform_color}:"
            f"scale=lin,format=rgba,colorchannelmixer=aa=0.85[wave]"
        )
        filters.append(
            f"{video_label}[wave]overlay=0:H-h-40[withwave]"
        )
        video_label = "[withwave]"

    filters.append(
        f"{video_label}fade=t=in:st=0:d={fade_seconds},"
        f"fade=t=out:st={max(0, duration - fade_seconds)}:d={fade_seconds}[vout]"
    )
    filters.append(
        f"[0:a]loudnorm=I=-14:TP=-1.5:LRA=11,"
        f"afade=t=in:st=0:d={fade_seconds},"
        f"afade=t=out:st={max(0, duration - fade_seconds)}:d={fade_seconds}[aout]"
    )

    filter_complex = ";".join(filters)

    encoder_args_by_type = {
        "nvenc": ["-c:v", "h264_nvenc", "-preset", "p5"],
        "amf": ["-c:v", "h264_amf", "-quality", "balanced"],
        "qsv": ["-c:v", "h264_qsv", "-preset", "medium"],
        None: ["-c:v", "libx264", "-preset", "veryfast"],
    }

    def _cmd(encoder_args):
        return [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", str(start), "-to", str(end), "-i", source_path,
            "-filter_complex", filter_complex,
            "-map", "[vout]", "-map", "[aout]",
            *encoder_args,
            "-crf", "20",
            "-c:a", "aac", "-b:a", "160k",
            out_path,
        ]

    try:
        subprocess.run(_cmd(encoder_args_by_type[gpu]), check=True)
    except subprocess.CalledProcessError:
        if gpu is None:
            raise  # already tried CPU, nothing left to fall back to
        # FFmpeg can report a hardware encoder as compiled-in even when the
        # driver isn't actually usable at runtime (exactly what happened in
        # testing: h264_nvenc listed, but "Cannot load libcuda.so.1"). Retry
        # once on CPU instead of failing the whole clip.
        print(f"  [!] {gpu} encoder failed at runtime, retrying on CPU ...")
        subprocess.run(_cmd(encoder_args_by_type[None]), check=True)


def make_thumbnail(clip_path: str, thumb_path: str, at_seconds: float = 0.5) -> None:
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", str(at_seconds), "-i", clip_path,
        "-frames:v", "1", "-update", "1", "-q:v", "2", thumb_path,
    ]
    subprocess.run(cmd, check=True)
