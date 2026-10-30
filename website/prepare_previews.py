"""Prepare lightweight homepage previews without changing physical motion or playback time.

Run with the project's sim environment (imageio-ffmpeg==0.6.0).
Full-resolution source files remain available for explicit playback.
"""
from pathlib import Path
import json
import subprocess

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parent
CURATED = ROOT / "assets" / "curated"
OUTPUT = ROOT / "assets" / "preview"

SOURCES = [
    ("tasks-paired", CURATED / "tasks-paired.mp4", 1280, 480),
    ("garment-collection", ROOT / "assets" / "garment-collection-web.mp4", 1280, 720),
    ("generation-200", CURATED / "generation-200.mp4", 1280, 480),
    *[
        (f"robot-{robot}", CURATED / f"robot-{robot}.mp4", 1280, 480)
        for robot in ("piper", "aloha2", "x5", "fr3_duo", "ur5e_2f85")
    ],
]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    records = []
    for name, source, width, height in SOURCES:
        target = OUTPUT / f"{name}.mp4"
        subprocess.run([
            ffmpeg, "-v", "error", "-y", "-i", str(source), "-an",
            "-vf", f"scale={width}:{height}:flags=lanczos,fps=24",
            "-c:v", "libx264", "-threads", "4", "-preset", "slow",
            "-crf", "25", "-maxrate", "1600k", "-bufsize", "3200k",
            "-g", "48", "-pix_fmt", "yuv420p", "-profile:v", "high",
            "-level:v", "4.0", "-movflags", "+faststart", str(target),
        ], check=True)
        records.append({
            "file": str(target.relative_to(ROOT)),
            "source": str(source.relative_to(ROOT)),
            "width": width, "height": height, "fps": 24,
            "playback_speed": 1,
            "source_bytes": source.stat().st_size,
            "preview_bytes": target.stat().st_size,
        })
        print(name, source.stat().st_size, "->", target.stat().st_size, flush=True)
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "schema": "clothdojo_homepage_previews_v1",
        "purpose": "Lightweight ambient playback; full-resolution replay loads on explicit request.",
        "records": records,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
