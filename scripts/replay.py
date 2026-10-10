"""Replay a recorded trajectory as a four-camera H.264 video."""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--assets", type=Path, default=Path("assets"))
    parser.add_argument("--output", type=Path, default=Path("outputs/replay.mp4"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cloth-visual", choices=("nocs", "white"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    import numpy as np

    from clothdojo.rendering import DATASET_CAMERA_NAMES
    from clothdojo.trajectory.replay import ReplayCamera

    camera = ReplayCamera(args.trajectory, assets_root=args.assets, device=args.device,
                          cloth_visual=args.cloth_visual)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".replay-", dir=args.output.parent) as staging:
        temporary = Path(staging) / "replay.mp4"
        fps = camera.reader.manifest.control_fps
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
                   "-pixel_format", "rgb24", "-video_size", "1280x960", "-framerate", str(fps),
                   "-i", "pipe:0", "-an", "-c:v", "libx264", "-crf", "18",
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary)]
        with (Path(staging) / "encoder.log").open("w+") as log:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
            try:
                for index in range(camera.reader.manifest.state_count):
                    views = camera(index)
                    panels = [views[f"observation.images.{name}"][0] for name in DATASET_CAMERA_NAMES]
                    frame = np.concatenate((np.concatenate(panels[:2], axis=1),
                                            np.concatenate(panels[2:], axis=1)), axis=0)
                    process.stdin.write(np.ascontiguousarray(frame).tobytes())
                process.stdin.close()
                if process.wait() != 0:
                    log.seek(0)
                    raise RuntimeError(f"FFmpeg encoding failed: {log.read()}")
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
        temporary.rename(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
