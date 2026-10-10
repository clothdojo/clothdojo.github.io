"""Re-render recorded states as RGB previews or a LeRobot dataset."""

import argparse
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path, nargs="?",
                        help="trajectory directory; omit with --selection")
    parser.add_argument("--assets", type=Path, default=Path("assets"))
    parser.add_argument("--visual-assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[0, -1])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export", action="store_true", help="export the complete episode as LeRobot v2.1")
    parser.add_argument("--selection", type=Path,
                        help="selection.json proving this rollout was chosen as a success")
    parser.add_argument("--export-python", type=Path,
                        help="Python in the separate ClothDojo export/policy environment")
    args = parser.parse_args()
    if args.output.exists() or args.gpu < 0 or args.seed < 0:
        parser.error("use a fresh output and non-negative GPU/seed")
    if args.export_python and (not args.export or not args.export_python.is_file()):
        parser.error("--export-python requires --export and an existing Python executable")
    if args.selection and not args.export:
        parser.error("--selection requires --export")
    if args.trajectory is None:
        if args.selection is None:
            parser.error("provide a trajectory or --selection")
        from clothdojo.data.selection import first_selected_trajectory

        try:
            args.trajectory = first_selected_trajectory(args.selection)
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.error(str(error))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    from clothdojo.data.isaac_render import (
        _preload_python_libstdcpp, _require_isaac5_runtime, _start_isaac_app,
    )

    _require_isaac5_runtime()
    _preload_python_libstdcpp()
    from clothdojo.teleop.isaac_compat import configure_replicator_warp

    configure_replicator_warp()
    portable = (Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
                / "clothdojo" / "isaac5" / f"gpu-{args.gpu}")
    portable.mkdir(parents=True, exist_ok=True)
    camera = None
    exit_code = 1
    try:
        app = _start_isaac_app(args.gpu, portable)
        import cv2
        import numpy as np

        from clothdojo.rendering import DATASET_CAMERA_NAMES
        from clothdojo.trajectory.isaac_replay import IsaacReplayCamera

        camera = IsaacReplayCamera(app, args.trajectory, assets_root=args.assets,
                                   visual_assets_root=args.visual_assets, seed=args.seed)
        if args.export:
            if args.export_python:
                from clothdojo.data.rgb_export import export_in_separate_python

                export_in_separate_python(camera, args.trajectory, args.output,
                                          assets_root=args.assets, python=args.export_python)
            else:
                from clothdojo.trajectory.lerobot import export_lerobot

                export_lerobot(args.trajectory, args.output, assets_root=args.assets,
                               replay_camera=camera)
            if args.selection:
                from clothdojo.data.selection import approve_selected_export

                approve_selected_export(args.selection, args.trajectory, args.output)
            print(args.output)
            exit_code = 0
            return
        count = camera.reader.manifest.state_count
        indices = list(dict.fromkeys(index if index >= 0 else count + index for index in args.frames))
        if any(index < 0 or index >= count for index in indices):
            raise ValueError("requested frame is outside the recording")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".rgb-", dir=args.output.parent) as temporary:
            staging = Path(temporary) / "frames"
            staging.mkdir()
            for index in indices:
                views = camera(index)
                panels = [views[f"observation.images.{name}"][0] for name in DATASET_CAMERA_NAMES]
                image = np.concatenate((np.concatenate(panels[:2], axis=1),
                                        np.concatenate(panels[2:], axis=1)), axis=0)
                if not cv2.imwrite(str(staging / f"frame_{index:06d}.png"),
                                   cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
                    raise RuntimeError("could not write RGB preview")
            (staging / "render.json").write_text(json.dumps({
                "renderer": "isaac-rtx", "frames": indices,
                "cameras": list(DATASET_CAMERA_NAMES), "layouts": camera.layouts,
            }, indent=2) + "\n")
            staging.rename(args.output)
        print(args.output)
        exit_code = 0
    except BaseException:
        traceback.print_exc()
    finally:
        try:
            if camera is not None:
                camera.close()
        except BaseException:
            traceback.print_exc()
            exit_code = 1
        # Keep Isaac's native teardown inside this disposable CLI process.
        # Plugin unloading can otherwise swallow the Python exception/exit code.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)


if __name__ == "__main__":
    main()
