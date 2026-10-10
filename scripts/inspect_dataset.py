"""Read one LeRobot v2.1 frame without PyTorch, a simulator or a GPU."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    args = parser.parse_args()
    root = args.dataset.resolve()
    info = json.loads((root / "meta/info.json").read_text())
    if info.get("codebase_version") != "v2.1":
        parser.error("expected an ordinary LeRobot v2.1 dataset")
    if not 0 <= args.episode < info["total_episodes"] or args.frame < 0:
        parser.error("episode or frame index out of range")

    import cv2
    import pyarrow.parquet as pq

    chunk = args.episode // info["chunks_size"]
    fields = {"episode_chunk": chunk, "episode_index": args.episode}
    data = root / info["data_path"].format(**fields)
    table = pq.read_table(data, columns=["observation.state", "action", "timestamp"])
    if args.frame >= table.num_rows:
        parser.error("frame index exceeds this episode")
    sample = table.slice(args.frame, 1).to_pylist()[0]
    views = {}
    for key, feature in info["features"].items():
        if feature.get("dtype") != "video":
            continue
        video = root / info["video_path"].format(**fields, video_key=key)
        capture = cv2.VideoCapture(str(video))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
            ok, pixels = capture.read()
            if not ok:
                raise RuntimeError(f"could not decode frame {args.frame} from {video}")
            views[key] = {"shape": list(pixels.shape), "dtype": str(pixels.dtype)}
        finally:
            capture.release()
    print(json.dumps({"episode": args.episode, "frame": args.frame,
                      **sample, "views": views}, indent=2))


if __name__ == "__main__":
    main()
