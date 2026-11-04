"""One recorded NOCS trajectory and two review videos, using the existing CLI."""
from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time

SERVER_START_TIMEOUT = 600


def _stop(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def _run(command: list[str], log: Path, root: Path, env: dict[str, str]) -> None:
    with log.open("w") as stream:
        process = subprocess.Popen(command, cwd=root, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait()
        finally:
            _stop(process)
    if code:
        raise RuntimeError(f"Command exited with status {code}; see {log}")


def run(task: str, root: Path, output: Path | None = None) -> Path:
    root = root.resolve()
    teacher = root / "models/nocs-teachers" / task
    states = root / f"benchmark/conditions/nocs-{task}/states"
    for path in (teacher / "weights", teacher / "norm_stats/norm_stats.json",
                 states / "meta.json", root / "benchmark/assets"):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run bash scripts/download.sh {task} first.")
    if output is None:
        parent = root / "outputs/quickstart"
        parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output = Path(tempfile.mkdtemp(prefix=f"{task}-{stamp}-", dir=parent))
    else:
        output = output.resolve()
        output.mkdir(parents=True, exist_ok=False)
    logs = output / "logs"
    logs.mkdir()
    collection = output / "rollout"
    env = os.environ.copy()
    env.update(PXR_WORK_THREAD_LIMIT="1", OPENPI_DATA_HOME=str(root / ".cache/openpi"),
               CLOTHDOJO_OPENPI_CACHE=str(root / ".cache/openpi"))
    python = sys.executable
    cli = [python, "-m", "clothdojo"]
    server = None
    print(f"Saving this run to {output}", flush=True)
    try:
        print("[1/4] Preparing the tokenizer", flush=True)
        _run([python, "-c", "from openpi.shared import download; "
              "download.maybe_download('gs://big_vision/paligemma_tokenizer.model', "
              "gs={'token': 'anon', 'session_kwargs': {'trust_env': True}})"],
             logs / "tokenizer.log", root, env)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        print("[2/4] Loading the NOCS teacher", flush=True)
        with (logs / "teacher.log").open("w") as stream:
            server = subprocess.Popen(cli + ["policy", "serve-sft", "--checkpoint",
                str(teacher / "weights"), "--norm-stats", str(teacher / "norm_stats"),
                "--num-steps", "10", "--image-preprocessing", "none", "--rtc-prefix-schedule",
                "exp", "--device", "cuda:0", "--port", str(port)], cwd=root, env=env,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + SERVER_START_TIMEOUT
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"Teacher exited; see {logs / 'teacher.log'}")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    break
            except OSError:
                if time.monotonic() > deadline:
                    raise RuntimeError(f"Teacher startup timed out; see {logs / 'teacher.log'}")
                time.sleep(0.2)
        seconds = 60 if task == "flatten" else 40
        scenario = "shirt_flatten.yaml" if task == "flatten" else "shirt_fold_eval_40s.yaml"
        print(f"[3/4] Generating one {seconds}-second trajectory", flush=True)
        _run(cli + ["rollout", "run", "--scenario", str(root / "configs/rollout" / scenario),
            "--states", str(states), "--policy", f"ws://127.0.0.1:{port}",
            "--assets", str(root / "benchmark/assets"), "--gpus", "0", "--batch-size", "1",
            "--seen-assets", "1", "--unseen-assets", "0", "--limit-per-asset", "1",
            "--execution", "chunk", "--exec-horizon", "10", "--storage-kind", "render_only_v1",
            "--output", str(collection)], logs / "generation.log", root, env)
        _stop(server)
        server = None
        print("[4/4] Exporting high and low camera views", flush=True)
        for camera, destination in (("cam_high", collection / "review"),
                                    ("cam_low", collection / "review-low")):
            _run(cli + ["data", "render-review", "--source", str(collection), "--assets",
                str(root / "benchmark/assets"), "--device", "cuda:0", "--camera", camera,
                "--fps", "10", "--output", str(destination)], logs / f"{camera}.log", root, env)
            videos = list((destination / "videos" / camera).glob("*.mp4"))
            if len(videos) != 1:
                raise RuntimeError(f"Expected one {camera} video; see {logs / f'{camera}.log'}")
            print(f"{camera}: {videos[0]}", flush=True)
        print(f"Trajectory: {collection}", flush=True)
        return collection
    finally:
        _stop(server)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", nargs="?", choices=("flatten", "fold"), default="flatten")
    parser.add_argument("--output", type=Path, help="new run directory; existing files are never replaced")
    args = parser.parse_args(argv)
    try:
        run(args.task, Path.cwd(), args.output)
    except KeyboardInterrupt:
        parser.exit(130, "Cancelled. Partial output and logs have been kept.\n")
    except (OSError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
