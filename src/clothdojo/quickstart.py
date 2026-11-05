"""Managed NOCS generation and dual-view replay using the existing CLI."""
from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import signal
import shlex
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


def _path(value: str | Path, root: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def generate(root: Path, *, task: str = "flatten", benchmark: str | Path = "benchmark",
             teachers: str | Path = "models/nocs-teachers", output: str | Path | None = None,
             gpu: int = 0, seen_assets: int = 1, unseen_assets: int = 0,
             states_per_asset: int = 1) -> Path:
    """Generate with user choices and print the matching replay command."""
    root = root.resolve()
    collection = run(task, root, output, benchmark=benchmark, teachers=teachers, gpu=gpu,
                     seen_assets=seen_assets, unseen_assets=unseen_assets,
                     states_per_asset=states_per_asset, render=False)
    command = [".venv/bin/clothdojo", "replay", str(collection)]
    if _path(benchmark, root) != root / "benchmark":
        command.extend(["--benchmark", str(_path(benchmark, root))])
    if gpu != 0:
        command.extend(["--gpu", str(gpu)])
    print(f"Replay both views: {shlex.join(command)}", flush=True)
    return collection


def replay(source: str | Path, root: Path, *, benchmark: str | Path = "benchmark",
           gpu: int = 0, logs: Path | None = None) -> None:
    """Export both native views of an existing trajectory collection."""
    if type(gpu) is not int or gpu < 0:
        raise ValueError("gpu must be a non-negative integer")
    collection = _path(source, root)
    assets = _path(benchmark, root) / "assets"
    if not assets.is_dir():
        raise FileNotFoundError(f"Missing benchmark assets: {assets}")
    expected = {p.parent.name for p in (collection / "trajectories").glob("*/manifest.json")}
    if not expected:
        raise ValueError(f"No trajectories found in collection: {collection}")
    logs = logs or collection / "review-logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PXR_WORK_THREAD_LIMIT="1")
    for camera, destination in (("cam_high", collection / "review"),
                                ("cam_low", collection / "review-low")):
        _run([sys.executable, "-m", "clothdojo", "data", "render-review", "--source",
              str(collection), "--assets", str(assets), "--device", f"cuda:{gpu}",
              "--camera", camera, "--fps", "10", "--output", str(destination)],
             logs / f"{camera}.log", root, env)
        videos = sorted((destination / "videos" / camera).glob("*.mp4"))
        if {p.stem for p in videos} != expected:
            raise RuntimeError(f"Missing or unexpected {camera} videos; see {logs / f'{camera}.log'}")
        for video in videos:
            print(f"{camera}: {video}", flush=True)


def run(task: str, root: Path, output: str | Path | None = None, *,
        benchmark: str | Path = "benchmark", teachers: str | Path = "models/nocs-teachers",
        gpu: int = 0, seen_assets: int = 1, unseen_assets: int = 0,
        states_per_asset: int = 1, render: bool = True) -> Path:
    if task not in ("flatten", "fold"):
        raise ValueError("task must be flatten or fold")
    for name, value in (("benchmark", benchmark), ("teachers", teachers), ("output", output)):
        if name == "output" and value is None:
            continue
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ValueError(f"{name} must be a non-empty path")
    for name, value, minimum in (("gpu", gpu, 0), ("seen_assets", seen_assets, 0),
                                  ("unseen_assets", unseen_assets, 0),
                                  ("states_per_asset", states_per_asset, 1)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if seen_assets + unseen_assets == 0:
        raise ValueError("Select at least one seen or unseen garment")
    root = root.resolve()
    benchmark = _path(benchmark, root)
    teacher = _path(teachers, root) / task
    states = benchmark / f"conditions/nocs-{task}/states"
    for path in (teacher / "weights", teacher / "norm_stats/norm_stats.json",
                 states / "meta.json", benchmark / "assets"):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Check resource paths or run bash scripts/download.sh {task} first.")
    if output is None:
        parent = root / "outputs/quickstart"
        parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output = Path(tempfile.mkdtemp(prefix=f"{task}-{stamp}-", dir=parent))
    else:
        output = _path(output, root)
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
    stages = 4 if render else 3
    print(f"Saving this run to {output}", flush=True)
    try:
        print(f"[1/{stages}] Preparing the tokenizer", flush=True)
        _run([python, "-c", "from openpi.shared import download; "
              "download.maybe_download('gs://big_vision/paligemma_tokenizer.model', "
              "gs={'token': 'anon', 'session_kwargs': {'trust_env': True}})"],
             logs / "tokenizer.log", root, env)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        print(f"[2/{stages}] Loading the NOCS teacher", flush=True)
        with (logs / "teacher.log").open("w") as stream:
            server = subprocess.Popen(cli + ["policy", "serve-sft", "--checkpoint",
                str(teacher / "weights"), "--norm-stats", str(teacher / "norm_stats"),
                "--num-steps", "10", "--image-preprocessing", "none", "--rtc-prefix-schedule",
                "exp", "--device", f"cuda:{gpu}", "--port", str(port)], cwd=root, env=env,
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
        count = (seen_assets + unseen_assets) * states_per_asset
        print(f"[3/{stages}] Generating {count} trajectories of {seconds} seconds", flush=True)
        _run(cli + ["rollout", "run", "--scenario", str(root / "configs/rollout" / scenario),
            "--states", str(states), "--policy", f"ws://127.0.0.1:{port}",
            "--assets", str(benchmark / "assets"), "--gpus", str(gpu), "--batch-size", "1",
            "--seen-assets", str(seen_assets), "--unseen-assets", str(unseen_assets),
            "--limit-per-asset", str(states_per_asset),
            "--execution", "chunk", "--exec-horizon", "10", "--storage-kind", "render_only_v1",
            "--output", str(collection)], logs / "generation.log", root, env)
        _stop(server)
        server = None
        if render:
            print("[4/4] Exporting high and low camera views", flush=True)
            replay(collection, root, benchmark=benchmark, gpu=gpu, logs=logs)
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
