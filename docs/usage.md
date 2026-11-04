# Quickstart

Generate a NOCS trajectory and watch it from two camera views on one GPU. Isaac Sim is not required.

You need Linux x86_64, Python 3.11, Git, `patch`, [uv](https://docs.astral.sh/uv/), and an NVIDIA GPU with native BF16 support. Tested on one RTX 5090 D v2 (24 GB).

The first download is about 13.4 GiB, plus space for the environment and caches.

## 1. Download and install

From the [GitHub checkout](../README.md#getting-started):

```bash
bash scripts/download.sh flatten
bash scripts/setup.sh
```

The download script fetches only benchmark assets, initial states, templates, licenses, and the selected teacher. It does not download or replace source code. The install script creates `.venv/` and installs the pinned Python dependencies without the Isaac SDK.

## 2. Generate and watch

```bash
bash scripts/quickstart.sh flatten
```

This prepares the upstream tokenizer, starts the teacher, generates one 60-second trajectory, stops the teacher, and exports both `cam_high` and `cam_low` videos. Everything runs in one terminal. The first run also compiles CUDA kernels.

The script prints the trajectory directory and both MP4 paths. Each video is 640×480 at 10 fps. Runs are saved separately under `outputs/quickstart/`; rerunning the command does not overwrite an earlier result. Logs are kept with each run. Ctrl+C stops the processes started by the script and keeps partial output.

[View an example replay](assets/nocs-flatten.mp4).

## Try folding

Use the same environment:

```bash
bash scripts/download.sh fold
bash scripts/quickstart.sh fold
```

This generates a 40-second folding trajectory and exports the same two camera views. Each teacher uses its own weights and normalization statistics.

See [more workflows](advanced.md) for individual CLI commands, success selection, evaluation, and RGB export, or [download details](data-and-assets.md) for the asset layout.
