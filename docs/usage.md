# Quickstart

Generate a trajectory and watch its high and low NOCS camera videos. This is the shortest path to seeing ClothDojo's actions; Isaac Sim is not required.

You need Linux x86_64, Python 3.11, Git, `patch`, [uv](https://docs.astral.sh/uv/), and an NVIDIA GPU with native BF16 support. Tested on one RTX 5090 D v2 (24 GB).

The first download is about 13.4 GiB, plus space for the environment and caches. Follow the steps in one terminal.

## 1. Download and install

```bash
git clone https://github.com/clothdojo/clothdojo.github.io.git ClothDojo
cd ClothDojo
bash scripts/download.sh flatten
bash scripts/setup.sh
source .venv/bin/activate
```

`download.sh` fetches assets and the selected teacher from Hugging Face. `setup.sh` creates `.venv/` and installs the library and its dependencies. Run the remaining commands from this checkout root.

## 2. Generate a trajectory

```bash
clothdojo generate --output outputs/first-flatten
```

This starts the flattening teacher on GPU 0, simulates one 60-second trajectory, saves it, and stops the teacher. The first run downloads the upstream tokenizer and compiles CUDA kernels, so it takes longer.

The trajectory collection is saved in `outputs/first-flatten/rollout/`; logs are in `outputs/first-flatten/logs/`. This is an unfiltered example, not a success-selected training trajectory.

## 3. Watch both NOCS views

```bash
clothdojo replay outputs/first-flatten/rollout
```

Open the two MP4 paths printed by the command in your video player:

```text
outputs/first-flatten/rollout/
  review/videos/cam_high/<trajectory-id>.mp4
  review-low/videos/cam_low/<trajectory-id>.mp4
```

Each NOCS video is 60 seconds at 640×480 and 10 fps.

You now have a generated trajectory and both NOCS videos. Optional next step: [randomized RGB rendering](randomized-rendering.md), with its own renderer installation and material preparation.

## Try folding

Use the installed environment:

```bash
bash scripts/download.sh fold
clothdojo generate --task fold --output outputs/first-fold
clothdojo replay outputs/first-fold/rollout
```

This produces a 40-second folding trajectory and its two NOCS videos.

## Change a run

Defaults use GPU 0, `benchmark/`, and `models/nocs-teachers/`. Change these only when needed with `--gpu`, `--benchmark`, or `--teachers`.

Paths are relative to the checkout root; absolute paths and `~` are also supported. Keep the downloaded folder layouts intact so task-matched states, weights and normalization can be selected automatically.

For eight trajectories:

```bash
clothdojo generate --seen-assets 2 --unseen-assets 2 --states-per-asset 2
```

Selection follows the recorded pool order, not random sampling. The released pool has four initial states per garment; generation uses one simulation world at a time.

Use a new output directory for each generation. Omit `--output` on `generate` for an automatic run directory; its printed replay command carries the matching path and GPU. Replaying an existing collection reuses matching NOCS videos.

Use `clothdojo generate --help` or `clothdojo replay --help` for all options. See [more workflows](advanced.md) for success selection, evaluation and training-data export.
