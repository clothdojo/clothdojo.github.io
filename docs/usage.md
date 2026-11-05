# Quickstart

Install ClothDojo, generate a trajectory, and watch the high and low camera videos. Isaac Sim is not required for this NOCS workflow.

You need Linux x86_64, Python 3.11, Git, `patch`, [uv](https://docs.astral.sh/uv/), and an NVIDIA GPU with native BF16 support. Tested on one RTX 5090 D v2 (24 GB).

The first download is about 13.4 GiB, plus space for the environment and caches.

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

The trajectory collection is saved in `outputs/first-flatten/rollout/`; logs are in `outputs/first-flatten/logs/`. Generation records an unfiltered trajectory; task success is assessed separately through [success selection](advanced.md#generate-successful-trajectories).

## 3. Replay and watch both views

```bash
clothdojo replay outputs/first-flatten/rollout
```

Open the two MP4 paths printed by the command in your video player:

```text
outputs/first-flatten/rollout/
  review/videos/cam_high/<trajectory-id>.mp4
  review-low/videos/cam_low/<trajectory-id>.mp4
```

Each video is 60 seconds at 640×480 and 10 fps. Replay uses the saved robot and cloth poses, so it does not run the teacher again.

For another generation run, use a new `--output` directory or omit the option for an automatically created directory. `generate` prints the corresponding replay command, including any custom benchmark path and GPU. Replaying the same collection reuses matching videos.

## Try folding

Use the installed environment:

```bash
bash scripts/download.sh fold
clothdojo generate --task fold --output outputs/first-fold
clothdojo replay outputs/first-fold/rollout
```

This produces a 40-second folding trajectory and its two camera videos.

## Change a run

Add only the options you need:

| Option | Default / purpose |
|---|---|
| `--task` | `flatten`; also accepts `fold` |
| `--benchmark` | `benchmark`; change when the released benchmark folder is elsewhere |
| `--teachers` | `models/nocs-teachers`; root containing `flatten/` and `fold/` |
| `--output` | Automatic new directory; specify a new path for a fixed location |
| `--gpu` | `0`; choose a visible CUDA GPU |
| `--seen-assets`, `--unseen-assets` | `1`, `0`; garment counts from each released split |
| `--states-per-asset` | `1`; initial states per garment, up to four in the released pool |

Paths are relative to the checkout root; absolute paths and `~` are also supported. Keep the downloaded benchmark and teacher folder layouts intact: task-matched initial states, weights and normalization are selected automatically.

For eight trajectories:

```bash
clothdojo generate --seen-assets 2 --unseen-assets 2 --states-per-asset 2
```

Selection uses the first garments and states in the recorded pool order, not random sampling. Generation uses one simulation world at a time. Task scenarios, inference settings, and action execution parameters keep the tested defaults.

Run `clothdojo generate --help` or `clothdojo replay --help` for option details. For separate policy/simulation commands, success selection, evaluation and RGB export, see [more workflows](advanced.md).
