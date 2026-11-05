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

## 2. Run a first example

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

## 3. Run the steps yourself

The quickstart connects three parts of ClothDojo:

| Part | Input → output | Code |
|---|---|---|
| Policy server | NOCS observations and robot state → action chunks | [`rollout/policy.py`](../src/clothdojo/rollout/policy.py) |
| Simulator and recorder | Initial state and policy actions → saved trajectory | [`rollout/`](../src/clothdojo/rollout/) |
| Replay | Saved robot and cloth poses → camera videos | [`data/rollout_review.py`](../src/clothdojo/data/rollout_review.py) |

Use the following commands after the first example. Run them from the checkout root. These are the same generation and replay steps used by `scripts/quickstart.sh`.

### Start the policy

In terminal A:

```bash
export PXR_WORK_THREAD_LIMIT=1
export OPENPI_DATA_HOME="$PWD/.cache/openpi"
export CLOTHDOJO_OPENPI_CACHE="$OPENPI_DATA_HOME"
.venv/bin/clothdojo policy serve-sft \
  --checkpoint models/nocs-teachers/flatten/weights \
  --norm-stats models/nocs-teachers/flatten/norm_stats \
  --num-steps 10 --image-preprocessing none --rtc-prefix-schedule exp \
  --device cuda:0 --port 8080
```

Wait for `server listening`. Keep this terminal running. Weights and normalization must belong to the same teacher.

### Generate a trajectory

In terminal B:

```bash
export PXR_WORK_THREAD_LIMIT=1
.venv/bin/clothdojo rollout run \
  --scenario configs/rollout/shirt_flatten.yaml \
  --states benchmark/conditions/nocs-flatten/states \
  --policy ws://127.0.0.1:8080 --assets benchmark/assets \
  --gpus 0 --batch-size 1 \
  --seen-assets 1 --unseen-assets 0 --limit-per-asset 1 \
  --execution chunk --exec-horizon 10 --storage-kind render_only_v1 \
  --output outputs/my-flatten
```

`--scenario` defines the task, prompt, duration and seed. `--states` supplies the released initial-state pool; this command does not generate a new initial state. `--policy` connects the simulator to terminal A.

This selects the first seen garment and its first saved initial state, producing one trajectory. To collect eight trajectories, change `--seen-assets 2 --unseen-assets 2 --limit-per-asset 2` and use a new output directory. Selection follows the pool's recorded order; these options do not randomly sample garments or states. Keep `--batch-size 1` initially; it controls simultaneous simulation worlds, not the total number of trajectories.

The collection contains:

```text
outputs/my-flatten/
  meta.json
  trajectories/<trajectory-id>/
    manifest.json          # Task, asset, timing and provenance
    actions.npz           # Arm targets and gripper commands
    states.npz            # Recorded robot and cloth poses
    selection_signals.npz
```

Generation records an unfiltered trajectory; it does not guarantee task success. See [success selection](advanced.md#generate-successful-trajectories) when collecting training data.

### Replay both cameras

After generation finishes, stop the policy in terminal A with Ctrl+C to free GPU memory. In terminal B:

```bash
for camera in cam_high cam_low; do
  .venv/bin/clothdojo data render-review \
    --source outputs/my-flatten --assets benchmark/assets \
    --device cuda:0 --camera "$camera" --fps 10 \
    --output "outputs/my-flatten/review-$camera"
done
```

Videos are saved to `outputs/my-flatten/review-<camera>/videos/<camera>/`. Replay reads recorded poses; it does not run the policy again. You can use the same command on an earlier quickstart result by replacing `--source` with its printed trajectory collection path.

## 4. Read a trajectory in Python

Use `TrajectoryReader` to inspect or analyze generated data:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from clothdojo.trajectory import TrajectoryReader

collection = Path("outputs/my-flatten")
trajectory = sorted((collection / "trajectories").iterdir())[0]
reader = TrajectoryReader(trajectory)

states = reader.state_arrays(["joint_q", "particle_q", "gripper_open_state"])
actions = reader.action_arrays()
print("States / actions:", reader.manifest.state_count, reader.manifest.action_count)
print("Cloth positions:", states["particle_q"].shape)
print("Arm targets:", actions["joint_target"].shape)
print("Gripper commands:", actions["gripper_open_command"].shape)
PY
```

Arrays are indexed by time. Action `i` is applied at state `i` and leads to state `i + 1`, so a trajectory has one more state than action. Arm targets are in radians; gripper commands are normalized openness in `[0, 1]`.

The compact `render_only_v1` format retains `joint_q`, `body_q`, `particle_q`, and `gripper_open_state`. Request retained fields explicitly, as above. It supports replay and geometric analysis; it omits internal dynamics and grasp state needed for a complete simulation restart.

For code changes, start with the [module index](project_index.md). For success selection, policy evaluation and RGB export, continue to [more workflows](advanced.md).
