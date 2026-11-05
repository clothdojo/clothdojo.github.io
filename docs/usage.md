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

Run from the checkout root and activate the installed environment:

```bash
source .venv/bin/activate
clothdojo generate --config configs/generation/flatten.yaml
```

`generate` starts the teacher, runs simulation and recording, then stops the teacher. It prints the saved collection path and a ready-to-copy replay command. For folding, use `configs/generation/fold.yaml` after downloading its teacher.

### Choose what to generate

The generation YAML contains the choices for this run:

```yaml
task: flatten
benchmark: benchmark
teachers: models/nocs-teachers
output: null
gpu: 0
seen_assets: 1
unseen_assets: 0
states_per_asset: 1
```

| Setting | When to change it |
|---|---|
| `task` | Choose `flatten` or `fold` |
| `benchmark` | The released benchmark folder is on another disk |
| `teachers` | The teacher root containing `flatten/` and `fold/` is elsewhere |
| `output` | Use a specific new run directory; `null` creates a separate directory automatically |
| `gpu` | Choose another visible CUDA GPU; policy, simulation and replay can share one GPU |
| `seen_assets`, `unseen_assets` | Choose how many garments from each released split |
| `states_per_asset` | Choose how many initial states per garment; the released pool has four |

Paths are relative to the checkout root where you run the command, including when the YAML is stored elsewhere. Absolute paths and `~` are supported. The benchmark root supplies assets and task-matched initial states; the teacher root supplies matching weights and normalization. Keep their downloaded layouts intact.

For eight trajectories, set `seen_assets: 2`, `unseen_assets: 2`, and `states_per_asset: 2`. Selection uses the first garments and states in the recorded pool order, not random sampling. Generation uses one simulation world at a time. Task scenarios, inference settings, and action execution parameters keep the tested defaults.

### Replay a saved collection

Copy the replay command printed by `generate`. For example, if you set `output: outputs/my-flatten`:

```bash
clothdojo replay outputs/my-flatten/rollout
```

This exports both cameras without starting a teacher or generating new actions. With external assets or a different GPU, the printed command includes `--benchmark` and `--gpu`; these are also available when replaying an older collection on another machine.

The collection contains:

```text
outputs/my-flatten/
  logs/
  rollout/
    meta.json
    trajectories/<trajectory-id>/
      manifest.json        # Task, asset, timing and provenance
      actions.npz         # Arm targets and gripper commands
      states.npz          # Recorded robot and cloth poses
      selection_signals.npz
    review/videos/cam_high/
    review-low/videos/cam_low/
```

Generation records an unfiltered trajectory; it does not guarantee task success. See [success selection](advanced.md#generate-successful-trajectories) when collecting training data.

Use a new output directory for a new generation run; existing directories are never overwritten. Replaying the same collection reuses matching videos. To manage the policy server and simulator separately, see the [individual commands](advanced.md#run-the-individual-steps).

## 4. Read a trajectory in Python

Use `TrajectoryReader` to inspect or analyze generated data:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from clothdojo.trajectory import TrajectoryReader

collection = Path("outputs/my-flatten/rollout")
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
