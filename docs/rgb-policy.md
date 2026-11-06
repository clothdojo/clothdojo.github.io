# Train and evaluate an RGB policy

Download Flatten or Fold demonstrations, train a Diffusion Policy, and evaluate it in
ClothDojo. Code comes from this Git checkout; Hugging Face supplies data and assets.

Use Linux, Python 3.11 and two available NVIDIA GPUs. Evaluation uses GPU 0 for
the policy and an Isaac-compatible RTX GPU 1 for simulation and RGB observations.
Install the environment once with `bash scripts/setup.sh --rgb --xpolicy`, then activate
it with `source .venv/bin/activate`. Reserve GPUs on shared machines before use.

## 1. Download the data

```bash
python scripts/rgb-policy.py download --task flatten --smoke
```

Use `--task fold` in all three commands for [Fold](https://huggingface.co/datasets/clothdojo/fold).
The example below selects [Flatten](https://huggingface.co/datasets/clothdojo/flatten).
This downloads one complete episode from the selected public LeRobot dataset
and the evaluation assets, using pinned revisions. The smoke fixture is separate
from the full training dataset. Keep its `outputs/rgb-flatten-smoke/` directory
to reuse downloads after interruption.

Downloaded visuals are linked into the renderer's standard `data/visual-assets`
directory without overwriting existing files. Evaluation, including this smoke run, also needs
the excluded NVIDIA background/material dependencies described in
[upstream dependencies](upstream-dependencies.md) and
`outputs/rgb-flatten-smoke/assets/upstream_visual_dependencies.json`.
The fixed-background RGB rendering example alone does not supply these scenes.

## 2. Train the policy

```bash
python scripts/rgb-policy.py train --task flatten --smoke
```

This uses the existing LeRobot Diffusion Policy trainer on GPU 0, performs two
updates, and saves `outputs/rgb-flatten-smoke/train/step_000002.pt` with its
normalization and EMA weights. It does not download the NOCS teacher or a VLA
base model. Two updates check the connection; they do not produce a useful policy.
The example uses the high camera and both wrist cameras; the dataset also supplies
the low camera for other uses. States and absolute joint/gripper targets are 14-dimensional.

## 3. Run evaluation

```bash
python scripts/rgb-policy.py evaluate --task flatten --smoke
```

The command checks upstream visual hashes before starting GPU processes.
The existing evaluator then starts the policy server, loads one repaired RGB initial
state, runs the task's simulated episode (Flatten: 60 seconds; Fold: 40 seconds), saves a `render_only_v1` trajectory,
scores its geometry offline, and stops its processes. Results are written under
`outputs/rgb-flatten-smoke/evaluations/<model>-<checkpoint-hash>-<code-hash>/`. Smoke results are not official
benchmark results. Training RGB videos are never used as geometry scoring inputs.

## Use the full dataset and benchmark

Omit `--smoke` in all three commands. Each task has its own directory:
`outputs/rgb-flatten/` or `outputs/rgb-fold/` (with `-smoke` for smoke runs).
Do not share a work directory between tasks.

| Task | LeRobot episodes | Dataset download | Evaluation duration |
|---|---:|---:|---:|
| Flatten | 3,992 | 196 GB | 60 seconds |
| Fold | 3,856 | 272 GB | 40 seconds |

Allow additional space for assets,
the environment, checkpoints, caches and evaluation trajectories.

The example recipe uses 80,000 steps and a single GPU with micro batch 8, global
batch 8 and no gradient accumulation. It is an example, not the paper's RGB
training recipe or a promised benchmark score. Edit the generated `train.yaml`
and `deployment.json` for your hardware; global batch must equal micro batch
times the number of training GPUs. Existing checkpoints require explicit resume
through the existing trainer.

Full evaluation preserves the released RGB panel: 200 garments, four physical
states each, two appearances (1,600 cases per task).
Flatten and Fold retain their own released scoring protocols; Fold uses fold quality.
It runs four simulation worlds per batch; edit `evaluate.json` and `deployment.json`
to match the available evaluation hardware.
RGB and NOCS seen/unseen definitions remain distinct. The download does not
regenerate initial states or relax the benchmark checks.

Add `--check` to `train` or `evaluate` to inspect configuration without GPU work.
Use `--work <directory>` in all three commands to select another output location.

For an existing checkpoint or your own model, follow [model adapters](model-adapters.md).
Evaluation-only users can use `download --assets-only`; training data is optional.
