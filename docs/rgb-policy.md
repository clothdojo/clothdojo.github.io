# Train and evaluate an RGB policy

Download Flatten or Fold demonstrations, train a Diffusion Policy, and evaluate it in
ClothDojo. Code comes from this Git checkout; Hugging Face supplies data and assets.

Use Linux, Python 3.11 and two available NVIDIA GPUs. Evaluation uses GPU 0 for
the policy and an Isaac-compatible RTX GPU 1 for simulation and RGB observations.
Install the environment once with `bash scripts/setup.sh --rgb --xpolicy`, then activate
it with `source .venv/bin/activate`. Reserve GPUs on shared machines before use.

## 1. Download the data

```bash
python scripts/rgb-policy.py download --task flatten
```

Use `--task fold` in all three commands for [Fold](https://huggingface.co/datasets/clothdojo/fold).
The commands shown select [Flatten](https://huggingface.co/datasets/clothdojo/flatten).
This downloads the complete selected LeRobot dataset and evaluation assets,
using pinned revisions. Keep its `outputs/rgb-flatten/` directory
to reuse downloads after interruption.

Downloaded visuals are linked into the renderer's standard `data/visual-assets`
directory without overwriting existing files. Evaluation also needs
the excluded NVIDIA background/material dependencies described in
[upstream dependencies](upstream-dependencies.md) and
`outputs/rgb-flatten/assets/upstream_visual_dependencies.json` (or the Fold equivalent).
The fixed-background RGB rendering example alone does not supply these scenes.

## 2. Train the policy

```bash
python scripts/rgb-policy.py train --task flatten
```

This uses the existing LeRobot Diffusion Policy trainer on GPU 0, performs 80,000
updates, and saves `outputs/rgb-flatten/train/step_080000.pt` (or the Fold equivalent) with its
normalization and EMA weights. It does not download the NOCS teacher or a VLA
base model.
The example uses the high camera and both wrist cameras; the dataset also supplies
the low camera for other uses. States and absolute joint/gripper targets are 14-dimensional.

## 3. Run evaluation

```bash
python scripts/rgb-policy.py evaluate --task flatten
```

The command checks upstream visual hashes before starting GPU processes.
The existing evaluator starts the policy server, runs the selected task's complete
repaired RGB initial-state panel, saves `render_only_v1` trajectories, scores their
geometry offline, and stops its processes. Each episode runs for 60 seconds in
Flatten or 40 seconds in Fold. Results are written under
`outputs/rgb-<task>/evaluations/<model>-<checkpoint-hash>-<code-hash>/`.
Training RGB videos are never used as geometry scoring inputs.

## Dataset size and configuration

The commands above use the full dataset and benchmark. Each task has its own
directory: `outputs/rgb-flatten/` or `outputs/rgb-fold/`.
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

## Optional: quickly check the workflow

Add `--smoke` to all three commands to download one complete training episode,
train for two updates and evaluate one initial state. For example:

```bash
python scripts/rgb-policy.py download --task fold --smoke
python scripts/rgb-policy.py train --task fold --smoke
python scripts/rgb-policy.py evaluate --task fold --smoke
```

Use `--task flatten` in all three commands for Flatten. Smoke runs use separate
`outputs/rgb-<task>-smoke/` directories and save `train/step_000002.pt`. They retain
the task's full episode duration and scoring criteria. Two updates only check the
workflow; they do not produce a useful policy, and smoke results are not benchmark
results. Use the commands above without `--smoke` for full training and evaluation.
