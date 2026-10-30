# ClothDojo: Learned Data Generation and Benchmarking for Bimanual Garment Flattening and Folding

ClothDojo is a simulation benchmark for bimanual garment **flattening** and **folding**. It provides fixed initial states, task-specific success criteria, and a common evaluation interface for comparing policies. The repository also includes tools for generating successful trajectories and randomized RGB training data.

[Project website](https://clothdojo.github.io/) · [Source code](https://github.com/clothdojo/clothdojo.github.io)

The project page lives in `website/`; GitHub Pages publishes that directory only. Code, benchmark fixtures, tests, and the website share this repository.

## Contents

- [1. Benchmark](#1-benchmark)
- [2. Installation](#2-installation)
- [3. Policy evaluation](#3-policy-evaluation)
- [4. Data generation](#4-data-generation)
- [5. Datasets](#5-datasets)
- [6. RGB policy training](#6-rgb-policy-training)
- [7. License and attribution](#7-license-and-attribution)

## 1. Benchmark

### 1.1 Tasks and protocol

| Task | Horizon | Success criterion |
| --- | ---: | --- |
| Flattening | 60 s | Cloth shape and orientation remain within the geometric thresholds for 1 s. |
| Folding | 40 s | Fold quality remains above the threshold for 3 s. |

The exact thresholds and timing are defined in `configs/evaluation/shirt_flatten.yaml` and `configs/evaluation/shirt_fold.yaml`. Evaluation records the model identifier, initial-state hash, protocol fingerprint, per-episode outcomes, and success rate. Use the same fixed states and protocol when comparing models.

The prepared full evaluation bundle covers 200 garments per task in both NOCS and RGB conditions (100 seen and 100 unseen). Its download and multi-garment runner are not yet public.

### 1.2 Resources

| Resource | Purpose | Availability |
| --- | --- | --- |
| [Code and one garment](assets/garments/converted/rand_ZZCWITH5HY/) | Run a small integration example | Included |
| [Example initial states](scripts/data/initial_states/) | Check flattening and folding on one garment | Included; not the paper evaluation set |
| Flattening and folding reference policies | Generate successful trajectories; place bundles in `models/flatten/` and `models/fold/` | 🔴 Pending release; checkpoints identified |
| Generated RGB flattening and folding datasets | Train RGB policies; place datasets in `data/generated-rgb/shirt_flatten/` and `data/generated-rgb/shirt_fold/` | 🔴 Pending release |
| Full evaluation bundle | Fixed garment assets, templates, initial states, and checksums | 🔴 Pending release; bundle prepared |
| Isaac visual materials | RGB export and online RGB evaluation | 🔴 Pending release |

🔴 indicates that a public download is not yet available.

This is a code preview. Public download links and end-to-end GPU validation are pending. The OpenXR teleoperation interface is included, but headset connection has not yet been validated.

## 2. Installation

Use Linux x86_64 with Python 3.11, a CUDA GPU, and [uv](https://docs.astral.sh/uv/). Prepare the main environment once from the repository root:

```bash
git clone https://github.com/clothdojo/clothdojo.github.io.git
cd clothdojo.github.io

# Main environment: Newton simulation, policy serving, training, and LeRobot export.
# The selected XPolicyLab communication runtime is included in this release.
uv sync --frozen --extra sim --extra policy --extra export --extra data
uv pip install --python .venv/bin/python -e XPolicyLab
```

Add the Isaac environment only when you need RGB rendering, RGB export, or online RGB evaluation:

```bash
# Isaac has its own Python dependencies.
uv venv --python 3.11 .venv-isaac
uv pip install --python .venv-isaac/bin/python --index-strategy unsafe-best-match '.[sim]' -r requirements/isaac.txt
uv pip install --python .venv-isaac/bin/python -e XPolicyLab
```

Use `.venv/bin/python` for data generation, training, policy serving, and dataset inspection. Use `.venv-isaac/bin/python` only for RGB rendering and the RGB benchmark client. Isaac rendering also requires the visual materials and system FFmpeg. The two environments are necessary because Isaac 5 and LeRobot require incompatible versions of some Python packages; RGB export passes frames between their processes.

The environment uses Newton for cloth simulation and Piper as the robot embodiment. RGB evaluation uses Isaac for rendering and XPolicyLab for policy serving.

## 3. Policy evaluation

### 3.1 Policy server

Place trained RGB model bundles at `models/flatten-rgb/` and `models/fold-rgb/`, or pass the bundle path printed by `scripts/train.py`. Serve one model at a time on GPU 0; stop the server before switching tasks:

```bash
# Flattening.
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/serve_rlinf_xpolicylab.py \
  --bundle models/flatten-rgb

# Folding: run after stopping the flattening server.
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/serve_rlinf_xpolicylab.py \
  --bundle models/fold-rgb
```

### 3.2 Benchmark client

In another terminal, evaluate the four included starts for the matching task on GPU 1:

```bash
# Flattening: writes outputs/evaluations/shirt_flatten/flatten-rgb/.
.venv-isaac/bin/python scripts/benchmark_xpolicylab.py \
  --policy ClothDojo_RLinf --model-id flatten-rgb \
  --task shirt_flatten --cloth-visual rgb --isaac-gpu 1 --device cuda:1

# Folding: first restart the server with --bundle models/fold-rgb.
# Writes outputs/evaluations/shirt_fold/fold-rgb/.
.venv-isaac/bin/python scripts/benchmark_xpolicylab.py \
  --policy ClothDojo_RLinf --model-id fold-rgb \
  --task shirt_fold --cloth-visual rgb --isaac-gpu 1 --device cuda:1
```

The client defaults to the included task-specific initial states, the matching template, and `assets/visual/` for Isaac materials. Override `--initial-states`, `--template`, `--visual-assets`, or `--output` when using another garment or evaluation set. These four-state pools check integration, not the paper success rate. The full evaluation bundle and multi-garment runner are not yet public; live Isaac RGB evaluation still needs end-to-end GPU validation in this repository.

## 4. Data generation

### 4.1 Flattening

With the reference bundle at `models/flatten/`, generate initial states for the included garment and select one successful trajectory:

```bash
# Writes outputs/initial-states/rand_ZZCWITH5HY-flatten.npz.
.venv/bin/python scripts/generate_initial_states.py --count 16
# Writes outputs/generated/shirt_flatten/selection.json and trajectories/.
.venv/bin/python scripts/generate.py \
  --model models/flatten \
  --initial-states outputs/initial-states/rand_ZZCWITH5HY-flatten.npz \
  --attempts 3
```

### 4.2 Folding

With the reference bundle at `models/fold/`, use the included fixed folding starts to select a successful trajectory:

```bash
# Writes outputs/generated/shirt_fold/selection.json and trajectories/.
.venv/bin/python scripts/generate.py \
  --model models/fold \
  --initial-states scripts/data/initial_states/shirt_fold.npz \
  --attempts 3
```

To create new folding starts from the included garment, run the generator below, then pass its output to `generate.py` with `--initial-states`:

```bash
# Writes outputs/initial-states/rand_ZZCWITH5HY-fold.npz.
.venv/bin/python scripts/generate_fold_initial_states.py
```

Both trajectory-generation commands infer the matching template from the initial-state pool and stop after the first success or three attempts. `selection.json` records successes and any shortfall. The folding initial-state generator still needs a full Newton GPU run. The reference model bundles are not yet public.

### 4.3 RGB export

For either task, use its `selection.json` to render the first selected success and export a trainable LeRobot v2.1 episode. The flattening example is:

```bash
.venv-isaac/bin/python scripts/prepare_visual_assets.py \
  --selection outputs/generated/shirt_flatten/selection.json
.venv-isaac/bin/python scripts/render_rgb.py \
  --selection outputs/generated/shirt_flatten/selection.json \
  --visual-assets assets/visual --output outputs/rgb-datasets/shirt_flatten \
  --export --export-python .venv/bin/python
```

For folding, use the matching selection and output paths:

```bash
.venv-isaac/bin/python scripts/prepare_visual_assets.py \
  --selection outputs/generated/shirt_fold/selection.json
.venv-isaac/bin/python scripts/render_rgb.py \
  --selection outputs/generated/shirt_fold/selection.json \
  --visual-assets assets/visual --output outputs/rgb-datasets/shirt_fold \
  --export --export-python .venv/bin/python
```

The exporter checks the selection record and success score before setting `training_eligible=true`. The released Generated RGB datasets use the same format and also apply trimming and cleaning. Isaac materials remain a separate download.

## 5. Datasets

The first planned training-data release contains the **Generated RGB** flattening and folding datasets. Place them under `data/generated-rgb/` using the task names above. Once a download is available, inspect its LeRobot metadata and camera frames:

```bash
.venv/bin/python scripts/inspect_dataset.py data/generated-rgb/shirt_flatten
# Use data/generated-rgb/shirt_fold for the folding dataset.
```

Dataset and model downloads have no public URLs yet. The RGB training data must have ClothDojo's LeRobot v2.1 sidecar and match the task of the starting model.

## 6. RGB policy training

With a Generated RGB LeRobot dataset and the matching Pi0.5 reference bundle, train each task separately:

```bash
# Flattening.
.venv/bin/python scripts/compute_norm_stats.py data/generated-rgb/shirt_flatten \
  --output outputs/norm-stats/shirt_flatten
.venv/bin/python scripts/train.py data/generated-rgb/shirt_flatten \
  --model models/flatten --norm-stats outputs/norm-stats/shirt_flatten \
  --output outputs/training/shirt_flatten

# Folding: use the corresponding dataset, model, statistics, and output paths.
.venv/bin/python scripts/compute_norm_stats.py data/generated-rgb/shirt_fold \
  --output outputs/norm-stats/shirt_fold
.venv/bin/python scripts/train.py data/generated-rgb/shirt_fold \
  --model models/fold --norm-stats outputs/norm-stats/shirt_fold \
  --output outputs/training/shirt_fold
```

Pass `--check` to validate inputs without starting training. The training command changes the policy's observation type to RGB and prints the saved model-bundle path. This RGB path still needs an end-to-end GPU run with the released dataset; earlier short train/save/load/inference checks used non-RGB data and do not establish RGB model quality.

Default output locations are used only when the target does not already exist. For another run, pass a new `--output` path; the scripts do not overwrite existing results.

## 7. License and attribution

The code is [MIT licensed](LICENSE). The Piper USD files derive from AgileX's [Piper model](https://github.com/agilexrobotics/agx_arm_urdf); its MIT notice is included in [`third_party/agilex/LICENSE`](third_party/agilex/LICENSE). The example garment derives from [GarmentCodeData v2](https://www.research-collection.ethz.ch/handle/20.500.11850/690432) (CC BY 4.0), by Maria Korosteleva, Timur Levent Kesdogan, Fabian Kemper, Stephan Wenninger, Jasmin Koller, Yuhan Zhang, Mario Botsch, and Olga Sorkine-Hornung. ClothDojo modified the converted assets; neither upstream group endorses these modifications.

Zilong Huang, Zipeng Ye, Caicheng Wang, Yuchen Xie, and Jiaxiang Luo (corresponding author).

## Citation and contact

Publication metadata is pending. Until a paper reference is available, cite this code preview without implying peer-reviewed results:

```bibtex
@misc{clothdojo2026,
  title = {ClothDojo: Learned Data Generation and Benchmarking for Bimanual Garment Flattening and Folding},
  author = {Huang, Zilong and Ye, Zipeng and Wang, Caicheng and Xie, Yuchen and Luo, Jiaxiang},
  year = {2026},
  note = {Code preview; publication details pending}
}
```

Report issues at https://github.com/clothdojo/clothdojo.github.io/issues. `maintainers@example.com` is a documentation placeholder, not an operational contact address.

## Release contents

This release includes ClothDojo and a selected XPolicyLab communication runtime. It does not require a full upstream XPolicyLab checkout or include its model subprojects. `integrations/xpolicylab/ClothDojo_RLinf` provides the project adapter. Install the included runtime using the local `XPolicyLab` directory as shown above.

Model bundles, full datasets, the full evaluation bundle and visual material downloads remain pending. Included fixtures are small algorithm examples, not substitutes for released training data. See [data and asset provenance](docs/data-and-assets.md) and [third-party notices](THIRD_PARTY_NOTICES.md).

The included Piper geometry is retained after OpenUSD metadata cleanup and authored/composed equivalence checks. See [USD asset hygiene](docs/usd-assets.md) for the conversion follow-up tool and reproducibility limits.
