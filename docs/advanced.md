# After your first trajectory

[Project overview](../README.md) · [Trajectory generation](usage.md) · [Downloads and assets](data-and-assets.md)

Run these commands from the GitHub checkout root after completing the quickstart.

## Run the individual steps

Follow the [step-by-step Quickstart](usage.md#3-run-the-steps-yourself) to start a teacher, generate a collection, and replay both cameras. It also explains garment/state selection and reading the resulting arrays in Python.

## Generate successful trajectories

Start the flattening teacher using the Quickstart's policy command. To retain a successful rollout, run a new collection with a success quota and the matching released protocol and templates:

```bash
.venv/bin/clothdojo rollout run \
  --scenario configs/rollout/shirt_flatten.yaml \
  --states benchmark/conditions/nocs-flatten/states \
  --policy ws://127.0.0.1:8080 --assets benchmark/assets \
  --gpus 0 --batch-size 1 --seen-assets 1 --unseen-assets 0 \
  --success-quota 1 --success-initial-attempts 1 \
  --success-protocol benchmark/conditions/nocs-flatten/runtime_protocol.json \
  --success-templates benchmark/conditions/nocs-flatten/templates \
  --execution chunk --exec-horizon 10 --storage-kind render_only_v1 \
  --output outputs/successful-nocs-flatten
```

The quota selects by the task success criterion and publishes compact trajectories. Inspect the collection metadata and selection records for any shortfall before exporting training data. Exhausting the available initial states may leave the quota unmet.

## Folding

Download the fold teacher as described in the [download guide](data-and-assets.md). Stop the flattening server and restart it with both paths changed to `models/nocs-teachers/fold/`.

Use `configs/rollout/shirt_fold_eval_40s.yaml`, `benchmark/conditions/nocs-fold/states`, and a new output directory in the rollout command. For success selection, use `benchmark/conditions/nocs-fold/protocol.json` and `benchmark/conditions/nocs-fold/templates`.

The released Fold generation teacher differs from the historical Fold benchmark policy. Its results must be reported under its own checkpoint identity.

## Evaluate on the released panel

Stop the manually launched policy server first. The managed benchmark starts and owns its own policy service.

Benchmark configurations are included under `configs/quickstart/`.

These configurations reference the released repaired states and matching templates and protocol. They use GPU 0 for inference and GPU 1 for simulation; edit `configs/quickstart/local.yaml` to match your machine.

In a terminal with the tokenizer cache configured:

```bash
export PXR_WORK_THREAD_LIMIT=1
export CLOTHDOJO_OPENPI_CACHE="$PWD/.cache/openpi"
export OPENPI_DATA_HOME="$CLOTHDOJO_OPENPI_CACHE"
.venv/bin/clothdojo benchmark run --config configs/quickstart/nocs-flatten.yaml
```

This runs the full NOCS flattening panel, rather than a quick smoke test. Results are written under `outputs/evaluations/nocs-flatten-teacher/`, including `summary.json` and saved trajectories. For folding:

```bash
.venv/bin/clothdojo benchmark run --config configs/quickstart/nocs-fold.yaml
```

Each NOCS condition has 200 garments and 800 cases. The RGB and NOCS conditions have their own split and case definitions; consult `benchmark/manifest.json`. Keep the released initial states, templates, protocol, observation appearance, and checkpoint identity together. Re-running a condition resumes according to the runtime's identity checks.

The protocol files in `benchmark/conditions/` describe this release. Do not substitute similarly named development configurations or infer paper reproduction from a smoke-test result.

## Export RGB training data

Stop the policy server. Install the Isaac 5 SDK into the same environment and restore the original visual assets using the [upstream dependency guide](upstream-dependencies.md#nvidia-visual-assets-for-rgb). Use an indexed collection of successful trajectories as the source:

```bash
.venv/bin/clothdojo data export-lerobot \
  --source outputs/successful-nocs-flatten \
  --output outputs/rgb-flatten --assets benchmark/assets \
  --visual-assets data/visual-assets --device cuda:0 \
  --isaac-seed 42 --randomize all
```

This performs Isaac RTX re-rendering and exports LeRobot v2.1 data. A single trajectory directory is not an indexed collection. Preserve success selection and dataset provenance when preparing training data.

See the [download guide](data-and-assets.md) for the original RGB training datasets. `configs/training/` contains the generated-data flattening and folding examples and their shared bases. They need a complete dataset, initial weights, normalization statistics, and local paths. Both examples disable gradient accumulation; global batch equals per-GPU micro batch times the data-parallel GPU count. The NOCS quickstart does not depend on those datasets.
