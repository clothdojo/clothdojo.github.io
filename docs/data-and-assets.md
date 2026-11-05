# Downloads and assets

[Quickstart](usage.md) · [Upstream dependencies](upstream-dependencies.md)

## Sources

| Location | Contents |
|---|---|
| This GitHub repository | Source code, scripts, configuration, dependencies, and documentation |
| [clothdojo/assets](https://huggingface.co/datasets/clothdojo/assets) | Garment and robot assets, initial states, templates, protocols, redistributable visuals, examples, and component licenses |
| [clothdojo/nocs-teachers](https://huggingface.co/clothdojo/nocs-teachers) | Flattening and folding weights, matching normalization, model cards, provenance, and model terms |

The download script pins assets to `a0628f42dde13ebb0505fd9dc255d4bda50edb94` and teachers to `1d9a6357557df2d722649709738b7a4a62617452`. GitHub is the only code source; Hugging Face supplies assets and weights.

## Layout

Run all commands from the GitHub checkout root:

```text
ClothDojo/
  src/clothdojo/                  # GitHub source
  scripts/                       # Download, installation, and quickstart
  configs/
  .venv/                         # Created by setup.sh
  benchmark/                     # Downloaded assets and evaluation data
    assets/
    manifest.json
    conditions/<condition>/
      states/
      templates/
      protocol.json
  licenses/assets/               # Downloaded component notices and licenses
  models/nocs-teachers/
    flatten/
      policy.json
      weights/
      norm_stats/
    fold/
  outputs/quickstart/             # Generated runs
```

Each NOCS condition has 200 garments and four initial states per garment, split into 100 seen and 100 unseen garments. The NOCS/RGB and flattening/folding conditions retain their own definitions; see `benchmark/manifest.json`.

```bash
bash scripts/download.sh flatten
# Add the other teacher later; existing downloads are reused.
bash scripts/download.sh fold
```

Use `--assets benchmark/assets` with the released initial states. Model weights, normalization, and `policy.json` must come from the same teacher. The released Fold generation teacher differs from the historical Fold benchmark policy.

## Generated files

Each quickstart run contains:

```text
outputs/quickstart/<run>/
  logs/
  rollout/
    meta.json
    trajectories/<trajectory-id>/
      manifest.json
      actions.npz
      states.npz
      selection_signals.npz
    review/videos/cam_high/<trajectory-id>.mp4
    review-low/videos/cam_low/<trajectory-id>.mp4
```

A flattening trajectory has 3000 actions and 3001 states; folding has 2000 actions and 2001 states. Generation and task success are separate: the quickstart records one unfiltered trajectory. Success selection and the benchmark are described in [more workflows](advanced.md).

## Optional RGB resources

The separate [RGB rendering guide](randomized-rendering.md) downloads only the public Poly Haven set. For the full appearance profiles, also obtain the redistributable visual assets and upstream omission manifest:

```bash
uvx --from 'huggingface-hub==0.32.3' huggingface-cli download clothdojo/assets \
  --repo-type dataset --revision a0628f42dde13ebb0505fd9dc255d4bda50edb94 \
  --include 'data/visual-assets/*' 'upstream_visual_dependencies.json' \
  --local-dir .
```

The [upstream guide](upstream-dependencies.md) covers the Isaac 5 SDK and excluded NVIDIA originals. The omission manifest lists 1,572 original asset-tree files and 18 extracted-material files; these are not needed for NOCS generation or the basic Poly Haven rendering example. Downloaded components retain their upstream terms.

The original RGB dataset repositories, [clothdojo/flatten](https://huggingface.co/datasets/clothdojo/flatten) and [clothdojo/fold](https://huggingface.co/datasets/clothdojo/fold), are separate from the assets and teachers. Check their dataset cards for current completeness before training; they are not required for the quickstart.

Code is Apache-2.0. See `licenses/assets/LICENSE_SCOPE.md` and component notices for downloaded assets, and each teacher's Gemma terms, NOTICE, and use restrictions for weights. The code license does not replace those terms.
