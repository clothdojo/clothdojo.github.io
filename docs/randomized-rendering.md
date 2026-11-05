# Randomized RGB rendering

After watching the [NOCS Quickstart](usage.md) videos, use this separate workflow to render the saved motion with different materials and lighting. The commands below reuse `outputs/first-flatten/rollout`.

## 1. Prepare the renderer

Use an RTX GPU with RT cores and Vulkan support, Python 3.11, and Linux with GLIBC 2.35 or newer. From the checkout root, activate the existing environment and install Isaac Sim 5.0:

```bash
source .venv/bin/activate
python -m pip install --extra-index-url https://pypi.nvidia.com --no-deps \
  'isaacsim==5.0.0' 'isaacsim-app==5.0.0' 'isaacsim-core==5.0.0' \
  'isaacsim-replicator==5.0.0' 'isaacsim-extscache-kit==5.0.0' \
  'isaacsim-extscache-kit-sdk==5.0.0' 'isaacsim-extscache-physics==5.0.0' \
  'isaacsim-kernel==5.0.0' 'omniverse-kit==107.3.1.206797'
```

Keep `--no-deps` to preserve the shared Python dependencies. Download the public Poly Haven materials:

```bash
uvx --from 'huggingface-hub==0.32.3' huggingface-cli download clothdojo/assets \
  --repo-type dataset --revision a0628f42dde13ebb0505fd9dc255d4bda50edb94 \
  --include 'data/visual-assets/polyhaven/*' --local-dir .
```

This example varies cloth, table and ground materials and lighting, with a fixed background. No additional NVIDIA scene assets are needed.

## 2. Render the saved trajectory

```bash
clothdojo data export-lerobot \
  --source outputs/first-flatten/rollout \
  --output outputs/rgb-seed42 \
  --assets benchmark/assets \
  --isaac-seed 42 --randomize all
```

This renders the recorded motion on GPU 0 without running the teacher or simulating new actions. The first RTX stage and shader setup can take several minutes.

## 3. Watch both RGB views

Open these MP4s in your video player:

```text
outputs/rgb-seed42/videos/chunk-000/
  observation.images.cam_high/episode_000000.mp4
  observation.images.cam_low/episode_000000.mp4
```

The full flattening videos remain 60 seconds long, now at the recorded control rate of 50 fps. Two wrist-camera videos and accompanying LeRobot v2.1 data are also exported.

To compare another appearance, rerun the command with `--isaac-seed 43` and `--output outputs/rgb-seed43`. The same recorded pose rendered with two seeds looks like this:

![High and low RGB views of the same recorded pose with seeds 42 and 43](assets/randomized-rgb-preview.jpg)

Use a new output directory for each export. For folding, use the saved folding collection as `--source`; the other steps are unchanged.

## Appearance controls

| Option | Purpose |
|---|---|
| `--isaac-seed` | Select another appearance |
| `--randomize cloth,lighting` | Choose which components vary between trajectories in one export; also accepts `table` and `ground` |
| `--device cuda:1` | Render on another visible GPU |
| `--visual-assets /path/to/visual-assets` | Use visual resources stored elsewhere |

A layout stays fixed throughout each trajectory. The same trajectory and base seed select the same layout. `--randomize` chooses which components vary between trajectories within one export; changing the base seed can also change components shared across trajectories.

The sampled layouts are recorded in the export's `meta/clothdojo.json`. Basic rendering has a fixed background; background variation requires room assets or a full appearance profile.

## Training data and full appearance profiles

Use [success-selected trajectories](advanced.md#generate-successful-trajectories) before preparing training data. Rendering the unfiltered Quickstart example does not make it a successful training trajectory.

For the full appearance profiles and their upstream backgrounds/materials, see the [NVIDIA asset guide](upstream-dependencies.md#nvidia-visual-assets-for-rgb).
