# RGB rendering reference

The [Quickstart](usage.md#4-prepare-rgb-rendering) contains the complete installation, material download, rendering commands and video paths. This page describes the optional appearance controls.

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
