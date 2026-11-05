# Project index

| Path | Purpose |
|---|---|
| `scripts/download.sh` | Download assets and the selected teacher; never download code |
| `scripts/setup.sh` | Create the NOCS environment; `--rgb` adds Isaac 5 in the same `.venv` |
| `scripts/quickstart.sh` | Generate one trajectory and both review views |
| `docs/usage.md` | NOCS quickstart: install, generate a trajectory, and watch both camera views |
| `docs/randomized-rendering.md` | Separate RGB workflow: renderer setup, materials, randomized export and videos |
| `src/clothdojo/quickstart.py` | Own the quickstart's child processes and logs; delegate work to the CLI |
| `src/clothdojo/cli.py` | Individual generation, evaluation, export, and training commands |
| `src/clothdojo/rollout/` | Policy inference and trajectory collection |
| `src/clothdojo/sim/` | Newton scene and simulation |
| `src/clothdojo/trajectory/` | Shared trajectory storage and readback |
| `src/clothdojo/evaluation/` | Task protocols and scores |
| `src/clothdojo/data/` | Review rendering, data transforms, and RGB export |
| `src/clothdojo/training/` | Training workflows |
| `configs/rollout/` | Task scenarios |
| `configs/quickstart/` | Released-panel benchmark configurations |
| `configs/training/` | Training configurations; set local dataset and model paths |
| `tests/` | Regression tests and their small numerical fixtures |

GitHub owns executable code and configuration. Hugging Face owns production assets, evaluation states/templates, and teacher weights. Downloaded directories and generated outputs are ignored by Git.

[Quickstart](usage.md) · [Advanced commands](advanced.md) · [Downloads](data-and-assets.md)
