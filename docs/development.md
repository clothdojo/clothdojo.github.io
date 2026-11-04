# Development

Use Python 3.11 and run commands from the repository root. `scripts/setup.sh` installs the pinned shared environment in `.venv/`; Isaac is an optional later installation for RGB export.

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check src tests
```

The tests exercise storage, task scoring, gripper feedback, download boundaries, and quickstart process ownership without starting GPU inference. Real generation requires a compatible NVIDIA GPU.

Use `clothdojo` for individual operations. The quickstart only orchestrates those commands; policy, simulation, scoring, and rendering implementations remain in their existing modules.

Keep executable code in GitHub. Downloaded `benchmark/`, `models/`, `data/`, and generated `outputs/` are ignored. Small numerical regression fixtures live under `tests/fixtures/` and are not production simulation assets.

[Project index](project_index.md) · [Quickstart](usage.md) · [Advanced workflows](advanced.md)
