# Project index

| Path | Purpose |
|---|---|
| `scripts/download.sh` | Download assets and the selected teacher; never download code |
| `scripts/setup.sh` | Create the base environment; `--rgb` adds Isaac 5, `--xpolicy` adds the policy runtime in the same `.venv` |
| `scripts/quickstart.sh` | Generate one trajectory and both review views |
| `docs/usage.md` | NOCS quickstart: install, generate a trajectory, and watch both camera views |
| `docs/randomized-rendering.md` | Separate RGB workflow: renderer setup, materials, randomized export and videos |
| `docs/rgb-policy.md` | Download public Flatten data, train an RGB policy, and run repaired-state evaluation |
| `scripts/rgb-policy.py` | Three-stage RGB policy example using the existing trainer and managed evaluator |
| `docs/model-adapters.md` | Checkpoint-only evaluation and the new-model adapter contract |
| `examples/policy_adapter.py` | Working model adapter template; change loading, prediction and reset |
| `src/clothdojo/rollout/xpolicy.py` | Observation/action mapping to the pinned official XPolicyLab serving stack |
| `XPolicyLab/` | Optional pinned upstream policy-serving submodule |
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
