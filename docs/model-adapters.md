# Evaluate your model

Run from this Git checkout, using the same repaired RGB states and scoring protocol
as the other policies. The model never supplies initial states or success labels.

## A supported checkpoint

Install the optional serving runtime once:

```bash
bash scripts/setup.sh --rgb --xpolicy
source .venv/bin/activate
```

The verified built-in model is `diffusion`: a self-contained checkpoint produced
by the ClothDojo Diffusion trainer. Its architecture, normalization, EMA weights
and execution horizon are read from the checkpoint. No model code edits or
separate normalization path are required.

To evaluate an existing checkpoint without downloading training demonstrations:

```bash
python scripts/rgb-policy.py download --task flatten --assets-only --smoke
python scripts/rgb-policy.py evaluate --task flatten --smoke --checkpoint /path/to/step_080000.pt
```

Complete the [upstream visual asset preparation](upstream-dependencies.md#nvidia-visual-assets-for-rgb)
before evaluation. The command verifies these asset hashes before allocating GPU
processes. Select `--task fold` in both commands for Fold. Omit `--smoke` in both
commands for the selected task's complete 1,600-case RGB panel. The smoke run
selects one case while preserving the task protocol: Flatten 60 seconds, Fold
40 seconds, each with its own scorer. Smoke results are not benchmark results.

Evaluation owns and stops the policy server and simulator. Results live under
`outputs/rgb-<task>[-smoke]/evaluations/<model>-<checkpoint-hash>-<code-hash>/`.
Use `--check` to validate inputs without running GPU work. Use `--output` for a
fresh result directory when changing an external adapter's code; incompatible
results are rejected rather than overwritten. GPU assignments remain in the
generated `deployment.json` (policy 0, RTX simulation 1 by default).

## A new model: edit your adapter file

```bash
cp examples/policy_adapter.py my_policy.py
```

Change these parts in `my_policy.py`:

1. `__init__`: load your model and its processor/normalization from `config["checkpoint"]`.
2. `predict`: process the observations and return physical, unnormalized actions.
3. `reset`: clear episode history or recurrent state if your model has it.

Then run:

```bash
python scripts/rgb-policy.py evaluate --task flatten --smoke \
  --model my_policy:Model --checkpoint /path/to/my-checkpoint
```

You do not edit the evaluator, simulator or scorer. The supplied template is a
working DP reference, so its loading and prediction calls can be tested before
you replace them. `examples.policy_adapter:Model` is its importable reference name.

The adapter uses XPolicyLab's standard `update_obs[_batch]`, `get_action[_batch]`,
`prepare_case` and `reset` lifecycle. `ArrayModel` handles the standard named-action
packing; your prediction method returns an array instead of implementing transport.

An existing XPolicyLab model can also be used through `module:Model` without
inheriting `ArrayModel`: its constructor must accept the checkpoint/device config,
it must declare the horizons, and its batch action method must return the standard
Piper joint-action dictionaries. Keep its prediction code and wrap only the
configuration or action conversion that differs. Do not write another server.

| Input / output | Contract |
|---|---|
| `observations` | List of independent environments; preserve list order |
| `vision[cam_high/cam_left_wrist/cam_right_wrist]["color"]` | Latest RGB image, uint8 HWC; never swap RGB/BGR |
| `state` | XPolicyLab named joint/gripper state dictionary |
| `state_history`, `vision[...]["history"]` | One current frame or two adjacent 50 Hz control frames, as declared by `observation_horizon` |
| `instruction`, `env_idx` | Task text and batch-local environment index |
| `predict` output | Finite float32 `[batch, action_horizon, 14]`, absolute targets |
| 14 action entries | Left six joints in radians, left gripper-open command in [0,1], right six joints, right gripper-open command |

Declare `observation_horizon` (1 or 2) and `action_horizon` on the model. The template
uses the standard 8-action execution chunk. For another chunk size, match the
scenario's action horizon and execution horizon in the generated configuration.
Joint deltas, normalized actions, end-effector poses and torque commands are not
silently converted: adapt them inside your own model file before returning.
Call `super().reset()` when using `ArrayModel`, and use `self.seed` for any
model-owned random generators. Run prediction in inference mode without updating weights.

Native image transport is 640x480. The built-in DP and working template declare
the same trained 224x224 letterbox used by the trainer. A different model can use
native RGB with `transport_image_size = (640, 480)` and perform its own preprocessing.
Keep preprocessing statistics in the checkpoint bundle; its contents and the
adapter source hash are recorded with evaluation results.

## What XPolicyLab supplies

The pinned [XPolicyLab revision](https://github.com/XPolicyLab/XPolicyLab/tree/6baccd57ff958570d0db1588370289ee36c8bb70)
owns the WebSocket client/server, message codec, RGB decoding, lifecycle RPCs and
retry semantics. ClothDojo maps its observations and actions into that contract
and retains the existing rollout/writer/scorer. New models do not need another
socket server or a copied evaluation loop.

This does not assert that every model in the upstream policy zoo is verified on
ClothDojo. Models with other checkpoint formats, dependencies or action spaces
need an adapter and a model environment first. Only registered and tested model
formats are described as checkpoint-only. XPolicyLab serving currently uses
synchronous chunks and one simulation worker per server; that worker may batch
multiple worlds. Separate workers need separate policy servers to keep model
state and episode resets isolated.
