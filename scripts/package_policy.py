"""Package a legacy SFT checkpoint after checking its normalization binding."""

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--norm-stats", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--initial-states", type=Path)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--task", required=True, choices=("shirt_flatten", "shirt_fold"))
    parser.add_argument("--cloth-visual", required=True, choices=("nocs", "white", "rgb"))
    parser.add_argument("--image-preprocessing", required=True,
                        choices=("none", "training_fixed_crop"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    checkpoint = args.checkpoint.resolve()
    launch = json.loads((checkpoint / "clothdojo_launch.json").read_text())
    provenance = json.loads((args.norm_stats / "clothdojo.json").read_text())
    if launch.get("schema") != "clothdojo_pi05_launch_preflight" or launch.get("compatible") is not True:
        parser.error("expected a validated native SFT launch manifest")
    stats = args.norm_stats / "norm_stats.json"
    for field, path in (("stats_sha256", stats),
                        ("provenance_sha256", args.norm_stats / "clothdojo.json")):
        if sha256(path) != launch["norm_stats"][field]:
            parser.error(f"normalization binding differs: {field}")
    if (provenance.get("schema") != "clothdojo_openpi_norm_stats"
            or provenance.get("config_name") != "pi05_aloha_robotwin"
            or provenance.get("action_horizon") != 50
            or provenance.get("data_kwargs", {}).get("adapt_to_pi") is not False
            or provenance.get("data_kwargs", {}).get("extra_delta_transform") is not True):
        parser.error("unsupported normalization contract")
    weights = checkpoint / "model_state_dict/full_weights.pt"
    if sha256(weights) != args.weights_sha256:
        parser.error("weights differ from the expected SHA-256")
    layout = json.loads((checkpoint / "clothdojo_weights.json").read_text())
    ema = {"decay": 0.99, "path": "model_state_dict/full_weights.pt",
           "purpose": "inference and rollout"}
    if (layout.get("schema") != "clothdojo_openpi_rlinf_checkpoint"
            or layout.get("format_version") != 1 or layout.get("ema") != ema):
        parser.error("expected native SFT wrapper weights")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".policy-", dir=args.output.parent))
    try:
        (staging / "weights/model_state_dict").mkdir(parents=True)
        (staging / "cache/big_vision").mkdir(parents=True)
        shutil.copyfile(weights, staging / "weights/model_state_dict/full_weights.pt")
        # Preserve only the loader's layout tag; omit legacy paths and launch metadata.
        (staging / "weights/clothdojo_weights.json").write_text(json.dumps({
            "schema": "clothdojo_openpi_rlinf_checkpoint", "format_version": 1,
            "ema": ema,
        }) + "\n")
        shutil.copyfile(stats, staging / "norm_stats.json")
        shutil.copyfile(args.tokenizer, staging / "cache/big_vision/paligemma_tokenizer.model")
        metadata = {
            "schema": "clothdojo_pi05_policy_v1", "task": args.task,
            "cloth_visual": args.cloth_visual, "image_preprocessing": args.image_preprocessing,
            "weights_sha256": args.weights_sha256, "norm_stats_sha256": sha256(stats),
            "tokenizer_sha256": sha256(args.tokenizer),
        }
        if args.initial_states is not None:
            from clothdojo.rollout.initial_states import InitialStatePool

            pool = InitialStatePool.load(args.initial_states)
            pool.metadata.pop("garment_path", None)
            pool.save(staging / "initial_states.npz")
            metadata["initial_states_sha256"] = sha256(staging / "initial_states.npz")
        (staging / "policy.json").write_text(json.dumps(metadata, indent=2) + "\n")
        staging.rename(args.output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(args.output)


if __name__ == "__main__":
    main()
