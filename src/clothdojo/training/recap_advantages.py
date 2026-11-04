"""RECAP value inference and provenance-bound policy labels for the frozen HIL plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from clothdojo.evaluation.recap import nstep_advantage


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def dataset_roots(project: Path, configured: list[Path] | None = None) -> list[Path]:
    if configured:
        roots = [path if path.is_absolute() else project / path for path in configured]
        missing = [path for path in roots if not path.is_dir()]
        if missing:
            raise FileNotFoundError(f"RECAP dataset roots do not exist: {missing}")
        return [path.resolve() for path in roots]
    root = project / "data/datasets"
    return [*sorted(root.glob("recap_hil800_20260904_expert_shard_*")),
            *sorted(root.glob("recap_hil800_20260904_hil_rollout_shard_*")),
            root / "recap_hil800_20260904_hil_correction"]


def _contract(
    project: Path,
    *,
    policy_plan: Path | None = None,
    configured_roots: list[Path] | None = None,
) -> tuple[dict, list[dict]]:
    plan = (
        policy_plan.resolve()
        if policy_plan is not None
        else project / "outputs/recap/hil800_20260904/policy_plan.json"
    )
    policy = json.loads(plan.read_text())
    datasets = []
    for root in dataset_roots(project, configured_roots):
        recap = json.loads((root / "meta/recap.json").read_text())
        info = json.loads((root / "meta/info.json").read_text())
        role = recap["dataset"]
        expected = {e["source_episode_id"]: e for e in policy["datasets"][role]["episodes"]}
        selected = []
        for ep in recap["episodes"]:
            reference = expected.get(ep["source_episode_id"])
            if reference is None:
                if role != "hil_correction":
                    raise ValueError(
                        f"dataset episode is outside policy plan: {root}:{ep['episode_index']}"
                    )
                continue
            for key in ("asset_id", "action_count", "action_start", "action_end_exclusive"):
                if ep[key] != reference[key]:
                    raise ValueError(f"policy plan differs: {root}:{ep['episode_index']}:{key}")
            selected.append(int(ep["episode_index"]))
        if not selected:
            raise ValueError(f"dataset has no policy episodes: {root}")
        if role != "hil_correction" and len(selected) != len(recap["episodes"]):
            raise ValueError(f"non-correction dataset contains unselected episodes: {root}")
        datasets.append({
            "root": str(root), "source_role": role, "info": info, "recap": recap,
            "selected_episode_indices": selected,
        })
    identities = [
        (dataset["source_role"], episode["source_episode_id"])
        for dataset in datasets
        for episode in dataset["recap"]["episodes"]
        if episode["episode_index"] in dataset["selected_episode_indices"]
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("policy episodes are duplicated across dataset roots")
    return {
        "policy_plan_sha256": _sha(plan),
        "policy_plan_path": str(plan),
        "dataset_roots": [dataset["root"] for dataset in datasets],
    }, datasets


def selection_frame_ids(sources: list[dict], selection: dict) -> np.ndarray:
    """Resolve score-blind clip endpoints into the combined evaluator dataset."""

    clips = selection.get("clips")
    if not isinstance(clips, list) or not clips:
        raise ValueError("RECAP diagnostic selection must contain clips")
    episodes: dict[tuple[str, str], tuple[int, dict]] = {}
    dataset_offset = 0
    for source in sources:
        if source["source_role"] == "hil_correction":
            continue
        frame_offset = 0
        for episode in source["recap"]["episodes"]:
            key = (Path(source["root"]).name, episode["source_episode_id"])
            if key in episodes:
                raise ValueError(f"duplicate evaluator episode identity: {key}")
            episodes[key] = (dataset_offset + frame_offset, episode)
            frame_offset += episode["action_count"]
        dataset_offset += frame_offset
    ids = []
    for clip in clips:
        root = Path(str(clip.get("dataset_root", ""))).name
        source_id = clip.get("source_episode_id")
        key = (root, source_id)
        if key not in episodes:
            raise ValueError(f"diagnostic clip is outside evaluator data: {key}")
        offset, episode = episodes[key]
        for field in ("episode_index", "action_count", "manifest_sha256"):
            if clip.get(field) != episode.get(field):
                raise ValueError(
                    f"diagnostic clip provenance differs: {clip.get('clip_id')}:{field}"
                )
        anchor = clip.get("anchor")
        horizon = clip.get("horizon", 50)
        if (
            isinstance(anchor, bool)
            or not isinstance(anchor, int)
            or isinstance(horizon, bool)
            or not isinstance(horizon, int)
            or horizon < 1
            or not 0 <= anchor < anchor + horizon < episode["action_count"]
        ):
            raise ValueError(f"invalid diagnostic clip endpoint: {clip.get('clip_id')}")
        ids.extend((offset + anchor, offset + anchor + horizon))
    return np.asarray(sorted(set(ids)), dtype=np.int64)


def infer(
    project: Path,
    output: Path,
    batch_size: int,
    max_batches: int | None,
    *,
    checkpoint: Path | None = None,
    checkpoint_config: Path | None = None,
    checkpoint_sha256: str | None = None,
    selection_path: Path | None = None,
    policy_plan: Path | None = None,
    configured_roots: list[Path] | None = None,
    return_normalization: float = 9000.0,
    returns_tag: str = "fail4500_state_v2",
) -> None:
    import torch
    from omegaconf import OmegaConf
    from rlinf.data.datasets.recap.value_dataset import ValueDataset
    from rlinf.models.embodiment.value_model.recap import get_model
    from rlinf.models.embodiment.value_model.recap.data_collator import ValueDataCollator
    from rlinf.models.embodiment.value_model.recap.processing import ValueProcessor

    from clothdojo.training.recap_value import RECAP_IMAGE_KEY_MAP

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"rank_{rank:02d}.npz"
    if destination.exists():
        raise FileExistsError(destination)
    if not np.isfinite(return_normalization) or return_normalization <= 0:
        raise ValueError("return normalization must be positive and finite")
    contract, sources = _contract(
        project, policy_plan=policy_plan, configured_roots=configured_roots
    )
    value_root = project / "outputs/recap/hil800_20260904/value_r0_16xh100"
    launch_path = (
        checkpoint_config.resolve()
        if checkpoint_config is not None
        else value_root / "clothdojo_launch.json"
    )
    launch = json.loads(launch_path.read_text())
    config = OmegaConf.create(launch["resolved_config"]["actor"]["model"])
    # A staged launch records its initialization checkpoint in model_path.
    # Inference must construct the named backbones and load exactly the requested
    # candidate once, rather than silently loading the Stage-1 checkpoint first.
    config.model_path = None
    weights = (
        checkpoint.resolve()
        if checkpoint is not None
        else value_root
        / "train/checkpoints/global_step_1000/actor/model_state_dict/full_weights.pt"
    )
    expected_sha = checkpoint_sha256 or (
        "f48eba62e8390e86de1817c8baf52151bc1bbaccdcd33e7250ba48eafe094cfa"
        if checkpoint is None
        else _sha(weights)
    )
    if rank == 0 and _sha(weights) != expected_sha:
        raise ValueError("evaluator checkpoint changed")
    # Deliberately use all policy episodes, including the old value validation
    # partition; final policy evaluation uses separate scenarios.
    datasets = [ValueDataset(
        dataset_path=d["root"], robot_type="clothdojo", model_type="pi05",
        action_horizon=1, action_dim=14, default_prompt="flatten the shirt",
        return_min=-return_normalization, return_max=0., normalize_to_minus_one_zero=True,
        tag=returns_tag, image_key_map=RECAP_IMAGE_KEY_MAP,
    ) for d in sources if d["source_role"] != "hil_correction"]
    combined = torch.utils.data.ConcatDataset(datasets)
    selection_sha256 = None
    if selection_path is not None:
        selection_path = selection_path.resolve()
        selection = json.loads(selection_path.read_text())
        selected = selection_frame_ids(sources, selection)
        selection_sha256 = _sha(selection_path)
        ids = selected[rank::world]
    else:
        ids = np.arange(rank, len(combined), world, dtype=np.int64)
    if max_batches is not None:
        ids = ids[:max_batches * batch_size]
    if not len(ids):
        raise ValueError(f"rank {rank} has no evaluator inference rows")
    processor = ValueProcessor(
        max_token_len=128,
        tokenizer_name_or_path=config.tokenizer_path,
        do_augment=False,
        image_keys=tuple(RECAP_IMAGE_KEY_MAP),
    )
    collator = ValueDataCollator(processor=processor, max_length=128, train=False)
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(combined, ids.tolist()),
                                        batch_size=batch_size, collate_fn=collator, num_workers=4,
                                        multiprocessing_context="spawn", persistent_workers=True,
                                        pin_memory=True, prefetch_factor=2)
    model = get_model(config)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    model.cuda().eval()
    for parameter in model.parameters():
        parameter.data = parameter.data.bfloat16()

    def cuda(value):
        if isinstance(value, torch.Tensor):
            return value.cuda()
        if isinstance(value, dict):
            return {key: cuda(item) for key, item in value.items()}
        return value

    predictions = []
    with torch.inference_mode():
        for number, batch in enumerate(loader):
            observation = batch["observation"]
            observation.pop("action_mask", None)
            values = model(observation=cuda(observation)).predicted_values.float().cpu().numpy()
            if not np.isfinite(values).all() or (values < -1.001).any() or (values > 0.001).any():
                raise RuntimeError("invalid evaluator output")
            predictions.append(values)
            if number % 100 == 0:
                print(json.dumps({"rank": rank, "batch": number, "batches": len(loader)}),
                      flush=True)
    values = np.concatenate(predictions)
    temporary = destination.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, sample_index=ids, value=values)
    temporary.replace(destination)
    _write(destination.with_suffix(".json"), {
        **contract, "rank": rank, "world_size": world, "rows": len(ids),
        "total_frames": len(combined),
        "return_normalization": return_normalization, "returns_tag": returns_tag,
        "coverage": "selection" if selection_path is not None else "full",
        "selection_sha256": selection_sha256,
        "partial_smoke": max_batches is not None or selection_path is not None,
        "checkpoint_weights_sha256": expected_sha, "checkpoint_config_sha256": _sha(launch_path),
        "predictions_sha256": _sha(destination),
        "source_sha256": _sha(Path(__file__)),
    })


def finalize(
    project: Path,
    output: Path,
    horizon: int,
    fraction: float,
    tag: str,
    *,
    policy_plan: Path | None = None,
    configured_roots: list[Path] | None = None,
    return_normalization: float = 9000.0,
    returns_tag: str = "fail4500_state_v2",
    failure_penalty: float = 4500.0,
    threshold_role: str = "all",
    forced_positive_roles: tuple[str, ...] = ("hil_correction",),
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from clothdojo.training.steam_advantages import _advantage_tag

    tag = _advantage_tag(tag)
    supported_forced_roles = {"expert", "hil_correction"}
    forced_positive_roles = tuple(sorted(set(forced_positive_roles)))
    if (
        horizon < 1
        or not 0 < fraction < 1
        or not np.isfinite(return_normalization)
        or return_normalization <= 0
        or not np.isfinite(failure_penalty)
        or not 0 <= failure_penalty <= return_normalization
        or threshold_role not in {"all", "hil_rollout"}
        or not set(forced_positive_roles) <= supported_forced_roles
        or "hil_correction" not in forced_positive_roles
    ):
        raise ValueError("invalid horizon, fraction, label routing, or return contract")
    contract, datasets = _contract(
        project, policy_plan=policy_plan, configured_roots=configured_roots
    )
    files = sorted(output.glob("rank_*.npz"))
    if not files:
        raise ValueError("no inference shards")
    records = [json.loads(p.with_suffix(".json").read_text()) for p in files]
    reference = records[0]
    for key, value in contract.items():
        if reference[key] != value:
            raise ValueError(f"current policy contract differs: {key}")
    for dataset in datasets:
        path = Path(dataset["root"]) / "meta" / f"advantages_{tag}.parquet"
        if path.exists() or path.with_suffix(".json").exists():
            raise FileExistsError(path)
    for path, record in zip(files, records, strict=True):
        for key in (*contract, "world_size", "total_frames", "checkpoint_weights_sha256",
                    "checkpoint_config_sha256", "source_sha256"):
            if record[key] != reference[key]:
                raise ValueError(f"inference shard contract differs: {key}")
        if record["partial_smoke"] or record["predictions_sha256"] != _sha(path):
            raise ValueError("partial or changed inference shard")
    if {x["rank"] for x in records} != set(range(reference["world_size"])):
        raise ValueError("missing inference ranks")
    arrays = [np.load(p) for p in files]
    ids = np.concatenate([a["sample_index"] for a in arrays])
    values = np.concatenate([a["value"] for a in arrays])[np.argsort(ids)]
    if not np.array_equal(np.sort(ids), np.arange(reference["total_frames"])):
        raise ValueError("inference frame coverage is not exact")
    if not np.isfinite(values).all():
        raise ValueError("nonfinite inference values")
    cursor, eligible_scores, prepared = 0, [], []
    for dataset in datasets:
        root = Path(dataset["root"])
        result = {key: [] for key in ("episode_index", "frame_index", "advantage_continuous")}
        selected = set(dataset["selected_episode_indices"])
        for ep in dataset["recap"]["episodes"]:
            if ep["episode_index"] not in selected:
                continue
            length = ep["action_count"]
            result["episode_index"].extend([ep["episode_index"]] * length)
            result["frame_index"].extend(range(length))
            if dataset["source_role"] == "hil_correction":
                advantage = np.zeros(length)  # no invented value prediction for corrections
            else:
                rewards = np.full(length, -1. / return_normalization)
                if ep["outcome"] == "failure":
                    rewards[-1] -= failure_penalty / return_normalization
                advantage = nstep_advantage(values[cursor:cursor + length], rewards, horizon)
                cursor += length
                # Only anchors whose complete H50 policy action chunk exists
                # determine the threshold.  A rollout-only population avoids
                # letting the much larger expert frame pool move the policy
                # label boundary; forced human data remain external positives.
                if threshold_role == "all" or dataset["source_role"] == threshold_role:
                    eligible_scores.append(advantage[:max(0, length - horizon + 1)])
            result["advantage_continuous"].extend(advantage.tolist())
        prepared.append((dataset, result))
    if cursor != len(values):
        raise ValueError("episode lengths differ from value inference")
    if not eligible_scores:
        raise ValueError(f"threshold role has no eligible H{horizon} anchors: {threshold_role}")
    threshold = float(np.quantile(np.concatenate(eligible_scores), 1 - fraction))
    threshold_population = (
        f"all eligible H{horizon} policy anchors excluding corrections"
        if threshold_role == "all"
        else f"eligible H{horizon} {threshold_role} anchors only"
    )
    summaries = []
    for dataset, result in prepared:
        root = Path(dataset["root"])
        path = root / "meta" / f"advantages_{tag}.parquet"
        if path.exists() or path.with_suffix(".json").exists():
            raise FileExistsError(path)
        forced = dataset["source_role"] in forced_positive_roles
        labels = np.ones(len(result["frame_index"]), dtype=bool) if forced else (
            np.asarray(result["advantage_continuous"]) >= threshold)
        result["advantage"] = labels
        table = pa.table(result)
        temporary = path.with_suffix(".tmp.parquet")
        pq.write_table(table, temporary)
        temporary.replace(path)
        manifest = {
            "schema": "clothdojo_recap_advantages", "format_version": 1, **contract,
            "role": "rollout" if dataset["source_role"] == "hil_rollout" else "sft",
            "source_role": dataset["source_role"], "forced_positive": forced,
            "prompt": "flatten the shirt", "state_key": None,
            "camera_keys": [f"observation.images.{c}" for c in
                            ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")],
            "checkpoint_weights_sha256": reference["checkpoint_weights_sha256"],
            "checkpoint_config_sha256": reference["checkpoint_config_sha256"],
            "dataset_info_sha256": _sha(root / "meta/info.json"),
            "dataset_recap_sha256": _sha(root / "meta/recap.json"),
            "advantages_sha256": _sha(path), "returns_tag": returns_tag,
            "gamma": 1., "return_normalization": return_normalization,
            "failure_penalty": failure_penalty, "terminal_bootstrap": False,
            "episodes": dataset["selected_episode_indices"],
            "stride": horizon, "threshold": threshold,
            "threshold_population": threshold_population,
            "forced_positive_roles": list(forced_positive_roles),
            "positive_fraction_requested": fraction, "rows": len(labels),
            "positive": int(labels.sum()), "negative": int((~labels).sum()),
        }
        _write(path.with_suffix(".json"), manifest)
        summaries.append({"dataset": str(root), **manifest})
    _write(output / "labels.json", {
        "tag": tag,
        "threshold": threshold,
        "threshold_population": threshold_population,
        "forced_positive_roles": list(forced_positive_roles),
        "datasets": summaries,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("infer", "finalize"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--positive-fraction", type=float, default=.3)
    parser.add_argument("--tag", default="recap_v1000_statev2_n50_q30_r0")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-config", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--policy-plan", type=Path)
    parser.add_argument("--dataset-root", type=Path, action="append")
    parser.add_argument("--return-normalization", type=float, default=9000.0)
    parser.add_argument("--returns-tag", default="fail4500_state_v2")
    parser.add_argument("--failure-penalty", type=float, default=4500.0)
    parser.add_argument(
        "--threshold-role", choices=("all", "hil_rollout"), default="all"
    )
    parser.add_argument(
        "--force-positive-role",
        action="append",
        choices=("expert", "hil_correction"),
        dest="forced_positive_roles",
    )
    args = parser.parse_args()
    if args.mode == "infer":
        infer(
            args.project_root.resolve(), args.output.resolve(), args.batch_size, args.max_batches,
            checkpoint=args.checkpoint, checkpoint_config=args.checkpoint_config,
            checkpoint_sha256=args.checkpoint_sha256, selection_path=args.selection,
            policy_plan=args.policy_plan, configured_roots=args.dataset_root,
            return_normalization=args.return_normalization, returns_tag=args.returns_tag,
        )
    else:
        finalize(args.project_root.resolve(), args.output.resolve(), args.horizon,
                 args.positive_fraction, args.tag, policy_plan=args.policy_plan,
                 configured_roots=args.dataset_root,
                 return_normalization=args.return_normalization,
                 returns_tag=args.returns_tag, failure_penalty=args.failure_penalty,
                 threshold_role=args.threshold_role,
                 forced_positive_roles=tuple(
                     args.forced_positive_roles or ("hil_correction",)
                 ))


if __name__ == "__main__":
    main()
