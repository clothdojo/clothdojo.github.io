"""Interpret a confirmed sequential HIL review in the console's actual order."""

from __future__ import annotations

import json
from pathlib import Path

from clothdojo.rollout.initial_states import InitialStatePool, _sha256


def classify_sequential_review(
    trajectories: list[dict], marked: set[str], *, causes: dict[str, str] | None = None,
) -> dict:
    """Apply the operator's explicit skip convention only through the last mark.

    The HIL console renders metadata trajectories in their existing order.
    It does not alphabetically sort the all-assets episode list. This inference
    is appropriate only after the operator confirms sequential review and the
    meaning of skips. A skipped trajectory identifies a quality issue, not its
    cause or onset; only a separate assessment can label an initial defect.
    """

    ids = [Path(item["path"]).name for item in trajectories]
    if len(ids) != len(set(ids)) or marked - set(ids):
        raise ValueError("review marks do not match unique rollout trajectories")
    causes = {} if causes is None else causes
    if set(causes) - set(ids) or any(
        cause not in {"initial_self_intersection", "runtime_table_issue"}
        for cause in causes.values()
    ):
        raise ValueError("quality causes must name known trajectories and supported defect types")
    groups: dict[str, list[int]] = {}
    for position, item in enumerate(trajectories):
        groups.setdefault(item["asset_id"], []).append(position)
    if any(
        positions != list(range(positions[0], positions[-1] + 1)) for positions in groups.values()
    ):
        raise ValueError("sequential asset review requires contiguous asset groups")
    frontier = max((index for index, eid in enumerate(ids) if eid in marked), default=-1)
    rejected_assets = {
        asset_id
        for asset_id, positions in groups.items()
        if positions[-1] <= frontier and not any(ids[index] in marked for index in positions)
    }
    entries = []
    for position, (eid, item) in enumerate(zip(ids, trajectories, strict=True)):
        if position > frontier:
            decision = "unreviewed"
        elif item["asset_id"] in rejected_assets:
            decision = "rejected_asset"
        elif eid in marked:
            decision = "marked"
        else:
            decision = "skipped"
        if eid in causes:
            if decision != "skipped":
                raise ValueError(
                    "quality causes require a skipped trajectory within the reviewed prefix"
                )
            decision = causes[eid]
        entries.append(
            {
                "position": position + 1,
                "episode_id": eid,
                "asset_id": item["asset_id"],
                "initial_state_index": item["initial_state_index"],
                "seed": item["seed"],
                "decision": decision,
            }
        )
    return {
        "order": "rollout_meta_trajectories",
        "reviewed_through_position": frontier + 1,
        "reviewed_through_episode": ids[frontier] if frontier >= 0 else None,
        "rejected_assets": [asset_id for asset_id in groups if asset_id in rejected_assets],
        "skipped_count": sum(
            e["decision"] in {"skipped", "initial_self_intersection", "runtime_table_issue"}
            for e in entries
        ),
        "initial_self_intersection_count": sum(
            e["decision"] == "initial_self_intersection" for e in entries
        ),
        "runtime_table_issue_count": sum(e["decision"] == "runtime_table_issue" for e in entries),
        "pending_cause_count": sum(e["decision"] == "skipped" for e in entries),
        "marked_count": len(marked),
        "unreviewed_count": len(ids) - frontier - 1,
        "entries": entries,
    }


def inspect_sequential_hil_review(rollout_root: str | Path) -> dict:
    root = Path(rollout_root).expanduser().resolve()
    meta_path, review_path = root / "meta.json", root / "hil_review.json"
    meta_bytes, review_bytes = meta_path.read_bytes(), review_path.read_bytes()
    meta, review = json.loads(meta_bytes), json.loads(review_bytes)
    if meta.get("schema") != "clothdojo_rollout" or meta.get("status") != "complete":
        raise ValueError("sequential review needs a complete canonical rollout collection")
    if review.get("schema") != "clothdojo_hil_review_v3":
        raise ValueError("sequential review requires the current HIL marks and breakpoints schema")
    quality_path = root / "hil_quality_review.json"
    quality_bytes = quality_path.read_bytes() if quality_path.exists() else None
    assessments = {}
    if quality_bytes is not None:
        quality = json.loads(quality_bytes)
        if (
            quality.get("schema") != "clothdojo_hil_quality_review"
            or quality.get("rollout_meta_sha256") != _sha256(meta_path)
        ):
            raise ValueError("HIL quality causes belong to a different rollout collection")
        assessments = quality["episodes"]
    marked = set(review["marks"]) | {eid for eid, points in review["breakpoints"].items() if points}
    result = classify_sequential_review(
        meta["trajectories"], marked,
        causes={eid: assessment["cause"] for eid, assessment in assessments.items()},
    )
    states_root = Path(meta["initial_states"]["path"])
    if _sha256(states_root / "meta.json") != meta["initial_states"]["meta_sha256"]:
        raise ValueError("reviewed rollout initial-state collection has changed")
    pools = {}
    for entry in result["entries"]:
        asset_id = entry["asset_id"]
        if asset_id not in pools:
            path = states_root / asset_id / "states.npz"
            pools[asset_id] = (InitialStatePool.load(path), _sha256(path))
        pool, digest = pools[asset_id]
        index = entry["initial_state_index"]
        if not 0 <= index < pool.state_count or int(pool.seeds[index]) != entry["seed"]:
            raise ValueError("reviewed rollout state index/seed does not match its pool")
        entry["pool_sha256"] = digest
        if entry["episode_id"] in assessments:
            entry["cause_assessment"] = assessments[entry["episode_id"]]
    if meta_path.read_bytes() != meta_bytes or review_path.read_bytes() != review_bytes:
        raise ValueError("HIL review changed while constructing its sequential audit")
    if (quality_path.read_bytes() if quality_path.exists() else None) != quality_bytes:
        raise ValueError("HIL quality causes changed while constructing their audit")
    return {
        "schema": "clothdojo_sequential_initial_state_review",
        "rollout_root": str(root),
        "rollout_meta_sha256": _sha256(meta_path),
        "review_sha256": _sha256(review_path),
        "quality_review_sha256": None if quality_bytes is None else _sha256(quality_path),
        "states_root": str(states_root),
        **result,
    }
