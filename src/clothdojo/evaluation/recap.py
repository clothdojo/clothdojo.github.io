"""CPU diagnostics for saved RECAP validation predictions, without policy labels.

Local visual labels are diagnostic observations, not ground-truth advantages.
Validation-derived quantiles below measure sensitivity only; they must never be
used as thresholds for policy training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def nstep_advantage(values: np.ndarray, rewards: np.ndarray, horizon: int) -> np.ndarray:
    """One episode, gamma=1; reward includes the final boundary exactly once.

    Both inputs use the same normalized units. There is no exported terminal
    observation: bootstrap is zero for windows reaching or crossing the end.
    """
    values, rewards = np.asarray(values, dtype=float), np.asarray(rewards, dtype=float)
    if values.ndim != 1 or not len(values) or rewards.shape != values.shape:
        raise ValueError("values and rewards must be nonempty, aligned 1-D episode arrays")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if not np.isfinite(values).all() or not np.isfinite(rewards).all():
        raise ValueError("non-finite value or reward")
    starts = np.arange(len(values))
    ends = np.minimum(starts + horizon, len(values))
    cumulative = np.r_[0.0, np.cumsum(rewards)]
    bootstrap = np.r_[values, 0.0][ends]
    return cumulative[ends] - cumulative[starts] + bootstrap - values


def episode_index(project: Path, launch: dict, split: dict) -> list[dict]:
    episodes, cursor = [], 0
    source_hashes = {Path(x["path"]).name: x for x in launch["datasets"]}
    for item in launch["resolved_config"]["data"]["eval_data_paths"]:
        root = project / item["dataset_path"]
        if sha256(root / "meta/recap.json") != source_hashes[root.name]["recap_sha256"]:
            raise ValueError(f"metadata differs from training: {root}")
        if sha256(root / "meta/info.json") != source_hashes[root.name]["info_sha256"]:
            raise ValueError(f"video/frame metadata differs from training: {root}")
        recap = json.loads((root / "meta/recap.json").read_text())
        info = json.loads((root / "meta/info.json").read_text())
        role = recap["dataset"]
        selected = set(split["datasets"][role]["eval_source_episode_ids"])
        for ep in recap["episodes"]:
            if ep["source_episode_id"] not in selected:
                continue
            episodes.append({**ep, "role": role, "dataset_root": str(root),
                             "fps": info["fps"], "video_path": info["video_path"],
                             "chunks_size": info["chunks_size"], "offset": cursor,
                             "domain": "seen" if ep["asset_id"] in
                             split["assets"]["eval_seen"] else "unseen"})
            cursor += ep["action_count"]
    if len(episodes) != 104 or cursor != 310949:
        raise ValueError("saved validation frame contract changed")
    if len({(e["role"], e["source_episode_id"]) for e in episodes}) != len(episodes):
        raise ValueError("duplicate validation episode identity")
    return episodes


def prepare_clips(episodes: list[dict], output: Path) -> list[dict]:
    """Choose clips by seed, metadata and position, without reading predictions."""
    path = output / "clips.json"
    if path.exists():
        return json.loads(path.read_text())["clips"]
    rng = np.random.default_rng(260904)
    clips = []
    for role, outcome in (("expert", "success"), ("hil_rollout", "success"),
                          ("hil_rollout", "failure")):
        for domain in ("seen", "unseen"):
            candidates = [i for i, ep in enumerate(episodes)
                          if (ep["role"], ep["outcome"], ep["domain"]) == (role, outcome, domain)
                          and ep["action_count"] > 300]
            selected_assets = set()
            for index in rng.permutation(candidates):
                ep = episodes[index]
                if ep["asset_id"] in selected_assets:
                    continue
                selected_assets.add(ep["asset_id"])
                for fraction in (0.1, 0.6):
                    start = int((ep["action_count"] - 251) * fraction)
                    clips.append({"episode": int(index), "start": start, "end": start + 250,
                                  "selection": "seeded_position"})
                if len(selected_assets) == 2:
                    break
    for index, ep in enumerate(episodes):
        for bad in ep.get("review", {}).get("bad_ranges", []):
            clips.append({"episode": index, "start": bad["action_start"],
                          "end": min(bad["action_end_exclusive"], ep["action_count"] - 1),
                          "selection": "legacy_bad_candidate_not_ground_truth"})
    rng.shuffle(clips)
    for number, clip in enumerate(clips):
        clip["clip_id"] = f"clip_{number:03d}"
    write_json(path, {"seed": 260904, "prediction_independent": True, "clips": clips})
    write_json(output / "review_template.json", {
        "reviewer": None, "method": "review imagery before opening clip_scores.json",
        "labels": {"progress": "visible improvement toward flattening",
                   "regress": "visible loss of flattening or controllability",
                   "stall": "no clear net progress during the window",
                   "ambiguous": "occlusion, necessary intermediate motion, or unclear net effect"},
        "limitations": "visual net progress is not a measured counterfactual advantage",
        "clips": [{"clip_id": c["clip_id"], "label": None, "confidence": None,
                   "reason": ""} for c in clips],
    })
    return clips


def render_panels(episodes: list[dict], clips: list[dict], output: Path) -> None:
    import imageio_ffmpeg
    from PIL import Image, ImageDraw

    panels = output / "panels"
    panels.mkdir(exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    for clip in clips:
        target = panels / f"{clip['clip_id']}.jpg"
        if target.exists():
            continue
        ep = episodes[clip["episode"]]
        frames = np.linspace(clip["start"], clip["end"], 7).round().astype(int)
        panel = Image.new("RGB", (7 * 224, 3 * 224 + 55), "white")
        draw = ImageDraw.Draw(panel)
        draw.text((5, 4), f"{clip['clip_id']} | time left to right | high / left / right",
                  fill="black")
        for row, camera in enumerate(("cam_high", "cam_left_wrist", "cam_right_wrist")):
            video = Path(ep["dataset_root"]) / ep["video_path"].format(
                episode_chunk=ep["episode_index"] // ep["chunks_size"],
                episode_index=ep["episode_index"], video_key=f"observation.images.{camera}")
            # Decode by exact frame number rather than a timestamp/keyframe seek.
            selection = "+".join(f"eq(n\\,{int(frame)})" for frame in frames)
            result = subprocess.run(
                [ffmpeg, "-v", "error", "-threads", "1", "-i", str(video), "-vf",
                 f"select={selection},scale=224:224", "-fps_mode", "passthrough",
                 "-frames:v", "7", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                check=True, capture_output=True)
            expected = 7 * 224 * 224 * 3
            if len(result.stdout) != expected:
                raise ValueError(f"wrong decoded frame count for {target}: {len(result.stdout)}")
            images = np.frombuffer(result.stdout, np.uint8).reshape(7, 224, 224, 3)
            for column, pixels in enumerate(images):
                panel.paste(Image.fromarray(pixels), (column * 224, 55 + row * 224))
                if row == 0:
                    draw.text((column * 224 + 5, 28),
                              f"+{(frames[column] - frames[0]) / ep['fps']:.2f}s", fill="black")
        panel.save(target, quality=92)
        print(f"rendered {target.name}", flush=True)


def describe(values: np.ndarray) -> dict:
    return {"count": len(values), "mean": float(np.mean(values)),
            "positive_fraction": float(np.mean(values > 0)),
            "near_zero_fraction": float(np.mean(np.abs(values) < 1e-7)),
            "quantiles": dict(zip(("p01", "p10", "p50", "p90", "p99"),
                                  np.quantile(values, [0.01, 0.1, 0.5, 0.9, 0.99]).tolist(),
                                  strict=True))}


def pair_auc(positive: np.ndarray, negative: np.ndarray) -> float | None:
    if not len(positive) or not len(negative):
        return None
    ranks = rankdata(np.r_[positive, negative])[:len(positive)]
    return float((ranks.sum() - len(positive) * (len(positive) + 1) / 2)
                 / (len(positive) * len(negative)))


def analyze(episodes: list[dict], clips: list[dict], output: Path) -> None:
    horizons = (1, 10, 50, 100, 250)
    expected, rewards, masks = [], [], {h: [] for h in horizons}
    for ep in episodes:
        length = ep["action_count"]
        boundary = 0 if ep["outcome"] == "success" else -4500
        expected.extend((-np.arange(length, 0, -1) + boundary) / 9000)
        reward = np.full(length, -1.0 / 9000)
        reward[-1] += boundary / 9000
        rewards.append(reward)
        for h in horizons:
            masks[h].extend(np.arange(length) + h < length)
    expected = np.asarray(expected)
    masks = {h: np.asarray(mask) for h, mask in masks.items()}
    summary = {"units": "normalized return, divide raw rewards by 9000",
               "policy_labels_written": False,
               "quantiles": "validation-only descriptive sensitivity, not policy thresholds",
               "checkpoints": {}, "agreement": {}}
    advantages, clip_scores = {}, {}
    for path in sorted((output / "predictions").glob("step_*.npz")):
        step = int(path.stem.split("_")[1])
        rows = np.load(path)["rows"]
        if rows.shape != (len(expected), 6) or not np.isfinite(rows).all():
            raise ValueError(f"bad validation rows: {path}")
        if not np.array_equal(rows[:, 0], np.arange(len(expected))):
            raise ValueError(f"missing/reordered validation identities: {path}")
        if not np.allclose(rows[:, 2], expected, atol=1e-7, rtol=0):
            raise ValueError(f"episode mapping does not match return targets: {path}")
        values = rows[:, 1]
        summary["checkpoints"][step] = {"sha256": sha256(path), "horizons": {}}
        advantages[step] = {}
        for h in horizons:
            adv = np.concatenate([
                nstep_advantage(values[ep["offset"]:ep["offset"] + ep["action_count"]], r, h)
                for ep, r in zip(episodes, rewards, strict=True)])
            advantages[step][h] = adv
            summary["checkpoints"][step]["horizons"][h] = {
                "interior": describe(adv[masks[h]]), "terminal": describe(adv[~masks[h]])}
        clip_scores[step] = []
        for clip in clips:
            ep = episodes[clip["episode"]]
            start, end = clip["start"], clip["end"]
            v = values[ep["offset"]:ep["offset"] + ep["action_count"]]
            clip_scores[step].append({"clip_id": clip["clip_id"], "value_start": float(v[start]),
                                     "value_end": float(v[end]), "duration": end - start,
                                     "advantage": float(v[end] - v[start] - (end - start) / 9000)})
    if not advantages:
        raise ValueError("no prediction files")
    for left, right in combinations(advantages, 2):
        results = {}
        for h in horizons:
            a, b = advantages[left][h][masks[h]], advantages[right][h][masks[h]]
            pa, pb = a >= np.quantile(a, 0.7), b >= np.quantile(b, 0.7)
            rho = spearmanr(a, b).statistic
            results[h] = {"advantage_spearman": float(rho) if np.isfinite(rho) else None,
                          "top30_jaccard": float(np.sum(pa & pb) / np.sum(pa | pb)),
                          "positive_sign_agreement": float(np.mean((a > 0) == (b > 0)))}
        summary["agreement"][f"{left}_vs_{right}"] = results
    review_path = output / "review.json"
    if review_path.exists():
        review = json.loads(review_path.read_text())
        if not review.get("frozen_before_scores"):
            raise ValueError("review must explicitly record whether it preceded model scores")
        labels = {x["clip_id"]: x for x in review["clips"]}
        if set(labels) != {x["clip_id"] for x in clips}:
            raise ValueError("review clip identity mismatch")
        summary["visual_review"] = {"sha256": sha256(review_path), "results": {}}
        for step, scores in clip_scores.items():
            groups = {key: [] for key in ("progress", "regress", "stall", "ambiguous")}
            seeded_ids = {c["clip_id"] for c in clips if c["selection"] == "seeded_position"}
            for score in scores:
                if score["clip_id"] not in seeded_ids:
                    continue
                label = labels[score["clip_id"]]
                groups[label["label"]].append(score["advantage"])
            summary["visual_review"]["results"][step] = {
                "selection": "seeded 250-step windows only; legacy candidate excluded",
                "group_counts": {k: len(v) for k, v in groups.items()},
                "mean_advantage": {k: float(np.mean(v)) if v else None for k, v in groups.items()},
                "progress_vs_regress_auc": pair_auc(np.array(groups["progress"]),
                                                    np.array(groups["regress"])),
                "progress_vs_stall_auc": pair_auc(np.array(groups["progress"]),
                                                  np.array(groups["stall"]))}
    write_json(output / "summary.json", summary)
    write_json(output / "clip_scores.json", clip_scores)
    print(json.dumps(summary.get("visual_review", {"status": "no independent review"}), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--render-panels", action="store_true")
    args = parser.parse_args()
    project, output = args.project_root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan_root = project / "outputs/recap/hil800_20260904"
    launch = json.loads((plan_root / "reliability_20260905/formal_launch.json").read_text())
    if sha256(plan_root / "value_split.json") != launch["split"]["sha256"]:
        raise ValueError("validation split differs from the completed training run")
    split = json.loads((plan_root / "value_split.json").read_text())
    episodes = episode_index(project, launch, split)
    write_json(output / "episodes.json", episodes)
    clips = prepare_clips(episodes, output)
    if args.render_panels:
        render_panels(episodes, clips, output)
    if not args.prepare_only:
        analyze(episodes, clips, output)


if __name__ == "__main__":
    main()
