"""Bind a selected successful rollout to its RGB training export."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def first_selected_trajectory(selection: Path) -> Path:
    """Return the first successful trajectory in a completed selection."""
    selection = selection.expanduser().resolve()
    report = json.loads(selection.read_text())
    if report.get("status") != "complete":
        raise ValueError("generation selection is not complete")
    selected = report.get("selected") or []
    if not selected:
        raise ValueError("generation selection has no successful trajectories")
    trajectory = (selection.parent / selected[0]).resolve()
    if not (trajectory / "manifest.json").is_file():
        raise FileNotFoundError(f"selected trajectory is missing: {trajectory}")
    return trajectory


def approve_selected_export(selection: Path, trajectory: Path, dataset: Path) -> None:
    selection = selection.expanduser().resolve()
    trajectory = trajectory.expanduser().resolve()
    dataset = dataset.expanduser().resolve()
    report = json.loads(selection.read_text())
    if report.get("status") != "complete":
        raise ValueError("generation selection is not complete")
    selected = [str(item) for item in report.get("selected", [])]
    if not any((selection.parent / item).resolve() == trajectory for item in selected):
        raise ValueError("trajectory is not listed among selected successes")
    score_path = trajectory / "score.json"
    score = json.loads(score_path.read_text())
    if score.get("success") is not True:
        raise ValueError("selected trajectory has no successful score")
    if score.get("protocol_fingerprint") != report.get("protocol_fingerprint"):
        raise ValueError("selection and trajectory scoring protocols differ")
    sidecar_path = dataset / "meta/clothdojo.json"
    sidecar = json.loads(sidecar_path.read_text())
    if (sidecar.get("schema") != "clothdojo_lerobot_export"
            or sidecar.get("export_status") != "complete"
            or (sidecar.get("observation") or {}).get("cloth_visual") != "rgb"):
        raise ValueError("expected a completed ClothDojo RGB export")
    if sidecar.get("training_eligible") is True:
        raise ValueError("RGB export is already marked training eligible")
    sidecar["training_eligible"] = True
    sidecar["selection"] = {
        "kind": "selected_successful_rollout",
        "selection_sha256": hashlib.sha256(selection.read_bytes()).hexdigest(),
        "score_sha256": hashlib.sha256(score_path.read_bytes()).hexdigest(),
        "protocol_fingerprint": report["protocol_fingerprint"],
    }
    temporary = sidecar_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(sidecar, indent=2) + "\n")
    temporary.replace(sidecar_path)
