"""The single folding completion score: five geometric features and a fixed cutoff.

Only current particle positions enter inference. An existing verified flat mesh
fixes material weights, length scale and support height; it is not a fold target.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from clothdojo.evaluation.folding import (
    FoldVisualReference,
    _planar_hull,
    _rectangle_union_pixel_area,
    fold_visual_debug_projection_from_reference,
    fold_visual_reference,
)

FEATURE_NAMES = (
    "log_area_compression_material_frame",
    "all_layer_flatness",
    "planar_rectangle_material_frame",
    "xyz_overlap_16",
    "support_10",
)


@dataclass(frozen=True)
class FoldQualityReference:
    """Fixed material geometry, valid only for this mesh's particle ordering."""

    triangles: np.ndarray
    visual: FoldVisualReference
    material_weights: np.ndarray
    sample: np.ndarray
    nonlocality: np.ndarray
    sample_area: float
    scale: float
    table_height: float


def _positions(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not len(array):
        raise ValueError(f"{name} must have shape (particles, 3)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite numeric values")
    return array


def _normals_and_weights(points, triangles):
    vertices = np.asarray(points, dtype=np.float64)[triangles]
    crosses = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
    lengths = np.linalg.norm(crosses, axis=1)
    normals = np.divide(
        crosses, lengths[:, None], out=np.zeros_like(crosses), where=lengths[:, None] > 1e-12
    )
    return normals, lengths * 0.5


def _material_frame(q: np.ndarray, triangles: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Orient XY using fixed reference-area weights; z cannot rotate this frame."""
    q = np.asarray(q, dtype=np.float64)
    faces = q[triangles]
    center = weights @ faces[:, :, :2].mean(axis=1)
    offset = faces[:, :, :2] - center
    covariance = np.einsum("t,tvi,tvj->ij", weights / 3, offset, offset)
    axis = np.linalg.eigh(covariance)[1][:, -1]
    projection = (q[:, :2] - center) @ axis
    anchors = np.flatnonzero(abs(projection) > 0.01 * np.sqrt(np.trace(covariance)))
    if len(anchors) and projection[anchors[0]] < 0:
        axis = -axis
    result = q.copy()
    result[:, :2] = (q[:, :2] - center) @ np.stack([axis, [-axis[1], axis[0]]]).T
    return result


def prepare_fold_quality_reference(
    flat_particle_q: np.ndarray,
    triangle_indices: np.ndarray,
) -> FoldQualityReference:
    """Prepare once from an existing verified flat mesh in the same z-up frame.

    A cut segment's first state is not automatically a valid reference. Flatness
    and matching material particle order are caller-owned preconditions.
    """
    flat = _positions(flat_particle_q, "flat_particle_q")
    triangles = np.asarray(triangle_indices)
    if triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles):
        raise ValueError("triangle_indices must have shape (triangles, 3)")
    if not np.issubdtype(triangles.dtype, np.integer):
        raise ValueError("triangle_indices must be integers")
    if triangles.min() < 0 or triangles.max() >= len(flat):
        raise ValueError("triangle indices exceed particle bounds")
    triangles = triangles.copy()
    _, areas = _normals_and_weights(flat, triangles)
    area = areas.sum()
    if not np.isfinite(area) or area <= 0:
        raise ValueError("flat mesh must have positive surface area")
    weights = areas / area
    scale = float(np.sqrt(_planar_hull(flat[:, :2])[0]))
    visual = fold_visual_reference(
        _material_frame(flat, triangles, weights), triangles, pixels_per_scale=128
    )
    sample = np.searchsorted(np.cumsum(areas) / area, (np.arange(512) + 0.5) / 512)
    rest = flat[triangles].mean(axis=1)[sample] / scale
    nonlocality = 1 - np.exp(-((cdist(rest, rest, "sqeuclidean") / 0.01) ** 2))
    for array in (triangles, weights, sample, nonlocality):
        array.setflags(write=False)
    return FoldQualityReference(
        triangles=triangles, visual=visual, material_weights=weights, sample=sample,
        nonlocality=nonlocality, sample_area=float(area / 512 / scale**2),
        scale=scale, table_height=float(np.quantile(flat[:, 2], 0.01)),
    )


def fold_quality_features(
    particle_q: np.ndarray,
    reference: FoldQualityReference,
) -> np.ndarray:
    """Extract the five features in FEATURE_NAMES order, without motion/history."""
    q = _positions(particle_q, "particle_q")
    if len(q) != reference.visual.particle_count:
        raise ValueError("state must match the reference particle count")
    tri = reference.triangles
    visual = fold_visual_debug_projection_from_reference(
        _material_frame(q, tri, reference.material_weights), tri, reference.visual
    )
    occupied = np.count_nonzero(visual.full_projection_mask)
    compression = -np.log(max(min(occupied / reference.visual.flat_projection_pixels, 1), 1e-8))
    union = _rectangle_union_pixel_area(
        visual.full_projection_mask, visual.effective_rectangle_xy
    )
    planar_rectangle = float(visual.planar_top_weight.sum(dtype=np.float64)) / union
    normals, _ = _normals_and_weights(q, tri)
    flatness = float(np.average(normals[:, 2] ** 2, weights=reference.material_weights))
    centers = q[tri[reference.sample]].mean(axis=1) / reference.scale
    distance2 = cdist(centers, centers, "sqeuclidean")
    overlap = (
        (np.exp(-distance2 / 0.16**2) * reference.nonlocality).sum(axis=1)
        * reference.sample_area / (np.pi * 0.16**2)
    )
    height = centers[:, 2] - reference.table_height / reference.scale
    vector = np.array([
        compression, flatness, planar_rectangle,
        np.log1p(overlap).mean(), np.exp(-((height / 0.1) ** 2)).mean(),
    ], dtype=np.float64)
    if not np.isfinite(vector).all():
        raise ValueError("state produced non-finite folding features")
    return vector


@dataclass(frozen=True)
class FoldQualityMetric:
    """Shared linear completion margin; Q is neither probability nor percentage."""

    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    score_center: float
    score_scale: float
    success_threshold: float

    @classmethod
    @cache
    def load(cls) -> FoldQualityMetric:
        """Load the sole packaged model; no selectable weights or legacy variants."""
        model = json.loads(Path(__file__).with_name("fold_quality_weights.json").read_text())
        if tuple(model["features"]) != FEATURE_NAMES:
            raise ValueError("invalid folding feature order")
        arrays = [
            np.asarray(model[key], dtype=np.float64) for key in ("mean", "scale", "coefficients")
        ]
        if any(a.shape != (5,) or not np.isfinite(a).all() for a in arrays):
            raise ValueError("invalid folding model dimensions or non-finite weights")
        scalars = [float(model[key]) for key in (
            "intercept", "score_center", "score_scale", "success_threshold"
        )]
        if not np.isfinite(scalars).all() or np.any(arrays[1] <= 0) or scalars[2] <= 0:
            raise ValueError("folding model requires finite weights and positive scales")
        for array in arrays:
            array.setflags(write=False)
        return cls(*arrays, *scalars)

    def score_features(self, vector: np.ndarray) -> np.ndarray:
        """Score a vector or batch in FEATURE_NAMES order."""
        raw = np.asarray(vector, dtype=np.float64)
        if raw.ndim not in (1, 2) or raw.shape[-1] != 5 or not np.isfinite(raw).all():
            raise ValueError("folding features must be finite with shape (5,) or (n, 5)")
        z = 3 * np.tanh((raw - self.mean) / (3 * self.scale))
        return (z @ self.coefficients + self.intercept - self.score_center) / self.score_scale

    def score(self, particle_q: np.ndarray, reference: FoldQualityReference) -> float:
        return float(self.score_features(fold_quality_features(particle_q, reference)))

    def passes_threshold(self, score: float | np.ndarray) -> np.ndarray:
        """Single-state qualification; trajectory success also requires a 3 s hold."""
        values = np.asarray(score, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("fold quality score must be finite")
        return values >= self.success_threshold
