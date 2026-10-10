"""Folding geometry and diagnostic primitives; the sole outcome metric is fold_quality."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import ConvexHull


@dataclass(frozen=True, slots=True)
class FoldStateDiagnostics:
    """Rotation/translation-invariant diagnostics for one terminal state."""

    initial_footprint_area_m2: float
    terminal_convex_footprint_area_m2: float
    convex_footprint_ratio: float
    convex_rectangularity: float
    footprint_aspect_ratio: float
    normalized_thickness_p90: float
    speed_rms_m_s: float


@dataclass(frozen=True, slots=True)
class FoldVisualMetrics:
    """Two target-free quantities describing a visually tidy folded package.

    ``planar_rectangle_iou`` softly weights the largest connected surface by
    agreement with a robustly fitted dominant top plane. Its numerator is that
    weighted surface area; its denominator is the union of the surface's
    minimum oriented rectangle and the complete garment projection. It falls
    for a rough or incomplete top, irregular boundary, or cloth protruding
    beyond the main package. ``compression_ratio`` rejects an otherwise tidy
    but unfolded garment without prescribing a template or aspect ratio.
    """

    planar_rectangle_iou: float
    compression_ratio: float
    terminal_projection_area_m2: float
    flat_projection_area_m2: float
    bounding_rectangle_area_m2: float
    planar_top_area_m2: float
    shape_union_area_m2: float


@dataclass(frozen=True, slots=True)
class FoldVisualReference:
    """Reusable flat-state raster quantities for scoring a trajectory."""

    particle_count: int
    normalization_scale_m: float
    pixel_size_m: float
    flat_projection_pixels: int


@dataclass(frozen=True, slots=True)
class FoldVisualDebugProjection:
    """Raster components used to visually audit ``planar_rectangle_iou``."""

    full_projection_mask: np.ndarray
    planar_top_mask: np.ndarray
    planar_top_weight: np.ndarray
    bounding_rectangle_xy: np.ndarray
    effective_rectangle_xy: np.ndarray


def aligned_fold_distance(
    particle_q: np.ndarray,
    target_q: np.ndarray,
    *,
    normalization_scale_m: float,
) -> float:
    """Mean corresponding-particle 3D distance after planar rigid alignment.

    The target is translated and continuously rotated in the table plane to fit
    the state.  Height is not translated: both states must use the same table
    frame, so lifted, wrinkled, or incorrectly stacked layers remain penalized.
    Reflection and correspondence permutations are never allowed.
    """

    current = _particle_positions(particle_q, "particle_q").astype(np.float64)
    target = _particle_positions(target_q, "target_q").astype(np.float64)
    if current.shape != target.shape:
        raise ValueError("particle_q and target_q must have identical shape")
    scale = float(normalization_scale_m)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("normalization_scale_m must be finite and positive")

    current_mean = current[:, :2].mean(axis=0)
    target_mean = target[:, :2].mean(axis=0)
    current_xy = current[:, :2] - current_mean
    target_xy = target[:, :2] - target_mean

    # arg(sum(conj(target) * current)) is the least-squares planar rotation.
    real = np.einsum("ni,ni->", target_xy, current_xy)
    imag = np.einsum("n,n->", target_xy[:, 0], current_xy[:, 1])
    imag -= np.einsum("n,n->", target_xy[:, 1], current_xy[:, 0])
    angle = np.arctan2(imag, real)
    cosine, sine = np.cos(angle), np.sin(angle)
    aligned_xy = np.empty_like(target_xy)
    aligned_xy[:, 0] = target_xy[:, 0] * cosine - target_xy[:, 1] * sine
    aligned_xy[:, 1] = target_xy[:, 0] * sine + target_xy[:, 1] * cosine
    aligned_xy += current_mean

    residual_xy = current[:, :2] - aligned_xy
    residual_z = current[:, 2] - target[:, 2]
    distance = np.sqrt(np.square(residual_xy).sum(axis=1) + np.square(residual_z))
    return float(distance.mean() / scale)


def fold_state_diagnostics(
    initial_particle_q: np.ndarray,
    terminal_particle_q: np.ndarray,
    terminal_particle_qd: np.ndarray,
) -> FoldStateDiagnostics:
    """Compute target-free fold diagnostics using the initial flat state as scale."""

    initial = _particle_positions(initial_particle_q, "initial_particle_q")
    terminal = _particle_positions(terminal_particle_q, "terminal_particle_q")
    velocity = _particle_positions(terminal_particle_qd, "terminal_particle_qd")
    if initial.shape != terminal.shape or velocity.shape != terminal.shape:
        raise ValueError("initial, terminal, and velocity particle arrays must match")

    initial_area, _initial_box_area, _initial_aspect = _planar_hull(initial[:, :2])
    terminal_area, terminal_box_area, terminal_aspect = _planar_hull(terminal[:, :2])
    scale = float(np.sqrt(initial_area))
    z05, z95 = np.percentile(terminal[:, 2], (5.0, 95.0))
    speed = np.sqrt(np.square(velocity.astype(np.float64)).sum(axis=1).mean())
    return FoldStateDiagnostics(
        initial_footprint_area_m2=initial_area,
        terminal_convex_footprint_area_m2=terminal_area,
        convex_footprint_ratio=terminal_area / initial_area,
        convex_rectangularity=terminal_area / terminal_box_area,
        footprint_aspect_ratio=terminal_aspect,
        normalized_thickness_p90=float((z95 - z05) / scale),
        speed_rms_m_s=float(speed),
    )


def fold_visual_metrics(
    flat_particle_q: np.ndarray,
    terminal_particle_q: np.ndarray,
    triangle_indices: np.ndarray,
    *,
    pixels_per_scale: int = 256,
    flatness_tolerance_ratio: float = 0.04,
    normal_z_min: float = 0.8,
) -> FoldVisualMetrics:
    """Measure planar rectangular appearance and projected-area compression.

    The raster scale is derived from the square root of the flat garment's
    projected convex-hull area.  A small CPU z-buffer approximates the visible
    top triangle at every pixel.  The dominant horizontal height band is then
    reduced to its largest connected component, so separate flat protrusions
    do not count as part of the complete top plane.

    This intentionally evaluates terminal appearance only.  It does not encode
    a folding path, layer order, target aspect ratio, or asset-specific folded
    template.
    """

    flat = _particle_positions(flat_particle_q, "flat_particle_q").astype(np.float64)
    terminal = _particle_positions(
        terminal_particle_q, "terminal_particle_q"
    ).astype(np.float64)
    if flat.shape != terminal.shape:
        raise ValueError("flat_particle_q and terminal_particle_q must match")
    triangles = _triangle_indices(triangle_indices, len(flat))
    if not isinstance(pixels_per_scale, (int, np.integer)) or pixels_per_scale < 32:
        raise ValueError("pixels_per_scale must be an integer of at least 32")
    tolerance_ratio = float(flatness_tolerance_ratio)
    if not np.isfinite(tolerance_ratio) or tolerance_ratio <= 0.0:
        raise ValueError("flatness_tolerance_ratio must be finite and positive")
    normal_threshold = float(normal_z_min)
    if not np.isfinite(normal_threshold) or not 0.0 <= normal_threshold <= 1.0:
        raise ValueError("normal_z_min must be in [0, 1]")

    reference = fold_visual_reference(
        flat, triangles, pixels_per_scale=pixels_per_scale
    )
    return fold_visual_metrics_from_reference(
        terminal,
        triangles,
        reference,
        flatness_tolerance_ratio=flatness_tolerance_ratio,
        normal_z_min=normal_z_min,
    )


def fold_visual_reference(
    flat_particle_q: np.ndarray,
    triangle_indices: np.ndarray,
    *,
    pixels_per_scale: int = 256,
) -> FoldVisualReference:
    """Precompute the invariant flat projection used by every trajectory frame."""

    flat = _particle_positions(flat_particle_q, "flat_particle_q").astype(np.float64)
    triangles = _triangle_indices(triangle_indices, len(flat))
    if not isinstance(pixels_per_scale, (int, np.integer)) or pixels_per_scale < 32:
        raise ValueError("pixels_per_scale must be an integer of at least 32")
    flat_hull_area, _box_area, _aspect = _planar_hull(flat[:, :2])
    normalization_scale = float(np.sqrt(flat_hull_area))
    pixel_size = normalization_scale / int(pixels_per_scale)
    flat_projection_pixels = _projection_pixel_count(flat, triangles, pixel_size)
    if flat_projection_pixels < 1:
        raise ValueError("flat garment projection is empty")
    return FoldVisualReference(
        particle_count=len(flat),
        normalization_scale_m=normalization_scale,
        pixel_size_m=pixel_size,
        flat_projection_pixels=flat_projection_pixels,
    )


def fold_visual_metrics_from_reference(
    terminal_particle_q: np.ndarray,
    triangle_indices: np.ndarray,
    reference: FoldVisualReference,
    *,
    flatness_tolerance_ratio: float = 0.04,
    normal_z_min: float = 0.8,
) -> FoldVisualMetrics:
    """Score one state using a flat-state reference computed once per trajectory."""

    terminal = _particle_positions(
        terminal_particle_q, "terminal_particle_q"
    ).astype(np.float64)
    if len(terminal) != reference.particle_count:
        raise ValueError("terminal particle count does not match visual reference")
    triangles = _triangle_indices(triangle_indices, len(terminal))
    tolerance_ratio = float(flatness_tolerance_ratio)
    if not np.isfinite(tolerance_ratio) or tolerance_ratio <= 0.0:
        raise ValueError("flatness_tolerance_ratio must be finite and positive")
    normal_threshold = float(normal_z_min)
    if not np.isfinite(normal_threshold) or not 0.0 <= normal_threshold <= 1.0:
        raise ValueError("normal_z_min must be in [0, 1]")

    pixel_size = reference.pixel_size_m
    flat_projection_pixels = reference.flat_projection_pixels
    debug = fold_visual_debug_projection_from_reference(
        terminal,
        triangles,
        reference,
        flatness_tolerance_ratio=tolerance_ratio,
        normal_z_min=normal_threshold,
    )
    terminal_pixels = int(np.count_nonzero(debug.full_projection_mask))
    if terminal_pixels < 1:
        raise ValueError("garment projection is empty")

    rectangle_pixels = _minimum_rectangle_pixel_area(debug.full_projection_mask)
    shape_union_pixels = _rectangle_union_pixel_area(
        debug.full_projection_mask, debug.effective_rectangle_xy
    )
    planar_pixels = float(debug.planar_top_weight.sum(dtype=np.float64))
    pixel_area = pixel_size * pixel_size
    return FoldVisualMetrics(
        planar_rectangle_iou=float(planar_pixels / shape_union_pixels),
        compression_ratio=float(terminal_pixels / flat_projection_pixels),
        terminal_projection_area_m2=float(terminal_pixels * pixel_area),
        flat_projection_area_m2=float(flat_projection_pixels * pixel_area),
        bounding_rectangle_area_m2=float(rectangle_pixels * pixel_area),
        planar_top_area_m2=float(planar_pixels * pixel_area),
        shape_union_area_m2=float(shape_union_pixels * pixel_area),
    )


def fold_visual_debug_projection_from_reference(
    terminal_particle_q: np.ndarray,
    triangle_indices: np.ndarray,
    reference: FoldVisualReference,
    *,
    flatness_tolerance_ratio: float = 0.04,
    normal_z_min: float = 0.8,
) -> FoldVisualDebugProjection:
    """Return the exact raster masks and rectangle used by the visual metric."""

    terminal = _particle_positions(
        terminal_particle_q, "terminal_particle_q"
    ).astype(np.float64)
    if len(terminal) != reference.particle_count:
        raise ValueError("terminal particle count does not match visual reference")
    triangles = _triangle_indices(triangle_indices, len(terminal))
    tolerance_ratio = float(flatness_tolerance_ratio)
    if not np.isfinite(tolerance_ratio) or tolerance_ratio <= 0.0:
        raise ValueError("flatness_tolerance_ratio must be finite and positive")
    normal_threshold = float(normal_z_min)
    if not np.isfinite(normal_threshold) or not 0.0 <= normal_threshold <= 1.0:
        raise ValueError("normal_z_min must be in [0, 1]")

    top = _top_projection(terminal, triangles, reference.pixel_size_m)
    planar, planar_weight = _dominant_plane_top_region(
        top.height,
        top.normal_x,
        top.normal_y,
        top.normal_z,
        top.occupancy,
        tolerance_m=tolerance_ratio * reference.normalization_scale_m,
        pixel_size_m=reference.pixel_size_m,
        normal_z_min=normal_threshold,
    )
    _area, rectangle = _minimum_rectangle(top.occupancy)
    if np.count_nonzero(planar) >= 3:
        _effective_area, effective_rectangle = _minimum_rectangle(planar)
    else:
        effective_rectangle = rectangle.copy()
    occupancy = top.occupancy.astype(bool)
    occupancy.setflags(write=False)
    planar.setflags(write=False)
    planar_weight.setflags(write=False)
    rectangle.setflags(write=False)
    effective_rectangle.setflags(write=False)
    return FoldVisualDebugProjection(
        full_projection_mask=occupancy,
        planar_top_mask=planar,
        planar_top_weight=planar_weight,
        bounding_rectangle_xy=rectangle,
        effective_rectangle_xy=effective_rectangle,
    )


@dataclass(frozen=True, slots=True)
class _TopProjection:
    occupancy: np.ndarray
    height: np.ndarray
    normal_x: np.ndarray
    normal_y: np.ndarray
    normal_z: np.ndarray


def _projection_pixel_count(
    points: np.ndarray, triangles: np.ndarray, pixel_size: float
) -> int:
    occupancy, _pixels = _projection_raster(points, triangles, pixel_size)
    return int(np.count_nonzero(occupancy))


def _top_projection(
    points: np.ndarray, triangles: np.ndarray, pixel_size: float
) -> _TopProjection:
    occupancy, pixels = _projection_raster(points, triangles, pixel_size)
    height = np.full(occupancy.shape, -np.inf, dtype=np.float32)
    normal_x = np.zeros(occupancy.shape, dtype=np.float32)
    normal_y = np.zeros(occupancy.shape, dtype=np.float32)
    normal_z = np.zeros(occupancy.shape, dtype=np.float32)

    vertices = points[triangles]
    centroids_z = vertices[:, :, 2].mean(axis=1)
    edges_a = vertices[:, 1] - vertices[:, 0]
    edges_b = vertices[:, 2] - vertices[:, 0]
    normals = np.cross(edges_a, edges_b)
    lengths = np.linalg.norm(normals, axis=1)
    unit_normals = np.divide(
        normals,
        lengths[:, None],
        out=np.zeros_like(normals),
        where=lengths[:, None] > 1.0e-12,
    )
    unit_normals[unit_normals[:, 2] < 0.0] *= -1.0
    # Dense cloth triangles are small.  Painter-ordering them by centroid height
    # is a deterministic and sufficiently local approximation to a CPU z-buffer.
    for face in np.argsort(centroids_z, kind="stable"):
        polygon = pixels[triangles[face]]
        cv2.fillConvexPoly(height, polygon, float(centroids_z[face]))
        cv2.fillConvexPoly(normal_x, polygon, float(unit_normals[face, 0]))
        cv2.fillConvexPoly(normal_y, polygon, float(unit_normals[face, 1]))
        cv2.fillConvexPoly(normal_z, polygon, float(unit_normals[face, 2]))
    return _TopProjection(
        occupancy=occupancy,
        height=height,
        normal_x=normal_x,
        normal_y=normal_y,
        normal_z=normal_z,
    )


def _projection_raster(
    points: np.ndarray, triangles: np.ndarray, pixel_size: float
) -> tuple[np.ndarray, np.ndarray]:
    xy = np.asarray(points[:, :2], dtype=np.float64)
    center = (xy.min(axis=0) + xy.max(axis=0)) * 0.5
    half = np.ceil(np.ptp(xy, axis=0) / (2 * pixel_size)).astype(np.int64) + 2
    shape_xy = 2 * half + 1
    if np.any(shape_xy < 2) or np.any(shape_xy > 4096):
        raise ValueError("garment projection raster has invalid dimensions")
    pixels = np.rint((xy - center) / pixel_size).astype(np.int32) + half
    occupancy = np.zeros((int(shape_xy[1]), int(shape_xy[0])), dtype=np.uint8)
    # fillPoly treats a list of mutually overlapping contours with an even-odd
    # rule, which would erase regions covered by an even number of cloth layers.
    # Union the faces explicitly instead; fillConvexPoly is idempotent on this
    # binary target and preserves every projected layer.
    for triangle in triangles:
        cv2.fillConvexPoly(occupancy, pixels[triangle], 1)
    return occupancy, pixels


def _minimum_rectangle_pixel_area(mask: np.ndarray) -> int:
    area, _corners = _minimum_rectangle(mask)
    return area


def _minimum_rectangle(mask: np.ndarray) -> tuple[int, np.ndarray]:
    y, x = np.nonzero(mask)
    if len(x) < 3:
        raise ValueError("garment projection has no non-degenerate rectangle")
    points = np.stack((x, y), axis=1).astype(np.float32)
    rectangle = cv2.minAreaRect(points)
    _center, (width, height), _angle = rectangle
    # minAreaRect measures between pixel centers. Add one pixel in both axes so
    # the denominator represents the enclosing pixel cells themselves.
    area = int(np.ceil((float(width) + 1.0) * (float(height) + 1.0)))
    return max(area, int(np.count_nonzero(mask))), cv2.boxPoints(rectangle)


def _rectangle_union_pixel_area(mask: np.ndarray, rectangle_xy: np.ndarray) -> int:
    rectangle_mask = np.zeros(mask.shape, dtype=np.uint8)
    corners = np.rint(rectangle_xy).astype(np.int32)
    cv2.fillConvexPoly(rectangle_mask, corners, 1)
    return int(np.count_nonzero(mask.astype(bool) | rectangle_mask.astype(bool)))


def _dominant_plane_top_region(
    height: np.ndarray,
    normal_x: np.ndarray,
    normal_y: np.ndarray,
    normal_z: np.ndarray,
    occupancy: np.ndarray,
    *,
    tolerance_m: float,
    pixel_size_m: float,
    normal_z_min: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit and softly score the largest connected top surface.

    A horizontal height-band component provides a deterministic seed. The seed
    is expanded by repeatedly fitting a plane and collecting connected pixels
    close to that plane. Final residual and normal agreement become continuous
    weights, avoiding the brittle all-or-nothing area changes of a height band.
    """

    seed = _dominant_planar_top_mask(
        height,
        normal_z,
        occupancy,
        tolerance_m=tolerance_m,
        normal_z_min=normal_z_min,
    )
    if np.count_nonzero(seed) < 3:
        return np.zeros_like(occupancy, dtype=bool), np.zeros_like(height)

    rows, columns = np.indices(height.shape, dtype=np.float64)
    x = (columns - 0.5 * (height.shape[1] - 1)) * pixel_size_m
    y = (rows - 0.5 * (height.shape[0] - 1)) * pixel_size_m
    plane = _fit_height_plane(x, y, height, seed)
    component = seed
    for _iteration in range(3):
        predicted = plane[0] * x + plane[1] * y + plane[2]
        residual = np.abs(height - predicted)
        plane_normal = np.asarray((-plane[0], -plane[1], 1.0), dtype=np.float64)
        plane_normal /= np.linalg.norm(plane_normal)
        alignment = (
            normal_x * plane_normal[0]
            + normal_y * plane_normal[1]
            + normal_z * plane_normal[2]
        )
        candidate = (
            occupancy.astype(bool)
            & np.isfinite(height)
            & (residual <= tolerance_m)
            & (alignment >= normal_z_min)
        )
        expanded = _largest_component(candidate)
        if np.count_nonzero(expanded) < 3:
            break
        component = expanded
        plane = _fit_height_plane(x, y, height, component)

    predicted = plane[0] * x + plane[1] * y + plane[2]
    normalized_residual = np.abs(height - predicted) / tolerance_m
    residual_weight = np.square(np.clip(1.0 - np.square(normalized_residual), 0.0, 1.0))
    plane_normal = np.asarray((-plane[0], -plane[1], 1.0), dtype=np.float64)
    plane_normal /= np.linalg.norm(plane_normal)
    alignment = (
        normal_x * plane_normal[0]
        + normal_y * plane_normal[1]
        + normal_z * plane_normal[2]
    )
    if normal_z_min >= 1.0:
        normal_weight = (alignment >= 1.0 - 1.0e-6).astype(np.float64)
    else:
        saturation = min(1.0, normal_z_min + 0.15)
        normal_weight = np.clip(
            (alignment - normal_z_min) / (saturation - normal_z_min),
            0.0,
            1.0,
        )
    weight = (residual_weight * normal_weight).astype(np.float32)
    connected = _largest_component(component & (weight > 0.0))
    weight[~connected] = 0.0
    return connected, weight


def _fit_height_plane(
    x: np.ndarray, y: np.ndarray, height: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    design = np.stack((x[mask], y[mask], np.ones(np.count_nonzero(mask))), axis=1)
    plane, _residuals, _rank, _singular = np.linalg.lstsq(
        design, height[mask].astype(np.float64), rcond=None
    )
    return plane


def _largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return np.zeros_like(mask, dtype=bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def _dominant_planar_top_mask(
    height: np.ndarray,
    normal_z: np.ndarray,
    occupancy: np.ndarray,
    *,
    tolerance_m: float,
    normal_z_min: float,
) -> np.ndarray:
    candidate = occupancy.astype(bool) & np.isfinite(height) & (normal_z >= normal_z_min)
    values = np.sort(np.asarray(height[candidate], dtype=np.float64))
    if not len(values):
        return np.zeros_like(occupancy, dtype=bool)

    # Find the height interval of fixed physical width containing the most
    # horizontal visible area. This accepts any absolute stack height while
    # requiring one dominant, nearly coplanar top surface.
    width = 2.0 * tolerance_m
    right = np.searchsorted(values, values + width, side="right")
    left_index = int(np.argmax(right - np.arange(len(values))))
    center = 0.5 * (values[left_index] + values[right[left_index] - 1])
    planar = candidate & (np.abs(height - center) <= tolerance_m)
    return _largest_component(planar)


def _triangle_indices(value: np.ndarray, particle_count: int) -> np.ndarray:
    triangles = np.asarray(value)
    if triangles.ndim == 1:
        if len(triangles) % 3:
            raise ValueError("triangle_indices length must be divisible by three")
        triangles = triangles.reshape((-1, 3))
    if triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles):
        raise ValueError("triangle_indices must have shape (faces, 3)")
    if triangles.dtype.kind not in "iu":
        raise ValueError("triangle_indices must be integers")
    triangles = triangles.astype(np.int32, copy=False)
    if np.any(triangles < 0) or np.any(triangles >= particle_count):
        raise ValueError("triangle index is outside the particle array")
    return triangles


def _particle_positions(value: np.ndarray, name: str) -> np.ndarray:
    points = np.asarray(value)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError(f"{name} must have shape (particles, 3)")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} must be finite")
    return points


def _planar_hull(points: np.ndarray) -> tuple[float, float, float]:
    try:
        hull = ConvexHull(np.asarray(points, dtype=np.float64))
    except Exception as exc:
        raise ValueError("particle projection has no non-degenerate convex hull") from exc
    hull_points = np.asarray(points[hull.vertices], dtype=np.float32)
    _center, (width, height), _angle = cv2.minAreaRect(hull_points)
    box_area = float(width * height)
    longest = max(float(width), float(height))
    area = float(hull.volume)
    if area <= 0.0 or box_area <= 0.0 or longest <= 0.0:
        raise ValueError("particle projection has a degenerate footprint")
    return area, box_area, min(float(width), float(height)) / longest
