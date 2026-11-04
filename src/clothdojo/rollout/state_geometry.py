"""Geometric self-intersection checks for cloth initial states."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import warp as wp

# Resolve real crossing lines before applying a physical length threshold.
SELF_INTERSECTION_TOLERANCE_M = 1.0e-7
INTERSECTION_ENDPOINT_TOLERANCE_M = 1.0e-6
MINIMUM_MATERIAL_SEPARATION_M = 0.02
MINIMUM_CROSSING_DIAMETER_M = 0.05
MINIMUM_SEVERE_REGION_FRACTION = 0.01
MINIMUM_SEVERE_REGION_RADIUS_M = 0.01
SEVERE_SELF_INTERSECTION_SCORE = 1.0


@dataclass(frozen=True, slots=True)
class SelfIntersectionMetrics:
    count: np.ndarray
    rest_area_fraction: np.ndarray
    total_length_m: np.ndarray
    maximum_component_length_m: np.ndarray
    maximum_region_fraction: np.ndarray
    maximum_region_radius_m: np.ndarray
    severity_score: np.ndarray

    @property
    def severe(self) -> np.ndarray:
        return self.severity_score >= SEVERE_SELF_INTERSECTION_SCORE


def geometry_diagnostics_settings() -> dict:
    """Version the generation rule independently of the local-crossing baseline."""
    return {
        "method": "self_intersection_severity_v2",
        "score_threshold": SEVERE_SELF_INTERSECTION_SCORE,
        "score_definition": (
            "max(max_regions(min(area/area_limit, radius/radius_limit)), "
            "nonlocal_diameter/diameter_limit)"
        ),
        "tolerance_m": SELF_INTERSECTION_TOLERANCE_M,
        "endpoint_tolerance_m": INTERSECTION_ENDPOINT_TOLERANCE_M,
        "target": "severe_self_intersection_only",
        "automatic_rejection": True,
        "minimum_region_fraction": MINIMUM_SEVERE_REGION_FRACTION,
        "minimum_region_radius_m": MINIMUM_SEVERE_REGION_RADIUS_M,
        "minimum_material_separation_m": MINIMUM_MATERIAL_SEPARATION_M,
        "minimum_crossing_diameter_m": MINIMUM_CROSSING_DIAMETER_M,
        "region_backend": "libigl_2.6.3_cgal",
        "scope": "separating_regions_and_connected_nonlocal_crossings",
    }


@wp.func
def _straddles(
    a: wp.vec3d,
    b: wp.vec3d,
    c: wp.vec3d,
    x: wp.vec3d,
    y: wp.vec3d,
    z: wp.vec3d,
    tolerance: wp.float64,
):
    normal = wp.normalize(wp.cross(b - a, c - a))
    dx = wp.dot(normal, x - a)
    dy = wp.dot(normal, y - a)
    dz = wp.dot(normal, z - a)
    return wp.min(dx, wp.min(dy, dz)) < -tolerance and wp.max(dx, wp.max(dy, dz)) > tolerance


@wp.kernel
def _intersections(
    mesh: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    faces: wp.array(dtype=wp.int32, ndim=2),
    tolerance: wp.float64,
    counts: wp.array(dtype=wp.int32),
    intersecting: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
    pairs: wp.array(dtype=wp.int32, ndim=2),
    write_pairs: bool,
):
    face = wp.tid()
    i = faces[face, 0]
    j = faces[face, 1]
    k = faces[face, 2]
    a = points[i]
    b = points[j]
    c = points[k]
    lower = wp.min(a, wp.min(b, c))
    upper = wp.max(a, wp.max(b, c))
    query = wp.mesh_query_aabb(mesh, lower, upper)
    count = int(0)  # noqa: UP018, RUF046 - Warp needs a dynamic loop accumulator.
    for other in query:
        if other <= face:
            continue
        x = faces[other, 0]
        y = faces[other, 1]
        z = faces[other, 2]
        # An edge shared by two noncoplanar faces is ordinary contact. Sharing
        # only one vertex does not exclude a crossing away from that vertex.
        shared = int(i == x or i == y or i == z)
        shared += int(j == x or j == y or j == z)
        shared += int(k == x or k == y or k == z)
        if shared >= 2:
            continue
        ad = wp.vec3d(a)
        bd = wp.vec3d(b)
        cd = wp.vec3d(c)
        xd = wp.vec3d(points[x])
        yd = wp.vec3d(points[y])
        zd = wp.vec3d(points[z])
        # Both surfaces must cross the other's plane. This excludes tangency,
        # coplanar contact and overlapping projections of separate cloth layers.
        if not _straddles(ad, bd, cd, xd, yd, zd, tolerance):
            continue
        if not _straddles(xd, yd, zd, ad, bd, cd, tolerance):
            continue
        # Warp 1.16's Moller test uses a fixed epsilon on unnormalised plane
        # distances. Metre-scale cloth triangles otherwise look coplanar.
        scale = wp.max(wp.length(bd - ad), wp.length(cd - ad))
        scale = wp.max(scale, wp.max(wp.length(yd - xd), wp.length(zd - xd)))
        if wp.intersect_tri_tri(
            wp.vec3d(0.0),
            (bd - ad) / scale,
            (cd - ad) / scale,
            (xd - ad) / scale,
            (yd - ad) / scale,
            (zd - ad) / scale,
        ):
            if write_pairs:
                pairs[offsets[face] + count, 0] = face
                pairs[offsets[face] + count, 1] = other
            count += 1
            wp.atomic_max(intersecting, face, 1)
            wp.atomic_max(intersecting, other, 1)
    counts[face] = count


def self_intersection_metrics(
    particle_q: np.ndarray, triangles: np.ndarray, rest_points: np.ndarray, *, device: str = "cpu"
) -> tuple[np.ndarray, np.ndarray]:
    """The pair-count/rest-area audit baseline; not the generation predicate."""
    result = measure_self_intersections(particle_q, triangles, rest_points, device=device)
    return result.count, result.rest_area_fraction


def measure_self_intersections(
    particle_q: np.ndarray, triangles: np.ndarray, rest_points: np.ndarray, *, device: str = "cpu"
) -> SelfIntersectionMetrics:
    """Measure proper transverse crossings of cloth triangles.

    Two BVH passes allocate exactly enough space for every pair, independent of
    the solver's contact buffer limit. Lengths use saved world positions in metres;
    rest_points must also be in metres, for material separation. No state is changed.
    """

    points = np.asarray(particle_q)
    faces = np.asarray(triangles)
    if points.ndim != 3 or points.shape[2] != 3 or not np.isfinite(points).all():
        raise ValueError("particle_q must contain finite (states, particles, 3) positions")
    if (
        faces.ndim != 2
        or faces.shape[1] != 3
        or not len(faces)
        or not np.issubdtype(faces.dtype, np.integer)
    ):
        raise ValueError("triangles must be a non-empty integer array of shape (faces, 3)")
    if np.any(faces < 0) or np.any(faces >= points.shape[1]):
        raise ValueError("triangle indices are outside the particle array")
    rest = np.asarray(rest_points, dtype=np.float64)
    if rest.shape != points.shape[1:] or not np.isfinite(rest).all():
        raise ValueError("rest_points must match the finite garment particle positions")
    rest_faces = rest[faces]
    areas = np.linalg.norm(
        np.cross(rest_faces[:, 1] - rest_faces[:, 0], rest_faces[:, 2] - rest_faces[:, 0]), axis=1
    )
    total_area = float(areas.sum())
    if total_area <= 0.0:
        raise ValueError("garment rest surface must have positive area")
    indices = wp.array(np.ascontiguousarray(faces, dtype=np.int32), dtype=wp.int32, device=device)
    counts = wp.empty(len(faces), dtype=wp.int32, device=device)
    intersecting = wp.zeros(len(faces), dtype=wp.int32, device=device)
    result = []
    fractions = []
    total_lengths = []
    maximum_lengths = []
    region_fractions = []
    region_radii = []
    scores = []
    offsets = wp.zeros(len(faces), dtype=wp.int32, device=device)
    empty_pairs = wp.empty((0, 2), dtype=wp.int32, device=device)
    for state in points:
        intersecting.zero_()
        vertices = wp.array(
            np.ascontiguousarray(state, dtype=np.float32), dtype=wp.vec3, device=device
        )
        mesh = wp.Mesh(points=vertices, indices=indices.flatten())
        wp.launch(
            _intersections,
            dim=len(faces),
            inputs=[
                mesh.id,
                vertices,
                indices,
                SELF_INTERSECTION_TOLERANCE_M,
                counts,
                intersecting,
                offsets,
                empty_pairs,
                False,
            ],
            device=device,
        )
        sizes = counts.numpy()
        pair_count = int(sizes.sum(dtype=np.int64))
        if pair_count == 0:
            result.append(0)
            fractions.append(0.0)
            total_lengths.append(0.0)
            maximum_lengths.append(0.0)
            region_fractions.append(0.0)
            region_radii.append(0.0)
            scores.append(0.0)
            continue
        if pair_count > np.iinfo(np.int32).max:
            raise ValueError("too many intersecting triangle pairs to represent")
        offsets.assign(np.cumsum(sizes, dtype=np.int32) - sizes)
        pairs = wp.empty((pair_count, 2), dtype=wp.int32, device=device)
        wp.launch(
            _intersections,
            dim=len(faces),
            inputs=[
                mesh.id,
                vertices,
                indices,
                SELF_INTERSECTION_TOLERANCE_M,
                counts,
                intersecting,
                offsets,
                pairs,
                True,
            ],
            device=device,
        )
        pair_indices = pairs.numpy()
        endpoints, lengths = _intersection_segments(state, faces, pair_indices)
        # A shared vertex alone can pass a triangle intersection predicate.
        # It is contact, not penetration, and must not join distinct contours.
        resolved = lengths > INTERSECTION_ENDPOINT_TOLERANCE_M
        pair_indices = pair_indices[resolved]
        endpoints, lengths = endpoints[resolved], lengths[resolved]
        result.append(len(pair_indices))
        fractions.append(float(areas[np.unique(pair_indices)].sum() / total_area))
        total_lengths.append(float(lengths.sum()))
        maximum_lengths.append(_maximum_component_length(faces, pair_indices, endpoints, lengths))
        from clothdojo.rollout.state_regions import intersection_regions

        regions = intersection_regions(state, faces, rest, pair_indices)
        region_fractions.append(max((r.rest_area_fraction for r in regions), default=0.0))
        region_radii.append(max((r.radius_m for r in regions), default=0.0))
        # Area and radius must describe the SAME region; unrelated tiny cuts
        # must not accumulate into a severe result.
        region_score = max(
            (min(r.rest_area_fraction / MINIMUM_SEVERE_REGION_FRACTION,
                 r.radius_m / MINIMUM_SEVERE_REGION_RADIUS_M) for r in regions),
            default=0.0,
        )
        crossing_diameter = _nonlocal_crossing_diameter(state, faces, rest, pair_indices, endpoints)
        scores.append(max(region_score, crossing_diameter / MINIMUM_CROSSING_DIAMETER_M))
    return SelfIntersectionMetrics(
        count=np.asarray(result, dtype=np.int64),
        rest_area_fraction=np.asarray(fractions, dtype=np.float64),
        total_length_m=np.asarray(total_lengths, dtype=np.float64),
        maximum_component_length_m=np.asarray(maximum_lengths, dtype=np.float64),
        maximum_region_fraction=np.asarray(region_fractions, dtype=np.float64),
        maximum_region_radius_m=np.asarray(region_radii, dtype=np.float64),
        severity_score=np.asarray(scores, dtype=np.float64),
    )


def _nonlocal_crossing_diameter(points, faces, rest_m, pairs, endpoints) -> float:
    """Spatial diameter of each contiguous crossing between distant material.

    Rest-space Euclidean distance is a lower bound on intrinsic separation,
    not a geodesic or penetration depth. Never bridge discarded local pieces.
    """
    if not len(pairs):
        return 0.0
    midpoints = np.mean(endpoints, axis=1)
    mapped = []
    for side in (0, 1):
        triangles = np.asarray(points, dtype=np.float64)[faces[pairs[:, side]]]
        origin = triangles[:, 0]
        e, f = triangles[:, 1] - origin, triangles[:, 2] - origin
        normal = np.cross(e, f)
        denominator = np.sum(normal * normal, axis=1)
        if np.any(denominator <= 0):
            raise ValueError("degenerate triangle in resolved crossing")
        u = np.einsum("ni,ni->n", np.cross(midpoints-origin, f), normal) / denominator
        v = np.einsum("ni,ni->n", np.cross(e, midpoints-origin), normal) / denominator
        reference = rest_m[faces[pairs[:, side]]]
        mapped.append(reference[:, 0] + u[:, None]*(reference[:, 1]-reference[:, 0]) +
                      v[:, None]*(reference[:, 2]-reference[:, 0]))
    nonlocal_pairs = np.linalg.norm(mapped[0]-mapped[1], axis=1) >= MINIMUM_MATERIAL_SEPARATION_M
    selected_pairs, selected_endpoints = pairs[nonlocal_pairs], endpoints[nonlocal_pairs]
    labels = _intersection_components(faces, selected_pairs, selected_endpoints)
    maximum_squared = 0.0
    for label in np.unique(labels):
        vertices = selected_endpoints[labels == label].reshape(-1, 3)
        for start in range(0, len(vertices), 128):
            squared = np.sum((vertices[start:start+128, None]-vertices[None])**2, axis=2)
            maximum_squared = max(maximum_squared, float(squared.max(initial=0.0)))
    return maximum_squared**0.5


def _plane_segments(triangles: np.ndarray, other: np.ndarray) -> np.ndarray:
    normal = np.cross(other[:, 1] - other[:, 0], other[:, 2] - other[:, 0])
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    distances = np.einsum("nkj,nj->nk", triangles - other[:, :1], normal)
    segments = np.empty((len(triangles), 2, 3), dtype=np.float64)
    used = np.zeros(len(triangles), dtype=np.int32)
    for i, j in ((0, 1), (1, 2), (2, 0)):
        crosses = ((distances[:, i] <= 0) & (distances[:, j] > 0)) | (
            (distances[:, j] <= 0) & (distances[:, i] > 0)
        )
        ids = np.flatnonzero(crosses)
        ratio = distances[ids, i] / (distances[ids, i] - distances[ids, j])
        segments[ids, used[ids]] = triangles[ids, i] + ratio[:, None] * (
            triangles[ids, j] - triangles[ids, i]
        )
        used[ids] += 1
    if np.any(used != 2):
        raise ValueError("could not resolve a transverse triangle intersection")
    return segments


def _intersection_segments(points, faces, pairs) -> tuple[np.ndarray, np.ndarray]:
    triangles = np.asarray(points, dtype=np.float64)[faces[pairs]]
    first = _plane_segments(triangles[:, 0], triangles[:, 1])
    second = _plane_segments(triangles[:, 1], triangles[:, 0])
    direction = first[:, 1] - first[:, 0]
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    first_interval = np.einsum("nij,nj->ni", first - first[:, :1], direction)
    second_interval = np.einsum("nij,nj->ni", second - first[:, :1], direction)
    low = np.maximum(first_interval.min(axis=1), second_interval.min(axis=1))
    high = np.minimum(first_interval.max(axis=1), second_interval.max(axis=1))
    lengths = np.maximum(high - low, 0.0)
    endpoints = first[:, :1] + np.stack([low, high], axis=1)[:, :, None] * direction[:, None]
    if not np.isfinite(endpoints).all() or not np.isfinite(lengths).all():
        raise ValueError("non-finite intersection segments")
    return endpoints, lengths


def _maximum_component_length(faces, pairs, endpoints, lengths) -> float:
    labels = _intersection_components(faces, pairs, endpoints)
    if not len(labels):
        return 0.0
    return float(np.bincount(labels, weights=lengths).max(initial=0.0))


def _intersection_components(faces, pairs, endpoints) -> np.ndarray:
    """Join coincident endpoints only when both intersecting sheets continue.

    Requiring mesh continuity prevents nearby, unrelated layers from being merged.
    Open contours at garment boundaries are handled just like closed contours.
    """
    if not len(pairs):
        return np.empty(0, dtype=np.int64)
    parent = np.arange(len(pairs))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    face_vertices = {int(f): set(map(int, faces[f])) for f in np.unique(pairs)}

    def continues(i, j):
        a, b = pairs[i]
        c, d = pairs[j]
        return (
            bool(face_vertices[a] & face_vertices[c]) and bool(face_vertices[b] & face_vertices[d])
        ) or (
            bool(face_vertices[a] & face_vertices[d]) and bool(face_vertices[b] & face_vertices[c])
        )

    tolerance = INTERSECTION_ENDPOINT_TOLERANCE_M
    buckets: dict[tuple[int, int, int], list[tuple[int, np.ndarray]]] = {}
    neighbors = tuple(product((-1, 0, 1), repeat=3))
    for index, segment in enumerate(endpoints):
        for point in segment:
            cell = tuple(np.floor(point / tolerance).astype(np.int64))
            for delta in neighbors:
                key = tuple(a + b for a, b in zip(cell, delta, strict=True))
                for other, previous in buckets.get(key, ()):
                    if other == index or root(other) == root(index):
                        continue
                    if np.linalg.norm(point - previous) <= tolerance and continues(index, other):
                        parent[root(index)] = root(other)
            buckets.setdefault(cell, []).append((index, point))
    return np.asarray([root(i) for i in range(len(pairs))], dtype=np.int64)
