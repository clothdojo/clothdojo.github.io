"""Material regions separated by individual cloth intersection contours.

These are geometric measurements for the unified score in state_geometry.
Open contours that do not separate the material surface have no enclosed region.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class IntersectionRegion:
    rest_area_fraction: float
    radius_m: float
    original_faces: np.ndarray

def _components(count, connections):
    parent = np.arange(count)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in connections:
        parent[root(a)] = root(b)
    return np.array([root(i) for i in range(count)], dtype=np.int64)


def _edges(faces):
    incident = defaultdict(list)
    for i, (a, b, c) in enumerate(faces):
        for x, y in ((a, b), (b, c), (c, a)):
            incident[tuple(sorted((int(x), int(y))))].append(i)
    return incident


def _areas(points, faces):
    t = points[faces]
    return np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1) / 2


def intersection_regions(points, faces, rest_points, resolved_pairs) -> list[IntersectionRegion]:
    """Cut along each proper crossing contour without joining the two sheets.

    CGAL subdivides faces exactly at intersections. Its original material vertex
    indices retain sheet connectivity; IM is used only to identify coincident
    crossing edges, never to weld intersecting cloth sheets. Each contour is cut
    separately so unrelated local crossings cannot jointly isolate a large area.
    Area is transported back to the garment's rest surface using birth triangles.
    """
    if not len(resolved_pairs):
        return []
    from igl.copyleft.cgal import remesh_self_intersections

    q = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    rest = np.asarray(rest_points, dtype=np.float64)
    original_edges = _edges(faces)
    if any(len(incident) > 2 for incident in original_edges.values()):
        raise ValueError("self-intersection region analysis requires manifold cloth edges")
    original = _components(len(faces), [v for v in original_edges.values() if len(v) == 2])
    vertices, mesh, _, birth, unique = remesh_self_intersections(q, faces)
    if not np.isfinite(vertices).all():
        raise ValueError("self-intersection remeshing produced non-finite positions")
    material = _edges(mesh)
    if any(len(incident) > 2 for incident in material.values()):
        raise ValueError("self-intersection remeshing changed material manifoldness")
    geometric_edges = {
        edge: tuple(sorted((int(unique[edge[0]]), int(unique[edge[1]]))))
        for edge in material
    }
    physical = defaultdict(set)
    for edge, incident in material.items():
        physical[geometric_edges[edge]].update(map(int, birth[incident]))
    valid = {tuple(sorted(map(int, p))) for p in resolved_pairs}
    cut = {
        edge for edge, incident in physical.items()
        if any(tuple(sorted((a, b))) in valid for a in incident for b in incident if a != b)
    }
    before = _components(len(mesh), [v for v in material.values() if len(v) == 2])
    correspondence = [np.unique(original[birth[before == g]]) for g in np.unique(before)]
    if (
        any(len(group) != 1 for group in correspondence)
        or len(correspondence) != len(np.unique(original))
        or len({int(group[0]) for group in correspondence}) != len(correspondence)
    ):
        raise ValueError("self-intersection remeshing changed material connectivity")
    source_area = _areas(q, faces)
    remeshed_area = _areas(vertices, mesh)
    if np.any(source_area <= 0) or not np.allclose(
        np.bincount(birth, weights=remeshed_area, minlength=len(faces)),
        source_area, rtol=1e-6, atol=1e-12,
    ):
        raise ValueError("self-intersection remeshing did not preserve triangle areas")
    rest_area = _areas(rest, faces)
    weights = remeshed_area / source_area[birth] * rest_area[birth]
    cut_list = sorted(cut)
    nodes = defaultdict(list)
    for i, edge in enumerate(cut_list):
        for v in edge:
            nodes[v].append(i)
    contour_labels = _components(
        len(cut_list), [(v[0], i) for v in nodes.values() for i in v[1:]]
    )
    regions = []
    for contour in np.unique(contour_labels):
        contour_edges = {cut_list[i] for i in np.flatnonzero(contour_labels == contour)}
        connections = [
            incident for edge, incident in material.items()
            if len(incident) == 2 and geometric_edges[edge] not in contour_edges
        ]
        after = _components(len(mesh), connections)
        boundary = defaultdict(list)
        for edge, incident in material.items():
            if geometric_edges[edge] in contour_edges:
                for face in incident:
                    boundary[int(after[face])].append(edge)
        totals = np.bincount(after, weights=weights, minlength=len(mesh))
        for group in np.unique(before):
            labels = np.unique(after[before == group])
            # The smaller side is the operational affected region, following GIA.
            # Disconnected garment panels are compared within their own component.
            largest = labels[np.argmax(totals[labels])]
            for label in labels[labels != largest]:
                region = np.flatnonzero(after == label)
                edges = sorted(set(boundary[int(label)]))
                if not edges:
                    raise ValueError("isolated cloth region has no intersection boundary")
                samples = np.concatenate([
                    vertices[np.unique(mesh[region])], vertices[mesh[region]].mean(axis=1)
                ])
                nearest = np.full(len(samples), np.inf)
                for a, b in vertices[np.asarray(edges)]:
                    direction = b - a
                    t = np.clip(
                        (samples - a) @ direction / max(float(direction @ direction), 1e-30),
                        0, 1,
                    )
                    nearest = np.minimum(
                        nearest, np.linalg.norm(samples - a - t[:, None] * direction, axis=1)
                    )
                regions.append(IntersectionRegion(
                    rest_area_fraction=float(totals[label] / rest_area.sum()),
                    radius_m=float(nearest.max(initial=0.0)),
                    original_faces=np.unique(birth[region]),
                ))
    return regions
