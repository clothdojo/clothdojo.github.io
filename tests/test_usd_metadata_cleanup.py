import hashlib
import runpy
from pathlib import Path

import numpy as np
import pytest

Usd = pytest.importorskip("pxr.Usd")
UsdGeom = pytest.importorskip("pxr.UsdGeom")

ROOT = Path(__file__).resolve().parents[1]
HELPERS = runpy.run_path(str(ROOT / "scripts/sanitize_usd_metadata.py"))
SANITIZE = HELPERS["sanitize_usd_metadata"]


def layer_with_mesh(tmp_path):
    source = tmp_path / "source.usda"
    stage = Usd.Stage.CreateNew(str(source))
    mesh = UsdGeom.Mesh.Define(stage, "/Geometry")
    mesh.CreatePointsAttr([(0.0, 0.0, 0.0), (0.25, 0.0, 0.0), (0.0, 0.5, 0.0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    stage.GetRootLayer().documentation = str(tmp_path / "old-source.urdf")
    stage.GetRootLayer().Save()
    return source, stage


def test_source_and_mesh_values_preserved(tmp_path):
    source, stage = layer_with_mesh(tmp_path)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path / "clean.usda"
    removed = SANITIZE(source, output)
    assert removed == [{"spec": "/", "field": "documentation"}]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    clean = Usd.Stage.Open(str(output))
    assert clean.GetRootLayer().documentation == ""
    for name in ["points", "faceVertexCounts", "faceVertexIndices"]:
        old = np.asarray(stage.GetPrimAtPath("/Geometry").GetAttribute(name).Get())
        new = np.asarray(clean.GetPrimAtPath("/Geometry").GetAttribute(name).Get())
        assert old.dtype == new.dtype and old.shape == new.shape
        assert old.tobytes() == new.tobytes()


def test_public_notices_and_creator_preserved(tmp_path):
    values = {
        "creator": "URDF USD Converter",
        "license": "MIT",
        "copyright": str(tmp_path / "notice"),
        "scale": 1.0,
    }
    clean, removed = HELPERS["clean_dictionary"](values)
    assert clean == values
    assert removed == []


def test_private_provenance_dictionary_only(tmp_path):
    values = {
        "author": "example author",
        "origin": {"sourcePath": str(tmp_path / "input.usd"), "revision": "public-v1"},
        "metersPerUnit": 1.0,
    }
    clean, removed = HELPERS["clean_dictionary"](values)
    assert clean == {"origin": {"revision": "public-v1"}, "metersPerUnit": 1.0}
    assert set(removed) == {"author", "origin.sourcePath"}


def test_existing_output_never_overwritten(tmp_path):
    source, stage = layer_with_mesh(tmp_path)
    output = tmp_path / "existing.usda"
    output.write_text("preserved")
    with pytest.raises(FileExistsError):
        SANITIZE(source, output)
    assert output.read_text() == "preserved"
