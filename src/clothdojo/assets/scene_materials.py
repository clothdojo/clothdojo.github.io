"""Material-only libraries extracted from pinned Isaac and GenieSim assets."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from functools import cache
from pathlib import Path

from clothdojo.assets.materials import MaterialAsset, MaterialCatalog

SCENE_MATERIALS = Path(__file__).with_name("rendering") / "scene_materials.json"


@cache
def load_scene_materials() -> tuple[dict, ...]:
    value = json.loads(SCENE_MATERIALS.read_text())
    if value["schema"] != "clothdojo_scene_materials_v1":
        raise ValueError("invalid scene material catalog")
    items = value["materials"]
    if len({item["id"] for item in items}) != len(items):
        raise ValueError("duplicate scene material IDs")
    for item in items:
        if not re.fullmatch(r"(?:isaac|geniesim)_[a-z0-9_]+", item["id"]):
            raise ValueError("invalid scene material ID")
        path = Path(item["path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("scene material source must stay within visual-assets")
        if item["split"] not in ("train", "test"):
            raise ValueError("invalid scene material split")
    return tuple(items)


def table_material_pool(catalog: MaterialCatalog) -> tuple[MaterialAsset, ...]:
    return catalog.pool("table") + tuple(
        MaterialAsset(item["id"], "table", ("source",), (1000.0, 1000.0))
        for item in load_scene_materials()
    )


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _recipe_digest(item: dict) -> str:
    return hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()


def atlas_top_uv(root: Path, item: dict):
    """Read the top faces through the source's authored alignment wrapper."""
    import numpy as np
    from pxr import Usd, UsdGeom

    source = root / item["geometry_path"]
    if _digest(source) != item["geometry_sha256"]:
        raise ValueError(f"atlas source alignment changed: {source}")
    stage = Usd.Stage.Open(str(source.resolve()))
    if UsdGeom.GetStageUpAxis(stage) != "Z":
        raise ValueError("atlas audit requires the aligned Z-up source")
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(item["mesh"]))
    if not mesh or not np.all(np.asarray(mesh.GetFaceVertexCountsAttr().Get()) == 3):
        raise ValueError("atlas audit requires the selected triangulated source mesh")
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    matrix = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(mesh.GetPrim()))
    points = (np.column_stack((points, np.ones(len(points)))) @ matrix)[:, :3]
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    faces = points[indices]
    normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    area2 = np.linalg.norm(normals, axis=1)
    selected = ((normals[:, 2] / np.maximum(area2, 1e-12) > 0.999)
                & (faces[:, :, 2].mean(axis=1) > points[:, 2].max() - 0.001))
    uv = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
    if uv.GetInterpolation() != "faceVarying" or not selected.any():
        raise ValueError("atlas audit could not identify top-face UVs")
    return np.asarray(uv.ComputeFlattened()).reshape(-1, 3, 2)[selected]


def validate_atlas_crop(root: Path, item: dict) -> dict:
    import numpy as np

    triangles = atlas_top_uv(root, item)
    x0, y0, x1, y1 = item["crop_xyxy"]
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("invalid atlas crop rectangle")
    # Image rows run downward, while USD texture V runs upward.
    x, y = np.meshgrid(np.linspace(x0, x1, 65), np.linspace(1 - y1, 1 - y0, 65))
    samples = np.column_stack((x.ravel(), y.ravel()))
    covered = np.zeros(len(samples), dtype=bool)
    for a, b, c in triangles:
        edges = np.column_stack((b - a, c - a))
        if abs(np.linalg.det(edges)) < 1e-12:
            continue
        bary = (samples - a) @ np.linalg.inv(edges).T
        covered |= ((bary >= -1e-6).all(axis=1) & (bary.sum(axis=1) <= 1 + 1e-6))
    if not covered.all():
        raise ValueError(f"atlas crop leaves the table top UV island: {item['id']}")
    return {"top_triangles": len(triangles), "crop_samples": len(samples),
            "top_uv_coverage": float(covered.mean()),
            "uv_min": triangles.reshape(-1, 2).min(axis=0).tolist(),
            "uv_max": triangles.reshape(-1, 2).max(axis=0).tolist()}


def extract_scene_materials(root: Path) -> list[dict]:
    """Keep MDL graphs and dependencies; crop only explicitly selected atlas regions."""
    from PIL import Image
    from pxr import Sdf, Usd, UsdShade

    root = root.resolve()
    origins = json.loads(SCENE_MATERIALS.read_text())["sources"]
    results = []
    for item in load_scene_materials():
        source = root / item["path"]
        if _digest(source) != item["source_sha256"]:
            raise ValueError(f"scene material source changed: {source}")
        output = root / "extracted-materials" / item["id"]
        output.mkdir(parents=True, exist_ok=True)
        stage = Usd.Stage.CreateInMemory()
        material = UsdShade.Material.Define(stage, "/Material")
        stage.SetDefaultPrim(material.GetPrim())
        inputs = {item["path"]: _digest(source)}
        uv_audit = None
        if item["kind"] == "mdl":
            # These selected NVIDIA wrappers import only runtime OmniPBR and
            # standard MDL modules. Keep their authored texture paths intact.
            shader = UsdShade.Shader.Define(stage, "/Material/Shader")
            shader.CreateImplementationSourceAttr("sourceAsset")
            shader.SetSourceAsset(Sdf.AssetPath(f"./{source.name}"), "mdl")
            shader.SetSourceAssetSubIdentifier(item["subidentifier"], "mdl")
            material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
            shutil.copy2(source, output / source.name)
            for relative in re.findall(r'texture_2d\("([^\"]+)"', source.read_text()):
                dependency = (source.parent / relative).resolve()
                target = output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(dependency, target)
                inputs[str(dependency.relative_to(root))] = _digest(dependency)
        elif item["kind"] == "usd":
            uv_audit = validate_atlas_crop(root, item)
            inputs[item["geometry_path"]] = item["geometry_sha256"]
            material.GetPrim().GetReferences().AddReference(str(source), item["prim"])
            if not material.GetSurfaceOutput("mdl").HasConnectedSource():
                raise ValueError(f"material has no MDL surface: {item['id']}")
            stage = Usd.Stage.Open(stage.Flatten())
            for prim in stage.Traverse():
                for attr in prim.GetAttributes():
                    if attr.GetTypeName() != Sdf.ValueTypeNames.Asset:
                        continue
                    value = attr.Get()
                    if not value or not value.path:
                        continue
                    if not value.resolvedPath:
                        if value.path == "OmniPBR.mdl":
                            continue
                        raise FileNotFoundError(value.path)
                    dependency = Path(value.resolvedPath)
                    inputs[str(dependency.relative_to(root))] = _digest(dependency)
                    target = output / "textures" / dependency.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    # All channels use the same normalized source rectangle;
                    # normal, roughness and metallic maps remain aligned.
                    with Image.open(dependency) as image:
                        box = tuple(round(v * image.size[i % 2])
                                    for i, v in enumerate(item["crop_xyxy"]))
                        image.crop(box).save(target.with_suffix(".png"))
                    attr.Set(Sdf.AssetPath(f"./textures/{dependency.stem}.png"))
                    if attr.GetName() == "inputs:diffuse_texture":
                        attr.SetMetadata("colorSpace", "sRGB")
                    elif attr.GetName() in {
                        "inputs:normalmap_texture", "inputs:metallic_texture",
                        "inputs:reflectionroughness_texture",
                    }:
                        attr.SetMetadata("colorSpace", "raw")
        else:
            raise ValueError(f"unknown scene material kind: {item['kind']}")
        stage.GetRootLayer().Export(str(output / "material.usda"))
        manifest = {
            "id": item["id"], "recipe_sha256": _recipe_digest(item),
            "source": item["source"], "license": item["license"],
            "origin": origins[item["source"]],
            "adaptation": {key: item[key] for key in ("path", "prim", "crop_xyxy") if key in item},
            "uv_audit": uv_audit,
            "input_sha256": inputs,
            "output_sha256": {
                str(path.relative_to(output)): _digest(path)
                for path in sorted(output.rglob("*"))
                if path.is_file() and path.name != "manifest.json"
            },
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        results.append(manifest)
    return results


def require_scene_materials(root: Path, ids: set[str]) -> None:
    for item in load_scene_materials():
        if item["id"] not in ids:
            continue
        folder = root / "extracted-materials" / item["id"]
        manifest_path = folder / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Extract scene materials first: scripts/data/extract_scene_materials.py ({folder})"
            )
        manifest = json.loads(manifest_path.read_text())
        if manifest["recipe_sha256"] != _recipe_digest(item):
            raise ValueError(f"scene material extraction recipe changed: {item['id']}")
        for name, digest in manifest["output_sha256"].items():
            if _digest(folder / name) != digest:
                raise ValueError(f"extracted material changed: {folder / name}")


def bind_scene_material(stage, root: Path, asset_id: str, *, look: str, prim: str) -> bool:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    item = next((item for item in load_scene_materials() if item["id"] == asset_id), None)
    if item is None:
        return False
    path = root / "extracted-materials" / asset_id / "material.usda"
    if not path.is_file():
        raise FileNotFoundError(path)
    if stage.GetPrimAtPath(look):
        stage.RemovePrim(look)
    material = UsdShade.Material.Define(stage, look)
    material.GetPrim().GetReferences().AddReference(str(path.resolve()), "/Material")
    if item["kind"] == "usd":
        # The project box authors its top face first, with metre-based UVs.
        # Atlas crops are not seamless: fit one crop across that top face.
        uv = UsdGeom.PrimvarsAPI(stage.GetPrimAtPath(prim)).GetPrimvar("st").Get()
        if uv is None or len(uv) < 4 or uv[2][0] <= 0 or uv[2][1] <= 0:
            raise ValueError("atlas table material requires the project box UVs")
        for child in Usd.PrimRange(material.GetPrim()):
            if child.IsA(UsdShade.Shader):
                UsdShade.Shader(child).CreateInput("texture_scale", Sdf.ValueTypeNames.Float2).Set(
                    Gf.Vec2f(1.0 / uv[2][0], 1.0 / uv[2][1])
                )
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(prim)).Bind(material)
    return True
