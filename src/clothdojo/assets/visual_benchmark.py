"""Fixed asset identities for paired visual-generalization experiments."""

from __future__ import annotations

import hashlib
import json
import math
from functools import cache
from pathlib import Path

from clothdojo.assets.materials import load_material_catalog
from clothdojo.assets.scene_materials import SCENE_MATERIALS, load_scene_materials

VISUAL_PROFILES = (
    "train", "test_cloth", "test_background", "test_lighting", "test_all"
)
VISUAL_PROTOCOL = Path(__file__).with_name("rendering") / "benchmark.json"
_COMPONENTS = ("cloth", "table", "ground", "lighting", "background")


@cache
def load_visual_protocol(path: str | Path = VISUAL_PROTOCOL) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema") != "clothdojo_visual_benchmark_v1":
        raise ValueError("invalid visual benchmark schema")
    if set(value["splits"]) != {"train", "test"}:
        raise ValueError("visual benchmark requires train/test splits only")
    catalog = load_material_catalog()
    lighting = value["indoor_lighting"]
    ranges = list(lighting["position_range"])
    for condition in lighting["conditions"].values():
        ranges.extend((condition["intensity"], condition["temperature"]))
        if condition["intensity"][0] <= 0 or not (
            1000 <= condition["temperature"][0] <= condition["temperature"][1] <= 10000
        ):
            raise ValueError("invalid indoor light intensity/temperature")
    if any(len(bounds) != 2 or not all(math.isfinite(x) for x in bounds)
           or bounds[0] > bounds[1] for bounds in ranges):
        raise ValueError("invalid indoor light sampling range")
    if len(lighting["position_range"]) != 3 or not 0 <= lighting["fill_ratio"] <= 1:
        raise ValueError("invalid indoor light geometry/fill")
    backgrounds = value["backgrounds"]
    background_ids = [item["id"] for item in backgrounds]
    if len(background_ids) != len(set(background_ids)):
        raise ValueError("duplicate background asset identities")
    for item in backgrounds:
        source = Path(item["path"])
        if source.is_absolute() or ".." in source.parts:
            raise ValueError("background paths must stay within visual-assets")
        if len(item["translation"]) != 3 or item["scale"] <= 0:
            raise ValueError("invalid background placement")
    known = {
        **{role: {item.id for item in catalog.pool(role)} for role in _COMPONENTS[:3]},
        "lighting": set(value["indoor_lighting"]["conditions"]),
        "background": set(background_ids),
    }
    for material in load_scene_materials():
        known["table"].add(material["id"])
        value["splits"][material["split"]]["table"].append(material["id"])
    for component in _COMPONENTS:
        assigned = set()
        for split in ("train", "test"):
            ids = value["splits"][split][component]
            if not ids or len(ids) != len(set(ids)):
                raise ValueError(f"empty or duplicated {split}/{component} identities")
            if assigned.intersection(ids):
                raise ValueError(f"visual split leakage: {component}")
            assigned.update(ids)
        if assigned != known[component]:
            raise ValueError(f"visual split coverage mismatch: {component}")
        for family in value.get("families", {}).get(component, {}).values():
            if not set(family) <= known[component]:
                raise ValueError(f"unknown visual family member: {component}")
            partitions = [
                split for split in ("train", "test")
                if set(family).intersection(value["splits"][split][component])
            ]
            if len(partitions) > 1:
                raise ValueError(f"visual family crosses splits: {component}")
    return value


def visual_pool(profile: str, component: str) -> tuple[str, ...]:
    if profile not in VISUAL_PROFILES or component not in _COMPONENTS:
        raise ValueError(f"unknown visual profile/component: {profile}/{component}")
    if profile == "test_all" or profile == f"test_{component}":
        split = "test"
    else:
        split = "train"
    return tuple(load_visual_protocol()["splits"][split][component])


def visual_protocol_identity() -> dict[str, str]:
    from clothdojo.assets.materials import DEFAULT_MATERIAL_CATALOG

    return {
        "protocol_sha256": hashlib.sha256(VISUAL_PROTOCOL.read_bytes()).hexdigest(),
        "materials_sha256": hashlib.sha256(DEFAULT_MATERIAL_CATALOG.read_bytes()).hexdigest(),
        "scene_materials_sha256": hashlib.sha256(SCENE_MATERIALS.read_bytes()).hexdigest(),
    }


def require_background_assets(root: Path, profile: str) -> None:
    ids = set(visual_pool(profile, "background"))
    for item in load_visual_protocol()["backgrounds"]:
        if item["id"] not in ids:
            continue
        source = root / item["path"]
        if not source.is_file():
            raise FileNotFoundError(f"background asset is missing: {source}")
        with source.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != item["source_sha256"]:
            raise ValueError(f"background source changed: {source}")


def apply_background(stage, root: Path, asset_id: str) -> None:
    """Place render-only scenery; task geometry and simulation stay unchanged."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

    item = next(
        (item for item in load_visual_protocol()["backgrounds"] if item["id"] == asset_id),
        None,
    )
    if item is None:
        raise ValueError(f"unknown background: {asset_id}")
    source = root / item["path"]
    if not source.is_file():
        raise FileNotFoundError(source)
    anchor = UsdGeom.Xform.Define(stage, "/World/BenchmarkBackground")
    asset = UsdGeom.Xform.Define(stage, "/World/BenchmarkBackground/Asset")
    asset.GetPrim().GetReferences().AddReference(str(source.resolve()))
    transform = UsdGeom.Xformable(anchor)
    transform.AddTranslateOp().Set(Gf.Vec3d(*item["translation"]))
    transform.AddRotateZOp().Set(float(item["yaw"]))
    transform.AddScaleOp().Set(Gf.Vec3f(item["scale"]))
    for relative in item["hide_prims"]:
        prim = stage.GetPrimAtPath(f"/World/BenchmarkBackground/Asset/{relative}")
        if not prim:
            raise ValueError(f"background prim is missing: {relative}")
        prim.SetActive(False)
    for prim in Usd.PrimRange(asset.GetPrim()):
        if prim.HasAPI(UsdLux.LightAPI):
            # Lighting is controlled by the sampled indoor area lights.
            UsdLux.LightAPI(prim).CreateIntensityAttr(0.0)
            legacy_intensity = prim.GetAttribute("intensity")
            if legacy_intensity:
                legacy_intensity.Set(0.0)
            for name in ("inputs:texture:file", "texture:file"):
                texture = prim.GetAttribute(name)
                if texture:
                    texture.Set(Sdf.AssetPath(""))
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)


def apply_indoor_lighting(stage, appearance) -> None:
    """Aim an indoor area light at the task, with a weak matching overhead fill."""
    from pxr import Gf, Sdf, UsdGeom, UsdLux

    settings = load_visual_protocol()["indoor_lighting"]
    # Remove the previous dome and task lights, including inherited HDR paths.
    for name in ("Dome", "Key", "BenchmarkFill"):
        path = f"/World/Lighting/{name}"
        if stage.GetPrimAtPath(path):
            stage.RemovePrim(path)
            # A referenced workstation can retain a weaker authored light.
            if stage.GetPrimAtPath(path):
                stage.OverridePrim(path).SetActive(False)
    for name, position, intensity in (
        ("IndoorKey", appearance.position, appearance.intensity),
        ("IndoorFill", settings["fill_position"], appearance.intensity * settings["fill_ratio"]),
    ):
        light = UsdLux.RectLight.Define(stage, f"/World/Lighting/{name}")
        light.CreateWidthAttr(settings["width"])
        light.CreateHeightAttr(settings["height"])
        light.CreateIntensityAttr(intensity)
        light.CreateExposureAttr(0.0)
        light.CreateNormalizeAttr(False)
        light.CreateColorAttr(Gf.Vec3f(1.0))
        light.CreateEnableColorTemperatureAttr(True)
        light.CreateColorTemperatureAttr(appearance.temperature)
        light.GetPrim().CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(False)
        transform = UsdGeom.Xformable(light)
        transform.ClearXformOpOrder()
        transform.AddTranslateOp().Set(Gf.Vec3d(*position))
        direction = Gf.Vec3d(*settings["target"]) - Gf.Vec3d(*position)
        rotation = Gf.Rotation(Gf.Vec3d(0, 0, -1), direction)
        transform.AddOrientOp().Set(Gf.Quatf(rotation.GetQuat()))
