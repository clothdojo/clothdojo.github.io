from __future__ import annotations


from pathlib import Path


import numpy as np


def build_render_stage(
    runtime,
    plan,
    *,
    cloth_visual: str = "nocs",
    output_path: str | Path | None = None,
) -> Path:
    """Author the visual-only stage consumed by Isaac RTX/OpenXR."""
    return _build_render_stage(
        runtime.spec,
        runtime.state_0.particle_q.numpy(),
        plan,
        cloth_visual=cloth_visual,
        robot_profile=runtime.scene.robot_profile,
        output_path=output_path,
    )


def build_replay_render_stage(
    initial_points,
    plan,
    *,
    robot_profile,
    cloth_visual: str = "normal",
    spec=None,
) -> Path:
    """Author the same stage directly from a recorded trajectory state."""
    from clothdojo.sim.scene import ShirtFlattenSpec

    return _build_render_stage(
        ShirtFlattenSpec() if spec is None else spec,
        initial_points,
        plan,
        cloth_visual=cloth_visual,
        robot_profile=robot_profile,
    )


def _build_render_stage(
    spec,
    initial_points,
    plan,
    *,
    cloth_visual: str,
    robot_profile,
    output_path: str | Path | None = None,
) -> Path:
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdShade, Vt

    from clothdojo.sim.cloth import load_garment_mesh

    if cloth_visual not in {"normal", "nocs"}:
        raise ValueError("cloth_visual must be 'normal' or 'nocs'")

    output_path = (
        plan.session_root / "teleop_stage.usda"
        if output_path is None
        else Path(output_path).expanduser().resolve()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    stage = Usd.Stage.CreateNew(str(output_path))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    _author_robot(stage, plan.assets_root, robot_profile)

    table_material = _preview_material(stage, "/World/Looks/Table", (0.82, 0.84, 0.87), 0.38)
    table = _define_textured_cube(
        stage, "/World/Table", spec.table_half_extents, physical_uvs=True
    )
    _set_cube(table, spec.table_position, spec.table_half_extents)
    UsdShade.MaterialBindingAPI.Apply(table.GetPrim()).Bind(table_material)

    points = np.asarray(initial_points, dtype=np.float32)
    rest_vertices, triangles, panel_vertices, panel_indices = load_garment_mesh(plan.asset_path)
    indices = np.asarray(triangles, dtype=np.int32).reshape(-1)
    cloth = UsdGeom.Mesh.Define(stage, "/World/Cloth")
    cloth.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    cloth.CreateFaceVertexCountsAttr(Vt.IntArray([3] * (indices.size // 3)))
    cloth.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(indices))
    cloth.CreateDoubleSidedAttr(True)
    cloth.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.catmullClark)
    _write_st_primvar(cloth, panel_vertices[panel_indices])
    if cloth_visual == "nocs":
        if len(rest_vertices) != len(points):
            raise RuntimeError(
                "teleoperation NOCS requires one rest vertex per cloth particle"
            )
        _write_nocs_primvar(cloth, rest_vertices)
        cloth_material = _nocs_material(stage, "/World/Looks/ClothNocs")
    else:
        cloth_material = _preview_material(
            stage, "/World/Looks/Cloth", (0.13, 0.36, 0.78), 0.58
        )
    UsdShade.MaterialBindingAPI.Apply(cloth.GetPrim()).Bind(cloth_material)

    ground_material = _preview_material(stage, "/World/Looks/Ground", (0.025, 0.027, 0.032), 0.82)
    ground_extents = (3.5, 3.5, 0.05)
    ground = _define_textured_cube(stage, "/World/Ground", ground_extents, physical_uvs=True)
    _set_cube(ground, (0.0, 0.0, -0.05), ground_extents)
    UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(ground_material)

    dome = UsdLux.DomeLight.Define(stage, "/World/Lighting/Dome")
    dome.CreateIntensityAttr(650.0)
    dome.CreateColorAttr(Gf.Vec3f(0.82, 0.88, 1.0))
    key = UsdLux.DistantLight.Define(stage, "/World/Lighting/Key")
    key.CreateIntensityAttr(2200.0)
    key.CreateAngleAttr(2.0)
    UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-38.0, 24.0, -28.0))
    stage.GetRootLayer().Save()
    return output_path


def _author_robot(stage, assets_root: Path, robot_profile) -> tuple[str, ...]:
    robot_root = stage.DefinePrim("/World/Robot", "Xform")
    robot_asset = str((assets_root / robot_profile.asset).resolve())
    robot_root.GetReferences().AddReference(robot_asset)
    physics = robot_root.GetVariantSet("Physics")
    if physics and "none" in physics.GetVariantNames():
        physics.SetVariantSelection("none")
    return (robot_root.GetPath().pathString,)


def _set_cube(cube, position, half_extents) -> None:
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xformable(cube)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in position]))
    xf.AddScaleOp().Set(Gf.Vec3d(*[float(v) for v in half_extents]))


def _define_textured_cube(stage, path, half_extents, *, physical_uvs):
    """Create a closed textured box with normalized or meter-scale UVs."""
    from pxr import Gf, Sdf, UsdGeom

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(-1.0, -1.0, -1.0),
            Gf.Vec3f(1.0, -1.0, -1.0),
            Gf.Vec3f(-1.0, 1.0, -1.0),
            Gf.Vec3f(1.0, 1.0, -1.0),
            Gf.Vec3f(-1.0, -1.0, 1.0),
            Gf.Vec3f(1.0, -1.0, 1.0),
            Gf.Vec3f(-1.0, 1.0, 1.0),
            Gf.Vec3f(1.0, 1.0, 1.0),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr(
        [
            4,
            5,
            7,
            6,
            1,
            0,
            2,
            3,
            2,
            6,
            7,
            3,
            0,
            1,
            5,
            4,
            5,
            1,
            3,
            7,
            0,
            4,
            6,
            2,
        ]
    )
    mesh.CreateNormalsAttr(
        [
            Gf.Vec3f(0.0, 0.0, 1.0),
            Gf.Vec3f(0.0, 0.0, -1.0),
            Gf.Vec3f(0.0, 1.0, 0.0),
            Gf.Vec3f(0.0, -1.0, 0.0),
            Gf.Vec3f(1.0, 0.0, 0.0),
            Gf.Vec3f(-1.0, 0.0, 0.0),
        ]
    )
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.uniform)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.faceVarying,
    )
    if physical_uvs:
        size_x, size_y, size_z = (2.0 * float(value) for value in half_extents)
    else:
        size_x = size_y = size_z = 1.0
    face_sizes = (
        (size_x, size_y),
        (size_x, size_y),
        (size_z, size_x),
        (size_x, size_z),
        (size_z, size_y),
        (size_z, size_y),
    )
    face_uvs = []
    for size_u, size_v in face_sizes:
        face_uvs.extend(
            [
                (0.0, 0.0),
                (size_u, 0.0),
                (size_u, size_v),
                (0.0, size_v),
            ]
        )
    st.Set(face_uvs)
    return mesh


def _preview_material(stage, path, color, roughness):
    from pxr import Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _write_nocs_primvar(cloth, rest_points) -> None:
    from pxr import Gf, Sdf, UsdGeom, Vt

    from clothdojo.sim.scene import rest_vertices_to_nocs

    colors = rest_vertices_to_nocs(rest_points)
    values = Vt.Vec3fArray(len(colors))
    for index, color in enumerate(colors):
        values[index] = Gf.Vec3f(*[float(value) for value in color])
    primvar = UsdGeom.PrimvarsAPI(cloth.GetPrim()).CreatePrimvar(
        "nocs",
        Sdf.ValueTypeNames.Color3fArray,
        UsdGeom.Tokens.vertex,
    )
    primvar.Set(values)


def _write_st_primvar(cloth, face_uvs) -> None:
    from pxr import Sdf, UsdGeom, Vt

    primvar = UsdGeom.PrimvarsAPI(cloth.GetPrim()).CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.faceVarying,
    )
    primvar.Set(Vt.Vec2fArray.FromNumpy(face_uvs))


def _nocs_material(stage, path):
    from pxr import Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.0, 0.0, 0.0))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    reader = UsdShade.Shader.Define(stage, f"{path}/NocsReader")
    reader.CreateIdAttr("UsdPrimvarReader_float3")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("nocs")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float3)
    shader.GetInput("diffuseColor").ConnectToSource(
        reader.ConnectableAPI(),
        "result",
    )
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material

