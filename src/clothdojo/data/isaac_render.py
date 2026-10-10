"""Optional Isaac RTX rendering for canonical Newton trajectory replay."""


from __future__ import annotations


import ctypes


import hashlib


import importlib.metadata




import math


import os


import re




import sys


import tempfile




from dataclasses import asdict, dataclass, replace


from pathlib import Path


from types import SimpleNamespace


from typing import Any, Literal


import numpy as np


from clothdojo.assets.materials import (
    MaterialAsset,
    MaterialCatalog,
    load_material_catalog,
    require_polyhaven_assets,
)


from clothdojo.rendering import CameraSpec


from clothdojo.robots import RobotProfile




ROBODOJO_REVISION = "c037c1b3183a030724e694d69a08cb62369ed285"


_ISAAC_MAJOR_MINOR = (5, 0)


_DEVICE = re.compile(r"cuda:(\d+)\Z")


VisualComponent = Literal["cloth", "table", "ground", "lighting", "background"]


VISUAL_COMPONENTS: tuple[VisualComponent, ...] = (
    "cloth",
    "table",
    "ground",
    "lighting",
    "background",
)


_ROOMS = tuple(
    f"Simple_Room_{color}"
    for color in (
        "azure",
        "blue",
        "deep_blue",
        "green",
        "lime",
        "magenta",
        "orange",
        "pink",
        "purple",
        "red",
        "teal",
        "yellow",
    )
)


@dataclass(frozen=True, slots=True)
class MaterialAppearance:
    material: str
    color_channel: str


@dataclass(frozen=True, slots=True)
class LightingAppearance:
    condition: str
    position: tuple[float, float, float]
    intensity: float
    temperature: float


@dataclass(frozen=True, slots=True)
class VisualLayout:
    episode_index: int
    source_episode_id: str | None
    seed: int
    cloth: MaterialAppearance
    table: MaterialAppearance
    ground: MaterialAppearance
    lighting: LightingAppearance
    room: str | None
    background: str | None = None
    visual_profile: str | None = None


def sample_visual_layout(
    base_seed: int,
    episode_index: int,
    *,
    source_episode_id: str | None = None,
    include_room: bool = False,
    randomize: tuple[VisualComponent, ...] = VISUAL_COMPONENTS,
    visual_profile: str | None = None,
) -> VisualLayout:
    """Sample one deterministic, semantically valid appearance."""

    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise TypeError("visual randomization seed must be an integer")
    if base_seed < 0:
        raise ValueError("visual randomization seed must be non-negative")
    if isinstance(episode_index, bool) or not isinstance(episode_index, int):
        raise TypeError("episode index must be an integer")
    if episode_index < 0:
        raise ValueError("episode index must be non-negative")
    if not isinstance(include_room, bool):
        raise TypeError("include_room must be boolean")
    randomized = _validate_visual_components(randomize)
    if visual_profile is not None and include_room:
        raise ValueError("visual profiles select their own backgrounds; omit include_room")
    if source_episode_id is not None:
        if (
            not isinstance(source_episode_id, str)
            or not source_episode_id
            or source_episode_id != source_episode_id.strip()
        ):
            raise ValueError("source_episode_id must be a non-empty string")
        episode_seed = int.from_bytes(
            hashlib.blake2s(source_episode_id.encode(), digest_size=4).digest(),
            "little",
        )
    else:
        episode_seed = episode_index
    sequence = np.random.SeedSequence([base_seed, episode_seed])
    catalog = load_material_catalog()
    if visual_profile is not None:
        from clothdojo.assets.scene_materials import table_material_pool
        from clothdojo.assets.visual_benchmark import visual_pool

        pools = {
            role: tuple(
                item for item in (
                    table_material_pool(catalog) if role == "table" else catalog.pool(role)
                ) if item.id in visual_pool(visual_profile, role)
            )
            for role in ("cloth", "table", "ground")
        }
        backgrounds = visual_pool(visual_profile, "background")
    else:
        pools = {role: catalog.pool(role) for role in ("cloth", "table", "ground")}
        backgrounds = ()
    table_rng = _component_rng(
        base_seed, episode_seed if "table" in randomized else 0, "table"
    )
    ground_rng = _component_rng(
        base_seed, episode_seed if "ground" in randomized else 0, "ground"
    )
    cloth_rng = _component_rng(
        base_seed, episode_seed if "cloth" in randomized else 0, "cloth"
    )
    lighting_rng = _component_rng(
        base_seed, episode_seed if "lighting" in randomized else 0, "lighting"
    )
    room_rng = _component_rng(
        base_seed, episode_seed if "background" in randomized else 0, "room"
    )
    background_rng = _component_rng(
        base_seed, episode_seed if "background" in randomized else 0, "background"
    )
    from clothdojo.assets.visual_benchmark import load_visual_protocol, visual_pool

    protocol = load_visual_protocol()
    light_ids = visual_pool(visual_profile or "train", "lighting")
    condition = light_ids[int(lighting_rng.integers(len(light_ids)))]
    light_range = protocol["indoor_lighting"]["conditions"][condition]
    position_range = protocol["indoor_lighting"]["position_range"]
    return VisualLayout(
        episode_index=episode_index,
        source_episode_id=source_episode_id,
        seed=int(sequence.generate_state(1, dtype=np.uint32)[0]),
        cloth=_sample_material(cloth_rng, pools["cloth"]),
        table=_sample_material(table_rng, pools["table"]),
        ground=_sample_material(ground_rng, pools["ground"]),
        lighting=LightingAppearance(
            condition=condition,
            position=tuple(_rounded_uniform(lighting_rng, *bounds) for bounds in position_range),
            intensity=_rounded_uniform(lighting_rng, *light_range["intensity"]),
            temperature=_rounded_uniform(lighting_rng, *light_range["temperature"]),
        ),
        room=(
            _ROOMS[int(room_rng.integers(len(_ROOMS)))] if include_room else None
        ),
        background=(
            backgrounds[int(background_rng.integers(len(backgrounds)))] if backgrounds else None
        ),
        visual_profile=visual_profile,
    )


def parse_visual_randomization(value: str) -> tuple[VisualComponent, ...]:
    """Parse the single user-facing visual randomization selector."""

    if value == "all":
        return VISUAL_COMPONENTS
    if value == "none":
        return ()
    names = value.split(",")
    if not names or any(not name or name != name.strip() for name in names):
        raise ValueError("expected none, all, or comma-separated visual components")
    if len(names) != len(set(names)):
        raise ValueError("visual randomization components must not be repeated")
    unknown = set(names).difference(VISUAL_COMPONENTS)
    if unknown:
        raise ValueError(f"unknown visual randomization components: {sorted(unknown)}")
    return tuple(name for name in VISUAL_COMPONENTS if name in names)


def _validate_visual_components(
    value: tuple[VisualComponent, ...],
) -> frozenset[VisualComponent]:
    if not isinstance(value, tuple):
        raise TypeError("visual randomization components must be a tuple")
    if len(value) != len(set(value)) or any(item not in VISUAL_COMPONENTS for item in value):
        raise ValueError("invalid visual randomization components")
    return frozenset(value)


def _sample_material(
    rng: np.random.Generator,
    materials: tuple[MaterialAsset, ...],
) -> MaterialAppearance:
    material = materials[int(rng.integers(len(materials)))]
    return MaterialAppearance(
        material=material.id,
        color_channel=material.color_channels[
            int(rng.integers(len(material.color_channels)))
        ],
    )


def _component_rng(base_seed: int, episode_seed: int, component: str) -> np.random.Generator:
    component_seed = int.from_bytes(
        hashlib.blake2s(component.encode(), digest_size=4).digest(), "little"
    )
    return np.random.default_rng(
        np.random.SeedSequence([base_seed, episode_seed, component_seed])
    )


def _rounded_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return round(float(rng.uniform(low, high)), 6)


class IsaacRandomizedCamera:
    """Project recorded Newton state into Isaac Fabric and RTX."""

    def __init__(
        self,
        app,
        initial_state,
        *,
        asset_path: Path,
        assets_root: Path,
        visual_assets_root: Path,
        device: str,
        base_seed: int,
        include_room: bool,
        randomize: tuple[VisualComponent, ...],
        layouts: list[dict[str, Any]],
        robot_profile: RobotProfile,
        cameras: tuple[CameraSpec, ...] | None = None,
        render_resolution: tuple[int, int] = (640, 480),
        pathtrace_spp: int | None = None,
        room_light_saturation: float = 1.0,
        room_light_intensity_scale: float = 1.0,
        use_recorded_body_q: bool = False,
        visual_profile: str | None = None,
        world_count: int = 1,
        simulation_spec=None,
        surface_winding_reversed: bool = False,
    ) -> None:
        import omni.usd
        from pxr import Usd

        particle_q = np.asarray(initial_state.particle_q, dtype=np.float32)
        if particle_q.ndim != 2 or particle_q.shape[1] != 3:
            raise ValueError(
                "Isaac randomized export requires particle_q shape (particles, 3), "
                f"got {particle_q.shape}"
            )
        self._app = app
        if isinstance(world_count, bool) or not isinstance(world_count, int) or world_count < 1:
            raise ValueError("Isaac camera world_count must be a positive integer")
        if world_count > 1 and not use_recorded_body_q:
            raise ValueError("batched Isaac rendering requires explicit body poses")
        self._world_count = world_count
        self._base_seed = base_seed
        self._include_room = include_room
        self._randomize = randomize
        self._visual_profile = visual_profile
        self._visual_assets_root = visual_assets_root
        self._layouts = layouts
        if not 0.0 <= room_light_saturation <= 1.0:
            raise ValueError("room light saturation must be in [0, 1]")
        if not 0.0 <= room_light_intensity_scale <= 1.0:
            raise ValueError("room light intensity scale must be in [0, 1]")
        self._room_light_saturation = room_light_saturation
        self._room_light_intensity_scale = room_light_intensity_scale
        self._cameras = robot_profile.cameras if cameras is None else cameras
        if not self._cameras:
            raise ValueError("Isaac randomized export requires at least one camera")
        if (
            len(render_resolution) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
                   for value in render_resolution)
        ):
            raise ValueError("Isaac render resolution must contain two positive integers")
        if pathtrace_spp is not None and (
            isinstance(pathtrace_spp, bool)
            or not isinstance(pathtrace_spp, int)
            or pathtrace_spp <= 0
        ):
            raise ValueError("Isaac path-tracing samples must be a positive integer")
        self._render_resolution = render_resolution
        self._pathtrace_spp = pathtrace_spp
        self._closed = False
        self._temporary = tempfile.TemporaryDirectory(prefix="clothdojo-isaac-export-")
        plan = _RenderPlan(
            session_root=Path(self._temporary.name),
            assets_root=assets_root,
            asset_path=asset_path,
        )
        from clothdojo.teleop.isaac_stage import build_replay_render_stage

        stage_path = build_replay_render_stage(
            particle_q,
            plan,
            cloth_visual="normal",
            robot_profile=robot_profile,
            spec=simulation_spec,
        )
        source_stage = Usd.Stage.Open(str(stage_path))
        if surface_winding_reversed:
            from pxr import UsdGeom, Vt

            cloth = UsdGeom.Mesh(source_stage.GetPrimAtPath("/World/Cloth"))
            faces = np.asarray(cloth.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).reshape(-1, 3)
            cloth.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(faces[:, ::-1].copy().reshape(-1)))
            st = UsdGeom.PrimvarsAPI(cloth).GetPrimvar("st")
            uv = np.asarray(st.Get(), dtype=np.float32).reshape(-1, 3, 2)
            st.Set(Vt.Vec2fArray.FromNumpy(uv[:, ::-1].copy().reshape(-1, 2)))
        paths = _author_cameras(
            source_stage, robot_profile.body_names,
            _newton_projection_cameras(self._cameras, render_resolution),
        )
        self._source_camera_paths = paths
        source_stage.GetRootLayer().Save()
        self._layout_stages, self._world_roots, stage_path = _render_world_stages(
            source_stage, world_count,
        )
        self._camera_paths = tuple(
            root + path.removeprefix("/World")
            for root in self._world_roots for path in paths
        )
        context = omni.usd.get_context()
        if not context.open_stage(str(stage_path)):
            raise RuntimeError(f"Isaac could not open render stage: {stage_path}")
        for _ in range(30):
            app.update()
        self._stage = context.get_stage()
        if self._stage is None:
            raise RuntimeError("Isaac USD context has no render stage")
        from clothdojo.teleop.isaac_compat import configure_replicator_warp

        configure_replicator_warp()
        import omni.replicator.core as rep

        self._rep = rep
        self._articulation = None
        if use_recorded_body_q:
            body_q = np.asarray(initial_state.body_q, dtype=np.float32)
            _validate_robot_installations(body_q, robot_profile)
        else:
            self._articulation = _NewtonArticulationReplay(
                initial_state,
                asset=assets_root / robot_profile.asset,
                device=device,
                robot_profile=robot_profile,
            )
            _validate_robot_installations(
                self._articulation.body_q.numpy(), robot_profile
            )
        self._model = SimpleNamespace(
            body_label=robot_profile.body_names,
            body_count=len(robot_profile.body_names),
            particle_count=int(particle_q.shape[0]),
        )
        self._usd_context = context
        self._device = device
        self._bridges = []
        self._apply_render_settings()
        self._render_outputs = []

    def _apply_render_settings(self) -> None:
        if self._pathtrace_spp is None:
            self._rep.settings.set_render_rtx_realtime(antialiasing="DLAA")
            from omni.kit.viewport.utility import get_active_viewport

            # Headless Kit still renders its default perspective viewport.
            # Only the dedicated camera render product is exported.
            viewport = get_active_viewport()
            if viewport is not None:
                viewport.updates_enabled = False
        else:
            self._rep.settings.set_render_pathtraced(samples_per_pixel=self._pathtrace_spp)

    def _create_render_outputs(self) -> None:
        product = self._rep.create.render_product_tiled(
            cameras=list(self._camera_paths),
            tile_resolution=self._render_resolution,
            force_new=True,
            name="ClothDojoCameras",
        )
        annotator = self._rep.AnnotatorRegistry.get_annotator(
            "rgb", device="cpu", do_array_copy=True,
        )
        annotator.attach(product)
        self._render_outputs.append((product, annotator))
        for _ in range(4):
            self._app.update()

    def _destroy_render_outputs(self) -> None:
        for product, annotator in self._render_outputs:
            product.hydra_texture.updates_enabled = False
            try:
                annotator.detach([product.path])
            finally:
                product.destroy()
        self._render_outputs.clear()

    def begin_episode(
        self,
        episode_index: int,
        *,
        source_episode_id: str | None,
        layout: VisualLayout | None = None,
    ) -> None:
        if self._world_count != 1:
            raise ValueError("use begin_episodes for a batched Isaac camera")
        if layout is None:
            layout = sample_visual_layout(
                self._base_seed,
                episode_index,
                source_episode_id=source_episode_id,
                include_room=self._include_room,
                randomize=self._randomize,
                visual_profile=self._visual_profile,
            )
        elif layout.episode_index != episode_index:
            raise ValueError("explicit visual layout has the wrong episode index")
        self._set_layouts((layout,))

    def begin_episodes(self, case_ids: tuple[str, ...], *,
                       visual_profiles: tuple[str, ...] | None = None) -> None:
        if len(case_ids) != self._world_count:
            raise ValueError("one case identity is required per render world")
        if visual_profiles is not None and len(visual_profiles) != self._world_count:
            raise ValueError("one visual profile is required per render world")
        profiles = visual_profiles or (self._visual_profile,) * self._world_count
        self._set_layouts(tuple(
            sample_visual_layout(
                self._base_seed, 0, source_episode_id=case_id,
                include_room=self._include_room, randomize=self._randomize,
                visual_profile=profile,
            ) for case_id, profile in zip(case_ids, profiles, strict=True)
        ))

    def _set_layouts(self, layouts: tuple[VisualLayout, ...]) -> None:
        from omni.usd import attach_all_hydra_engines, release_all_hydra_engines
        from pxr import Usd

        stage_path = None
        if self._render_outputs:
            # Background references may contain instanced/skinned geometry.
            # Retire the old imaging adapters before removing their USD prims;
            # editing a live RTX stage can leave dangling instancer references.
            self._destroy_render_outputs()
            self._bridges.clear()
            stage_path = self._stage.GetRootLayer().identifier
            release_all_hydra_engines(self._usd_context)
            self._usd_context.close_stage()
            # Context-owned Stage wrappers expire on close. Author the retained
            # source layers through a new standalone stage until Kit reopens it.
            self._stage = Usd.Stage.Open(stage_path)
        for stage, layout in zip(self._layout_stages, layouts, strict=True):
            _apply_layout(
                stage, layout, self._visual_assets_root,
                room_light_saturation=self._room_light_saturation,
                room_light_intensity_scale=self._room_light_intensity_scale,
            )
            self._layouts.append(asdict(layout))
        if self._world_count > 1:
            _separate_render_worlds(
                self._stage, self._layout_stages, self._world_roots, self._source_camera_paths,
            )
        if stage_path is not None:
            # Kit reloads the root and referenced layers on open. Publish the
            # temporary appearance edits before returning stage ownership to it.
            for stage in self._layout_stages:
                stage.GetRootLayer().Save()
            self._stage.GetRootLayer().Save()
            if not self._usd_context.open_stage(stage_path):
                raise RuntimeError(f"Isaac could not reopen render stage: {stage_path}")
            # Hydra engines belong to the context's active stage.  Attaching
            # before open_stage creates an engine without a stage and leaves
            # stale imaging state behind on every appearance switch.
            attach_all_hydra_engines(self._usd_context)
            self._stage = self._usd_context.get_stage()
        for _ in range(3):
            self._app.update()
        # Reopening a stage restores Kit defaults, including DLSS. Reapply the
        # requested render mode after those updates and before creating cameras.
        self._apply_render_settings()
        self._create_render_outputs()
        # Layout edits can rebuild Fabric and discard custom attributes, so bind
        # only after the episode's USD appearance has settled.
        from clothdojo.teleop.isaac_fabric import NewtonFabricBridge

        self._bridges = [
            NewtonFabricBridge(
                self._model, self._usd_context, device=self._device, root_path=root,
            ) for root in self._world_roots
        ]

    def __call__(self, state) -> dict[str, np.ndarray]:
        return self.capture_batch((state,))

    def capture_batch(self, states) -> dict[str, np.ndarray]:
        if len(states) != self._world_count:
            raise ValueError("one state is required per render world")
        if not self._bridges:
            raise RuntimeError("Isaac camera episode has not been initialized")
        bodies = [state.body_q for state in states]
        if self._articulation is not None:
            self._articulation.update(states[0])
            bodies = [self._articulation.body_q.numpy()]

        def publish():
            for bridge, state, body_q in zip(self._bridges, states, bodies, strict=True):
                bridge.update_arrays(state.particle_q, body_q)

        publish()
        if self._pathtrace_spp is not None:
            for _ in range(4):
                self._app.update()
        # Schedule the first capture, then republish before the final capture.
        # Its blocking read below also waits for the earlier GPU work to finish.
        # Path-traced paper renders retain their separately validated updates.
        self._step_render(wait_for_render=self._pathtrace_spp is not None)
        publish()
        if self._pathtrace_spp is not None:
            for _ in range(4):
                self._app.update()
        images = self._read_tiled_rgb()
        return {
            f"observation.images.{camera.name}": np.stack(images[index::len(self._cameras)])
            for index, camera in enumerate(self._cameras)
        }

    def _step_render(self, *, wait_for_render: bool = True) -> None:
        self._rep.orchestrator.step(
            rt_subframes=3,
            pause_timeline=True,
            delta_time=0.0,
            wait_for_render=wait_for_render,
        )

    def _read_tiled_rgb(self) -> list[np.ndarray]:
        self._step_render()
        images = []
        for _, annotator in self._render_outputs:
            data = annotator.get_data(device="cpu", do_array_copy=True)
            if isinstance(data, dict):
                data = data.get("data")
            images.extend(_split_tiled_rgb(
                data, camera_count=self._world_count * len(self._cameras),
                tile_resolution=self._render_resolution,
            ))
        return images

    def close(self) -> None:
        from omni.usd import release_all_hydra_engines

        if self._closed:
            return
        self._closed = True
        try:
            # Replicator keeps its SDG graph and pending render bookkeeping
            # across render-product destruction.  If the stage is closed while
            # that graph is still running, the next product can wait forever
            # for a frame associated with the retired Hydra scene index.
            self._rep.orchestrator.stop()
            self._rep.orchestrator.wait_until_complete()
            self._destroy_render_outputs()
            self._bridges.clear()
            release_all_hydra_engines(self._usd_context)
            self._usd_context.close_stage()
            for _ in range(3):
                self._app.update()
        finally:
            self._temporary.cleanup()


def _render_world_stages(source_stage, world_count):
    """Reuse one authored workstation with independent per-world appearance layers."""
    from pxr import Usd, UsdGeom

    source = Path(source_stage.GetRootLayer().realPath)
    if world_count == 1:
        return (source_stage,), ("/World",), source
    master_path = source.with_name("batch.usda")
    master = Usd.Stage.CreateNew(str(master_path))
    root = UsdGeom.Xform.Define(master, "/World")
    master.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageUpAxis(master, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(master, 1.0)
    stages, roots = [], []
    for index in range(world_count):
        path = source.with_name(f"world_{index}.usda")
        stage = Usd.Stage.CreateNew(str(path))
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        world.GetPrim().GetReferences().AddReference(str(source), "/World")
        stage.GetRootLayer().Save()
        root_path = f"/World/env_{index}"
        prim = UsdGeom.Xform.Define(master, root_path)
        prim.GetPrim().GetReferences().AddReference(str(path), "/World")
        prim.AddTranslateOp().Set((0.0, 0.0, 0.0))
        stages.append(stage)
        roots.append(root_path)
    master.GetRootLayer().Save()
    return tuple(stages), tuple(roots), master_path


def _separate_render_worlds(master, stages, roots, camera_paths):
    """Separate visible bounds and restrict each light and shadow to its own world."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

    bounds_by_world = []
    for stage in stages:
        far = max(
            float(UsdGeom.Camera(stage.GetPrimAtPath(path)).GetClippingRangeAttr().Get()[1])
            for path in camera_paths
        )
        for prim in stage.Traverse():
            if prim.HasAPI(UsdLux.LightAPI):
                light = UsdLux.LightAPI(prim)
                # Background lights are disabled. Linking them needlessly
                # invalidates large USD collections whenever scenery changes.
                if light.GetIntensityAttr().Get() == 0:
                    continue
                for link in (light.GetLightLinkCollectionAPI(), light.GetShadowLinkCollectionAPI()):
                    link.CreateIncludeRootAttr(False)
                    link.CreateIncludesRel().SetTargets([Sdf.Path("/World")])
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
            useExtentsHint=True,
        )
        bounds = cache.ComputeWorldBound(stage.GetPrimAtPath("/World")).ComputeAlignedRange()
        lower, upper = np.asarray(bounds.GetMin()), np.asarray(bounds.GetMax())
        if not np.isfinite([*lower, *upper, far]).all() or np.any(upper < lower):
            raise RuntimeError("render world has invalid visible bounds")
        bounds_by_world.append((lower, upper, far))
    # Large imported floors can span a kilometre. Use the shortest axis and
    # center the placements to avoid RTX precision artifacts on the cloth.
    axis = int(np.argmin(np.sum([hi - lo for lo, hi, _ in bounds_by_world], axis=0)))
    offsets = []
    previous_right, previous_far = None, 0.0
    for lower, upper, far in bounds_by_world:
        left, right = lower[axis], upper[axis]
        offset = 0.0 if previous_right is None else previous_right - left + max(far, previous_far)
        offsets.append(offset)
        previous_right, previous_far = right + offset, far
    center = (offsets[0] + offsets[-1]) / 2
    for root, offset in zip(roots, offsets, strict=True):
        translation = [0.0, 0.0, 0.0]
        translation[axis] = float(offset - center)
        master.GetPrimAtPath(root).GetAttribute("xformOp:translate").Set(Gf.Vec3d(*translation))


class _NewtonArticulationReplay:
    """Reconstruct robot bodies from importer-stable generalized coordinates."""

    def __init__(
        self,
        initial_state,
        *,
        asset: Path,
        device: str,
        robot_profile: RobotProfile,
    ) -> None:
        import newton
        import warp as wp

        from clothdojo.sim.scene import _add_robot

        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        _add_robot(
            newton,
            wp,
            builder,
            asset,
            robot_profile,
            load_visual_shapes=False,
        )
        self._newton = newton
        self._model = builder.finalize(device=device, requires_grad=False)
        self._state = self._model.state()
        self.update(initial_state)

    @property
    def body_q(self):
        return self._state.body_q

    def update(self, recorded_state) -> None:
        joint_q = np.asarray(recorded_state.joint_q, dtype=np.float32)
        joint_qd = np.asarray(recorded_state.joint_qd, dtype=np.float32)
        body_q = np.asarray(recorded_state.body_q, dtype=np.float32)
        body_qd = np.asarray(recorded_state.body_qd, dtype=np.float32)
        expected = {
            "joint_q": tuple(self._state.joint_q.shape),
            "joint_qd": tuple(self._state.joint_qd.shape),
            "body_q": (*tuple(self._state.body_q.shape), 7),
            "body_qd": (*tuple(self._state.body_qd.shape), 6),
        }
        actual = {
            "joint_q": joint_q.shape,
            "joint_qd": joint_qd.shape,
            "body_q": body_q.shape,
            "body_qd": body_qd.shape,
        }
        mismatched = [
            f"{name} expected {expected[name]}, got {actual[name]}"
            for name in expected
            if expected[name] != actual[name]
        ]
        if mismatched:
            raise ValueError("Isaac articulation replay shape mismatch: " + "; ".join(mismatched))
        self._state.joint_q.assign(np.ascontiguousarray(joint_q))
        self._state.joint_qd.assign(np.ascontiguousarray(joint_qd))
        # Preserve recorded poses for any non-articulated bodies before FK. The
        # articulated body array is reconstructed because legacy trajectories
        # predate a persisted, importer-stable body-name order.
        self._state.body_q.assign(np.ascontiguousarray(body_q))
        self._state.body_qd.assign(np.ascontiguousarray(body_qd))
        self._newton.eval_fk(
            self._model,
            self._state.joint_q,
            self._state.joint_qd,
            self._state,
        )


@dataclass(frozen=True, slots=True)
class _RenderPlan:
    session_root: Path
    assets_root: Path
    asset_path: Path


def _validate_robot_installations(
    body_q: np.ndarray, profile: RobotProfile
) -> None:
    """Fail before rendering if Newton roots disagree with the robot profile."""

    roots = np.flatnonzero(np.asarray(profile.body_parents) == -1)
    expected_positions = np.asarray(
        [frame.position for frame in profile.root_body_frames], dtype=np.float32
    )
    expected_rotations = np.asarray(
        [frame.rotation_xyzw for frame in profile.root_body_frames], dtype=np.float32
    )
    actual = np.asarray(body_q, dtype=np.float32)[roots]
    position_error = np.linalg.norm(actual[:, :3] - expected_positions, axis=1)
    actual_rotation = actual[:, 3:7]
    actual_norms = np.linalg.norm(actual_rotation, axis=1, keepdims=True)
    expected_norms = np.linalg.norm(expected_rotations, axis=1, keepdims=True)
    if (
        not np.isfinite(actual).all()
        or np.any(actual_norms <= 1.0e-8)
        or np.any(expected_norms <= 1.0e-8)
    ):
        raise RuntimeError(
            f"Newton {profile.robot_id} root body poses disagree with its "
            "installation frames"
        )
    actual_rotation /= actual_norms
    expected_rotations /= expected_norms
    rotation_alignment = np.abs(np.sum(actual_rotation * expected_rotations, axis=1))
    if (
        np.any(position_error > 1.0e-5)
        or np.any(rotation_alignment < 1.0 - 1.0e-6)
    ):
        raise RuntimeError(
            f"Newton {profile.robot_id} root body poses disagree with its "
            "installation frames"
        )


def _newton_projection_cameras(cameras, resolution: tuple[int, int]):
    """Match the effective intrinsics of the existing Newton ray sampler."""
    width, height = resolution
    if all(math.isclose(view.vertical_aperture, view.horizontal_aperture * height / width,
                        rel_tol=1.0e-6) for view in cameras):
        # Newton's established 4:3 path uses the horizontal aperture as its
        # vertical FOV input. Match its wider view without changing old policies.
        return tuple(replace(view, horizontal_aperture=view.horizontal_aperture * width / height,
                             vertical_aperture=view.horizontal_aperture) for view in cameras)
    return cameras


def _author_cameras(stage, body_labels, cameras) -> tuple[str, ...]:
    from pxr import Gf, UsdGeom

    from clothdojo.teleop.isaac_fabric import _find_render_body_paths

    requested_names = tuple(view.name for view in cameras)
    embedded = {
        str(prim.GetAttribute("clothdojo:cameraName").Get() or prim.GetName()): str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.IsA(UsdGeom.Camera)
        and str(prim.GetAttribute("clothdojo:cameraName").Get() or prim.GetName())
        in requested_names
    }
    if embedded:
        if set(embedded) != set(requested_names):
            raise RuntimeError("robot USD contains only part of the requested camera set")
        # CameraSpec is the rendering contract even when the workstation already
        # embeds the camera prim. This permits paper-only lens overrides while
        # preserving the authored pose and clipping range.
        for view in cameras:
            camera = UsdGeom.Camera(stage.GetPrimAtPath(embedded[view.name]))
            camera.GetFocalLengthAttr().Set(view.focal_length)
            camera.GetHorizontalApertureAttr().Set(view.horizontal_aperture)
            camera.GetVerticalApertureAttr().Set(view.vertical_aperture)
        return tuple(embedded[name] for name in requested_names)

    body_paths = _find_render_body_paths(stage, body_labels)
    label_to_path = {
        str(body_labels[index]).strip("/"): path
        for index, path in body_paths.items()
    }
    paths = []
    for view in cameras:
        parent = "/World/Cameras"
        if view.parent_body is not None:
            candidates = [
                path
                for label, path in label_to_path.items()
                if _matches_semantic_body(label, view.parent_body)
            ]
            if len(candidates) != 1:
                raise RuntimeError(
                    f"Isaac camera mount {view.parent_body!r} matched "
                    f"{len(candidates)} render bodies"
                )
            parent = f"{candidates[0]}/ClothDojoCameras"
        UsdGeom.Xform.Define(stage, parent)
        path = f"{parent}/{view.name}"
        camera = UsdGeom.Camera.Define(stage, path)
        matrix = np.asarray(view.parent_from_camera, dtype=np.float64)
        transform = Gf.Matrix4d(*[float(value) for value in matrix.T.reshape(-1)])
        UsdGeom.Xformable(camera).AddTransformOp().Set(transform)
        camera.CreateFocalLengthAttr(view.focal_length)
        camera.CreateHorizontalApertureAttr(view.horizontal_aperture)
        camera.CreateVerticalApertureAttr(view.vertical_aperture)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
        paths.append(path)
    return tuple(paths)


def _matches_semantic_body(label: str, semantic_name: str) -> bool:
    """Match a compact name such as ``left/camera`` to a full body path."""

    label_parts = label.strip("/").split("/")
    semantic_parts = semantic_name.strip("/").split("/")
    if not label_parts or not semantic_parts or label_parts[-1] != semantic_parts[-1]:
        return False
    positions = iter(label_parts[:-1])
    return all(part in positions for part in semantic_parts[:-1])


def _split_tiled_rgb(
    data,
    *,
    camera_count: int,
    tile_resolution: tuple[int, int] = (640, 480),
) -> list[np.ndarray]:
    columns = int(np.ceil(np.sqrt(camera_count)))
    rows = int(np.ceil(camera_count / columns))
    tiled = np.asarray(data)
    width, height = tile_resolution
    expected = (rows * height, columns * width)
    if tiled.ndim != 3 or tiled.shape[:2] != expected or tiled.shape[2] < 3:
        raise RuntimeError(
            f"Isaac tiled RGB shape {tiled.shape} does not match "
            f"{(*expected, '3 or 4 channels')}"
        )
    if tiled.shape[2] == 4:
        import cv2

        # Pack RGB once with SIMD instead of four strided RGBA-to-RGB copies.
        tiled = cv2.cvtColor(tiled, cv2.COLOR_RGBA2RGB)
    return [
        np.ascontiguousarray(
            tiled[
                row * height : (row + 1) * height,
                column * width : (column + 1) * width,
                :3,
            ]
        )
        for row, column in (
            divmod(index, columns) for index in range(camera_count)
        )
    ]


def _apply_layout(
    stage,
    layout: VisualLayout,
    root: Path,
    *,
    room_light_saturation: float = 1.0,
    room_light_intensity_scale: float = 1.0,
) -> None:
    catalog = load_material_catalog()
    previous_background = stage.GetPrimAtPath("/World/BenchmarkBackground")
    if previous_background:
        stage.RemovePrim(previous_background.GetPath())
    material_root = root / "polyhaven"
    from clothdojo.assets.scene_materials import bind_scene_material

    if not bind_scene_material(
        stage, root, layout.table.material,
        look="/World/Looks/RandomizedTable", prim="/World/Table",
    ):
        _bind_polyhaven_material(
            stage,
            look="/World/Looks/RandomizedTable",
            prim="/World/Table",
            material_root=material_root,
            catalog=catalog,
            appearance=layout.table,
        )
    _bind_polyhaven_material(
        stage,
        look="/World/Looks/RandomizedGround",
        prim="/World/Ground",
        material_root=material_root,
        catalog=catalog,
        appearance=layout.ground,
    )
    _bind_polyhaven_material(
        stage,
        look="/World/Looks/RandomizedCloth",
        prim="/World/Cloth",
        material_root=material_root,
        catalog=catalog,
        appearance=layout.cloth,
    )
    from clothdojo.data.two_sided_cloth import bind_two_sided_cloth

    bind_two_sided_cloth(stage)
    room = stage.GetPrimAtPath("/World/RoboDojoRoom")
    if layout.room is None:
        if room:
            stage.RemovePrim(room.GetPath())
    else:
        _reference_room(
            stage,
            root / "robodojo/Assets/Room",
            layout.room,
            light_saturation=room_light_saturation,
            light_intensity_scale=room_light_intensity_scale,
        )
    from clothdojo.assets.visual_benchmark import apply_background, apply_indoor_lighting

    if layout.background is not None:
        apply_background(stage, root, layout.background)
    apply_indoor_lighting(stage, layout.lighting)


def _reference_room(
    stage,
    room_root: Path,
    room_name: str,
    *,
    light_saturation: float,
    light_intensity_scale: float,
) -> None:
    from pxr import Gf, UsdGeom

    source = room_root / room_name / "base0.usd"
    room = UsdGeom.Xform.Define(stage, "/World/RoboDojoRoom")
    references = room.GetPrim().GetReferences()
    references.ClearReferences()
    references.AddReference(str(source.resolve()))
    xform = UsdGeom.Xformable(room)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(0.782, 0.0, 0.0))
    xform.AddRotateZOp().Set(-90.0)
    xform.AddScaleOp().Set(Gf.Vec3f(0.5, 0.5, 0.5))
    light = stage.GetPrimAtPath("/World/RoboDojoRoom/simple_room/RectLight")
    if light:
        color = light.GetAttribute("inputs:color")
        authored = np.asarray(color.Get(), dtype=np.float32)
        mild = 1.0 + light_saturation * (authored - 1.0)
        color.Set(Gf.Vec3f(*(float(value) for value in mild)))
        for name in ("inputs:intensity", "intensity"):
            intensity = light.GetAttribute(name)
            if intensity and intensity.HasAuthoredValue():
                intensity.Set(float(intensity.Get()) * light_intensity_scale)


def _bind_polyhaven_material(
    stage,
    *,
    look: str,
    prim: str,
    material_root: Path,
    catalog: MaterialCatalog,
    appearance: MaterialAppearance,
) -> None:
    existing = stage.GetPrimAtPath(look)
    if existing:
        stage.RemovePrim(existing.GetPath())
    from pxr import Gf, Sdf, UsdShade

    source = catalog.material(appearance.material)
    directory = source.directory(material_root)
    material = UsdShade.Material.Define(stage, look)
    shader = UsdShade.Shader.Define(stage, f"{look}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.5)

    reader = UsdShade.Shader.Define(stage, f"{look}/PrimvarReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    transform = UsdShade.Shader.Define(stage, f"{look}/UvTransform")
    transform.CreateIdAttr("UsdTransform2d")
    transform.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.ConnectableAPI(), "result"
    )
    transform.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(
        Gf.Vec2f(*source.uv_scale())
    )
    transform.CreateInput("rotation", Sdf.ValueTypeNames.Float).Set(0.0)
    transform.CreateInput("translation", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(0.0, 0.0))
    transform.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    base_color = _uv_texture(
        stage,
        f"{look}/BaseColor",
        directory / source.base_color_name(appearance.color_channel),
        transform,
        source_color_space="sRGB",
    )
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        base_color.ConnectableAPI(), "rgb"
    )
    roughness = _uv_texture(
        stage,
        f"{look}/Roughness",
        directory / "roughness.jpg",
        transform,
        source_color_space="raw",
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(
        roughness.ConnectableAPI(), "r"
    )
    normal = _uv_texture(
        stage,
        f"{look}/Normal",
        directory / "normal_gl.jpg",
        transform,
        source_color_space="raw",
    )
    normal.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(
        Gf.Vec4f(2.0, 2.0, 2.0, 1.0)
    )
    normal.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(
        Gf.Vec4f(-1.0, -1.0, -1.0, 0.0)
    )
    normal.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
        normal.ConnectableAPI(), "rgb"
    )
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    _bind(stage, prim, material)


def _uv_texture(stage, path: str, source: Path, transform, *, source_color_space: str):
    from pxr import Sdf, UsdShade

    shader = UsdShade.Shader.Define(stage, path)
    shader.CreateIdAttr("UsdUVTexture")
    shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(str(source.resolve()))
    )
    shader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        transform.ConnectableAPI(), "result"
    )
    shader.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(source_color_space)
    shader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    shader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    shader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    shader.CreateOutput("r", Sdf.ValueTypeNames.Float)
    return shader


def _bind(stage, prim_path: str, material) -> None:
    from pxr import UsdShade

    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise RuntimeError(f"render prim is missing: {prim_path}")
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def _require_visual_assets(
    root: Path,
    catalog: MaterialCatalog,
    *,
    include_room: bool,
) -> None:
    try:
        require_polyhaven_assets(root / "polyhaven", catalog)
    except FileNotFoundError as exc:
        room_flag = " --include-room" if include_room else ""
        raise FileNotFoundError(
            f"{exc}; run `clothdojo data prepare-isaac-assets{room_flag}` first"
        ) from None
    if include_room:
        _require_robodojo_rooms(root / "robodojo")


def _require_robodojo_rooms(root: Path) -> None:
    revision_path = root / "clothdojo_revision.txt"
    try:
        revision = revision_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"RoboDojo rooms are not prepared at {root}; run "
            "`clothdojo data prepare-isaac-assets --include-room` first"
        ) from None
    if revision != ROBODOJO_REVISION:
        raise RuntimeError(
            f"RoboDojo assets use revision {revision!r}, expected {ROBODOJO_REVISION!r}"
        )
    missing = [
        room
        for room in _ROOMS
        if not (root / "Assets/Room" / room / "base0.usd").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"RoboDojo room assets are incomplete: {sorted(missing)}")


def _require_isaac5_runtime() -> None:
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(
            f"Isaac RTX export requires Python 3.11, got {sys.version_info.major}."
            f"{sys.version_info.minor}"
        )
    try:
        version = importlib.metadata.version("isaacsim")
    except importlib.metadata.PackageNotFoundError:
        raise RuntimeError(
            "Isaac RTX export requires the optional Isaac Sim 5 runtime"
        ) from None
    numbers = tuple(int(value) for value in re.findall(r"\d+", version)[:2])
    if numbers != _ISAAC_MAJOR_MINOR:
        raise RuntimeError(f"Isaac RTX export requires Isaac Sim 5.0.x, got {version}")


def _start_isaac_app(gpu_id: int, portable: Path):
    import isaacsim
    from isaacsim import SimulationApp

    extension_cache = Path(isaacsim.__file__).resolve().parent / "extscache"
    if not extension_cache.is_dir():
        raise RuntimeError(
            "Isaac RTX export requires the local Isaac Sim extension cache; "
            "install the Isaac environment as shown in README.md"
        )
    cpu_threads = int(os.environ.get("CLOTHDOJO_ISAAC_CPU_THREADS", "4"))
    if cpu_threads < 2:
        raise ValueError("Isaac RTX rendering requires at least 2 CPU worker threads")
    original_argv = sys.argv
    # SimulationApp applies limit_cpu_threads to both OpenUSD and the renderer.
    # Keep USD imports serial, then override the independent RTX worker pools
    # through final Kit arguments so shader compilation can make progress.
    sys.argv = [
        sys.argv[0], "--portable-root", str(portable),
        f"--/plugins/carb.tasking.plugin/threadCount={cpu_threads}",
        f"--/plugins/omni.tbb.globalcontrol/maxThreadCount={cpu_threads}",
    ]
    try:
        return SimulationApp(
            {
                "headless": True,
                # Replay opens its own stage. An empty startup stage can leave
                # Isaac 5 waiting indefinitely for a headless viewport handle.
                "create_new_stage": False,
                "renderer": "RaytracedLighting",
                "width": 640,
                "height": 480,
                "multi_gpu": False,
                # Vulkan keeps host GPU ordinals when CUDA is isolated.
                "active_gpu": gpu_id,
                "physics_gpu": 0,
                "enable_crashreporter": False,
                # SimulationApp sets PXR_WORK_THREAD_LIMIT before loading USD.
                "limit_cpu_threads": 1,
                "extra_args": [
                    "--ext-folder",
                    str(extension_cache),
                    # Batched RGB rollout uses one tile per world and camera.
                    # Eight worlds with the three policy cameras require 24,
                    # above Isaac's default limit of 16.
                    "--/rtx/viewTile/limit=32",
                    "--/rtx/hydra/supportMultiTickRate=false",
                    "--/rtx/post/motionBlur/enabled=false",
                ],
            },
            experience=str(Path(__file__).with_name("isaac_export.kit")),
        )
    finally:
        sys.argv = original_argv


def _preload_python_libstdcpp() -> None:
    """Keep Kit compatible with Python distributions that bundle a newer C++ ABI."""

    library = Path(sys.base_prefix) / "lib" / "libstdc++.so.6"
    if sys.platform.startswith("linux") and library.is_file():
        ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)


def _physical_gpu(device: str) -> int:
    match = _DEVICE.fullmatch(device)
    if match is None:
        raise ValueError("Isaac RTX export device must be a physical CUDA index such as cuda:0")
    return int(match.group(1))
