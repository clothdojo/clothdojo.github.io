"""Live Newton observations through the same RTX camera used for RGB export."""

from __future__ import annotations

import os
import socket
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from clothdojo.data.isaac_render import (
    VISUAL_COMPONENTS,
    IsaacRandomizedCamera,
    _preload_python_libstdcpp,
    _require_isaac5_runtime,
    _start_isaac_app,
)
from clothdojo.rendering import CAMERA_NAMES, POLICY_RENDER_SIZE
from clothdojo.trajectory.newton import _local_body_q, _world_offsets


def _capture_render_states(runtime) -> tuple[SimpleNamespace, ...]:
    """Read only rendered geometry, using the trajectory's world-local frame."""
    state = runtime.state_0
    body_q = runtime.scene.robot.get_attribute("body_q", state).numpy()[:, 0]
    particle_q = state.particle_q.numpy()
    offsets = _world_offsets(runtime)
    return tuple(
        SimpleNamespace(
            body_q=_local_body_q(body_q[world], offsets[world]),
            particle_q=particle_q[start:end] - offsets[world],
        )
        for world, (start, end) in enumerate(runtime.grasp.particle_ranges())
    )


@contextmanager
def isaac_application(gpu: str):
    """One Kit application per worker; individual asset stages are disposable."""
    if not gpu.isdecimal():
        raise ValueError("Isaac RGB observations require a numeric GPU ordinal")
    _require_isaac5_runtime()
    _preload_python_libstdcpp()
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")
    os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")
    from clothdojo.teleop.isaac_compat import configure_replicator_warp

    configure_replicator_warp()
    cache = Path("data/cache/isaac-rgb-rollout").resolve()
    cache.mkdir(parents=True, exist_ok=True)
    app = _start_isaac_app(int(gpu), cache / f"{socket.gethostname()}-{gpu}")
    try:
        yield app
    finally:
        if not app.is_exiting():
            app.close(wait_for_replicator=False)


class IsaacCameraObservation:
    """Batched world-local observations; rendering never advances Newton physics."""

    def __init__(self, app, runtime, *, asset_path: Path, assets_root: Path,
                 visual_assets: Path, visual_profile: str | None,
                 visual_profiles: tuple[str, ...] | None = None,
                 visual_seed: int) -> None:
        from clothdojo.assets.scene_materials import require_scene_materials
        from clothdojo.assets.visual_benchmark import require_background_assets, visual_pool

        visual_assets = visual_assets.expanduser().resolve()
        profiles = visual_profiles or (visual_profile,)
        for profile_name in profiles:
            require_background_assets(visual_assets, profile_name)
        require_scene_materials(visual_assets, {
            material for profile_name in profiles for material in visual_pool(profile_name, "table")
        })
        self.layouts = []
        profile = runtime.scene.robot_profile
        cameras = {camera.name: camera for camera in profile.cameras}
        self.camera = IsaacRandomizedCamera(
            app, _capture_render_states(runtime)[0], asset_path=asset_path,
            assets_root=assets_root, visual_assets_root=visual_assets,
            device=str(runtime.model.device), base_seed=visual_seed,
            include_room=False, randomize=VISUAL_COMPONENTS, layouts=self.layouts,
            robot_profile=profile, cameras=tuple(cameras[name] for name in CAMERA_NAMES),
            render_resolution=POLICY_RENDER_SIZE, visual_profile=visual_profile,
            use_recorded_body_q=True,
            world_count=int(runtime.model.world_count),
        )

    def begin_episodes(self, case_ids: tuple[str, ...], *,
                       visual_profiles: tuple[str, ...] | None = None) -> list[dict]:
        # Case identity, not worker order or model identity, determines appearance.
        self.layouts.clear()
        self.camera.begin_episodes(case_ids, visual_profiles=visual_profiles)
        return self.layouts.copy()

    def __call__(self, runtime):
        return self.camera.capture_batch(_capture_render_states(runtime))

    def close(self) -> None:
        self.camera.close()
