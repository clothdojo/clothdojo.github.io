"""Optional RGB replay adapter; create the Isaac application before constructing it."""

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from clothdojo.trajectory import TrajectoryReader


class IsaacReplayCamera:
    def __init__(self, app, trajectory, *, assets_root, visual_assets_root,
                 device="cuda:0", seed=42):
        from clothdojo.data.isaac_render import IsaacRandomizedCamera, sample_visual_layout
        from clothdojo.assets.materials import load_material_catalog, require_polyhaven_assets
        from clothdojo.rendering import DATASET_CAMERA_NAMES
        from clothdojo.robots import robot_profile
        from clothdojo.sim.scene import ShirtFlattenSpec

        self.reader = TrajectoryReader(trajectory)
        self.reader.validate_render_data()
        metadata = self.reader.manifest.metadata
        garment = metadata.get("asset_id")
        if not isinstance(garment, str) or not garment:
            raise ValueError("RGB replay requires a recorded asset_id")
        assets_root = Path(assets_root).resolve()
        profile = robot_profile(self.reader.manifest.scene.robot_id, assets_root=assets_root)
        garment_path = assets_root / "garments/converted" / garment / f"{garment}.usdc"
        for key, path in {"garment_usdc": garment_path,
                          f"{profile.robot_id}_entry_usda": assets_root / profile.asset}.items():
            expected = self.reader.manifest.scene.asset_fingerprints.get(key)
            if expected is not None and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"RGB replay asset differs from recording: {key}")
        spec = ShirtFlattenSpec(**metadata.get("simulation_spec", {"garment_id": garment}))
        if spec.garment_id != garment or spec.world_count != 1:
            raise ValueError("RGB replay requires the recorded garment in one world")
        mirrored = metadata.get("augmentation", {}).get("surface_winding_reversed", False)
        if type(mirrored) is not bool:
            raise ValueError("surface_winding_reversed must be boolean")
        self.scene = SimpleNamespace(robot_profile=profile)
        self.arrays = self.reader.state_arrays(["particle_q", "joint_q", "body_q"])
        self.layouts = []
        views = {view.name: view for view in profile.cameras}
        identity = hashlib.sha256((Path(trajectory) / "manifest.json").read_bytes()).hexdigest()
        layout = sample_visual_layout(seed, 0, source_episode_id=identity,
                                      randomize=("cloth", "table", "ground", "lighting"))
        catalog = load_material_catalog()
        selected = {layout.cloth.material, layout.table.material, layout.ground.material}
        require_polyhaven_assets(Path(visual_assets_root) / "polyhaven", replace(
            catalog, materials=tuple(item for item in catalog.materials if item.id in selected),
            lighting=()))
        self.camera = IsaacRandomizedCamera(
            app, self._state(0), asset_path=garment_path, assets_root=assets_root,
            visual_assets_root=Path(visual_assets_root).resolve(), device=device,
            base_seed=seed, include_room=False,
            randomize=("cloth", "table", "ground", "lighting"), layouts=self.layouts,
            robot_profile=profile, cameras=tuple(views[name] for name in DATASET_CAMERA_NAMES),
            simulation_spec=spec, surface_winding_reversed=mirrored,
        )
        try:
            self.camera.begin_episode(0, source_episode_id=identity, layout=layout)
        except BaseException:
            self.camera.close()
            raise

    def _state(self, index):
        profile = self.scene.robot_profile
        # Velocities do not affect FK or rendering and are absent in render-only recordings.
        return SimpleNamespace(**{key: value[index] for key, value in self.arrays.items()},
                               joint_qd=np.zeros(profile.joint_dof_count, np.float32),
                               body_qd=np.zeros((len(profile.body_names), 6), np.float32))

    def __call__(self, index):
        if type(index) is not int or not 0 <= index < self.reader.manifest.state_count:
            raise IndexError("RGB replay sample index out of range")
        return self.camera(self._state(index))

    def close(self):
        self.camera.close()
