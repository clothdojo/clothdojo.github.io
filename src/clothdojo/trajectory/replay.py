"""Render saved positions without constructing or advancing a physics solver."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from clothdojo.trajectory import TrajectoryReader


class ReplayCamera:
    """Single-world, four-view replay of full or render-only trajectories."""

    def __init__(self, trajectory: str | Path, *, assets_root: str | Path,
                 device: str = "cuda:0", cloth_visual: str | None = None):
        from clothdojo.rendering import DATASET_CAMERA_NAMES
        from clothdojo.rollout.render import ShirtFlattenCameraObservation
        from clothdojo.sim.scene import ShirtFlattenSpec, build_scene

        self.reader = TrajectoryReader(trajectory)
        self.reader.validate_render_data()
        metadata = self.reader.manifest.metadata
        garment = metadata.get("asset_id")
        if not isinstance(garment, str) or not garment:
            raise ValueError("trajectory must record asset_id for replay")
        recorded_visual = metadata.get("policy", {}).get("cloth_visual")
        visual = cloth_visual or recorded_visual
        if visual not in {"nocs", "white"}:
            raise ValueError("specify cloth_visual='nocs' or 'white' for this trajectory")
        if recorded_visual is not None and visual != recorded_visual:
            raise ValueError("cloth appearance differs from the recorded policy")
        from clothdojo.robots import robot_profile

        profile = robot_profile(self.reader.manifest.scene.robot_id, assets_root=assets_root)
        files = {
            "garment_usdc": Path(assets_root) / "garments/converted" / garment / f"{garment}.usdc",
            f"{profile.robot_id}_entry_usda": Path(assets_root) / profile.asset,
        }
        for key, path in files.items():
            expected = self.reader.manifest.scene.asset_fingerprints.get(key)
            if expected is not None and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"replay asset differs from the recording: {key}")
        spec = ShirtFlattenSpec(**metadata.get("simulation_spec", {"garment_id": garment}))
        if spec.garment_id != garment or spec.world_count != 1:
            raise ValueError("replay requires the recorded garment in one world")
        self.scene = build_scene(
            assets_root, spec=spec, device=device,
            visuals=True, cloth_visual=visual, robot_id=self.reader.manifest.scene.robot_id,
        )
        self.runtime = SimpleNamespace(scene=self.scene, state_0=self.scene.model.state())
        self.camera = ShirtFlattenCameraObservation(self.scene, camera_names=DATASET_CAMERA_NAMES)
        mirrored = metadata.get("augmentation", {}).get("surface_winding_reversed", False)
        if type(mirrored) is not bool:
            raise ValueError("surface_winding_reversed must be boolean")
        self.camera.set_cloth_surface_winding_reversed(mirrored)
        self.arrays = self.reader.state_arrays(["particle_q", "joint_q", "body_q"])

    def __call__(self, sample_index: int) -> dict[str, np.ndarray]:
        import newton

        if type(sample_index) is not int or not 0 <= sample_index < self.reader.manifest.state_count:
            raise IndexError("replay sample index out of range")
        state = self.runtime.state_0
        offset = np.asarray(self.scene.world_offsets[0], dtype=np.float32)
        state.particle_q.assign(np.ascontiguousarray(self.arrays["particle_q"][sample_index] + offset))
        joint_q = self.scene.robot.get_attribute("joint_q", state)
        joint_q.assign(np.ascontiguousarray(self.arrays["joint_q"][sample_index].reshape(joint_q.shape)))
        body_q = self.arrays["body_q"][sample_index].copy()
        body_q[:, :3] += offset
        target = self.scene.robot.get_attribute("body_q", state)
        target.assign(np.ascontiguousarray(body_q.reshape((*target.shape, 7))))
        # As in the original exporter, recorded joint coordinates own articulated poses.
        newton.eval_fk(self.scene.model, state.joint_q, state.joint_qd, state)
        return self.camera(self.runtime)
