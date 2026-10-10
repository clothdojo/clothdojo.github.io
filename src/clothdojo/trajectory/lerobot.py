"""Export recorded observations and applied controls as ordinary LeRobot v2.1."""

import json
import tempfile
from pathlib import Path
from types import MethodType

import numpy as np

from clothdojo.rendering import DATASET_CAMERA_NAMES
from clothdojo.rollout.newton import pack_policy_state
from clothdojo.trajectory import RUNTIME_ACTION_SEMANTICS
from clothdojo.trajectory.replay import ReplayCamera


def _encode_videos(writer, episode_index):
    import av
    from lerobot.common.datasets.video_utils import encode_video_frames

    paths = {}
    expected = len(writer.episode_buffer["frame_index"])
    for key in writer.meta.video_keys:
        images = writer._get_image_file_path(episode_index=episode_index,
                                             image_key=key, frame_index=0).parent
        output = writer.root / writer.meta.get_video_file_path(episode_index, key)
        encode_video_frames(images, output, writer.fps, vcodec="h264", g=2,
                            crf=23, log_level=None, overwrite=True)
        with av.open(str(output)) as video:
            if sum(1 for _ in video.decode(video=0)) != expected:
                raise RuntimeError(f"encoded frame count differs: {key}")
        paths[key] = str(output)
    return paths


def export_lerobot(trajectory: str | Path, output: str | Path, *,
                   assets_root: str | Path, device: str = "cuda:0",
                   cloth_visual: str | None = None, replay_camera=None) -> Path:
    """Export one trajectory; terminal state is excluded because it has no action."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    replay = replay_camera if replay_camera is not None else ReplayCamera(
        trajectory, assets_root=assets_root, device=device, cloth_visual=cloth_visual)
    reader = replay.reader
    if reader.manifest.metadata.get("action_semantics") != RUNTIME_ACTION_SEMANTICS:
        raise ValueError("export requires recorded runtime-applied control targets")
    count = reader.manifest.action_count
    if count < 1:
        raise ValueError("cannot export a trajectory without actions")
    fps = reader.manifest.control_fps
    if fps != int(fps):
        raise ValueError("LeRobot export requires an integer frame rate")
    profile = replay.scene.robot_profile
    names = (*profile.arm_joint_names[:6], "left_gripper_open_state",
             *profile.arm_joint_names[6:], "right_gripper_open_state")
    action_names = (*[f"{name}_target" for name in profile.arm_joint_names[:6]],
                    "left_gripper_open_cmd",
                    *[f"{name}_target" for name in profile.arm_joint_names[6:]],
                    "right_gripper_open_cmd")
    features = {
        "observation.state": {"dtype": "float32", "shape": (14,), "names": names},
        "action": {"dtype": "float32", "shape": (14,), "names": action_names},
        **{f"observation.images.{name}": {"dtype": "video", "shape": (480, 640, 3),
                                          "names": ("height", "width", "channels")}
           for name in DATASET_CAMERA_NAMES},
    }
    metadata = reader.manifest.metadata
    task = reader.manifest.scene.task_name
    prompt = metadata.get("prompt") or {"shirt_flatten": "Flatten the shirt.",
                                        "shirt_fold": "Fold the shirt."}[task]
    states = reader.state_arrays(["joint_q", "gripper_open_state"])
    actions = reader.action_arrays()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".lerobot-", dir=output.parent) as temporary:
        staging = Path(temporary) / "dataset"
        repo_id = f"local/{output.name}"
        writer = LeRobotDataset.create(repo_id=repo_id, root=staging, fps=int(fps),
                                       robot_type="piper", features=features, use_videos=True,
                                       image_writer_processes=0, image_writer_threads=4)
        writer.encode_episode_videos = MethodType(_encode_videos, writer)
        try:
            for index in range(count):
                arm_q = states["joint_q"][index][list(profile.arm_coord_indices)][None]
                state = pack_policy_state(arm_q, states["gripper_open_state"][index][None])[0]
                action = pack_policy_state(actions["joint_target"][index][None],
                                           actions["gripper_open_command"][index][None])[0]
                views = replay(index)
                writer.add_frame({"observation.state": state, "action": action, "task": prompt,
                                  **{name: image[0] for name, image in views.items()}})
            writer.save_episode()
        finally:
            writer.stop_image_writer()
        dataset = LeRobotDataset(repo_id, root=staging, download_videos=False, video_backend="pyav")
        if len(dataset) != count or dataset.meta.info["codebase_version"] != "v2.1":
            raise RuntimeError("LeRobot could not reopen the complete exported episode")
        # Decode actual first/last samples, including all four videos.
        for index in sorted({0, count - 1}):
            sample = dataset[index]
            if not np.isfinite(sample["observation.state"].numpy()).all():
                raise RuntimeError("exported state contains non-finite values")
        sidecar = {
            "schema": "clothdojo_lerobot_export", "format_version": 1,
            "export_status": "complete", "training_eligible": metadata.get("training_eligible") is True,
            "source": {"kind": "recorded_trajectory", "output_format": "LeRobot v2.1"},
            "counts": {"output_episodes": 1, "output_frames": count},
            "scenario": {"task_name": task, "prompt": prompt, "asset_id": metadata["asset_id"]},
            "observation": {"cloth_visual": cloth_visual or metadata.get("policy", {}).get("cloth_visual")},
            "action_semantics": RUNTIME_ACTION_SEMANTICS,
            "episodes": [{"episode_index": 0, "length": count, "full_trajectory": True}],
        }
        if replay_camera is not None:
            sidecar["observation"].update(renderer="isaac-rtx", cloth_visual="rgb",
                                          layouts=replay.layouts)
        (staging / "meta/clothdojo.json").write_text(json.dumps(sidecar, indent=2) + "\n")
        staging.rename(output)
    return output
