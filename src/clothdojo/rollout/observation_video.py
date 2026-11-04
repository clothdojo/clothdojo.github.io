"""Small review videos from the observations passed to a chunk policy."""

from contextlib import contextmanager
from pathlib import Path

import numpy as np

from clothdojo.rendering import CAMERA_NAMES


@contextmanager
def record_policy_observations(policy, outputs, *, fps):
    """Record only published worlds; never render or modify policy inputs."""
    import imageio_ffmpeg

    encoders = {}

    class RecordingPolicy:
        def reset(self, *args, **kwargs):
            return policy.reset(*args, **kwargs)

        def infer(self, observation, prompt, **kwargs):
            for world, output in enumerate(outputs):
                if output is None:
                    continue
                frame = np.ascontiguousarray(np.concatenate([
                    observation[f"observation.images.{name}"][world]
                    for name in CAMERA_NAMES
                ], axis=1))
                if world not in encoders:
                    path = Path(output) / "policy_obs.mp4"
                    encoder = imageio_ffmpeg.write_frames(
                        str(path), (frame.shape[1], frame.shape[0]),
                        fps=fps, codec="libx264", quality=None, macro_block_size=1,
                        output_params=["-vf", "scale=960:240", "-crf", "28",
                                       "-threads", "1", "-movflags", "+faststart"],
                    )
                    encoder.send(None)
                    encoders[world] = encoder
                encoders[world].send(frame)
            return policy.infer(observation, prompt, **kwargs)

        @property
        def last_semantic(self):
            return getattr(policy, "last_semantic", None)

    try:
        yield RecordingPolicy()
    finally:
        for encoder in encoders.values():
            encoder.close()
