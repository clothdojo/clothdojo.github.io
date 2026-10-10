"""Stream Isaac pixels to an independently installed LeRobot environment."""

import argparse
import json
import os
import socket
import struct
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from clothdojo.rendering import DATASET_CAMERA_NAMES

_IMAGE_SHAPE = (len(DATASET_CAMERA_NAMES), 480, 640, 3)
_IMAGE_BYTES = int(np.prod(_IMAGE_SHAPE))


def _receive(connection, count):
    result = bytearray(count)
    view = memoryview(result)
    while view:
        received = connection.recv_into(view)
        if not received:
            raise EOFError("RGB export worker disconnected before completing the transfer")
        view = view[received:]
    return result


def _send_json(connection, value):
    data = json.dumps(value).encode()
    if len(data) > 65536:
        raise ValueError("RGB export metadata exceeds 64 KiB")
    connection.sendall(struct.pack("!I", len(data)) + data)


def _receive_json(connection):
    count, = struct.unpack("!I", _receive(connection, 4))
    if count > 65536:
        raise ValueError("RGB export metadata exceeds 64 KiB")
    return json.loads(_receive(connection, count))


def _serve(connection, camera):
    profile = camera.scene.robot_profile
    _send_json(connection, {
        "arm_joint_names": profile.arm_joint_names,
        "arm_coord_indices": profile.arm_coord_indices,
        "layouts": camera.layouts,
    })
    expected = 0
    while True:
        message = _receive_json(connection)
        if message == {"done": True}:
            if expected != camera.reader.manifest.action_count:
                raise RuntimeError("RGB export worker stopped before reading all action frames")
            return
        if message != {"frame": expected} or type(message.get("frame")) is not int:
            raise ValueError("RGB export worker requested an unexpected frame")
        if expected >= camera.reader.manifest.action_count:
            raise IndexError("RGB export worker requested a terminal state without an action")
        views = camera(expected)
        images = [views[f"observation.images.{name}"] for name in DATASET_CAMERA_NAMES]
        if any(image.shape != (1, 480, 640, 3) or image.dtype != np.uint8 for image in images):
            raise ValueError("RGB export requires uint8[1,480,640,3] camera images")
        connection.sendall(np.concatenate(images).tobytes())
        expected += 1


def export_in_separate_python(camera, trajectory, output, *, assets_root, python):
    """Keep Isaac and LeRobot native/Python dependencies in separate processes."""
    parent, child = socket.socketpair()
    parent.settimeout(600)
    process = None
    try:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        env.pop("PYTHONPATH", None)
        process = subprocess.Popen([
            str(python), "-m", "clothdojo.data.rgb_export", str(Path(trajectory).resolve()),
            str(Path(output).resolve()), "--assets", str(Path(assets_root).resolve()),
            "--fd", str(child.fileno()),
        ], pass_fds=(child.fileno(),), env=env)
        child.close()
        _serve(parent, camera)
        returncode = process.wait(timeout=60)
        if returncode:
            raise subprocess.CalledProcessError(returncode, process.args)
        return Path(output)
    finally:
        parent.close()
        child.close()
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def _worker(connection, trajectory, output, assets_root):
    from clothdojo.trajectory import TrajectoryReader
    from clothdojo.trajectory.lerobot import export_lerobot

    metadata = _receive_json(connection)

    class Camera:
        reader = TrajectoryReader(trajectory)
        scene = SimpleNamespace(robot_profile=SimpleNamespace(
            arm_joint_names=metadata["arm_joint_names"],
            arm_coord_indices=metadata["arm_coord_indices"],
        ))
        layouts = metadata["layouts"]

        def __call__(self, index):
            _send_json(connection, {"frame": index})
            images = np.frombuffer(_receive(connection, _IMAGE_BYTES), np.uint8).reshape(_IMAGE_SHAPE)
            return {f"observation.images.{name}": images[i:i + 1]
                    for i, name in enumerate(DATASET_CAMERA_NAMES)}

    export_lerobot(trajectory, output, assets_root=assets_root, replay_camera=Camera())
    _send_json(connection, {"done": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--fd", type=int, required=True)
    args = parser.parse_args()
    with socket.socket(fileno=args.fd) as connection:
        connection.settimeout(600)
        _worker(connection, args.trajectory, args.output, args.assets)
