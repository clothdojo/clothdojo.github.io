"""Exercise the real upstream WebSocket protocol, not a mocked policy socket."""

import asyncio
import hashlib
import importlib.util
import json
import threading

import numpy as np
import pytest

from clothdojo.rendering import CAMERA_NAMES
from clothdojo.rollout.xpolicy import ArrayModel, HostedModel, validate_model_name


def test_checkpoint_only_config_and_model_session_isolation(tmp_path):
    from clothdojo.evaluation.managed import _policy_command, _policy_slots
    from clothdojo.evaluation.run_config import load_benchmark_run_config

    def write(name, value):
        (tmp_path / name).write_text(json.dumps(value))

    write(
        "benchmark.json",
        {
            "schema": "clothdojo_frozen_benchmark",
            "panel_id": "rgb_panel",
            "scenario": "scenario.json",
            "protocol": "protocol.json",
            "states": "states",
            "templates": "templates",
            "assets": "assets",
        },
    )
    deployment = {
        "schema": "clothdojo_evaluation_deployment",
        "policy": {"nodes": [{"host": "local", "gpus": [0], "port_start": 28000}]},
        "simulation": {"nodes": [{"host": "local", "gpus": [1]}]},
    }
    write("deployment.json", deployment)
    run = {
        "schema": "clothdojo_benchmark_run",
        "condition_id": "rgb_test",
        "benchmark": "benchmark.json",
        "deployment": "deployment.json",
        "policy": {"kind": "xpolicy", "checkpoint": "model.pt"},
        "execution": {"cloth_visual": "rgb", "visual_profiles": ["train", "test_all"]},
    }
    write("run.json", run)
    config = load_benchmark_run_config(tmp_path / "run.json", project_root=tmp_path)
    assert config.policy.norm_stats is None
    slot = _policy_slots(config)[0]
    assert slot.endpoint.startswith("xpolicy://")
    command = _policy_command(config, slot)
    assert "serve-xpolicy" in command
    assert "--norm-stats" not in command
    assert "--image-preprocessing" not in command
    deployment["max_simulation_workers_per_policy"] = 2
    write("deployment.json", deployment)
    with pytest.raises(ValueError, match="one simulation worker"):
        load_benchmark_run_config(tmp_path / "run.json", project_root=tmp_path)
    deployment["max_simulation_workers_per_policy"] = 1
    write("deployment.json", deployment)
    run["execution"]["mode"] = "rtc"
    write("run.json", run)
    with pytest.raises(ValueError, match="chunk execution"):
        load_benchmark_run_config(tmp_path / "run.json", project_root=tmp_path)


class EchoModel(ArrayModel):
    observation_horizon = 2
    transport_image_size = (224, 224)

    def predict(self, observations):
        for obs in observations:
            # Catch an unnoticed RGB/BGR swap and history sampled at inference rate.
            assert obs["vision"]["cam_high"]["color"][0, 0].tolist() == [253, 31, 17]
            assert obs["state_history"].shape == (2, 14)
        return np.stack(
            [
                np.repeat(obs["state_history"][-1][None], self.action_horizon, axis=0)
                for obs in observations
            ]
        )


@pytest.mark.parametrize("name", ["../../bad:Model", "bad", "pkg:Model:extra"])
def test_adapter_names_fail_before_launch(name):
    with pytest.raises(ValueError, match="importable"):
        validate_model_name(name)


def test_official_transport_preserves_rgb_state_history_and_reset(tmp_path):
    if (
        importlib.util.find_spec("XPolicyLab") is None
        or importlib.util.find_spec("msgpack_numpy") is None
    ):
        pytest.skip("install the optional XPolicyLab runtime with setup.sh --xpolicy")
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig
    from clothdojo.rollout.policy import RemotePolicy

    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"protocol test artifact")
    model = EchoModel({})
    server = PolicyServer(
        HostedModel(model, checkpoint), PolicyServerConfig(host="127.0.0.1", port=0)
    )
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def run():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.start())
        ready.set()
        loop.run_forever()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert ready.wait(10)
    client = None
    try:
        client = RemotePolicy(
            server.url.replace("ws://", "xpolicy://"),
            expected_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            action_horizon=8,
        )
        client.reset(17, mode="chunk", exec_horizon=8)
        state = np.arange(56, dtype=np.float32).reshape(2, 2, 14) / 100
        image = np.broadcast_to(np.array([253, 31, 17], dtype=np.uint8), (2, 2, 224, 224, 3)).copy()
        observations = {
            "observation.state": state,
            **{f"observation.images.{name}": image for name in CAMERA_NAMES},
        }
        actions = client.infer(observations, "flatten the shirt")
        np.testing.assert_array_equal(actions, np.repeat(state[:, -1:, :], 8, axis=1))
        assert model.seed == 17
        large_seed = 2**63 + 29
        client.reset(large_seed, mode="chunk", exec_horizon=8)
        client.infer(observations, "flatten the shirt")
        assert model.seed == large_seed
    finally:
        if client is not None:
            client._client.close()
        asyncio.run_coroutine_threadsafe(server.stop(), loop).result(timeout=10)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=10)
        loop.close()
