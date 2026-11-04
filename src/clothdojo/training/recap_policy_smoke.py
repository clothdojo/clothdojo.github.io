"""Exercise RECAP CFG routing, short corrections, two updates and checkpoint reload."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from clothdojo.training.cfg import inspect_cfg_training, write_cfg_launch_manifest
from clothdojo.training.rlinf import _load_runtime, compose_config
from clothdojo.training.rlinf_cfg import ClothDojoCfgWorker, _masked_cfg_loader


class PolicySmokeWorker(ClothDojoCfgWorker):
    def probe_short_correction(self):
        from rlinf.utils.pytree import register_pytree_dataclasses
        from torch.utils._pytree import tree_map

        loader = self.data_loader._data_loader
        corrections = loader.dataset.mixture.datasets[-1]
        index = int(np.flatnonzero(corrections._valid_steps < 50)[0])
        example = corrections[index]
        assert example["advantage"] and example["action_loss_mask"].sum() == 20
        batch = loader.collate_fn([example] * int(self.cfg.actor.micro_batch_size))
        observation, actions, advantage = next(iter(_masked_cfg_loader(None, [batch])))
        register_pytree_dataclasses(observation)
        observation = tree_map(lambda value: torch.as_tensor(value).to(self.device), observation)
        self.model.train()
        loss, _ = self.model(data={"observation": observation,
                                   "actions": actions.to(self.device),
                                   "advantage": advantage.to(self.device)})
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite short-correction loss")
        loss.backward()
        self.optimizer.zero_grad(set_to_none=True)
        return {"rank": self._rank, "short_correction_steps": 20, "loss": float(loss.detach())}

    def checkpoint_roundtrip(self, path: str):
        def snapshot():
            return torch.cat([p.detach().flatten()[:16].clone() for p in self.model.parameters()
                              if p.numel()])

        before = snapshot()
        self.save_checkpoint(path, 2)
        with torch.no_grad():
            next(p for p in self.model.parameters() if p.numel()).add_(0.1)
        assert not torch.equal(before, snapshot())
        self.load_checkpoint(path)
        torch.testing.assert_close(snapshot(), before, rtol=0, atol=0)
        return {"rank": self._rank, "checkpoint_exact_reload": True,
                "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    overrides = [*args.overrides, f"runner.logger.log_path={output}"]
    cfg = compose_config(args.config, overrides)
    report = inspect_cfg_training(args.config, overrides)
    report["resolved_config"] = OmegaConf.to_container(cfg, resolve=True)
    write_cfg_launch_manifest(cfg, report)
    runtime = _load_runtime("cfg")
    cfg = runtime.validate_cfg(cfg)
    runtime.multiprocessing.set_start_method("spawn", force=True)
    cluster = runtime.cluster_type(cluster_cfg=cfg.cluster)
    placement = runtime.placement_type(cfg, cluster)
    actor = PolicySmokeWorker.create_group(cfg).launch(
        cluster, name="ReCapPolicySmoke", placement_strategy=placement.get_strategy("actor"))
    actor.init_worker().wait()
    probe = actor.probe_short_correction().wait()
    print(json.dumps({"short_correction": probe}), flush=True)
    updates = []
    for step in range(2):
        actor.set_global_step(step).wait()
        metrics = actor.run_training().wait()
        updates.append(metrics[0])
        print(json.dumps({"step": step + 1, "metrics": metrics[0]}), flush=True)
    checkpoint = output / "checkpoint/actor"
    roundtrip = actor.checkpoint_roundtrip(str(checkpoint)).wait()
    from clothdojo.rollout.checkpoint import inspect_rlinf_openpi_checkpoint

    inference = inspect_rlinf_openpi_checkpoint(checkpoint, action_horizon=50)
    result = {"status": "passed", "short_correction": probe, "updates": updates,
              "checkpoint": roundtrip, "inference_checkpoint": inference}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
