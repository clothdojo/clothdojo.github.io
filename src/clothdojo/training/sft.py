"""Single-task reference SFT with explicit data windows and EMA inference weights."""

import hashlib
import json
import os
import shutil
from pathlib import Path

import torch
from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker

from clothdojo.training.augmentation import install_training_augmentation
from clothdojo.training.ema import (
    _initialize_ema_state,
    _swap_parameter_data_with_ema,
    _update_ema_state,
    _validate_fp32_training_state,
)
from clothdojo.training.openpi import validate_action_chunk_boundary
from clothdojo.training.sft_data import _adapt_official_openpi_loader, _set_training_prompt


class Pi05SftWorker(FSDPVlaSftWorker):
    def __init__(self, cfg):
        import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
        from rlinf.utils.utils import seed_everything

        lerobot_dataset.get_safe_default_codec = lambda: "pyav"
        seed_everything(int(cfg.actor.seed) + int(os.environ.get("RANK", 0)))
        self.boundary = validate_action_chunk_boundary(cfg.clothdojo.action_chunk_boundary)
        super().__init__(cfg)

    def build_dataloader(self, data_paths, eval_dataset=False):
        loader, config = super().build_dataloader(data_paths, eval_dataset=eval_dataset)
        _set_training_prompt(loader, self.cfg.actor.model.openpi_data.default_prompt)
        if not eval_dataset:
            _adapt_official_openpi_loader(loader, seed=int(self.cfg.actor.seed),
                                         rank=int(self._rank), action_horizon=50,
                                         action_chunk_boundary=self.boundary)
        return loader, config

    def init_worker(self):
        super().init_worker()
        module = getattr(self.model, "module", self.model)
        install_training_augmentation(module.model, enabled=False)
        _validate_fp32_training_state(self.model, self.optimizer)
        self.ema_decay = float(self.cfg.actor.optim.ema_decay)
        if self.ema_decay != 0.99:
            raise ValueError("the reference policy bundle requires EMA decay 0.99")
        self.ema_state = _initialize_ema_state(self.model)

    def optimizer_step(self):
        grad_norm, lr_list = super().optimizer_step()
        if not torch.isfinite(torch.as_tensor(grad_norm)).item():
            raise FloatingPointError("SFT encountered non-finite gradients")
        _update_ema_state(self.model, self.ema_state, self.ema_decay)
        return grad_norm, lr_list

    def save_checkpoint(self, save_path, step=0):
        # This reference workflow publishes inference weights. It does not
        # write a second model and optimizer archive or advertise resume.
        _swap_parameter_data_with_ema(self.model, self.ema_state)
        try:
            state = self.get_model_state_dict(cpu_offload=True, full_state_dict=True)
            if torch.distributed.get_rank() == 0:
                root = Path(save_path)
                weights = root / "weights/model_state_dict/full_weights.pt"
                weights.parent.mkdir(parents=True, exist_ok=True)
                if weights.exists():
                    raise FileExistsError(weights)
                temporary = weights.with_suffix(".pt.tmp")
                try:
                    torch.save(state, temporary)
                    temporary.replace(weights)
                finally:
                    temporary.unlink(missing_ok=True)
                layout = {
                    "schema": "clothdojo_openpi_rlinf_checkpoint", "format_version": 1,
                    "step": int(step), "ema": {"decay": self.ema_decay,
                    "path": "model_state_dict/full_weights.pt", "purpose": "inference and rollout"},
                }
                (root / "weights/clothdojo_weights.json").write_text(json.dumps(layout, indent=2) + "\n")
                stats = Path(self.cfg.actor.model.openpi_data.norm_stats_path)
                stats = stats if stats.is_dir() else stats.parent
                binding = {"schema": "clothdojo_pi05_launch_preflight", "compatible": True,
                           "norm_stats": {key: hashlib.sha256((stats / name).read_bytes()).hexdigest()
                           for key, name in (("stats_sha256", "norm_stats.json"),
                                             ("provenance_sha256", "clothdojo.json"))}}
                (root / "weights/clothdojo_launch.json").write_text(json.dumps(binding, indent=2) + "\n")
                shutil.copyfile(stats / "norm_stats.json", root / "norm_stats.json")
                tokenizer = Path(self.cfg.data.openpi_data_home) / "big_vision/paligemma_tokenizer.model"
                (root / "cache/big_vision").mkdir(parents=True, exist_ok=True)
                shutil.copyfile(tokenizer, root / "cache/big_vision/paligemma_tokenizer.model")
                from clothdojo.rollout.checkpoint import rlinf_openpi_weights_sha256

                policy = {"schema": "clothdojo_pi05_policy_v1", "task": self.cfg.clothdojo.task,
                          "cloth_visual": self.cfg.clothdojo.cloth_visual, "image_preprocessing": "none",
                          "weights_sha256": rlinf_openpi_weights_sha256(root / "weights"),
                          "norm_stats_sha256": binding["norm_stats"]["stats_sha256"],
                          "tokenizer_sha256": hashlib.sha256(tokenizer.read_bytes()).hexdigest()}
                initial = self.cfg.clothdojo.get("initial_states")
                if initial:
                    shutil.copyfile(initial, root / "initial_states.npz")
                    policy["initial_states_sha256"] = hashlib.sha256((root / "initial_states.npz").read_bytes()).hexdigest()
                (root / "policy.json").write_text(json.dumps(policy, indent=2) + "\n")
            torch.distributed.barrier()
        finally:
            _swap_parameter_data_with_ema(self.model, self.ema_state)

    def load_checkpoint(self, load_path):
        raise ValueError("start a new fine-tuning run; optimizer resume is not supported")
