"""Train the pinned LeRobot CNN Diffusion Policy on the two existing task datasets.

Run with torch.distributed.run; one optimizer update consumes exactly one batch per rank.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
import yaml

from clothdojo.training.diffusion_data import (
    IMAGE_KEYS,
    BalancedWindowSampler,
    DiffusionDataset,
)


def make_policy(cfg, stats=None, *, device="cpu"):
    from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.configs.types import FeatureType, PolicyFeature

    config = DiffusionConfig(
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (14,)),
            **{
                key: PolicyFeature(FeatureType.VISUAL, (3, cfg["image_size"], cfg["image_size"]))
                for key in IMAGE_KEYS
            },
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (14,))},
        device=device,
        n_obs_steps=cfg["n_obs_steps"],
        horizon=cfg["horizon"],
        n_action_steps=cfg["n_action_steps"],
        down_dims=tuple(cfg["down_dims"]),
        crop_shape=None,
        pretrained_backbone_weights=None,
        use_group_norm=True,
        use_separate_rgb_encoder_per_camera=True,
        noise_scheduler_type="DDIM",
        num_train_timesteps=cfg["num_train_timesteps"],
        num_inference_steps=cfg["num_inference_steps"],
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        do_mask_loss_for_padding=False,
    )
    return DiffusionPolicy(config, dataset_stats=stats)


def validate_batch(cfg, world_size):
    if cfg["gradient_accumulation_steps"] != 1:
        raise ValueError("gradient accumulation is forbidden")
    if cfg["global_batch_size"] != cfg["micro_batch_size"] * world_size:
        raise ValueError("global batch must equal micro batch times world size")
    if cfg["n_obs_steps"] != 2 or cfg["fps"] != 50:
        raise ValueError("this adapter requires two adjacent observations at 50 Hz")
    if not 1 <= cfg["n_action_steps"] <= cfg["horizon"] - 1:
        raise ValueError("invalid execution horizon")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def save_checkpoint(output, step, cfg, model, ema, optimizer, scheduler, rng_states, provenance):
    path = output / f"step_{step:06d}.pt"
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "schema": "clothdojo_diffusion_v1",
            "step": step,
            "config": cfg,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng_states": rng_states,
            "datasets": provenance,
        },
        temporary,
    )
    os.replace(temporary, path)
    atomic_json(output / "latest.json", {"step": step, "checkpoint": path.name})
    return path


def train(args):
    from diffusers.optimization import get_scheduler
    from diffusers.training_utils import EMAModel
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader

    cfg = yaml.safe_load(Path(args.config).read_text())
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    validate_batch(cfg, world)
    torch.set_num_threads(1)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world > 1:
        torch.distributed.init_process_group("nccl")
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dataset = DiffusionDataset(Path(args.project).resolve(), cfg)
    model = make_policy(cfg, dataset.normalization_stats()).to(device)
    optimizer = torch.optim.AdamW(
        model.get_optim_params(),
        lr=cfg["lr"],
        betas=tuple(cfg["betas"]),
        weight_decay=cfg["weight_decay"],
        eps=1e-8,
    )
    scheduler = get_scheduler(
        "cosine",
        optimizer,
        num_warmup_steps=cfg["warmup_steps"],
        num_training_steps=cfg["max_steps"],
    )
    ema = EMAModel(
        model.parameters(),
        decay=cfg["ema_max_decay"],
        use_ema_warmup=True,
        inv_gamma=1.0,
        power=cfg["ema_power"],
        foreach=True,
    )
    start = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint["config"] != cfg or checkpoint["datasets"] != dataset.provenance:
            raise ValueError("resume config or dataset identity differs")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        ema.load_state_dict(checkpoint["ema"])
        ema.to(device)
        start = checkpoint["step"]
    ddp = DistributedDataParallel(model, device_ids=[local_rank]) if world > 1 else model
    torch.manual_seed(cfg["seed"] + rank)
    torch.cuda.manual_seed(cfg["seed"] + rank)
    if args.resume:
        state = checkpoint["rng_states"][rank]
        torch.set_rng_state(state["cpu"])
        torch.cuda.set_rng_state(state["cuda"], device)
        del checkpoint
    steps = min(cfg["max_steps"], args.stop_after or cfg["max_steps"])
    sampler = BalancedWindowSampler(
        dataset,
        rank=rank,
        start_step=start,
        steps=steps,
        batch_size=cfg["micro_batch_size"],
        seed=cfg["seed"],
    )
    workers = cfg["num_workers"]
    loader = DataLoader(
        dataset,
        batch_size=cfg["micro_batch_size"],
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        **({"prefetch_factor": 2, "persistent_workers": True} if workers else {}),
        # Keep DataLoader worker seeding separate from model noise on resume.
        generator=torch.Generator().manual_seed(cfg["seed"] + rank),
    )
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        if (output / "latest.json").exists() and not args.resume:
            raise FileExistsError("existing checkpoint; use --resume or a fresh output")
        atomic_json(
            output / "manifest.json",
            {
                "schema": "clothdojo_diffusion_training_v1",
                "config": cfg,
                "datasets": dataset.provenance,
                "world_size": world,
                "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "started_at": datetime.now(UTC).isoformat(),
                "upstream": "huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5",
                "source_sha256": {
                    name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                    for name in ("diffusion.py", "diffusion_data.py")
                },
            },
        )
        freeze = subprocess.run(
            [os.sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
        )
        (output / "pip_freeze.txt").write_text(freeze.stdout)
        print(
            json.dumps({"event": "ready", "sources": dataset.provenance, "steps": steps}),
            flush=True,
        )
    ddp.train()
    started = time.perf_counter()
    last_log = started
    losses = []
    for step, batch in enumerate(loader, start=start + 1):
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = ddp(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite loss at step {step}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg["grad_clip_norm"], error_if_nonfinite=True
        )
        optimizer.step()
        scheduler.step()
        ema.step(model.parameters())
        losses.append(loss.detach())
        if step % cfg["log_interval"] == 0 or step == steps:
            value = torch.stack(losses).mean()
            if world > 1:
                torch.distributed.all_reduce(value)
                value /= world
            torch.cuda.synchronize()
            now = time.perf_counter()
            if rank == 0:
                record = {
                    "step": step,
                    "loss": value.item(),
                    "grad_norm": float(norm),
                    "lr": scheduler.get_last_lr()[0],
                    "seconds_per_step": (now - last_log) / len(losses),
                    "max_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "elapsed_seconds": now - started,
                    "ema_decay": ema.cur_decay_value,
                }
                print(json.dumps(record), flush=True)
                with (output / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
            last_log, losses = now, []
        if step % cfg["save_interval"] == 0 or step == steps:
            state = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}
            states = [None] * world
            if world > 1:
                torch.distributed.all_gather_object(states, state)
            else:
                states = [state]
            if rank == 0:
                path = save_checkpoint(
                    output, step, cfg, model, ema, optimizer, scheduler, states, dataset.provenance
                )
                print(json.dumps({"event": "checkpoint", "path": str(path)}), flush=True)
            if world > 1:
                torch.distributed.barrier()
    if rank == 0:
        atomic_json(
            output / "status.json",
            {
                "status": "complete" if steps == cfg["max_steps"] else "pilot_complete",
                "step": steps,
            },
        )
    if world > 1:
        torch.distributed.destroy_process_group()


def launch(config, arguments, *, check=False):
    """Public CLI: inspect data or launch one local DDP job after resource reservation."""
    cfg = yaml.safe_load(Path(config).read_text())
    world = cfg["global_batch_size"] // cfg["micro_batch_size"]
    validate_batch(cfg, world)
    if check:
        parser = argparse.ArgumentParser()
        parser.add_argument("--project", default=".")
        args = parser.parse_args(arguments)
        dataset = DiffusionDataset(Path(args.project).resolve(), cfg)
        dataset.normalization_stats()
        print(
            json.dumps(
                {"config": cfg, "world_size": world, "datasets": dataset.provenance}, indent=2
            )
        )
        return
    subprocess.run(
        [
            os.sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={world}",
            "-m",
            "clothdojo.training.diffusion",
            "--config",
            config,
            *arguments,
        ],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--project", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--stop-after", type=int)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
