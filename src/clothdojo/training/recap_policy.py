"""Run the checked RECAP labels through the existing ClothDojo CFG trainer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from clothdojo.runtime_identity import runtime_source_sha256
from clothdojo.training.cfg import inspect_cfg_training
from clothdojo.training.rlinf import compose_config, run_config


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = compose_config(args.config, args.overrides)
    if cfg.get("clothdojo", {}).get("advantage_method") != "recap":
        raise ValueError("RECAP policy requires a RECAP config")
    report = inspect_cfg_training(args.config, args.overrides)
    report["resolved_config"] = OmegaConf.to_container(cfg, resolve=True)
    report["runtime_sha256"] = runtime_source_sha256(Path(__file__).resolve().parents[1])
    print(json.dumps(report, indent=2), flush=True)
    if not args.preflight:
        run_config(kind="cfg", cfg=cfg, cfg_manifest=report, config_path=args.config)


if __name__ == "__main__":
    main()
