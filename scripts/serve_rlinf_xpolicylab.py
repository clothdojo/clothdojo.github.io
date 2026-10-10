"""Serve a historical ClothDojo RLinf policy bundle through XPolicyLab."""

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--xpl-root", type=Path, default=Path("XPolicyLab"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-steps", type=int, default=10)
    args = parser.parse_args()
    xpl_root = args.xpl_root.expanduser().resolve()
    bundle = args.bundle.expanduser().resolve()
    if not (xpl_root / "setup_policy_server.py").is_file():
        parser.error(f"XPolicyLab checkout not found: {xpl_root}")
    if not (bundle / "policy.json").is_file():
        parser.error(f"ClothDojo RLinf policy bundle not found: {bundle}")
    if not 1 <= args.port <= 65535 or args.seed < 0 or args.num_steps < 1:
        parser.error("port, seed or num_steps is invalid")

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(xpl_root.parent))
    sys.path.insert(0, str(xpl_root))
    import XPolicyLab.policy as policy_package
    from XPolicyLab.setup_policy_server import main as serve

    policy_package.__path__.append(str(root / "integrations/xpolicylab"))
    serve({
        "policy_name": "ClothDojo_RLinf",
        "protocol": "ws",
        "host": args.host,
        "port": args.port,
        "env_cfg_type": "piper",
        "action_type": "joint",
        "bundle": str(bundle),
        "device": args.device,
        "seed": args.seed,
        "num_steps": args.num_steps,
    })


if __name__ == "__main__":
    main()
