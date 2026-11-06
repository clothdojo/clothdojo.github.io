#!/usr/bin/env bash
set -euo pipefail
usage() {
  echo 'Usage: bash scripts/setup.sh [--rgb] [--xpolicy]'
  echo '  --rgb  Add Isaac 5 to the same environment; install base dependencies if needed.'
  echo '  --xpolicy  Install the pinned XPolicyLab policy-serving runtime.'
}
install_rgb=false
install_xpolicy=false
while (( $# )); do
  case "$1" in
    --rgb) install_rgb=true ;;
    --xpolicy) install_xpolicy=true ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
  shift
done
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
python_bin="${PYTHON:-python3.11}"
if [[ ! -x .venv/bin/python ]]; then
  "$python_bin" -m venv .venv
fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 11), "Python 3.11 is required"'
base_ready=false
if { $install_rgb || $install_xpolicy; } && [[ -x .venv/bin/clothdojo ]] && \
    .venv/bin/clothdojo env check --training >/dev/null 2>&1; then
  base_ready=true
  if $install_xpolicy; then
    echo 'Existing environment found; adding XPolicyLab and requested components.'
  else
    echo 'Existing NOCS environment found; adding Isaac only.'
  fi
fi
if ! $base_ready; then
  .venv/bin/python -m pip install -r requirements.txt
  site_packages="$(.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')"
  if patch --dry-run --batch --forward --silent -p1 -d "$site_packages" < patches/rlinf-recap-value.patch; then
    patch --batch --forward --silent -p1 -d "$site_packages" < patches/rlinf-recap-value.patch
  elif ! patch --dry-run --batch --silent -R -p1 -d "$site_packages" < patches/rlinf-recap-value.patch; then
    echo 'Installed RLinf source does not match the pinned patch.' >&2
    exit 1
  fi
  .venv/bin/python -m pip install --no-deps -e .
fi
if $install_rgb; then
  .venv/bin/python -m pip install --extra-index-url https://pypi.nvidia.com --no-deps \
    'isaacsim==5.0.0' 'isaacsim-app==5.0.0' 'isaacsim-core==5.0.0' \
    'isaacsim-replicator==5.0.0' 'isaacsim-extscache-kit==5.0.0' \
    'isaacsim-extscache-kit-sdk==5.0.0' 'isaacsim-extscache-physics==5.0.0' \
    'isaacsim-kernel==5.0.0' 'omniverse-kit==107.3.1.206797'
  .venv/bin/python -c 'from clothdojo.data.isaac_render import _require_isaac5_runtime; _require_isaac5_runtime()'
fi
if $install_xpolicy; then
  git submodule update --init XPolicyLab
  .venv/bin/python -m pip install 'msgpack-numpy==0.4.8'
  .venv/bin/python -m pip install --no-deps -e XPolicyLab
fi
.venv/bin/clothdojo env check --training
echo 'Ready. Activate the environment: source .venv/bin/activate'
