#!/usr/bin/env bash
set -euo pipefail
if (( $# )); then
  case "$1" in
    -h|--help) echo 'Usage: bash scripts/setup.sh'; exit 0 ;;
    *) echo 'Usage: bash scripts/setup.sh' >&2; exit 2 ;;
  esac
fi
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
python_bin="${PYTHON:-python3.11}"
if [[ ! -x .venv/bin/python ]]; then
  "$python_bin" -m venv .venv
fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 11), "Python 3.11 is required"'
.venv/bin/python -m pip install -r requirements.txt
site_packages="$(.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')"
if patch --dry-run --batch --forward --silent -p1 -d "$site_packages" < patches/rlinf-recap-value.patch; then
  patch --batch --forward --silent -p1 -d "$site_packages" < patches/rlinf-recap-value.patch
elif ! patch --dry-run --batch --silent -R -p1 -d "$site_packages" < patches/rlinf-recap-value.patch; then
  echo 'Installed RLinf source does not match the pinned patch.' >&2
  exit 1
fi
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/clothdojo env check --training
echo 'Ready. Activate the environment: source .venv/bin/activate'
