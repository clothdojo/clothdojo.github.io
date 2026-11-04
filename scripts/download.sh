#!/usr/bin/env bash
set -euo pipefail

task="${1:-flatten}"
case "$task" in
  -h|--help) echo "Usage: bash scripts/download.sh [flatten|fold]"; exit 0 ;;
  flatten|fold) ;;
  *) echo "Usage: bash scripts/download.sh [flatten|fold]" >&2; exit 2 ;;
esac
if (( $# > 1 )); then
  echo "Usage: bash scripts/download.sh [flatten|fold]" >&2
  exit 2
fi
command -v uvx >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/' >&2; exit 1; }
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1

# Download only the assets required for trajectory generation and replay.
uvx --from 'huggingface-hub==0.32.3' huggingface-cli download clothdojo/assets \
  --repo-type dataset --revision a0628f42dde13ebb0505fd9dc255d4bda50edb94 \
  --include 'benchmark/*' --local-dir "$project_root"

uvx --from 'huggingface-hub==0.32.3' huggingface-cli download clothdojo/assets \
  --repo-type dataset --revision a0628f42dde13ebb0505fd9dc255d4bda50edb94 \
  --include 'LICENSE' 'LICENSE-DATA' 'LICENSE_SCOPE.md' 'THIRD_PARTY_NOTICES.md' 'licenses/*' \
  --local-dir "$project_root/licenses/assets"

uvx --from 'huggingface-hub==0.32.3' huggingface-cli download clothdojo/nocs-teachers \
  --revision 1d9a6357557df2d722649709738b7a4a62617452 \
  --include "$task/*" '*.md' '*LICENSE*' '*NOTICE*' \
  --local-dir "$project_root/models/nocs-teachers"

echo "Assets and $task teacher downloaded. Code is supplied by this GitHub checkout."
