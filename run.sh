#!/usr/bin/env bash
# One-shot test inference (foreground). Run after `bash setup.sh`.
set -euo pipefail
_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${_DIR}/_env.sh"
exec python "${_DIR}/model_infer.py"
