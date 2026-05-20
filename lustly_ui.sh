#!/usr/bin/env bash
# Launch the FLUX.1-dev + Lustly LoRA Gradio UI in the foreground.
# Run after `bash setup.sh && bash image_setup.sh` and after HF login.
set -euo pipefail
_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${_DIR}/_env.sh"
exec python "${_DIR}/lustly_app.py"
