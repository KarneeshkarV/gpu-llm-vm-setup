#!/usr/bin/env bash
# Launch the FLUX.2 Klein Gradio UI in the foreground (real TTY).
# Run after `bash setup.sh && bash image_setup.sh` and after
# `huggingface-cli login` (FLUX.2 Klein base is gated).
set -euo pipefail
_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${_DIR}/_env.sh"
exec python "${_DIR}/image_app.py"
