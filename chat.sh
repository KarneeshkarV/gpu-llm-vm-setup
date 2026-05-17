#!/usr/bin/env bash
# Interactive streaming chat REPL (foreground TTY). Run after `bash setup.sh`.
set -euo pipefail
_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${_DIR}/_env.sh"
exec python "${_DIR}/chatbot.py"
