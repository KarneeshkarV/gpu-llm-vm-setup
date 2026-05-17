#!/usr/bin/env bash
#
# setup.sh — single idempotent bootstrap for running a local GGUF LLM on a
# fresh GPU VM (AWS "Deep Learning OSS Nvidia Driver AMI", driver-only, no
# CUDA toolkit). Safe to re-run.
#
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${REPO_DIR}/.venv"
PY_VERSION="3.12"

# cu124 prebuilt wheels are backward-compatible with newer CUDA drivers.
LLAMA_WHEEL_INDEX="https://abetlen.github.io/llama-cpp-python/whl/cu124"

echo "=============================================================="
echo " gpu-llm-vm-setup :: bootstrap"
echo " repo dir : ${REPO_DIR}"
echo "=============================================================="

# -------------------------------------------------------------------------
echo
echo "[1/6] Checking for a GPU (nvidia-smi)..."
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
else
    echo "  WARNING: nvidia-smi not found. This setup REQUIRES an NVIDIA GPU."
    echo "           GPU offload is mandatory (CPU inference OOM-kills on 16 GiB RAM)."
    echo "           Continuing anyway, but inference will likely fail."
fi

# -------------------------------------------------------------------------
echo
echo "[2/6] Ensuring uv is installed..."
export PATH="${HOME}/.local/bin:${PATH}"
if command -v uv >/dev/null 2>&1; then
    echo "  uv already present: $(command -v uv)"
else
    echo "  Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi
uv --version

# -------------------------------------------------------------------------
echo
echo "[3/6] Creating Python ${PY_VERSION} virtualenv at ${VENV_DIR} ..."
if [ -d "${VENV_DIR}" ]; then
    echo "  .venv already exists, reusing it (idempotent)."
else
    uv venv --python "${PY_VERSION}" "${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python --version

# -------------------------------------------------------------------------
echo
echo "[4/6] Installing llama-cpp-python from the PREBUILT cu124 wheel index..."
echo "      (Driver-only AMI has no nvcc; a CUDA source build WILL fail."
echo "       --index-strategy unsafe-best-match is REQUIRED or uv ignores"
echo "       the wheel index and falls back to the broken PyPI source build.)"
uv pip install llama-cpp-python \
    --extra-index-url "${LLAMA_WHEEL_INDEX}" \
    --index-strategy unsafe-best-match

# -------------------------------------------------------------------------
echo
echo "[5/6] Installing huggingface_hub + CUDA 12 runtime shared libs..."
echo "      (Driver-only AMI lacks libcudart.so.12 / libcublas; the prebuilt"
echo "       wheel needs them at runtime via LD_LIBRARY_PATH — see _env.sh.)"
uv pip install \
    huggingface_hub \
    nvidia-cuda-runtime-cu12 \
    nvidia-cublas-cu12

# -------------------------------------------------------------------------
echo
echo "[6/6] Verifying 'import llama_cpp' works (with CUDA libs on path)..."
# import llama_cpp needs libcudart.so.12 -> LD_LIBRARY_PATH. _env.sh sets it
# (and re-activates the venv, which is idempotent).
# shellcheck disable=SC1091
source "${REPO_DIR}/_env.sh"
python - <<'PYEOF'
import llama_cpp
print("  llama_cpp OK, version:", getattr(llama_cpp, "__version__", "unknown"))
PYEOF

# -------------------------------------------------------------------------
echo
echo "=============================================================="
echo " SUCCESS — environment is ready."
echo "=============================================================="
echo
echo " Next steps (run these in a REAL terminal/TTY, foreground):"
echo
echo "   bash run.sh     # one-shot test inference (capital of France)"
echo "   bash chat.sh    # interactive streaming chat REPL"
echo
echo " NOTE: first run downloads the model (~16 GB) from Hugging Face."
echo "=============================================================="
