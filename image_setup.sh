#!/usr/bin/env bash
#
# image_setup.sh — additive bootstrap for FLUX.2-klein-9B image generation
# with the ponpoke uncensored Qwen3 text encoder. Run AFTER setup.sh on the
# same VM (it reuses the .venv that setup.sh creates). Safe to re-run.
#
# Target hardware: AWS g6.xlarge (NVIDIA L4, 22.5 GB VRAM, 16 GiB sys RAM).
# That's tight for a 9B diffusion transformer + an 8B Qwen3 text encoder, so
# we enable sequential CPU offload at runtime and load the text encoder in
# 4-bit via bitsandbytes. Disk footprint is large (~35-40 GB once cached) —
# the EC2 root volume is sized for this (50 GB gp3).
#
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${REPO_DIR}/.venv"

# Prebuilt PyTorch cu124 wheels — same CUDA major as the llama-cpp wheel,
# so they coexist cleanly in the same venv on this driver-only AMI.
TORCH_INDEX="https://download.pytorch.org/whl/cu124"

echo "=============================================================="
echo " gpu-llm-vm-setup :: image-gen bootstrap (FLUX.2 Klein 9B)"
echo " repo dir : ${REPO_DIR}"
echo "=============================================================="

if [ ! -d "${VENV_DIR}" ]; then
    echo "  ERROR: ${VENV_DIR} not found. Run 'bash setup.sh' first."
    exit 1
fi

export PATH="${HOME}/.local/bin:${PATH}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python --version

echo
echo "[1/4] Installing PyTorch + torchvision (cu124 prebuilt wheels)..."
uv pip install \
    --extra-index-url "${TORCH_INDEX}" \
    --index-strategy unsafe-best-match \
    "torch==2.5.1" "torchvision==0.20.1"

echo
echo "[2/4] Installing diffusers + transformers + accelerate stack..."
uv pip install \
    "diffusers>=0.32.0" \
    "transformers>=4.46.0" \
    "accelerate>=1.1.0" \
    "safetensors>=0.4.5" \
    "sentencepiece" \
    "protobuf" \
    "peft" \
    "Pillow"

echo
echo "[3/4] Installing bitsandbytes (4-bit quant for the text encoder)..."
echo "      Reason: L4 has 22.5 GB VRAM; FLUX.2 Klein 9B transformer + an 8B"
echo "      Qwen3 text encoder in fp16 do NOT both fit. 4-bit text encoder"
echo "      drops it from ~16 GB -> ~5 GB and leaves room for the diffuser."
uv pip install "bitsandbytes>=0.44.0"

echo
echo "[4/4] Installing Gradio UI + Hugging Face CLI..."
uv pip install \
    "gradio>=5.0.0" \
    "huggingface_hub[cli]>=0.26.0"

echo
echo "--------------------------------------------------------------"
echo " Verifying torch + CUDA visibility..."
echo "--------------------------------------------------------------"
# shellcheck disable=SC1091
source "${REPO_DIR}/_env.sh"
python - <<'PYEOF'
import torch
print("  torch:", torch.__version__)
print("  CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("  device:", torch.cuda.get_device_name(0))
    free, total = torch.cuda.mem_get_info(0)
    print(f"  VRAM: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")
import diffusers, transformers, gradio
print("  diffusers:", diffusers.__version__)
print("  transformers:", transformers.__version__)
print("  gradio:", gradio.__version__)
PYEOF

echo
echo "=============================================================="
echo " SUCCESS — image-gen environment ready."
echo "=============================================================="
echo
echo " Before running the UI you MUST:"
echo "   1. 'huggingface-cli login'  (a token with 'read' is enough)"
echo "      — FLUX.2 Klein is gated; you must accept the license once at"
echo "        https://huggingface.co/black-forest-labs/FLUX.2-klein-9B"
echo "   2. The ponpoke text encoder is gated='auto' — first download"
echo "      auto-approves once you're logged in."
echo
echo " Then in a REAL foreground TTY:"
echo
echo "   bash image_ui.sh        # Gradio UI on http://<vm-ip>:7860"
echo
echo " NOTE: first run downloads ~35 GB (FLUX.2 Klein base + Qwen3 encoder)."
echo "=============================================================="
