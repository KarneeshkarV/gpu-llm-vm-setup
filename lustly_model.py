"""lustly_model.py — load FLUX.1-dev with the Lustly.ai uncensored LoRA.

Uses the standard diffusers `FluxPipeline` (FLUX.1 uses CLIP + T5 text
encoders — no custom encoder swap needed). The LoRA is a ~344 MB
safetensors file from momofuhshoh/Flux_Lustly.ai_Uncensored_nsfw_v1.

Sizing:
  * Base FLUX.1-dev:  12B params, ~24 GB in bf16
  * Lustly LoRA:      ~344 MB
  * Comfortably fits on the L40S (46 GB), no offload.
  * On the L4 (22.5 GB), enable_sequential_cpu_offload kicks in below
    30 GB free VRAM (same threshold as the FLUX.2 path).
"""

from __future__ import annotations

import os

import torch
from diffusers import FluxPipeline
from huggingface_hub import hf_hub_download

BASE_REPO = "black-forest-labs/FLUX.1-dev"
LORA_REPO = "momofuhshoh/Flux_Lustly.ai_Uncensored_nsfw_v1"
LORA_FILE = "flux_lustly-ai_v1.safetensors"
LORA_ADAPTER_NAME = "lustly"

DTYPE = torch.bfloat16
FAST_PATH_VRAM_GB = 30.0


def _mode() -> str:
    forced = os.environ.get("IMAGE_MODE", "").strip().lower()
    if forced in {"fast", "tight"}:
        return forced
    if not torch.cuda.is_available():
        return "tight"
    free, _ = torch.cuda.mem_get_info(0)
    return "fast" if (free / 1e9) >= FAST_PATH_VRAM_GB else "tight"


def load_pipeline(lora_strength: float = 0.85):
    """Load FLUX.1-dev + Lustly LoRA. Returns the pipe ready for inference."""
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    mode = _mode()
    print(f"[lustly_model] mode={mode}")

    print(f"[lustly_model] Loading {BASE_REPO} (bf16)...")
    pipe = FluxPipeline.from_pretrained(
        BASE_REPO,
        torch_dtype=DTYPE,
        token=hf_token,
    )

    print(f"[lustly_model] Downloading LoRA from {LORA_REPO}...")
    lora_path = hf_hub_download(LORA_REPO, LORA_FILE, token=hf_token)

    print(f"[lustly_model] Loading LoRA weights (strength={lora_strength})...")
    pipe.load_lora_weights(lora_path, adapter_name=LORA_ADAPTER_NAME)
    pipe.set_adapters([LORA_ADAPTER_NAME], adapter_weights=[lora_strength])

    if mode == "tight":
        print("[lustly_model] Enabling sequential CPU offload (peak VRAM ~20 GB)...")
        pipe.enable_sequential_cpu_offload()
        if hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
        if hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
    else:
        print("[lustly_model] Moving pipeline to CUDA (fast path)...")
        pipe.to("cuda")

    return pipe
