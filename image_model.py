"""image_model.py — load FLUX.2 Klein 9B with the ponpoke uncensored Qwen3
text encoder swapped in.

Two paths, auto-selected from the available VRAM at load time:

  * "tight"  (default on L4 22.5 GB):
      - text encoder in 4-bit nf4 (bitsandbytes), bf16 compute
      - pipeline.enable_sequential_cpu_offload()
      - VAE slicing + tiling
      - Peak VRAM ~20 GB. Slow (~30-90 s/image).

  * "fast"   (default on L40S 48 GB and up):
      - text encoder in bf16 fully on GPU
      - whole pipe on GPU, no offload
      - Peak VRAM ~30-35 GB. Full speed.

Override with IMAGE_MODE=fast | tight in the environment.

If diffusers gains an official FluxPipeline2 / FLUX.2 class, this loader
picks it up via DiffusionPipeline.from_pretrained() auto-detection.
"""

from __future__ import annotations

import os

import torch
from diffusers import DiffusionPipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

BASE_REPO = "black-forest-labs/FLUX.2-klein-9B"
TEXT_ENCODER_REPO = "ponpoke/flux2-klein-9b-uncensored-text-encoder"

DTYPE = torch.bfloat16
FAST_PATH_VRAM_GB = 30.0  # if free VRAM >= this on device 0, use the fast path


def _choose_mode() -> str:
    forced = os.environ.get("IMAGE_MODE", "").strip().lower()
    if forced in {"fast", "tight"}:
        return forced
    if not torch.cuda.is_available():
        return "tight"
    free, _total = torch.cuda.mem_get_info(0)
    return "fast" if (free / 1e9) >= FAST_PATH_VRAM_GB else "tight"


def _bnb_4bit_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=DTYPE,
        bnb_4bit_use_double_quant=True,
    )


def load_pipeline():
    """Load FLUX.2 Klein with the uncensored Qwen3 text encoder.

    Returns the diffusers pipeline, ready to call with `pipe(prompt=...)`.
    """
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    mode = _choose_mode()
    print(f"[image_model] Selected mode: {mode}")

    print(f"[image_model] Loading text encoder from {TEXT_ENCODER_REPO}...")
    tokenizer = AutoTokenizer.from_pretrained(
        TEXT_ENCODER_REPO,
        token=hf_token,
        trust_remote_code=True,
    )

    if mode == "tight":
        text_encoder = AutoModelForCausalLM.from_pretrained(
            TEXT_ENCODER_REPO,
            token=hf_token,
            trust_remote_code=True,
            quantization_config=_bnb_4bit_config(),
            torch_dtype=DTYPE,
            device_map="auto",
        )
    else:
        text_encoder = AutoModelForCausalLM.from_pretrained(
            TEXT_ENCODER_REPO,
            token=hf_token,
            trust_remote_code=True,
            torch_dtype=DTYPE,
        )

    print(f"[image_model] Loading base pipeline from {BASE_REPO} (bf16)...")
    pipe = DiffusionPipeline.from_pretrained(
        BASE_REPO,
        torch_dtype=DTYPE,
        token=hf_token,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    )

    if mode == "tight":
        print("[image_model] Enabling sequential CPU offload (peak VRAM ~20 GB)...")
        pipe.enable_sequential_cpu_offload()
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
    else:
        print("[image_model] Moving full pipeline to CUDA (fast path)...")
        pipe.to("cuda")

    return pipe
