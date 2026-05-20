"""image_model.py — load FLUX.2 Klein 9B with the ponpoke uncensored Qwen3
text encoder swapped in, sized to fit on a 22.5 GB L4.

Strategy:
  * Base pipeline:  black-forest-labs/FLUX.2-klein-9B   (gated — accept once)
  * Text encoder:   ponpoke/flux2-klein-9b-uncensored-text-encoder
                    loaded from safetensors in 4-bit via bitsandbytes
                    (8 GB fp16 -> ~5 GB in 4-bit; otherwise we OOM with the
                    9B diffusion transformer also on the GPU).
  * Diffusion model: kept in bfloat16 on the GPU
  * VAE: kept in bfloat16 on the GPU
  * pipeline.enable_sequential_cpu_offload() — offloads each submodule to
    CPU between calls so peak VRAM stays under ~20 GB on the L4.

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

    print(f"[image_model] Loading text encoder from {TEXT_ENCODER_REPO} (4-bit nf4)...")
    tokenizer = AutoTokenizer.from_pretrained(
        TEXT_ENCODER_REPO,
        token=hf_token,
        trust_remote_code=True,
    )
    text_encoder = AutoModelForCausalLM.from_pretrained(
        TEXT_ENCODER_REPO,
        token=hf_token,
        trust_remote_code=True,
        quantization_config=_bnb_4bit_config(),
        torch_dtype=DTYPE,
        device_map="auto",
    )

    print(f"[image_model] Loading base pipeline from {BASE_REPO} (bf16)...")
    pipe = DiffusionPipeline.from_pretrained(
        BASE_REPO,
        torch_dtype=DTYPE,
        token=hf_token,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    )

    print("[image_model] Enabling sequential CPU offload (peak VRAM ~20 GB)...")
    pipe.enable_sequential_cpu_offload()

    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()

    return pipe
