"""Shared model loader.

Both model_infer.py and chatbot.py import load_llm() from here so the
load logic exists in exactly one place.

IMPORTANT: we resolve the model via huggingface_hub.hf_hub_download and then
load it with Llama(model_path=...). We deliberately do NOT use
Llama.from_pretrained(): that path uses huggingface_hub `local_dir` mode
(emits a `local_dir_use_symlinks` deprecation warning) and re-fetches the
~16 GB model on every run instead of reusing the hub cache. hf_hub_download
uses ~/.cache/huggingface/hub and returns instantly when already cached.
"""

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# --- Model selection (swap these to use a different GGUF) -----------------
MODEL_REPO = "Jiunsong/supergemma4-26b-uncensored-gguf-v2"
MODEL_FILE = "supergemma4-26b-uncensored-fast-v2-Q4_K_M.gguf"

# --- Inference settings ---------------------------------------------------
# n_gpu_layers=-1 offloads ALL layers to the GPU. This is MANDATORY: the
# target VM has only 16 GiB system RAM, so CPU inference of this 26B model
# OOM-kills. All 31/31 layers fit in a ~16 GB CUDA buffer on the 22.5 GB L4.
N_GPU_LAYERS = -1
N_CTX = 4096


def load_llm(n_ctx: int = N_CTX, verbose: bool = False) -> Llama:
    """Resolve the model from the HF hub cache and load it fully on the GPU."""
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    return Llama(
        model_path=model_path,
        n_gpu_layers=N_GPU_LAYERS,
        n_ctx=n_ctx,
        verbose=verbose,
    )
