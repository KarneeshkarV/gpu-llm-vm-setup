# gpu-llm-vm-setup

One-command bootstrap to run a local GGUF LLM **fully on the GPU** on a fresh
cloud GPU VM. Built from a real debugging session — every non-obvious choice
here exists because the obvious one failed.

## Hardware assumptions

- NVIDIA GPU with ~22.5 GB VRAM (tested on an **L4** / AWS `g6.xlarge`)
- Only **16 GiB system RAM** → CPU inference OOM-kills, GPU offload is mandatory
- AWS **"Deep Learning OSS Nvidia Driver AMI"**: NVIDIA driver only, **no CUDA
  toolkit** (`nvcc` absent), Ubuntu, user `ubuntu`

## Usage

On a fresh VM, clone/copy this repo and run:

```bash
bash setup.sh        # installs uv, venv, llama-cpp-python (CUDA), CUDA libs
bash run.sh          # one-shot test: "What is the capital of France?"
bash chat.sh         # interactive streaming chat REPL  (/reset, /exit)
```

Run `run.sh` / `chat.sh` in a **real terminal/TTY in the foreground**.

The model (~16 GB) downloads **once** from Hugging Face into
`~/.cache/huggingface/hub` and is reused on every subsequent run.

## Files

| File | Purpose |
|---|---|
| `setup.sh` | idempotent bootstrap (safe to re-run) |
| `_env.sh` | sourced helper: activates venv + sets `LD_LIBRARY_PATH` |
| `_model.py` | shared model loader (single source of truth) |
| `model_infer.py` | one-shot test inference |
| `chatbot.py` | interactive streaming chat REPL |
| `run.sh` / `chat.sh` | foreground launchers |
| `image_setup.sh` | additive bootstrap for FLUX.2 Klein image generation |
| `image_model.py` | loads FLUX.2 Klein + ponpoke uncensored Qwen3 text encoder |
| `image_app.py` | Gradio UI for image generation |
| `image_ui.sh` | foreground launcher for the Gradio UI |

To use a different model, edit `MODEL_REPO` / `MODEL_FILE` in `_model.py`.

## Image-generation mode (FLUX.2 Klein 9B, uncensored text encoder)

On top of the chat path above, this branch adds a Gradio UI that runs
`black-forest-labs/FLUX.2-klein-9B` with
`ponpoke/flux2-klein-9b-uncensored-text-encoder` swapped in as the text
encoder. The text encoder is loaded in 4-bit (nf4) and the whole pipeline
uses sequential CPU offload to fit on a 22.5 GB L4.

```bash
bash setup.sh                          # base bootstrap (uv, venv, llama-cpp)
bash image_setup.sh                    # adds torch cu124, diffusers, gradio, bnb
huggingface-cli login                  # FLUX.2 Klein base is gated
bash image_ui.sh                       # Gradio UI at http://<vm-ip>:7860
```

First run downloads ~35 GB into `~/.cache/huggingface/hub`. Generation is
~30-90 s/image on an L4 because of sequential offload; a bigger GPU
(A10G 24 GB, L40S 48 GB) skips offload and is much faster.

## Why these specific choices (the gotchas)

1. **No source build.** The driver-only AMI has no `nvcc`; building
   `llama-cpp-python` from source fails with "CUDA Toolkit not found". We
   install the **prebuilt CUDA wheel** from
   `https://abetlen.github.io/llama-cpp-python/whl/cu124`.
2. **`--index-strategy unsafe-best-match` is required.** Without it `uv`
   ignores the extra wheel index and falls back to the broken PyPI source
   build. `cu124` wheels are backward-compatible with newer CUDA drivers.
3. **CUDA runtime libs + `LD_LIBRARY_PATH`.** The prebuilt wheel needs
   `libcudart.so.12` / `libcublas`, which the driver-only AMI lacks. We pip
   install `nvidia-cuda-runtime-cu12` / `nvidia-cublas-cu12` and point
   `LD_LIBRARY_PATH` at the venv's nvidia lib dirs. The dirs are located via
   `sysconfig` purelib — **not** `nvidia.__file__`, which is `None` because
   `nvidia` is a namespace package.
4. **GPU offload is mandatory.** `n_gpu_layers=-1`. The 26B Q4_K_M model
   (~16 GB) loads into a CUDA buffer on the GPU; on 16 GiB RAM, CPU inference
   gets OOM-killed.
5. **`hf_hub_download`, not `Llama.from_pretrained`.** `from_pretrained` uses
   huggingface_hub `local_dir` mode and re-fetches the ~16 GB model on every
   run. We resolve the path via `hf_hub_download` (proper hub cache) and load
   with `Llama(model_path=...)`.
