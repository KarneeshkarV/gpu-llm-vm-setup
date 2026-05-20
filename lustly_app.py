"""lustly_app.py — Gradio UI for FLUX.1-dev + Lustly.ai uncensored LoRA.

Launch via `bash lustly_ui.sh`. Binds to 0.0.0.0:7860 so the EC2 public IP
serves the UI directly.
"""

from __future__ import annotations

import time

import gradio as gr
import torch

from lustly_model import load_pipeline

print("Loading FLUX.1-dev + Lustly LoRA (first run downloads ~24 GB)...")
PIPE = load_pipeline()
print("Pipeline ready.")


def generate(
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance: float,
    width: int,
    height: int,
    seed: int,
    lora_strength: float,
):
    if not prompt or not prompt.strip():
        raise gr.Error("Enter a prompt.")

    # Allow live LoRA strength changes without reloading the model.
    PIPE.set_adapters(["lustly"], adapter_weights=[float(lora_strength)])

    generator = None
    if seed and seed > 0:
        generator = torch.Generator(device="cuda").manual_seed(int(seed))

    t0 = time.time()
    kwargs = dict(
        prompt=prompt,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        width=int(width),
        height=int(height),
        generator=generator,
        max_sequence_length=512,
    )
    if negative_prompt and negative_prompt.strip():
        kwargs["negative_prompt"] = negative_prompt

    image = PIPE(**kwargs).images[0]
    elapsed = time.time() - t0
    info = (
        f"Generated in {elapsed:.1f}s — {width}x{height}, {steps} steps, "
        f"cfg={guidance}, LoRA={lora_strength}"
    )
    return image, info


with gr.Blocks(title="FLUX.1-dev + Lustly Uncensored LoRA", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "## FLUX.1-dev + Lustly.ai Uncensored LoRA\n"
        "Base: `black-forest-labs/FLUX.1-dev` (gated, accepted) — "
        "LoRA: `momofuhshoh/Flux_Lustly.ai_Uncensored_nsfw_v1`.\n"
        "Running on an L40S — no offload, full speed (~6-15s/image)."
    )
    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="cinematic film still of a woman with red hair, 35mm, golden hour",
                lines=3,
            )
            negative = gr.Textbox(
                label="Negative prompt (optional)",
                placeholder="blurry, low quality, watermark",
                lines=2,
            )
            with gr.Row():
                steps = gr.Slider(label="Steps", minimum=4, maximum=50, step=1, value=24)
                guidance = gr.Slider(label="Guidance (CFG)", minimum=0.0, maximum=15.0, step=0.1, value=3.5)
            with gr.Row():
                width = gr.Slider(label="Width",  minimum=512, maximum=1280, step=64, value=1024)
                height = gr.Slider(label="Height", minimum=512, maximum=1280, step=64, value=1024)
            with gr.Row():
                seed = gr.Number(label="Seed (0 = random)", value=0, precision=0)
                lora_str = gr.Slider(label="LoRA strength", minimum=0.0, maximum=1.5, step=0.05, value=0.85)
            run = gr.Button("Generate", variant="primary")
        with gr.Column(scale=3):
            out_image = gr.Image(label="Output", type="pil")
            out_info = gr.Markdown()

    run.click(
        generate,
        inputs=[prompt, negative, steps, guidance, width, height, seed, lora_str],
        outputs=[out_image, out_info],
    )


if __name__ == "__main__":
    demo.queue(max_size=8).launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
