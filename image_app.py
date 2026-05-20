"""image_app.py — Gradio UI for FLUX.2 Klein 9B + ponpoke uncensored Qwen3
text encoder, running on a single L4 (22.5 GB VRAM).

Launch via `bash image_ui.sh` from a foreground TTY. The UI binds to
0.0.0.0:7860 so the EC2 instance's public IP serves it directly.
"""

from __future__ import annotations

import time

import gradio as gr
import torch

from image_model import load_pipeline

print("Loading FLUX.2 Klein pipeline (first run downloads ~35 GB)...")
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
):
    if not prompt or not prompt.strip():
        raise gr.Error("Enter a prompt.")

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
    )
    if negative_prompt and negative_prompt.strip():
        kwargs["negative_prompt"] = negative_prompt

    result = PIPE(**kwargs)
    image = result.images[0]
    elapsed = time.time() - t0
    info = f"Generated in {elapsed:.1f}s — {width}x{height}, {steps} steps, cfg={guidance}"
    return image, info


with gr.Blocks(title="FLUX.2 Klein — Uncensored", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "## FLUX.2 Klein 9B — image generation\n"
        "Base: `black-forest-labs/FLUX.2-klein-9B` (gated) — "
        "Text encoder: `ponpoke/flux2-klein-9b-uncensored-text-encoder` (4-bit nf4).\n"
        "Running on an L4 with sequential CPU offload — expect ~30-90s / image."
    )
    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="A cinematic portrait of a cyberpunk samurai in neon rain, 35mm film",
                lines=3,
            )
            negative = gr.Textbox(
                label="Negative prompt (optional)",
                placeholder="blurry, low quality, watermark",
                lines=2,
            )
            with gr.Row():
                steps = gr.Slider(label="Steps", minimum=4, maximum=50, step=1, value=20)
                guidance = gr.Slider(label="Guidance (CFG)", minimum=0.0, maximum=15.0, step=0.1, value=3.5)
            with gr.Row():
                width = gr.Slider(label="Width",  minimum=512, maximum=1280, step=64, value=1024)
                height = gr.Slider(label="Height", minimum=512, maximum=1280, step=64, value=1024)
            seed = gr.Number(label="Seed (0 = random)", value=0, precision=0)
            run = gr.Button("Generate", variant="primary")
        with gr.Column(scale=3):
            out_image = gr.Image(label="Output", type="pil")
            out_info = gr.Markdown()

    run.click(
        generate,
        inputs=[prompt, negative, steps, guidance, width, height, seed],
        outputs=[out_image, out_info],
    )


if __name__ == "__main__":
    demo.queue(max_size=8).launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
