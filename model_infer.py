#!/usr/bin/env python3
"""model_infer.py — load the model fully on the GPU and run one test
chat completion. Run via `bash run.sh` (which sets the CUDA env first).
"""

from _model import MODEL_FILE, MODEL_REPO, load_llm


def main() -> None:
    print(f"Loading {MODEL_REPO} :: {MODEL_FILE}")
    print("(first run downloads ~16 GB; subsequent runs use the HF cache)")
    llm = load_llm()

    out = llm.create_chat_completion(
        messages=[
            {"role": "user", "content": "What is the capital of France?"},
        ]
    )

    print("\n--- Model response -------------------------------------------")
    print(out["choices"][0]["message"]["content"])
    print("--------------------------------------------------------------")


if __name__ == "__main__":
    main()
