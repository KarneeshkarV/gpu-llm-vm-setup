#!/usr/bin/env python3
"""chatbot.py — interactive streaming chat REPL on the GPU-loaded model.

Run via `bash chat.sh` (sets the CUDA env first) in a REAL terminal/TTY.
Running it backgrounded via nohup through chained SSH gets SIGKILLed during
the VRAM weight-copy, so keep it foreground.

Commands:  /reset  clear conversation history   |   /exit  quit
"""

import sys

from _model import load_llm

SYSTEM_PROMPT = "You are a helpful, concise assistant running locally on a GPU."


def new_history():
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def main() -> None:
    print("Loading model onto GPU (~1-2 min the first time)...")
    llm = load_llm()
    print("Model ready.  (/reset to clear history, /exit to quit)\n")

    history = new_history()
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            break

        if not user:
            continue
        if user == "/exit":
            print("bye.")
            break
        if user == "/reset":
            history = new_history()
            print("[conversation reset]\n")
            continue

        history.append({"role": "user", "content": user})

        print("bot> ", end="", flush=True)
        assistant_text = ""
        for chunk in llm.create_chat_completion(
            messages=history, stream=True, temperature=0.7, max_tokens=512
        ):
            piece = chunk["choices"][0]["delta"].get("content")
            if piece:
                assistant_text += piece
                sys.stdout.write(piece)
                sys.stdout.flush()
        print("\n")

        history.append({"role": "assistant", "content": assistant_text})


if __name__ == "__main__":
    main()
