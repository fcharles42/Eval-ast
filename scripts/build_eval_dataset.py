import json
import random
from datasets import load_dataset

OUT_PATH = "eval_dataset.jsonl"
NUM_SAMPLES = 150
MAX_CHARS = 2000

def extract_prompt(code):
    lines = code.strip().split("\n")
    return "\n".join(lines[:3])  # simple partial prompt

def main():
    dataset = load_dataset("code_search_net", "python", split="train")

    samples = []
    for ex in dataset:
        code = ex.get("code", "")
        if not code or len(code) > MAX_CHARS:
            continue

        prompt = extract_prompt(code)
        if len(prompt.strip()) < 5:
            continue

        samples.append({
            "prompt": prompt,
            "reference": code
        })

        if len(samples) >= NUM_SAMPLES:
            break

    with open(OUT_PATH, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"[OK] Saved {len(samples)} samples → {OUT_PATH}")

if __name__ == "__main__":
    main()