import json
import ast
import random
from datasets import load_dataset
from transformers import AutoTokenizer

OUT_PATH = "/content/Eval-ast/eval_dataset.jsonl"

NUM_SAMPLES = 150
MAX_CHARS = 16000

USE_BOUNDARY_TOKENS = False
AST_START = "<ast_start>"
AST_END   = "<ast_end>"

TOKENIZER_NAME = "Qwen/Qwen2.5-0.5B"
PROMPT_RATIO = 0.2


def ast_to_json(node):
    """Must exactly match the function used during training."""
    if isinstance(node, ast.AST):
        result = {"_type": type(node).__name__}
        for field, value in ast.iter_fields(node):
            result[field] = ast_to_json(value)
        return result
    elif isinstance(node, list):
        return [ast_to_json(item) for item in node]
    elif isinstance(node, (str, int, float, bool, type(None))):
        return node
    else:
        return str(node)


def extract_function_modules(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield ast.Module(body=[node], type_ignores=[])


def make_ast_dump(code):
    """
    Parse code → extract first function → serialize to JSON string.
    Returns None if anything fails.
    """
    try:
        mod = ast.parse(code)
    except Exception:
        return None

    for fn_mod in extract_function_modules(mod):
        try:
            ast_json = ast_to_json(fn_mod)
            dump_str = json.dumps(ast_json, ensure_ascii=False)
            return dump_str
        except Exception:
            return None

    return None


def main():
    dataset = load_dataset("code_search_net", "python", split="train")
    dataset = list(dataset)
    random.shuffle(dataset)
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME,
        trust_remote_code=True
    )

    samples = []

    for ex in dataset:
        code = ex.get("func_code_string", "")

        if not code or len(code) > MAX_CHARS:
            continue

        dump_str = make_ast_dump(code)

        if dump_str is None:
            continue

        if len(dump_str) > MAX_CHARS:
            continue

        # Wrap with boundary tokens if evaluating that model variant
        if USE_BOUNDARY_TOKENS:
            full_sequence = f"{AST_START}{dump_str}{AST_END}"
        else:
            full_sequence = dump_str

        tokens = tokenizer.encode(
            full_sequence,
            add_special_tokens=False
        )

        if len(tokens) < 50:
            continue

        prompt_len = int(len(tokens) * PROMPT_RATIO)
        prompt_ids = tokens[:prompt_len]
        reference_ids = tokens[prompt_len:]

        prompt = tokenizer.decode(prompt_ids)
        reference = tokenizer.decode(reference_ids)

        if not prompt.strip() or not reference.strip():
            continue

        samples.append({
            "prompt": prompt,
            "reference": reference,
            # store original code for debugging
            "source_code": code,
        })

        if len(samples) >= NUM_SAMPLES:
            break

    with open(OUT_PATH, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"[OK] Saved {len(samples)} samples → {OUT_PATH}")


if __name__ == "__main__":
    main()
