import json
import csv
import ast
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Optional: install beforehand → pip install codebleu
from codebleu import calc_codebleu


# ---------------- CONFIG LOADER ----------------

def load_config(path):
    with open(path) as f:
        return json.load(f)


# ---------------- DATA ----------------

def load_dataset(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


# ---------------- MODEL ----------------

def load_model(cfg):
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model_name"],
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    if cfg.get("checkpoint_path"):
        model = PeftModel.from_pretrained(base, cfg["checkpoint_path"])
    else:
        model = base

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model_name"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


# ---------------- GENERATION ----------------

def generate(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
        )

    return tokenizer.decode(out[0], skip_special_tokens=True)


# ---------------- METRICS ----------------

def is_valid_python(code):
    try:
        ast.parse(code)
        return True
    except:
        return False


def compute_valid_ratio(preds):
    valid = sum(is_valid_python(p) for p in preds)
    return valid / len(preds)


def compute_codebleu(preds, refs):
    result = calc_codebleu(refs, preds, lang="python")
    return result["codebleu"]


# --- Simple TreeBLEU approximation (AST → tokens → BLEU) ---

def ast_to_tokens(code):
    try:
        tree = ast.parse(code)
        return [type(node).__name__ for node in ast.walk(tree)]
    except:
        return []


def compute_treebleu(preds, refs):
    from nltk.translate.bleu_score import corpus_bleu

    pred_tokens = [ast_to_tokens(p) for p in preds]
    ref_tokens = [[ast_to_tokens(r)] for r in refs]

    return corpus_bleu(ref_tokens, pred_tokens)


# ---------------- MAIN EVAL ----------------

def evaluate(cfg):
    data = load_dataset(cfg["eval_dataset_path"])
    model, tokenizer = load_model(cfg)

    preds = []
    refs = []

    for ex in tqdm(data):
        pred = generate(model, tokenizer, ex["prompt"])
        preds.append(pred)
        refs.append(ex["reference"])

    valid = compute_valid_ratio(preds)
    codebleu = compute_codebleu(preds, refs)
    treebleu = compute_treebleu(preds, refs)

    return {
        "valid_ratio": valid,
        "codebleu": codebleu,
        "treebleu": treebleu
    }


# ---------------- LOGGING ----------------

def log_results(cfg, metrics):
    row = {
        "model": cfg["model_type"],
        "ft_percent": cfg["ft_percent"],
        "checkpoint": cfg["ft_percent"],
        "valid_ratio": metrics["valid_ratio"],
        "codebleu": metrics["codebleu"],
        "treebleu": metrics["treebleu"],
    }

    file_exists = False
    try:
        with open(cfg["output_path"], "r"):
            file_exists = True
    except:
        pass

    with open(cfg["output_path"], "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

    print("[OK] Logged results")


# ---------------- ENTRY ----------------

if __name__ == "__main__":
    import sys

    cfg_path = sys.argv[1]
    cfg = load_config(cfg_path)

    for model_cfg in cfg["models"]:
        full_cfg = {**cfg, **model_cfg}
        metrics = evaluate(full_cfg)
        print(f"Metrics for {model_cfg['model_type']} {model_cfg['ft_percent']}: {metrics}")
        log_results(full_cfg, metrics)