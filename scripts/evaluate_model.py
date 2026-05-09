import json
import csv
import ast
import torch
import os

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from safetensors.torch import load_file

from codebleu import calc_codebleu


# ---------------- CONFIG ----------------

def load_config(path):
    with open(path) as f:
        return json.load(f)


# ---------------- DATA ----------------

def load_dataset_file(path):
    data = []

    with open(path) as f:
        for line in f:
            data.append(json.loads(line))

    return data


# ---------------- MODEL ----------------

def load_model(cfg):
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model_name"],
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load adapter weights
    adapter_path = os.path.join(
        cfg["checkpoint_path"],
        "adapter_model.safetensors"
    )

    state_dict = load_file(adapter_path)

    # Detect vocab size if embeddings exist
    target_vocab_size = base.get_input_embeddings().weight.shape[0]

    for k in state_dict:
        if "embed_tokens.weight" in k:
            target_vocab_size = state_dict[k].shape[0]
            break

    current_vocab = base.get_input_embeddings().weight.shape[0]

    if current_vocab != target_vocab_size:
        print(f"[INFO] Resizing embeddings {current_vocab} → {target_vocab_size}")
        base.resize_token_embeddings(target_vocab_size)

    model = PeftModel.from_pretrained(
        base,
        cfg["checkpoint_path"]
    )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["base_model_name"],
        trust_remote_code=True
    )

    tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    return model, tokenizer


# ---------------- GENERATION ----------------

def generate(model, tokenizer, prompt):

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():

        out = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(
        out[0],
        skip_special_tokens=True
    )


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
    result = calc_codebleu(
        refs,
        preds,
        lang="python"
    )

    return result["codebleu"]


# ---------------- TREEBLEU ----------------

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


# ---------------- EVALUATION ----------------
def evaluate(cfg):
    data = load_dataset_file(cfg["eval_dataset_path"])

    # DEBUG MODE
    data = data[:5]

    model, tokenizer = load_model(cfg)

    preds = []
    refs = []

    predictions_path = "/content/Eval-ast/predictions.jsonl"

    with open(predictions_path, "w") as pred_file:

        for ex in tqdm(data):

            pred = generate(
                model,
                tokenizer,
                ex["prompt"]
            )

            preds.append(pred)
            refs.append(ex["reference"])

            # CACHE EACH PREDICTION IMMEDIATELY
            pred_file.write(json.dumps({
                "prompt": ex["prompt"],
                "prediction": pred,
                "reference": ex["reference"]
            }) + "\n")

            pred_file.flush()

    print(f"[OK] Saved predictions → {predictions_path}")

    valid = compute_valid_ratio(preds)

    codebleu = compute_codebleu(
        preds,
        refs
    )

    treebleu = compute_treebleu(
        preds,
        refs
    )

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
        "checkpoint": cfg["checkpoint_label"],
        "valid_ratio": metrics["valid_ratio"],
        "codebleu": metrics["codebleu"],
        "treebleu": metrics["treebleu"],
    }

    file_exists = os.path.exists(cfg["output_path"])

    with open(cfg["output_path"], "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=row.keys()
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

    print("[OK] Logged results")


# ---------------- ENTRY ----------------

if __name__ == "__main__":
    cfg = load_config(
        "/content/Eval-ast/config/config.json"
    )

    metrics = evaluate(cfg)

    print(metrics)

    log_results(cfg, metrics)