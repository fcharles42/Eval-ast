import json
import csv
import torch
import os

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from safetensors.torch import load_file
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

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

    # ---------------- FULL FINETUNED MODEL ----------------

    if cfg["ft_percent"] == "100%":

        print("[INFO] Loading full finetuned model")

        model = AutoModelForCausalLM.from_pretrained(
            cfg["checkpoint_path"],
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    # ---------------- PEFT / LORA MODEL ----------------

    else:

        print("[INFO] Loading PEFT adapter")

        base = AutoModelForCausalLM.from_pretrained(
            cfg["base_model_name"],
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        adapter_path = os.path.join(
            cfg["checkpoint_path"],
            "adapter_model.safetensors"
        )

        state_dict = load_file(adapter_path)

        target_vocab_size = (
            base.get_input_embeddings().weight.shape[0]
        )

        for k in state_dict:

            if "lm_head.base_layer.weight" in k:
                target_vocab_size = state_dict[k].shape[0]
                break

            if "embed_tokens.weight" in k:
                target_vocab_size = state_dict[k].shape[0]
                break

        current_vocab = (
            base.get_input_embeddings().weight.shape[0]
        )

        if current_vocab != target_vocab_size:

            print(
                f"[INFO] Resizing embeddings "
                f"{current_vocab} → {target_vocab_size}"
            )

            base.resize_token_embeddings(
                target_vocab_size
            )

        base.config.tie_word_embeddings = False

        model = PeftModel.from_pretrained(
            base,
            cfg["checkpoint_path"],
            ignore_mismatched_sizes=True
        )

        # Remove incompatible lm_head LoRA cleanly
        if hasattr(model.base_model.model.lm_head, "base_layer"):

            model.base_model.model.lm_head = (
                model.base_model.model.lm_head.base_layer
            )

    # ---------------- TOKENIZER ----------------

    try:

        tokenizer = AutoTokenizer.from_pretrained(
            cfg["checkpoint_path"],
            trust_remote_code=True
        )

    except:

        tokenizer = AutoTokenizer.from_pretrained(
            cfg["base_model_name"],
            trust_remote_code=True
        )

    tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    return model, tokenizer


# ---------------- GENERATION ----------------

def generate(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    eos_token_id = tokenizer.eos_token_id

    if "<ast_end>" in tokenizer.get_vocab():
        eos_token_id = tokenizer.convert_tokens_to_ids("<ast_end>")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
            temperature=0.0,
            repetition_penalty=1.05,
            use_cache=True,
            eos_token_id=eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(
        out[0][input_len:],
        skip_special_tokens=False
    )

    if "<ast_end>" in generated:
        generated = generated.split("<ast_end>")[0] + "<ast_end>"

    return generated.strip()


# ---------------- METRICS ----------------

def is_valid_json_continuation(prompt, pred):
    """Check if prompt + pred forms valid JSON."""
    try:
        json.loads(prompt + pred)
        return True
    except Exception:
        return False


def compute_json_valid_ratio(prompts, preds):
    """How often does prompt+pred reconstruct valid JSON?"""
    valid = sum(
        is_valid_json_continuation(p, r)
        for p, r in zip(prompts, preds)
    )
    return valid / len(preds)


def compute_exact_prefix_match(preds, refs, n=50):
    """Fraction of predictions whose first n chars match the reference."""
    matches = sum(
        p[:n] == r[:n]
        for p, r in zip(preds, refs)
    )
    return matches / len(preds)


def compute_codebleu(preds, refs):
    """Compute CodeBLEU and return the aggregate score."""
    result = calc_codebleu(
        refs,
        preds,
        lang="python"
    )

    if isinstance(result, dict):
        return result.get("codebleu", 0.0)

    return float(result)


# ---------------- TREEBLEU ----------------

def extract_json_ast_node_types(obj):
    nodes = []

    if isinstance(obj, dict):

        if "_type" in obj:
            nodes.append(obj["_type"])

        for value in obj.values():
            nodes.extend(
                extract_json_ast_node_types(value)
            )

    elif isinstance(obj, list):

        for item in obj:
            nodes.extend(
                extract_json_ast_node_types(item)
            )

    return nodes


def ast_json_to_node_tokens(text):

    try:
        parsed = json.loads(text)
        return extract_json_ast_node_types(parsed)

    except Exception:
        return []


def compute_treebleu(prompts, preds, refs):

    pred_tokens = []
    ref_tokens = []

    for p, pred, ref in zip(prompts, preds, refs):

        pred_full = p + pred
        ref_full = p + ref

        pred_tok = ast_json_to_node_tokens(pred_full)
        ref_tok = ast_json_to_node_tokens(ref_full)

        pred_tokens.append(pred_tok)
        ref_tokens.append([ref_tok])

    smoothie = SmoothingFunction().method1

    return corpus_bleu(
        ref_tokens,
        pred_tokens,
        smoothing_function=smoothie
    )


# ---------------- EVALUATION ----------------
def evaluate(cfg):
    data = load_dataset_file(cfg["eval_dataset_path"])

    model, tokenizer = load_model(cfg)

    preds  = []
    refs   = []
    prompts = []

    predictions_path = "/content/Eval-ast/predictions.jsonl"

    with open(predictions_path, "w") as pred_file:
        for ex in tqdm(data):
            pred = generate(model, tokenizer, ex["prompt"])

            preds.append(pred)
            refs.append(ex["reference"])
            prompts.append(ex["prompt"])

            pred_file.write(json.dumps({
                "prompt":     ex["prompt"],
                "prediction": pred,
                "reference":  ex["reference"],
            }) + "\n")
            pred_file.flush()

    print(f"[OK] Saved predictions → {predictions_path}")

    return {
        # Replaces valid_ratio — checks if prompt+pred is valid JSON
        "json_valid_ratio":   compute_json_valid_ratio(prompts, preds),
        # Exact character match on first 50 chars of the completion
        "prefix_match_50":    compute_exact_prefix_match(preds, refs, n=50),
        # Token-level structural similarity via AST node names
        "treebleu":           compute_treebleu(prompts, preds, refs),
        # String-level n-gram similarity
        "codebleu":           compute_codebleu(preds, refs),
    }


# ---------------- LOGGING ----------------

def log_results(cfg, metrics):
    row = {
        "model":            cfg["model_type"],
        "ft_percent":       cfg["ft_percent"],
        "checkpoint":       cfg["checkpoint_label"],
        "json_valid_ratio": metrics["json_valid_ratio"],
        "prefix_match_50":  metrics["prefix_match_50"],
        "codebleu":         metrics["codebleu"],
        "treebleu":         metrics["treebleu"],
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
