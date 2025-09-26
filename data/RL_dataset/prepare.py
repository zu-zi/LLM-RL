# prepare.py
import os
import re
import json
import math
import hashlib
from collections import Counter
import torch
from datasets import load_dataset
import tiktoken

DATASET_NAME = "openbmb/UltraFeedback"
FIXED_SIZE = 1024
EVAL_SIZE  = 16
GLOBAL_SEED = 1337

MAX_PROMPT_TOKENS = 152
MIN_PROMPT_TOKENS = 32
ALLOW_EMPTY_ASSISTANT = True

SAVE_DIR = os.path.dirname(__file__)
os.makedirs(SAVE_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(SAVE_DIR, "prompt.bin")

enc = tiktoken.get_encoding("gpt2")
EOS_ID = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"} )[0]

def _stable_hash(s: str, seed: int) -> int:
    h = hashlib.sha1((str(seed) + "§" + s).encode("utf-8")).hexdigest()
    return int(h, 16)

def _likely_english(text: str) -> bool:
    if not text: return False
    total = len(text)
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    if (cjk / max(total, 1)) >= 0.02: return False
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    return (non_ascii / max(total, 1)) < 0.25

def _norm_text(s: str) -> str:
    return (s or "").replace("\r\n", "\n").replace("\r", "\n").strip()

def _ensure_human_lead(text: str) -> str:
    t = text.lstrip()
    return t if t.startswith("Human:") else ("Human: " + t)

def _cut_to_prompt_prefix(full_dialogue: str) -> str:
    s = _norm_text(full_dialogue)
    if not s: return ""
    m = re.search(r"(?:\n\n)?Assistant:", s)
    if m:
        prefix = _ensure_human_lead(s[:m.start()])
        prompt = (prefix.rstrip() + "\n\nAssistant:")
        return re.sub(r"(?:\n+)?Assistant:$", "Assistant:", prompt)
    if ALLOW_EMPTY_ASSISTANT:
        head = _ensure_human_lead(s)
        return head.rstrip() + "\n\nAssistant:"
    return ""

def _encode_prompt(prompt_text: str):
    return enc.encode(prompt_text, allowed_special="all")

def _choose_source_column(ds):
    cols = ds.column_names
    if "instruction" in cols: return "instruction"
    if "chosen" in cols: return "chosen"
    if "prompt" in cols: return "prompt"
    if "text" in cols: return "text"
    raise RuntimeError(
        f"Dataset {DATASET_NAME} must have one of: instruction/chosen/prompt/text. Available: {cols}"
    )

def _row_to_prompt(row: dict, col: str) -> str:
    val = row.get(col, None)
    if not isinstance(val, str): return ""
    if col == "instruction":
        instr = _norm_text(val)
        return f"Human: {instr}\n\nAssistant:" if instr else ""
    return _cut_to_prompt_prefix(val)

def collect_fixed_prompts(ds, fixed_size=FIXED_SIZE, seed=GLOBAL_SEED):
    pool, seen_md5, lengths = [], set(), []
    col = _choose_source_column(ds)
    print(f"[prepare] using column: '{col}' from dataset (available: {ds.column_names})")

    for row in ds:
        prompt = _row_to_prompt(row, col)
        if not prompt or not _likely_english(prompt): continue
        ids = _encode_prompt(prompt)
        if len(ids) < MIN_PROMPT_TOKENS: continue
        ids = ids[:MAX_PROMPT_TOKENS]
        if not ids: continue
        k = hashlib.md5(prompt.encode("utf-8")).hexdigest()
        if k in seen_md5: continue
        seen_md5.add(k)
        lengths.append(len(ids))
        pool.append((_stable_hash(prompt, seed), prompt, ids))

    if not pool:
        raise RuntimeError("No valid prompts. Check column mapping or filters.")

    pool.sort(key=lambda x: x[0])
    pool = pool[:fixed_size]
    prompts = [p[1] for p in pool]
    token_ids = [p[2] for p in pool]

    if lengths:
        hist = Counter(int(8 * math.floor(l / 8)) for l in lengths)
        bars = ", ".join(f"{k:>3}-{k+7:>3}:{v}" for k, v in sorted(hist.items()))
        print(f"[prepare] prompt length histogram (bucket=8): {bars}")

    return prompts, token_ids, lengths

if __name__ == "__main__":
    assert 0 < EVAL_SIZE <= FIXED_SIZE, "EVAL_SIZE must be in (0, FIXED_SIZE]"
    print("Loading dataset...")
    ds = load_dataset(DATASET_NAME, split="train")
    print(f"Dataset size: {len(ds)}")

    print(f"Selecting fixed {FIXED_SIZE} prompts (seed={GLOBAL_SEED}) ...")
    prompts, gpt2_token_ids, lengths = collect_fixed_prompts(ds)

    g = torch.Generator().manual_seed(GLOBAL_SEED)
    perm = torch.randperm(len(gpt2_token_ids), generator=g).tolist()
    eval_indices  = perm[:EVAL_SIZE]
    train_indices = perm[EVAL_SIZE:]

    obj = {
        "prompts": prompts,
        "gpt2_token_ids": gpt2_token_ids,
        "lengths": lengths,
        "eos_id": EOS_ID,
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "seed": GLOBAL_SEED,
        "dataset_name": DATASET_NAME,
        "format_note": "Each prompt is 'Human: ...\\n\\nAssistant:'.",
        "lang_hint": "en",
        "train_indices": train_indices,
        "eval_indices":  eval_indices,
        "fixed_size": FIXED_SIZE,
        "eval_size": EVAL_SIZE,
    }
    torch.save(obj, OUTPUT_FILE)

    try:
        preview_path = os.path.join(SAVE_DIR, "prompt_preview.json")
        with open(preview_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset": DATASET_NAME,
                    "seed": GLOBAL_SEED,
                    "fixed_size": FIXED_SIZE,
                    "eval_size": EVAL_SIZE,
                    "max_prompt_tokens": MAX_PROMPT_TOKENS,
                    "examples": prompts[:8],
                },
                f, ensure_ascii=False, indent=2,
            )
        print(f"[prepare] preview saved: {preview_path}")
    except Exception:
        pass

    print(f"Saved {len(prompts)} prompts to {OUTPUT_FILE}")
    print(f"[prepare] split -> train={len(train_indices)} eval={len(eval_indices)}")