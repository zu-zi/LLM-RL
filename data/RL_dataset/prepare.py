# data/RL_dataset/prepare.py
import os
import re
import json
import math
import hashlib
from collections import Counter

import torch
from datasets import load_dataset
import tiktoken

# ================== 基本配置（保持与你现有训练脚本对齐） ==================
DATASET_NAME = "openbmb/UltraFeedback"  # 英文指令类，适合英文 RM
FIXED_SIZE   = 1024                     # 训练集总条数
EVAL_SIZE    = 16                       # eval 子集条数（从 FIXED_SIZE 中划出）
GLOBAL_SEED  = 1337

# 仅裁剪“prompt”侧长度；留出足够空间给回复（block_size=256 时 152 足够）
MAX_PROMPT_TOKENS = 152
MIN_PROMPT_TOKENS = 32

# UltraFeedback/指令类语料常没有 Assistant 段，允许补一个空位
ALLOW_EMPTY_ASSISTANT = True

# 保存位置
SAVE_DIR    = os.path.dirname(__file__)
os.makedirs(SAVE_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(SAVE_DIR, "prompt.bin")

# actor 侧 tokenizer（GPT-2 BPE）
enc    = tiktoken.get_encoding("gpt2")
EOS_ID = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

# ================== 小工具 ==================
def _stable_hash(s: str, seed: int) -> int:
    # 同一个 seed 下，文本对应的哈希稳定/可复现
    h = hashlib.sha1((str(seed) + "§" + s).encode("utf-8")).hexdigest()
    return int(h, 16)

def _likely_english(text: str) -> bool:
    """非常轻量的英文启发式：汉字占比<2%，非 ASCII 占比<25%"""
    if not text:
        return False
    total = len(text)
    if total == 0:
        return False
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    if (cjk / total) >= 0.02:
        return False
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    return (non_ascii / total) < 0.25

def _norm_text(s: str) -> str:
    return (s or "").replace("\r\n", "\n").replace("\r", "\n").strip()

def _ensure_human_lead(text: str) -> str:
    t = text.lstrip()
    return t if t.startswith("Human:") else ("Human: " + t)

def _cut_to_prompt_prefix(full_dialogue: str) -> str:
    """
    从包含 Human/Assistant 的长串里裁到第一个 Assistant: 之前，
    并以 '\\n\\nAssistant:' 结尾，供 actor 续写回答。
    """
    s = _norm_text(full_dialogue)
    if not s:
        return ""
    m = re.search(r"(?:\n\n)?Assistant:", s)
    if m:
        prefix = s[:m.start()]
        prefix = _ensure_human_lead(prefix)
        prompt = prefix.rstrip() + "\n\nAssistant:"
        # 避免出现多余换行
        prompt = re.sub(r"(?:\n+)?Assistant:$", "Assistant:", prompt)
        return prompt
    if ALLOW_EMPTY_ASSISTANT:
        head = _ensure_human_lead(s)
        return head.rstrip() + "\n\nAssistant:"
    return ""

def _encode_prompt(prompt_text: str):
    return enc.encode(prompt_text, allowed_special="all")

def _choose_source_column(ds):
    cols = ds.column_names
    # UltraFeedback 主列通常是 "instruction"
    if "instruction" in cols:
        return "instruction"
    if "chosen" in cols:
        return "chosen"
    if "prompt" in cols:
        return "prompt"
    if "text" in cols:
        return "text"
    raise RuntimeError(
        f"Dataset {DATASET_NAME} must have one of columns "
        f"'instruction' (preferred), 'chosen', 'prompt', or 'text'. Available: {cols}"
    )

def _row_to_prompt(row: dict, col: str) -> str:
    """
    统一把不同数据集的一行转成对齐格式的 prompt：
    - UltraFeedback: 用 instruction，包成 'Human: ...\\n\\nAssistant:'
    - 其它列：若已有完整对话，裁到 Assistant 之前
    """
    val = row.get(col, None)
    if not isinstance(val, str):
        return ""
    if col == "instruction":
        instr = _norm_text(val)
        if not instr:
            return ""
        return f"Human: {instr}\n\nAssistant:"
    else:
        return _cut_to_prompt_prefix(val)

# ================== 主逻辑 ==================
def collect_fixed_prompts(ds, fixed_size=FIXED_SIZE, seed=GLOBAL_SEED):
    """
    选出 fixed_size 条“提示前缀”（prompt），每条形如：
      Human: ...
      
      Assistant:
    并保存对应的 GPT-2 token ids（仅裁剪 prompt 段到 MAX_PROMPT_TOKENS）。
    """
    pool = []
    seen_md5 = set()
    lengths_all = []

    col = _choose_source_column(ds)
    print(f"[prepare] using column: '{col}' (available: {ds.column_names})")

    for row in ds:
        prompt = _row_to_prompt(row, col)
        if not prompt:
            continue

        # 过滤非英文样本（奖励模型匹配英文）
        if not _likely_english(prompt):
            continue

        ids = _encode_prompt(prompt)
        if len(ids) < MIN_PROMPT_TOKENS:
            continue

        ids = ids[:MAX_PROMPT_TOKENS]
        if not ids:
            continue

        # 基于原始文本去重
        k = hashlib.md5(prompt.encode("utf-8")).hexdigest()
        if k in seen_md5:
            continue
        seen_md5.add(k)

        lengths_all.append(len(ids))
        pool.append((_stable_hash(prompt, seed), prompt, ids))

    if not pool:
        raise RuntimeError(
            "No valid prompts were built. Check dataset column mapping or language/length filters."
        )

    # 用稳定哈希排序，截断为 fixed_size，保证可复现
    pool.sort(key=lambda x: x[0])
    pool = pool[:fixed_size]

    prompts   = [p[1] for p in pool]
    token_ids = [p[2] for p in pool]
    lengths   = [len(t) for t in token_ids]

    # 统计（对保留样本）
    if lengths:
        hist = Counter(int(8 * math.floor(l / 8)) for l in lengths)
        bars = ", ".join(f"{k:>3}-{k+7:>3}:{v}" for k, v in sorted(hist.items()))
        print(f"[prepare] prompt length histogram (bucket=8): {bars}")

    return prompts, token_ids, lengths

# ================== 脚本入口 ==================
if __name__ == "__main__":
    assert 0 < EVAL_SIZE <= FIXED_SIZE, "EVAL_SIZE must be in (0, FIXED_SIZE]"

    print("Loading dataset...")
    ds = load_dataset(DATASET_NAME, split="train")
    print(f"Dataset size: {len(ds)}")

    print(f"Selecting fixed {FIXED_SIZE} prompts (seed={GLOBAL_SEED}) ...")
    prompts, gpt2_token_ids, lengths = collect_fixed_prompts(ds)

    # 可复现的 train/eval 拆分
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
        "format_note": "Each prompt is 'Human: ...\\n\\nAssistant:' so the actor generates the answer.",
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
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[prepare] preview saved: {preview_path}")
    except Exception:
        pass

    print(f"Saved {len(prompts)} prompts to {OUTPUT_FILE}")
    print(f"[prepare] split -> train={len(train_indices)} eval={len(eval_indices)}")
