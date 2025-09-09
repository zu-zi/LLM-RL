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

# ================= config =================
# 选用英文数据以匹配英文奖励模型（如 OpenAssistant/reward-model-deberta-v3-*）
DATASET_NAME = "Anthropic/hh-rlhf"

# 固定采样规模（决定 prompt.bin 中的条数；与离线生成池“理论上限”无强耦合）
FIXED_SIZE = 1024
EVAL_SIZE  = 16
GLOBAL_SEED = 1337

# 仅裁剪“prompt”长度，保留 response 空间；与训练的 block_size 要匹配
MAX_PROMPT_TOKENS = 224
MIN_PROMPT_TOKENS = 8

# 若原始文本没有出现 Assistant 段，是否允许“造”一个空回答位（一般不需要）
ALLOW_EMPTY_ASSISTANT = False

# 保存位置
SAVE_DIR = os.path.dirname(__file__)
os.makedirs(SAVE_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(SAVE_DIR, "prompt.bin")

# actor 侧（nanoGPT）使用的分词器（GPT-2 BPE）
enc = tiktoken.get_encoding("gpt2")
EOS_ID = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"} )[0]

# ================= helpers =================
def _stable_hash(s: str, seed: int) -> int:
    h = hashlib.sha1((str(seed) + "§" + s).encode("utf-8")).hexdigest()
    return int(h, 16)

def _likely_english(text: str) -> bool:
    """
    简易英文检测：中文 CJK 比例过高则过滤；与英文 RM 对齐即可。
    """
    if not text:
        return False
    total = len(text)
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return (cjk / max(total, 1)) < 0.05

def _norm_text(s: str) -> str:
    return (s or "").replace("\r\n", "\n").replace("\r", "\n").strip()

def _ensure_human_lead(text: str) -> str:
    """
    若不以 'Human:' 开头，则补上标签，统一格式：
    Human: <user content>\n\nAssistant:
    """
    t = text.lstrip()
    if not t.startswith("Human:"):
        return "Human: " + t
    return t

def _cut_to_prompt_prefix(full_dialogue: str) -> str:
    """
    从一段对话文本中裁出“用于生成前缀”的 prompt：
    - 截到**第一个 'Assistant:' 出现之前**，然后以 '\n\nAssistant:' 结尾
    - 这样生成时，模型会从 Assistant 段位继续生成
    """
    s = _norm_text(full_dialogue)
    if not s:
        return ""

    # 常见数据里 'Assistant:' 之前有时有空行，这里用正则兼容
    m = re.search(r"(?:\n\n)?Assistant:", s)
    if m:
        prefix = s[:m.start()]  # 不含 'Assistant:' 本身
        prefix = _ensure_human_lead(prefix)
        prompt = prefix.rstrip() + "\n\nAssistant:"
        return prompt

    if ALLOW_EMPTY_ASSISTANT:
        parts = re.split(r"\n{2,}", s)
        head = _ensure_human_lead(parts[0]) if parts else _ensure_human_lead(s)
        return head.rstrip() + "\n\nAssistant:"

    return ""

def _encode_prompt(prompt_text: str):
    return enc.encode(prompt_text, allowed_special="all")

def _choose_source_column(ds):
    cols = ds.column_names
    # 优先顺序：chosen -> prompt -> text（hh-rlhf 一般有 chosen）
    if "chosen" in cols:
        return "chosen"
    if "prompt" in cols:
        return "prompt"
    if "text" in cols:
        return "text"
    raise RuntimeError(
        f"Dataset {DATASET_NAME} must have one of columns "
        f"'chosen' (preferred), 'prompt', or 'text'. Available columns: {cols}"
    )

# ================= builder =================
def collect_fixed_prompts(ds, fixed_size=FIXED_SIZE, seed=GLOBAL_SEED):
    pool = []
    seen_md5 = set()

    col = _choose_source_column(ds)
    print(f"[prepare] using column: '{col}' from dataset (available: {ds.column_names})")

    kept_len = []

    for row in ds:
        val = row.get(col, None)
        if not isinstance(val, str):
            continue

        prompt = _cut_to_prompt_prefix(val)
        if not prompt:
            continue

        # 过滤非英语，尽量与英文 RM 匹配
        if not _likely_english(prompt):
            continue

        ids = _encode_prompt(prompt)
        if len(ids) < MIN_PROMPT_TOKENS:
            continue

        # 仅裁剪 prompt，不动 response 空间
        ids = ids[:MAX_PROMPT_TOKENS]
        if not ids:
            continue

        # 文本级去重（更稳，避免同义重复）
        k = hashlib.md5(prompt.encode("utf-8")).hexdigest()
        if k in seen_md5:
            continue
        seen_md5.add(k)

        kept_len.append(len(ids))
        pool.append((_stable_hash(prompt, seed), prompt, ids))

    if not pool:
        raise RuntimeError(
            "No valid prompts were built. This usually means the input column "
            "does not contain 'Human:' / 'Assistant:' style dialogues, or everything "
            "was filtered out by language/length constraints."
        )

    # 稳定排序后取前 fixed_size
    pool.sort(key=lambda x: x[0])
    pool = pool[:fixed_size]

    prompts = [p[1] for p in pool]     # list[str]
    token_ids = [p[2] for p in pool]   # list[list[int]]
    lengths = [len(p[2]) for p in pool]

    # 可观测性：长度分布（粗略）
    if lengths:
        hist = Counter(int(8 * math.floor(l / 8)) for l in lengths)
        bars = ", ".join(f"{k:>3}-{k+7:>3}:{v}" for k, v in sorted(hist.items()))
        print(f"[prepare] prompt length histogram (bucket=8): {bars}")

    return prompts, token_ids, lengths

# =================== main ==================
if __name__ == "__main__":
    assert 0 < EVAL_SIZE <= FIXED_SIZE, "EVAL_SIZE must be in (0, FIXED_SIZE]"

    print("Loading dataset...")
    ds = load_dataset(DATASET_NAME, split="train")
    print(f"Dataset size: {len(ds)}")

    print(f"Selecting fixed {FIXED_SIZE} prompts (seed={GLOBAL_SEED}) ...")
    prompts, gpt2_token_ids, lengths = collect_fixed_prompts(ds)

    # ===== 切分固定评测/训练索引（仅在固定集合内随机划分，稳定可复现）=====
    g = torch.Generator().manual_seed(GLOBAL_SEED)
    perm = torch.randperm(len(gpt2_token_ids), generator=g).tolist()
    eval_indices  = perm[:EVAL_SIZE]
    train_indices = perm[EVAL_SIZE:]

    obj = {
        # 文本仅用于抽检/可视化，训练/推理不强依赖
        "prompts": prompts,                 # list[str]
        # actor 直接使用（不 pad）
        "gpt2_token_ids": gpt2_token_ids,   # list[list[int]]
        "lengths": lengths,
        "eos_id": EOS_ID,
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "seed": GLOBAL_SEED,
        "dataset_name": DATASET_NAME,
        "format_note": "Each prompt ends with '\\n\\nAssistant:' so the model should generate the response next.",
        "lang_hint": "en",
        # 与 rollout_worker 对齐的固定划分
        "train_indices": train_indices,
        "eval_indices":  eval_indices,
        "fixed_size": FIXED_SIZE,
        "eval_size": EVAL_SIZE,
    }
    torch.save(obj, OUTPUT_FILE)

    # 额外保存一份 JSON 供人眼快速查看（可选）
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
