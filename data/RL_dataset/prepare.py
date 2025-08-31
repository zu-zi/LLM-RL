# prepare.py
import os
import torch
import hashlib
from datasets import load_dataset
import tiktoken

# config
DATASET_NAME = "Anthropic/hh-rlhf"
FIXED_SIZE = 1024           # 固定只训练这 1024 条
GLOBAL_SEED = 1337      
MAX_PROMPT_TOKENS = 256     # 用 gpt2 分词器裁剪的上限（只裁剪 prompt）
MIN_PROMPT_TOKENS = 8

SAVE_DIR = os.path.join(os.path.dirname(__file__), "data", "RL_dataset")
os.makedirs(SAVE_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(SAVE_DIR, "prompt.bin")

# actor 侧（nanoGPT）使用的分词器
enc = tiktoken.get_encoding("gpt2")
EOS_ID = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"} )[0]

def stable_hash(s: str, seed: int) -> int:
    # 用 seed 注入的 sha1 哈希做稳定子集抽样
    h = hashlib.sha1((str(seed) + "§" + s).encode("utf-8")).hexdigest()
    return int(h, 16)

def load_rl_dataset(split="train"):
    ds = load_dataset(DATASET_NAME, split=split)
    # 统一到 "text" 列
    if "text" not in ds.column_names:
        for c in ds.column_names:
            try:
                if ds.features[c].dtype == "string":
                    ds = ds.rename_column(c, "text")
                    break
            except Exception:
                pass
    return ds

def normalize_text(s: str) -> str:
    return (s or "").strip()

def collect_fixed_prompts(ds, fixed_size=FIXED_SIZE, seed=GLOBAL_SEED):
    pool = []
    seen_keys = set()

    for s in ds["text"]:
        if not isinstance(s, str):
            continue
        s = normalize_text(s)
        if not s:
            continue

        # 用 gpt2 分词以便做长度过滤与后续 actor 侧直接使用
        ids = enc.encode(s, allowed_special="all")
        if len(ids) < MIN_PROMPT_TOKENS:
            continue

        # 只裁到 prompt 上限，保证训练时 response 还能放得下
        ids = ids[:MAX_PROMPT_TOKENS]
        if not ids:
            continue

        # 文本去重（按 MD5）
        k = hashlib.md5(s.encode("utf-8")).hexdigest()
        if k in seen_keys:
            continue
        seen_keys.add(k)

        pool.append((stable_hash(s, seed), s, ids))

    # 稳定排序后取前 fixed_size
    pool.sort(key=lambda x: x[0])
    pool = pool[:fixed_size]

    prompts = [p[1] for p in pool]      # list[str]
    token_ids = [p[2] for p in pool]    # list[list[int]]（未 pad）
    lengths = [len(p[2]) for p in pool]
    return prompts, token_ids, lengths

# ------------ main ------------
if __name__ == "__main__":
    print("Loading dataset...")
    ds = load_rl_dataset("train")
    print(f"Dataset size: {len(ds)}")

    print(f"Selecting fixed {FIXED_SIZE} prompts (seed={GLOBAL_SEED}) ...")
    prompts, gpt2_token_ids, lengths = collect_fixed_prompts(ds)

    obj = {
        "prompts": prompts,                 # list[str]  —— 训练时用于拼接 reward 文本
        "gpt2_token_ids": gpt2_token_ids,   # list[list[int]] —— actor 侧直接用（不 pad）
        "lengths": lengths,                 # list[int]
        "eos_id": EOS_ID,
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "seed": GLOBAL_SEED,
        "dataset_name": DATASET_NAME,
    }
    torch.save(obj, OUTPUT_FILE)
    print(f"Saved {len(prompts)} prompts to {OUTPUT_FILE}")
