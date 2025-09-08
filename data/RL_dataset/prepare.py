import os
import re
import torch
import hashlib
from datasets import load_dataset
import tiktoken

# ================= config =================
# 选用英文数据以匹配英文奖励模型（如 OpenAssistant/reward-model-deberta-v3-base）
DATASET_NAME = "Anthropic/hh-rlhf"
FIXED_SIZE = 1024            # 固定抽样条数（总体池大小）
EVAL_SIZE  = 16             # 从 FIXED_SIZE 中切出的固定评测集大小
GLOBAL_SEED = 1337
MAX_PROMPT_TOKENS = 224      # 只截断 prompt，给 response 留空间
MIN_PROMPT_TOKENS = 8
ALLOW_EMPTY_ASSISTANT = False  # 若文本无 Assistant 段，是否允许合成一个空回答位

SAVE_DIR = os.path.join(os.path.dirname(__file__))
os.makedirs(SAVE_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(SAVE_DIR, "prompt.bin")

# actor 侧（nanoGPT）使用的分词器（GPT-2 BPE）
enc = tiktoken.get_encoding("gpt2")
EOS_ID = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"} )[0]


# ================ helpers =================
def stable_hash(s: str, seed: int) -> int:
    h = hashlib.sha1((str(seed) + "§" + s).encode("utf-8")).hexdigest()
    return int(h, 16)

def load_rl_dataset(split="train"):
    # Anthropic/hh-rlhf: 通常包含 'chosen' / 'rejected' 两列
    ds = load_dataset(DATASET_NAME, split=split)
    return ds

def norm_text(s: str) -> str:
    return (s or "").replace("\r\n", "\n").replace("\r", "\n").strip()

def ensure_human_lead(text: str) -> str:
    """
    若不以 'Human:' 开头，则补上标签，以便统一格式：
    Human: <user content>\n\nAssistant:
    """
    t = text.lstrip()
    if not t.startswith("Human:"):
        return "Human: " + t
    return t

def cut_to_prompt_prefix(full_dialogue: str) -> str:
    """
    从一段对话文本中裁出“用于生成前缀”的 prompt：
    - 截到**第一个 'Assistant:' 出现之前**，然后以 '\n\nAssistant:' 结尾
    - 这样生成时，模型会从 Assistant 段位继续生成
    """
    s = norm_text(full_dialogue)
    if not s:
        return ""

    # 找第一个 'Assistant:' 的位置；常见格式前面会有空行
    m = re.search(r"(?:\n\n)?Assistant:", s)
    if m:
        prefix = s[:m.start()]  # 不含 'Assistant:' 本身
        prefix = ensure_human_lead(prefix)
        prompt = prefix.rstrip() + "\n\nAssistant:"
        return prompt

    # 如果整段里没有 Assistant 段
    if ALLOW_EMPTY_ASSISTANT:
        # 取第一段（通常是 Human 内容），并补上 Assistant:
        parts = re.split(r"\n{2,}", s)
        head = ensure_human_lead(parts[0]) if parts else ensure_human_lead(s)
        return head.rstrip() + "\n\nAssistant:"

    # 否则丢弃该样本
    return ""

def encode_prompt(prompt_text: str):
    ids = enc.encode(prompt_text, allowed_special="all")
    return ids


# ================ builder =================
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

def _likely_english(text: str) -> bool:
    """
    简单英文性检测：中文字符占比太高则过滤，避免非英语 prompt 进入。
    不追求极准，只求与英文 RM 大致对齐。
    """
    if not text:
        return False
    total = len(text)
    cjk = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
    return (cjk / max(total, 1)) < 0.05  # 允许极少量非英文字符

def collect_fixed_prompts(ds, fixed_size=FIXED_SIZE, seed=GLOBAL_SEED):
    pool = []
    seen = set()

    col = _choose_source_column(ds)
    print(f"[prepare] using column: '{col}' from dataset (available: {ds.column_names})")

    for row in ds:
        val = row.get(col, None)
        if not isinstance(val, str):
            continue

        prompt = cut_to_prompt_prefix(val)
        if not prompt:
            continue

        # 过滤非英语对话，尽量与英文 RM 匹配
        if not _likely_english(prompt):
            continue

        ids = encode_prompt(prompt)
        if len(ids) < MIN_PROMPT_TOKENS:
            continue

        # 截断到 prompt 上限（只裁剪 prompt，不动 response 空间）
        ids = ids[:MAX_PROMPT_TOKENS]
        if not ids:
            continue

        # 文本去重（基于 prompt 文本）
        k = hashlib.md5(prompt.encode("utf-8")).hexdigest()
        if k in seen:
            continue
        seen.add(k)

        pool.append((stable_hash(prompt, seed), prompt, ids))

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
    return prompts, token_ids, lengths


# =================== main ==================
if __name__ == "__main__":
    assert 0 < EVAL_SIZE <= FIXED_SIZE, "EVAL_SIZE must be in (0, FIXED_SIZE]"

    print("Loading dataset...")
    ds = load_rl_dataset("train")
    print(f"Dataset size: {len(ds)}")

    print(f"Selecting fixed {FIXED_SIZE} prompts (seed={GLOBAL_SEED}) ...")
    prompts, gpt2_token_ids, lengths = collect_fixed_prompts(ds)

    # ===== 切分固定评测/训练索引（只在固定 1024 内部再随机划分）=====
    g = torch.Generator().manual_seed(GLOBAL_SEED)
    perm = torch.randperm(len(gpt2_token_ids), generator=g).tolist()
    eval_indices  = perm[:EVAL_SIZE]
    train_indices = perm[EVAL_SIZE:]

    obj = {
        "prompts": prompts,                 # list[str] —— 给奖励模型/调试查看
        "gpt2_token_ids": gpt2_token_ids,   # list[list[int)] —— actor 侧直接使用（不 pad）
        "lengths": lengths,
        "eos_id": EOS_ID,
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "seed": GLOBAL_SEED,
        "dataset_name": DATASET_NAME,
        "format_note": "Each prompt ends with '\\n\\nAssistant:' so the model should generate the response next.",
        "lang_hint": "en",                  # 标注英语，方便其他脚本做一致性检查
        # 关键：落盘固定划分索引
        "train_indices": train_indices,
        "eval_indices":  eval_indices,
        "fixed_size": FIXED_SIZE,
        "eval_size": EVAL_SIZE,
    }
    torch.save(obj, OUTPUT_FILE)
    print(f"Saved {len(prompts)} prompts to {OUTPUT_FILE}")
    print(f"[prepare] split -> train={len(train_indices)} eval={len(eval_indices)}")
