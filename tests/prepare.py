# prepare.py
import os, json, random, math, warnings
from typing import List, Dict, Tuple
from dataclasses import dataclass

import torch

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except Exception:
    HAS_DATASETS = False

RNG_SEED_DEFAULT = 1337

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _read_jsonl(p: str) -> List[Dict]:
    out = []
    with open(p, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out

def _write_jsonl(p: str, rows: List[Dict]):
    with open(p, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _sample_ultrafeedback(n_train: int, n_eval: int, seed: int) -> Tuple[List[str], List[str]]:
    # 仅抽取 prompt/instruction 字段用于 RL 阶段
    if not HAS_DATASETS:
        raise RuntimeError("datasets 未安装或不可用，且本地 data/*.jsonl 不存在。请提供本地数据或安装 datasets。")
    ds = load_dataset("openbmb/UltraFeedback", split="train")
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    prompts = []
    for i in idxs:
        row = ds[i]
        text = row.get("instruction") or row.get("prompt") or row.get("question") or ""
        text = text.strip()
        if text:
            prompts.append(text)
    if len(prompts) < (n_train + n_eval):
        raise RuntimeError("UltraFeedback 可用样本不足。")
    train = prompts[:n_train]
    evals = prompts[n_train:n_train+n_eval]
    return train, evals

def _dedupe_keep_short_first(texts: List[str]) -> List[str]:
    seen = set(); out=[]
    for t in sorted(texts, key=len):
        k = t.strip()
        if not k or k in seen: continue
        seen.add(k); out.append(k)
    return out

@dataclass
class DataConfig:
    data_dir: str = "./data"
    train_filename: str = "train_prompts.jsonl"
    eval_filename: str  = "eval_prompts.jsonl"
    train_size: int = 20000         # 训练 prompt 池规模（RL 阶段不需要很大）
    eval_size: int  = 32            # 固定评测集大小
    seed:   int = RNG_SEED_DEFAULT

class PromptPool:
    def __init__(self, prompts: List[str], seed: int = RNG_SEED_DEFAULT):
        self.prompts = prompts
        self.rng = random.Random(seed)
        self._cursor = 0
        self.rng.shuffle(self.prompts)

    def sample_batch(self, bs: int) -> List[str]:
        # 简单循环队列
        out = []
        for _ in range(bs):
            if self._cursor >= len(self.prompts):
                self.rng.shuffle(self.prompts)
                self._cursor = 0
            out.append(self.prompts[self._cursor])
            self._cursor += 1
        return out

def load_or_build_data(cfg: DataConfig) -> Tuple[PromptPool, List[str]]:
    _ensure_dir(cfg.data_dir)
    train_p = os.path.join(cfg.data_dir, cfg.train_filename)
    eval_p  = os.path.join(cfg.data_dir, cfg.eval_filename)

    if os.path.exists(train_p) and os.path.exists(eval_p):
        train_rows = _read_jsonl(train_p)
        eval_rows  = _read_jsonl(eval_p)
        trains = [r.get("prompt","").strip() for r in train_rows if r.get("prompt")]
        evals  = [r.get("prompt","").strip() for r in eval_rows  if r.get("prompt")]
    else:
        # 回退到 HF UltraFeedback 自动抽样
        trains, evals = _sample_ultrafeedback(cfg.train_size, cfg.eval_size, cfg.seed)
        trains = _dedupe_keep_short_first(trains)
        evals  = _dedupe_keep_short_first(evals)[:cfg.eval_size]
        _write_jsonl(train_p, [{"prompt": t} for t in trains])
        _write_jsonl(eval_p,   [{"prompt": t} for t in evals])

    if len(evals) != cfg.eval_size:
        # 强制固定 32 条，若不足则回填训练集前若干
        need = cfg.eval_size - len(evals)
        evals = evals + trains[:need]
        evals = evals[:cfg.eval_size]
        _write_jsonl(os.path.join(cfg.data_dir, cfg.eval_filename), [{"prompt": t} for t in evals])

    pool = PromptPool(trains, seed=cfg.seed)
    return pool, evals

# ================= W&B（离线）=================
def init_wandb_offline(project: str, run_name: str, config: dict):
    import wandb
    return wandb.init(project=project, name=run_name, mode="offline", config=config)

# ================== 设备选择 ==================
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"
