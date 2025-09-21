# rollout_worker.py
import argparse
import os
import json
import time
import uuid
import random
import glob
import sys
from typing import List, Dict, Optional

import torch
import tiktoken

try:
    import sglang as sgl
except Exception as e:
    print(f"[worker][warn] failed to import sglang: {e}", flush=True)
    sgl = None

TMP_SUFFIX = ".tmp"

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def _atomic_write_jsonl(path: str, rows: List[dict]) -> None:
    tmp = path + TMP_SUFFIX
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def _flush_incremental(out_dir: str, stash: List[dict]) -> None:
    if not stash: return
    path = os.path.join(out_dir, f"roll_{uuid.uuid4().hex}.jsonl")
    _atomic_write_jsonl(path, stash); stash.clear()
    print(f"[worker] flushed {path}", flush=True)

def _clean_tmp_leftovers(out_dir: str) -> None:
    for fp in glob.glob(os.path.join(out_dir, f"*{TMP_SUFFIX}")):
        try: os.remove(fp)
        except Exception: pass

def _load_json_list(path: Optional[str]) -> Optional[List[int]]:
    if not path: return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            arr = json.load(f)
        if isinstance(arr, list) and all(isinstance(x, int) for x in arr):
            return arr
    except Exception as e:
        print(f"[worker][warn] fail to read indices from {path}: {e}", flush=True)
    return None

def _read_pointer(path: Optional[str]) -> Optional[str]:
    if not path: return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            line = f.readline().strip()
        return line if line else None
    except Exception:
        return None

def _cuda_free_mb() -> int:
    try:
        free, total = torch.cuda.mem_get_info()
        return int(free // (1024 * 1024))
    except Exception:
        return 0

def _retry_sleep(backoff_s: float) -> float:
    time.sleep(backoff_s)
    return min(backoff_s * 2.0, 8.0)

# ====== 导出目录“签名”，用于与训练侧严格对齐 ======
def _dir_mtime_signature(path: str, glob_pat: str = "**/*", limit: int = 4096) -> float:
    sig = 0.0
    if not os.path.isdir(path): return sig
    cnt = 0
    for p in glob.iglob(os.path.join(path, glob_pat), recursive=True):
        try:
            sig = max(sig, os.path.getmtime(p)); cnt += 1
            if cnt >= limit: break
        except Exception:
            pass
    return sig

def _realpath_signature(path: str) -> str:
    try: return os.path.realpath(path)
    except Exception: return ""

class EngineWrapper:
    def __init__(self, init_path: str, strategy: str, pointer: Optional[str], mtime_glob: str):
        self.path = init_path
        self.strategy = strategy
        self.pointer = pointer
        self.mtime_glob = mtime_glob or "**/*"
        self.eng = None
        self._last_sig = None
        self._load(init_path)
        self._update_signature(first=True)

    def _load(self, path: str):
        if sgl is None:
            self.eng = None; return
        self._destroy()
        backoff = 0.5
        for attempt in range(5):
            try:
                print(f"[worker] sglang.Engine -> {path}", flush=True)
                self.eng = sgl.Engine(model_path=path)
                self.path = path
                self._free_cuda()
                return
            except Exception as e:
                print(f"[worker][warn] engine init failed (try {attempt+1}/5): {e}", flush=True)
                backoff = _retry_sleep(backoff)
        self.eng = None

    def _destroy(self):
        if self.eng is not None:
            try:
                if hasattr(self.eng, "shutdown"): self.eng.shutdown()
            except Exception: pass
            try: del self.eng
            except Exception: pass
            self.eng = None
            self._free_cuda()

    def _free_cuda(self):
        try: torch.cuda.synchronize()
        except Exception: pass
        try: torch.cuda.empty_cache()
        except Exception: pass

    def _current_signature(self):
        if self.strategy == "pointer":
            return _read_pointer(self.pointer) or ""
        elif self.strategy == "dir_mtime":
            return _dir_mtime_signature(self.path, self.mtime_glob)
        elif self.strategy == "realpath":
            return _realpath_signature(self.path)
        else:
            return None

    def _update_signature(self, first=False):
        self._last_sig = self._current_signature()
        if first:
            print(f"[worker] reload signature init: {self._last_sig}", flush=True)

    def maybe_reload(self) -> bool:
        if self.strategy in ("none", None): return False
        new_sig = self._current_signature()
        if new_sig != self._last_sig:
            print(f"[worker] detected change in signature: {self._last_sig} -> {new_sig}", flush=True)
            self._load(self.path)
            self._last_sig = new_sig
            return True
        return False

    def generate(self, inputs: List[str], params: Dict) -> Optional[List[dict]]:
        if self.eng is None: return None
        return self.eng.generate(inputs, params)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="sglang 初始/固定目录（可为符号链接）")
    ap.add_argument("--prompt-bin", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--count", type=int, default=256)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--min-resp", type=int, default=16)
    ap.add_argument("--mb", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--flush-interval", type=float, default=2.0)
    ap.add_argument("--flush-batch", type=int, default=256)

    # 采样
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)

    # 索引使用
    ap.add_argument("--use-only-train", action="store_true")
    ap.add_argument("--train-indices", type=str, default=None)
    ap.add_argument("--eval-indices", type=str, default=None)

    # 热重载策略
    ap.add_argument("--reload-strategy", type=str, default="dir_mtime",
                    choices=["none", "pointer", "dir_mtime", "realpath"])
    ap.add_argument("--model-pointer", type=str, default=None,
                    help="仅当 reload-strategy=pointer 时使用")
    ap.add_argument("--mtime-glob", type=str, default="**/*",
                    help="dir_mtime 策略下用于扫描的文件通配符")
    ap.add_argument("--refresh-every-batches", type=int, default=30)
    ap.add_argument("--min-free-mb", type=int, default=6000)
    ap.add_argument("--sleep-on-lowmem", type=float, default=1.0)

    # GRPO：每个 prompt 生成多条
    ap.add_argument("--per-prompt", type=int, default=1,
                    help="为同一 prompt 生成的样本数量（GRPO 设为 group_size）")

    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    ensure_dir(args.out_dir); _clean_tmp_leftovers(args.out_dir)

    try:
        blob = torch.load(args.prompt_bin, map_location="cpu")
        prompts_text = blob["prompts"]
        prompt_token_ids = blob["gpt2_token_ids"]
        eos_id = blob["eos_id"]
        pb_train_idx = blob.get("train_indices", None)
        pb_eval_idx  = blob.get("eval_indices", None)
    except Exception as e:
        print(f"[worker][fatal] failed to load prompts: {e}", flush=True); sys.exit(0)

    N = len(prompt_token_ids); all_indices = list(range(N))
    tok = tiktoken.get_encoding("gpt2")

    # train / eval 选择
    train_idx_from_file = _load_json_list(args.train_indices)
    eval_idx_from_file  = _load_json_list(args.eval_indices)
    if args.use_only_train:
        if train_idx_from_file:
            train_indices = sorted(set(i for i in train_idx_from_file if 0 <= i < N))
        elif isinstance(pb_train_idx, list) and pb_train_idx:
            train_indices = sorted(set(i for i in pb_train_idx if 0 <= i < N))
        else:
            if isinstance(pb_eval_idx, list) and pb_eval_idx:
                ev = set(i for i in pb_eval_idx if 0 <= i < N)
                train_indices = [i for i in all_indices if i not in ev]
            else:
                train_indices = all_indices
    else:
        train_indices = all_indices

    # 引擎
    engw = EngineWrapper(
        init_path=args.model,
        strategy=args.reload_strategy,
        pointer=args.model_pointer,
        mtime_glob=args.mtime_glob
    )

    total_ok = 0; total_try = 0
    stash: List[dict] = []
    last_flush = time.time()
    batch_counter = 0

    import hashlib
    def _hash_ids(ids: List[int]) -> str:
        h = hashlib.sha1(); h.update((",".join(map(str, ids))).encode("utf-8")); return h.hexdigest()

    seen_hid = set(); seen_pid_resp = set()

    def _is_garbage(resp: str) -> bool:
        s = resp.strip()
        if len(s) < 8: return True
        rep = s.count("\ufffd")
        digits = sum(ch.isdigit() for ch in s)
        puncts = sum(ch in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~" for ch in s)
        ratio = (rep + digits + puncts) / max(len(s), 1)
        return (rep > 0) or (ratio > 0.40)

    per_prompt = max(1, int(args.per_prompt))

    while total_ok < args.count:
        # 热重载检查
        if args.refresh_every_batches > 0 and batch_counter > 0 and (batch_counter % args.refresh_every_batches == 0):
            try:
                if engw.maybe_reload():
                    print("[worker] model hot-reloaded.", flush=True)
            except Exception as e:
                print(f"[worker][warn] hot-reload check failed: {e}", flush=True)

        # 显存保护
        free_mb = _cuda_free_mb()
        mb = int(args.mb)
        if free_mb < int(args.min_free_mb):
            if mb > 1:
                mb = max(1, mb // 2); print(f"[worker] low mem {free_mb}MB -> shrink mb to {mb}", flush=True)
            else:
                print(f"[worker] low mem {free_mb}MB -> sleep {args.sleep_on_lowmem}s", flush=True)
                time.sleep(max(0.1, float(args.sleep_on_lowmem)))

        # “prompt 组”的数量（每个组会被复制 per_prompt 次）
        take_groups = min(mb, args.count - total_ok)
        if len(train_indices) >= take_groups:
            base_idxs = random.sample(train_indices, k=take_groups)
        else:
            base_idxs = [train_indices[(total_ok + i) % len(train_indices)] for i in range(take_groups)]

        # 将每个选中的 idx 重复 per_prompt 次
        idxs = []
        for i in base_idxs:
            idxs.extend([i] * per_prompt)

        batch_prompts_ids  = [prompt_token_ids[i] for i in idxs]
        batch_prompts_txt  = [tok.decode(prompt_token_ids[i]) for i in idxs]

        params = {
            "max_new_tokens": int(args.max_new),
            "ignore_eos": False,
            "stop": ["\nHuman:", "\n\nHuman:"],
            "stop_token_ids": [int(eos_id)],
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
        }
        if args.top_k and args.top_k > 0:
            params["top_k"] = int(args.top_k)
        params["repetition_penalty"] = float(args.repetition_penalty)

        # 生成
        outs = None; backoff = 0.5
        for attempt in range(4):
            try:
                outs = engw.generate(batch_prompts_txt, params); break
            except Exception as e:
                print(f"[worker][warn] generate failed (try {attempt+1}/4): {e}", flush=True)
                backoff = _retry_sleep(backoff)

        if outs is None:
            outs = [{"text": p} for p in batch_prompts_txt]

        for k, out in enumerate(outs):
            total_try += 1
            text = out.get("text") or out.get("output_text") or out.get("generated_text") or ""
            if not isinstance(text, str): continue
            prompt_txt = batch_prompts_txt[k]
            resp = text[len(prompt_txt):] if text.startswith(prompt_txt) else text
            if _is_garbage(resp): continue

            # 截到第一轮结束
            cut_markers = ["\n\nHuman:", "\nHuman:", "\n\nAssistant:", "\nAssistant:"]
            cut_pos = min([resp.find(m) for m in cut_markers if m in resp] + [len(resp)])
            resp = resp[:cut_pos].lstrip()
            if resp.startswith("Assistant:"): resp = resp[len("Assistant:"):].lstrip()
            if len(resp.strip()) == 0: continue

            # 打包
            full_txt = prompt_txt + (resp if resp.endswith("<|endoftext|>") else resp + "<|endoftext|>")
            full_ids = tiktoken.get_encoding("gpt2").encode(full_txt, allowed_special="all")
            if (len(full_ids) - len(batch_prompts_ids[k])) < int(args.min_resp): continue

            key_hid = ",".join(map(str, full_ids))
            resp_md5 = hashlib.md5(resp.encode("utf-8")).hexdigest()
            pid = _hash_ids(batch_prompts_ids[k])
            if key_hid in seen_hid: continue
            if (pid, resp_md5) in seen_pid_resp: continue
            seen_hid.add(key_hid); seen_pid_resp.add((pid, resp_md5))

            # 关键：写入 model_sig 与 _mtime，便于训练侧“按导出版本过滤”
            stash.append({
                "prompt_ids": batch_prompts_ids[k],
                "full_ids": full_ids,
                "ts": time.time(),
                "pid": pid,
                "hid": _hash_ids(full_ids),
                "model_sig": _realpath_signature(engw.path),
                "_mtime": time.time(),
            })

            total_ok += 1
            if len(stash) >= max(1, int(args.flush_batch)):
                _flush_incremental(args.out_dir, stash); last_flush = time.time()

        if time.time() - last_flush >= max(0.2, float(args.flush_interval)):
            _flush_incremental(args.out_dir, stash); last_flush = time.time()

        batch_counter += 1

    _flush_incremental(args.out_dir, stash)
    acc = (total_ok / max(1, total_try)) * 100.0
    print(f"[worker] generated_ok={total_ok} tried={total_try} acc={acc:.1f}% model={engw.path}", flush=True)
    sys.exit(0)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[worker][fatal] unexpected error: {e}", flush=True); sys.exit(0)
