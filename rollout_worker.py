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

# sglang 可能加载较慢/偶发失败，这里按需重试
try:
    import sglang as sgl
except Exception as e:
    print(f"[worker][warn] failed to import sglang: {e}", flush=True)
    sgl = None

# ---------------- utils ----------------
TMP_SUFFIX = ".tmp"

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def _atomic_write_jsonl(path: str, rows: List[dict]) -> None:
    tmp = path + TMP_SUFFIX
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)  # 原子改名

def _flush_incremental(out_dir: str, stash: List[dict]) -> None:
    """把当前缓冲写成一个 roll_*.jsonl 并清空缓冲。"""
    if not stash:
        return
    path = os.path.join(out_dir, f"roll_{uuid.uuid4().hex}.jsonl")
    _atomic_write_jsonl(path, stash)
    stash.clear()
    print(f"[worker] flushed {path}", flush=True)

def _clean_tmp_leftovers(out_dir: str) -> None:
    for fp in glob.glob(os.path.join(out_dir, f"*{TMP_SUFFIX}")):
        try:
            os.remove(fp)
        except Exception:
            pass

def _load_json_list(path: Optional[str]) -> Optional[List[int]]:
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            arr = json.load(f)
        if isinstance(arr, list) and all(isinstance(x, int) for x in arr):
            return arr
    except Exception as e:
        print(f"[worker][warn] fail to read indices from {path}: {e}", flush=True)
    return None

def load_prompts(pb_path: str):
    blob = torch.load(pb_path, map_location="cpu")
    # 可能包含 train_indices / eval_indices（由新版 prepare.py 提供）
    return (
        blob["prompts"],          # 仅用于解码检查/回显（可选）
        blob["gpt2_token_ids"],   # List[List[int]]: prompt 的 token ids
        blob["eos_id"],           # int: <|endoftext|>
        blob.get("train_indices", None),
        blob.get("eval_indices", None),
    )

def _pick_text(obj: dict) -> str:
    # 兼容多种 sglang 输出字段
    return (
        obj.get("text")
        or obj.get("output_text")
        or obj.get("generated_text")
        or ""
    )

def _retry_sleep(backoff_s: float) -> float:
    """指数退避的简易封装，返回下一次 backoff。"""
    time.sleep(backoff_s)
    return min(backoff_s * 2.0, 8.0)

def _is_garbage(resp: str) -> bool:
    s = resp.strip()
    if len(s) < 8:
        return True
    rep = s.count("\ufffd")
    digits = sum(ch.isdigit() for ch in s)
    puncts = sum(ch in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~" for ch in s)
    ratio = (rep + digits + puncts) / max(len(s), 1)
    return (rep > 0) or (ratio > 0.40)

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-bin", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--count", type=int, default=256)       # 生成“合格样本”总数
    ap.add_argument("--max-new", type=int, default=128)     # 每条最大新 token
    ap.add_argument("--min-resp", type=int, default=8,      # 与池默认一致：最少回复 tokens
                    help="最短回复 token 数，过短则丢弃（默认 8）")
    ap.add_argument("--mb", type=int, default=8)            # 每批并发条数
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--block-size", type=int, default=512)  # 训练侧 block_size 对齐
    ap.add_argument("--flush-interval", type=float, default=2.0)  # 秒
    ap.add_argument("--flush-batch", type=int, default=256)       # 累计到这么多就落一文件
    # sglang 采样参数（更稳的默认）
    ap.add_argument("--temperature", type=float, default=0.75)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=40)  # 0 表示不限
    ap.add_argument("--repetition-penalty", type=float, default=1.2)

    # ===== 新增：只用训练集 / 索引来源 =====
    ap.add_argument("--use-only-train", action="store_true",
                    help="若给出，则仅从训练索引采样（优先使用 prompt.bin 内的 train_indices）。")
    ap.add_argument("--train-indices", type=str, default=None,
                    help="JSON 文件，整数数组；明确指定训练索引。")
    ap.add_argument("--eval-indices", type=str, default=None,
                    help="JSON 文件，整数数组；若未提供训练索引，则用全部索引减去评测索引作为训练索引。")

    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    ensure_dir(args.out_dir)
    _clean_tmp_leftovers(args.out_dir)

    try:
        prompts_text, prompt_token_ids, eos_id, pb_train_idx, pb_eval_idx = load_prompts(args.prompt_bin)
    except Exception as e:
        print(f"[worker][fatal] failed to load prompts: {e}", flush=True)
        # 不让主流程挂，直接退出 0，但确实没产出
        sys.exit(0)

    N = len(prompt_token_ids)
    all_indices = list(range(N))

    # 解析训练索引优先级：
    # 1) --train-indices 文件
    # 2) prompt.bin 内的 train_indices
    # 3) 若给了 --eval-indices 或 prompt.bin 内有 eval_indices，则 train = ALL - eval
    # 4) 否则 train = ALL
    train_idx_from_file = _load_json_list(args.train_indices)
    eval_idx_from_file  = _load_json_list(args.eval_indices)

    train_indices = None
    if args.use_only_train:
        if train_idx_from_file:
            train_indices = sorted(set(i for i in train_idx_from_file if 0 <= i < N))
            src = f"--train-indices({len(train_indices)})"
        elif isinstance(pb_train_idx, list) and pb_train_idx:
            train_indices = sorted(set(i for i in pb_train_idx if 0 <= i < N))
            src = f"prompt.bin/train_indices({len(train_indices)})"
        else:
            # 回退：若能拿到 eval，则做 ALL - eval
            eval_indices = None
            if eval_idx_from_file:
                eval_indices = set(i for i in eval_idx_from_file if 0 <= i < N)
                src_eval = "--eval-indices"
            elif isinstance(pb_eval_idx, list) and pb_eval_idx:
                eval_indices = set(i for i in pb_eval_idx if 0 <= i < N)
                src_eval = "prompt.bin/eval_indices"
            if eval_indices is not None:
                train_indices = sorted([i for i in all_indices if i not in eval_indices])
                src = f"ALL - {src_eval}({len(eval_indices)})"
            else:
                # 最后兜底：仍然使用全部（但给出警告）
                train_indices = all_indices
                src = "ALL (fallback)"
                print("[worker][warn] --use-only-train 给出，但没有可用的训练/评测索引；回退为用全部索引。", flush=True)
        print(f"[worker] sampling from TRAIN indices source: {src}; train_size={len(train_indices)} / total={N}", flush=True)
    else:
        train_indices = all_indices
        print(f"[worker] sampling from ALL indices (no --use-only-train); total={N}", flush=True)

    tok = tiktoken.get_encoding("gpt2")
    eos_token = "<|endoftext|>"

    # ---------- sglang 引擎：带重试 ----------
    eng = None
    if sgl is None:
        print("[worker][warn] sglang not available; no generation will happen.", flush=True)
    else:
        backoff = 0.5
        for attempt in range(5):
            try:
                eng = sgl.Engine(model_path=args.model)
                break
            except Exception as e:
                print(f"[worker][warn] engine init failed (try {attempt+1}/5): {e}", flush=True)
                backoff = _retry_sleep(backoff)
        if eng is None:
            print("[worker][warn] give up initializing engine; exiting gracefully.", flush=True)
            # 不让主流程挂，退出 0
            sys.exit(0)

    total_ok = 0     # 写入的合格样本数
    total_try = 0    # 生成的尝试条数
    stash: List[dict] = []
    last_flush = time.time()

    # 去重（本次进程内，按 full_ids 文本去重）
    # seen = set()
    seen_hid = set()
    seen_pid_resp = set()  # (pid, resp_md5) 级别去重，缓和模式坍塌

    def _make_sample(prompt_ids: List[int], prompt_txt: str, resp_txt: str) -> Optional[dict]:
        # 结尾保证 eos
        if not resp_txt.endswith(eos_token):
            resp_txt = resp_txt + eos_token

        # 拼接 & 编码
        full_txt = prompt_txt + resp_txt
        full_ids = tok.encode(full_txt, allowed_special="all")

        # 裁到 block_size：优先保留完整 prompt，再截断 response
        if len(full_ids) > args.block_size:
            # 能保留的 response 长度
            room = max(args.block_size - len(prompt_ids), 1)
            full_ids = prompt_ids + full_ids[len(prompt_ids):len(prompt_ids)+room]
        # 若确实有 response，则无论是否被截断，强制以 EOS 结尾
        if len(full_ids) > len(prompt_ids):
            full_ids[-1] = eos_id

        # 严格要求有 response
        if len(full_ids) <= len(prompt_ids):
            return None

        # 最短回复 token 过滤
        if (len(full_ids) - len(prompt_ids)) < max(int(args.min_resp), 0):
            return None

        # key = ",".join(map(str, full_ids))
        # if key in seen:
        #     return None
        # seen.add(key)
        key_hid = ",".join(map(str, full_ids))
        resp_md5 = hashlib.md5(resp_txt.encode("utf-8")).hexdigest()
        pid = _hash_ids(prompt_ids)
        if key_hid in seen_hid:
            return None
        if (pid, resp_md5) in seen_pid_resp:
            return None
        seen_hid.add(key_hid)
        seen_pid_resp.add((pid, resp_md5))

        # 与“新鲜池”保持字段对齐：ts/pid/hid
        ts = time.time()
        pid = _hash_ids(prompt_ids)
        hid = _hash_ids(full_ids)

        return {
            "prompt_ids": prompt_ids,
            "full_ids": full_ids,
            "ts": ts,
            "pid": pid,
            "hid": hid,
            # 可选透传（便于离线排错/抽检）——体积考虑默认不放文本
            # "prompt_text": prompt_txt,
            # "response_text": resp_txt,
        }

    # 小工具：为 _make_sample 计算哈希
    import hashlib
    def _hash_ids(ids: List[int]) -> str:
        h = hashlib.sha1()
        h.update((",".join(map(str, ids))).encode("utf-8"))
        return h.hexdigest()

    while total_ok < args.count:
        take = min(args.mb, args.count - total_ok)  # 基于“成功样本”配额来取 batch

        # 只从 train_indices 里抽样
        if len(train_indices) >= take:
            idxs = random.sample(train_indices, k=take)
        else:
            # 训练索引太少就循环覆盖
            idxs = [train_indices[(total_ok + i) % len(train_indices)] for i in range(take)]

        batch_prompts_ids  = [prompt_token_ids[i] for i in idxs]
        batch_prompts_txt  = [tok.decode(prompt_token_ids[i]) for i in idxs]

        params = {
            "max_new_tokens": int(args.max_new),
            "ignore_eos": False,
            "stop_token_ids": [int(eos_id)],
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            # 明确的字符串停词：遇到下一轮 Human 或重复 Assistant 时停
            "stop": ["\nHuman:", "\n\nHuman:"],
        }
        if args.top_k and args.top_k > 0:
            params["top_k"] = int(args.top_k)

        outs = None
        backoff = 0.5
        for attempt in range(4):
            try:
                params["repetition_penalty"] = float(args.repetition_penalty)
                outs = eng.generate(batch_prompts_txt, params)
                break
            except Exception as e:
                print(f"[worker][warn] generate failed (try {attempt+1}/4): {e}", flush=True)
                # 整批失败也不终止，退避后再试
                backoff = _retry_sleep(backoff)

        if outs is None:
            # 仍然失败：用空输出占位继续循环，不退出
            outs = [{"text": p} for p in batch_prompts_txt]

        for k, out in enumerate(outs):
            total_try += 1
            text = _pick_text(out)
            if not isinstance(text, str):
                continue

            prompt_txt = batch_prompts_txt[k]
            # 若返回已包含 prompt，则切出 resp；否则认为 text 就是 resp
            resp = text[len(prompt_txt):] if text.startswith(prompt_txt) else text
            if _is_garbage(resp):
                continue
            BAD_MARKERS = (
                "Advertisements", "rawdownloadcloneembedreport", "\ufffd",
                "Recognize the homophones of the following word"
            )
            if any(m in resp for m in BAD_MARKERS):
                continue
                
            # 额外后处理：只保留第一轮 Assistant 内容
            cut_markers = ["\n\nHuman:", "\nHuman:", "\n\nAssistant:", "\nAssistant:"]
            cut_pos = min([resp.find(m) for m in cut_markers if m in resp] + [len(resp)])
            resp = resp[:cut_pos]
            
            # 去掉开头可能的空格/重复标签
            resp = resp.lstrip()
            if resp.startswith("Assistant:"):
                # 有些模型会把标签又打一遍，去掉它
                resp = resp[len("Assistant:"):].lstrip()
            
            # 进一步的弱垃圾规则（可选）
            bad_fragments = ["Advertisements", "rawdownloadcloneembedreportprint"]
            if any(b in resp for b in bad_fragments):
                continue

            # 简单质量过滤（字符级）：过短直接跳
            if len(resp.strip()) == 0:
                continue

            sample = _make_sample(batch_prompts_ids[k], prompt_txt, resp)
            if sample is None:
                # 该条无效（空/重复/过短/被截断为空），跳过
                continue

            stash.append(sample)
            total_ok += 1

            # 达到 flush 批次立即落盘
            if len(stash) >= max(1, int(args.flush_batch)):
                _flush_incremental(args.out_dir, stash)
                last_flush = time.time()

        # 定时落盘，避免长时间缓存
        if time.time() - last_flush >= max(0.2, float(args.flush_interval)):
            _flush_incremental(args.out_dir, stash)
            last_flush = time.time()

    # 收尾：还有缓冲则落最后一份
    _flush_incremental(args.out_dir, stash)

    acc = (total_ok / max(1, total_try)) * 100.0
    print(f"[worker] generated_ok={total_ok} tried={total_try} acc={acc:.1f}% model={args.model}", flush=True)
    # 始终 0 退出，避免主进程崩
    sys.exit(0)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # 任何未捕获异常也不让主流程挂
        print(f"[worker][fatal] unexpected error: {e}", flush=True)
        sys.exit(0)
