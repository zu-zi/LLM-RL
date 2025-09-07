# rollout_worker.py
import argparse
import os
import json
import time
import uuid
import random

import torch
import tiktoken
import sglang as sgl

# ------------- utils -------------
def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def atomic_write_jsonl(path: str, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)  # 原子改名，避免训练进程读到半文件

def load_prompts(pb_path: str):
    blob = torch.load(pb_path, map_location="cpu")
    return blob["prompts"], blob["gpt2_token_ids"], blob["eos_id"]

# ------------- main -------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-bin", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--count", type=int, default=256)     # 生成总条数
    ap.add_argument("--max-new", type=int, default=192)   # 每条最大新 token
    ap.add_argument("--mb", type=int, default=8)          # 每批并发条数
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    random.seed(args.seed)

    ensure_dir(args.out_dir)
    prompts_text, prompt_token_ids, eos_id = load_prompts(args.prompt_bin)

    tok = tiktoken.get_encoding("gpt2")
    eos_token = "<|endoftext|>"

    # sglang 引擎（保持最小配置，方便在子进程内干净加载/卸载）
    eng = sgl.Engine(model_path=args.model)

    total = 0
    rows = []

    while total < args.count:
        take = min(args.mb, args.count - total)

        # 随机抽样（避免总是同一批 prompt）
        idxs = random.sample(range(len(prompt_token_ids)), k=take) \
            if len(prompt_token_ids) >= take else \
            [(total + i) % len(prompt_token_ids) for i in range(take)]

        # 把 token ids 解回文本作为输入
        batch_prompts = [tok.decode(prompt_token_ids[i]) for i in idxs]

        # sglang 参数：遇到 eos_id 就停
        params = {
            "max_new_tokens": int(args.max_new),
            "ignore_eos": False,
            "stop_token_ids": [int(eos_id)],
        }

        try:
            outs = eng.generate(batch_prompts, params)
        except Exception as e:
            # 出错则为这一批产出空字符串，避免阻塞主流程
            outs = [{"text": p} for p in batch_prompts]

        for k, out in enumerate(outs):
            text = (
                out.get("text")
                or out.get("output_text")
                or out.get("generated_text")
                or ""
            )

            prompt_txt = batch_prompts[k]
            # 把返回文本中的“回复部分”切出来（若没拼接回 prompt，就直接当全量）
            resp = text[len(prompt_txt):] if text.startswith(prompt_txt) else text

            # 结尾兜底加 eos，确保训练侧能在 response 末尾定位
            if not resp.endswith(eos_token):
                resp += eos_token

            full_ids = tok.encode(prompt_txt + resp, allowed_special="all")
            rows.append({
                "prompt_ids": prompt_token_ids[idxs[k]],
                "full_ids": full_ids
            })

        total += take

    # 文件名改为 roll_*.jsonl，匹配训练侧的出队规则
    out_path = os.path.join(args.out_dir, f"roll_{uuid.uuid4().hex}.jsonl")
    atomic_write_jsonl(out_path, rows)
    print(f"[worker] wrote {total} -> {out_path}")

if __name__ == "__main__":
    main()
