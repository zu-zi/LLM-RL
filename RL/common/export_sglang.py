# RL/common/export_sglang.py
import os, glob, random, shutil
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def atomic_symlink_update(target_dir: str, link_path: str):
    tmp = link_path + ".tmp"
    try:
        if os.path.lexists(tmp):
            os.remove(tmp)
    except Exception:
        pass
    # 若原路径存在且不是符号链接，清掉目录（避免 replace 失败）
    if os.path.exists(link_path) and not os.path.islink(link_path):
        shutil.rmtree(link_path)
    os.symlink(target_dir, tmp)
    os.replace(tmp, link_path)

# 将当前 actor 权重对齐到 HF 目录
@torch.no_grad()
def export_actor_for_sglang(raw_actor, init_from: str, export_base: str, symlink_path: str):
    os.makedirs(export_base, exist_ok=True)

    hf_model = AutoModelForCausalLM.from_pretrained(init_from, torch_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(init_from, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    hf_model.eval()

    sd_src = raw_actor.state_dict()
    sd_tgt = hf_model.state_dict()

    TRANSPOSE_SUFFIXES = (
        "attn.c_attn.weight",
        "attn.c_proj.weight",
        "mlp.c_fc.weight",
        "mlp.c_proj.weight",
    )

    matched = transposed = skipped = 0
    for k, w_tgt in sd_tgt.items():
        w_src = sd_src.get(k, None)
        if w_src is None:
            skipped += 1
            continue

        if any(k.endswith(suf) for suf in TRANSPOSE_SUFFIXES):
            if w_src.T.shape == w_tgt.shape:
                w_tgt.copy_(w_src.T.to(w_tgt.dtype))
                transposed += 1
                continue

            if w_src.shape == w_tgt.shape:
                w_tgt.copy_(w_src.to(w_tgt.dtype))
                matched += 1
                continue
            skipped += 1
            continue

        if w_src.shape == w_tgt.shape:
            w_tgt.copy_(w_src.to(w_tgt.dtype))
            matched += 1
        else:
            skipped += 1

    try:
        hf_model.transformer.wte.weight = hf_model.lm_head.weight
    except Exception:
        pass

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(export_base, f"ts_{ts}_{random.randint(1000,9999)}")
    os.makedirs(out_dir, exist_ok=True)
    hf_model.save_pretrained(out_dir, safe_serialization=True, max_shard_size="2GB")
    tok.save_pretrained(out_dir)

    atomic_symlink_update(out_dir, symlink_path)

    # 期望转置数：4 * n_layer（gpt2 为 c_attn/c_proj/c_fc/c_proj）
    try:
        exp_transpose = 4 * int(getattr(hf_model.config, "n_layer", 0))
    except Exception:
        exp_transpose = None

    note = f" expected_transposed={exp_transpose}" if exp_transpose is not None else ""
    print(f"[export] to={out_dir} matched={matched} transposed={transposed} skipped={skipped}{note}", flush=True)

    # 仅保留最新 1 份
    ds = sorted([d for d in glob.glob(os.path.join(export_base, "ts_*")) if os.path.isdir(d)])
    for d in ds[:-1]:
        try:
            shutil.rmtree(d)
        except Exception:
            pass
    return out_dir