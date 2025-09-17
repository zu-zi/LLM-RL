# -*- coding: utf-8 -*-
"""
RL 训练主流程（适配 32GB GPU）：
- 全局超参可被外部 .py 配置文件覆盖（运行时传该文件路径作为第一个参数）
- 移除环境变量依赖；默认值稳健
- 支持 W&B 日志分享（wandb_log=True 时启用）
- sglang 离线补货：子进程生成，完成后释放显存；自动导出 actor 为 HF 目录，并通过 pointer 热重载
"""

import os, sys, time, random, json, subprocess, glob, tempfile, shutil
from datetime import datetime
import numpy as np
import torch, tiktoken
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from RL.PPO import PPOTrainer, Critic
from RL.common import (
    Samples,
    normalize_for_reward,
    forward_values_via_actor,
)

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from utils.rollout_pool import dequeue_items, estimate_size, ensure_dir

# =============== 可被外部 config.py 覆盖的全局 ===============

# 基本运行 & IO
out_dir               = "/root/autodl-tmp/Results"
eval_interval         = 10
max_iters             = 1000
backend               = "nccl"
device                = "cuda"
compile               = False
seed_base             = 1337

# 可选：W&B
wandb_log             = True
wandb_project         = "hlhf"
wandb_run_name        = "ppo_gpt2l_vgpu32"

# 模型结构（从 gpt2-large 加载后会被覆盖为一致）
n_layer = 12; n_head = 12; n_embd = 768
block_size            = 256    # 重要：与 rollout_worker 对齐（prompt 96 + max_new 128 余量）
dropout               = 0.0
bias                  = False
init_from             = "gpt2-large"   # 不再支持 resume

# 训练批次
batch_size                        = 4
gradient_accumulation_steps       = 1   # 当前 Trainer 内部自 step，这里保持 1

# PPO/优化（32GB 稳健默认）
RL_learning_rate      = 1.5e-6
weight_decay          = 5e-3
beta1                 = 0.9
beta2                 = 0.95
kl_ctl_init           = 0.7
max_grad_norm         = 1.0
vf_clip               = 0.2

# 生成/停词（与 rollout 对齐）
SAMPLE_TEMPERATURE    = 0.8
SAMPLE_TOP_P          = 0.9
SAMPLE_TOP_K          = 0
SAMPLE_REP_PENALTY    = 1.1
SAMPLE_STOPS          = ["\nHuman:", "\n\nHuman:"]  # 不拦 "Assistant:"，避免误停
MIN_RESP_TOK          = 16

# sglang 离线池
SGLANG_ON             = True
SGLANG_OFFLINE        = True
SGLANG_MODEL_PATH     = "gpt2-large"          # 初始生成模型（可与 actor 不同）
# —— 热更相关（本脚本自动维护，无需手工改 txt）——
SGLANG_EXPORT_BASE    = "/root/autodl-tmp/actor_exports"   # 导出目录的“父”路径
# SGLANG_POINTER_FILE   = "/root/autodl-tmp/actor_current.txt"  # 指向“当前可用 HF 模型目录”的 pointer 文件
SGLANG_EXPORT_EVERY   = 20                                   # 每多少 iter 导出一次给 sglang
SGLANG_SYNC_DIR       = "/root/autodl-tmp/sgl_pool"
SGLANG_MAX_NEW        = 128                   # 新 token 上限
ROLL_LOW_WATERMARK_FACTOR = 3
SGLANG_ROLLOUT_TARGET = 96
SGLANG_REFILL_BATCH   = 48
ROLL_REFILL_COUNT     = 24
ROLL_COOLDOWN_SEC     = 18
ROLL_MIN_FREE_MB      = 6000
FRESH_RATIO           = 0.50
POOL_STALE_WARN_SEC   = 600
REFRESH_EVERY_BATCHES = 30                    # worker 每处理 N 批检查一次指针

# 奖励模型（EN）
REWARD_MODEL_NAME     = "OpenAssistant/reward-model-deberta-v3-large-v2"

# ===========================================================

def _apply_cli_config_once():
    """若以 `python train_RL_only.py your_cfg.py` 运行，则执行该 py 覆盖全局变量。"""
    is_main = (os.environ.get("RANK","-1") == "-1") or (os.environ.get("LOCAL_RANK") in (None,"0"))
    if is_main and len(sys.argv)>=2 and sys.argv[1].endswith(".py"):
        cfg=sys.argv[1]
        print(f"[config] Overriding with {cfg}:\n{'-'*60}\n{open(cfg,'r').read()}\n{'-'*60}")
        exec(open(cfg,"r").read(), globals(), globals())

# ========= 简单 GPT-2 分词器 =========
class GPT2Tok:
    def __init__(self):
        enc=tiktoken.get_encoding("gpt2")
        self.enc=enc
        self.eos_token="<|endoftext|>"
        self.eos_id=enc.encode(self.eos_token, allowed_special={self.eos_token})[0]
        self.pad_token_id = 0
        self.eos_token_id = self.eos_id
    def encode(self,s): return self.enc.encode(s, allowed_special="all")
    def decode(self,ids):
        if torch.is_tensor(ids): ids=ids.tolist()
        return self.enc.decode(ids)

# ========= 数据/固定 prompts =========
def _load_prompts():
    pf=os.path.join(os.path.dirname(__file__),"data/RL_dataset/prompt.bin")
    print(f"Loading fixed prompts from {pf} ...")
    blob=torch.load(pf,map_location="cpu")
    return (
        blob["prompts"],
        blob["gpt2_token_ids"],
        blob["eos_id"],
        blob.get("seed",seed_base),
        pf,
        blob.get("train_indices", None),
        blob.get("eval_indices", None),
    )

def pack_samples(gen_list, pad_id, block_size, device):
    if not gen_list:
        L=1; z=torch.zeros((0,L),dtype=torch.long,device=device)
        return Samples(seqs=z, attention_mask=z, action_mask=z,
                       num_actions=torch.zeros(0,dtype=torch.long,device=device),
                       response_length=torch.zeros(0,dtype=torch.long,device=device),
                       total_length=torch.zeros(0,dtype=torch.long,device=device))
    L=min(block_size, max(len(x["full_ids"]) for x in gen_list))
    B=len(gen_list)
    seqs=torch.full((B,L), pad_id, dtype=torch.long, device=device)
    attn=torch.zeros((B,L),dtype=torch.long,device=device)
    amsk=torch.zeros((B,L),dtype=torch.long,device=device)
    num_actions=torch.zeros(B,dtype=torch.long,device=device)
    resp_len=torch.zeros(B,dtype=torch.long,device=device)
    total_len=torch.zeros(B,dtype=torch.long,device=device)
    for i,it in enumerate(gen_list):
        full=torch.tensor(it["full_ids"][:L],dtype=torch.long,device=device)
        p_len=min(len(it["prompt_ids"]),L)
        t=full.numel()
        seqs[i,:t]=full; attn[i,:t]=1; total_len[i]=t
        if p_len<t:
            amsk[i,p_len:t]=1
            na=int((amsk[i]==1).sum().item())
            num_actions[i]=na; resp_len[i]=na
    return Samples(seqs=seqs, attention_mask=attn, action_mask=amsk,
                   num_actions=num_actions, response_length=resp_len, total_length=total_len)

# ========= 采样解码（与 rollout 完全对齐）=========
@torch.no_grad()
def _decode_with_sampling(model, idx, max_new_tokens, eos_id, block_size,
                          temperature=0.8, top_p=0.9, top_k=0, repetition_penalty=1.1,
                          stop_strs=None, tokenizer_decode=None, min_resp: int = 8):
    device = idx.device
    was_train = model.training
    model.eval()
    try:
        out = idx
        start_len = out.size(1)
        for _ in range(int(max_new_tokens)):
            idx_cond = out[:, -int(block_size):]
            logits = model(idx_cond)
            if isinstance(logits, tuple):
                logits = logits[0]
            last = logits[:, -1, :]

            if repetition_penalty and out.numel() > 0:
                uniq = torch.unique(out); last[:, uniq] = last[:, uniq] / float(repetition_penalty)
            last = last / max(float(temperature), 1e-6)
            if top_k and top_k > 0:
                kth = torch.topk(last, k=min(int(top_k), last.size(-1)), dim=-1).values[..., -1:]
                last = torch.where(last < kth, torch.full_like(last, -1e10), last)
            probs = torch.softmax(last, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            cutoff = (cumsum > float(top_p)).float().argmax(dim=-1, keepdim=True)
            mask = torch.arange(probs.size(-1), device=probs.device).view(1, -1) <= cutoff
            kept = torch.where(mask, sorted_probs, torch.zeros_like(sorted_probs))
            kept = kept / kept.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            next_sorted = torch.multinomial(kept, num_samples=1)
            next_id = sorted_idx.gather(1, next_sorted)

            if (out.size(1) - start_len) < int(min_resp) and eos_id is not None and int(next_id.item()) == int(eos_id):
                alt = sorted_idx[:, 1:2] if sorted_idx.size(1) > 1 else next_id
                next_id = alt

            out = torch.cat((out, next_id.to(device)), dim=1)

            if (out.size(1) - start_len) >= int(min_resp):
                if eos_id is not None and int(next_id.item()) == int(eos_id):
                    break
                if stop_strs and tokenizer_decode is not None:
                    tail = tokenizer_decode(out[0][-min(out.size(1), block_size):].tolist())
                    if any(s in tail for s in stop_strs):
                        break
        return out
    finally:
        if was_train: model.train()

# ========= 显存/新鲜度辅助 =========
def _cuda_free_mb(device_str="cuda"):
    try:
        if not torch.cuda.is_available():
            return 0
        free, total = torch.cuda.mem_get_info(torch.device(device_str))
        return int(free // (1024 * 1024))
    except Exception:
        return 0

def _pool_freshness(dir_path: str, take_last:int=50):
    try:
        fs = sorted(glob.glob(os.path.join(dir_path, "roll_*.jsonl")), key=os.path.getmtime)
        if not fs: return 0, (float("inf"), float("inf"))
        fs = fs[-min(len(fs), max(1,int(take_last))):]
        ages = [time.time() - os.path.getmtime(p) for p in fs]
        med = float(np.median(ages)); p90 = float(np.percentile(ages, 90))
        return len(fs), (med, p90)
    except Exception:
        return 0, (float("inf"), (float("inf")))

# ========= sglang 导出 + pointer 原子切换 =========
def _atomic_write_text(path: str, text: str):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def export_actor_for_sglang(raw_actor, init_from, export_base: str, symlink_path: str):
    """
    将当前 actor 导出为 HF 目录，并“原子切换” symlink 到该目录。
    symlink_path = /root/autodl-tmp/actor_exports/current
    """
    os.makedirs(export_base, exist_ok=True)

    # 1) 准备骨架，与 init_from 对齐
    hf_model = AutoModelForCausalLM.from_pretrained(init_from, torch_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(init_from, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    # 2) 拷贝参数
    sd_src = raw_actor.state_dict()
    sd_tgt = hf_model.state_dict()
    matched, skipped = 0, 0
    for k in sd_tgt.keys():
        if k in sd_src and sd_tgt[k].shape == sd_src[k].shape:
            sd_tgt[k].copy_(sd_src[k].to(sd_tgt[k].dtype))
            matched += 1
        else:
            skipped += 1
    hf_model.eval()

    # 3) 导出到新路径
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(export_base, f"ts_{ts}_{random.randint(1000,9999)}")
    os.makedirs(out_dir, exist_ok=True)
    hf_model.save_pretrained(out_dir, safe_serialization=True, max_shard_size="2GB")
    tok.save_pretrained(out_dir)

    # 4) 原子切换符号链接 symlink_path -> out_dir
    symlink_tmp = symlink_path + ".tmp"
    try:
        if os.path.lexists(symlink_tmp):
            os.remove(symlink_tmp)
    except Exception:
        pass
    # 若目标已存在且是目录，先移除（或重命名备份）
    if os.path.exists(symlink_path) and not os.path.islink(symlink_path):
        shutil.rmtree(symlink_path)  # 已 import shutil
    os.symlink(out_dir, symlink_tmp)
    os.replace(symlink_tmp, symlink_path)

    print(f"[export] sglang HF model updated. matched={matched}, skipped={skipped}, path={out_dir}", flush=True)
    def _prune_exports(base, keep_last=2):
        import os, glob
        ds = sorted([d for d in glob.glob(os.path.join(base, "ts_*")) if os.path.isdir(d)])
        for d in ds[:-keep_last]:
            try:
                import shutil; shutil.rmtree(d)
                print(f"[export][gc] remove old export: {d}")
            except Exception as e:
                print(f"[export][gc][warn] {d}: {e}")
    
    # 在 export_actor_for_sglang(...) 成功后调一次：
    _prune_exports(export_base, keep_last=1)
    return out_dir


# ========= sglang 离线补货 =========
def _spawn_rollout_subprocess(prompt_bin_path, count, sync_dir, max_new, rollout_log_dir, quiet=True):
    ensure_dir(sync_dir)
    os.makedirs(rollout_log_dir, exist_ok=True)
    cmd = [
        "python", "-u", "rollout_worker.py",
        "--model", SGLANG_MODEL_PATH,
        "--prompt-bin", prompt_bin_path,
        "--out-dir", sync_dir,
        "--count", str(int(count)),
        "--max-new", str(int(max_new)),
        "--block-size", str(int(block_size)),
        "--mb", "6",
        "--use-only-train",
        "--min-resp","16",
        "--refresh-every-batches", str(int(REFRESH_EVERY_BATCHES)),
        "--reload-strategy", "realpath",
        "--min-free-mb", str(int(ROLL_MIN_FREE_MB)),  # ★ 加这一行
    ]

    logf = os.path.join(rollout_log_dir, f"rollout_{int(time.time())}.log")
    if quiet:
        print(f"[rollout] spawn (quiet, cuda) -> {logf}")
        with open(logf, "a") as f:
            ret = subprocess.call(cmd, stdout=f, stderr=f)
    else:
        print(f"[rollout] spawning: {' '.join(cmd)}")
        ret = subprocess.call(cmd)
    if ret != 0:
        raise RuntimeError(f"rollout_worker exit with code {ret}")


def _is_master(ddp): return (not ddp) or (os.environ.get("RANK","0") == "0")

def _ema_update(prev, x, alpha=0.1):
    if prev is None or np.isnan(prev): return float(x)
    return float(alpha * x + (1.0 - alpha) * prev)

# ========= 评测（采样口径，与 rollout 一致）=========
@torch.no_grad()
def eval_fixed_raw_reward_sampled(actor_model, gpt2_tok, eval_prompt_ids,
                                  reward_tokenizer, reward_model,
                                  block_size, max_new_eval=64):
    dev = next(actor_model.parameters()).device
    eos_id = gpt2_tok.eos_id
    texts = []
    was_training = actor_model.training
    actor_model.eval()
    try:
        for ids in eval_prompt_ids:
            idx = torch.tensor(ids, dtype=torch.long, device=dev).unsqueeze(0)
            room = block_size - idx.size(1) - 1
            if room <= 0:
                full_ids = ids[:block_size]
                texts.append(gpt2_tok.decode(full_ids)); continue
            gen_len = max(8, min(int(max_new_eval), int(room)))
            out = _decode_with_sampling(
                actor_model, idx, gen_len, eos_id, block_size,
                temperature=SAMPLE_TEMPERATURE, top_p=SAMPLE_TOP_P, top_k=SAMPLE_TOP_K,
                repetition_penalty=SAMPLE_REP_PENALTY,
                stop_strs=SAMPLE_STOPS, tokenizer_decode=gpt2_tok.decode,
                min_resp=MIN_RESP_TOK
            )
            full = out[0].tolist()[:block_size]
            texts.append(gpt2_tok.decode(full))
    finally:
        if was_training: actor_model.train()

    texts = [normalize_for_reward(t, reward_tokenizer) for t in texts]
    toks = reward_tokenizer(texts, padding=True, truncation=True, max_length=1024, return_tensors="pt")
    outs = reward_model(**toks)
    logits = getattr(outs, "logits", None)
    if logits is None: return float("nan")
    if logits.dim() == 2 and logits.size(-1) == 1: logits = logits.squeeze(-1)
    return float(logits.mean().item()) if len(texts) > 0 else float("nan")

@torch.no_grad()
def eval_fixed_raw_reward_sampled(actor_model, gpt2_tok, eval_prompt_ids,
                                  reward_tokenizer, reward_model,
                                  block_size, max_new_eval=64,
                                  mode: str = "greedy",   # "greedy" or "sample"
                                  seeds: int = 1):        # mode=="sample" 时可>1
    dev = next(actor_model.parameters()).device
    eos_id = gpt2_tok.eos_id
    agg = []

    def _set_all_seeds(base):
        import random, numpy as np, torch
        random.seed(base); np.random.seed(base); torch.manual_seed(base)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(base)

    for rep in range(max(1, seeds)):
        if mode == "sample":
            _set_all_seeds(1337 + rep)

        texts = []
        was_training = actor_model.training
        actor_model.eval()
        try:
            for ids in eval_prompt_ids:
                idx = torch.tensor(ids, dtype=torch.long, device=dev).unsqueeze(0)
                room = block_size - idx.size(1) - 1
                if room <= 0:
                    texts.append(gpt2_tok.decode(ids[:block_size])); continue
                gen_len = max(8, min(int(max_new_eval), int(room)))

                if mode == "greedy":
                    # 贪心：温度=0，等价于取最大概率 token
                    out = _decode_with_sampling(
                        actor_model, idx, gen_len, eos_id, block_size,
                        temperature=1e-6, top_p=1.0, top_k=0,  # 实现“近似贪心”
                        repetition_penalty=1.0,                # 评测不引入惩罚
                        stop_strs=SAMPLE_STOPS, tokenizer_decode=gpt2_tok.decode,
                        min_resp=MIN_RESP_TOK
                    )
                else:
                    out = _decode_with_sampling(
                        actor_model, idx, gen_len, eos_id, block_size,
                        temperature=SAMPLE_TEMPERATURE, top_p=SAMPLE_TOP_P, top_k=SAMPLE_TOP_K,
                        repetition_penalty=SAMPLE_REP_PENALTY,
                        stop_strs=SAMPLE_STOPS, tokenizer_decode=gpt2_tok.decode,
                        min_resp=MIN_RESP_TOK
                    )

                full = out[0].tolist()[:block_size]
                texts.append(gpt2_tok.decode(full))
        finally:
            if was_training: actor_model.train()

        texts = [normalize_for_reward(t, reward_tokenizer) for t in texts]
        toks = reward_tokenizer(texts, padding=True, truncation=True, max_length=1024, return_tensors="pt")
        outs = reward_model(**toks)
        logits = getattr(outs, "logits", None)
        if logits is None:
            agg.append(float("nan"))
        else:
            if logits.dim() == 2 and logits.size(-1) == 1: logits = logits.squeeze(-1)
            agg.append(float(logits.mean().item()))

    # 返回均值（若 seeds>1 可再记录方差）
    return float(np.mean(agg)) if len(agg) > 0 else float("nan")


# ============================= 主流程 =============================
def main():
    global wandb_log
    _apply_cli_config_once()

    # 目录/日志
    if _is_master(False):
        os.makedirs(out_dir, exist_ok=True)
    METRICS_CSV    = os.path.join(out_dir, "metrics.csv")
    ROLLOUT_LOG_DIR= os.path.join(out_dir, "rollout_logs")
    ensure_dir(SGLANG_SYNC_DIR); os.makedirs(ROLLOUT_LOG_DIR, exist_ok=True)

    # DDP
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend=backend)
        rank=int(os.environ['RANK']); local=int(os.environ['LOCAL_RANK']); world=int(os.environ['WORLD_SIZE'])
        dev=f'cuda:{local}'; torch.cuda.set_device(dev); master=(rank==0)
        assert gradient_accumulation_steps % world == 0
    else:
        dev=device; master=True

    # 随机种
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    torch.manual_seed(seed_base + (int(os.environ.get("RANK","0")) if ddp else 0))

    # prompts
    tok = GPT2Tok()
    (
        PROMPTS_TEXT, PROMPT_TOKEN_IDS, EOS_ID, data_seed, PROMPT_BIN_PATH,
        TRAIN_INDICES, EVAL_INDICES
    ) = _load_prompts()
    pad_id = tok.eos_id

    ALL_IDX = list(range(len(PROMPT_TOKEN_IDS)))
    if isinstance(TRAIN_INDICES, list) and TRAIN_INDICES:
        TRAIN_IDX = sorted(i for i in TRAIN_INDICES if 0 <= i < len(PROMPT_TOKEN_IDS))
    else:
        if isinstance(EVAL_INDICES, list) and EVAL_INDICES:
            EVAL_SET = set(i for i in EVAL_INDICES if 0 <= i < len(PROMPT_TOKEN_IDS))
            TRAIN_IDX = [i for i in ALL_IDX if i not in EVAL_SET]
        else:
            TRAIN_IDX = ALL_IDX[:]
    TRAIN_PROMPT_IDS = [PROMPT_TOKEN_IDS[i] for i in TRAIN_IDX]

    if isinstance(EVAL_INDICES, list) and EVAL_INDICES:
        EVAL_PROMPT_IDS = [PROMPT_TOKEN_IDS[i] for i in EVAL_INDICES if 0 <= i < len(PROMPT_TOKEN_IDS)]
    else:
        rng = np.random.RandomState(int(data_seed))
        eval_count = min(16, len(PROMPT_TOKEN_IDS))
        choice = rng.choice(len(PROMPT_TOKEN_IDS), size=eval_count, replace=False) if eval_count>0 else []
        EVAL_PROMPT_IDS = [PROMPT_TOKEN_IDS[int(i)] for i in choice]

    # 初始化 actor/ref/critic
    from model import GPT, GPTConfig
    model_args=dict(n_layer=n_layer,n_head=n_head,n_embd=n_embd,block_size=block_size,bias=bias,dropout=dropout,vocab_size=None)

    if isinstance(init_from,str) and init_from.startswith("gpt2"):
        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
        m=GPT.from_pretrained(init_from, dict(dropout=dropout))
        for k in ['n_layer','n_head','n_embd','block_size','bias','vocab_size']: model_args[k]=getattr(m.config,k)
        if block_size < model_args['block_size']:
            m.crop_block_size(block_size); model_args['block_size']=block_size
        base_state=m.state_dict(); del m
    else:
        print("Initializing a new model from scratch")
        model_args['vocab_size']=50304
        m=GPT(GPTConfig(**model_args)); base_state=m.state_dict(); del m

    ref=GPT(GPTConfig(**model_args)).to(dev); ref.load_state_dict(base_state)
    for p in ref.parameters(): p.requires_grad=False
    ref.eval()
    ref_dtype=torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    ref=ref.to(device=dev, dtype=ref_dtype)

    actor=GPT(GPTConfig(**model_args)).to(dev); actor.load_state_dict(base_state)
    if compile: actor=torch.compile(actor)
    if ddp: actor=DDP(actor, device_ids=[int(str(dev).split(':')[-1])])
    raw_actor=actor.module if ddp else actor

    critic=Critic(raw_actor).to(dev)

    # —— 初始引导导出（让 symlink 有东西，worker 才能正常加载）——
    if SGLANG_ON and SGLANG_OFFLINE and (os.environ.get("RANK","0") == "0"):
        try:
            os.makedirs(os.path.dirname(SGLANG_MODEL_PATH), exist_ok=True)
            export_actor_for_sglang(raw_actor, init_from, SGLANG_EXPORT_BASE, SGLANG_MODEL_PATH)
        except Exception as e:
            print(f"[export][warn] initial export failed: {e}", flush=True)

    # 优化器（8bit 优先）
    lr_c = max(2e-6, RL_learning_rate * 1.5)
    weight_decay_c = 1e-3
    try:
        from bitsandbytes.optim import AdamW8bit
        opt_a=AdamW8bit(raw_actor.parameters(), lr=RL_learning_rate, betas=(beta1,beta2), weight_decay=weight_decay)
        opt_c=AdamW8bit(critic.parameters(),    lr=lr_c,          betas=(beta1,beta2), weight_decay=weight_decay_c)
        print("[optim] using bitsandbytes AdamW8bit")
    except Exception as e:
        print(f"[optim] bitsandbytes not available ({e}), fallback to torch.optim.AdamW")
        opt_a=torch.optim.AdamW(raw_actor.parameters(), lr=RL_learning_rate, betas=(beta1,beta2), weight_decay=weight_decay)
        opt_c=torch.optim.AdamW(critic.parameters(),    lr=lr_c,          betas=(beta1,beta2), weight_decay=weight_decay_c)

    # 奖励模型（CPU）
    print(f"[reward] loading {REWARD_MODEL_NAME} on CPU ...")
    reward_hf = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL_NAME, device_map="cpu", torch_dtype=torch.float32).eval()
    reward_model = reward_hf.eval()  # 修复：不再使用错误的下标语法
    reward_tokenizer=AutoTokenizer.from_pretrained(REWARD_MODEL_NAME, use_fast=True)
    try: reward_tokenizer.padding_side = "right"
    except Exception: pass
    if getattr(reward_tokenizer, "pad_token", None) is None and getattr(reward_tokenizer, "eos_token", None) is not None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token

    # PPO 训练器
    trainer=PPOTrainer(
        actor_model=raw_actor, ref_model=ref, reward_model=reward_model,
        critic_model=critic, actor_tokenizer=tok, reward_tokenizer=reward_tokenizer,
        optimizer_actor=opt_a, optimizer_critic=opt_c,
        device=dev,
        mb_size_logits=1,
        mb_size_values=1,
        kl_ctl=kl_ctl_init,
        max_grad_norm=max_grad_norm,
        vf_clip=vf_clip,
        ppo_clip=ppo_clip,
        entropy_coef=entropy_coef,
    )

    # W&B
    if wandb_log:
        try:
            import wandb, json
            os.environ["WANDB_MODE"] = "offline"   # 离线缓存
            os.environ["WANDB_SILENT"] = "true"
    
            # 组装“可序列化”的扁平配置（把 list/tuple 等先 JSON 化成字符串，避免被当嵌套 dict 合并）
            cfg_for_log = {}
            white = {
                "out_dir","eval_interval","max_iters","seed_base","compile","backend","device",
                "wandb_log","wandb_project","wandb_run_name",
                "init_from","block_size","bias","dropout",
                "batch_size","gradient_accumulation_steps",
                "RL_learning_rate","weight_decay","beta1","beta2","max_grad_norm","vf_clip",
                "kl_ctl_init","ppo_clip","entropy_coef",
                "SAMPLE_TEMPERATURE","SAMPLE_TOP_P","SAMPLE_TOP_K","SAMPLE_REP_PENALTY","SAMPLE_STOPS","MIN_RESP_TOK",
                "SGLANG_ON","SGLANG_OFFLINE","SGLANG_MODEL_PATH","SGLANG_SYNC_DIR",
                "SGLANG_ROLLOUT_TARGET","SGLANG_REFILL_BATCH","SGLANG_MAX_NEW",
                "ROLL_LOW_WATERMARK_FACTOR","ROLL_REFILL_COUNT","ROLL_COOLDOWN_SEC","ROLL_MIN_FREE_MB",
                "REFRESH_EVERY_BATCHES","FRESH_RATIO",
                "REWARD_MODEL_NAME"
            }
            for k, v in globals().items():
                if k in white:
                    try:
                        json.dumps(v)
                        cfg_for_log[k] = v
                    except Exception:
                        cfg_for_log[k] = json.dumps(v, ensure_ascii=False)
    
            # 关键：把 config 直接传给 init，而不是再调 config.update
            wandb.init(project=wandb_project, name=wandb_run_name, dir=out_dir, config=cfg_for_log)
        except Exception as e:
            print(f"[wandb] disabled: {e}")
            wandb_log = False



    # ====== sglang 池启用 ======
    if master and SGLANG_ON and SGLANG_OFFLINE:
        print(f"[sglang] offline pool: dir={SGLANG_SYNC_DIR} target={SGLANG_ROLLOUT_TARGET} batch={SGLANG_REFILL_BATCH}")

    r_raw_ema = None
    last_rollout_t = 0.0

    # KL 迟滞控制
    KL_HALT, KL_RESUME = 0.12, 0.08
    HALT_STREAK, RESUME_STREAK = 2, 2
    actor_frozen = False; halt_hits = 0; resume_hits = 0; freeze_steps = 0
    FORCE_FRESH_STEPS = 7
    EMERG_SHORT_GEN_STEPS = 4
    BASE_LR_ACTOR = RL_learning_rate

    for iter_num in range(1, max_iters+1):
        # ---- 池子估算 + 新鲜度 ----
        pool_est = estimate_size(SGLANG_SYNC_DIR, SGLANG_REFILL_BATCH) if (SGLANG_ON and SGLANG_OFFLINE) else -1
        n_files, (age_med, age_p90) = _pool_freshness(SGLANG_SYNC_DIR) if (SGLANG_ON and SGLANG_OFFLINE) else (0,(float("inf"),float("inf")))

        # ---- 离线补货（主进程）----
        if master and SGLANG_ON and SGLANG_OFFLINE:
            LOW_WATERMARK = max(batch_size * ROLL_LOW_WATERMARK_FACTOR, 8)
            if pool_est < LOW_WATERMARK:
                now = time.time()
                cool_ok = (now - last_rollout_t) >= ROLL_COOLDOWN_SEC
                free_mb = _cuda_free_mb(device_str=device)
                mem_ok = free_mb >= ROLL_MIN_FREE_MB
                if cool_ok and mem_ok:
                    refill = min(ROLL_REFILL_COUNT, SGLANG_REFILL_BATCH)
                    try:
                        _spawn_rollout_subprocess(PROMPT_BIN_PATH, refill, SGLANG_SYNC_DIR, SGLANG_MAX_NEW, ROLLOUT_LOG_DIR, quiet=True)
                        last_rollout_t = now
                    except Exception as e:
                        print(f"[rollout][skip] spawn failed: {e}", flush=True)
                else:
                    why=[]
                    if not cool_ok: why.append("cooldown")
                    if not mem_ok:  why.append(f"free_mem={free_mb}MB<{ROLL_MIN_FREE_MB}MB")
                    print(f"[rollout][skip] pool={pool_est} reasons={'+'.join(why)}", flush=True)

        # ---- 取样：保证“新鲜样本占比” ----
        want_fresh_base = max(1, int(np.ceil(FRESH_RATIO * batch_size)))
        force_fresh = (FORCE_FRESH_STEPS > 0)
        if actor_frozen or force_fresh:
            want_fresh = batch_size; batch = []
            if force_fresh: FORCE_FRESH_STEPS -= 1
        else:
            want_fresh = want_fresh_base
            batch = dequeue_items(SGLANG_SYNC_DIR, batch_size - want_fresh) if (SGLANG_ON and SGLANG_OFFLINE) else []

        # 在线补齐
        def _gen_one_from_train(use_ref: bool):
            ids = random.choice(TRAIN_PROMPT_IDS)
            ids_t = torch.tensor(ids, dtype=torch.long, device=dev).unsqueeze(0)
            room = block_size - ids_t.size(1) - 1
            if room <= 0: return None
            max_new_cap = SGLANG_MAX_NEW
            if use_ref or EMERG_SHORT_GEN_STEPS > 0: max_new_cap = min(max_new_cap, 32)
            out = _decode_with_sampling(
                ref if use_ref else raw_actor, ids_t,
                max_new_tokens=max(8, min(room, max_new_cap)),
                eos_id=tok.eos_id, block_size=block_size,
                temperature=SAMPLE_TEMPERATURE, top_p=SAMPLE_TOP_P, top_k=SAMPLE_TOP_K,
                repetition_penalty=SAMPLE_REP_PENALTY,
                stop_strs=SAMPLE_STOPS, tokenizer_decode=tok.decode,
                min_resp=MIN_RESP_TOK
            )
            resp_len = out.size(1) - ids_t.size(1)
            if resp_len >= MIN_RESP_TOK:
                return {"prompt_ids": ids, "full_ids": out[0].tolist()}
            return None

        fresh = []
        for _ in range(want_fresh):
            g = _gen_one_from_train(use_ref=actor_frozen)
            if g is not None: fresh.append(g)
        batch.extend(fresh)

        while len(batch) < batch_size:
            g = _gen_one_from_train(use_ref=actor_frozen)
            if g is None: break
            batch.append(g)

        # 打包
        samples = pack_samples(batch, pad_id=tok.eos_id, block_size=block_size, device=torch.device(dev))

        # 若整批没有 response，跳过
        if int(samples.action_mask.sum().item()) == 0:
            if master: print(f"[iter {iter_num:4d}] skip(empty-response-batch) pool={pool_est}", flush=True)
            continue

        # 评估经验（不更新参数）
        experiences, report_kl, r_raw, r_shaped, r_ctr, safe_kl = trainer.evaluate_experience(samples)

        # —— 安全保险丝 —— #
        critic_only_this_iter = False
        if (not np.isfinite(safe_kl)) or (safe_kl > 1.7):
            actor_frozen = True; critic_only_this_iter = True
            FORCE_FRESH_STEPS = max(FORCE_FRESH_STEPS, 3)
            EMERG_SHORT_GEN_STEPS = max(EMERG_SHORT_GEN_STEPS, 3)
            if master: print(f"[guard] skip actor (abnormal KL={report_kl:.4g}) -> critic-only", flush=True)

        stats_last = getattr(trainer, "last_stats", {}) or {}
        clip_frac    = float(stats_last.get("clip_frac", 0.0) or 0.0)
        approx_kl_pi = float(stats_last.get("approx_kl_pi", float("nan")))
        entropy_tok  = float(stats_last.get("entropy", float("nan")))
        v_mae        = float(stats_last.get("v_mae", float("nan")))
        explained_var= float(stats_last.get("explained_var", float("nan")))
        ratio_qs     = stats_last.get("ratio_q50_q90_q99", (float("nan"),)*3)
        ratio_max    = float(stats_last.get("ratio_max",  float("nan")))
        adv_abs_m    = float(stats_last.get("adv_abs_mean", float("nan")))
        sel_tokens   = int(stats_last.get("sel_tokens", 0))
        ppo_clip_v   = float(stats_last.get("ppo_clip", float("nan")))
        kl_ctl_now   = float(stats_last.get("kl_ctl_now", float("nan")))

         # 当 report_kl 或 clip_frac 异常时，临时只训 critic 2 个 iter
        if (np.isfinite(report_kl) and report_kl > 0.80) or (clip_frac > 0.40):
            actor_frozen = True
            critic_only_this_iter = True
            FORCE_FRESH_STEPS = max(FORCE_FRESH_STEPS, 4)

        if np.isfinite(safe_kl) and safe_kl > 1.7:
            actor_frozen = True
            FORCE_FRESH_STEPS = max(FORCE_FRESH_STEPS, 3)
            EMERG_SHORT_GEN_STEPS = max(EMERG_SHORT_GEN_STEPS, 3)
            trainer.kl_ctl = float(min(2.0, trainer.kl_ctl * 1.2))
            for g in opt_a.param_groups:
                g['lr'] = max(BASE_LR_ACTOR * 0.25, 8e-7)

        if EMERG_SHORT_GEN_STEPS > 0: EMERG_SHORT_GEN_STEPS -= 1

        # 自适应 KL
        kl_target = 0.2
        if np.isfinite(safe_kl):
            err = safe_kl / max(kl_target, 1e-8) - 1.0
            up = 0.05; down = 0.03
            trainer.kl_ctl *= float(np.exp(np.clip(err, -down, up)))
            trainer.kl_ctl = float(np.clip(trainer.kl_ctl, 0.25, 2.0))

        r_raw_ema = _ema_update(r_raw_ema, r_raw, alpha=0.1)

        # 临时降 LR（actor）
        KL_STOP = 1
        if np.isfinite(safe_kl):
            err = max(0.0, safe_kl / KL_STOP - 1.0)
            scale = 1.0 / (1.0 + 2.5 * err)
            scale = float(max(0.2, min(1.0, scale)))
        else:
            scale = 0.25
        for g in opt_a.param_groups:
            g['lr'] = BASE_LR_ACTOR * scale

        # 迟滞冻结/解冻
        if np.isfinite(safe_kl):
            if safe_kl > KL_HALT:  halt_hits += 1; resume_hits = 0
            elif safe_kl < KL_RESUME: resume_hits += 1; halt_hits = 0
            else: halt_hits = 0; resume_hits = 0

            if (not actor_frozen) and (halt_hits >= HALT_STREAK):
                actor_frozen = True; halt_hits = 0
                if master: print(f"[guard] freeze actor (kl={report_kl:.3f})", flush=True)
            if actor_frozen and (resume_hits >= RESUME_STREAK):
                actor_frozen = False; resume_hits = 0; freeze_steps = 0
                if master: print(f"[guard] unfreeze actor (kl={report_kl:.3f})", flush=True)

        if actor_frozen:
            freeze_steps += 1
            if freeze_steps >= 8:
                trainer.kl_ctl = float(min(1.2, trainer.kl_ctl * 1.2))
                for g in opt_a.param_groups:
                    g['lr'] = max(BASE_LR_ACTOR * 0.4, 6e-7)
                if master:
                    print(f"[guard] long-freeze fallback: kl_ctl={trainer.kl_ctl:.3f} actor_lr={opt_a.param_groups[0]['lr']:.2e}", flush=True)
        else:
            freeze_steps = 0

        # === 训练 ===
        if not experiences:
            continue

        pl, vl = [], []
        if actor_frozen or critic_only_this_iter:
            # 只训 critic
            for exp in experiences:
                critic.train(); opt_c.zero_grad(set_to_none=True)
                values_full = forward_values_via_actor(
                    trainer.actor, trainer.critic, exp.seqs, trainer.device_type,
                    ptdtype=None, micro_batch_size=getattr(trainer, "mb_values", 1), detach_hidden=True
                )
                values_new = values_full[:, 1:]
                v_loss_tok = torch.nn.functional.huber_loss(values_new, exp.returns, delta=1.0, reduction='none')
                v_loss = (v_loss_tok * exp.action_mask).sum() / exp.action_mask.sum().clamp_min(1e-8)
                v_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainer.critic.parameters(), 0.5)
                trainer.opt_critic.step()
                vl.append(float(v_loss.detach().item()))
            mean_p = 0.0; mean_v = float(np.mean(vl)) if vl else None
        else:
            # 正常 PPO（actor + critic）
            KL_EPOCH_DROP = 1.10
            LOW_EPOCH, HIGH_EPOCH = 1, 2
            POLICY_EPOCHS = LOW_EPOCH if (np.isfinite(safe_kl) and safe_kl > KL_EPOCH_DROP) else HIGH_EPOCH
            if (iter_num > 5) and np.isfinite(safe_kl) and (safe_kl < 0.25) and (clip_frac < 0.01):
                POLICY_EPOCHS = 3
                for g in opt_a.param_groups:
                    g['lr'] = min(BASE_LR_ACTOR * 1.2, 5e-6)
            for _ in range(POLICY_EPOCHS):
                for exp in experiences:
                    out = trainer.train_on_experience(exp, use_token_entropy=False)
                    if isinstance(out, tuple):
                        p, v = out; pl.append(float(p.detach().item())); vl.append(float(v.detach().item()))
                    else:
                        pl.append(float(out.detach().item()))
            mean_p = float(np.mean(pl)) if pl else 0.0
            mean_v = float(np.mean(vl)) if vl else None

        # 评测
        r_eval_greedy = float("nan")
        if (iter_num % eval_interval == 0) and master and len(EVAL_PROMPT_IDS) > 0:
            try:
                r_eval_greedy = eval_fixed_raw_reward_sampled(
                    actor_model=raw_actor, gpt2_tok=tok, eval_prompt_ids=EVAL_PROMPT_IDS,
                    reward_tokenizer=reward_tokenizer, reward_model=reward_model,
                    block_size=block_size, max_new_eval=min(SGLANG_MAX_NEW, 64),
                    mode="greedy", seeds=1
                )
            except Exception:
                r_eval_greedy = float("nan")

        # ===== 日志输出（控制台 / CSV / W&B）=====
        # response 长度统计
        resp_lengths = samples.response_length.detach().cpu().numpy().tolist()
        resp_len_stats = {
            "resp_len_min": int(np.min(resp_lengths)) if resp_lengths else 0,
            "resp_len_p50": float(np.percentile(resp_lengths, 50)) if resp_lengths else 0.0,
            "resp_len_p90": float(np.percentile(resp_lengths, 90)) if resp_lengths else 0.0,
            "resp_len_max": int(np.max(resp_lengths)) if resp_lengths else 0,
        }

        if (iter_num % eval_interval == 0) and master:
            cur_lr = opt_a.param_groups[0]['lr']
            stale_note = ""
            if age_med != float("inf"):
                stale_note = f" age_med={age_med:.0f}s p90={age_p90:.0f}s"
                if age_med > POOL_STALE_WARN_SEC: stale_note += "(!stale)"
            core = (
                f"[iter {iter_num:4d}] p={mean_p:.4f} " + (f"v={mean_v:.4f} " if mean_v is not None else "") +
                f"kl={report_kl:.6f} safe_kl={safe_kl:.6f} r_raw={r_raw:.4f} r_ema={r_raw_ema:.4f} "
                f"r_ctr={r_ctr:.4f} r_shp={r_shaped:.4f} r_eval_greedy={r_eval_greedy:.4f} "
                f"clip={clip_frac:.3f} akl_pi={approx_kl_pi:.4f} H={entropy_tok:.3f} "
                f"v_mae={v_mae:.4f} ev={explained_var:.3f} pool={pool_est} "
                f"| sel_tok={sel_tokens} adv|={adv_abs_m:.3e} "
                f"rΔ q50/90/99={ratio_qs[0]:.3e}/{ratio_qs[1]:.3e}/{ratio_qs[2]:.3e} "
                f"rΔmax={ratio_max:.3e} clip_th={ppo_clip_v:.3f} kl_ctl={kl_ctl_now:.3f} "
                f"lr={cur_lr:.2e} "
                f"resp_len[min/p50/p90/max]={resp_len_stats['resp_len_min']}/{resp_len_stats['resp_len_p50']:.1f}/{resp_len_stats['resp_len_p90']:.1f}/{resp_len_stats['resp_len_max']} "
                f"{stale_note}"
            )
            print(core, flush=True)

            hdr = not os.path.exists(METRICS_CSV)
            with open(METRICS_CSV, "a") as f:
                if hdr:
                    f.write(
                        "iter,p_loss,v_loss,avg_kl,safe_kl,r_raw,r_raw_ema,r_ctr,r_shp,r_eval_greedy,"
                        "clip_frac,approx_kl_pi,entropy,v_mae,explained_var,pool_est,age_med,age_p90,lr,"
                        "resp_min,resp_p50,resp_p90,resp_max\n"
                    )
                f.write(
                    f"{iter_num},{mean_p},{'' if mean_v is None else mean_v},"
                    f"{report_kl},{safe_kl},{r_raw},{r_raw_ema},{r_ctr},{r_shaped},{r_eval_greedy},"
                    f"{clip_frac},{approx_kl_pi},{entropy_tok},{v_mae},{explained_var},"
                    f"{pool_est},{age_med},{age_p90},{cur_lr},"
                    f"{resp_len_stats['resp_len_min']},{resp_len_stats['resp_len_p50']},"
                    f"{resp_len_stats['resp_len_p90']},{resp_len_stats['resp_len_max']}\n"
                )

            if wandb_log:
                try:
                    import wandb
                    wandb.log({
                        "iter": iter_num,
                        "loss/policy": mean_p,
                        "loss/value": mean_v if mean_v is not None else float("nan"),
                        "kl/report": report_kl,
                        "kl/safe": safe_kl,
                        "reward/raw": r_raw,
                        "reward/ema": r_raw_ema,
                        "reward/ctr": r_ctr,
                        "reward/shaped": r_shaped,
                        "reward/eval_sampled": r_eval_greedy,
                        "ppo/clip_frac": clip_frac,
                        "ppo/approx_kl_pi": approx_kl_pi,
                        "entropy/token": entropy_tok,
                        "value/mae": v_mae,
                        "value/explained_var": explained_var,
                        "lr/actor": opt_a.param_groups[0]['lr'],
                        "pool/estimate": pool_est,
                        "pool/age_med_sec": 0 if age_med==float("inf") else age_med,
                        "pool/age_p90_sec": 0 if age_p90==float("inf") else age_p90,
                        "resp/min": resp_len_stats['resp_len_min'],
                        "resp/p50": resp_len_stats['resp_len_p50'],
                        "resp/p90": resp_len_stats['resp_len_p90'],
                        "resp/max": resp_len_stats['resp_len_max'],
                        "kl_ctl": kl_ctl_now,
                    })
                except Exception as e:
                    print(f"[wandb] log error: {e}")

            # 保存 ckpt（简明）
            ckpt = {
                'model': raw_actor.state_dict(),
                'critic': critic.state_dict(),
                'ref': ref.state_dict(),
                'optimizer_actor': opt_a.state_dict(),
                'optimizer_critic': opt_c.state_dict(),
                'iter_num': iter_num,
            }
            torch.save(ckpt, os.path.join(out_dir, "PPO_ckpt.pt"))

            # —— 每隔 SGLANG_EXPORT_EVERY 步导出给 sglang（HF 目录 + pointer 原子切换） ——
            if SGLANG_ON and SGLANG_OFFLINE and (iter_num % int(SGLANG_EXPORT_EVERY) == 0):
                try:
                    export_actor_for_sglang(raw_actor, init_from, SGLANG_EXPORT_BASE, SGLANG_MODEL_PATH)
                except Exception as e:
                    print(f"[export][warn] export failed: {e}", flush=True)

    if ddp: destroy_process_group()

if __name__ == "__main__":
    main()
