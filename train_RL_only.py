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

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
)
from utils.rollout_pool import dequeue_items, dequeue_groups, estimate_size, ensure_dir

# =============== 可被外部 config.py 覆盖的全局 ===============
use_ppo  = True
use_grpo = False
use_dapo = False

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

# 模型结构（会被 init_from 覆盖）
n_layer = 12; n_head = 12; n_embd = 768
block_size            = 256
dropout               = 0.0
bias                  = False
init_from             = "gpt2-large"  # "gpt2-*", 或本地 ckpt

# 训练批次
batch_size                        = 4
gradient_accumulation_steps       = 1

# PPO/优化
RL_learning_rate      = 1.5e-6
weight_decay          = 5e-3
beta1                 = 0.9
beta2                 = 0.95
kl_ctl_init           = 0.7
max_grad_norm         = 1.0
vf_clip               = 0.2
ppo_clip              = 0.2
entropy_coef          = 0.0

# GRPO
grpo_group_size       = 4
ratio_min             = 0.75
ratio_max_stat        = 1.25
kl_token_cap          = 0.5
k3_cap                = 1.5
ent_mask_keep         = 0.20
mb_size_logits        = 1
mb_size_values        = 1  # 仅 PPO 用

# 生成/停词
SAMPLE_TEMPERATURE    = 0.8
SAMPLE_TOP_P          = 0.9
SAMPLE_TOP_K          = 0
SAMPLE_REP_PENALTY    = 1.1
SAMPLE_STOPS          = ["\nHuman:", "\n\nHuman:"]
MIN_RESP_TOK          = 16

# sglang 离线池
SGLANG_ON             = True
SGLANG_OFFLINE        = True
SGLANG_MODEL_PATH     = "gpt2-large"  # 将被替换成符号链接
SGLANG_EXPORT_BASE    = "/root/autodl-tmp/actor_exports"
SGLANG_EXPORT_EVERY   = 30
SGLANG_SYNC_DIR       = "/root/autodl-tmp/sgl_pool"
SGLANG_MAX_NEW        = 128
ROLL_LOW_WATERMARK_FACTOR = 3
SGLANG_ROLLOUT_TARGET = 96
SGLANG_REFILL_BATCH   = 48
ROLL_REFILL_COUNT     = 24
ROLL_COOLDOWN_SEC     = 18
ROLL_MIN_FREE_MB      = 6000
FRESH_RATIO           = 0.50
POOL_STALE_WARN_SEC   = 600
REFRESH_EVERY_BATCHES = 30

# 奖励模型（EN）
REWARD_MODEL_NAME     = "OpenAssistant/reward-model-deberta-v3-large-v2"

# ===========================================================
def _apply_cli_config_once():
    is_main = (os.environ.get("RANK","-1") == "-1") or (os.environ.get("LOCAL_RANK") in (None,"0"))
    if is_main and len(sys.argv)>=2 and sys.argv[1].endswith(".py"):
        cfg=sys.argv[1]
        print(f"[config] Overriding with {cfg}:\n{'-'*60}\n{open(cfg,'r').read()}\n{'-'*60}")
        exec(open(cfg,"r").read(), globals(), globals())

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
                uniq = torch.unique(out)
                last[:, uniq] = last[:, uniq] / float(repetition_penalty)

            # 温度
            last = last / max(float(temperature), 1e-6)

            # top-k
            if top_k and top_k > 0:
                kth = torch.topk(last, k=min(int(top_k), last.size(-1)), dim=-1).values[..., -1:]
                last = torch.where(last < kth, torch.full_like(last, -1e10), last)

            # softmax
            probs = torch.softmax(last, dim=-1)

            # top-p（p>=0.999 时直接跳过以免数值抖动）
            if (top_p is not None) and (top_p < 0.999):
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                cutoff = (cumsum > float(top_p)).float().argmax(dim=-1, keepdim=True)
                arng = torch.arange(probs.size(-1), device=probs.device).view(1, -1)
                mask = arng <= cutoff
                kept = torch.where(mask, sorted_probs, torch.zeros_like(sorted_probs))
                kept = kept / kept.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                next_sorted = torch.multinomial(kept, num_samples=1)
                next_id = sorted_idx.gather(1, next_sorted)
            else:
                next_id = torch.multinomial(probs, num_samples=1)

            # 最小回复长度保护
            if (out.size(1) - start_len) < int(min_resp) and eos_id is not None and int(next_id.item()) == int(eos_id):
                # 选择下一个概率最大的非 eos
                top2 = torch.topk(probs, k=2, dim=-1).indices
                alt = top2[:, :1]
                next_id = torch.where(alt == eos_id, top2[:, 1:2], alt)

            out = torch.cat((out, next_id.to(device)), dim=1)

            # 结束条件
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

def _atomic_write_text(path: str, text: str):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

# -------- 关键：对齐友好的、轻量的导出给 sglang --------
def export_actor_for_sglang(
    raw_actor,
    init_from,
    export_base: str,
    symlink_path: str,
    keep_last_exports: int = 1,   # 只保留最新 N 个
):
    import os, glob, shutil, random
    from datetime import datetime
    from transformers import AutoModelForCausalLM, AutoTokenizer
    os.makedirs(export_base, exist_ok=True)

    # 1) 准备 HF 骨架（与 init_from 对齐）
    hf_model = AutoModelForCausalLM.from_pretrained(init_from, torch_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(init_from, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    sd_src = raw_actor.state_dict()      # 你的 Actor (nanoGPT 风格)
    sd_tgt = hf_model.state_dict()       # HF GPT-2 风格

    # 需要强制转置的 4 类权重（Conv1D 口径差异）
    NEED_T = ("attn.c_attn.weight", "attn.c_proj.weight",
              "mlp.c_fc.weight",   "mlp.c_proj.weight")

    def _need_t(k: str) -> bool:
        return any(k.endswith(suf) for suf in NEED_T)

    matched = 0
    transposed = 0
    skipped = 0

    # 为了更清晰的词表报告
    src_vocab = None
    tgt_vocab = None
    if "transformer.wte.weight" in sd_src and "transformer.wte.weight" in sd_tgt:
        src_vocab = sd_src["transformer.wte.weight"].shape[0]
        tgt_vocab = sd_tgt["transformer.wte.weight"].shape[0]

    # 逐参数复制
    for k_tgt in list(sd_tgt.keys()):
        if k_tgt not in sd_src:
            # 权重名基本一致；若 key 不在源模型，跳过
            skipped += 1
            continue

        w_src = sd_src[k_tgt]
        w_tgt = sd_tgt[k_tgt]

        # 1) 需要转置的四类：无条件 t()（即使形状相等也转）
        if _need_t(k_tgt):
            try:
                w_tgt.copy_(w_src.t().to(w_tgt.dtype))
                transposed += 1
            except Exception:
                skipped += 1
            continue

        # 2) 词嵌入 / lm_head：下采样到 HF 词表（通常 50304 -> 50257）
        if k_tgt.endswith(("wte.weight", "lm_head.weight")):
            rows = min(w_src.shape[0], w_tgt.shape[0])
            try:
                w_tgt[:rows, :].copy_(w_src[:rows, :].to(w_tgt.dtype))
                matched += 1
            except Exception:
                skipped += 1
            continue

        # 3) 位置嵌入：按最短 rows 复制（block_size 可能不同）
        if k_tgt.endswith("wpe.weight"):
            rows = min(w_src.shape[0], w_tgt.shape[0])
            try:
                w_tgt[:rows, :].copy_(w_src[:rows, :].to(w_tgt.dtype))
                matched += 1
            except Exception:
                skipped += 1
            continue

        # 4) 其余：形状必须一致直接复制
        if w_src.shape == w_tgt.shape:
            try:
                w_tgt.copy_(w_src.to(w_tgt.dtype))
                matched += 1
            except Exception:
                skipped += 1
        else:
            skipped += 1

    hf_model.eval()

    # 3) 导出到新目录
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
    # 若目标已存在且是目录（非符号链接），先删掉释放空间
    if os.path.exists(symlink_path) and not os.path.islink(symlink_path):
        shutil.rmtree(symlink_path)
    os.symlink(out_dir, symlink_tmp)
    os.replace(symlink_tmp, symlink_path)

    # 汇总日志
    vocab_note = ""
    if src_vocab is not None and tgt_vocab is not None:
        if src_vocab != tgt_vocab:
            vocab_note = f", vocab_downsize={src_vocab}->{tgt_vocab}"
        else:
            vocab_note = f", vocab={tgt_vocab}"
    print(
        f"[export] sglang HF model updated. matched={matched}, transposed={transposed}, "
        f"skipped={skipped}{vocab_note}, path={out_dir}",
        flush=True
    )

    # 5) 仅保留最新导出
    def _prune_exports(base, keep_last=1):
        keep_last = max(int(keep_last), 1)
        ds = sorted([d for d in glob.glob(os.path.join(base, "ts_*")) if os.path.isdir(d)])
        for d in ds[:-keep_last]:
            try:
                shutil.rmtree(d)
                print(f"[export][gc] remove old export: {d}")
            except Exception as e:
                print(f"[export][gc][warn] {d}: {e}")

    _prune_exports(export_base, keep_last=keep_last_exports)
    return out_dir


def _spawn_rollout_subprocess(prompt_bin_path, count, sync_dir, max_new, rollout_log_dir, quiet=True):
    ensure_dir(sync_dir)
    os.makedirs(rollout_log_dir, exist_ok=True)
    per_prompt = int(globals().get("grpo_group_size", 4)) if globals().get("use_grpo", False) else 1
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
        "--min-free-mb", str(int(ROLL_MIN_FREE_MB)),
        "--per-prompt", str(per_prompt),
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

@torch.no_grad()
def eval_fixed_raw_reward_sampled(actor_model, gpt2_tok, eval_prompt_ids,
                                  reward_tokenizer, reward_model,
                                  block_size, max_new_eval=64,
                                  mode: str = "greedy",
                                  seeds: int = 1):
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
                    out = _decode_with_sampling(
                        actor_model, idx, gen_len, eos_id, block_size,
                        temperature=1e-6, top_p=1.0, top_k=0,
                        repetition_penalty=1.0,
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
        assert gradient_accumulation_steps % max(world,1) == 0
    else:
        dev=device; master=True

    # 随机种
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    torch.manual_seed(seed_base + (int(os.environ.get("RANK","0")) if ddp else 0))

    # prompts
    tok = GPT2Tok()
    (PROMPTS_TEXT, PROMPT_TOKEN_IDS, EOS_ID, data_seed, PROMPT_BIN_PATH, TRAIN_INDICES, EVAL_INDICES) = _load_prompts()
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
        # 这里走我们强化过的 from_pretrained：默认 vocab=50304，自动复制/零填充
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

    # 初始导出给 sglang（极轻量）
    if SGLANG_ON and SGLANG_OFFLINE and (os.environ.get("RANK","0") == "0"):
        try:
            os.makedirs(os.path.dirname(SGLANG_MODEL_PATH), exist_ok=True)
            export_actor_for_sglang(raw_actor, init_from, SGLANG_EXPORT_BASE, SGLANG_MODEL_PATH)
        except Exception as e:
            print(f"[export][warn] initial export failed: {e}", flush=True)

    # 优化器
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
    reward_model = reward_hf.eval()
    reward_tokenizer=AutoTokenizer.from_pretrained(REWARD_MODEL_NAME, use_fast=True)
    try: reward_tokenizer.padding_side = "right"
    except Exception: pass
    if getattr(reward_tokenizer, "pad_token", None) is None and getattr(reward_tokenizer, "eos_token", None) is not None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token

    # === 选择算法：PPO 或 GRPO ===
    if use_ppo:
        from RL.PPO import PPOTrainer
        trainer = PPOTrainer(
            actor_model=raw_actor, ref_model=ref, reward_model=reward_model,
            critic_model=critic, actor_tokenizer=tok, reward_tokenizer=reward_tokenizer,
            optimizer_actor=opt_a, optimizer_critic=opt_c,
            device=dev,
            mb_size_logits=mb_size_logits,
            mb_size_values=mb_size_values,
            kl_ctl=kl_ctl_init,
            max_grad_norm=max_grad_norm,
            vf_clip=vf_clip,
            ppo_clip=ppo_clip,
            entropy_coef=entropy_coef,
        )
    elif use_grpo:
        from RL.GRPO import GRPOTrainer
        trainer = GRPOTrainer(
            actor_model=raw_actor, ref_model=ref, reward_model=reward_model,
            actor_tokenizer=tok, reward_tokenizer=reward_tokenizer,
            optimizer_actor=opt_a,
            device=dev,
            mb_size_logits=globals().get("mb_size_logits", 1),
            group_size=globals().get("grpo_group_size", 4),
            kl_ctl=globals().get("kl_ctl_init", 0.6),
            ppo_clip=globals().get("ppo_clip", 0.2),
            entropy_coef=globals().get("entropy_coef", 0.0),
            max_grad_norm=globals().get("max_grad_norm", 1.0),
            ratio_min=globals().get("ratio_min", 0.75),
            ratio_max=globals().get("ratio_max_stat", 1.25),
            kl_token_cap=globals().get("kl_token_cap", 0.5),
            k3_cap=globals().get("k3_cap", 1.5),
            ent_mask_keep=globals().get("ent_mask_keep", 0.20),
        )
    elif use_dapo:
        from RL.DAPO import DAPOTrainer
        trainer = DAPOTrainer(
            actor_model=raw_actor, ref_model=ref, reward_model=reward_model,
            actor_tokenizer=tok, reward_tokenizer=reward_tokenizer,
            optimizer_actor=opt_a,
            device=dev,
            mb_size_logits=globals().get("mb_size_logits", 1),
            kl_ctl=globals().get("kl_ctl_init", 1.0),
            ppo_clip=globals().get("ppo_clip", 0.2),
            entropy_coef=globals().get("entropy_coef", 0.004),
            max_grad_norm=globals().get("max_grad_norm", 0.5),
            ratio_min=globals().get("ratio_min", 0.75),
            ratio_max=globals().get("ratio_max", 1.25),
            kl_token_cap=globals().get("kl_token_cap", 0.5),
            k3_cap=globals().get("k3_cap", 1.5),
            ent_mask_keep=globals().get("ent_mask_keep", 0.20),
            ema_alpha=globals().get("ema_alpha", 0.10),
            ema_warmup=globals().get("ema_warmup", 1),
            use_batch_center_fallback=globals().get("use_batch_center_fallback", True),
        )
    else:
        raise RuntimeError("Please set either use_ppo=True or use_grpo=True in your config.")

    # ===== Baseline eval (before training) =====
    if master and len(EVAL_PROMPT_IDS) > 0:
        try:
            # 统一为评测固定随机种，避免抖动（sampled 会用 1337 + rep）
            torch.manual_seed(seed_base)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_base)

            r_eval_greedy_baseline = eval_fixed_raw_reward_sampled(
                actor_model=raw_actor, gpt2_tok=tok, eval_prompt_ids=EVAL_PROMPT_IDS,
                reward_tokenizer=reward_tokenizer, reward_model=reward_model,
                block_size=block_size, max_new_eval=min(SGLANG_MAX_NEW, 64),
                mode="greedy", seeds=1
            )
            r_eval_sampled_baseline = eval_fixed_raw_reward_sampled(
                actor_model=raw_actor, gpt2_tok=tok, eval_prompt_ids=EVAL_PROMPT_IDS,
                reward_tokenizer=reward_tokenizer, reward_model=reward_model,
                block_size=block_size, max_new_eval=min(SGLANG_MAX_NEW, 64),
                mode="sample", seeds=3   # 多次采样更稳，3 次均值即可
            )

            print(f"[baseline] r_eval_greedy={r_eval_greedy_baseline:.4f} "
                  f"r_eval_sampled={r_eval_sampled_baseline:.4f}", flush=True)

            # 追加到 metrics.csv（iter=0 代表训练前）
            METRICS_CSV = os.path.join(out_dir, "metrics.csv")
            hdr = not os.path.exists(METRICS_CSV)
            with open(METRICS_CSV, "a") as f:
                if hdr:
                    f.write(
                        "iter,p_loss,v_loss,avg_kl,safe_kl,r_raw,r_raw_ema,r_ctr,r_shp,r_eval_greedy,"
                        "clip_frac,approx_kl_pi,entropy,v_mae,explained_var,pool_est,age_med,age_p90,lr,"
                        "resp_min,resp_p50,resp_p90,resp_max,algo,grpo_group_eff,grpo_ctr_abs,"
                        "r_eval_sampled\n"
                    )
                f.write(
                    f"0,,,nan,nan,nan,nan,nan,nan,{r_eval_greedy_baseline},"
                    f",,, , , , , , , , , , ,{'PPO' if use_ppo else 'GRPO'},,,"  # 保持列数
                    f"{r_eval_sampled_baseline}\n"
                )

            if wandb_log:
                import wandb
                wandb.log({
                    "iter": 0,
                    "reward/eval_greedy": r_eval_greedy_baseline,
                    "reward/eval_sampled": r_eval_sampled_baseline,
                    "phase": "baseline",
                })
        except Exception as e:
            print(f"[baseline][warn] eval failed: {e}", flush=True)
        
    # W&B
    if wandb_log:
        try:
            import wandb, json
            os.environ["WANDB_MODE"] = "offline"
            os.environ["WANDB_SILENT"] = "true"
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
                "REWARD_MODEL_NAME","use_ppo","use_grpo","grpo_group_size"
            }
            for k, v in globals().items():
                if k in white:
                    try: json.dumps(v); cfg_for_log[k] = v
                    except Exception: cfg_for_log[k] = json.dumps(v, ensure_ascii=False)
            wandb.init(project=wandb_project, name=wandb_run_name, dir=out_dir, config=cfg_for_log)
        except Exception as e:
            print(f"[wandb] disabled: {e}")
            wandb_log = False

    # ====== sglang 池启用 ======
    if master and SGLANG_ON and SGLANG_OFFLINE:
        print(f"[sglang] offline pool: dir={SGLANG_SYNC_DIR} target={SGLANG_ROLLOUT_TARGET} batch={SGLANG_REFILL_BATCH}")

    r_raw_ema = None
    last_rollout_t = 0.0

    # KL 迟滞控制等
    KL_HALT, KL_RESUME = 0.12, 0.08
    HALT_STREAK, RESUME_STREAK = 2, 2
    actor_frozen = False; halt_hits = 0; resume_hits = 0; freeze_steps = 0
    FORCE_FRESH_STEPS = 4
    EMERG_SHORT_GEN_STEPS = 4
    BASE_LR_ACTOR = RL_learning_rate

    for iter_num in range(1, max_iters+1):
        # ---- 池子估算 + 新鲜度 ----
        pool_est = estimate_size(SGLANG_SYNC_DIR, SGLANG_REFILL_BATCH) if (SGLANG_ON and SGLANG_OFFLINE) else -1
        n_files, (age_med, age_p90) = _pool_freshness(SGLANG_SYNC_DIR) if (SGLANG_ON and SGLANG_OFFLINE) else (0,(float("inf"),float("inf")))

        # ---- 离线补货 ----
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

        # === 取样 ===
        groups = None
        if use_grpo and SGLANG_ON and SGLANG_OFFLINE:
            groups = dequeue_groups(SGLANG_SYNC_DIR, group_size=int(grpo_group_size), num_groups=int(batch_size))

        def _gen_one_from_train(use_ref: bool, prompt_ids=None):
            ids = prompt_ids if (isinstance(prompt_ids, list) and prompt_ids) else random.choice(TRAIN_PROMPT_IDS)
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

        experiences_all = []
        report_kl = r_raw = r_shaped = r_ctr = safe_kl = float("nan")
        stats_last = {}
        sel_tokens = 0

        if use_grpo:
            groups = groups or []
            while len(groups) < int(batch_size):
                base_ids = random.choice(TRAIN_PROMPT_IDS)
                g = []
                for _ in range(int(grpo_group_size)):
                    it = _gen_one_from_train(use_ref=actor_frozen, prompt_ids=base_ids)
                    if it is not None: g.append(it)
                if len(g) == int(grpo_group_size):
                    groups.append(g)
                else:
                    break

            for grp in groups[:int(batch_size)]:
                samples = pack_samples(grp, pad_id=tok.eos_id, block_size=block_size, device=torch.device(dev))
                if int(samples.action_mask.sum().item()) == 0:
                    continue
                exps, rep_kl, rr, rshp, rctr, skl = trainer.evaluate_experience(samples)
                if exps:
                    experiences_all.extend(exps)
                report_kl = rep_kl if np.isfinite(rep_kl) else report_kl
                r_raw = rr if np.isfinite(rr) else r_raw
                r_shaped = rshp if np.isfinite(rshp) else r_shaped
                r_ctr = rctr if np.isfinite(rctr) else r_ctr
                safe_kl = skl if np.isfinite(skl) else safe_kl
                stats_last = getattr(trainer, "last_stats", {}) or stats_last
                sel_tokens += int(stats_last.get("sel_tokens", 0) or 0)

            if not experiences_all:
                if master: print(f"[iter {iter_num:4d}] skip(empty-GRPO-exps) pool={pool_est}", flush=True)
                continue

            trainer.last_stats = stats_last
            report_kl = float(report_kl) if np.isfinite(report_kl) else float("nan")
            safe_kl = float(safe_kl) if np.isfinite(safe_kl) else float("nan")
        else:
            want_fresh_base = max(1, int(np.ceil(FRESH_RATIO * batch_size)))
            force_fresh = (FORCE_FRESH_STEPS > 0)
            if actor_frozen or force_fresh:
                want_fresh = batch_size; batch = []
                if force_fresh: FORCE_FRESH_STEPS -= 1
            else:
                want_fresh = want_fresh_base
                batch = dequeue_items(SGLANG_SYNC_DIR, batch_size - want_fresh) if (SGLANG_ON and SGLANG_OFFLINE) else []
            fresh = []
            for _ in range(want_fresh):
                g = _gen_one_from_train(use_ref=actor_frozen)
                if g is not None: fresh.append(g)
            batch.extend(fresh)
            while len(batch) < batch_size:
                g = _gen_one_from_train(use_ref=actor_frozen)
                if g is None: break
                batch.append(g)
            samples = pack_samples(batch, pad_id=tok.eos_id, block_size=block_size, device=torch.device(dev))
            if int(samples.action_mask.sum().item()) == 0:
                if master: print(f"[iter {iter_num:4d}] skip(empty-response-batch) pool={pool_est}", flush=True)
                continue
            experiences_all, report_kl, r_raw, r_shaped, r_ctr, safe_kl = trainer.evaluate_experience(samples)
            stats_last = getattr(trainer, "last_stats", {}) or {}
            sel_tokens = int(stats_last.get("sel_tokens", 0) or 0)

        # —— 安全保险丝 —— #
        critic_only_this_iter = False
        if (not np.isfinite(safe_kl)) or (safe_kl > 1.7):
            actor_frozen = True; critic_only_this_iter = True
            FORCE_FRESH_STEPS = max(FORCE_FRESH_STEPS, 3)
            EMERG_SHORT_GEN_STEPS = max(EMERG_SHORT_GEN_STEPS, 3)
            if master: print(f"[guard] skip actor (abnormal KL={report_kl:.4g}) -> critic-only", flush=True)

        stats_last = getattr(trainer, "last_stats", {}) or {}

        def _as_float(x, default=float("nan")):
            try:
                return float(x)
            except Exception:
                return default

        clip_frac      = _as_float(stats_last.get("clip_frac", 0.0), 0.0)
        approx_kl_pi   = _as_float(stats_last.get("approx_kl_pi"))
        entropy_tok    = _as_float(stats_last.get("entropy"))
        v_mae          = _as_float(stats_last.get("v_mae"))
        explained_var  = _as_float(stats_last.get("explained_var"))
        sel_tokens     = int(stats_last.get("sel_tokens", 0) or 0)
        ppo_clip_v     = _as_float(stats_last.get("ppo_clip"))
        kl_ctl_now     = _as_float(stats_last.get("kl_ctl_now"))
        adv_abs_m      = _as_float(stats_last.get("adv_abs_mean"))

        ratio_qs_stat = stats_last.get("ratio_q50_q90_q99")
        if not (isinstance(ratio_qs_stat, (list, tuple)) and len(ratio_qs_stat) >= 3):
            ratio_qs_stat = (float("nan"), float("nan"), float("nan"))
        ratio_max_stat = _as_float(stats_last.get("ratio_max_stat"))

        grpo_group_eff = _as_float(stats_last.get("grpo/group_eff_mean"))
        grpo_ctr_abs   = _as_float(stats_last.get("grpo/r_center_mean_abs"), 0.0)

        if np.isfinite(safe_kl) and safe_kl > 1.7:
            actor_frozen = True
            FORCE_FRESH_STEPS = max(FORCE_FRESH_STEPS, 3)
            EMERG_SHORT_GEN_STEPS = max(EMERG_SHORT_GEN_STEPS, 3)
            trainer.kl_ctl = float(min(2.0, trainer.kl_ctl * 1.2))
            for g in opt_a.param_groups:
                g['lr'] = max(RL_learning_rate * 0.25, 8e-7)
        if EMERG_SHORT_GEN_STEPS > 0: EMERG_SHORT_GEN_STEPS -= 1

        # 自适应 KL
        kl_target = 0.25
        if np.isfinite(safe_kl):
            err = safe_kl / max(kl_target, 1e-8) - 1.0
            up = 0.05; down = 0.03
            trainer.kl_ctl *= float(np.exp(np.clip(err, -down, up)))
            trainer.kl_ctl = float(np.clip(trainer.kl_ctl, 0.15, 2.0))

        # EMA 奖励
        r_raw_ema = _ema_update(r_raw_ema, r_raw, alpha=0.1)

        # 临时降 LR（actor）
        KL_STOP = 1.5
        if np.isfinite(safe_kl):
            err = max(0.0, safe_kl / KL_STOP - 1.0)
            scale = 1.0 / (1.0 + 2.0 * err)
            scale = float(max(0.25, min(1.0, scale)))
        else:
            scale = 0.25
        for g in opt_a.param_groups:
            g['lr'] = RL_learning_rate * scale

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
                    g['lr'] = max(RL_learning_rate * 0.4, 6e-7)
                if master:
                    print(f"[guard] long-freeze fallback: kl_ctl={trainer.kl_ctl:.3f} actor_lr={opt_a.param_groups[0]['lr']:.2e}", flush=True)
        else:
            freeze_steps = 0

        # === 训练 ===
        if not experiences_all:
            continue

        pl, vl = [], []
        if use_ppo and (actor_frozen or critic_only_this_iter):
            # 只训 critic（仅 PPO）
            for exp in experiences_all:
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
            POLICY_EPOCHS = 2
            if use_ppo:
                KL_EPOCH_DROP = 1.10
                LOW_EPOCH, HIGH_EPOCH = 1, 2
                POLICY_EPOCHS = LOW_EPOCH if (np.isfinite(safe_kl) and safe_kl > KL_EPOCH_DROP) else HIGH_EPOCH
                if (iter_num > 5) and np.isfinite(safe_kl) and (safe_kl < 0.25) and (clip_frac < 0.01):
                    POLICY_EPOCHS = 3
                    for g in opt_a.param_groups:
                        g['lr'] = min(RL_learning_rate * 1.2, 5e-6)
            for _ in range(POLICY_EPOCHS):
                for exp in experiences_all:
                    out = trainer.train_on_experience(exp, use_token_entropy=False)
                    if isinstance(out, tuple):
                        p, v = out; pl.append(float(p.detach().item()))
                        if v is not None: vl.append(float(v.detach().item()))
                    else:
                        pl.append(float(out.detach().item()))
            mean_p = float(np.mean(pl)) if pl else 0.0
            mean_v = float(np.mean(vl)) if vl else None

        # 评测（统一贪心）
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

        # ===== 日志输出 =====
        resp_len_stats = {"resp_len_min": 0, "resp_len_p50": 0.0, "resp_len_p90": 0.0, "resp_len_max": 0}

        if (iter_num % eval_interval == 0) and master:
            cur_lr = opt_a.param_groups[0]['lr']
            stale_note = ""
            if age_med != float("inf"):
                stale_note = f" age_med={age_med:.0f}s p90={age_p90:.0f}s"
                if age_med > POOL_STALE_WARN_SEC: stale_note += "(!stale)"
            algo = "PPO" if use_ppo else "GRPO"
            core = (
                f"[{algo}] [iter {iter_num:4d}] p={mean_p:.4f} "
                + (f"v={mean_v:.4f} " if (use_ppo and mean_v is not None) else "")
                + f"kl={report_kl:.6f} safe_kl={safe_kl:.6f} r_raw={r_raw:.4f} r_ema={_ema_update(None, r_raw_ema, 1.0):.4f} "
                  f"r_ctr={r_ctr:.4f} r_shp={r_shaped:.4f} r_eval_greedy={r_eval_greedy:.4f} "
                  f"clip={clip_frac:.3f} akl_pi={approx_kl_pi:.4f} H={entropy_tok:.3f} "
                + (f"v_mae={v_mae:.4f} ev={explained_var:.3f} " if use_ppo else "")
                + (f"grp_eff={grpo_group_eff:.2f} ctr_abs={grpo_ctr_abs:.3e} " if use_grpo else "")
                + f"| sel_tok={sel_tokens} adv|={adv_abs_m:.3e} pool={pool_est} "
                  f"rΔ q50/90/99={ratio_qs_stat[0]:.3e}/{ratio_qs_stat[1]:.3e}/{ratio_qs_stat[2]:.3e} "
                  f"rΔmax={ratio_max_stat:.3e} clip_th={ppo_clip_v:.3f} kl_ctl={kl_ctl_now:.3f} "
                  f"lr={cur_lr:.2e} "
                  f"{stale_note}"
            )
            print(core, flush=True)

            hdr = not os.path.exists(METRICS_CSV)
            with open(METRICS_CSV, "a") as f:
                if hdr:
                    f.write(
                        "iter,p_loss,v_loss,avg_kl,safe_kl,r_raw,r_raw_ema,r_ctr,r_shp,r_eval_greedy,"
                        "clip_frac,approx_kl_pi,entropy,v_mae,explained_var,pool_est,age_med,age_p90,lr,"
                        "resp_min,resp_p50,resp_p90,resp_max,algo,grpo_group_eff,grpo_ctr_abs\n"
                    )
                f.write(
                    f"{iter_num},{mean_p},{'' if mean_v is None else mean_v},"
                    f"{report_kl},{safe_kl},{r_raw},{r_raw_ema},{r_ctr},{r_shaped},{r_eval_greedy},"
                    f"{clip_frac},{approx_kl_pi},{entropy_tok},{v_mae},{explained_var},"
                    f"{pool_est},{age_med},{age_p90},{cur_lr},"
                    f"{resp_len_stats['resp_len_min']},{resp_len_stats['resp_len_p50']},"
                    f"{resp_len_stats['resp_len_p90']},{resp_len_stats['resp_len_max']},"
                    f"{'PPO' if use_ppo else 'GRPO'},{grpo_group_eff},{grpo_ctr_abs}\n"
                )

            if wandb_log:
                try:
                    import wandb
                    wb = {
                        "algo": ("PPO" if use_ppo else "GRPO"),
                        "iter": iter_num,
                        "loss/policy": mean_p,
                        "loss/value": mean_v if (use_ppo and mean_v is not None) else float("nan"),
                        "kl/report": report_kl,
                        "kl/safe": safe_kl,
                        "reward/raw": r_raw,
                        "reward/ema": r_raw_ema,
                        "reward/ctr": r_ctr,
                        "reward/shaped": r_shaped,
                        "reward/eval_greedy": r_eval_greedy,
                        "ppo/clip_frac": clip_frac,
                        "ppo/approx_kl_pi": approx_kl_pi,
                        "entropy/token": entropy_tok,
                        "value/mae": v_mae if use_ppo else float("nan"),
                        "value/explained_var": explained_var if use_ppo else float("nan"),
                        "lr/actor": cur_lr,
                        "pool/estimate": pool_est,
                        "pool/age_med_sec": 0 if age_med==float("inf") else age_med,
                        "pool/age_p90_sec": 0 if age_p90==float("inf") else age_p90,
                        "resp/min": resp_len_stats['resp_len_min'],
                        "resp/p50": resp_len_stats['resp_len_p50'],
                        "resp/p90": resp_len_stats['resp_len_p90'],
                        "resp/max": resp_len_stats['resp_len_max'],
                        "kl_ctl": kl_ctl_now,
                        "adv/abs_mean": adv_abs_m,
                        "sel/tokens": sel_tokens,
                    }
                    if use_grpo:
                        wb.update({
                            "grpo/group_eff_mean": grpo_group_eff,
                            "grpo/r_center_mean_abs": grpo_ctr_abs,
                        })
                    wandb.log(wb)
                except Exception as e:
                    print(f"[wandb] log error: {e}")

            # 保存 ckpt
            ckpt = {
                'model': raw_actor.state_dict(),
                'critic': critic.state_dict(),
                'ref': ref.state_dict(),
                'optimizer_actor': opt_a.state_dict(),
                'optimizer_critic': opt_c.state_dict(),
                'iter_num': iter_num,
            }
            torch.save(ckpt, os.path.join(out_dir, "PPO_ckpt.pt"))

            # 导出给 sglang（轻量）
            if SGLANG_ON and SGLANG_OFFLINE and (iter_num % int(SGLANG_EXPORT_EVERY) == 0):
                try:
                    export_actor_for_sglang(raw_actor, init_from, SGLANG_EXPORT_BASE, SGLANG_MODEL_PATH)
                except Exception as e:
                    print(f"[export][warn] export failed: {e}", flush=True)

    if ddp: destroy_process_group()

if __name__ == "__main__":
    main()
