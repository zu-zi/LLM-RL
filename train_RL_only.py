# train_RL_only.py
import os, sys, time, random, json, subprocess, shlex
import numpy as np
import torch, tiktoken
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from RL.PPO import PPOTrainer, Critic
from RL.GRPO import GRPOTrainer
from RL.DAPO import DAPOTrainer
from RL.common import Samples
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils.rollout_pool import dequeue_items, estimate_size, ensure_dir

# ========= 基本环境 =========
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("NANOGPT_SILENT_INIT", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")  # 降噪

# ========= 默认配置（可被外部 config 覆盖）=========
use_ppo = True; use_grpo=False; use_dapo=False; use_token_entropy=False
out_dir = "Results"
eval_interval = 20
max_iters = 200
batch_size = 4
block_size = 512
gradient_accumulation_steps = 1

init_from = "gpt2-large"
dropout = 0.0; bias=False
n_layer=12; n_head=12; n_embd=768
RL_learning_rate=1e-5; weight_decay=1e-1; beta1=0.9; beta2=0.95
backend="nccl"; device="cuda"; compile=False
kl_ctl = 0.20  # 默认，可被 config 覆盖

dataset="openwebtext"
wandb_log=False; wandb_project="hlhf"; wandb_run_name="run"

# sglang 离线池（默认值，可被 config 覆盖）
SGLANG_ON=True
SGLANG_OFFLINE=True
SGLANG_MODEL_PATH="gpt2-large"
SGLANG_SYNC_DIR="./sgl_pool"  # 建议在 config 中指向大盘
SGLANG_ROLLOUT_TARGET=1024
SGLANG_REFILL_BATCH=256
SGLANG_MAX_NEW=192  # 可在 config 里覆盖为 128

# ====== rollout 调度阈值（避免与训练抢显存）======
# 低水位阈值：只有池子低到这么少才补货（越小越不容易重叠）
ROLL_LOW_WATERMARK_FACTOR = 2     # 实际低水位 = batch_size * 2
# 单次补货条数（小颗粒更安全，不与训练抢）
ROLL_REFILL_COUNT = 32
# 两次补货之间最少间隔（秒）
ROLL_COOLDOWN_SEC = 60
# 触发补货前，GPU 至少需要这么多空闲显存（MB）
ROLL_MIN_FREE_MB = 12000

# 奖励模型（默认英文 RM，和 HH-RLHF prompt 对齐好）
REWARD_MODEL_NAME = "OpenAssistant/reward-model-deberta-v3-large"

# ====== CLI 覆盖 ======
def _apply_cli_config_once():
    is_main = (os.environ.get("RANK","-1") == "-1") or (os.environ.get("LOCAL_RANK") in (None,"0"))
    if is_main and len(sys.argv)>=2 and sys.argv[1].endswith(".py"):
        cfg=sys.argv[1]
        print(f"Overriding config with {cfg}:")
        print(open(cfg,"r").read())
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

# ========= 奖励模型 CPU 包装器（自动把输入搬到 CPU）=========
class RewardOnCPU(torch.nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model.eval()
        self._device = torch.device("cpu")
        self.model.to(self._device)
    @property
    def device(self):
        return self._device
    def forward(self, **kwargs):
        moved = {k: (v.to(self._device) if torch.is_tensor(v) else v) for k, v in kwargs.items()}
        with torch.no_grad():
            return self.model(**moved)
    def parameters(self, *args, **kwargs):
        return self.model.parameters(*args, **kwargs)
    def eval(self):
        self.model.eval(); return self

# ========= 数据/打包 =========
def _load_prompts():
    pf=os.path.join(os.path.dirname(__file__),"data/RL_dataset/prompt.bin")
    print(f"Loading fixed prompts from {pf} ...")
    blob=torch.load(pf,map_location="cpu")
    # 兼容新版 prepare.py：若含 train_indices/eval_indices 一并返回
    return (
        blob["prompts"],
        blob["gpt2_token_ids"],
        blob["eos_id"],
        blob.get("seed",1337),
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

# ========= 纯贪心解码（避免 multinomial/CUDA 断言）=========
@torch.no_grad()
def _greedy_decode_argmax(actor_model, idx, max_new_tokens, eos_id, block_size):
    """
    纯 argmax 贪心解码（逐 token 前向），不走 model.generate，
    因此不会触发 multinomial 路径；支持遇到 eos 提前停止。
    - idx: [1, T] Long
    - 返回: [1, T + L_new]
    """
    device = idx.device
    was_train = actor_model.training
    actor_model.eval()
    try:
        out = idx
        for _ in range(int(max_new_tokens)):
            idx_cond = out[:, -int(block_size):]
            logits = actor_model(idx_cond)
            if isinstance(logits, tuple):  # 兼容返回 (logits, ...)
                logits = logits[0]
            next_token_logits = logits[:, -1, :]              # (B, V)
            next_id = torch.argmax(next_token_logits, dim=-1) # (B,)
            next_id = next_id.view(1, 1).to(device)
            out = torch.cat((out, next_id), dim=1)
            if eos_id is not None and int(next_id.item()) == int(eos_id):
                break
        return out
    finally:
        if was_train:
            actor_model.train()

# ========= 显存检测 =========
def _cuda_free_mb(device_str="cuda"):
    try:
        if not torch.cuda.is_available():
            return 0
        free, total = torch.cuda.mem_get_info(torch.device(device_str))
        return int(free // (1024 * 1024))
    except Exception:
        return 0

# ========= sglang 离线补货 =========
def _spawn_rollout_subprocess(prompt_bin_path, count, sync_dir, max_new, rollout_log_dir, quiet=True):
    ensure_dir(sync_dir)
    os.makedirs(rollout_log_dir, exist_ok=True)

    # 用更小的并发，降低峰值；其它采样参数保持
    cmd = [
        "python", "-u", "rollout_worker.py",
        "--model", SGLANG_MODEL_PATH,
        "--prompt-bin", prompt_bin_path,
        "--out-dir", sync_dir,
        "--count", str(int(count)),
        "--max-new", str(int(max_new)),
        "--block-size", str(int(block_size)),
        "--mb", "1",
        "--use-only-train",
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

def _is_master(ddp):
    return (not ddp) or (os.environ.get("RANK","0") == "0")

# ========= 评测 & EMA =========
def _ema_update(prev, x, alpha=0.1):
    if prev is None or np.isnan(prev):
        return float(x)
    return float(alpha * x + (1.0 - alpha) * prev)

@torch.no_grad()
def _score_texts_raw_reward(texts, reward_tokenizer, reward_model, max_length=1024):
    if len(texts) == 0:
        return []
    toks = reward_tokenizer(
        texts, padding=True, truncation=True, max_length=int(max_length), return_tensors="pt"
    )
    outs = reward_model(**toks)
    logits = getattr(outs, "logits", None)
    if logits is None:
        return [float("nan")] * len(texts)
    if logits.dim() == 2 and logits.size(-1) == 1:
        logits = logits.squeeze(-1)
    return [float(v) for v in logits.detach().cpu().tolist()]

@torch.no_grad()
def eval_fixed_raw_reward(actor_model, gpt2_tok, eval_prompt_ids, reward_tokenizer, reward_model, block_size, max_new_eval=128):
    dev = next(actor_model.parameters()).device
    eos_id = gpt2_tok.eos_id
    texts = []

    was_training = actor_model.training
    actor_model.eval()
    try:
        for ids in eval_prompt_ids:
            ids_t = torch.tensor(ids, dtype=torch.long, device=dev).unsqueeze(0)
            prompt_len = ids_t.size(1)
            room = block_size - prompt_len - 1
            if room <= 0:
                full_ids = ids[:block_size]
                texts.append(gpt2_tok.decode(full_ids))
                continue

            gen_len = max(8, min(int(max_new_eval), int(room)))  # 按你要求：最多 128
            out = _greedy_decode_argmax(actor_model, ids_t, gen_len, eos_id, block_size)
            full = out[0].tolist()[:block_size]
            texts.append(gpt2_tok.decode(full))
    finally:
        if was_training:
            actor_model.train()

    scores = _score_texts_raw_reward(texts, reward_tokenizer, reward_model, max_length=1024)
    return float(np.mean(scores)) if len(scores) > 0 else float("nan")

# ========= 主流程 =========
def main():
    _apply_cli_config_once()

    # 依赖外部 config 的量，必须在 exec(config) 之后再确定
    EVAL_LOG_EVERY = eval_interval
    DEBUG_SAMPLE_EVERY = eval_interval
    METRICS_CSV = os.path.join(out_dir, "metrics.csv")
    ROLLOUT_LOG_DIR = os.path.join(out_dir, "rollout_logs")
    ROLLOUT_QUIET = True

    # 固定 prompts / tokenizer
    tok = GPT2Tok()
    (
        PROMPTS_TEXT, PROMPT_TOKEN_IDS, EOS_ID, seed, PROMPT_BIN_PATH,
        TRAIN_INDICES, EVAL_INDICES
    ) = _load_prompts()
    pad_id = 0

    # ====== 构建“固定评测集”：优先使用 prepare.py 写入的 eval_indices ======
    if isinstance(EVAL_INDICES, list) and len(EVAL_INDICES) > 0:
        EVAL_PROMPT_IDS = [PROMPT_TOKEN_IDS[i] for i in EVAL_INDICES if 0 <= i < len(PROMPT_TOKEN_IDS)]
    else:
        rng = np.random.RandomState(int(seed))
        eval_count = min(16, len(PROMPT_TOKEN_IDS))  # 评测固定 16 条（稳定且开销小）
        EVAL_PROMPT_IDS = [PROMPT_TOKEN_IDS[i] for i in rng.choice(len(PROMPT_TOKEN_IDS), size=eval_count, replace=False)] if eval_count>0 else []

    # DDP
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend=backend)
        rank=int(os.environ['RANK']); local=int(os.environ['LOCAL_RANK']); world=int(os.environ['WORLD_SIZE'])
        dev=f'cuda:{local}'; torch.cuda.set_device(dev); master=(rank==0)
        assert gradient_accumulation_steps % world == 0
    else:
        dev=device; master=True

    if master: os.makedirs(out_dir, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    torch.manual_seed(1337 + (int(os.environ.get("RANK","0")) if ddp else 0))

    # ========== 在 CPU 构建 base_state（只一次） ==========
    from model import GPT, GPTConfig
    model_args=dict(n_layer=n_layer,n_head=n_head,n_embd=n_embd,block_size=block_size,bias=bias,dropout=dropout,vocab_size=None)
    if isinstance(init_from,str) and init_from.startswith("gpt2"):
        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
        m=GPT.from_pretrained(init_from, dict(dropout=dropout))  # 默认在 CPU
        for k in ['n_layer','n_head','n_embd','block_size','bias','vocab_size']: model_args[k]=getattr(m.config,k)
        if block_size < model_args['block_size']:
            m.crop_block_size(block_size); model_args['block_size']=block_size
        base_state=m.state_dict(); del m
    else:
        print("Initializing a new model from scratch")
        model_args['vocab_size']=50304
        m=GPT(GPTConfig(**model_args)); base_state=m.state_dict(); del m

    # ========== 构建 ref/actor/critic ==========
    ref=GPT(GPTConfig(**model_args)).to(dev); ref.load_state_dict(base_state)
    for p in ref.parameters(): p.requires_grad=False
    ref.eval()
    ref_dtype=torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    ref=ref.to(device=dev, dtype=ref_dtype)

    actor=GPT(GPTConfig(**model_args)).to(dev); actor.load_state_dict(base_state)
    if compile: actor=torch.compile(actor)
    if ddp: actor=DDP(actor, device_ids=[int(dev.split(':')[-1])])
    raw_actor=actor.module if ddp else actor

    critic=Critic(raw_actor).to(dev)

    # 优化器（8bit 优先，失败回退）
    try:
        from bitsandbytes.optim import AdamW8bit
        opt_a=AdamW8bit(raw_actor.parameters(), lr=RL_learning_rate, betas=(beta1,beta2), weight_decay=weight_decay)
        opt_c=AdamW8bit(critic.parameters(),    lr=RL_learning_rate, betas=(beta1,beta2), weight_decay=weight_decay)
        print("[optim] using bitsandbytes AdamW8bit")
    except Exception as e:
        print(f"[optim] bitsandbytes not available ({e}), fallback to torch.optim.AdamW")
        opt_a=torch.optim.AdamW(raw_actor.parameters(), lr=RL_learning_rate, betas=(beta1,beta2), weight_decay=weight_decay)
        opt_c=torch.optim.AdamW(critic.parameters(),    lr=RL_learning_rate, betas=(beta1,beta2), weight_decay=weight_decay)

    # 奖励模型（强制 CPU + 包装器）
    rw_name = REWARD_MODEL_NAME
    print(f"[reward] loading {rw_name} on CPU ...")
    reward_hf = AutoModelForSequenceClassification.from_pretrained(rw_name, device_map="cpu", torch_dtype=torch.float32).eval()
    reward_model = RewardOnCPU(reward_hf)
    reward_tokenizer=AutoTokenizer.from_pretrained(rw_name, use_fast=True)

    # 训练器
    if use_ppo:
        trainer=PPOTrainer(actor_model=raw_actor, ref_model=ref, reward_model=reward_model,
                           critic_model=critic, actor_tokenizer=tok, reward_tokenizer=reward_tokenizer,
                           optimizer_actor=opt_a, optimizer_critic=opt_c,
                           device=dev, mb_size_logits=1, mb_size_values=1, kl_ctl=kl_ctl)
    elif use_grpo:
        trainer=GRPOTrainer(actor_model=raw_actor, ref_model=ref, reward_model=reward_model,
                            actor_tokenizer=tok, reward_tokenizer=reward_tokenizer,
                            optimizer_actor=opt_a, group_size=4, kl_coef=0.0, clip_reward=5.0,
                            device=dev, mb_size_logits=2)
    else:
        trainer=DAPOTrainer(actor_model=raw_actor, ref_model=ref, reward_model=reward_model,
                            actor_tokenizer=tok, reward_tokenizer=reward_tokenizer,
                            optimizer_actor=opt_a, beta=1.0, adv_norm="zscore", adv_clip=5.0, kl_coef=0.01,
                            device=dev, mb_size_logits=2)

    # ========== 训练循环 ==========
    if _is_master(ddp) and SGLANG_ON and SGLANG_OFFLINE:
        print(f"[sglang] offline pool enabled: dir={SGLANG_SYNC_DIR}, target={SGLANG_ROLLOUT_TARGET}, batch={SGLANG_REFILL_BATCH}")
    ensure_dir(SGLANG_SYNC_DIR)
    os.makedirs(ROLLOUT_LOG_DIR, exist_ok=True)

    r_raw_ema = None  # 训练 batch raw reward 的 EMA
    last_rollout_t = 0.0

    for iter_num in range(0, max_iters+1):
        # ========== 智能补货：严格避免与训练抢显存 ==========
        if SGLANG_ON and SGLANG_OFFLINE:
            LOW_WATERMARK = max(batch_size * ROLL_LOW_WATERMARK_FACTOR, 8)
            pool_est = estimate_size(SGLANG_SYNC_DIR, SGLANG_REFILL_BATCH)
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
                        if _is_master(ddp):
                            print(f"[rollout][skip] spawn failed: {e}", flush=True)
                else:
                    if _is_master(ddp):
                        why=[]
                        if not cool_ok: why.append("cooldown")
                        if not mem_ok:  why.append(f"free_mem={free_mb}MB<{ROLL_MIN_FREE_MB}MB")
                        print(f"[rollout][skip] pool={pool_est} reasons={'+'.join(why)}", flush=True)

        # 2) 从池里取一批（不足则用 actor 现场补齐；使用纯贪心）
        batch = dequeue_items(SGLANG_SYNC_DIR, batch_size)
        if len(batch) < batch_size:
            for _ in range(batch_size - len(batch)):
                ids = random.choice(PROMPT_TOKEN_IDS)
                ids_t=torch.tensor(ids,dtype=torch.long,device=dev).unsqueeze(0)
                prompt_len=ids_t.size(1)
                room = block_size - prompt_len - 1
                if room <= 0:
                    continue
                out = _greedy_decode_argmax(raw_actor, ids_t, max_new_tokens=max(8,room),
                                            eos_id=tok.eos_id, block_size=block_size)
                batch.append({"prompt_ids": ids, "full_ids": out[0].tolist()})

        # 3) 打包 Samples
        samples = pack_samples(batch, pad_id=0, block_size=block_size, device=torch.device(dev))

        # >>> 若这一批没有任何 response token，则现场补样/跳过（纯贪心）
        if int(samples.action_mask.sum().item()) == 0:
            tries = 0
            while int(samples.action_mask.sum().item()) == 0 and tries < 8:
                ids = random.choice(PROMPT_TOKEN_IDS)
                ids_t = torch.tensor(ids, dtype=torch.long, device=dev).unsqueeze(0)
                prompt_len = ids_t.size(1)
                room = block_size - prompt_len - 1
                if room > 0:
                    out = _greedy_decode_argmax(raw_actor, ids_t, max_new_tokens=max(8, room),
                                                eos_id=tok.eos_id, block_size=block_size)
                    gen = {"prompt_ids": ids, "full_ids": out[0].tolist()}
                    batch.append(gen)
                samples = pack_samples(batch[-batch_size:], pad_id=0, block_size=block_size, device=torch.device(dev))
                tries += 1
            if int(samples.action_mask.sum().item()) == 0:
                if _is_master(ddp):
                    pool_now = estimate_size(SGLANG_SYNC_DIR, SGLANG_REFILL_BATCH) if (SGLANG_ON and SGLANG_OFFLINE) else -1
                    print(f"[iter {iter_num:4d}] skip(empty-response-batch) pool={pool_now}", flush=True)
                continue

        # 4) 评估经验 & 5) 参数更新
        experiences, report_kl, r_raw, r_shaped, r_ctr = trainer.evaluate_experience(samples)

        # --- 自适应 KL 系数 ---
        kl_target = 0.15
        if not np.isnan(report_kl):
            err = report_kl / max(kl_target, 1e-8) - 1.0
            trainer.kl_ctl *= float(np.exp(np.clip(err, -0.05, 0.05)))  # 每步最多 ±5%

        # 记录 EMA
        r_raw_ema = _ema_update(r_raw_ema, r_raw, alpha=0.1)

        pl, vl = [], []
        for exp in experiences:
            out = trainer.train_on_experience(exp, use_token_entropy=use_token_entropy)
            if isinstance(out, tuple):
                p,v = out; pl.append(float(p.detach().item())); vl.append(float(v.detach().item()))
            else:
                pl.append(float(out.detach().item()))
        mean_p = float(np.mean(pl)) if pl else 0.0
        mean_v = float(np.mean(vl)) if vl else None

        # ========== 固定评测集 r_eval_raw ==========
        r_eval_raw = float("nan")
        if (iter_num % EVAL_LOG_EVERY == 0) and _is_master(ddp) and len(EVAL_PROMPT_IDS) > 0:
            try:
                r_eval_raw = eval_fixed_raw_reward(
                    actor_model=raw_actor, gpt2_tok=tok, eval_prompt_ids=EVAL_PROMPT_IDS,
                    reward_tokenizer=reward_tokenizer, reward_model=reward_model,
                    block_size=block_size, max_new_eval=min(SGLANG_MAX_NEW, 128)  # 保留 128 上限
                )
            except Exception:
                r_eval_raw = float("nan")

        # 6) 日志 & ckpt
        if (iter_num % EVAL_LOG_EVERY == 0) and _is_master(ddp):
            pool_now = estimate_size(SGLANG_SYNC_DIR, SGLANG_REFILL_BATCH) if (SGLANG_ON and SGLANG_OFFLINE) else -1

            stats = getattr(trainer, "last_stats", {}) or {}
            clip_frac     = stats.get("clip_frac", float("nan"))
            approx_kl_pi  = stats.get("approx_kl_pi", float("nan"))
            entropy_tok   = stats.get("entropy", float("nan"))
            v_mae         = stats.get("v_mae", float("nan"))
            explained_var = stats.get("explained_var", float("nan"))

            core = (
                f"[iter {iter_num:4d}] "
                f"p={mean_p:.4f} "
                + (f"v={mean_v:.4f} " if mean_v is not None else "")
                + f"kl={report_kl:.6f} r_raw={r_raw:.4f} r_raw_ema={r_raw_ema:.4f} "
                  f"r_ctr={r_ctr:.4f} r_shp={r_shaped:.4f} r_eval_raw={r_eval_raw:.4f} "
                  f"clip={clip_frac:.3f} akl_pi={approx_kl_pi:.4f} H={entropy_tok:.3f} "
                  f"v_mae={v_mae:.4f} ev={explained_var:.3f} pool={pool_now}"
            )
            print(core, flush=True)

            hdr = not os.path.exists(METRICS_CSV)
            with open(METRICS_CSV, "a") as f:
                if hdr:
                    f.write(
                        "iter,p_loss,v_loss,avg_kl,r_raw,r_raw_ema,r_ctr,r_shp,r_eval_raw,"
                        "clip_frac,approx_kl_pi,entropy,v_mae,explained_var,pool_est\n"
                    )
                f.write(
                    f"{iter_num},{mean_p},{'' if mean_v is None else mean_v},"
                    f"{report_kl},{r_raw},{r_raw_ema},{r_ctr},{r_shaped},{r_eval_raw},"
                    f"{clip_frac},{approx_kl_pi},{entropy_tok},{v_mae},{explained_var},{pool_now}\n"
                )

            if DEBUG_SAMPLE_EVERY and (iter_num % DEBUG_SAMPLE_EVERY == 0) and samples.seqs.size(0) > 0:
                try:
                    i0 = int(torch.randint(0, samples.seqs.size(0), (1,)).item())
                    L0 = int(samples.attention_mask[i0].sum().item())
                    txt0 = tok.decode(samples.seqs[i0, :L0].detach().cpu())
                    r0 = experiences[0].reward[i0].item() if isinstance(experiences, list) and len(experiences)>0 else float("nan")
                    print(f"[sample] reward={r0:.4f} text={txt0[:200].replace(chr(10),' ')}", flush=True)
                except Exception as e:
                    print(f"[sample] skip(print) due to error: {e}", flush=True)

            algo = "PPO" if use_ppo else ("GRPO" if use_grpo else "DAPO")
            ckpt = {
                'model': raw_actor.state_dict(), 'critic': critic.state_dict(), 'ref': ref.state_dict(),
                'optimizer_actor': opt_a.state_dict(), 'optimizer_critic': opt_c.state_dict(),
                'vocab_size': model_args.get('vocab_size', 50304), 'iter_num': iter_num,
            }
            torch.save(ckpt, os.path.join(out_dir, f"{algo}_ckpt.pt"))

    if ddp: destroy_process_group()

if __name__ == "__main__":
    main()
