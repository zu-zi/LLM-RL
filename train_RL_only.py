# train_RL_only.py
import os, sys, time, random, json, subprocess, glob
os.environ["PPO_RATIO_MIN"] = "0.75"   # 重要性采样下限更紧
os.environ["PPO_RATIO_MAX"] = "1.25"   # 上限更紧，防止爆比值
os.environ["PPO_KL_TOKEN_CAP"] = "0.5" # 夹紧 Δlogp 幅度（配合 PPO.K3_CAP）
os.environ["PPO_K3_CAP"] = "1.5"       # 对 k3 再上限，削尖峰
os.environ["PPO_ENT_MASK_KEEP"] = "0.1"  # 熵正则子采样更保守
os.environ["ROLL_MIN_RESP_TOKENS"]="16"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("NANOGPT_SILENT_INIT", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")  # 降噪
MIN_RESP_TOK = int(os.getenv("ROLL_MIN_RESP_TOKENS", "16"))
import numpy as np
import torch, tiktoken
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from RL.PPO import PPOTrainer, Critic
from RL.GRPO import GRPOTrainer
from RL.DAPO import DAPOTrainer
from RL.common import (
    Samples,
    normalize_for_reward,
    compute_actor_ref_logprobs,
    masked_mean,
    forward_values_via_actor,   # 用于“只训 critic”路径
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils.rollout_pool import dequeue_items, estimate_size, ensure_dir

# ================== Core switches ==================
use_ppo = True
use_grpo = False
use_dapo  = False
use_token_entropy = False        # 开：逐 token 熵正则
ENT_MASK_KEEP=0.2

# ================== IO / runtime ==================
out_dir = "/root/autodl-tmp/Results"
eval_interval = 8
max_iters = 1000
wandb_log = False
wandb_project = "hlhf"
wandb_run_name = "run"

# ================== System / model init ==================
init_from = "gpt2-large"        # 继续训练时改成 "resume"
RESUME_CKPT = None              # 或显式指定 ckpt 路径
backend = "nccl"
device  = "cuda"
compile = False

# GPT 结构/初始化（从 gpt2 权重时这些会被覆盖到相同值；从 scratch 时生效）
n_layer = 12; n_head = 12; n_embd = 768
block_size = 384
dropout = 0.0
bias = False

# ================== Optim / PPO ==================
batch_size = 4
gradient_accumulation_steps = 2  # 当前 Trainer 内部自行 step，这里留作将来扩展

# ——基础超参（PPO/熵系数在 RL/PPO.py 内部，如需可再改 Trainer）——
RL_learning_rate = 1.5e-6
kl_ctl  = 0.7

# 优化器通用超参
weight_decay = 5e-3
beta1 = 0.9
beta2 = 0.95

# ================== Reward model (EN) ==================
REWARD_MODEL_NAME = "OpenAssistant/reward-model-deberta-v3-large-v2"

# ================== sglang 离线池（离线补货生成） ==================
SGLANG_ON = True
SGLANG_OFFLINE = True
SGLANG_MODEL_PATH = "gpt2-large"
SGLANG_SYNC_DIR   = "/root/autodl-tmp/sgl_pool"

# ——保新鲜，缓解 (!stale) 与 KL 尖峰——
SGLANG_ROLLOUT_TARGET = 96
SGLANG_REFILL_BATCH  = 48
SGLANG_MAX_NEW       = 48  # 先稳住 KL，后面再拉回？

# 调度阈值（别让池老化）
ROLL_LOW_WATERMARK_FACTOR = 3
ROLL_REFILL_COUNT = 24      # 从 12 → 24，补货更积极一些
ROLL_COOLDOWN_SEC = 18
ROLL_MIN_FREE_MB  = 6000

# 每个迭代保证“新鲜样本占比”（在线条数）
FRESH_RATIO = 0.50
POOL_STALE_WARN_SEC = 600  # 10min

# ===== 统一采样口径（与 rollout_worker.py 对齐）=====
SAMPLE_TEMPERATURE = 0.8
SAMPLE_TOP_P = 0.9
SAMPLE_TOP_K = 0
SAMPLE_REP_PENALTY = 1.1
SAMPLE_STOPS = ["\nHuman:", "\n\nHuman:"]

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

# ========= 奖励模型 CPU 包装器 =========
class RewardOnCPU(torch.nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model.eval()
        self._device = torch.device("cpu")
        self.model.to(self._device)
    @property
    def device(self): return self._device
    def forward(self, **kwargs):
        moved = {k: (v.to(self._device) if torch.is_tensor(v) else v) for k, v in kwargs.items()}
        with torch.no_grad():
            return self.model(**moved)
    def parameters(self, *args, **kwargs):
        return self.model.parameters(*args, **kwargs)
    def eval(self):
        self.model.eval(); return self

# ========= 数据/固定 prompts =========
def _load_prompts():
    pf=os.path.join(os.path.dirname(__file__),"data/RL_dataset/prompt.bin")
    print(f"Loading fixed prompts from {pf} ...")
    blob=torch.load(pf,map_location="cpu")
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

# ========= 贪心解码（保留以兼容旧评测）=========
# @torch.no_grad()
# def _greedy_decode_argmax(actor_model, idx, max_new_tokens, eos_id, block_size):
#     device = idx.device
#     was_train = actor_model.training
#     actor_model.eval()
#     try:
#         out = idx
#         for _ in range(int(max_new_tokens)):
#             idx_cond = out[:, -int(block_size):]
#             logits = actor_model(idx_cond)
#             if isinstance(logits, tuple):  # 兼容 (logits, ...)
#                 logits = logits[0]
#             next_token_logits = logits[:, -1, :]
#             next_id = torch.argmax(next_token_logits, dim=-1).view(1,1).to(device)
#             out = torch.cat((out, next_id), dim=1)
#             if eos_id is not None and int(next_id.item()) == int(eos_id):
#                 break
#         return out
#     finally:
#         if was_train: actor_model.train()

# ========= 采样解码（与 rollout 完全对齐）=========
@torch.no_grad()
def _sample_next_token(logits, top_p=0.9, temperature=0.75, top_k=0, repetition_penalty=1.1, prev=None):
    if repetition_penalty and prev is not None and prev.numel() > 0:
        uniq = torch.unique(prev)
        logits[:, uniq] = logits[:, uniq] / float(repetition_penalty)
    logits = logits / max(float(temperature), 1e-6)
    if top_k and top_k > 0:
        kth = torch.topk(logits, k=min(int(top_k), logits.size(-1)), dim=-1).values[..., -1:]
        logits = torch.where(logits < kth, torch.full_like(logits, -1e10), logits)
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    cutoff = (cumsum > float(top_p)).float().argmax(dim=-1, keepdim=True)
    mask = torch.arange(probs.size(-1), device=probs.device).view(1, -1) <= cutoff
    kept = torch.where(mask, sorted_probs, torch.zeros_like(sorted_probs))
    kept = kept / kept.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    next_sorted = torch.multinomial(kept, num_samples=1)
    next_id = sorted_idx.gather(1, next_sorted)  # [1,1]
    return next_id

# @torch.no_grad()
# def _decode_with_sampling(model, idx, max_new_tokens, eos_id, block_size,
#                           temperature=0.75, top_p=0.9, top_k=0, repetition_penalty=1.1,
#                           stop_strs=None, tokenizer_decode=None):
#     device = idx.device
#     was_train = model.training
#     model.eval()
#     try:
#         out = idx
#         for _ in range(int(max_new_tokens)):
#             idx_cond = out[:, -int(block_size):]
#             logits = model(idx_cond)
#             if isinstance(logits, tuple):
#                 logits = logits[0]
#             last = logits[:, -1, :]
#             next_id = _sample_next_token(
#                 last, top_p=top_p, temperature=temperature, top_k=top_k,
#                 repetition_penalty=repetition_penalty, prev=out
#             )
#             out = torch.cat((out, next_id.to(device)), dim=1)
#             # eos 停
#             if eos_id is not None and int(next_id.item()) == int(eos_id):
#                 break
#             # 明确字符串停词
#             if stop_strs and tokenizer_decode is not None:
#                 tail = tokenizer_decode(out[0][-min(out.size(1), block_size):].tolist())
#                 if any(s in tail for s in stop_strs):
#                     break
#         return out
#     finally:
#         if was_train: model.train()

@torch.no_grad()
def _decode_with_sampling(model, idx, max_new_tokens, eos_id, block_size,
                          temperature=0.75, top_p=0.9, top_k=0, repetition_penalty=1.1,
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

            # 轻度去重复 + 温度/核采样
            if repetition_penalty and out.numel() > 0:
                uniq = torch.unique(out)
                last[:, uniq] = last[:, uniq] / float(repetition_penalty)
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
            next_id = sorted_idx.gather(1, next_sorted)  # [1,1]

            # 如果还没达到最短响应长度 -> 禁止 EOS
            if (out.size(1) - start_len) < int(min_resp) and eos_id is not None and int(next_id.item()) == int(eos_id):
                # 直接改抽到的 id 为概率里的第二名，避免卡死；退而求其次：
                alt = sorted_idx[:, 1:2] if sorted_idx.size(1) > 1 else next_id
                next_id = alt

            out = torch.cat((out, next_id.to(device)), dim=1)

            # 仅在达到最短长度之后，才允许这些停条件生效
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



# ========= 显存检测 =========
def _cuda_free_mb(device_str="cuda"):
    try:
        if not torch.cuda.is_available():
            return 0
        free, total = torch.cuda.mem_get_info(torch.device(device_str))
        return int(free // (1024 * 1024))
    except Exception:
        return 0

# ========= 池子新鲜度（文件年龄）=========
def _pool_freshness(dir_path: str, take_last:int=50):
    try:
        fs = sorted(glob.glob(os.path.join(dir_path, "roll_*.jsonl")), key=os.path.getmtime)
        if not fs: return 0, (float("inf"), float("inf"))
        fs = fs[-min(len(fs), max(1,int(take_last))):]
        ages = [time.time() - os.path.getmtime(p) for p in fs]
        med = float(np.median(ages))
        p90 = float(np.percentile(ages, 90))
        return len(fs), (med, p90)
    except Exception:
        return 0, (float("inf"), (float("inf")))

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
        "--mb", "4",
        "--use-only-train",
        "--min-resp","16",
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

# ========= EMA =========
def _ema_update(prev, x, alpha=0.1):
    if prev is None or np.isnan(prev): return float(x)
    return float(alpha * x + (1.0 - alpha) * prev)

# ========= RM 打分（贪心口径，保留） =========
# @torch.no_grad()
# def eval_fixed_raw_reward(actor_model, gpt2_tok, eval_prompt_ids, reward_tokenizer, reward_model, block_size, max_new_eval=128):
#     dev = next(actor_model.parameters()).device
#     eos_id = gpt2_tok.eos_id
#     texts = []
#     was_training = actor_model.training
#     actor_model.eval()
#     try:
#         for ids in eval_prompt_ids:
#             ids_t = torch.tensor(ids, dtype=torch.long, device=dev).unsqueeze(0)
#             prompt_len = ids_t.size(1)
#             room = block_size - prompt_len - 1
#             if room <= 0:
#                 full_ids = ids[:block_size]
#                 texts.append(gpt2_tok.decode(full_ids)); continue
#             gen_len = max(8, min(int(max_new_eval), int(room)))
#             out = _greedy_decode_argmax(actor_model, ids_t, gen_len, eos_id, block_size)
#             full = out[0].tolist()[:block_size]
#             texts.append(gpt2_tok.decode(full))
#     finally:
#         if was_training: actor_model.train()

#     texts = [normalize_for_reward(t, reward_tokenizer) for t in texts]
#     toks = reward_tokenizer(texts, padding=True, truncation=True, max_length=1024, return_tensors="pt")
#     outs = reward_model(**toks)
#     logits = getattr(outs, "logits", None)
#     if logits is None: return float("nan")
#     if logits.dim() == 2 and logits.size(-1) == 1: logits = logits.squeeze(-1)
#     return float(np.mean([float(v) for v in logits.detach().cpu().tolist()])) if len(texts) > 0 else float("nan")

# ========= 采样口径评测（与 rollout 对齐）=========
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

# ========= 主流程 =========
def main():
    _apply_cli_config_once()

    # 依赖外部 config 的量
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
    pad_id = tok.eos_id

    # ====== 训练/评测索引严格构造 ======
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
        rng = np.random.RandomState(int(seed))
        eval_count = min(16, len(PROMPT_TOKEN_IDS))
        choice = rng.choice(len(PROMPT_TOKEN_IDS), size=eval_count, replace=False) if eval_count>0 else []
        EVAL_PROMPT_IDS = [PROMPT_TOKEN_IDS[int(i)] for i in choice]

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

    # ========== 在 CPU 构建 base_state ==========
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

    # ========== ref / actor / critic ==========
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
    lr_c = max(2e-6, RL_learning_rate * 1.5)  # critic 更快跟上
    weight_decay_c = 1e-3
    try:
        from bitsandbytes.optim import AdamW8bit
        opt_a=AdamW8bit(raw_actor.parameters(), lr=RL_learning_rate, betas=(beta1,beta2), weight_decay=weight_decay)
        opt_c=AdamW8bit(critic.parameters(),    lr=lr_c, betas=(beta1,beta2), weight_decay=weight_decay_c)
        print("[optim] using bitsandbytes AdamW8bit")
    except Exception as e:
        print(f"[optim] bitsandbytes not available ({e}), fallback to torch.optim.AdamW")
        opt_a=torch.optim.AdamW(raw_actor.parameters(), lr=RL_learning_rate, betas=(beta1,beta2), weight_decay=weight_decay)
        opt_c=torch.optim.AdamW(critic.parameters(),    lr=lr_c, betas=(beta1,beta2), weight_decay=weight_decay_c)

    # ====== resume（可选）======
    iter_start = 0
    if isinstance(init_from, str) and init_from.lower() == "resume":
        algo = "PPO" if use_ppo else ("GRPO" if use_grpo else "DAPO")
        ckpt_path = RESUME_CKPT or os.path.join(out_dir, f"{algo}_ckpt.pt")
        print(f"[resume] loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=dev)
    
        # 模型
        raw_actor.load_state_dict(ckpt["model"])
        critic.load_state_dict(ckpt["critic"])
        try:
            ref.load_state_dict(ckpt["ref"])
        except Exception as e:
            print(f"[resume] warn: ref state not fully matched ({e}); continue.")
    
        # 优化器
        try:
            opt_a.load_state_dict(ckpt["optimizer_actor"])
            opt_c.load_state_dict(ckpt["optimizer_critic"])
        except Exception as e:
            print(f"[resume] warn: optimizer states not loaded ({e}); continue with fresh.")
    
        # 迭代起点
        iter_start = int(ckpt.get("iter_num", 0)) + 1
        print(f"[resume] resume from iter={iter_start}")
    else:
        iter_start = 0

    # 奖励模型（CPU）
    rw_name = REWARD_MODEL_NAME
    print(f"[reward] loading {rw_name} on CPU ...")
    reward_hf = AutoModelForSequenceClassification.from_pretrained(rw_name, device_map="cpu", torch_dtype=torch.float32).eval()
    reward_model = RewardOnCPU(reward_hf)
    reward_tokenizer=AutoTokenizer.from_pretrained(rw_name, use_fast=True)
    try: reward_tokenizer.padding_side = "right"
    except Exception: pass
    if getattr(reward_tokenizer, "pad_token", None) is None and getattr(reward_tokenizer, "eos_token", None) is not None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token

    # 训练器
    if use_ppo:
        trainer=PPOTrainer(
            actor_model=raw_actor, ref_model=ref, reward_model=reward_model,
            critic_model=critic, actor_tokenizer=tok, reward_tokenizer=reward_tokenizer,
            optimizer_actor=opt_a, optimizer_critic=opt_c,
            device=dev,
            mb_size_logits=1,
            mb_size_values=1,
            kl_ctl=kl_ctl,
            ppo_clip=ppo_clip,
            entropy_coef=entropy_coef,
            max_grad_norm=1.0,
            vf_clip=0.2,
            # use_token_entropy=use_token_entropy,
            # ent_keep_ratio=ent_keep_ratio,
        )
    elif use_grpo:
        trainer = GRPOTrainer(
            actor_model=raw_actor, ref_model=ref, reward_model=reward_model,
            actor_tokenizer=tok, reward_tokenizer=reward_tokenizer,
            optimizer_actor=opt_a, device=dev,
            group_size=GRPO_GROUP_SIZE,
            kl_coef=GRPO_KL_COEF,
            clip_reward=GRPO_CLIP_REWARD,
            mb_size_logits=MB_SIZE_LOGITS,
            block_size=block_size,
            max_new_tokens=min(SGLANG_MAX_NEW, 96),
            # use_token_entropy=use_token_entropy,
            # ent_keep_ratio=ent_keep_ratio,
        )
    elif use_dapo:
        trainer = DAPOTrainer(
            actor_model=raw_actor, ref_model=ref, reward_model=reward_model,
            actor_tokenizer=tok, reward_tokenizer=reward_tokenizer,
            optimizer_actor=opt_a,
            device=dev,
            group_size=group_size,
            kl_coef=kl_coef,
            beta=beta,
            adv_norm=adv_norm,
            adv_clip=adv_clip,
            mb_size_logits=MB_SIZE_LOGITS,
            max_new_tokens=min(DAPO_MAX_NEW, SGLANG_MAX_NEW),
            min_resp_tokens=MIN_RESP_TOK,
            block_size=block_size,
            # use_token_entropy=use_token_entropy,
            # ent_keep_ratio=ent_keep_ratio,
        )


    # ========== 训练循环 ==========
    if _is_master(ddp) and SGLANG_ON and SGLANG_OFFLINE:
        print(f"[sglang] offline pool enabled: dir={SGLANG_SYNC_DIR}, target={SGLANG_ROLLOUT_TARGET}, batch={SGLANG_REFILL_BATCH}")
    ensure_dir(SGLANG_SYNC_DIR)
    os.makedirs(ROLLOUT_LOG_DIR, exist_ok=True)

    r_raw_ema = None
    last_rollout_t = 0.0

    # ====== KL 迟滞（hysteresis）控制器 ======
    KL_HALT   = 0.12     # 连续命中该阈值 → 冻结 actor
    KL_RESUME = 0.08     # 连续低于该阈值 → 解冻 actor
    HALT_STREAK = 2
    RESUME_STREAK = 2
    actor_frozen = False
    halt_hits = 0
    resume_hits = 0
    freeze_steps = 0   # 连续冻结计数

    # ====== 紧急保险丝状态 ======
    FORCE_FRESH_STEPS = 4
    EMERG_SHORT_GEN_STEPS = 4

    # 保存基础 LR
    BASE_LR_ACTOR = RL_learning_rate

    for iter_num in range(iter_start, max_iters+1):
        # ---- 池子估算 + 新鲜度观测 ----
        pool_est = estimate_size(SGLANG_SYNC_DIR, SGLANG_REFILL_BATCH) if (SGLANG_ON and SGLANG_OFFLINE) else -1
        n_files, (age_med, age_p90) = _pool_freshness(SGLANG_SYNC_DIR) if (SGLANG_ON and SGLANG_OFFLINE) else (0,(float("inf"),float("inf")))

        # ---- 离线补货（仅主进程）----
        if _is_master(ddp) and SGLANG_ON and SGLANG_OFFLINE:
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
            want_fresh = batch_size
            batch = []
            if force_fresh:
                FORCE_FRESH_STEPS -= 1
        else:
            want_fresh = want_fresh_base
            batch = dequeue_items(SGLANG_SYNC_DIR, batch_size - want_fresh) if (SGLANG_ON and SGLANG_OFFLINE) else []

        # 在线“新鲜样本”改为采样解码（统一与 rollout）
        # def _gen_one_from_train(use_ref: bool):
        #     ids = random.choice(TRAIN_PROMPT_IDS)
        #     ids_t = torch.tensor(ids, dtype=torch.long, device=dev).unsqueeze(0)
        #     room = block_size - ids_t.size(1) - 1
        #     if room <= 0: return None
        #     max_new_cap = SGLANG_MAX_NEW
        #     if use_ref or EMERG_SHORT_GEN_STEPS > 0:
        #         max_new_cap = min(max_new_cap, 32)  # 紧急/冻结：更短生成
        #     out = _decode_with_sampling(
        #         ref if use_ref else raw_actor, ids_t,
        #         max_new_tokens=max(8, min(room, max_new_cap)),
        #         eos_id=tok.eos_id, block_size=block_size,
        #         temperature=SAMPLE_TEMPERATURE, top_p=SAMPLE_TOP_P, top_k=SAMPLE_TOP_K,
        #         repetition_penalty=SAMPLE_REP_PENALTY,
        #         stop_strs=SAMPLE_STOPS, tokenizer_decode=tok.decode
        #     )
        #     return {"prompt_ids": ids, "full_ids": out[0].tolist()}

        def _gen_one_from_train(use_ref: bool):
            ids = random.choice(TRAIN_PROMPT_IDS)
            ids_t = torch.tensor(ids, dtype=torch.long, device=dev).unsqueeze(0)
            room = block_size - ids_t.size(1) - 1
            if room <= 0: return None
            max_new_cap = SGLANG_MAX_NEW
            if use_ref or EMERG_SHORT_GEN_STEPS > 0:
                max_new_cap = min(max_new_cap, 32)
        
            # 允许出现下一轮 Assistant，但必须先写够最短回复
            stop_list = ["\nHuman:", "\n\nHuman:", "\nAssistant:"]
        
            tries = 0
            while tries < 6:
                tries += 1
                out = _decode_with_sampling(
                    ref if use_ref else raw_actor, ids_t,
                    max_new_tokens=max(8, min(room, max_new_cap)),
                    eos_id=tok.eos_id, block_size=block_size,
                    temperature=SAMPLE_TEMPERATURE, top_p=SAMPLE_TOP_P, top_k=SAMPLE_TOP_K,
                    repetition_penalty=SAMPLE_REP_PENALTY,                       # 稍强一点，缓解复读
                    stop_strs=stop_list, tokenizer_decode=tok.decode,
                    min_resp=MIN_RESP_TOK                          # 关键：达到最短才可停
                )
                resp_len = out.size(1) - ids_t.size(1)
                if resp_len >= MIN_RESP_TOK:
                    return {"prompt_ids": ids, "full_ids": out[0].tolist()}
            return None


        # 补“新鲜样本”
        fresh = []
        for _ in range(want_fresh):
            g = _gen_one_from_train(use_ref=actor_frozen)
            if g is not None: fresh.append(g)
        batch.extend(fresh)

        # 若仍不足，再补在线生成
        while len(batch) < batch_size:
            g = _gen_one_from_train(use_ref=actor_frozen)
            if g is None: break
            batch.append(g)

        # 打包 Samples
        samples = pack_samples(batch, pad_id=tok.eos_id, block_size=block_size, device=torch.device(dev))

        # 若整批没有 response，继续尝试几次；仍不行则跳过
        if int(samples.action_mask.sum().item()) == 0:
            tries = 0
            while int(samples.action_mask.sum().item()) == 0 and tries < 8:
                g = _gen_one_from_train(use_ref=actor_frozen)
                if g is not None: batch.append(g)
                samples = pack_samples(batch[-batch_size:], pad_id=tok.eos_id, block_size=block_size, device=torch.device(dev))
                tries += 1
            if int(samples.action_mask.sum().item()) == 0:
                if _is_master(ddp):
                    print(f"[iter {iter_num:4d}] skip(empty-response-batch) pool={pool_est}", flush=True)
                continue

        # 评估经验（不会更新参数）
        experiences, report_kl, r_raw, r_shaped, r_ctr, safe_kl = trainer.evaluate_experience(samples)

        # === 硬保险丝：异常 KL → 冻结 actor + 只训 critic（不再整步 continue） ===
        critic_only_this_iter = False
        if (not np.isfinite(safe_kl)) or (safe_kl > 1.7):
            actor_frozen = True
            critic_only_this_iter = True
            FORCE_FRESH_STEPS = max(FORCE_FRESH_STEPS, 3)
            EMERG_SHORT_GEN_STEPS = max(EMERG_SHORT_GEN_STEPS, 3)
            if _is_master(ddp):
                print(f"[guard] skip actor (abnormal KL={report_kl:.4g}) -> critic-only this iter", flush=True)

        _stats = getattr(trainer, "last_stats", {}) or {}
        clip_frac = float(_stats.get("clip_frac", float("nan")))
        if np.isnan(clip_frac): clip_frac = 0.0

        # ---- Emergency KL fuse (hard stop on actor) ----
        if np.isfinite(safe_kl) and safe_kl > 1.7:
            actor_frozen = True
            FORCE_FRESH_STEPS = max(FORCE_FRESH_STEPS, 3)
            EMERG_SHORT_GEN_STEPS = max(EMERG_SHORT_GEN_STEPS, 3)
            trainer.kl_ctl = float(min(2.0, trainer.kl_ctl * 1.2))
            for g in opt_a.param_groups:
                g['lr'] = max(BASE_LR_ACTOR * 0.25, 8e-7)

        # 紧急“短生成”计时递减
        if EMERG_SHORT_GEN_STEPS > 0:
            EMERG_SHORT_GEN_STEPS -= 1

        # --- 自适应 KL（温和） ---
        kl_target = 0.3
        if np.isfinite(safe_kl):
            err = safe_kl / max(kl_target, 1e-8) - 1.0
            up = 0.05; down = 0.03
            trainer.kl_ctl *= float(np.exp(np.clip(err, -down, up)))
            trainer.kl_ctl = float(np.clip(trainer.kl_ctl, 0.05, 2.0))

        r_raw_ema = _ema_update(r_raw_ema, r_raw, alpha=0.1)

        # === KL 驱动的临时降学习率（只作用于 actor；不改基础超参）===
        KL_STOP = 1.5
        if np.isfinite(safe_kl):
            err = max(0.0, safe_kl / KL_STOP - 1.0)
            scale = 1.0 / (1.0 + 2.0 * err)
            scale = float(max(0.25, min(1.0, scale)))
        else:
            scale = 0.25
        for g in opt_a.param_groups:
            g['lr'] = BASE_LR_ACTOR * scale

        # ===== KL 迟滞状态机：决定是否冻结/解冻 actor =====
        if np.isfinite(safe_kl):
            if safe_kl > KL_HALT:
                halt_hits += 1; resume_hits = 0
            elif safe_kl < KL_RESUME:
                resume_hits += 1; halt_hits = 0
            else:
                halt_hits = 0; resume_hits = 0

            if (not actor_frozen) and (halt_hits >= HALT_STREAK):
                actor_frozen = True; halt_hits = 0
                if _is_master(ddp): print(f"[guard] freeze actor (kl={report_kl:.3f})", flush=True)

            if actor_frozen and (resume_hits >= RESUME_STREAK):
                actor_frozen = False; resume_hits = 0; freeze_steps = 0
                if _is_master(ddp): print(f"[guard] unfreeze actor (kl={report_kl:.3f})", flush=True)

        # 记录连续冻结步数 & 长期冻结回退
        if actor_frozen:
            freeze_steps += 1
            if freeze_steps >= 8:  # 连续冻结过久则增强回拉
                trainer.kl_ctl = float(min(1.2, trainer.kl_ctl * 1.2))
                for g in opt_a.param_groups:
                    g['lr'] = max(BASE_LR_ACTOR * 0.4, 6e-7)
                if _is_master(ddp):
                    print(f"[guard] long-freeze fallback: kl_ctl={trainer.kl_ctl:.3f}, actor_lr={opt_a.param_groups[0]['lr']:.2e}", flush=True)
        else:
            freeze_steps = 0

        # === policy/critic 训练 ===
        if not experiences:
            continue

        pl, vl = [], []

        if actor_frozen or critic_only_this_iter:
            # ------- 只训 critic（不更新 actor）-------
            for exp in experiences:
                critic.train()
                opt_c.zero_grad(set_to_none=True)
                values_full = forward_values_via_actor(
                    trainer.actor, trainer.critic, exp.seqs, trainer.device_type,
                    ptdtype=None, micro_batch_size=getattr(trainer, "mb_values", 1), detach_hidden=True
                )  # [B, T]
                values_new = values_full[:, 1:]  # 对齐 action 轴
                v_loss_tok = torch.nn.functional.huber_loss(
                    values_new, exp.returns, delta=1.0, reduction='none'
                )
                v_loss = (v_loss_tok * exp.action_mask).sum() / exp.action_mask.sum().clamp_min(1e-8)
                v_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainer.critic.parameters(), 0.5)
                trainer.opt_critic.step()
                vl.append(float(v_loss.detach().item()))
            mean_p = 0.0
            mean_v = float(np.mean(vl)) if vl else None
        else:
            # ------- 正常 PPO 训练（actor+critic）-------
            KL_EPOCH_DROP = 1.10
            LOW_EPOCH = 1
            HIGH_EPOCH = 2
            POLICY_EPOCHS = LOW_EPOCH if (np.isfinite(safe_kl) and safe_kl > KL_EPOCH_DROP) else HIGH_EPOCH

            if (not actor_frozen) and (iter_num > iter_start + 4) \
               and np.isfinite(safe_kl) and (safe_kl < 0.25) and (clip_frac < 0.01):
                POLICY_EPOCHS = 3
                for g in opt_a.param_groups:
                    g['lr'] = min(BASE_LR_ACTOR * 1.2, 5e-6)

            for _ in range(POLICY_EPOCHS):
                for exp in experiences:
                    out = trainer.train_on_experience(exp, use_token_entropy=use_token_entropy)
                    if isinstance(out, tuple):
                        p, v = out
                        pl.append(float(p.detach().item()))
                        vl.append(float(v.detach().item()))
                    else:
                        pl.append(float(out.detach().item()))
            mean_p = float(np.mean(pl)) if pl else 0.0
            mean_v = float(np.mean(vl)) if vl else None

        # 固定评测集：保留greedy，同时新增sampled（与rollout对齐）
        # r_eval_raw     = float("nan")  # greedy口径（旧字段，保持原名）
        r_eval_sampled = float("nan")  # 新增：采样口径
        if (iter_num % EVAL_LOG_EVERY == 0) and _is_master(ddp) and len(EVAL_PROMPT_IDS) > 0:
            # try:
            #     r_eval_raw = eval_fixed_raw_reward(
            #         actor_model=raw_actor, gpt2_tok=tok, eval_prompt_ids=EVAL_PROMPT_IDS,
            #         reward_tokenizer=reward_tokenizer, reward_model=reward_model,
            #         block_size=block_size, max_new_eval=min(SGLANG_MAX_NEW, 64)
            #     )
            # except Exception:
            #     r_eval_raw = float("nan")
            try:
                r_eval_sampled = eval_fixed_raw_reward_sampled(
                    actor_model=raw_actor, gpt2_tok=tok, eval_prompt_ids=EVAL_PROMPT_IDS,
                    reward_tokenizer=reward_tokenizer, reward_model=reward_model,
                    block_size=block_size, max_new_eval=min(SGLANG_MAX_NEW, 64)
                )
            except Exception:
                r_eval_sampled = float("nan")

        # 日志 & ckpt（主进程）
        if (iter_num % EVAL_LOG_EVERY == 0) and _is_master(ddp):
            stats = getattr(trainer, "last_stats", {}) or {}
            clip_frac     = stats.get("clip_frac", float("nan"))
            approx_kl_pi  = stats.get("approx_kl_pi", float("nan"))
            entropy_tok   = stats.get("entropy", float("nan"))
            v_mae         = stats.get("v_mae", float("nan"))
            explained_var = stats.get("explained_var", float("nan"))
            ratio_qs   = stats.get("ratio_q50_q90_q99", (float("nan"),)*3)
            ratio_max  = stats.get("ratio_max",  float("nan"))
            adv_abs_m  = stats.get("adv_abs_mean", float("nan"))
            sel_tokens = stats.get("sel_tokens", 0)
            ppo_clip_v = stats.get("ppo_clip", float("nan"))
            kl_ctl_now = stats.get("kl_ctl_now", float("nan"))
            cur_lr     = opt_a.param_groups[0]['lr']

            stale_note = ""
            if age_med != float("inf"):
                stale_note = f" age_med={age_med:.0f}s age_p90={age_p90:.0f}s"
                if age_med > POOL_STALE_WARN_SEC:
                    stale_note += "(!stale)"

            core = (
                f"[iter {iter_num:4d}] "
                f"p={mean_p:.4f} " + (f"v={mean_v:.4f} " if mean_v is not None else "") +
                f"kl={report_kl:.6f} safe_kl={safe_kl:.6f} r_raw={r_raw:.4f} r_raw_ema={r_raw_ema:.4f} "
                f"r_ctr={r_ctr:.4f} r_shp={r_shaped:.4f} r_eval_samp={r_eval_sampled:.4f} "
                f"clip={clip_frac:.3f} akl_pi={approx_kl_pi:.4f} H={entropy_tok:.3f} "
                f"v_mae={v_mae:.4f} ev={explained_var:.3f} pool={pool_est}"
                f"| sel_tok={sel_tokens} adv|={adv_abs_m:.3e} "
                f"rΔ q50/90/99={ratio_qs[0]:.3e}/{ratio_qs[1]:.3e}/{ratio_qs[2]:.3e} "
                f"rΔmax={ratio_max:.3e} clip_th={ppo_clip_v:.3f} kl_ctl={kl_ctl_now:.3f} "
                f"lr={cur_lr:.2e}"
                f"{stale_note}"
            )
            print(core, flush=True)

            hdr = not os.path.exists(METRICS_CSV)
            with open(METRICS_CSV, "a") as f:
                if hdr:
                    f.write(
                        "iter,p_loss,v_loss,avg_kl,safe_kl,r_raw,r_raw_ema,r_ctr,r_shp,r_eval_samp,"
                        "clip_frac,approx_kl_pi,entropy,v_mae,explained_var,pool_est,age_med,age_p90,lr\n"
                    )
                f.write(
                    f"{iter_num},{mean_p},{'' if mean_v is None else mean_v},"
                    f"{report_kl},{safe_kl},{r_raw},{r_raw_ema},{r_ctr},{r_shaped},{r_eval_sampled},"
                    f"{clip_frac},{approx_kl_pi},{entropy_tok},{v_mae},{explained_var},"
                    f"{pool_est},{age_med},{age_p90},{cur_lr}\n"
                )

            algo = "PPO" if use_ppo else ("GRPO" if use_grpo else "DAPO")
            ckpt = {
                'model': raw_actor.state_dict(), 'critic': critic.state_dict(), 'ref': ref.state_dict(),
                'optimizer_actor': opt_a.state_dict(), 'optimizer_critic': opt_c.state_dict(),
                'vocab_size': model_args.get('vocab_size', 50304), 'iter_num': iter_num,
            }
            torch.save(ckpt, os.path.join(out_dir, f"{algo}_ckpt.pt"))

            if DEBUG_SAMPLE_EVERY and (iter_num % DEBUG_SAMPLE_EVERY == 0) and samples.seqs.size(0) > 0:
                try:
                    B = samples.seqs.size(0)
                    i0 = int(torch.randint(0, B, (1,)).item())
                    L0 = int(samples.attention_mask[i0].sum().item())
                    txt0 = tok.decode(samples.seqs[i0, :L0].detach().cpu()).replace("\n", " ")
                    r0 = float(experiences[i0].reward[0].item())
                    print(f"[sample] reward={r0:.4f} text={txt0[:200]}", flush=True)
                except Exception as e:
                    print(f"[sample] skip(print) due to error: {e}", flush=True)

    if ddp: destroy_process_group()

if __name__ == "__main__":
    main()
