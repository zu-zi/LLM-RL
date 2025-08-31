"""
训练脚本：支持单卡/多卡（DDP）
- 当前启用 RL 分支：PPO / GRPO / DAPO / Token Entropy
- 只在固定 prompts 子集（prepare.py 生成）上训练，便于观测 RL 效果
- 节约显存：block_size=512（prompt≤256，response≤255 含EOS）
"""

import os
import time
import math
import pickle
import random
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from RL.PPO import PPOTrainer, Critic
from RL.GRPO import GRPOTrainer
from RL.DAPO import DAPOTrainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import tiktoken

# --------------------- RL 开关（可被 configurator.py 覆盖） ---------------------
use_ppo = False
use_grpo = False
use_dapo = True         # 示例：默认 DAPO
use_token_entropy = False  # 熵掩码（2/8 规则）在各 Trainer 的 train_on_experience 中启用

# --------------------- 基础超参（可被 configurator.py 覆盖） ---------------------
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'    # 'scratch' | 'resume' | 'gpt2*' 例如 'gpt2', 'gpt2-medium', ...

# wandb（如需）
wandb_log = False
wandb_project = 'owt'
wandb_run_name = 'gpt2'

# data（SFT用；RL 因为用固定 prompts，不依赖）
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8
batch_size = 12                      # RL：每步使用的 prompt 条数
block_size = 512                     # 直接缩小到 512（prepare.py 已裁剪 prompt≤256）

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# AdamW（SFT学习率；RL 有单独 LR）
learning_rate = 6e-4
RL_learning_rate = 1e-5
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# 学习率衰减（SFT用，占位）
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5

# DDP
backend = 'nccl'

# 系统/精度
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True

# --------------------- 允许外部配置覆盖 ---------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
if os.path.exists('configurator.py'):
    exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}

# --------------------- GPT2 分词器（actor 侧） ---------------------
class GPT2Tok:
    def __init__(self):
        self.enc = tiktoken.get_encoding("gpt2")
        self.eos_id = self.enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
    def encode(self, s: str):
        return self.enc.encode(s, allowed_special="all")
    def decode(self, ids):
        if torch.is_tensor(ids):
            ids = ids.tolist()
        return self.enc.decode(ids)

gpt2tok = GPT2Tok()

# --------------------- 固定 prompts ---------------------
PROMPT_FILE = os.path.join(os.path.dirname(__file__), "data/RL_dataset/prompt.bin")
print(f"Loading fixed prompts from {PROMPT_FILE} ...")
blob = torch.load(PROMPT_FILE)
PROMPTS_TEXT = blob["prompts"]            # list[str]（供奖励模型分词）
PROMPT_TOKEN_IDS = blob["gpt2_token_ids"] # list[list[int]]（未 pad 的 GPT-2 ids）
EOS_ID = blob["eos_id"]
NUM_PROMPTS = len(PROMPT_TOKEN_IDS)
print(f"Loaded {NUM_PROMPTS} fixed prompts.")

class FixedPromptSampler:
    """固定顺序循环采样，便于对比训练效果"""
    def __init__(self, token_ids, seed=1337):
        self.ids = token_ids
        self.order = list(range(len(token_ids)))
        random.Random(seed).shuffle(self.order)
        self.i = 0
    def sample_ids(self, batch_size: int):
        out = []
        for _ in range(batch_size):
            out.append(self.ids[self.order[self.i]])
            self.i = (self.i + 1) % len(self.order)
        return out

sampler = FixedPromptSampler(PROMPT_TOKEN_IDS, seed=blob.get("seed", 1337))

# --------------------- DDP 初始化 ---------------------
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# --------------------- 尝试从数据派生 vocab_size（若存在 meta.pkl，SFT 习惯） ---------------------
data_dir = os.path.join('data', dataset)
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta.get('vocab_size', None)
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# ======================================================================
# ========== 模型初始化 & 断点续训（支持 scratch / gpt2* / resume） ==========
# ======================================================================
model_args = dict(
    n_layer=n_layer, n_head=n_head, n_embd=n_embd,
    block_size=block_size, bias=bias, vocab_size=None, dropout=dropout
)

def _build_fresh_gpt(args):
    cfg = GPTConfig(**args)
    m = GPT(cfg)
    return m

# 算法标签（用于优先选择 RL ckpt 文件名）
algo_tag = "PPO" if use_ppo else ("GRPO" if use_grpo else ("DAPO" if use_dapo else "RL"))
rl_ckpt_name_candidates = [
    f"{algo_tag}_ckpt.pt",  # 新：按算法区分
    "RL_ckpt.pt",           # 旧：通用名
]
rl_ckpt_path = None
for name in rl_ckpt_name_candidates:
    p = os.path.join(out_dir, name)
    if os.path.exists(p):
        rl_ckpt_path = p
        break

sft_ckpt_path = os.path.join(out_dir, "ckpt.pt")  # NanoGPT 原 SFT ckpt
base_state = None
resume_payload = None
iter_num = 0

if init_from == "resume" and rl_ckpt_path is not None:
    # ---------- 恢复 RL 训练 ----------
    print(f"[resume] loading RL checkpoint: {rl_ckpt_path}")
    resume_payload = torch.load(rl_ckpt_path, map_location=device)
    # 统一 vocab_size
    model_args['vocab_size'] = resume_payload.get('vocab_size', 50304)
    base_model = _build_fresh_gpt(model_args).to(device)
    base_model.load_state_dict(resume_payload['model'])
    base_state = base_model.state_dict()
    iter_num = resume_payload.get('iter_num', 0)

elif init_from == "resume" and os.path.exists(sft_ckpt_path):
    # ---------- 没有 RL ckpt，回退 SFT ckpt ----------
    print(f"[resume] RL ckpt not found, fallback to SFT checkpoint: {sft_ckpt_path}")
    sft_ckpt = torch.load(sft_ckpt_path, map_location=device)
    ckpt_model_args = sft_ckpt['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = ckpt_model_args[k]
    # 强制裁剪到当前 block_size（512）
    if block_size < model_args['block_size']:
        model_args['block_size'] = block_size
    base_model = _build_fresh_gpt(model_args).to(device)
    state_dict = sft_ckpt['model']
    # 兼容 _orig_mod. 前缀
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    base_model.load_state_dict(state_dict)
    base_state = base_model.state_dict()
    iter_num = sft_ckpt.get('iter_num', 0)

else:
    # ---------- 新开训练 ----------
    if isinstance(init_from, str) and init_from.startswith("gpt2"):
        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
        override_args = dict(dropout=dropout)
        base_model = GPT.from_pretrained(init_from, override_args)
        # 读取结构并对齐本地 block_size
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = getattr(base_model.config, k)
        if block_size < model_args['block_size']:
            base_model.crop_block_size(block_size)
            model_args['block_size'] = block_size
        base_model = base_model.to(device)
        base_state = base_model.state_dict()
    else:
        print("Initializing a new model from scratch")
        model_args['vocab_size'] = 50304 if meta_vocab_size is None else meta_vocab_size
        base_model = _build_fresh_gpt(model_args).to(device)
        base_state = base_model.state_dict()

# --------------------- ref / actor 的构建 ---------------------
# ref：优先使用 RL ckpt 中保存的 ref；否则用 base_state
gptconf_ref = GPTConfig(**model_args)
ref_model = GPT(gptconf_ref).to(device)
if resume_payload is not None and 'ref' in resume_payload:
    print("[resume] loading ref model from RL checkpoint")
    ref_model.load_state_dict(resume_payload['ref'])
else:
    ref_model.load_state_dict(base_state)
for p in ref_model.parameters():
    p.requires_grad = False
ref_model.eval()

# actor：从 RL ckpt 恢复，或者从 base_state 起步
gptconf_actor = GPTConfig(**model_args)
actor_model = GPT(gptconf_actor).to(device)
if resume_payload is not None:
    print("[resume] loading actor model from RL checkpoint")
    actor_model.load_state_dict(resume_payload['model'])
else:
    actor_model.load_state_dict(base_state)

# 编译 & DDP
if compile:
    print("compiling the actor model... (takes ~1 minute)")
    actor_model = torch.compile(actor_model)
if ddp:
    actor_model = DDP(actor_model, device_ids=[int(device.split(':')[-1])])
raw_actor = actor_model.module if ddp else actor_model

# critic：PPO/DAPO 需要；从 RL ckpt 恢复（如有）
critic_model = Critic(raw_actor).to(device)
if resume_payload is not None and 'critic' in resume_payload:
    print("[resume] loading critic model from RL checkpoint")
    critic_model.load_state_dict(resume_payload['critic'])

# 优化器：从 RL ckpt 恢复（如有）
optimizer_actor = torch.optim.AdamW(raw_actor.parameters(), lr=RL_learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
optimizer_critic = torch.optim.AdamW(critic_model.parameters(), lr=RL_learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
if resume_payload is not None:
    if 'optimizer_actor' in resume_payload:
        optimizer_actor.load_state_dict(resume_payload['optimizer_actor'])
    if 'optimizer_critic' in resume_payload:
        optimizer_critic.load_state_dict(resume_payload['optimizer_critic'])

# --------------------- sglang 引擎（offline模式） ---------------------
try:
    import sglang as sgl
    engine = sgl.Engine(model=raw_actor, mode="offline")  # 离线推理引擎（不可用时自动回退）
    print("SGLang engine loaded (offline mode).")
except Exception as e:
    engine = None
    print(f"SGLang not available, fallback to raw_actor.generate. Error: {e}")

def generate_with_engine(ids_t, max_new_tokens, eos_token_id):
    if engine is not None:
        # 注意：不同 sglang 版本接口可能不同，这里做最简封装
        return engine.generate(ids_t, max_new_tokens=max_new_tokens, eos_token_id=eos_token_id)
    else:
        return raw_actor.generate(
            idx=ids_t,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_k=None,
            eos_token_id=eos_token_id
        )

# --------------------- 奖励模型（HF） ---------------------
reward_model_name = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_name).to(device).eval()
reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)

# --------------------- wandb ---------------------
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# --------------------- Trainer 构造 ---------------------
trainer = None
if use_ppo:
    trainer = PPOTrainer(
        actor_model=raw_actor, ref_model=ref_model, reward_model=reward_model,
        critic_model=critic_model, actor_tokenizer=gpt2tok, reward_tokenizer=reward_tokenizer,
        optimizer_actor=optimizer_actor, optimizer_critic=optimizer_critic, device=device
    )
elif use_grpo:
    trainer = GRPOTrainer(
        actor_model=raw_actor, ref_model=ref_model, reward_model=reward_model,
        actor_tokenizer=gpt2tok, reward_tokenizer=reward_tokenizer,
        optimizer_actor=optimizer_actor, group_size=4, kl_coef=0.0, clip_reward=5.0, device=device
    )
elif use_dapo:
    trainer = DAPOTrainer(
        actor_model=raw_actor, ref_model=ref_model, reward_model=reward_model,
        actor_tokenizer=gpt2tok, reward_tokenizer=reward_tokenizer,
        optimizer_actor=optimizer_actor, kl_coef=0.01, device=device
    )

# --------------------- 训练主循环（只跑 RL） ---------------------
t0 = time.time()
local_iter_num = 0
running_mfu = -1.0

while trainer is not None:
    # 1) 采样 batch 条 prompts
    batch_ids = sampler.sample_ids(batch_size)

    # 2) 逐条生成（动态限制 response，保证总长 ≤ block_size）
    generations = []  # list[dict]: {"prompt_ids": Tensor, "full_ids": Tensor, "response_ids": Tensor}
    for ids in batch_ids:
        ids_t = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, T_prompt]
        prompt_len = ids_t.size(1)
        room = block_size - prompt_len - 1  # 预留 EOS
        if room <= 0:
            continue
        max_new_tokens = max(1, room)

        out = generate_with_engine(ids_t, max_new_tokens=max_new_tokens, eos_token_id=EOS_ID)
        full = out[0]                   # [T_prompt + T_resp]
        resp = full[prompt_len:]        # response 段（含/不含 EOS）
        generations.append({"prompt_ids": ids_t.squeeze(0), "full_ids": full, "response_ids": resp})

    # 3) 评估经验（不同算法内部会构建 Samples/Experience）
    experiences, avg_kl, avg_reward = trainer.evaluate_experience(generations)

    # 4) 更新（PPO 通常多 step；GRPO/DAPO 一般 1 step）
    for exp in experiences:
        # token entropy 插口：use_token_entropy=True 时，trainer 内部对 logits 应用熵掩码
        trainer.train_on_experience(exp, use_token_entropy=use_token_entropy)

    # 5) 日志 & 保存
    if iter_num % eval_interval == 0 and master_process:
        print(f"iter {iter_num}: avg_kl={avg_kl:.6f}, avg_reward={avg_reward:.4f}")
        if wandb_log:
            import wandb
            wandb.log({
                "iter": iter_num,
                "train/avg_kl": float(avg_kl),
                "train/avg_reward": float(avg_reward),
            })
        # 保存 RL ckpt（包含 actor/critic/optimizer/ref）
        ckpt_name = f"{algo_tag}_ckpt.pt"
        ckpt = {
            'model': raw_actor.state_dict(),
            'critic': critic_model.state_dict(),
            'ref': ref_model.state_dict(),
            'optimizer_actor': optimizer_actor.state_dict(),
            'optimizer_critic': optimizer_critic.state_dict(),
            'vocab_size': model_args.get('vocab_size', 50304),
            'iter_num': iter_num,
        }
        path = os.path.join(out_dir, ckpt_name)
        print(f"saving RL checkpoint to {path}")
        torch.save(ckpt, path)

    iter_num += 1
    if iter_num > max_iters:
        break

# --------------------- DDP 清理 ---------------------
if ddp:
    destroy_process_group()
