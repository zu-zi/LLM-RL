# train_RL_only.py
import os, sys, time, random, pickle, json, subprocess, shlex
import numpy as np
import torch, tiktoken
from contextlib import nullcontext
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
kl_ctl = 0.10  # 默认，可被 config 覆盖

dataset="openwebtext"
wandb_log=False; wandb_project="rl"; wandb_run_name="run"

# sglang 离线池（默认值，可被 config 覆盖）
SGLANG_ON=True
SGLANG_OFFLINE=True
SGLANG_MODEL_PATH="gpt2-large"
SGLANG_SYNC_DIR="./sgl_pool"  # 建议在 config 中指向大盘
SGLANG_ROLLOUT_TARGET=1024
SGLANG_REFILL_BATCH=256
SGLANG_MAX_NEW=192  # 稍降长度，吞吐更稳

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
    return blob["prompts"], blob["gpt2_token_ids"], blob["eos_id"], blob.get("seed",1337), pf

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

# ========= sglang 离线补货 =========
def _spawn_rollout_subprocess(prompt_bin_path, count, sync_dir, max_new, rollout_log_dir, quiet=True):
    ensure_dir(sync_dir)
    os.makedirs(rollout_log_dir, exist_ok=True)
    cmd = f'python -u rollout_worker.py --model "{SGLANG_MODEL_PATH}" --prompt-bin "{prompt_bin_path}" ' \
          f'--out-dir "{sync_dir}" --count {int(count)} --max-new {int(max_new)}'
    if quiet:
        logf = os.path.join(rollout_log_dir, f"rollout_{int(time.time())}.log")
        print(f"[rollout] spawn (quiet) -> {logf}")
        with open(logf, "a") as f:
            ret = subprocess.call(shlex.split(cmd), stdout=f, stderr=f)
    else:
        print(f"[rollout] spawning: {cmd}")
        ret = subprocess.call(shlex.split(cmd))
    if ret != 0:
        raise RuntimeError(f"rollout_worker exit with code {ret}")

def _is_master(ddp):
    return (not ddp) or (os.environ.get("RANK","0") == "0")

# ========= 主流程 =========
def main():
    _apply_cli_config_once()

    # 这些依赖外部 config 的量，必须在 exec(config) 之后再确定
    EVAL_LOG_EVERY = eval_interval
    DEBUG_SAMPLE_EVERY = eval_interval
    METRICS_CSV = os.path.join(out_dir, "metrics.csv")
    LOW_WATERMARK = max(batch_size * 8, SGLANG_REFILL_BATCH // 2)
    REFILL_COUNT  = max(SGLANG_REFILL_BATCH, batch_size * 8)
    ROLLOUT_LOG_DIR = os.path.join(out_dir, "rollout_logs")
    ROLLOUT_QUIET = True

    # 固定 prompts / tokenizer
    tok = GPT2Tok()
    PROMPTS_TEXT, PROMPT_TOKEN_IDS, EOS_ID, seed, PROMPT_BIN_PATH = _load_prompts()
    pad_id = 0

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
    ref.to(dtype=ref_dtype)

    actor=GPT(GPTConfig(**model_args)).to(dev); actor.load_state_dict(base_state)
    if compile: actor=torch.compile(actor)
    if ddp: actor=DDP(actor, device_ids=[int(dev.split(':')[-1])])
    raw_actor=actor.module if ddp else actor

    critic=Critic(raw_actor).to(dev)

    # 优化器（8bit）
    from bitsandbytes.optim import AdamW8bit
    opt_a=AdamW8bit(raw_actor.parameters(), lr=RL_learning_rate, betas=(beta1,beta2), weight_decay=weight_decay)
    opt_c=AdamW8bit(critic.parameters(),    lr=RL_learning_rate, betas=(beta1,beta2), weight_decay=weight_decay)

    # 奖励模型（强制 CPU + 包装器）
    rw_name="Skywork/Skywork-Reward-V2-Qwen3-0.6B"
    reward_hf = AutoModelForSequenceClassification.from_pretrained(
        rw_name, device_map="cpu", torch_dtype=torch.float32
    ).eval()
    reward_model = RewardOnCPU(reward_hf)
    reward_tokenizer=AutoTokenizer.from_pretrained(rw_name)

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
    if master and SGLANG_ON and SGLANG_OFFLINE:
        print(f"[sglang] offline pool enabled: dir={SGLANG_SYNC_DIR}, target={SGLANG_ROLLOUT_TARGET}, batch={SGLANG_REFILL_BATCH}")
    ensure_dir(SGLANG_SYNC_DIR)
    os.makedirs(ROLLOUT_LOG_DIR, exist_ok=True)

    for iter_num in range(0, max_iters+1):
        # 1) 低水位触发，批量补货（子进程生成完退出释放显存）
        if SGLANG_ON and SGLANG_OFFLINE:
            pool_est = estimate_size(SGLANG_SYNC_DIR, SGLANG_REFILL_BATCH)
            if pool_est < LOW_WATERMARK:
                _spawn_rollout_subprocess(PROMPT_BIN_PATH, REFILL_COUNT, SGLANG_SYNC_DIR, SGLANG_MAX_NEW, ROLLOUT_LOG_DIR, quiet=True)

        # 2) 从池里取一批（不足则用 actor 现场补齐）
        batch = dequeue_items(SGLANG_SYNC_DIR, batch_size)
        if len(batch) < batch_size:
            for _ in range(batch_size - len(batch)):
                ids = random.choice(PROMPT_TOKEN_IDS)
                ids_t=torch.tensor(ids,dtype=torch.long,device=dev).unsqueeze(0)
                prompt_len=ids_t.size(1)
                room = block_size - prompt_len - 1
                if room <= 0: continue
                out = raw_actor.generate(idx=ids_t, max_new_tokens=max(1,room),
                                         temperature=1.0, top_k=None, eos_token_id=tok.eos_id)
                batch.append({"prompt_ids": ids, "full_ids": out[0].tolist()})

        # 3) 打包 Samples
        samples = pack_samples(batch, pad_id=0, block_size=block_size, device=torch.device(dev))

        # 4) 评估经验 & 5) 参数更新
        experiences, avg_kl, avg_reward = trainer.evaluate_experience(samples)

        # --- 自适应 KL 系数：目标 0.02，偏离就动态调 ---
        kl_target = 0.02
        if avg_kl > 0:
            err = avg_kl / kl_target - 1.0
            trainer.kl_ctl *= float(np.exp(np.clip(err, -0.2, 0.2)))  # 每步最多 ±22% 调整

        pl, vl = [], []
        for exp in experiences:
            out = trainer.train_on_experience(exp, use_token_entropy=use_token_entropy)
            if isinstance(out, tuple):
                p,v = out; pl.append(float(p.detach().item())); vl.append(float(v.detach().item()))
            else:
                pl.append(float(out.detach().item()))
        mean_p = float(np.mean(pl)) if pl else 0.0
        mean_v = float(np.mean(vl)) if vl else None

        # 6) 日志 & ckpt（精简一行 + CSV 记录 + 抽查样本）
        if (iter_num % EVAL_LOG_EVERY == 0) and _is_master(ddp):
            pool_now = estimate_size(SGLANG_SYNC_DIR, SGLANG_REFILL_BATCH) if (SGLANG_ON and SGLANG_OFFLINE) else -1
            if mean_v is not None:
                print(f"[iter {iter_num:4d}] p={mean_p:.4f} v={mean_v:.4f} kl={avg_kl:.6f} r={avg_reward:.4f} pool={pool_now}", flush=True)
            else:
                print(f"[iter {iter_num:4d}] loss={mean_p:.4f} kl={avg_kl:.6f} r={avg_reward:.4f} pool={pool_now}", flush=True)

            # CSV
            hdr = not os.path.exists(METRICS_CSV)
            with open(METRICS_CSV, "a") as f:
                if hdr: f.write("iter,p_loss,v_loss,avg_kl,avg_reward,pool_est\n")
                f.write(f"{iter_num},{mean_p},{'' if mean_v is None else mean_v},{avg_kl},{avg_reward},{pool_now}\n")

            # 抽查一条
            if DEBUG_SAMPLE_EVERY and (iter_num % DEBUG_SAMPLE_EVERY == 0) and samples.seqs.size(0) > 0:
                i0 = int(torch.randint(0, samples.seqs.size(0), (1,)).item())
                L0 = int(samples.attention_mask[i0].sum().item())
                txt0 = tok.decode(samples.seqs[i0, :L0])
                r0 = experiences[0].reward[i0].item() if isinstance(experiences, list) and len(experiences)>0 else float("nan")
                print(f"[sample] reward={r0:.4f} text={txt0[:200].replace(chr(10),' ')}", flush=True)

            # ckpt
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
