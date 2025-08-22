"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from RL.PPO import PPOTrainer,Critic
from data.RL_dataset.prepare import enc_wrapper
import tiktoken
from transformers import AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification, AutoTokenizer

# -----------------------------------------------------------------------------
# RLHF
use_ppo = False
use_grpo = False
use_dapo = False
use_token_entropy = False

# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024#生成时能看多长的上下文
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
RL_learning_rate=1e-5
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
# 允许外部configurator.py覆盖
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------
# RL
PROMPT_FILE = os.path.join(os.path.dirname(__file__), "data/RL_dataset/prompt.bin")
print(f"Loading pre-tokenized prompts from {PROMPT_FILE} ...")
prompt_data = torch.load(PROMPT_FILE)   # {'input_ids': Tensor, 'attention_mask': Tensor}
num_prompts = prompt_data['input_ids'].size(0)
print(f"Loaded {num_prompts} prompts.")

def get_prompt_batch(batch_size, device):
    #从 prompt.bin 中随机取 batch
    idx = torch.randint(0, num_prompts, (batch_size,))
    input_ids = prompt_data['input_ids'][idx].to(device)
    attention_mask = prompt_data['attention_mask'][idx].to(device)
    return input_ids, attention_mask

# 加载Skywork奖励模型
reward_model_name = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_name).to(device).eval()
reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)

# various inits, derived attributes, I/O setup
# DDP 分布式训练初始化
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

#系统设置
if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
# def get_batch(split):
#     # We recreate np.memmap every batch to avoid a memory leak, as per
#     # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
#     if split == 'train':
#         data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
#     else:
#         data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
#     #每次采样一批随机位置ix，构建长度为block_size的输入x和目标y
#     ix = torch.randint(len(data) - block_size, (batch_size,))#y是x向后移动一个token
#     x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
#     y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
#     if device_type == 'cuda':
#         # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
#         x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
#     else:
#         x, y = x.to(device), y.to(device)
#     return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
# 如果找到了就加载vocab_size,如果没找到，用GPT-2默认值50304
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init # 3种模型情况
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']#保存时记录的模型配置
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']#保存的state_dict
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
# 如果配置的block_size比模型原始支持的短，就动态裁剪
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value

model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
# 混合精度训练用，能避免 float16 的数值不稳定
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer：定义在model里的函数，返回一个AdamW优化器；如果是resume，还会恢复之前的优化器状态
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])

checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
# unwrap raw model for saving / for using custom methods (generate, forward return_all_logits)
raw_model = model.module if ddp else model
# RL,后续再添加逻辑：选择哪种RL算法
import copy

actor_model = raw_model  # 确保 raw_model 是未DDP包裹的原始模型

# 直接深拷贝 actor_model 作为 ref_model
ref_model = copy.deepcopy(actor_model).to(device)

# 冻结 ref_model 参数
for param in ref_model.parameters():
    param.requires_grad = False
ref_model.eval()

critic_model = Critic(actor_model).to(device)

optimizer_actor = torch.optim.AdamW(actor_model.parameters(), lr=RL_learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
optimizer_critic = torch.optim.AdamW(critic_model.parameters(), lr=RL_learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)

ppo_trainer = PPOTrainer(
    actor_model=actor_model,
    ref_model=ref_model,
    reward_model=reward_model,
    critic_model=critic_model,
    actor_tokenizer=enc_wrapper,
    reward_tokenizer=reward_tokenizer,
    optimizer_actor=optimizer_actor,
    optimizer_critic=optimizer_critic,
    device=device
)
# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():#每隔一段时间会用这个函数来估算当前模型在train和val上的loss
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
# 前期使用线性warmup，然后使用余弦衰减
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
# X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
running_mfu = -1.0

if use_ppo:
    iter_num = globals().get("iter_num", 0)
    policy_loss = value_loss = 0.0
    while True:
        # --- 1) sample prompts ---
        input_ids, attention_mask = get_prompt_batch(batch_size, device)  # (B, Lp), (B, Lp)
        # 生成 samples（PPOTrainer.generate_samples 返回 list[Samples]）
        max_new_tokens = block_size - 256
        samples_list = ppo_trainer.generate_samples(
            (input_ids, attention_mask),
            max_length=block_size,
            max_new_tokens=max_new_tokens
        )
        samples = samples_list[0]  # 批次 Samples

        # --- 2) evaluate samples -> experiences (and avg_kl for logging) ---
        experiences, avg_kl = ppo_trainer.evaluate_experience(samples)

        # --- 3) 多次/多轮更新 PPO（你原本是 1-step，保持不变） ---
        for exp in experiences:
            policy_loss, value_loss = ppo_trainer.train_on_experience(exp)

        # --- 4) 日志与保存 ---
        if iter_num % eval_interval == 0 and master_process:
            print(f"iter {iter_num}: policy_loss={policy_loss:.4f}, value_loss={value_loss:.4f}, avg_kl={avg_kl:.6f}")
            if wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/policy_loss": policy_loss,
                    "train/value_loss": value_loss,
                    "train/avg_kl": avg_kl,
                })
            checkpoint = {
                'model': actor_model.state_dict(),
                'critic': critic_model.state_dict(),
                'optimizer_actor': optimizer_actor.state_dict(),
                'optimizer_critic': optimizer_critic.state_dict(),
                'iter_num': iter_num,
            }
            print(f"saving PPO checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, 'RL_ckpt.pt'))

        iter_num += 1
        if iter_num > max_iters:
            break

else:
    
    # while True:
    #     # determine and set the learning rate for this iteration
    #     lr = get_lr(iter_num) if decay_lr else learning_rate
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr

    #     # evaluate the loss on train/val sets and write checkpoints
    #     if iter_num % eval_interval == 0 and master_process:
    #         losses = estimate_loss()
    #         print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    #         if wandb_log:
    #             wandb.log({
    #                 "iter": iter_num,
    #                 "train/loss": losses['train'],
    #                 "val/loss": losses['val'],
    #                 "lr": lr,
    #                 "mfu": running_mfu*100, # convert to percentage
    #             })
    #         if losses['val'] < best_val_loss or always_save_checkpoint:
    #             best_val_loss = losses['val']
    #             if iter_num > 0: # 保存了model state_dict、优化器状态、当前迭代、最佳loss等 # RLHF自定义保存逻辑
    #                 checkpoint = {
    #                     'model': raw_model.state_dict(),
    #                     'optimizer': optimizer.state_dict(),
    #                     'model_args': model_args,
    #                     'iter_num': iter_num,
    #                     'best_val_loss': best_val_loss,
    #                     'config': config,
    #                 }
    #                 print(f"saving checkpoint to {out_dir}")
    #                 torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    #     if iter_num == 0 and eval_only:
    #         break

    #     # forward backward update, with optional gradient accumulation to simulate larger batch size
    #     # and using the GradScaler if data type is float16
    #     # 主循环逻辑
    #     for micro_step in range(gradient_accumulation_steps):
    #         if ddp:
    #             # in DDP training we only need to sync gradients at the last micro step.
    #             # the official way to do this is with model.no_sync() context manager, but
    #             # I really dislike that this bloats the code and forces us to repeat code
    #             # looking at the source of that context manager, it just toggles this variable
    #             model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
    #         with ctx:
    #             logits, loss = model(X, Y)
    #             loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
    #         # immediately async prefetch next batch while model is doing the forward pass on the GPU
    #         # X, Y = get_batch('train')
    #         # backward pass, with gradient scaling if training in fp16
    #         scaler.scale(loss).backward()# 缩放loss保证float16稳定性
    #     # clip the gradient梯度裁剪
    #     if grad_clip != 0.0:
    #         scaler.unscale_(optimizer)
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    #     # step the optimizer and scaler if training in fp16
    #     scaler.step(optimizer)
    #     scaler.update()
    #     # flush the gradients as soon as we can, no need for this memory anymore
    #     optimizer.zero_grad(set_to_none=True)

    #     # timing and logging
    #     t1 = time.time()
    #     dt = t1 - t0
    #     t0 = t1
    #     if iter_num % log_interval == 0 and master_process:
    #         # get loss as float. note: this is a CPU-GPU sync point
    #         # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
    #         lossf = loss.item() * gradient_accumulation_steps
    #         if local_iter_num >= 5: # let the training loop settle a bit
    #             mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
    #             running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
    #         print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    #     iter_num += 1
    #     local_iter_num += 1

    #     # termination conditions
    #     if iter_num > max_iters:
    #         break

 if ddp:
    destroy_process_group()