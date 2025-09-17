import os
os.environ["WANDB_MODE"] = "offline"   # 纯离线记录到本地
os.environ["WANDB_SILENT"] = "true"    # 少打日志
wandb_log  = True
from datetime import datetime

# -------- 运行 / 日志 --------
out_dir        = "/root/autodl-tmp/Results"
eval_interval  = 10
max_iters      = 1500
seed_base      = 1337
compile        = False
backend        = "nccl"
device         = "cuda"

# -------- Weights & Biases --------
wandb_project  = "LLM-RL-4"
wandb_run_name = f"ppo_gpt2l_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# -------- 模型 / 上下文 --------
init_from   = "gpt2-large"  # actor/ref 初始化
block_size  = 256           # 96(prompt) + ~96(new) + 余量；更省显存更稳
bias        = False
dropout     = 0.0

# -------- 批次 / 优化 --------
batch_size                  = 4        # 32GB 稳健
gradient_accumulation_steps = 1        # Trainer 内部有 micro-batch
RL_learning_rate            = 1.7e-6   # 与 train.py 默认一致
weight_decay                = 5e-3
beta1, beta2                = 0.9, 0.95
max_grad_norm               = 0.5
vf_clip                     = 0.1

# -------- PPO 关键超参 --------
kl_ctl_init  = 1.2          # 前期更稳，配合自适应调整
ppo_clip     = 0.2
entropy_coef = 0.005        # 轻微熵正则

# -------- 采样口径（与离线池一致）--------
SAMPLE_TEMPERATURE = 0.7
SAMPLE_TOP_P       = 0.9
SAMPLE_TOP_K       = 0
SAMPLE_REP_PENALTY = 1.1
SAMPLE_STOPS       = ["\nHuman:", "\n\nHuman:"]
MIN_RESP_TOK       = 16      # 回到 16，增加有效动作位

# -------- sglang 离线样本池 --------
SGLANG_ON              = True
SGLANG_OFFLINE         = True
SGLANG_MODEL_PATH      = "/root/autodl-tmp/actor_exports/current"   # 供 worker 热切换
SGLANG_EXPORT_BASE     = "/root/autodl-tmp/actor_exports"
SGLANG_MODEL_POINTER   = None
SGLANG_SYNC_DIR        = "/root/autodl-tmp/sgl_pool"
SGLANG_ROLLOUT_TARGET  = 200
SGLANG_REFILL_BATCH    = 32
SGLANG_MAX_NEW         = 96            # 关键：别设 128，先稳住 96

# —— 补货调度（避免与训练抢显存）——
ROLL_LOW_WATERMARK_FACTOR = 2      # 低于 batch_size*2 才补
ROLL_REFILL_COUNT         = 40     # 触发的小颗粒补货
ROLL_COOLDOWN_SEC         = 6      # 两次补货最小间隔
ROLL_MIN_FREE_MB          = 2500   # 触发前至少空闲显存（MB）
REFRESH_EVERY_BATCHES     = 17     # worker 每 N 批检查一次指针
FRESH_RATIO               = 0.5    # 训练 batch 中“在线新鲜样本”占比

# -------- 奖励模型（英文）--------
REWARD_MODEL_NAME = "OpenAssistant/reward-model-deberta-v3-large-v2"
