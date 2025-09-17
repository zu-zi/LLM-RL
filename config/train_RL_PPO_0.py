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
RL_learning_rate            = 1.7e-6   # 比 1.5e-6 略激进，收敛更快
weight_decay                = 5e-3
beta1, beta2                = 0.9, 0.95
max_grad_norm               = 0.5
vf_clip                     = 0.1

# -------- PPO 关键超参 --------
kl_ctl_init  = 1.2         # 稍低于 0.9；前期探索更自由，配合自适应调整
ppo_clip     = 0.2
entropy_coef = 0.005      # 轻微熵正则，稳定但不过分抑制

# -------- 采样口径（与离线池一致）--------
SAMPLE_TEMPERATURE = 0.7
SAMPLE_TOP_P       = 0.9
SAMPLE_TOP_K       = 0
SAMPLE_REP_PENALTY = 1.1
SAMPLE_STOPS       = ["\nHuman:", "\n\nHuman:"]
MIN_RESP_TOK       = 16

# -------- sglang 离线样本池 --------
SGLANG_ON              = True
SGLANG_OFFLINE         = True
SGLANG_MODEL_PATH   = "/root/autodl-tmp/actor_exports/current"                # 可换更快推理模型
SGLANG_EXPORT_BASE  = "/root/autodl-tmp/actor_exports" 
SGLANG_MODEL_POINTER   = None                         # 需要热切换时填指针文件路径
SGLANG_SYNC_DIR        = "/root/autodl-tmp/sgl_pool"
SGLANG_ROLLOUT_TARGET  = 200                          # 目标可用样本估算上限
SGLANG_REFILL_BATCH    = 32                           # 一次子进程总生成量（粗粒度）
SGLANG_MAX_NEW         = 128                           # 与 block_size 配合（96 new 更稳）

# —— 补货调度（避免与训练抢显存）——
ROLL_LOW_WATERMARK_FACTOR = 2      # 低于 batch_size*2 才补
ROLL_REFILL_COUNT         = 40     # 实际触发的小颗粒补货
ROLL_COOLDOWN_SEC         = 6    # 两次补货最小间隔
ROLL_MIN_FREE_MB          = 2500   # 触发前至少空闲显存（MB）
REFRESH_EVERY_BATCHES     = 26     # worker 每 N 批检查一次指针
FRESH_RATIO               = 0.5    # 训练 batch 中“在线新鲜样本”占比（更稳）

# -------- 奖励模型（英文）--------
# 资源更友好（CPU/RAM 压力小），总体效果稳定；若追求更强可切回 large-v2
REWARD_MODEL_NAME = "OpenAssistant/reward-model-deberta-v3-large-v2"