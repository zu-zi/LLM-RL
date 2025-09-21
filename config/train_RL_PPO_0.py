import os
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_SILENT"] = "true"
from datetime import datetime

# —— 日志 —— #
wandb_log      = True
wandb_project  = "LLM-RL-PPO"
wandb_run_name = f"ppo_gpt2l_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# —— 运行 —— #
out_dir        = "/root/autodl-tmp/Results"
eval_interval  = 10
max_iters      = 1500
seed_base      = 1337
compile        = False
backend        = "nccl"
device         = "cuda"

# —— 模型 / 上下文 —— #
init_from   = "gpt2-large"
block_size  = 256
bias        = False
dropout     = 0.0

# —— 批次 / 优化 —— #
batch_size                  = 4
gradient_accumulation_steps = 1
RL_learning_rate            = 1.7e-6
CRITIC_LR_MULT              = 1.8    # 训练脚本里若支持：critic_lr = RL_learning_rate * CRITIC_LR_MULT
weight_decay                = 5e-3
beta1, beta2                = 0.9, 0.95
max_grad_norm               = 0.5
vf_clip                     = 0.15   # 稍放松（日志里 v_mae 偏大）

# —— PPO 关键超参 —— #
kl_ctl_init  = 1.4           # 提高初值，抑制你日志中期 KL 失控
kl_ctl_min   = 0.35          # 加下限（训练代码里若有自适应，别降到 0.15 以下）
ppo_clip     = 0.15          # 收紧步长，减少 clip 爆表
entropy_coef = 0.003         # 略降，避免与 KL 冲突；高 KL 时建议训练里把它门控到 0

# —— PPO 安全帽（需要在构造 PPOTrainer 时传入这些）—— #
ratio_min     = 0.75
ratio_max     = 1.25
kl_token_cap  = 0.40         # 0.5 -> 0.4
k3_cap        = 1.20         # 1.5 -> 1.2
ent_mask_keep = 0.20

# —— iter0 —— #
EVAL_ITER0     = True
ITER0_BATCHES  = 2

# —— 采样口径 —— #
SAMPLE_TEMPERATURE = 0.7
SAMPLE_TOP_P       = 0.9
SAMPLE_TOP_K       = 0
SAMPLE_REP_PENALTY = 1.1
SAMPLE_STOPS       = ["\nHuman:", "\n\nHuman:"]
MIN_RESP_TOK       = 16

# —— sglang 样本池 —— #
SGLANG_ON              = True
SGLANG_OFFLINE         = True
SGLANG_MODEL_PATH      = "/root/autodl-tmp/actor_exports/current"
SGLANG_EXPORT_BASE     = "/root/autodl-tmp/actor_exports"
SGLANG_MODEL_POINTER   = None
SGLANG_SYNC_DIR        = "/root/autodl-tmp/sgl_pool"
SGLANG_ROLLOUT_TARGET  = 300   # 200 -> 300，更稳
SGLANG_REFILL_BATCH    = 32
SGLANG_MAX_NEW         = 128
SGLANG_EXPORT_EVERY   = 30
# —— 补货调度 —— #
ROLL_LOW_WATERMARK_FACTOR = 3  # 2 -> 3
ROLL_REFILL_COUNT         = 48 # 40 -> 48
ROLL_COOLDOWN_SEC         = 4  # 6 -> 4
ROLL_MIN_FREE_MB          = 2500
REFRESH_EVERY_BATCHES     = 26
FRESH_RATIO               = 0.5

# —— 奖励模型 —— #
REWARD_MODEL_NAME = "OpenAssistant/reward-model-deberta-v3-large-v2"
